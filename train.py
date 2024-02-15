#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, **kwargs):
    testing_iterations = kwargs["test_iterations"]
    saving_iterations = kwargs["save_iterations"]
    checkpoint_iterations = kwargs["checkpoint_iterations"]
    checkpoint = kwargs["start_checkpoint"]
    debug_from = kwargs["debug_from"]
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, **kwargs)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, **kwargs)
    gaussians.training_setup(opt)
    
    if kwargs['rgb_fix']:
        gaussians._features_dc.requires_grad = False
        gaussians._features_rest.requires_grad = False
    if kwargs['xyz_fix']:
        gaussians._xyz.requires_grad = False
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", dynamic_ncols=True)
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0 and not kwargs['no_sh']:
            gaussians.oneupSHdegree()
            
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # Covaraince Loss
        cov_loss = None
        if kwargs['cov_loss']:
            if kwargs['cov_loss_type'] == 'scale':
                scale = gaussians.get_scaling
                scale_x = torch.mean(scale[:,[0]])
                scale_y = torch.mean(scale[:,[1]])
                scale_z = torch.mean(scale[:,[2]])
                cov_loss = (scale_x + scale_y + scale_z) / 3
            elif kwargs['cov_loss_type'] == 'cov':
                cov = gaussians.get_covariance() 
                cov_x = cov[:,[0]] 
                cov_y = cov[:,[3]]
                cov_z = cov[:,[5]]
                cov_mean = torch.mean(torch.cat((cov_x, cov_y, cov_z), dim=1), dim=1) # (num_gaussians, 1)
                cov_loss = torch.mean(cov_mean)
            
            if kwargs['cov_loss'] == 'high':
                loss += kwargs['cov_weight'] * (1 / (cov_loss + 1e-7))
            elif kwargs['cov_loss'] == 'low':
                loss += kwargs['cov_weight'] * cov_loss
        
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), cov_loss, kwargs['deblur'])

            # Densification
            if iteration < opt.densify_until_iter and kwargs['no_densify'] == False:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, kwargs['reset_opacity_threshold'], scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    if kwargs['no_reset_opacity'] == False:
                        gaussians.reset_opacity()
                    
                if kwargs['rgb_fix']:
                    gaussians._features_dc.requires_grad = False
                    gaussians._features_rest.requires_grad = False
                if kwargs['xyz_fix']:
                    gaussians._xyz.requires_grad = False

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        # Every 7000, deblur image
        if iteration % kwargs['deblur_every_iter'] == 0 and kwargs['blur'] and kwargs['deblur'] :
            
            # Decrease blur filter size by 10
            kwargs['filter_size'] = kwargs['filter_size'] - kwargs['deblur_step']
            
            # Turn off blur if filter size is less than 3
            if kwargs['filter_size'] < 3:
                kwargs['blur'] = False
                kwargs['gaussian_blur'] = False
                
            scene.reloadCameras(dataset, **kwargs)

def prepare_output_and_logger(args, **kwargs):
    
    output_path = kwargs['output_path']
    exp_name = kwargs['exp_name']
    project_name = kwargs['project_name']    
    
    if (not args.model_path) and (not exp_name):
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    elif (not args.model_path) and exp_name:
        args.model_path = os.path.join("./output", exp_name)

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    with open(os.path.join(args.model_path, 'command_line.txt'), 'w') as file:
        # Write the log information to the file.
        file.write(' '.join(sys.argv))

    # Create Tensorboard writer
    tb_writer = None
    # Prepare Wandb
    # wandb.init(project=project_name, name=exp_name, dir=args.model_path, config=args, sync_tensorboard=True)
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
        print("Logging progress to Tensorboard at {}".format(args.model_path))
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, cov_loss, deblur):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        if cov_loss and (iteration % 500 == 0):
            rot = scene.gaussians.get_rotation
            rot_x = torch.mean(rot[:,[0]])
            rot_y = torch.mean(rot[:,[1]])
            rot_z = torch.mean(rot[:,[2]])
            scale = scene.gaussians.get_scaling
            scale_x = torch.mean(scale[:,[0]])
            scale_y = torch.mean(scale[:,[1]])
            scale_z = torch.mean(scale[:,[2]])
            
            tb_writer.add_scalar('train_loss_patches/cov_loss', cov_loss.item(), iteration)
            tb_writer.add_scalar('train_loss_patches/rot_x', rot_x.item(), iteration)
            tb_writer.add_scalar('train_loss_patches/rot_y', rot_y.item(), iteration)
            tb_writer.add_scalar('train_loss_patches/rot_z', rot_z.item(), iteration)
            tb_writer.add_scalar('train_loss_patches/scale_x', scale_x.item(), iteration)
            tb_writer.add_scalar('train_loss_patches/scale_y', scale_y.item(), iteration)
            tb_writer.add_scalar('train_loss_patches/scale_z', scale_z.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)   
    
        if (iteration % 250 == 0 and iteration <= 5000) or iteration % 3000 == 0:
            cov_x = scene.gaussians.get_covariance()[:,[0]].detach().cpu()
            cov_y = scene.gaussians.get_covariance()[:,[3]].detach().cpu()
            cov_z = scene.gaussians.get_covariance()[:,[5]].detach().cpu()
            cov_mean = torch.mean(torch.cat((cov_x, cov_y, cov_z), dim=1), dim=1)
            
            # tb_writer.add_histogram("cov/X_hist", cov_x, iteration, 'auto')
            # tb_writer.add_histogram("cov/Y_hist", cov_y, iteration, 'auto')
            # tb_writer.add_histogram("cov/Z_hist", cov_z, iteration, 'auto')
            # tb_writer.add_histogram("cov/mean_hist", cov_mean, iteration, 'auto')
            
            # tb_writer.add_histogram("cov/X_hist_negLog", -torch.log(cov_x), iteration, 'auto')
            # tb_writer.add_histogram("cov/Y_hist_negLog", -torch.log(cov_y), iteration, 'auto')
            # tb_writer.add_histogram("cov/Z_hist_negLog", -torch.log(cov_z), iteration, 'auto')
            # tb_writer.add_histogram("cov/mean_hist_negLog", -torch.log(cov_mean), iteration, 'auto')
            
            # tb_writer.add_scalar('cov/X_mean', torch.mean(cov_x), iteration)
            # tb_writer.add_scalar('cov/Y_mean', torch.mean(cov_y), iteration)
            # tb_writer.add_scalar('cov/Z_mean', torch.mean(cov_z), iteration)
            # tb_writer.add_scalar('cov/mean', torch.mean(cov_mean), iteration)
            
            # Plot only 90% of covaraince since high skewness (outlier)
            cov_x = cov_x[cov_x < torch.quantile(cov_x, 0.9)]
            cov_y = cov_y[cov_y < torch.quantile(cov_y, 0.9)]
            cov_z = cov_z[cov_z < torch.quantile(cov_z, 0.9)]
            cov_mean = cov_mean[cov_mean < torch.quantile(cov_mean, 0.9)]
            
            # tb_writer.add_histogram("cov(90%)/X_hist", cov_x, iteration, 'auto')
            # tb_writer.add_histogram("cov(90%)/Y_hist", cov_y, iteration, 'auto')
            # tb_writer.add_histogram("cov(90%)/Z_hist", cov_z, iteration, 'auto')
            tb_writer.add_histogram("cov(90%)/mean_hist", cov_mean, iteration, 'auto')
            
            # Negative log histogram for better visualization
            # tb_writer.add_histogram("cov(90%)/X_hist_negLog", -torch.log(cov_x), iteration, 'auto')
            # tb_writer.add_histogram("cov(90%)/Y_hist_negLog", -torch.log(cov_y), iteration, 'auto')
            # tb_writer.add_histogram("cov(90%)/Z_hist_negLog", -torch.log(cov_z), iteration, 'auto')
            tb_writer.add_histogram("cov(90%)/mean_hist_negLog", -torch.log(cov_mean), iteration, 'auto')
            
            tb_writer.add_scalar('cov(90%)/X_mean', torch.mean(cov_x), iteration)
            tb_writer.add_scalar('cov(90%)/Y_mean', torch.mean(cov_y), iteration)
            tb_writer.add_scalar('cov(90%)/Z_mean', torch.mean(cov_z), iteration)
            tb_writer.add_scalar('cov(90%)/mean', torch.mean(cov_mean), iteration)
            
            tb_writer.add_histogram("opacity/hist", scene.gaussians.get_opacity, iteration)
            

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}
                              )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0] or deblur:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    if deblur:
                        gt_image = torch.clamp(viewpoint.original_image_deblur.to("cuda"), 0.0, 1.0)
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                with open(os.path.join(args.output_path, args.exp_name, 'log_file.txt'), 'a') as file:
                    # Write the log information to the file.
                    file.write("\n[ITER {}] Evaluating {}: L1 {} PSNR {}\n".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    


        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    try :
        # Set up command line argument parser
        parser = ArgumentParser(description="Training script parameters")
        lp = ModelParams(parser)
        op = OptimizationParams(parser)
        pp = PipelineParams(parser)
        parser.add_argument('--ip', type=str, default="127.0.0.1")
        parser.add_argument('--port', type=int, default=6009)
        parser.add_argument('--debug_from', type=int, default=-1)
        parser.add_argument('--detect_anomaly', action='store_true', default=False)
        parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
        parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
        parser.add_argument("--quiet", action="store_true")
        parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
        parser.add_argument("--start_checkpoint", type=str, default = None)
        parser.add_argument("--exp_name", type=str, default=None)
        parser.add_argument("--output_path", type=str, default='./output/')
        parser.add_argument("--project_name", type=str, default="gaussian-splatting")
        parser.add_argument("--rgb_fix", action="store_true", default = False, help="Fix the gradient of RGB features of the Gaussians")
        parser.add_argument("--xyz_fix", action="store_true", default = False, help="Fix the gradient of XYZ features of the Gaussians")
        parser.add_argument("--no_densify", action="store_true", default = False, help="Disable densification")
        parser.add_argument("--no_reset_opacity", action="store_true", default = False, help="Disable opacity reset")
        parser.add_argument("--reset_opacity_threshold", type=float, default=0.005, help="Threshold for opacity reset")
        parser.add_argument("--random_gaussian", action="store_true", default = False, help="Randomize the point cloud")
        parser.add_argument("--cuda", type=str, default=None)

        # Covariance Loss Arguments
        parser.add_argument("--cov_loss", type=str, default=None, help="Set 'high' if you want to maximize the covariance loss, 'low' if you want to minimize the covariance loss")
        parser.add_argument("--cov_loss_type", type=str, default='scale', help = "one of 'scale', 'rot', 'cov'")
        parser.add_argument("--cov_weight", type=float, default=10.0, help="Weight for covariance loss")
        
        # Deblurring Arguments
        parser.add_argument("--deblur", action="store_true", default = False, help="Enable Coarse to Fine deblurring")
        parser.add_argument("--deblur_step", type = int, default=10, help="Number of deblurring steps")
        parser.add_argument("--deblur_every_iter", type=int, default=7000, help="Deblur every n iterations")
        
        # Blur Arguments
        parser.add_argument("--blur", action="store_true", default = False, help="Enable Gaussian blur")
        parser.add_argument("--gaussian_blur", action="store_true", default = False, help="Enable Gaussian blur")
        parser.add_argument("--filter_size", type=int, default=31, help="Filter size for blur")
        
        # Spherical Harmonics Arguments
        parser.add_argument("--no_sh", action="store_true", default = False, help="Disable Spherical Harmonics")
        

        args = parser.parse_args(sys.argv[1:])
        args.test_iterations.append(args.iterations)
        
        # For every end of deblurring, do testing
        if args.deblur:
            tmp = args.deblur_every_iter
            while tmp < args.iterations:
                args.test_iterations.append(tmp)
                tmp += args.deblur_every_iter
            
        args.save_iterations.append(args.iterations)
        
        # Set cuda device if args.cuda is set
        if args.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
        
        # filter_size should be odd
        if args.filter_size % 2 == 0:
            print(f'Filter size should be odd, adding 1 to filter size : {args.filter_size + 1}')
            args.filter_size += 1
        
        print("Optimizing " + args.model_path)
        print(f'args.densify_until_iter : {args.densify_until_iter}')

        # Initialize system state (RNG)
        safe_state(args.quiet)

        # Start GUI server, configure and run training
        while True :
            try:
                network_gui.init(args.ip, args.port)
                print(f"GUI server started at {args.ip}:{args.port}")
                break
            except Exception as e:
                args.port = args.port + 1
                print(f"Failed to start GUI server, retrying with port {args.port}...")
        
        torch.autograd.set_detect_anomaly(args.detect_anomaly)
        
        training(lp.extract(args), op.extract(args), pp.extract(args), **vars(args))
        

        # All done
        print("\nTraining complete.")
    except Exception as e:
        import pdb
        import sys, traceback
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
