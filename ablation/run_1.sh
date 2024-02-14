python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/bicycle -r 8 --exp_name deblur51 --cuda 0 --blur --deblur --filter_size 51 --deblur_every_iter 6000 --iterations 40000

python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/bicycle -r 8 --exp_name deblur51_noDensify --cuda 0 --blur --no_densify --deblur --filter_size 51 --deblur_every_iter 6000 --iterations 40000

python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/bicycle -r 8 --exp_name deblur51_xyzfix_noDensify --cuda 0 --xyz_fix --no_densify --blur --deblur --filter_size 51 --deblur_every_iter 6000 --iterations 40000

python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/bicycle -r 8 --exp_name deblur51_random --random_gaussian --cuda 0 --blur --deblur --filter_size 51 --deblur_every_iter 6000 --iterations 40000

python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/bicycle -r 8 --exp_name deblur51_random_noDensify --random_gaussian --cuda 0 --blur --no_densify --deblur --filter_size 51 --deblur_every_iter 6000 --iterations 40000

python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/bicycle -r 8 --exp_name deblur51_random_xyzfix_noDensify --random_gaussian --cuda 0 --xyz_fix --no_densify --blur --deblur --filter_size 51 --deblur_every_iter 6000 --iterations 40000