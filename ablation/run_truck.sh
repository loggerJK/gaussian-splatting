python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/truck -r 8 --exp_name baseline --cuda 1

python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/truck -r 8 --exp_name baseline_noDensify --cuda 1 --no_densify

python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/truck -r 8 --exp_name baseline_xyzfix_noDensify --cuda 1 --xyz_fix --no_densify

python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/truck -r 8 --exp_name baseline_random --random_gaussian --cuda 1

python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/truck -r 8 --exp_name baseline_random_noDensify --random_gaussian --cuda 1 --no_densify

python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/truck -r 8 --exp_name baseline_random_xyzfix_noDensify --random_gaussian --cuda 1 --xyz_fix --no_densify

python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/truck -r 8 --exp_name blur31 --cuda 1 --blur

python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/truck -r 8 --exp_name blur31_noDensify --cuda 1 --blur --no_densify

python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/truck -r 8 --exp_name blur31_xyzfix_noDensify --cuda 1 --xyz_fix --no_densify --blur

python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/truck -r 8 --exp_name blur31_random --random_gaussian --cuda 1 --blur

python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/truck -r 8 --exp_name blur31_random_noDensify --random_gaussian --cuda 1 --blur --no_densify

python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/truck -r 8 --exp_name blur31_random_xyzfix_noDensify --random_gaussian --cuda 1 --xyz_fix --no_densify --blur