# python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/bicycle -r 8 --exp_name baseline --cuda 0

# python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/bicycle -r 8 --exp_name baseline_noDensify --cuda 0 --no_densify

# python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/bicycle -r 8 --exp_name baseline_xyzfix_noDensify --cuda 0 --xyz_fix --no_densify

# python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/bicycle -r 8 --exp_name baseline_random --random_gaussian --cuda 0

# python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/bicycle -r 8 --exp_name baseline_random_noDensify --random_gaussian --cuda 0 --no_densify

# python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/bicycle -r 8 --exp_name baseline_random_xyzfix_noDensify --random_gaussian --cuda 0 --xyz_fix --no_densify

# python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/bicycle -r 8 --exp_name blur31 --cuda 0 --blur

# python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/bicycle -r 8 --exp_name blur31_noDensify --cuda 0 --blur --no_densify

# python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/bicycle -r 8 --exp_name blur31_xyzfix_noDensify --cuda 0 --xyz_fix --no_densify --blur

# python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/bicycle -r 8 --exp_name blur31_random --random_gaussian --cuda 0 --blur

# python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/bicycle -r 8 --exp_name blur31_random_noDensify --random_gaussian --cuda 0 --blur --no_densify

# python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/bicycle -r 8 --exp_name blur31_random_xyzfix_noDensify --random_gaussian --cuda 0 --xyz_fix --no_densify --blur

python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/bicycle -r 8 --exp_name deblur31 --cuda 0 --blur --deblur

python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/bicycle -r 8 --exp_name deblur31_noDensify --cuda 0 --blur --no_densify --deblur

python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/bicycle -r 8 --exp_name deblur31_xyzfix_noDensify --cuda 0 --xyz_fix --no_densify --blur --deblur

python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/bicycle -r 8 --exp_name deblur31_random --random_gaussian --cuda 0 --blur --deblur

python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/bicycle -r 8 --exp_name deblur31_random_noDensify --random_gaussian --cuda 0 --blur --no_densify --deblur

python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/bicycle -r 8 --exp_name deblur31_random_xyzfix_noDensify --random_gaussian --cuda 0 --xyz_fix --no_densify --blur --deblur