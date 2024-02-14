#!/bin/bash

for cov_weight in 0.1 1.0 10.0 100.0; do
    python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/bicycle -r 8 --exp_name baseline_random_lowcov_$cov_weight --random_gaussian --cuda 3 --cov_loss low --cov_weight $cov_weight --eval
done

