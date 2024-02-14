#!/bin/bash

for thresh in 0.005 0.01 0.05 0.1; do
    python train.py -s /mnt/sda/home/cvlab03/project/dataset/colmap/bicycle -r 8 --exp_name baseline_random_$thresh --random_gaussian --cuda 1 --eval --reset_opacity_threshold $thresh
done

