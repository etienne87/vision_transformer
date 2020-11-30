import os
import glob
import torch
import numpy as np


def normalize(im):
    low, high = im.min(), im.max()
    return (im - low) / (1e-5 + high - low)
    
def filter_outliers(input_val, num_std=3):
    val_range = num_std * input_val.std()
    img_min = input_val.mean() - val_range
    img_max = input_val.mean() + val_range
    normed = input_val.clamp_(img_min, img_max)
    return normed    

def search_latest_checkpoint(log_dir):
    """looks for latest checkpoint in latest sub-directory"""
    vdir = os.path.join(log_dir, 'checkpoints')
    ckpt = sorted(glob.glob(vdir + '/*.ckpt'))[-1]
    return ckpt

