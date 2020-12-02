import os
import glob
import torch
import numpy as np
import torch.distributed as dist


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


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


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def cuda_tick():
    torch.cuda.synchronize()
    return time.time()

def cuda_time(func):
    def wrapper(*args, **kwargs):
        start = cuda_tick()
        out = func(*args, **kwargs)
        end = cuda_tick()
        rt = end-start
        freq = 1./rt
        if freq > 0:
            print(freq, ' it/s @ ', func)
        else:
            print(rt, ' s/it @ ', func)
        return out
    return wrapper
