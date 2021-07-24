"""
main tools
"""
import os
import glob

def extract_num(path):
    filename = os.path.splitext(path)[0]
    num = filename.split('=')[1]
    return int(num) if num.isdigit() else -1


def search_latest_checkpoint(root_dir):
    """looks for latest checkpoint in latest sub-directory"""
    vdir = os.path.join(root_dir, 'checkpoints')
    ckpts = sorted(glob.glob(os.path.join(vdir, '*.ckpt')), key=extract_num)
    return ckpts[-1]
