"""
main tools
"""
import os
import glob
import pytorch_lightning as pl


class ModelCallback(pl.callbacks.base.Callback):
    def __init__(self, data_module, video_every=2):
        super().__init__()
        self.video_every = int(video_every)
        self.data_module = data_module

    def on_epoch_end(self, trainer, pl_module):
        if not trainer.current_epoch % self.video_every:
            pl_module.demo_video(
                self.data_module.test_dataloader(),
                epoch=trainer.current_epoch)


def extract_num(path):
    filename = os.path.splitext(path)[0]
    num = filename.split('=')[1]
    return int(num) if num.isdigit() else -1


def search_latest_checkpoint(root_dir):
    """looks for latest checkpoint in latest sub-directory"""
    vdir = os.path.join(root_dir, 'checkpoints')
    ckpts = sorted(glob.glob(os.path.join(vdir, '*.ckpt')), key=extract_num)
    return ckpts[-1]
