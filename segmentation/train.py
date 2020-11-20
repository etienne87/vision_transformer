import os
import argparse
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger

from segmentation.lightning_model import SegmentationModel
from segmentation.utils import search_latest_checkpoint
from core.temporal import SequenceWise
from arch.vit import ViT



def train_moving_mnist_segmentation(train_dir, height=128, width=128, max_epochs=100, num_tbins=10, batch_size=64, num_classes=11, num_workers=1, max_frames_per_video=10,
                                    max_frames_per_epoch=10000, max_objects=1, resume=False, just_demo=False):
    """
    Example: 

    >> python3 segmentation/train.py test_drive --max_frames_per_video 16 --max_frames_per_epoch 50000 --height 64 --width 64
    """

    params = argparse.Namespace(**locals())
    net = SequenceWise(ViT(3,11))
    model = SegmentationModel(net, params)

    if resume:
        ckpt = search_latest_checkpoint(train_dir)
    else:
        ckpt = None
    
    tmpdir = os.path.join(train_dir, 'checkpoints')
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, period=2)

    logger = TestTubeLogger(
        save_dir=os.path.join(train_dir, 'logs'),
        version=1)

    if just_demo:
        checkpoint = torch.load(ckpt)
        unet.cuda()
        unet.load_state_dict(checkpoint['state_dict'])
        unet.demo_video()
    else:
        trainer = pl.Trainer(checkpoint_callback=checkpoint_callback, logger=logger, gpus=1,precision=32,resume_from_checkpoint=ckpt)
        trainer.fit(model)

  
  
if __name__ == "__main__" :
    import fire
    fire.Fire(train_moving_mnist_segmentation)
