import os
import argparse
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger

from detection.lightning_model import DetectionModel 
from detection.utils import search_latest_checkpoint
from detection.mnist_data_module import DetMNISTDataModule
from core.temporal import SequenceWise
import arch


def get_model(model_name, num_layers=3):
    model = getattr(arch, model_name)(3, 11, num_layers=num_layers, dropout=0.0)
    return SequenceWise(model) if model_name == 'ViT' else model


def train_mnist(train_dir, model_name, num_layers=3, lr=1e-3, height=64, width=64, max_epochs=100, num_tbins=12, batch_size=64, num_classes=11, num_workers=1, max_frames_per_video=20,
    demo_every=2,                                
    max_frames_per_epoch=10000, val_max_frames_per_epoch=1000, max_objects=1, precision=32, resume=False, just_demo=False,
    eos_coef=0.1, bbox_loss_coef=1, giou_loss_coef=1, cost_class=1, cost_bbox=5, cost_giou=2
    ):
    """
    Example: 

    >> python3 segmentation/train.py test_drive model_name --max_frames_per_video 16 --max_frames_per_epoch 50000 --height 64 --width 64
    """

    params = argparse.Namespace(**locals())
    net = get_model(model_name)

    model = DetectionModel(net, params)
    dm = DetMNISTDataModule(params)

    if resume or just_demo:
        ckpt = search_latest_checkpoint(train_dir)
    else:
        ckpt = None
    
    tmpdir = os.path.join(train_dir, 'checkpoints')
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, save_top_k=-1, period=5) 

    logger = TestTubeLogger(
        save_dir=os.path.join(train_dir, 'logs'),
        version=1)

    if just_demo:
        checkpoint = torch.load(ckpt)
        model.cuda()
        model.load_state_dict(checkpoint['state_dict'])
        model.demo_video(dm.val_dataloader())
    else:
        trainer = pl.Trainer(checkpoint_callback=checkpoint_callback, logger=logger, gpus=1, precision=precision, resume_from_checkpoint=ckpt)
        trainer.fit(model, dm)

  
  
if __name__ == "__main__" :
    import fire
    fire.Fire(train_mnist)
