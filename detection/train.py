from __future__ import absolute_import

import os
import argparse
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger

from utils import box_api
from utils.misc import search_latest_checkpoint

from detection.lightning_model import DetectionModel
from detection.mnist_data_module import DetMNISTDataModule
from core.temporal import SequenceWise
import arch




def get_model(model_name, num_layers=3):
    if model_name == 'SparseInputPerceiver':
        model = getattr(arch, model_name)(3, 11 + 4, depth=num_layers)
    elif model_name == 'UnetConv':
        model = getattr(arch, model_name)(3, 11 + 4, num_layers_enc=4, num_layers_dec=2)
    else:
        model = getattr(arch, model_name)(3, 11 + 4, num_layers=num_layers, dropout=0.0)
    if model_name == 'ViT' or model_name == 'DetViT' or model_name == 'CNN4' or model_name == 'UnetConv':
        model = SequenceWise(model)
    return model


def train_mnist(train_dir, model_name, num_layers=3, lr=1e-3, height=64, width=64, max_epochs=100, tbins=12, batch_size=64, num_classes=11, num_workers=2, max_frames_per_video=20,
    demo_every=2, val_every=1,
    max_frames_per_epoch=10000, val_max_frames_per_epoch=5000, min_objects=1, max_objects=2, precision=32, resume=False, just_val=False, just_demo=False,
    eos_coef=0.1, bbox_loss_coef=1, giou_loss_coef=1, cost_class=1, cost_bbox=5, cost_giou=2
    ):
    """
    Example:

    >> python3 segmentation/train.py test_drive model_name --max_frames_per_video 16 --max_frames_per_epoch 50000 --height 64 --width 64

    Last to have worked well:
    >> python3 detection/train.py det_exps/vit_2d_pos_encoding_small_batches_accumulate DetViT --max_frames_per_epoch 200000 --height 128 --width 128 --tbins 1 --batch_size 32 --num_workers 0
    """

    params = argparse.Namespace(**locals())
    net = get_model(model_name, num_layers)

    model = DetectionModel(net, params)
    dm = DetMNISTDataModule(params)

    if resume or just_demo or just_val:
        ckpt = search_latest_checkpoint(train_dir)
        print("resume from: ", ckpt)
    else:
        ckpt = None

    if params.just_demo or params.just_val:
        checkpoint = torch.load(ckpt)
        model.load_state_dict(checkpoint['state_dict'])

    tmpdir = os.path.join(train_dir, 'checkpoints')
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, filename='weights#{epoch}', save_top_k=None, period=1)

    logger = TestTubeLogger(
        save_dir=os.path.join(train_dir, 'logs'),
        version=1)

    if just_demo:
        checkpoint = torch.load(ckpt)
        model.cuda()
        model.load_state_dict(checkpoint['state_dict'])
        model.demo_video(dm.train_dataloader(), show_video=True)
    elif just_val:
        pl.Trainer().test(model, test_dataloaders=dm.val_dataloader())
    else:
        trainer = pl.Trainer(checkpoint_callback=checkpoint_callback, logger=logger, gpus=1, precision=precision, resume_from_checkpoint=ckpt, check_val_every_n_epoch=val_every, reload_dataloaders_every_epoch=True, accumulate_grad_batches=8)
        trainer.fit(model, dm)



if __name__ == "__main__" :
    import fire
    fire.Fire(train_mnist)
