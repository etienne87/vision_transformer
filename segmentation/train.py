import os
import argparse
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger

from segmentation.lightning_model import SegmentationModel
from segmentation.mnist_data_module import SegMNISTDataModule
from core.temporal import SequenceWise
from utils.main_tools import search_latest_checkpoint
from utils.main_tools import ModelCallbacks

import arch

def unet_conv(num_layers):
    return SequenceWise(arch.UnetConv(3,11,num_layers_enc=num_layers, num_layers_dec=num_layers))

def vit(num_layers):
    return SequenceWise(arch.ViT(3,11,num_layers=num_layers))

def get_model(model_name, num_layers=3):
    fun = globals()[model_name]
    return fun(num_layers)




def train_mnist(train_dir, model_name, num_layers=3, lr=1e-3, height=64, width=64, max_epochs=100, tbins=1, batch_size=64, num_classes=11, num_workers=2, max_frames_per_video=100,
    demo_every=2,
    max_frames_per_epoch=10000, val_max_frames_per_epoch=1000, max_objects=1, precision=32, resume=False, just_demo=False):
    """
    Example:

    >> python3 segmentation/train.py test_drive model_name --max_frames_per_video 16 --max_frames_per_epoch 50000 --height 64 --width 64
    """

    params = argparse.Namespace(**locals())
    net = get_model(model_name)

    model = SegmentationModel(net, params)
    dm = SegMNISTDataModule(params)

    if resume or just_demo:
        ckpt = search_latest_checkpoint(train_dir)
    else:
        ckpt = None

    tmpdir = os.path.join(train_dir, 'checkpoints')
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, filename='weights#{epoch}', save_top_k=None, period=1)

    logger = TestTubeLogger(
        save_dir=os.path.join(train_dir, 'logs'),
        version=1)

    callbacks = [ModelCallback(params.demo_every), checkpoint_callback]

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
