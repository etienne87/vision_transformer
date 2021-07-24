"""
Lightning-DataModule
"""
import pytorch_lightning as pl
import torchvision

import argparse
import cv2
import random

from moving_mnist.moving_mnist_segmentation import MovingMNISTSegDataset
from torchvision.utils import make_grid


class SegMNISTDataModule(pl.LightningDataModule):

    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.hparams = hparams
        a_signature = inspect.signature(MovingMNISTSegDataset)
        parameters = a_signature.parameters
        parameter_list = list(parameters)
        self.co_args = set(self.hparams.__dict__.keys()).intersection(parameter_list)

    def train_dataloader(self):
        seed = random.randint(0, 100)
        kwargs = {k:self.hparams.__dict__[k] for k in self.co_args}
        kwargs['random_seed'] = random.randint(0, 100)
        kwargs['train'] = True
        train_dataloader, _ = MovingMNISTSegDataset(**kwargs)
        return train_dataloader

    def val_dataloader(self):
        seed = random.randint(0, 100)
        kwargs = {k:self.hparams.__dict__[k] for k in self.co_args}
        kwargs['max_frames_per_epoch'] = self.hparams.val_max_frames_per_epoch
        kwargs['random_seed'] = random.randint(0, 100)
        kwargs['train'] = False
        dataloader, _ = MovingMNISTSegDataset(**kwargs)
        return dataloader
