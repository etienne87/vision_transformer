"""
Lightning-DataModule
"""
import pytorch_lightning as pl
import torchvision

import argparse
import cv2

from moving_mnist.moving_mnist_detection import make_moving_mnist
from torchvision.utils import make_grid

import numpy as np
import random


class DetMNISTDataModule(pl.LightningDataModule):

    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.hparams = hparams
        self.co_args = set(self.hparams.__dict__.keys()).intersection(make_moving_mnist.__code__.co_varnames)

    def train_dataloader(self):
        seed = random.randint(0, 100)
        kwargs = {k:self.hparams.__dict__[k] for k in self.co_args}
        kwargs['random_seed'] = random.randint(0, 100)
        train_dataloader, _ = make_moving_mnist(**kwargs) 
        return train_dataloader   
        
    def val_dataloader(self):
        seed = random.randint(0, 100)
        kwargs = {k:self.hparams.__dict__[k] for k in self.co_args}
        kwargs['max_frames_per_epoch'] = self.hparams.val_max_frames_per_epoch
        kwargs['random_seed'] = random.randint(0, 100)
        dataloader, _ = make_moving_mnist(**kwargs)
        return dataloader 
