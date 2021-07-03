"""
Lightning-DataModule
"""
import pytorch_lightning as pl
import torchvision

import argparse
import cv2

from moving_mnist.moving_mnist_detection import MovingMNISTDetDataset
from torchvision.utils import make_grid

import numpy as np
import random
import inspect


class DetMNISTDataModule(pl.LightningDataModule):

    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.hparams = hparams
        a_signature = inspect.signature(MovingMNISTDetDataset)
        parameters = a_signature.parameters
        parameter_list = list(parameters)
        self.co_args = set(self.hparams.__dict__.keys()).intersection(parameter_list)

    def train_dataloader(self):
        seed = random.randint(0, 100)
        kwargs = {k:self.hparams.__dict__[k] for k in self.co_args}
        if 0: #random scaling in training time
            scale = np.random.randint(0, 2)
            kwargs['height'] = 64 * 2**scale
            kwargs['width'] = 64 * 2**scale
            print('Size: height, width: ', kwargs['height'], kwargs['width'])
        kwargs['train'] = True
        train_dataloader = MovingMNISTDetDataset(**kwargs)
        return train_dataloader

    def val_dataloader(self):
        seed = random.randint(0, 100)
        kwargs = {k:self.hparams.__dict__[k] for k in self.co_args}
        kwargs['max_frames_per_epoch'] = self.hparams.val_max_frames_per_epoch
        kwargs['train'] = False
        val_dataloader = MovingMNISTDetDataset(**kwargs)
        return val_dataloader
