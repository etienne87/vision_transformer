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

    def train_dataloader(self):
        seed = random.randint(0, 100)
        train_dataloader, _ = make_moving_mnist(batch_size=self.hparams.batch_size, 
                                                train=True, 
                                                min_objects=self.hparams.min_objects, 
                                                max_objects=self.hparams.max_objects,
                                                height=self.hparams.height,
                                                width=self.hparams.width,
                                                tbins=self.hparams.num_tbins, 
                                                max_frames_per_video=self.hparams.max_frames_per_video, 
                                                max_frames_per_epoch=self.hparams.max_frames_per_epoch, 
                                                num_workers=self.hparams.num_workers, 
                                                random_seed=seed)
        return train_dataloader   
        
    def val_dataloader(self):
        seed = random.randint(0, 100)
        dataloader, _ = make_moving_mnist(batch_size=self.hpparams.batch_size, 
                                        train=False, 
                                        min_objects=self.hparams.min_objects, 
                                        max_objects=self.hparams.max_objects,
                                        height=self.hparams.height,
                                        width=self.hparams.width,
                                        tbins=self.hparams.num_tbins, 
                                        max_frames_per_video=self.hparams.max_frames_per_video, 
                                        max_frames_per_epoch=self.hparams.val_max_frames_per_epoch, 
                                        num_workers=self.hparams.num_workers, 
                                        random_seed=seed)
        return dataloader 
