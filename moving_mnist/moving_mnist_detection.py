"""
Toy Problem Dataset that serves as an example
of our streamer dataloader.

This time it is detection
"""
from __future__ import absolute_import

import urllib.request
import tarfile
import os
import shutil


def download_mnist(url='https://www.di.ens.fr/~lelarge/MNIST.tar.gz',
                  path='/tmp/mnist/'):
    tar_name = os.path.basename(url)
    tar_filename = os.path.join(path, tar_name)
    filename = os.path.join(path, 'MNIST')
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(tar_filename):
        filedata = urllib.request.urlretrieve(url, tar_filename)
    if not os.path.exists(filename):
        shutil.unpack_archive(tar_filename, path+'/')

PATH = '/tmp/mnist'
download_mnist(path=PATH)


import sys
import time
import cv2
import torch
import numpy as np
import uuid

from moving_mnist import moving_box as toy
from pytorch_stream_dataloader.stream_dataloader import StreamDataLoader
from pytorch_stream_dataloader.stream_dataset import StreamDataset

from torchvision import datasets, transforms
from functools import partial

TRAIN_DATASET = datasets.MNIST(PATH, train=True, download=False,
                                       transform=transforms.Compose([transforms.ToTensor(),
                                                                     transforms.Normalize((0.1307,), (0.3081,))]))
VAL_DATASET = datasets.MNIST(PATH, train=False, download=False,
                                       transform=transforms.Compose([transforms.ToTensor(),
                                                                     transforms.Normalize((0.1307,), (0.3081,))]))


class FileMetadata(object):
    """Video Infos
    """
    def __init__(self, path, duration, delta_t, tbins):
        self.path = path
        self.delta_t = delta_t
        self.duration = duration
        self.tbins = tbins
        self.padding = False
        self.start_ts = 0


class MovingMnist(toy.Animation):
    """Moving Mnist Animation

    Args:
        idx: unique id
        tbins: number of steps delivered at once
        height: frame height
        width: frame width
        channels: 1 or 3 gray or color
        max_stop: random pause in animation
        max_objects: maximum number of objects per animation
        train: use training/ validation part of MNIST
        max_frames_per_video: maximum frames per video before reset
    """
    def __init__(self, idx, tbins=10, height=128, width=128, channels=3, max_stop=15,
                 min_objects=1, max_objects=2, train=True, max_frames_per_video=100, data_caching_path="/tmp/mnist_data", colorized=True):
        self.train = train
        self.channels = channels
        self.steps = 0
        self.tbins = tbins
        self.max_frames_per_video = max_frames_per_video
        max_classes = 10
        random_name = str(uuid.uuid4())
        self.video_info = FileMetadata(random_name, 50000 * max_frames_per_video, 50000, tbins)
        self.colorized = colorized
        self.colormap = cv2.applyColorMap(np.array([i for i in range(255)], dtype=np.uint8), cv2.COLORMAP_HSV)
        self.colormap[0] = 0
        super(MovingMnist, self).__init__(height, width, channels, max_stop, max_classes, min_objects, max_objects)
        self.label_offset = 0

    def reset(self):
        super(MovingMnist, self).reset()
        dataset = TRAIN_DATASET if self.train else VAL_DATASET
        self.steps = 0
        self.ids = [i for i in range(1, 255)]
        np.random.shuffle(self.ids)
        self.ids = self.ids[:len(self.objects)]
        for i in range(len(self.objects)):
            idx = np.random.randint(0, len(dataset))
            x, y = dataset[idx]
            self.objects[i].class_id = y
            self.objects[i].idx = idx
            img = x.numpy()[0]
            img = (img-img.min())/(img.max()-img.min())
            abs_img = np.abs(img)
            y, x = np.where(abs_img > 0.45)
            x1, x2 = np.min(x), np.max(x)
            y1, y2 = np.min(y), np.max(y)
            # choose a random color
            id = self.ids[i]
            assert id > 0
            img = img[y1:y2, x1:x2]
            img = (img >= 0.1) * id + (img < 0.1) * 0
            labelrgb = self.colormap[img][:,:,0,:]
            labelrgb = labelrgb / 255.0
            if self.colorized:
                self.objects[i].img = labelrgb
            else:
                self.objects[i].img = np.repeat(img[y1:y2, x1:x2][...,None], self.channels, 2)

    def step(self):
        self.img[...] = 0
        boxes = np.zeros((len(self.objects), 5), dtype=np.float32)
        for i, digit in enumerate(self.objects):
            x1, y1, x2, y2 = next(digit)
            boxes[i] = np.array([x1, y1, x2, y2, digit.class_id + self.label_offset])
            thumbnail = cv2.resize(digit.img, (x2-x1, y2-y1), cv2.INTER_LINEAR)
            self.img[y1:y2, x1:x2] = np.maximum(self.img[y1:y2, x1:x2], thumbnail)
        output = self.img
        self.steps += 1
        return (output, boxes)

    def __iter__(self):
        for r in range(self.max_frames_per_video//self.tbins):
            reset = self.steps > 0
            imgs, targets = [], []
            for t in range(self.tbins):
                img, target = self.step()
                imgs.append(img[None].copy())

                target = torch.from_numpy(target)
                targets.append(target)

            imgs = np.concatenate(imgs, axis=0)
            imgs = torch.from_numpy(imgs)

            video_info = (self.video_info, self.steps * self.video_info.delta_t, self.tbins * self.video_info.delta_t)
            yield imgs, targets, reset, video_info


def collate_fn(data_list):
    """collate_fn
    this collates batch parts to a single dictionary
    Args:
        data_list: batch parts
    """
    batch, boxes, resets, video_infos = zip(*data_list)
    batch = torch.cat([item[:, None] for item in batch], dim=1)
    batch = batch.permute(0,1,4,2,3).contiguous()
    t, n = batch.shape[:2]
    boxes = [[boxes[i][t] for i in range(n)] for t in range(t)]
    resets = torch.FloatTensor(resets)[:,None,None,None]
    frame_is_labelled = torch.ones((t, n), dtype=torch.float32)
    return {'inputs': batch, 'labels': boxes, "frame_is_labelled": frame_is_labelled,
            'mask_keep_memory': resets, "video_infos": video_infos}


class MovingMNISTDetDataset(StreamDataLoader):
    """Creates the dataloader for moving mnist

    Args:
        tbins: number of steps per batch
        num_workers: number of parallel workers
        batch_size: number of animations
        height: animation height
        width: animation width
        max_frames_per_video: maximum frames per animation (must be greater than tbins)
        max_frames_per_epoch: maximum frames per epoch
        train: use training part of MNIST dataset.
    """

    def __init__(self, tbins, num_workers, batch_size, height, width,
                 max_frames_per_video, max_frames_per_epoch, train):

        assert max_frames_per_video >= tbins
        height, width, cin = height, width, 3
        n = max_frames_per_epoch // max_frames_per_video
        stream_list = list(range(n))

        def iterator_fun(idx): return MovingMnist(idx, tbins, height, width, 3, 15, 1, 2, train, max_frames_per_video)

        dataset = StreamDataset(stream_list, iterator_fun, batch_size, "data", None)
        super().__init__(dataset, num_workers, collate_fn)
        self.vis_func = lambda img: (np.moveaxis(img, 0, 2).copy() * 255).astype(np.int32)
        self.label_map = [str(i) for i in range(10)]

    def get_vis_func(self):
        return self.vis_func
