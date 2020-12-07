"""
Toy Problem Dataset that serves as an example
of our streamer dataloader.

This time it is detection
"""
from __future__ import absolute_import

import sys
import time
import cv2
import torch
import numpy as np
import uuid

from moving_mnist import moving_box as toy
from moving_mnist.multistream_dataloader import MultiStreamDataLoader, MultiStreamDataset

from torchvision import datasets, transforms
from functools import partial

TRAIN_DATASET = datasets.MNIST('/tmp/mnist', train=True, download=True,
                                       transform=transforms.Compose([transforms.ToTensor(),
                                                                     transforms.Normalize((0.1307,), (0.3081,))]))
VAL_DATASET = datasets.MNIST('/tmp/mnist', train=False, download=True,
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
        self.ids = [i for i in range(255)]
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
                targets.append(target)

            imgs = np.concatenate(imgs, axis=0)

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


class MovingMNISTDetDataset(MultiStreamDataLoader):
    """Multi-MNIST Animations Parallel Streamer

    Args:
        datasets: MultiStream datasets with MovingMnist streamers
        collate_fn: above batch collation function
        parallel: use several workers or not
    """
    def __init__(self, datasets, collate_fn, parallel=True):
        super().__init__(datasets, collate_fn, parallel)
        self.vis_func = lambda img:(np.moveaxis(img, 0, 2).copy()*255).astype(np.int32)

    def get_vis_func(self):
        return self.vis_func

    def __len__(self):
        """Here we know in advance the exact duration of each stream.
        In practice you can either discard this function or make an estimate using a few streams.
        """
        args = self.datasets[0].stream_kwargs
        mini_epochs = len(self.datasets[0].stream_list) // self.datasets[0].batch_size
        max_batch_per_epoch = mini_epochs * args['max_frames_per_video'] // args['tbins']
        return max_batch_per_epoch


def make_moving_mnist(tbins=10, num_workers=1, batch_size=8, height=256, width=256,
                      max_frames_per_video=100, max_frames_per_epoch=1000, min_objects=1, max_objects=2, train=True, random_seed=0):
    """Creates the dataloader for moving mnist

    Args:
        tbins: number of steps per batch
        num_workers: number of parallel workers
        batch_size: number of animations
        height: animation height
        width: animation width
        max_frames_per_video: maximum frames per animation
        max_frames_per_epoch: maximum frames per epoch
        train: use training part of MNIST dataset.
    """
    if train:
        print('DATALOADER IS IN TRAINING MODE')
    else:
        print('DATALOADER IS IN VALIDATION MODE')
    max_frames_per_video = max(max_frames_per_video, tbins)
    height, width, cin = height, width, 3
    n = max_frames_per_epoch // max_frames_per_video
    dummy_list = list(range(n))
    parallel = num_workers > 0

    datasets = MultiStreamDataset.split_datasets(dummy_list, batch_size=batch_size, max_workers=num_workers, streamer=MovingMnist, random_seed=random_seed, tbins=tbins, max_frames_per_video=max_frames_per_video, height=height, width=width, min_objects=min_objects, max_objects=max_objects, train=train)
    dataset = MovingMNISTDetDataset(datasets, collate_fn, parallel=parallel)
    dataset.label_map = [str(i) for i in range(10)]

    return dataset, dataset.label_map
