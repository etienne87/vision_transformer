"""
Toy Problem Dataset that serves as an example
of our streamer dataloader.

It is the same as "moving_mnist_detection", but
this times sends a mask.
"""
from __future__ import absolute_import

import sys
import time
import cv2
import torch
import numpy as np
import uuid

from moving_mnist import moving_box as toy
from moving_mnist.download_moving_mnist import TRAIN_DATASET, VAL_DATASET
from pytorch_stream_dataloader.stream_dataloader import StreamDataLoader
from pytorch_stream_dataloader.stream_dataset import StreamDataset

from torchvision import datasets, transforms
from functools import partial




class MovingMnist(toy.Animation):
    """Moving Mnist Animation

    Args:
        idx: unique id
        tbins: number of steps delivered at once
        height: frame height
        width: frame width
        channels: 1 or 3 gray or color
        max_stop: random pause in animation
        min_objects: minimum number of objects per animation
        max_objects: maximum number of objects per animation
        train: use training/ validation part of MNIST
        max_frames_per_video: maximum frames per video before reset
    """
    def __init__(self, idx, tbins=10, height=128, width=128, channels=3, max_stop=15,
            min_objects=1, max_objects=2,
            train=True, max_frames_per_video=100, colorization_problem=False, data_caching_path='/tmp/mnist_data'):
        self.dataset_ = TRAIN_DATASET if train else VAL_DATASET
        self.channels = channels
        self.steps = 0
        self.tbins = tbins
        self.max_frames_per_video = max_frames_per_video
        max_classes = 10
        self.video_info = str(uuid.uuid4())
        self.label_img = np.zeros((height, width), dtype=np.uint8)
        self.colorization_problem = colorization_problem
        super(MovingMnist, self).__init__(height, width, channels, max_stop, max_classes, min_objects, max_objects)
        self.label_offset = 1

    def reset(self):
        super(MovingMnist, self).reset()
        self.steps = 0
        for i in range(len(self.objects)):
            idx = np.random.randint(0, len(self.dataset_))
            x, y = self.dataset_[idx]
            self.objects[i].class_id = y
            self.objects[i].id = np.random.randint(1, 100)
            self.objects[i].idx = idx
            img = x.numpy()[0]
            img = (img-img.min())/(img.max()-img.min())
            abs_img = np.abs(img)
            y, x = np.where(abs_img > 0.45)
            x1, x2 = np.min(x), np.max(x)
            y1, y2 = np.min(y), np.max(y)
            self.objects[i].img = np.repeat(img[y1:y2, x1:x2][...,None], self.channels, 2)

    def step(self):
        self.img[...] = 0
        self.label_img[...] = 0
        boxes = np.zeros((len(self.objects), 5), dtype=np.float32)
        for i, digit in enumerate(self.objects):
            x1, y1, x2, y2 = next(digit)
            boxes[i] = np.array([x1, y1, x2, y2, digit.class_id + self.label_offset])
            thumbnail = cv2.resize(digit.img, (x2-x1, y2-y1), cv2.INTER_LINEAR)
            #this line is to disambiguate a bit: objects are in different gray level
            #thumbnail2 = thumbnail * ((i+1)/len(self.objects))

            #otherwise if you want you can be voluntarily ambiguous: test the memory
            thumbnail2 = thumbnail
            self.img[y1:y2, x1:x2] = np.maximum(self.img[y1:y2, x1:x2], thumbnail2)
            thumbnail = thumbnail.mean(axis=2)
            mask_thumb = np.zeros((thumbnail.shape[0], thumbnail.shape[1]), dtype=np.uint8)
            if self.colorization_problem:
                print('fuuuuuck')
                mask_thumb[thumbnail >= 0.1] = digit.id
            else:
                mask_thumb[thumbnail >= 0.1] = digit.class_id + self.label_offset
            self.label_img[y1:y2, x1:x2] = mask_thumb * (mask_thumb > 0) + self.label_img[y1:y2, x1:x2] * (mask_thumb == 0)

        output = self.img
        mask = self.label_img
        self.steps += 1
        return (output, mask)

    def __iter__(self):
        for r in range(self.max_frames_per_video//self.tbins):
            reset = self.steps > 0
            imgs, targets = [], []
            for t in range(self.tbins):
                img, target = self.step()
                imgs.append(img[None].copy())
                targets.append(target[None].copy())

            imgs = np.concatenate(imgs, axis=0)
            masks = np.concatenate(targets, axis=0)

            imgs = torch.from_numpy(imgs)
            masks = torch.from_numpy(masks)

            video_info = (self.video_info, self.steps)
            yield imgs, masks, reset, video_info


def collate_fn(data_list):
    """collate_fn
    this collates batch parts to a single dictionary
    Args:
        data_list: batch parts
    """
    batch, masks, resets, video_infos = zip(*data_list)
    batch = torch.cat([item[:, None] for item in batch], dim=1)
    batch = batch.permute(0,1,4,2,3).contiguous()
    t, n = batch.shape[:2]
    masks = torch.cat([item[:, None] for item in masks], dim=1)
    resets = torch.FloatTensor(resets)[:,None,None,None]
    frame_is_labelled = torch.ones((t, n), dtype=torch.float32)
    return {'inputs': batch, 'labels': masks, "frame_is_labelled": frame_is_labelled,
            'mask_keep_memory': resets, "video_infos": video_infos}



class MovingMNISTSegDataset(StreamDataLoader):
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

    def get_vis_func(self):
        return self.vis_func


