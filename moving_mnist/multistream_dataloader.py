"""
Module that enables Parallel Multistreaming.
Can be used in alternative to the Sequential Dataset,
when it is important to avoid seeking and reopening files.


To fully understand the code here, you might want to read:

(1) https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd
"""
import glob
import torch
import numpy as np
import threading
import cv2
import time
import random

from torch.utils.data import IterableDataset, DataLoader

from torch._six import queue, container_abcs, string_classes
from torch.utils.data._utils.pin_memory import pin_memory
from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL
from torch._utils import ExceptionWrapper

from itertools import chain 

        
class MultiStreamDataset(IterableDataset):
    '''The class simply iterates several python iterators.
    Here the iterators are *not* partitioned into a balanced partition.
    One can change this to simply go through randomly over the entire dataset like in
    (1)
    
    Args:
        stream_list: list of files to stream
        streamer: user's class to instanciate per file to stream
        batch_size: number of streams
        stream_kwargs: user's class ctor arguments
    '''
    def __init__(self, stream_list, streamer, batch_size=4, partition=False, random_seed=0, **stream_kwargs):
        self.stream_list = stream_list
        self.batch_size = batch_size
        self.streamer = streamer
        self.stream_kwargs = stream_kwargs
        self.partition = partition
        self.seed = random_seed


    @property
    def shuffled_data_list(self):
        return random.sample(self.stream_list, len(self.stream_list))

    def chain_streamers(self, data_list):
        streamers = iter(self.streamer(data, **self.stream_kwargs) for data in data_list)
        return chain.from_iterable(streamers)

    def __iter__(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        return self.iter_partition() if self.partition else self.iter_random()

    def iter_random(self):
        return zip(*[self.chain_streamers(self.shuffled_data_list) for i in range(self.batch_size)]) 

    def iter_partition(self):
        stream_list = random.sample(self.stream_list, len(self.stream_list)) +\
                      random.choices(self.stream_list,k=self.batch_size-len(self.stream_list)%self.batch_size)
        chunk_size = len(stream_list) // self.batch_size
        return zip(
            *[self.chain_streamers(stream_list[i*chunk_size:(i+1)*chunk_size]) for i in range(self.batch_size)]
        )
        
    @classmethod
    def make_datasets(cls, stream_list, batch_size, max_workers, streamer, **stream_kwargs):
        max_workers = max(1, max_workers)
        for n in range(max_workers, 0, -1):
            if batch_size % n == 0:
                num_workers = n
                break
        split_size = batch_size // num_workers
        out = [cls(stream_list, streamer, split_size, **stream_kwargs) for _ in range(num_workers)]
        return out
    
    @classmethod
    def split_datasets(cls, stream_list, batch_size, max_workers, streamer, random_seed, **stream_kwargs):
        max_workers = min(len(stream_list), max(1, max_workers))
        for n in range(max_workers, 0, -1):
            if batch_size % n == 0:
                num_workers = n
                break
        print('using',num_workers,'workers')
        #here we partition the original data_list.
        split_size = batch_size // num_workers
        num_files_per_worker = len(stream_list) // num_workers
        out = []
        for i in range(num_workers):
            start = i * num_files_per_worker
            end = (i + 1) * num_files_per_worker
            if i == num_workers-1:
                stream_files = stream_list[start:]
            else:
                stream_files = stream_list[start:end]
            item = cls(stream_files, streamer=streamer, batch_size=split_size, random_seed=random_seed + i, **stream_kwargs)
            out.append(item)

        return out


class MultiStreamDataLoader:
    '''MultiStreamDataLoader for RNN-like training.

    Several Dataloaders, where every dataloader is responsible for a group of streams.
    One dataloader can only use 1 worker, otherwise video-clips are not contiguous across batches.

    Args:
        datasets: 1 iterable dataset per dataloader (with only 1 worker per dataloader)
        collate_fn: to collate batch parts. note that dataloader internal collate is the default one.
        parallel: if set to False it sets num_workers to 0 for each dataloader. otherwise every 
        dataloader's num_workers is set to 1.
    '''
    def __init__(self, datasets, collate_fn, parallel=True):
        self.datasets = datasets
        self.collate_fn = collate_fn
        self.parallel = parallel

    def get_stream_loaders(self):
        dataloaders = [
            DataLoader(dataset, num_workers=self.parallel, batch_size=None, pin_memory=True)
            for dataset in self.datasets
        ]
        return zip(*dataloaders)

    def __iter__(self):
        for batch_parts in self.get_stream_loaders():
            data = list(chain(*batch_parts))
            batch = self.collate_fn(data)
            yield batch
