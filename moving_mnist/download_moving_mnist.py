"""
MNIST Data
"""
from __future__ import absolute_import

import urllib.request
import tarfile
import os
import shutil
from torchvision import datasets, transforms


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


TRAIN_DATASET = datasets.MNIST(PATH, train=True, download=False,
                                       transform=transforms.Compose([transforms.ToTensor(),
                                                                     transforms.Normalize((0.1307,), (0.3081,))]))
VAL_DATASET = datasets.MNIST(PATH, train=False, download=False,
                                       transform=transforms.Compose([transforms.ToTensor(),
                                                                     transforms.Normalize((0.1307,), (0.3081,))]))


