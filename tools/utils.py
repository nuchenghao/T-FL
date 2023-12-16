import time

import numpy as np
import torch
import torchvision
from torchvision import transforms



class Accumulator:
    """For accumulating sums over `n` variables."""

    def __init__(self, n):
        """Defined in :numref:`sec_utils`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



class Timer:
    """Record multiple running times."""

    def __init__(self):
        """Defined in :numref:`sec_minibatch_sgd`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()



def load_data_fashion_mnist(batch_size, which, resize=None):
    """Download the Fashion-MNIST dataset and then load it into memory.

    Defined in :numref:`sec_utils`"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    if which == 'train':
        mnist_train = torchvision.datasets.FashionMNIST(
            root="./fashionmnist", train=True, transform=trans, download=True)
        return torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True)
    if which == 'test':
        mnist_test = torchvision.datasets.FashionMNIST(
            root="./fashionmnist", train=False, transform=trans, download=True)
        return torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=False)
