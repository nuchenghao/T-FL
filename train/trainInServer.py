from collections import OrderedDict

import numpy as np
import torch
import torchvision

from torch import nn
from torchvision import transforms

from tools import utils

argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)


class Trainer:
    def __init__(self, batchSize, net, which):
        self.batchSize = batchSize
        self.net = net
        self.data_iter = self.load_data_fashion_mnist(batchSize, which)
        self.net.initNet()  # 初始化全局模型

    def load_data_fashion_mnist(self, batch_size, which, resize=None):
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

    # test_iter = load_data_fashion_mnist(batchSize, "test")

    def accuracy(self, y_hat, y):
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = argmax(y_hat, axis=1)
        cmp = astype(y_hat, y.dtype) == y
        return float(reduce_sum(astype(cmp, y.dtype)))

    def evaluate_accuracy(self):
        self.net.net.to(self.net.device)
        if isinstance(self.net, nn.Module):
            self.net.net.eval()  # 设置为评估模式

        metric = utils.Accumulator(2)
        with torch.no_grad():
            for X, y in self.data_iter:
                if isinstance(X, list):
                    X = [x.to(self.net.device) for x in X]
                else:
                    X = X.to(self.net.device)
                y = y.to(self.net.device)
                metric.add(self.accuracy(self.net.net(X), y), y.numel())
        return metric[0] / metric[1]

    def aggregatrion(self, modellist):
        length = len(modellist)
        for name, param in self.net.net.named_parameters():
            param.data = sum([model.state_dict()[name] for model in modellist]) / length
        print(f"The accuracy of the aggregated models is {self.evaluate_accuracy()}")
