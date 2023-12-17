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
    def __init__(self, numLocalTrain, batchSize, learningRate, numGlobalTrain, net, device, which):
        self.numLocalTrain = numLocalTrain
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.numGlobalTrain = numGlobalTrain
        self.net = net
        self.data_iter = self.load_data_fashion_mnist(batchSize, which)
        self.device = device
        self.established = False
        self.initNet()  # 初始化网络

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

    def init_weights(self, m):  # 初始化参数
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    def initNet(self):
        self.net.apply(self.init_weights)

    def accuracy(self, y_hat, y):
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = argmax(y_hat, axis=1)
        cmp = astype(y_hat, y.dtype) == y
        return float(reduce_sum(astype(cmp, y.dtype)))

    def getNetParams(self):
        return [val.cpu().numpy().tolist() for _, val in self.net.state_dict().items()]

    def getNewGloablModel(self, listOfParameter):
        listOfNdarrat = [np.array(l, dtype=np.float64) for l in listOfParameter]
        params_dict = zip(self.net.state_dict().keys(), listOfNdarrat)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict)
        self.net.to(self.device)

    def evaluate_accuracy(self):
        if isinstance(self.net, nn.Module):
            self.net.eval()  # 设置为评估模式

        metric = utils.Accumulator(2)
        with torch.no_grad():
            for X, y in self.data_iter:
                if isinstance(X, list):
                    X = [x.to(self.device) for x in X]
                else:
                    X = X.to(self.device)
                y = y.to(self.device)
                metric.add(self.accuracy(self.net(X), y), y.numel())
        return metric[0] / metric[1]

# net = LeNet.lenet()
# initNet(net)
