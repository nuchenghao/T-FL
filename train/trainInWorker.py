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


class trainInWorker():
    def __init__(self, net):
        self.numLocalTrain = 0
        self.batchSize = 0
        self.learningRate = 0
        self.net = net
        self.data_iter = None
        self.optimizer = None
        self.device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'
        self.timer = utils.Timer()

    def initrain(self, numLocalTrain, batchSize, learningRate, paramslist):
        self.numLocalTrain = numLocalTrain
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.data_iter = self.load_data_fashion_mnist(self.batchSize, 'train')
        self.getNewGloablModel(paramslist)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learningRate)

    def load_data_fashion_mnist(self, batch_size, which, resize=None):
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

    def train(self):
        metric = utils.Accumulator(2)
        loss = nn.CrossEntropyLoss()
        for epoch in range(self.numLocalTrain):
            print(f"In local training epoch {epoch + 1}")
            self.net.train()
            for i, (X, y) in enumerate(self.data_iter):
                # self.timer.start()
                self.optimizer.zero_grad()
                X, y = X.to(self.device), y.to(self.device)
                y_hat = self.net(X)
                l = loss(y_hat, y)
                l.backward()
                self.optimizer.step()
                with torch.no_grad():
                    metric.add(self.accuracy(y_hat, y), X.shape[0])

        train_acc = metric[0] / metric[1]
        # self.timer.stop()
        print(f'train acc is {train_acc:.3f}')
        # return self.timer.sum()  # 返回训练时间
