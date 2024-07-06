import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils import data
import torchvision
from torchvision import transforms
from rich.console import Console
from rich.padding import Padding
from collections import OrderedDict

console = Console()  # 终端输出对象


class Net():
    def __init__(self, net, trainConfigJSON, init_weights, loss, optimizer):
        self.net = net
        self.trainConfigJSON = trainConfigJSON
        self.init_weights = init_weights
        self.loss = nn.CrossEntropyLoss(reduction='none') if loss == 'CrossEntropyLoss' else None
        self.optimizer = torch.optim.SGD(self.net.parameters(),
                                         lr=self.trainConfigJSON['lr']) if optimizer == "SGD" else None

    def initNet(self):
        self.net.apply(self.init_weights)

    def getNetParams(self):
        return self.net.state_dict()  # 将模型移到CPU，然后导出模型参数

    def updateNetParams(self, clientModelList):
        self.net.eval()  # 如果网络使用了如BatchNorm这样的层，那么在计算平均参数之前，应该将网络置于评估模式（.eval()），以避免BatchNorm层的运行时统计数据影响参数的值。
        with torch.no_grad():
            # 使用 OrderedDict 保持参数的顺序
            avg_state_dict = OrderedDict()
            for net in clientModelList:
                state_dict = net.state_dict()
                for key, param in state_dict.items():
                    if key in avg_state_dict:
                        avg_state_dict[key] += param
                    else:
                        avg_state_dict[key] = param.clone()
            # 计算平均值
            for key in avg_state_dict.keys():
                avg_state_dict[key] = avg_state_dict[key] / len(clientModelList)
            self.net.load_state_dict(avg_state_dict)
        # print(self.net.state_dict())

    class Accumulator:
        """在n个变量上累加，内部类"""

        def __init__(self, n):
            """Defined in :numref:`sec_softmax_scratch`"""
            self.data = [0.0] * n

        def add(self, *args):
            self.data = [a + float(b) for a, b in zip(self.data, args)]

        def reset(self):
            self.data = [0.0] * len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    reduce_sum = lambda self, x, *args, **kwargs: x.sum(*args, **kwargs)
    argmax = lambda self, x, *args, **kwargs: x.argmax(*args, **kwargs)
    astype = lambda self, x, *args, **kwargs: x.type(*args, **kwargs)
    size = lambda self, x, *args, **kwargs: x.numel(*args, **kwargs)

    def accuracy(self, y_hat, y):
        """计算预测正确的数量

        Defined in :numref:`sec_softmax_scratch`"""
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = self.argmax(y_hat, axis=1)
        cmp = self.astype(y_hat, y.dtype) == y
        return float(self.reduce_sum(self.astype(cmp, y.dtype)))

    def train(self, train_iter, name, iteration, client=True):

        if client:
            # client训练
            numEpochs = self.trainConfigJSON["totalEpochesInClient"]
        else:
            # 服务器预训练
            numEpochs = self.trainConfigJSON["totalEpochesInServer"]

        # 指定设备，client端拿cpu模拟，所以这里直接指定cpu
        device = "cpu"
        self.net.to(device)

        for epoch in range(numEpochs):
            if isinstance(self.net, torch.nn.Module):
                self.net.train()
            metric = self.Accumulator(3)
            for X, y in train_iter:
                self.optimizer.zero_grad()
                X, y = X.to(device), y.to(device)
                y_hat = self.net(X)
                l = self.loss(y_hat, y)
                l.mean().backward()
                self.optimizer.step()
                metric.add(float(l.sum()), self.accuracy(y_hat, y), y.numel())
            train_acc = metric[1] / metric[2]
            console.log(
                Padding(f"{name}'s train accuracy in iteration {iteration} is {train_acc * 100:.4f}%", style='bold red',
                        pad=(0, 0, 0, 20)))
        self.net.to('cpu')  # 防止server上聚合时出错

    def evaluate_accuracy(self, data_iter, stateInServer):
        if isinstance(self.net, torch.nn.Module):
            self.net.eval()  # 将模型设置为评估模式
        metric = self.Accumulator(2)  # 正确预测数、预测总数

        # 指定设备
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.net.to(device)

        with torch.no_grad():
            for X, y in data_iter:
                X, y = X.to(device), y.to(device)
                metric.add(self.accuracy(self.net(X), y), self.size(y))
        test_acc = metric[0] / metric[1]
        stateInServer.resultRecord.append((stateInServer.timer.stop(), test_acc))  # (分钟，精度)
        console.log(Padding(f"the test accuracy is {test_acc}", style='bold red', pad=(0, 0, 0, 4)))
        self.net.to('cpu')  # 防止下发时出错
