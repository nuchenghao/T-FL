from collections import OrderedDict

import numpy as np
import torch
from torch import nn


class trainNet:
    def __init__(self, net, device):
        self.net = net  # 训练使用的网络
        self.device = device  # 训练使用的设备

    def init_weights(self, m):  # 初始化参数
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    def initNet(self):
        self.net.apply(self.init_weights)

    def getNetParams(self):
        return [val.cpu().numpy().tolist() for _, val in self.net.state_dict().items()]

    def getModel(self, listOfParameter):
        listOfNdarrat = [np.array(l, dtype=np.float64) for l in listOfParameter]
        params_dict = zip(self.net.state_dict().keys(), listOfNdarrat)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict)
        self.net.to(self.device)
