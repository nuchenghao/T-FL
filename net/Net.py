from collections import OrderedDict

import numpy as np
import torch
from torch import nn


class trainNet:
    def __init__(self, net, device, record):
        self.net = net  # 训练使用的网络
        self.device = device  # 训练使用的设备
        self.record = record
        self.countGetNetParams = 1  # 计数器
        self.countGetModel = 1

    def init_weights(self, m):  # 初始化参数
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    def initNet(self):
        self.net.apply(self.init_weights)

    def getNetParams(self):
        params = self.net.cpu().state_dict()  # 统一移到CPU中，然后导出参数
        if self.record:
            torch.save(params, f'./recordModel/{self.countGetNetParams} round params in getNetParams.params')
            self.countGetNetParams += 1
        return params

    def getModel(self, modelStateDict):
        if self.record:
            torch.save(modelStateDict, f'./recordModel/{self.countGetModel} round params in getModel.params')
            self.countGetModel += 1
        self.net.load_state_dict(modelStateDict)

    def printNet(self):
        print(self.net.state_dict())
