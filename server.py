import copy
import sys
import socket
import selectors
import traceback

import numpy as np
import torch
import torchvision

from net import LeNet, Net
from protocol import libserver
from train import trainInServer
from tools import options
from tools import stateInServer

sel = selectors.DefaultSelector()


def accept_wrapper(sock, state):
    # 为每个新连接创建socket
    conn, addr = sock.accept()  # Should be ready to read
    # print(f"Accepted connection from {addr}")
    conn.setblocking(False)
    message = libserver.Message(sel, conn, addr, state.net)
    sel.register(conn, selectors.EVENT_READ, data=message)


args = options.args_server()
# 设定训练参数
numLocalTrain = args.numLocalTrain
batchSize = args.batchSize
learningRate = args.learningRate
numGlobalTrain = args.numGlobalTrain
host, port = args.host, args.port
numCliets = args.numClient
splitDataSet = args.splitDataSet
record = args.record

device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')  # server上的训练设备
globalNet = Net.trainNet(LeNet.lenet(), device, record)  # 全局网络

trainer = trainInServer.Trainer(batchSize, globalNet, 'test')
state = stateInServer.messageInServer(globalNet, numGlobalTrain, numCliets, numLocalTrain, batchSize, learningRate,
                                      splitDataSet)

# 创建socket监听设备
lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
lsock.bind((host, port))
lsock.listen()
print(f"Listening on {(host, port)}")
lsock.setblocking(False)
sel.register(lsock, selectors.EVENT_READ, data=None)


def splitDataSet(clientlist):
    train_data = torchvision.datasets.FashionMNIST(root='./fashionmnist', train=True, download=True)
    numClients = len(clientlist)  # client数量
    numSamples = len(train_data)
    part_size = numSamples // numClients  # 每个client分到多少数据
    indices = list(range(numSamples))
    np.random.shuffle(indices)
    indicesClient = []
    for i in range(numClients):
        indicesClient.append(indices[i * part_size:(i + 1) * part_size])
    for i in range(numClients):
        clientlist[i].data = [train_data[j] for j in indicesClient[i]]


try:
    while True:
        events = sel.select(timeout=None)
        for key, mask in events:
            if key.data is None:  # 连接
                accept_wrapper(key.fileobj, state)
            else:
                message = key.data
                try:
                    message.process_events(mask, state)
                except Exception:
                    print(
                        f"Main: Error: Exception for {message.addr}:\n"
                        f"{traceback.format_exc()}"
                    )
                    message.close()
        if state.ready():  # 所有client就绪
            if state.register and state.splitDataset:  # 注册过且需要划分数据
                clientlist = []  # 用户列表，元素是Message
                socket_map = sel.get_map()
                for fd, key in socket_map.items():
                    if key.data != None:
                        clientlist.append(key.data)
                splitDataSet(clientlist)

                for fd, key in socket_map.items():
                    if key.data != None:
                        key.data.net = copy.deepcopy(state.net)
                        sel.modify(key.fileobj, selectors.EVENT_WRITE, key.data)

                state.splitDataset = False

            elif state.register:  # 注册过了，训练
                modellist = []
                socket_map = sel.get_map()  # 获取注册的socket和data的字典
                for fd, key in socket_map.items():
                    if key.data != None:
                        modellist.append(key.data.net.net)  # 获取网络
                trainer.aggregatrion(modellist)  # 聚合
                for fd, key in socket_map.items():
                    if key.data != None:
                        key.data.net = copy.deepcopy(state.net)
                        sel.modify(key.fileobj, selectors.EVENT_WRITE, key.data)
                state.addEpoch()

            else:  # 注册
                socket_map = sel.get_map()  # 获取注册的socket和data的字典
                for fd, key in socket_map.items():
                    data = key.data
                    if data != None:
                        sel.modify(key.fileobj, selectors.EVENT_WRITE, data)  # 将挂起的事件激活
                state.register = True

            state.clearClient()

        if state.finish() and len(sel.get_map()) == 1:  # 训练完成且所有通信socket都已经完成
            break

except KeyboardInterrupt:
    print("Caught keyboard interrupt, exiting")
finally:
    sel.close()
