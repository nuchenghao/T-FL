import sys
import socket
import selectors
import traceback

import torch

from net import LeNet, Net
from train import trainInClient
from protocol import libclient
from tools import options
from tools import stateInClient

sel = selectors.DefaultSelector()

args = options.args_client()  # 解析客户端参数

host, port = args.server_ip, args.server_port
name = args.name
numLocalTrain = 0
batchSzie = 0
learningRate = 0
device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
net = Net.trainNet(LeNet.lenet(), device)
trainer = trainInClient.trainInWorker(net)
state = stateInClient.messageInClient(net)


def create_request(name, action, value):
    return dict(
        encoding="utf-8",
        content=dict(name=name, action=action, value=value),
    )


def connection(host, port, request):
    addr = (host, port)
    print(f"Starting connection to {addr}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(False)
    sock.connect_ex(addr)
    events = selectors.EVENT_READ | selectors.EVENT_WRITE
    message = libclient.Message(sel, sock, addr, request)
    sel.register(sock, events, data=message)


def registrt():
    # 向服务器注册
    request = create_request(name, "register", "")
    connection(host, port, request)
    try:
        while True:
            events = sel.select(timeout=-1)
            for key, mask in events:
                message = key.data
                try:
                    message.process_events(mask, state)
                except Exception:
                    print(
                        f"Main: Error: Exception for {message.addr}:\n"
                        f"{traceback.format_exc()}"
                    )
                    message.close()
            if not sel.get_map():
                break
    except Exception:
        print("Caught Exception in register, exiting")


def requestData():
    request = create_request(name, "requestData", "")
    connection(host, port, request)
    try:
        while True:
            events = sel.select(timeout=-1)
            for key, mask in events:
                message = key.data
                try:
                    message.process_events(mask, state)
                except Exception:
                    print(
                        f"Main: Error: Exception for {message.addr}:\n"
                        f"{traceback.format_exc()}"
                    )
                    message.close()
            if not sel.get_map():
                break
    except Exception:
        print("Caught Exception in requesting data, exiting")


def client():
    registrt()
    if state.splitDataset:  # 请求数据
        requestData()
    trainer.initrain(state)
    # print(trainer.numLocalTrain, trainer.batchSize, trainer.learningRate)
    while True:
        if state.finished:
            break
        trainer.train()  # 训练

        # 通信
        request = create_request(name, 'upload', trainer.net.getNetParams())
        connection(host, port, request)
        try:
            while True:
                events = sel.select(timeout=-1)
                for key, mask in events:
                    message = key.data
                    try:
                        message.process_events(mask, state)
                    except Exception:
                        print(
                            f"Main: Error: Exception for {message.addr}:\n"
                            f"{traceback.format_exc()}"
                        )
                        message.close()
                if not sel.get_map():
                    break
        except Exception:
            print("Caught Exception in register, exiting")


if __name__ == "__main__":
    client()
