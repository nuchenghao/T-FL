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
record = args.record
name = args.name
numLocalTrain = 0
batchSzie = 0
learningRate = 0


def get_available_gpu():
    # 检查是否有可用的 GPU
    if torch.cuda.is_available():
        # 获取可用的 GPU 数量
        gpu_count = torch.cuda.device_count()
        # 遍历所有可用的 GPU，选择第一个未被使用的 GPU
        for i in range(gpu_count):
            if torch.cuda.get_device_properties(i).is_initialized():
                continue  # 跳过已经被使用的 GPU
            else:
                selected_gpu = torch.device(f"cuda:{i}")
                return selected_gpu
        return torch.device(f"cuda:{gpu_count - 1}")  # 如果所有 GPU 都被使用，则选择最大序号的 GPU
    else:
        return torch.device("cpu")  # 如果没有可用的 GPU，则使用 CPU 进行训练


device = get_available_gpu()

net = Net.trainNet(LeNet.lenet(), device, record)
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
