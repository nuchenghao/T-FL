import json
import sys
import socket
import traceback
import pickle
from lib import libclient
import argparse
# 输出设置----------------------------------------------------------
from rich.console import Console
from rich.padding import Padding
from tool import data, Timer

console = Console()  # 终端输出对象

# 解析相关参数---------------------------------------------------
with open('./config/client.json', 'r', encoding='utf-8') as file:
    config = json.load(file)

host = config['server_ip']
port = config['server_port']

# name由命令行提供
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True, help="name of client")
args = parser.parse_args()
name = args.name  # client名
timer = Timer.Timer()
stateInClient = libclient.stateInClient(name, timer)
# 通信相关-------------------------------------------------------
addr = (host, port)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 每个client创建一个连接server的socket
sock.setblocking(False)
sock.connect_ex(addr)
message = libclient.Message(sock, None, name)


def create_content(name, action, value):  # 这个就是上传的内容格式

    return dict(name=name,
                action=action,
                content=value)


def register():
    # 向服务器注册
    content = create_content(name, "register", "")
    message.content = content
    writeThread = libclient.WriteThread(message)  # 发送注册信息
    writeThread.start()
    writeThread.join()  # 等待读结束
    readThread = libclient.ReadThread(message)  # 等待serve返回信息
    readThread.start()
    readThread.join()
    stateInClient.finished = message.content['finished']


def client():
    console.log(f"{stateInClient.name} start registering", style="bold blue")
    register()  # 向服务器注册
    # console.log(f"{stateInClient.name} has register", style='bold yellow')
    if stateInClient.finished is False:  # 如果没有结束，在register时用
        stateInClient.Net = message.content.get('content').get('net')
        stateInClient.globalepoch = message.content.get('content').get('globalepoch')
        dataIter = data.load_data_fashion_mnist(stateInClient.Net.trainConfigJSON['config']['batchSize'], 'train',
                                                f"./data/noniid/{name}")
        stateInClient.dataIter = dataIter

    while True:
        if stateInClient.finished:
            sock.close()  # 关闭socket
            break
        console.log(f"{stateInClient.name} has been selected in globalepoch {stateInClient.globalepoch}",
                    style='bold yellow')
        stateInClient.timer.start()  # 开始计时
        trainAcc = stateInClient.Net.train(stateInClient.dataIter, stateInClient.name, stateInClient.globalepoch)
        trainTime = stateInClient.timer.stop("s")  # 记录训练时间
        stateInClient.Net.net.eval()

        contentSendToServer = dict(net=stateInClient.Net, trainTime=trainTime, trainAcc=trainAcc)
        contentSendToServer = pickle.dumps(contentSendToServer)  # 上传的模型需要先序列化，因为通过multiprocessing.Queue()直接传递模型会有问题

        content = create_content(name, 'upload', contentSendToServer)
        message.content = content
        writeThread = libclient.WriteThread(message)
        writeThread.start()
        writeThread.join()
        readThread = libclient.ReadThread(message)
        readThread.start()
        readThread.join()
        # 更新全局状态
        stateInClient.finished = message.content['finished']
        if not stateInClient.finished:
            stateInClient.Net = message.content.get('content').get('net')
            stateInClient.globalepoch = message.content.get('content').get('globalepoch')


if __name__ == "__main__":
    client()
