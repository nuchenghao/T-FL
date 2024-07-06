import json
import sys
import socket
import selectors
import traceback
import pickle
from lib import libclient
import argparse
# 输出设置----------------------------------------------------------
from rich.console import Console
from rich.padding import Padding
from tool import data

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

stateInClient = libclient.stateInClient(name)
# 通信相关-------------------------------------------------------
sel = selectors.DefaultSelector()


def create_request(name, action, value):
    return dict(name=name, action=action, content=value)


def connection(host, port, variableLenContent2):
    addr = (host, port)
    # console.log(Padding("start connecting to server...", style='bold magenta'))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(False)
    sock.connect_ex(addr)
    events = selectors.EVENT_READ | selectors.EVENT_WRITE
    message = libclient.Message(sel, sock, addr, variableLenContent2)
    sel.register(sock, events, data=message)


def register():
    # 向服务器注册
    request = create_request(name, "register", "")
    variableLenContent2 = pickle.dumps(request)  # 一定要是二进制文件
    connection(host, port, variableLenContent2)
    try:
        while True:
            events = sel.select(timeout=-1)
            for key, mask in events:
                message = key.data
                try:
                    message.process_events(mask, stateInClient)
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


def client():
    # console.rule("[bold red]In register stage")
    console.log(f"{stateInClient.name} start registering", style="bold blue")
    register()  # 向服务器注册
    console.log(f"{stateInClient.name} have registered", style='bold yellow')

    dataIter = data.load_data_fashion_mnist(stateInClient.Net.trainConfigJSON['batchSize'], 'train',
                                            "./data/noniid/{}".format(name))
    stateInClient.dataIter = dataIter
    # console.rule("[bold red]In training stage")
    while True:
        if stateInClient.finished:
            break
        stateInClient.addIteration()
        # console.log(f"training iteration {stateInClient.trainingIterations}...", style='bold red on white')
        stateInClient.Net.train(stateInClient.dataIter, stateInClient.name, stateInClient.trainingIterations)
        stateInClient.Net.net.eval()
        request = create_request(name, 'upload', stateInClient.Net)
        variableLenContent2 = pickle.dumps(request)
        connection(host, port, variableLenContent2)
        try:
            while True:
                events = sel.select(timeout=-1)
                for key, mask in events:
                    message = key.data
                    try:
                        message.process_events(mask, stateInClient)
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
