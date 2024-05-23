import json
import pickle
import sys
import socket
import selectors
import traceback

from lib import libserver

from train import Net
from train import MLP

# 设置server的state -------------------------------------------------------------------
with open("./config/server.json", 'r', encoding='utf-8') as file:
    configOfServer = json.load(file)
with open("./train/train.json", 'r', encoding='utf-8') as file:
    trainConfigJSON = json.load(file)
host = configOfServer['host']
port = configOfServer['port']
numOfClients = configOfServer['numOfClients']
totalEpochesInServer = trainConfigJSON['totalEpochesInServer']

with open("./train/train.json", 'r', encoding="utf-8") as file:
    trainConfig2 = pickle.dumps(json.load(file))

Net = Net.Net(MLP.net, trainConfigJSON, MLP.init_weights)

stateInServer = libserver.stateInServer(numOfClients, totalEpochesInServer, Net)

# 输出设置----------------------------------------------------------
from rich.console import Console
from rich.padding import Padding

console = Console()  # 终端输出对象
# 通信部分-------------------------------------------------------------
sel = selectors.DefaultSelector()


def accept_wrapper(sock):
    # 为每个新连接创建socket
    conn, addr = sock.accept()  # Should be ready to read
    conn.setblocking(False)
    message = libserver.Message(sel, conn, addr)
    sel.register(conn, selectors.EVENT_READ, data=message)
    # console.log(f"Accepted connection from {addr}", style="bold green")


lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
lsock.bind((host, port))
lsock.listen()
console.log(f"Listening on {(host, port)}", style="bold white on blue")
lsock.setblocking(False)
sel.register(lsock, selectors.EVENT_READ, data=None)
console.rule("[bold red]In register stage")

registered = False  # 处理输出用
aggregated = True
try:
    while True:
        events = sel.select(timeout=None)
        for key, mask in events:
            if key.data is None:  # 服务端的listening socket；意味着有一个新的连接到来
                accept_wrapper(key.fileobj)
            else:
                # or a client socket that’s already been accepted
                message = key.data
                try:
                    message.process_events(mask, stateInServer)
                except Exception:
                    print(
                        f"Main: Error: Exception for {message.addr}:\n"
                        f"{traceback.format_exc()}"
                    )
                    message.close()
        if stateInServer.ready():
            if stateInServer.allRegistered:
                socket_map = sel.get_map()  # 获取注册的socket和data的字典
                for fd, key in socket_map.items():
                    if key.data != None:
                        sel.modify(key.fileobj, selectors.EVENT_WRITE, key.data)

                # 聚合，然后分发模型
                stateInServer.addEpoch()
                console.log(
                    Padding(f"Training iteration {stateInServer.currentEpoch} has been finished",
                            style="bold red on white"))
                if stateInServer.finish():
                    console.log(f"Training has been finished. Send finished flag to clients", style="bold red on white")
                aggregated = True
            else:  # 注册
                socket_map = sel.get_map()  # 获取注册的socket和data的字典
                for fd, key in socket_map.items():
                    data = key.data
                    if data != None:
                        sel.modify(key.fileobj, selectors.EVENT_WRITE, data)  # 将挂起的事件激活
                stateInServer.allRegistered = True
                console.log("Register has been finished", style="bold red on white")
                console.rule("[bold red]In training stage")

                # 下发参数

            stateInServer.clearClient()

        if len(sel.get_map()) - 1 == numOfClients and stateInServer.allRegistered and aggregated and not stateInServer.finish():
            console.log(Padding(f"Training iteration {stateInServer.currentEpoch + 1}...", style="bold red on white"))
            aggregated = False

        if stateInServer.finish() and len(sel.get_map()) == 1:  # 训练完成且所有通信socket都已经完成
            console.log("Finish training", style="bold red on white")
            break




except KeyboardInterrupt:
    print("Caught keyboard interrupt, exiting")
finally:
    sel.close()

# 每个链接传输一次，然后就断开
