import json
import sys
import socket
import selectors
import traceback
import pickle
from lib import libclient

# 输出设置----------------------------------------------------------
from rich.console import Console
from rich.padding import Padding

console = Console()  # 终端输出对象

# 解析相关参数---------------------------------------------------
with open('./config/client.json', 'r', encoding='utf-8') as file:
    config = json.load(file)

host = config['server_ip']
port = config['server_port']
name = config['name']

stateInClient = libclient.stateInClient()
# 通信相关-------------------------------------------------------
sel = selectors.DefaultSelector()


def create_request(name, action, value):
    return dict(name=name, action=action, value=value)


def connection(host, port, variableLenContent2):
    addr = (host, port)
    console.log(Padding("start connecting to server...", style='bold magenta'))
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
    console.rule("[bold red]In register stage")
    register()  # 向服务器注册
    console.log("register has been finished", style='bold red on white')

    console.rule("[bold red]In training stage")
    while True:
        if stateInClient.finished:
            break
        stateInClient.addIteration()
        console.log(f"training iteration {stateInClient.trainingIterations}...", style='bold red on white')
        request = create_request(name, 'upload', "")
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
        console.log(f"training iteration {stateInClient.trainingIterations} has been finished",
                    style='bold red on white')


if __name__ == "__main__":
    client()
