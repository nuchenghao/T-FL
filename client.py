import sys
import socket
import selectors
import traceback
from net import LeNet
from train import trainInWorker
from protocol import libclient
from tools import options

sel = selectors.DefaultSelector()


def create_request(action, value):
    return dict(
        encoding="utf-8",
        content=dict(action=action, value=value),
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


args = options.args_client()  # 解析客户端参数

host, port = args.server_ip, args.server_port

numLocalTrain = 0
batchSzie = 0
learningRate = 0

net = LeNet.lenet()
trainer = trainInWorker.trainInWorker(net)


def registrt():
    # 向服务器注册
    request = create_request("register", "")
    connection(host, port, request)
    try:
        while True:
            events = sel.select(timeout=-1)
            for key, mask in events:
                message = key.data
                try:
                    message.process_events(mask, trainer)
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
    registrt()
    trainer.train()
