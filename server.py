import sys
import socket
import selectors
import traceback

import torch

from net import LeNet
from protocol import libserver
from train import trainInServer
from tools import options
from tools import msgInServer

sel = selectors.DefaultSelector()


def accept_wrapper(sock):
    conn, addr = sock.accept()  # Should be ready to read
    print(f"Accepted connection from {addr}")
    conn.setblocking(False)
    message = libserver.Message(sel, conn, addr)
    sel.register(conn, selectors.EVENT_READ, data=message)


args = options.args_server()
# 设定训练参数
numLocalTrain = args.numLocalTrain
batchSize = args.batchSize
learningRate = args.learningRate
numGlobalTrain = args.numGlobalTrain
net = LeNet.lenet()
device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')  # server上的训练设备
trainer = trainInServer.Trainer(numLocalTrain, batchSize, learningRate, numGlobalTrain, net, device, 'test')
msg = msgInServer.messageInServer(numGlobalTrain)

host, port = args.host, args.port

# 创建socket监听设备
lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
lsock.bind((host, port))
lsock.listen()
print(f"Listening on {(host, port)}")
lsock.setblocking(False)
sel.register(lsock, selectors.EVENT_READ, data=None)

# 已经训练的次数
epoch = -1
try:
    while True:
        events = sel.select(timeout=None)
        for key, mask in events:
            if key.data is None:  # 连接
                accept_wrapper(key.fileobj)
            else:
                message = key.data
                try:
                    message.process_events(mask, trainer, msg)
                except Exception:
                    print(
                        f"Main: Error: Exception for {message.addr}:\n"
                        f"{traceback.format_exc()}"
                    )
                    message.close()
        if msg.finish() == True and len(sel.get_map()) == 1:
            break

except KeyboardInterrupt:
    print("Caught keyboard interrupt, exiting")
finally:
    sel.close()
