import json
import pickle
import sys
import socket
import selectors
import traceback

from lib import libserver
from tool import Timer
from train import Net
from train import Resnet18
from tool import data, drawResult
from rich.console import Console
from rich.padding import Padding

import multiprocessing

# 设置server的state -------------------------------------------------------------------
with open("./config/server.json", 'r', encoding='utf-8') as file:
    configOfServer = json.load(file)  # server的配置文件
with open("./train/train.json", 'r', encoding='utf-8') as file:
    trainConfigJSON = json.load(file)  # 训练的配置文件

ipOfServer = configOfServer['host']  # server的ip地址
portOfServer = configOfServer['port']  # server的监听端口

numOfClients = configOfServer['numOfClients']  # 客户端数量

totalTrainIterations = trainConfigJSON['totalTrainIterations']  # 总的全局训练轮数
loss = trainConfigJSON['loss']  # 训练的损失函数
optimizer = trainConfigJSON['optimizer']  # 优化器

Net = Net.Net(Resnet18.net, trainConfigJSON, Resnet18.init_weights, loss, optimizer)  # 定义全局网络
Net.initNet()  # 网络初始化

evalDataIter = data.load_data_fashion_mnist(trainConfigJSON["batchSize"], 'test', name='server', resize=96)  # 测试集迭代器
timer = Timer.Timer()  # 计时器
allClientMessageQueue = []  # 存储所有client对应的message
multiprocessingSharedQueue = multiprocessing.Queue()  # 所有子进程可以访问的共享队列
stateInServer = libserver.stateInServer(numOfClients, totalTrainIterations, allClientMessageQueue,
                                        multiprocessingSharedQueue, Net, evalDataIter, timer)

# 输出设置----------------------------------------------------------

console = Console()  # 终端输出对象
# 通信部分-------------------------------------------------------------
sel = selectors.DefaultSelector()
lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
lsock.bind((ipOfServer, portOfServer))
lsock.listen()
console.log(f"Listening on {(ipOfServer, portOfServer)}", style="bold white on blue")
lsock.setblocking(False)
sel.register(lsock, selectors.EVENT_READ, data=None)
console.rule("[bold red]In register stage")


def accept_wrapper(sock):
    # 为每个新连接创建socket
    conn, addr = sock.accept()
    conn.setblocking(False)
    message = libserver.Message(conn, stateInServer.currentClients)  # 按照连接的顺序给每个socket一个编号，从0开始
    sel.register(conn, selectors.EVENT_READ, data=message)
    stateInServer.addClient()  # 注册一个新的用户，因为我们需要stateInServer.currentClients作为message的编号
    stateInServer.allClientMessageQueue.append(message)  # 将这个message放入队列


# 其他--------------------------------------------------------------------
printLock = multiprocessing.RLock()

if __name__ == '__main__':
    try:
        while True:
            events = sel.select(timeout=0)  # 非阻塞调用,立即返回可用的文件描述符,而不等待
            for key, mask in events:
                if key.data is None:  # 服务端的listening socket；意味着有一个新的连接到来
                    accept_wrapper(key.fileobj)
                else:
                    # 一个已经注册过的socket
                    message = key.data  # 获得该client对应的message
                    sel.modify(message.sock, 0, data=message)  # 将这个sock在sel中的状态暂时挂起
                    if mask & selectors.EVENT_READ:  # client的读事件
                        readProcess = libserver.ReadProcess(message, printLock, console,
                                                            stateInServer.multiprocessingSharedQueue)
                        readProcess.start()
                    if mask & selectors.EVENT_WRITE:  # client的写事件
                        writeProcess = libserver.WriteProcess(message, printLock, console,
                                                              stateInServer.multiprocessingSharedQueue,
                                                              stateInServer.finish())
                        writeProcess.start()
            # 子进程处理完，同步父进程相关内容
            if stateInServer.multiprocessingSharedQueue.qsize() == stateInServer.numOfClients:  # 所有的子进程都已经结束
                while stateInServer.multiprocessingSharedQueue.qsize():
                    option, msgID, value = stateInServer.multiprocessingSharedQueue.get()
                    # with printLock:
                    #     print(f"{option},{msgID},{value}")
                    if option == "finishRegister":
                        msg = stateInServer.allClientMessageQueue[msgID]
                        assert msg.messageId == msgID, f"the msgID {msgID} is not equal to msg.messageId{msg.messageId}"
                        stateInServer.allClientMessageQueue[msgID].name = value
                        stateInServer.optionState = 'finishRegister'
                    elif option == "finishUpload":
                        msg = stateInServer.allClientMessageQueue[msgID]
                        assert msg.messageId == msgID, f"the msgID {msgID} is not equal to msg.messageId{msg.messageId}"
                        value = pickle.loads(value)  # 这里通过multiprocessing.Queue()来传递，需要先传序列化的内容，父进程接收到后再序列化
                        msg.clientModel = value  # 记录上传的网络
                        stateInServer.optionState = 'finishUpload'
                    elif option == "finishDownload":  # 下发完成
                        msg = stateInServer.allClientMessageQueue[msgID]
                        assert msg.messageId == msgID, f"the msgID {msgID} is not equal to msg.messageId{msg.messageId}"
                        stateInServer.optionState = "finishDownload"
            # 同步完成，业务处理
            if stateInServer.optionState is not None:
                if stateInServer.optionState == 'finishRegister':  #
                    console.rule("[bold red]In training stage")
                    for msg in stateInServer.allClientMessageQueue:
                        msg.clientModel = stateInServer.Net  # 开始下发模型
                        sel.modify(msg.sock, selectors.EVENT_WRITE, data=msg)  # 修改该socket为可写
                elif stateInServer.optionState == 'finishUpload':  # 聚合，验证，更新每个client的下发模型
                    clientModelQueue = []
                    for msg in stateInServer.allClientMessageQueue:
                        clientModelQueue.append(msg.clientModel.net)  # 只需要模型
                    stateInServer.Net.updateNetParams(clientModelQueue)  # 聚合
                    stateInServer.Net.evaluate_accuracy(stateInServer)  # 验证
                    stateInServer.addEpoch()
                    for msg in stateInServer.allClientMessageQueue:
                        msg.clientModel = stateInServer.Net
                        sel.modify(msg.sock, selectors.EVENT_WRITE, data=msg)  # 可以下发模型

                    stateInServer.timer.start()  # 开始计时
                elif stateInServer.optionState == 'finishDownload':  # 全都下发完成
                    if stateInServer.finish():  # 如果结束训练，则关闭socket，退出
                        with printLock:
                            console.rule("[bold red]finish !!!!")
                        for msg in stateInServer.allClientMessageQueue:
                            sel.unregister(msg.sock)  # 注销该socket
                            msg.sock.close()
                        break
                    for msg in stateInServer.allClientMessageQueue:
                        sel.modify(msg.sock, selectors.EVENT_READ, data=msg)  # 等待模型上传
                stateInServer.optionState = None



    except KeyboardInterrupt:
        print("Caught keyboard interrupt, exiting")
    finally:
        sel.close()
