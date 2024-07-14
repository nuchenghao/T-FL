import json
import pickle
import socket
import selectors
import random
import logging
import time
from lib import libserver
from tool import Timer
from train import Net
from train import Resnet18
from tool import data, drawResult
from rich.console import Console
from rich.padding import Padding

import multiprocessing

# 日志模块-------------------------------------------------------------------------------------------------
logger = logging.getLogger("T-FL")
logger.setLevel(logging.INFO)
f_handler = logging.FileHandler(f'./log/{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}.log', 'w')
f_handler.setLevel(logging.INFO)
f_format = logging.Formatter("""%(asctime)s:
%(message)s""", datefmt='%Y-%m-%d %H:%M:%S')  # 输出格式
f_handler.setFormatter(f_format)
logger.addHandler(f_handler)
# 设置server的state -------------------------------------------------------------------
with open("./config/server.json", 'r', encoding='utf-8') as file:
    configOfServer = json.load(file)  # server的配置文件
with open("./train/train.json", 'r', encoding='utf-8') as file:
    trainConfigJSON = json.load(file)  # 训练的配置文件

ipOfServer = configOfServer['host']  # server的ip地址
portOfServer = configOfServer['port']  # server的监听端口

numOfClients = configOfServer['numOfClients']  # 客户端数量
numOfSelectedClients = configOfServer['numOfSelectedClients']  # 选择的客户端数量

totalEpoches = trainConfigJSON['totalEpoches']  # 总的全局训练轮数
loss = trainConfigJSON['loss']  # 训练的损失函数
optimizer = trainConfigJSON['optimizer']  # 优化器

Net = Net.Net(Resnet18.net, trainConfigJSON, Resnet18.init_weights, loss, optimizer)  # 定义全局网络
Net.initNet()  # 网络初始化

evalDataIter = data.load_data_fashion_mnist(trainConfigJSON["batchSize"], 'test', name='server', resize=96)  # 测试集迭代器
timer = Timer.Timer()  # 计时器
allClientMessageQueue = []  # 存储所有client对应的message
selectedClientMessageIdQueue = []  # 存储被选中的client的messageId
multiprocessingSharedQueue = multiprocessing.Queue()  # 所有子进程可以访问的共享队列
stateInServer = libserver.stateInServer(numOfClients, numOfSelectedClients, totalEpoches, allClientMessageQueue,
                                        selectedClientMessageIdQueue,
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
    timer = Timer.Timer()  # 为每个client的message创建一个计时器
    message = libserver.Message(conn, stateInServer.currentClients, timer)  # 按照连接的顺序给每个socket一个编号，从0开始
    sel.register(conn, selectors.EVENT_READ, data=message)
    stateInServer.addClient()  # 注册一个新的用户，因为我们需要stateInServer.currentClients作为message的编号
    stateInServer.allClientMessageQueue.append(message)  # 将这个message放入队列


# 其他--------------------------------------------------------------------
printLock = multiprocessing.RLock()


# --------------------------------------------------------------------------------------------

def selectClientMethod(method="random") -> list:
    numbers = list(range(0, stateInServer.numOfClients))
    random_sequence = random.sample(numbers, stateInServer.numOfSelectedClients)
    return random_sequence


def registerStage():
    try:
        while True:
            events = sel.select(timeout=0)  # 非阻塞调用,立即返回可用的文件描述符,而不等待
            for key, mask in events:
                if key.data is None:  # 服务端的listening socket；意味着有一个新的连接到来，需要注册
                    accept_wrapper(key.fileobj)
                else:
                    message = key.data  # 获得该client对应的message
                    sel.modify(message.sock, 0, data=message)  # 将这个sock在sel中的状态暂时挂起
                    if mask & selectors.EVENT_READ:  # 注册阶段只有读事件
                        readProcess = libserver.ReadProcess(message, printLock, console,
                                                            stateInServer.multiprocessingSharedQueue)
                        readProcess.start()
            if stateInServer.multiprocessingSharedQueue.qsize() == stateInServer.numOfClients:  # 所有的用户都已经注册了
                while stateInServer.multiprocessingSharedQueue.qsize():
                    option, msgID, value = stateInServer.multiprocessingSharedQueue.get()
                    msg = stateInServer.allClientMessageQueue[msgID]
                    assert msg.messageId == msgID, f"the msgID {msgID} is not equal to msg.messageId{msg.messageId}"
                    stateInServer.allClientMessageQueue[msgID].name = value
                    stateInServer.optionState = 'finishRegister'
                logger.info(
                    f"all the clients has registered! The stateInServer.allClientMessageQueue is {stateInServer.allClientMessageQueue}")
            if stateInServer.optionState is not None:
                console.rule("[bold red]In training stage")
                if not stateInServer.finish():
                    console.log(f"Start globalepoch {stateInServer.currentEpoch + 1}", style="bold white on green")
                    stateInServer.timer.start()  # 开始计时
                    logger.info(
                        f"---------------------- Start globalepoch {stateInServer.currentEpoch + 1} !-----------------------------------------")
                stateInServer.selectedClientMessageIdQueue = selectClientMethod() if not stateInServer.finish() else list(
                    range(0, stateInServer.numOfClients))
                if not stateInServer.finish():
                    logger.info(
                        f"the selected clients are {[stateInServer.allClientMessageQueue[clientID] for clientID in stateInServer.selectedClientMessageIdQueue]}")
                else:
                    logger.info(f"Training will finish! Start closing socket to all clients")
                for clintID in stateInServer.selectedClientMessageIdQueue:
                    msg = stateInServer.allClientMessageQueue[clintID]
                    assert msg.messageId == clintID, f"the messageId {msg} is not equal to selected clientId {clintID}"
                    msg.content = dict(net=stateInServer.Net, globalepoch=stateInServer.currentEpoch + 1)  # 开始下发信息
                    sel.modify(msg.sock, selectors.EVENT_WRITE, data=msg)  # 修改该socket为可写
                    if not stateInServer.finish():
                        msg.timer.start()  # 开始计时
                break
    except KeyboardInterrupt:
        print("Caught keyboard interrupt, exiting")


def trainingStage():
    try:
        while True:
            events = sel.select(timeout=0)  # 非阻塞调用,立即返回可用的文件描述符,而不等待
            for key, mask in events:
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
            if stateInServer.multiprocessingSharedQueue.qsize() == len(
                    stateInServer.selectedClientMessageIdQueue):  # 所有的子进程都已经结束
                while stateInServer.multiprocessingSharedQueue.qsize():
                    option, msgID, value = stateInServer.multiprocessingSharedQueue.get()
                    if option == "finishUpload":
                        msg = stateInServer.allClientMessageQueue[msgID]
                        assert msg.messageId == msgID and msg.messageId in stateInServer.selectedClientMessageIdQueue, f"the msgID {msgID} is not equal to msg.messageId{msg.messageId}"
                        value = pickle.loads(value)  # 这里通过multiprocessing.Queue()来传递，需要先传序列化的内容，父进程接收到后再序列化
                        msg.content = value  # 记录上传的网络
                        stateInServer.optionState = 'finishUpload'
                        totalTime = msg.timer.stop("s")  # 这个时间记录的有些问题，因为要等待所有的client上传后才能统计，所以有很大延迟
                        # 记录的格式为：(总时间，传输耗费时间，训练时间，轮次，该轮次的局部训练精度)
                        msg.record.append((totalTime, totalTime - msg.content['trainTime'], msg.content['trainTime'],
                                           stateInServer.currentEpoch + 1, msg.content['trainAcc']))
                    elif option == "finishDownload":  # 下发完成
                        msg = stateInServer.allClientMessageQueue[msgID]
                        assert msg.messageId == msgID, f"the msgID {msgID} is not equal to msg.messageId{msg.messageId}"
                        stateInServer.optionState = "finishDownload"
                with printLock:
                    logger.info(
                        "Finish synchronizing all selected clients' upload" if stateInServer.optionState == "finishUpload" else "Finish sending to all selected clients")
            # 同步完成，业务处理
            if stateInServer.optionState is not None:
                if stateInServer.optionState == 'finishUpload':  # 聚合，验证，更新每个client的下发模型
                    logger.info("Start aggregating all selected clients' model")
                    clientModelQueue = []
                    for clientId in stateInServer.selectedClientMessageIdQueue:
                        clientModelQueue.append(
                            stateInServer.allClientMessageQueue[clientId].content['net'].net)  # 更新被选中的client的Net
                    stateInServer.Net.updateNetParams(clientModelQueue)  # 聚合
                    logger.info("start evaluating aggregrated model")
                    test_acc = stateInServer.Net.evaluate_accuracy(stateInServer)  # 验证
                    stateInServer.addEpoch()  # 至此，一轮结束
                    logger.info(
                        f"Finish evaluating ! The accuracy on test dataset is {test_acc * 100:.4f}% in globalepoch {stateInServer.currentEpoch}")
                    summary = ""
                    for clientID in stateInServer.selectedClientMessageIdQueue:
                        summary += stateInServer.allClientMessageQueue[clientID].summaryOutput()
                    logger.info(f"Summary:\n{summary}")

                    if not stateInServer.finish():
                        console.log(f"Start globalepoch {stateInServer.currentEpoch + 1}", style="bold white on green")
                        logger.info(
                            f"---------------------- Start globalepoch {stateInServer.currentEpoch + 1} !-----------------------------------------")
                    stateInServer.selectedClientMessageIdQueue = selectClientMethod() if not stateInServer.finish() else list(
                        range(0, stateInServer.numOfClients))
                    if not stateInServer.finish():
                        logger.info(
                            f"the selected clients are {[stateInServer.allClientMessageQueue[clientID] for clientID in stateInServer.selectedClientMessageIdQueue]}")
                    else:
                        logger.info(f"Training will finish! Start closing socket to all clients")
                    for clintID in stateInServer.selectedClientMessageIdQueue:
                        msg = stateInServer.allClientMessageQueue[clintID]
                        msg.content = dict(net=stateInServer.Net, globalepoch=stateInServer.currentEpoch + 1)  # 开始下发信息
                        sel.modify(msg.sock, selectors.EVENT_WRITE, data=msg)  # 修改该socket为可写
                        if not stateInServer.finish():
                            msg.timer.start()  # 开始计时

                elif stateInServer.optionState == 'finishDownload':  # 全都下发完成
                    if stateInServer.finish():  # 如果结束训练，则关闭socket，退出
                        with printLock:
                            console.rule("[bold red]finish !!!!")
                        for msg in stateInServer.allClientMessageQueue:
                            sel.unregister(msg.sock)  # 注销该socket
                            msg.sock.close()
                        logger.info("---------------------------finish---------------------------------")
                        break
                    for clientID in stateInServer.selectedClientMessageIdQueue:
                        msg = stateInServer.allClientMessageQueue[clientID]
                        sel.modify(msg.sock, selectors.EVENT_READ, data=msg)  # 等待模型上传
                stateInServer.optionState = None

    except KeyboardInterrupt:
        print("Caught keyboard interrupt, exiting")
    finally:
        sel.close()


def main():
    logger.info(
        f"""
Start training
There are {stateInServer.numOfClients} clients. Each time {stateInServer.numOfSelectedClients} selected to participate in the training.
The training config is :
{json.dumps(trainConfigJSON, indent=4)}
""")  # 训练开始前做一次log
    registerStage()
    trainingStage()


if __name__ == '__main__':
    main()
