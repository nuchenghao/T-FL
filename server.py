import json
import socket
import selectors
import random
import logging
import threading
import time
import queue
from lib import libserver
from tool import Timer
from train import Net
from train import Resnet18
from tool import data
from rich.console import Console
from rich.padding import Padding
import swanlab

import multiprocessing

# 日志模块-------------------------------------------------------------------------------------------------
logger = logging.getLogger("T-FL")
logger.setLevel(logging.INFO)
f_handler = logging.FileHandler(f'./log/{time.strftime("%Y-%m-%d %Hh%Mm%Ss", time.localtime())}.log', 'w')
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

totalEpoches = trainConfigJSON['config']['totalEpoches']  # 总的全局训练轮数
loss = trainConfigJSON['config']['loss']  # 训练的损失函数
optimizer = trainConfigJSON['config']['optimizer']  # 优化器

Net = Net.Net(Resnet18.net, trainConfigJSON, Resnet18.init_weights, loss, optimizer)  # 定义全局网络
Net.initNet()  # 网络初始化

evalDataIter = data.load_data_fashion_mnist(trainConfigJSON['config']["batchSize"], 'test', name='server',
                                            resize=96)  # 测试集迭代器
timer = Timer.Timer()  # 计时器
allClientMessageQueue = []  # 存储所有client对应的message
selectedClientMessageIdQueue = []  # 存储被选中的client的messageId
sharedQueue = queue.Queue()  # 多线程共享队列
stateInServer = libserver.stateInServer(numOfClients, numOfSelectedClients, totalEpoches, allClientMessageQueue,
                                        selectedClientMessageIdQueue,
                                        sharedQueue, Net, evalDataIter, timer)

# -----------------------实验数据同步--------------------------------------------------------
run = swanlab.init(load="./train/train.json")
swanlab.log({'accuracy': 0}, step=0)
# 输出设置----------------------------------------------------------

console = Console()  # 终端输出对象

# 其他。注意锁的申请顺序，先stateInServerLock，后printLock------------------------------------------
stateInServerLock = threading.RLock()  # 多线程的stateInServer锁
printLock = multiprocessing.RLock()  # 多进程的输出锁

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
    stateInServer.allClientMessageQueue.append(message)  # 将这个message放入队列，message的编号从0开始记，所以取的时候直接用编号即可


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
                        event = multiprocessing.Event()
                        multiprocessingSharedQueue = multiprocessing.Queue()
                        readProcess = libserver.ReadProcess(message, printLock, console,
                                                            event, multiprocessingSharedQueue)
                        readThread = libserver.MyThread(stateInServer, stateInServerLock, event,
                                                        multiprocessingSharedQueue)
                        readThread.start()
                        readProcess.start()
            with stateInServerLock:
                if stateInServer.optionState is not None:  # 注册同步完成
                    logger.info(
                        f"all the clients has registered! The stateInServer.allClientMessageQueue is {stateInServer.allClientMessageQueue}")
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
                    while stateInServer.sharedQueue.qsize():  # 一定要清空队列
                        stateInServer.sharedQueue.get()
                    stateInServer.optionState = None
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
                    event = multiprocessing.Event()
                    multiprocessingSharedQueue = multiprocessing.Queue()
                    readProcess = libserver.ReadProcess(message, printLock, console,
                                                        event, multiprocessingSharedQueue)
                    readThread = libserver.MyThread(stateInServer, stateInServerLock, event, multiprocessingSharedQueue)
                    readThread.start()
                    readProcess.start()
                if mask & selectors.EVENT_WRITE:  # client的写事件
                    event = multiprocessing.Event()
                    multiprocessingSharedQueue = multiprocessing.Queue()
                    writeProcess = libserver.WriteProcess(message, printLock, console,
                                                          event, multiprocessingSharedQueue,
                                                          stateInServer.finish())
                    writeThread = libserver.MyThread(stateInServer, stateInServerLock, event,
                                                     multiprocessingSharedQueue)
                    writeThread.start()
                    writeProcess.start()
            # 同步完成，业务处理
            with stateInServerLock:
                if stateInServer.optionState is not None:
                    with printLock:
                        logger.info(
                            "Finish synchronizing all selected clients' upload" if stateInServer.optionState == "finishUpload" else "Finish sending to all selected clients")
                    if stateInServer.optionState == 'finishUpload':  # 聚合，验证，更新每个client的下发模型
                        logger.info("Start aggregating all selected clients' model")
                        clientModelQueue = []
                        while stateInServer.sharedQueue.qsize():  # 顺带清空队列
                            clientModelQueue.append(stateInServer.sharedQueue.get())
                        stateInServer.Net.updateNetParams(clientModelQueue)  # 聚合
                        logger.info("start evaluating aggregrated model")
                        time, test_acc = stateInServer.Net.evaluate_accuracy(stateInServer)  # 验证
                        stateInServer.totalTrainingTime += time
                        swanlab.log({'accuracy': test_acc * 100},
                                    step=stateInServer.totalTrainingTime)
                        stateInServer.addEpoch()  # 至此，一轮结束
                        # # 将该轮参与训练的设备的训练信息进行同步-----------------------------------
                        # for clientID in stateInServer.selectedClientMessageIdQueue:
                        #     msg = stateInServer.allClientMessageQueue[clientID]
                        #     assert msg.messageId == clientID, f"the messageId {msg} is not equal to selected clientId {clientID}"
                        #     swanlab.log({f"{msg.name}_total_time": msg.record[-1][0]}, step=msg.record[-1][3])
                        #     swanlab.log({f"{msg.name}_training_time": msg.record[-1][2]}, step=msg.record[-1][3])
                        # 日志输出--------------------------------------------------------------------------------
                        logger.info(
                            f"Finish evaluating ! The accuracy on test dataset is {test_acc * 100:.4f}% in globalepoch {stateInServer.currentEpoch}")
                        summary = ""
                        for clientID in stateInServer.selectedClientMessageIdQueue:
                            summary += stateInServer.allClientMessageQueue[clientID].summaryOutput()
                        logger.info(
                            f"Summary:\n{summary}\ntime for globalepoch {stateInServer.currentEpoch} is {time} and the total training time is {stateInServer.totalTrainingTime}")
                        # --------------------------------------------------------------------------------------
                        if not stateInServer.finish():
                            console.log(f"Start globalepoch {stateInServer.currentEpoch + 1}",
                                        style="bold white on green")
                            logger.info(
                                f"---------------------- Start globalepoch {stateInServer.currentEpoch + 1} !-----------------------------------------")
                        stateInServer.timer.start()  # 全局时钟开始计时
                        stateInServer.selectedClientMessageIdQueue = selectClientMethod() if not stateInServer.finish() else list(
                            range(0, stateInServer.numOfClients))
                        if not stateInServer.finish():
                            logger.info(
                                f"the selected clients are {[stateInServer.allClientMessageQueue[clientID] for clientID in stateInServer.selectedClientMessageIdQueue]}")
                        else:
                            logger.info(f"Training will finish! Start closing socket to all clients")
                        for clintID in stateInServer.selectedClientMessageIdQueue:
                            msg = stateInServer.allClientMessageQueue[clintID]
                            msg.content = dict(net=stateInServer.Net,
                                               globalepoch=stateInServer.currentEpoch + 1)  # 开始下发信息
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
                        while stateInServer.sharedQueue.qsize():  # 一定要清空队列
                            stateInServer.sharedQueue.get()
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
    swanlab.finish()
