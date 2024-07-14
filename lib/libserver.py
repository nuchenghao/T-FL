import multiprocessing
import sys
import logging
import json
import io
import struct
import pickle
import datetime
from rich.padding import Padding
import time

logger = logging.getLogger("T-FL")


def encode(obj):
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def decode(pickle_bytes, encoding='utf-8'):
    obj = pickle.loads(pickle_bytes, encoding=encoding)
    return obj


def json_encode(obj, encoding):
    return json.dumps(obj, ensure_ascii=False).encode(encoding)


def json_decode(json_bytes, encoding):
    tiow = io.TextIOWrapper(
        io.BytesIO(json_bytes), encoding=encoding, newline=""
    )
    obj = json.load(tiow)
    tiow.close()
    return obj


class Message:
    def __init__(self, sock, messageId, timer):
        self.sock = sock
        self.name = ""  # 这个message对应的client的名称
        self.messageId = messageId  # 每个消息的全局ID
        self.content = None  # 记录上传或下发的内容
        self.timer = timer  # 每个client一个独有的时钟
        self.record = []  # 记录训练数据，记录格式：(总时间，传输耗费时间，训练时间，轮次，该轮次的局部训练精度)

    def __repr__(self):  # 日志输出用
        return f"\n{self.name}'s message id is {self.messageId}"  # 为了适应日志输出，加上换行符

    def summaryOutput(self):
        return f"{self.name}'s message id is {self.messageId}. In globalepoch {self.record[-1][3]},he's time is {self.record[-1][0]},transmission time is {self.record[-1][1]},training time is {self.record[-1][2]} and the local accuracy is {self.record[-1][4]}\n"


class stateInServer:
    def __init__(self, numOfClients, numOfSelectedClients, totalEpoches, allClientMessageQueue,
                 selectedClientMessageIdQueue, multiprocessingSharedQueue, Net, dataIter,
                 timer):
        self.currentClients = 0
        self.numOfClients = numOfClients  # 所有参与训练的客户数
        self.numOfSelectedClients = numOfSelectedClients  # 选中的客户端数量
        self.currentEpoch = 0  # 当前的轮次
        self.totalEpoches = totalEpoches  # 总共需要训练的轮次

        self.allClientMessageQueue = allClientMessageQueue  # 记录所有的client的message
        self.selectedClientMessageIdQueue = selectedClientMessageIdQueue  # 记录每轮被选中的client的message

        self.multiprocessingSharedQueue = multiprocessingSharedQueue  # 多进程分享队列，用于server读取client的上传信息
        self.optionState = None  # 表示当前的状态，register/upload/download
        # 训练相关
        self.Net = Net  # 网络
        self.dataIter = dataIter

        self.timer = timer  # 计时器
        self.resultRecord = [(0, 0.0)]  # 统计一下结果，后面画图用;从(0,0)开始计

    def addEpoch(self):
        self.currentEpoch += 1

    def addClient(self):
        self.currentClients += 1

    def clearClient(self):
        self.currentClients = 0

    def finish(self):
        if self.currentEpoch == self.totalEpoches:
            return True
        else:
            return False


class ReadProcess(multiprocessing.Process):
    def __init__(self, message, printLock, console, multiprocessingSharedQueue):
        super().__init__()
        self.message = message
        self.printLock = printLock
        self.console = console
        self.multiprocessingSharedQueue = multiprocessingSharedQueue
        self._recv_buffer = b""  # 接收缓冲区
        self._jsonheader_len = None
        self.jsonheader = None
        self.request = None

    def _read(self):
        try:
            data = self.message.sock.recv(20_971_520)  # read 20MB once
        except BlockingIOError:
            pass
        else:
            if data:
                self._recv_buffer += data
            else:
                raise RuntimeError("Peer closed.")

    def process_protoheader(self):
        hdrlen = 2
        if len(self._recv_buffer) >= hdrlen:
            self._jsonheader_len = struct.unpack(">H", self._recv_buffer[:hdrlen])[0]
            self._recv_buffer = self._recv_buffer[hdrlen:]

    def process_jsonheader(self):
        hdrlen = self._jsonheader_len
        if len(self._recv_buffer) >= hdrlen:
            self.jsonheader = json_decode(self._recv_buffer[:hdrlen], "utf-8")
            self._recv_buffer = self._recv_buffer[hdrlen:]

    def process_request(self):
        content_len = self.jsonheader["content-length"]
        if not len(self._recv_buffer) >= content_len:
            return
        # 全部数据均已接收
        data = self._recv_buffer[:content_len]
        self._recv_buffer = self._recv_buffer[content_len:]
        self.request = decode(data)

        if self.request.get('action') == 'register':
            self.name = self.request.get('name')
            with self.printLock:
                self.console.log(Padding(f"Received {self.name} register request", style="bold yellow",
                                         pad=(0, 0, 0, 4)))
            self.multiprocessingSharedQueue.put(
                ("finishRegister", self.message.messageId, self.name))  # 返回给server修改,第2个参数表示对应的message

        elif self.request.get('action') == 'upload':
            self.name = self.request.get('name')
            uploadedNet = self.request.get('content')
            with self.printLock:
                self.console.log(
                    Padding(f"Received {self.name} upload request", style='bold yellow', pad=(0, 0, 0, 4)))
                # print(uploadedNet)
            self.multiprocessingSharedQueue.put(("finishUpload", self.message.messageId, uploadedNet))

    def run(self):
        with self.printLock:
            logger.info(
                f"start reading {self.message.name}'s upload whose messageId is {self.message.messageId}" if self.message.name != "" else "A new connection!")
        while True:
            self._read()
            if self._jsonheader_len is None:
                self.process_protoheader()
            if self._jsonheader_len is not None:
                if self.jsonheader is None:
                    self.process_jsonheader()
            if self.jsonheader:
                if self.request is None:
                    self.process_request()
                if self.request is not None:
                    break
        with self.printLock:
            logger.info(
                f"finish reading {self.message.name}'s upload whose messageId is {self.message.messageId}" if self.message.name != "" else "A new connection accepted!")


class WriteProcess(multiprocessing.Process):
    def __init__(self, message, printLock, console, multiprocessingSharedQueue, finished):
        super().__init__()
        self.message = message
        self.printLock = printLock
        self.console = console
        self.multiprocessingSharedQueue = multiprocessingSharedQueue  # 共享队列
        self._send_buffer = b""  # 写缓冲区
        self.finished = finished  # 是否训练结束

    def _create_message(
            self, content
    ):
        jsonheader = {
            "content-length": len(content),
        }
        jsonheader_bytes = json_encode(jsonheader, "utf-8")
        message_hdr = struct.pack(">H", len(jsonheader_bytes))
        message = message_hdr + jsonheader_bytes + content
        return message

    def _create_response(self):
        if self.finished:  # 训练结束
            response = dict(finished=self.finished, content="")
        else:
            response = dict(finished=self.finished, content=self.message.content)
        response = encode(response)
        return response

    def run(self):
        with self.printLock:
            logger.info(f"start sending to {self.message.name} whose messageId is {self.message.messageId}")
        response = self._create_response()
        message = self._create_message(response)  # 加两个头文件
        self._send_buffer += message
        while True:
            if self._send_buffer:
                try:
                    sent = self.message.sock.send(self._send_buffer)
                except BlockingIOError:
                    pass
                else:
                    self._send_buffer = self._send_buffer[sent:]
            else:
                break
        self.multiprocessingSharedQueue.put(("finishDownload", self.message.messageId, ""))
        with self.printLock:
            self.console.log(
                Padding(f"Sent to {self.message.name}", style='bold yellow', pad=(0, 0, 0, 4)))
            logger.info(f"Sent to {self.message.name} whose messageId is {self.message.messageId}")
