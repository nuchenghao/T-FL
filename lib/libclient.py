import pickle
import sys
import json
import io
import struct
import threading

# 输出设置
from rich.console import Console
from rich.padding import Padding

console = Console()  # 终端输出对象


def encode(obj):
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def decode(pickle_bytes, encoding='utf-8'):
    obj = pickle.loads(pickle_bytes, encoding=encoding)
    return obj


def json_encode(obj, encoding):
    return json.dumps(obj, ensure_ascii=False).encode(encoding)


def json_decode(json_bytes, encoding):
    tiow = io.TextIOWrapper(io.BytesIO(json_bytes), encoding=encoding, newline="")
    obj = json.load(tiow)
    tiow.close()
    return obj


class Message:
    def __init__(self, sock, content,name):
        self.sock = sock
        self.content = content  # 发送和接收的内容(非序列化的)
        self.name=name


class ReadThread(threading.Thread):
    def __init__(self, message):
        threading.Thread.__init__(self)
        self.message = message

        self._recv_buffer = b""
        self._jsonheader_len = None
        self.jsonheader = None
        self.finishedRead = False

    def _read(self):
        try:
            data = self.message.sock.recv(20_971_520)  # 20MB
        except BlockingIOError:
            pass
        else:
            if data:
                self._recv_buffer += data
            else:
                raise RuntimeError("Peer closed.")

    def process_jsonheader(self):
        hdrlen = self._jsonheader_len
        if len(self._recv_buffer) >= hdrlen:
            self.jsonheader = json_decode(
                self._recv_buffer[:hdrlen], "utf-8"
            )
            self._recv_buffer = self._recv_buffer[hdrlen:]

    def process_protoheader(self):
        hdrlen = 2
        if len(self._recv_buffer) >= hdrlen:
            self._jsonheader_len = struct.unpack(">H", self._recv_buffer[:hdrlen])[0]
            self._recv_buffer = self._recv_buffer[hdrlen:]

    def process_response(self):
        content_len = self.jsonheader["content-length"]
        if not len(self._recv_buffer) >= content_len:  # 还没收到完整数据
            return
        data = self._recv_buffer[:content_len]
        self._recv_buffer = self._recv_buffer[content_len:]
        self.message.content = decode(data)  # 反序列化
        self.finishedRead = True

    def run(self):
        while True:
            self._read()
            if self._jsonheader_len is None:
                self.process_protoheader()
            if self._jsonheader_len is not None:
                if self.jsonheader is None:
                    self.process_jsonheader()
            if self.jsonheader:
                if not self.finishedRead:
                    self.process_response()
                else:
                    break


class WriteThread(threading.Thread):
    def __init__(self, message):
        threading.Thread.__init__(self)
        self.message = message

        self._send_buffer = b""
        self._request_queued = False

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

    def run(self):
        content = pickle.dumps(self.message.content)  # 先序列化
        message = self._create_message(content)  # 添加两个Header
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
        print(f"{self.message.name} has sent to server")


class stateInClient:
    def __init__(self, name):
        self.trainingIterations = 0
        self.finished = False
        self.Net = None

        self.dataIter = None
        self.name = name

    def addIteration(self):
        self.trainingIterations += 1
