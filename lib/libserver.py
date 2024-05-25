import sys
import selectors
import json
import io
import struct
import pickle

from rich.console import Console
from rich.padding import Padding

console = Console()  # 终端输出对象


class Message:
    def __init__(self, selector, sock, addr):
        self.selector = selector
        self.sock = sock
        self.addr = addr
        self._recv_buffer = b""
        self._send_buffer = b""
        self._jsonheader_len = None
        self.jsonheader = None
        self.request = None
        self.response_created = False
        self.name = ""  # 这个message对应的client的名称

    # 设置socket状态--------------------------------------------------------
    def _set_selector_events_mask(self, mode):
        """Set selector to listen for events: mode is 'r', 'w', or 'rw'."""
        if mode == "r":
            events = selectors.EVENT_READ
        elif mode == "w":
            events = selectors.EVENT_WRITE
        elif mode == "rw":
            events = selectors.EVENT_READ | selectors.EVENT_WRITE
        elif mode == "hold":
            events = 0
        else:
            raise ValueError(f"Invalid events mask mode {mode!r}.")
        self.selector.modify(self.sock, events, data=self)

    def clear(self):
        self._jsonheader_len = None
        self.jsonheader = None
        self.request = None
        self.response_created = False

    # 编码解码-----------------------------------------------------------------------
    def _encode(self, obj):
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    def _decode(self, pickle_bytes, encoding='utf-8'):
        obj = pickle.loads(pickle_bytes, encoding=encoding)
        return obj

    def _json_encode(self, obj, encoding):
        return json.dumps(obj, ensure_ascii=False).encode(encoding)

    def _json_decode(self, json_bytes, encoding):
        tiow = io.TextIOWrapper(
            io.BytesIO(json_bytes), encoding=encoding, newline=""
        )
        obj = json.load(tiow)
        tiow.close()
        return obj

    # 读数据------------------------------------------------------------------------
    def _read(self):
        try:
            # Should be ready to read
            data = self.sock.recv(1_048_576)  # read 1MB once
        except BlockingIOError:
            # Resource temporarily unavailable (errno EWOULDBLOCK)
            pass
        else:
            if data:
                self._recv_buffer += data
            else:
                raise RuntimeError("Peer closed.")

    def process_protoheader(self):
        hdrlen = 2
        if len(self._recv_buffer) >= hdrlen:
            self._jsonheader_len = struct.unpack(
                ">H", self._recv_buffer[:hdrlen]
                # ">H": This is the format string. The > character specifies big-endian byte order,
                # and H indicates that you are unpacking an unsigned short (2 bytes) from the buffer.
            )[0]
            self._recv_buffer = self._recv_buffer[hdrlen:]

    def process_jsonheader(self):
        hdrlen = self._jsonheader_len
        if len(self._recv_buffer) >= hdrlen:
            self.jsonheader = self._json_decode(
                self._recv_buffer[:hdrlen], "utf-8"
            )
            self._recv_buffer = self._recv_buffer[hdrlen:]
            # for reqhdr in (
            #         "byteorder",
            #         "content-length",
            #         "content-type",
            #         "content-encoding",
            # ):
            #     if reqhdr not in self.jsonheader:
            #         raise ValueError(f"Missing required header '{reqhdr}'.")

    def process_request(self, stateInServer):
        content_len = self.jsonheader["content-length"]
        if not len(self._recv_buffer) >= content_len:
            return
        data = self._recv_buffer[:content_len]
        self._recv_buffer = self._recv_buffer[content_len:]
        self.request = self._decode(data)

        if self.request.get('action') == 'register':
            self.name = self.request.get('name')
            console.log(Padding(f"Received {self.name} register request from {self.addr}", style="bold yellow",
                                pad=(0, 0, 0, 4)))
            stateInServer.addClient()
        elif self.request.get('action') == 'upload':
            self.name = self.request.get('name')
            console.log(
                Padding(f"Received {self.name} upload request from {self.addr}", style='bold yellow', pad=(0, 0, 0, 4)))
            stateInServer.addClient()

        # 以下挂起的方式在win中有问题，服务端只能在linux中
        self._set_selector_events_mask("hold")  # 挂起该client，等待其他client

    def read(self, stateInServer):
        self._read()

        if self._jsonheader_len is None:
            self.process_protoheader()

        if self._jsonheader_len is not None:
            if self.jsonheader is None:
                self.process_jsonheader()

        if self.jsonheader:
            if self.request is None:
                self.process_request(stateInServer)

    # 写数据---------------------------------------------------------------------
    def _write(self):
        if self._send_buffer:
            # console.log(Padding(f"Sending to {self.name}", style="bold cyan", pad=(0, 0, 0, 4)))
            try:
                # Should be ready to write
                sent = self.sock.send(self._send_buffer)
            except BlockingIOError:
                # Resource temporarily unavailable (errno EWOULDBLOCK)
                pass
            else:  # 如果try中语句成功执行，那么else语句中的代码也执行。如果出现了异常，那么else中语句就不执行
                # 这里也是保证全部都能写掉
                self._send_buffer = self._send_buffer[sent:]
                # Close when the buffer is drained. The response has been sent.
                if sent and not self._send_buffer:
                    self.close()  # 关闭该轮该client的socket连接

    def _create_message(
            self, content
    ):
        jsonheader = {
            "content-length": len(content),
        }
        jsonheader_bytes = self._json_encode(jsonheader, "utf-8")
        message_hdr = struct.pack(">H", len(jsonheader_bytes))
        message = message_hdr + jsonheader_bytes + content
        return message

    def _create_response(self, stateInServer):
        # 直接传输网络
        # print(stateInServer.Net.getNetParams())  # 网络参数的输出只能用print
        if stateInServer.finish():  # 训练结束
            response = dict(finished=stateInServer.finish(), content="")
        else:
            response = dict(finished=stateInServer.finish(), content=stateInServer.Net)
        response = pickle.dumps(response)
        return response

    def create_response(self, stateInServer):
        response = self._create_response(stateInServer)
        message = self._create_message(response)  # 加两个头文件
        self.response_created = True
        self._send_buffer += message

    def write(self, stateInServer):
        if self.request:
            if not self.response_created:
                self.create_response(stateInServer)

        self._write()

    # 关闭连接---------------------------------------------------------
    def close(self):
        console.log(Padding(f"Closing connection to {self.name}", style="bold cyan", pad=(0, 0, 0, 4)))
        try:
            self.selector.unregister(self.sock)
        except Exception as e:
            print(
                f"Error: selector.unregister() exception for "
                f"{self.addr}: {e!r}"
            )

        try:
            self.sock.close()
        except OSError as e:
            print(f"Error: socket.close() exception for {self.addr}: {e!r}")
        finally:
            # Delete reference to socket object for garbage collection
            self.sock = None

    # handle a client connection ----------------------------------------------------------
    def process_events(self, mask, stateInServer):
        if mask & selectors.EVENT_READ:
            self.read(stateInServer)

        if mask & selectors.EVENT_WRITE:
            self.write(stateInServer)


class stateInServer:
    def __init__(self, numOfClients, totalEpoches, Net, dataIter, timer):
        self.allRegistered = False  # 记录是否已经注册过，用于区分当前是连接还是训练
        self.currentClients = 0
        self.numOfClients = numOfClients  # 所有参与训练的客户数

        self.currentEpoch = 0  # 当前的轮次
        self.totalEpoches = totalEpoches  # 总共需要训练的轮次

        # 训练相关
        self.Net = Net  # 网络
        self.dataIter = dataIter

        self.timer = timer  # 计时器
        self.resultRecord = []  # 统计一下结果，后面画图用

    def addEpoch(self):
        self.currentEpoch += 1

    def addClient(self):
        self.currentClients += 1

    def clearClient(self):
        self.currentClients = 0

    def ready(self):
        if self.currentClients == self.numOfClients:
            return True
        else:
            return False

    def finish(self):
        if self.currentEpoch == self.totalEpoches:
            return True
        else:
            return False
