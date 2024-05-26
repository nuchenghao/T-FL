import pickle
import sys
import selectors
import json
import io
import struct

# 输出设置
from rich.console import Console
from rich.padding import Padding

console = Console()  # 终端输出对象


class Message:
    def __init__(self, selector, sock, addr, variableLenContent2):
        self.selector = selector
        self.sock = sock
        self.addr = addr
        self.variableLenContent2 = variableLenContent2  # 必须是二进制文件
        self._recv_buffer = b""
        self._send_buffer = b""
        self._request_queued = False
        self._jsonheader_len = None
        self.jsonheader = None
        self.response = None  # 从服务器接收到内容(经过pickle反序列化)

    # 设置socket状态-----------------------------------------------------------
    def _set_selector_events_mask(self, mode):
        """Set selector to listen for events: mode is 'r', 'w', or 'rw'."""
        if mode == "r":
            events = selectors.EVENT_READ
        elif mode == "w":
            events = selectors.EVENT_WRITE
        elif mode == "rw":
            events = selectors.EVENT_READ | selectors.EVENT_WRITE
        else:
            raise ValueError(f"Invalid events mask mode {mode!r}.")
        self.selector.modify(self.sock, events, data=self)

    # 编解码----------------------------------------------------------------
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

    # 读数据------------------------------------------------------------------
    def _read(self):
        try:
            # Should be ready to read
            data = self.sock.recv(2_097_152)
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
            )[0]
            self._recv_buffer = self._recv_buffer[hdrlen:]

    def process_jsonheader(self):
        hdrlen = self._jsonheader_len
        if len(self._recv_buffer) >= hdrlen:
            self.jsonheader = self._json_decode(
                self._recv_buffer[:hdrlen], "utf-8"
            )
            self._recv_buffer = self._recv_buffer[hdrlen:]
            # 检查header是否正确，这里暂时用不到
            # for reqhdr in (
            #         "byteorder",
            #         "content-length",
            #         "content-type",
            #         "content-encoding",
            # ):
            #     if reqhdr not in self.jsonheader:
            #         raise ValueError(f"Missing required header '{reqhdr}'.")

    def process_response(self, stateInClient):
        content_len = self.jsonheader["content-length"]
        if not len(self._recv_buffer) >= content_len:  # 还没收到完整数据
            return
        # console.log(Padding(f"Received data from server", style="bold green", pad=(0, 0, 0, 4)))
        data = self._recv_buffer[:content_len]
        self._recv_buffer = self._recv_buffer[content_len:]

        self.response = self._decode(data)  # 反序列化

        stateInClient.finished = self.response.get('finished')  # 是否结束训练
        # console.print(Padding("Get Network", style="white", pad=(0, 0, 0, 20)))
        stateInClient.Net = self.response.get("content")
        # print(stateInClient.Net.getNetParams())  # 网络参数的输出只能用print
        self.close()

    def read(self, stateInClient):
        self._read()

        if self._jsonheader_len is None:
            self.process_protoheader()

        if self._jsonheader_len is not None:
            if self.jsonheader is None:
                self.process_jsonheader()

        if self.jsonheader:
            if self.response is None:
                self.process_response(stateInClient)

    # 写数据------------------------------------------------------------------
    def _write(self):
        if self._send_buffer:
            try:
                # Should be ready to write
                sent = self.sock.send(self._send_buffer)
            except BlockingIOError:
                # Resource temporarily unavailable (errno EWOULDBLOCK)
                pass
            else:
                self._send_buffer = self._send_buffer[sent:]

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

    def queue_request(self):
        content = self.variableLenContent2  # 发送的内容
        message = self._create_message(content)  # 添加两个Header
        self._send_buffer += message
        self._request_queued = True

    def write(self, stateInClient):
        if not self._request_queued:
            self.queue_request()

        self._write()

        if self._request_queued:
            if not self._send_buffer:
                # Set selector to listen for read events, we're done writing.
                self._set_selector_events_mask("r")
                # if stateInClient.trainingIterations > 0:
                #     console.log(
                #         f"Model has been send to server. Local training iteration {stateInClient.trainingIterations} has been finished",
                #         style='bold red on white')

    # 关闭socket-------------------------------------------------------
    def close(self):
        # console.log(Padding(f"Closing connection to server", style="bold magenta"))
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

    # 事件处理函数----------------------------------------------------
    def process_events(self, mask, stateInClient):
        if mask & selectors.EVENT_READ:
            self.read(stateInClient)
        if mask & selectors.EVENT_WRITE:
            self.write(stateInClient)


class stateInClient:
    def __init__(self, name):
        self.finished = False
        self.trainingIterations = 0
        self.Net = None

        self.dataIter = None

        self.name = name

    def addIteration(self):
        self.trainingIterations += 1
