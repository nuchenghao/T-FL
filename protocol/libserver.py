import copy
import sys
import selectors
import json
import io
import struct


class Message:
    def __init__(self, selector, sock, addr, net):
        self.selector = selector
        self.sock = sock
        self.addr = addr
        self._recv_buffer = b""
        self._send_buffer = b""
        self._jsonheader_len = None
        self.jsonheader = None
        self.request = None
        self.response_created = False
        self.accuracy = 0.0
        self.net = copy.deepcopy(net)  # 为每个连接(client)创建一个net，用于传递全局模型和接收client模型
        self.data = None

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

    def _write(self):
        if self._send_buffer:
            # print(f"Sending {self._send_buffer!r} to {self.addr}")
            try:
                # Should be ready to write
                sent = self.sock.send(self._send_buffer)
            except BlockingIOError:
                # Resource temporarily unavailable (errno EWOULDBLOCK)
                pass
            else:
                self._send_buffer = self._send_buffer[sent:]
                # Close when the buffer is drained. The response has been sent.
                if sent and not self._send_buffer:
                    self.close()

    def _json_encode(self, obj, encoding):
        return json.dumps(obj, ensure_ascii=False).encode(encoding)

    def _json_decode(self, json_bytes, encoding):
        tiow = io.TextIOWrapper(
            io.BytesIO(json_bytes), encoding=encoding, newline=""
        )
        obj = json.load(tiow)
        tiow.close()
        return obj

    def process_events(self, mask, state):
        if mask & selectors.EVENT_READ:
            self.read(state)

        if mask & selectors.EVENT_WRITE:
            self.write(state)

    def read(self, msg):
        self._read()

        if self._jsonheader_len is None:
            self.process_protoheader()

        if self._jsonheader_len is not None:
            if self.jsonheader is None:
                self.process_jsonheader()

        if self.jsonheader:
            if self.request is None:
                self.process_request(msg)

    def write(self, msg):
        if self.request:
            if not self.response_created:
                self.create_response(msg)

        self._write()

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
            for reqhdr in (
                    "byteorder",
                    "content-length",
                    "content-encoding",
            ):
                if reqhdr not in self.jsonheader:
                    raise ValueError(f"Missing required header '{reqhdr}'.")

    def process_request(self, state):
        content_len = self.jsonheader["content-length"]
        if not len(self._recv_buffer) >= content_len:
            return
        data = self._recv_buffer[:content_len]
        self._recv_buffer = self._recv_buffer[content_len:]

        encoding = self.jsonheader["content-encoding"]
        self.request = self._json_decode(data, encoding)

        if self.request.get('action') == 'register':  # 注册的处理
            print(f"Received {self.request.get('name')} register request from {self.addr}")
            state.addClient()
        elif self.request.get('action') == 'requestData':
            print(f"Received {self.request.get('name')} request for data from {self.addr}")
            state.addClient()
        elif self.request.get('action') == 'upload':  # 训练过程中的处理
            print(f"Received {self.request.get('name')} upload request from {self.addr}")
            self.net.getModel(self.request.get('value'))  # 得到client的模型
            state.addClient()

        # 以下挂起的方式在win中有问题，服务端只能在linux中
        self._set_selector_events_mask("hold")  # 挂起该client，等待其他client

    def _create_message(
            self, *, content_bytes, content_encoding
    ):
        jsonheader = {
            "byteorder": sys.byteorder,
            "content-encoding": content_encoding,
            "content-length": len(content_bytes),
        }
        jsonheader_bytes = self._json_encode(jsonheader, "utf-8")
        message_hdr = struct.pack(">H", len(jsonheader_bytes))
        message = message_hdr + jsonheader_bytes + content_bytes
        return message

    def _create_response_json_content(self, state):
        if self.request.get('action') == "register":
            content = {
                # register阶段返回的内容
                'action': 'register',
                'numLocalTrain': state.numLocalTrain,
                'batchSize': state.batchSize,
                'learningRate': state.learningRate,
                'splitDataSet': state.splitDataset,
                'value': state.net.getNetParams(),
            }
            content_encoding = "utf-8"
            response = {
                "content_bytes": self._json_encode(content, content_encoding),
                "content_encoding": content_encoding,
            }

        elif self.request.get('action') == "requestData":
            content = {
                'action': 'sendData',
                'value': self.data
            }
            content_encoding = "utf-8"
            response = {
                "content_bytes": self._json_encode(content, content_encoding),
                "content_encoding": content_encoding,
            }
        elif self.request.get('action') == "upload":
            content = {
                # upload阶段返回的内容
                "action": "download",
                "value": state.net.getNetParams(),
                "finished": state.finish()
            }
            content_encoding = "utf-8"
            response = {
                "content_bytes": self._json_encode(content, content_encoding),
                "content_encoding": content_encoding,
            }

        return response

    def create_response(self, state):
        response = self._create_response_json_content(state)
        message = self._create_message(**response)
        self.response_created = True
        self._send_buffer += message

    def close(self):
        print(f"Closing connection to {self.addr}")
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
