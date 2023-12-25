import sys
import selectors
import pickle

import struct


class Message:
    def __init__(self, selector, sock, addr, request):
        self.selector = selector
        self.sock = sock
        self.addr = addr
        self.request = request
        self._recv_buffer = b""
        self._send_buffer = b""
        self._request_queued = False
        self._header_len = None
        self.header = None
        self.response = None

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

    def _read(self):
        try:
            # Should be ready to read
            data = self.sock.recv(1_048_576)
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

    def _encode(self, obj):
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    def _decode(self, pickle_byte, encoding):
        obj = pickle.loads(pickle_byte, encoding=encoding)
        return obj

    def _create_message(
            self, *, content_bytes, content_encoding
    ):
        header = {
            "byteorder": sys.byteorder,
            "content-encoding": content_encoding,
            "content-length": len(content_bytes),
        }
        header_bytes = self._encode(header)
        message_hdr = struct.pack(">H", len(header_bytes))
        message = message_hdr + header_bytes + content_bytes
        return message

    def queue_request(self):
        content = self.request["content"]
        content_encoding = self.request["encoding"]

        req = {
            "content_bytes": self._encode(content),
            "content_encoding": content_encoding,
        }
        message = self._create_message(**req)
        self._send_buffer += message
        self._request_queued = True

    def process_events(self, mask, state):
        if mask & selectors.EVENT_READ:
            self.read(state)
        if mask & selectors.EVENT_WRITE:
            self.write()

    def read(self, state):
        self._read()

        if self._header_len is None:
            self.process_protocalheader()

        if self._header_len is not None:
            if self.header is None:
                self.process_header()

        if self.header:
            if self.response is None:
                self.process_response(state)

    def write(self):
        if not self._request_queued:
            self.queue_request()

        self._write()

        if self._request_queued:
            if not self._send_buffer:
                # Set selector to listen for read events, we're done writing.
                self._set_selector_events_mask("r")

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
            self.sock = None

    def process_protocalheader(self):
        hdrlen = 2
        if len(self._recv_buffer) >= hdrlen:
            self._header_len = struct.unpack(
                ">H", self._recv_buffer[:hdrlen]
            )[0]
            self._recv_buffer = self._recv_buffer[hdrlen:]

    def process_header(self):
        hdrlen = self._header_len
        if len(self._recv_buffer) >= hdrlen:
            self.header = self._decode(
                self._recv_buffer[:hdrlen], "utf-8"
            )
            self._recv_buffer = self._recv_buffer[hdrlen:]
            for reqhdr in (
                    "byteorder",
                    "content-length",
                    "content-encoding",
            ):
                if reqhdr not in self.header:
                    raise ValueError(f"Missing required header '{reqhdr}'.")

    def process_response(self, state):
        content_len = self.header["content-length"]
        if not len(self._recv_buffer) >= content_len:
            return
        data = self._recv_buffer[:content_len]
        self._recv_buffer = self._recv_buffer[content_len:]

        encoding = self.header["content-encoding"]
        self.response = self._decode(data, encoding)  # 得到传输的内容
        if self.response.get('action') == 'register':
            state.numLocalTrain = self.response.get("numLocalTrain")
            state.batchSize = self.response.get("batchSize")
            state.learningRate = self.response.get("learningRate")
            state.splitDataset = self.response.get("splitDataset")
            state.net.getModel(self.response.get('value'))
        elif self.response.get('action') == 'sendData':
            state.data = self.response.get("value")
        elif self.response.get('action') == 'download':
            state.finished = self.response.get('finished')
            if state.finished == False:
                state.net.getModel(self.response.get('value'))  # 获得全局模型
        self.close()
