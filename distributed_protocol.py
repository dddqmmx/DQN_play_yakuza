import pickle
import socket
import struct
import time

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


HEADER_STRUCT = struct.Struct("!I")
MAX_MESSAGE_SIZE = 256 * 1024 * 1024
NDARRAY_MARKER = "__dqn_numpy_array__"


class ProtocolError(RuntimeError):
    pass


def configure_socket(sock):
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    return sock


def recvall(sock, size):
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("socket closed")
        chunks.extend(chunk)
    return bytes(chunks)


def encode_payload(value):
    if np is not None and isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            NDARRAY_MARKER: True,
            "dtype": str(array.dtype),
            "shape": array.shape,
            "data": array.tobytes(order="C"),
        }
    if isinstance(value, dict):
        return {key: encode_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [encode_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(encode_payload(item) for item in value)
    return value


def decode_payload(value):
    if isinstance(value, dict):
        if value.get(NDARRAY_MARKER):
            if np is None:
                raise ProtocolError("received numpy array but numpy is not installed")
            array = np.frombuffer(value["data"], dtype=np.dtype(value["dtype"]))
            return array.reshape(tuple(value["shape"])).copy()
        return {key: decode_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(decode_payload(item) for item in value)
    return value


def send_message(sock, message):
    payload = pickle.dumps(encode_payload(message), protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) > MAX_MESSAGE_SIZE:
        raise ProtocolError(f"message too large: {len(payload)} bytes")
    sock.sendall(HEADER_STRUCT.pack(len(payload)))
    sock.sendall(payload)


def recv_message(sock):
    header = recvall(sock, HEADER_STRUCT.size)
    (size,) = HEADER_STRUCT.unpack(header)
    if size > MAX_MESSAGE_SIZE:
        raise ProtocolError(f"message too large: {size} bytes")
    return decode_payload(pickle.loads(recvall(sock, size)))


def connect_with_retry(host, port, retry_delay=1.0):
    while True:
        try:
            sock = configure_socket(socket.create_connection((host, port), timeout=5.0))
            sock.settimeout(None)
            return sock
        except OSError as exc:
            print(f">> 连接 {host}:{port} 失败: {exc}，{retry_delay:.1f}s 后重试")
            time.sleep(retry_delay)
