# -*- coding: utf-8 -*-
"""兼容层：协议实现已迁至 core.protocol。"""
from core.protocol import (  # noqa: F401
    HEADER_STRUCT,
    MAX_MESSAGE_SIZE,
    NDARRAY_MARKER,
    ProtocolError,
    configure_socket,
    connect_with_retry,
    decode_payload,
    encode_payload,
    recv_message,
    recvall,
    send_message,
)
