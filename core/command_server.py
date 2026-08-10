# -*- coding: utf-8 -*-
"""
AI 侧命令分发：解析客户端消息并回调 handler。

与 DecisionClient 对称——客户端发命令，这里收命令执行。
具体训练/推理逻辑由 handler 注入，本模块只做编解码与路由。
"""
from __future__ import annotations

import socket
import threading
import time
from typing import Any, Callable, Dict, Optional

from core.protocol import configure_socket, recv_message, send_message


MessageHandler = Callable[[str, Dict[str, Any]], Optional[Dict[str, Any]]]


class CommandServer:
    """
    监听 TCP，按 message['type'] 分发。

    内置类型约定（与 DecisionClient 对齐）:
      action_request -> handler 返回 {"type":"plan", "plan":[int], "commit":int, ...}
      transition     -> handler 返回 {"type":"ack"} 或 None（自动 ack）
      reset_hidden   -> 同上
      save           -> 同上
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 15001):
        self.host = host
        self.port = int(port)
        self.is_running = True
        self._handlers: Dict[str, MessageHandler] = {}
        self._default_handler: Optional[MessageHandler] = None

    def on(self, msg_type: str, handler: MessageHandler):
        self._handlers[msg_type] = handler
        return self

    def on_default(self, handler: MessageHandler):
        self._default_handler = handler
        return self

    def _dispatch(self, client_id: str, msg: Dict[str, Any]) -> Dict[str, Any]:
        msg_type = msg.get("type")
        handler = self._handlers.get(msg_type) or self._default_handler
        if handler is None:
            return {"type": "error", "message": f"unknown type: {msg_type}"}
        result = handler(client_id, msg)
        if result is None:
            return {"type": "ack"}
        return result

    def _handle_client(self, conn, addr):
        client_id = f"{addr[0]}:{addr[1]}"
        print(f">> 客户端已连接: {client_id}")
        try:
            while self.is_running:
                msg = recv_message(conn)
                try:
                    reply = self._dispatch(client_id, msg)
                except Exception as exc:
                    reply = {"type": "error", "message": str(exc)}
                send_message(conn, reply)
        except ConnectionError:
            print(f">> 客户端断开: {client_id}")
        except Exception as exc:
            print(f">> 连接错误 {client_id}: {exc}")
        finally:
            try:
                # 通知业务层清理
                if "client_disconnect" in self._handlers:
                    self._handlers["client_disconnect"](client_id, {"type": "client_disconnect"})
            except Exception:
                pass
            conn.close()

    def serve_forever(self, ready_callback: Callable = None):
        if ready_callback:
            ready_callback()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            configure_socket(server)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.settimeout(1.0)
            try:
                server.bind((self.host, self.port))
            except PermissionError as exc:
                raise PermissionError(
                    f"无法监听 {self.host}:{self.port}。请换端口或检查权限。"
                ) from exc
            server.listen()
            print(f">> CommandServer 监听 {self.host}:{self.port}")
            try:
                while self.is_running:
                    try:
                        conn, addr = server.accept()
                    except socket.timeout:
                        continue
                    configure_socket(conn)
                    threading.Thread(
                        target=self._handle_client, args=(conn, addr), daemon=True
                    ).start()
            finally:
                self.is_running = False

    def stop(self):
        self.is_running = False
