# -*- coding: utf-8 -*-
"""
Linux 游戏窗口观测（Wayland / Proton 优先）。

捕获优先级（均非整桌面“截图工具”路径）:
  1. pipewire  — xdg-desktop-portal ScreenCast → PipeWire 视频流（Wayland 原生、可后台）
  2. x11shm    — 对 XWayland 上的 Proton 窗口做 MIT-SHM 窗口缓冲读取
  3. mss       — 最后回退

环境变量:
  DQN_CAPTURE_METHOD=pipewire|x11shm|mss|auto
  DQN_PIPEWIRE_NODE=<node_id>          # 跳过 portal，直接绑已有 PW 节点
  DQN_PIPEWIRE_RESTORE=<token>         # portal 恢复令牌
  DQN_PIPEWIRE_TOKEN_FILE=path         # 持久化 restore token（默认 .pw_restore_token）
"""
from __future__ import annotations

import atexit
import ctypes
import ctypes.util
import os
import struct
import sys
import threading
import time
from typing import Optional, Tuple

import numpy as np
import psutil

try:
    from mss import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

_libx11 = None
_libxext = None
_x_error_handler = None


# ========================= X11 / XWayland 结构 =========================

class XErrorEvent(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("display", ctypes.c_void_p),
        ("resourceid", ctypes.c_ulong),
        ("serial", ctypes.c_ulong),
        ("error_code", ctypes.c_ubyte),
        ("request_code", ctypes.c_ubyte),
        ("minor_code", ctypes.c_ubyte),
    ]


class XWindowAttributes(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int), ("y", ctypes.c_int),
        ("width", ctypes.c_int), ("height", ctypes.c_int),
        ("border_width", ctypes.c_int), ("depth", ctypes.c_int),
        ("visual", ctypes.c_void_p), ("root", ctypes.c_ulong),
        ("class", ctypes.c_int), ("bit_gravity", ctypes.c_int),
        ("win_gravity", ctypes.c_int), ("backing_store", ctypes.c_int),
        ("backing_planes", ctypes.c_ulong), ("backing_pixel", ctypes.c_ulong),
        ("save_under", ctypes.c_int), ("colormap", ctypes.c_ulong),
        ("map_installed", ctypes.c_int), ("map_state", ctypes.c_int),
        ("all_event_masks", ctypes.c_long), ("your_event_mask", ctypes.c_long),
        ("do_not_propagate_mask", ctypes.c_long), ("override_redirect", ctypes.c_int),
        ("screen", ctypes.c_void_p),
    ]


class XImage(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int), ("height", ctypes.c_int),
        ("xoffset", ctypes.c_int), ("format", ctypes.c_int),
        ("data", ctypes.c_void_p), ("byte_order", ctypes.c_int),
        ("bitmap_unit", ctypes.c_int), ("bitmap_bit_order", ctypes.c_int),
        ("bitmap_pad", ctypes.c_int), ("depth", ctypes.c_int),
        ("bytes_per_line", ctypes.c_int), ("bits_per_pixel", ctypes.c_int),
        ("red_mask", ctypes.c_ulong), ("green_mask", ctypes.c_ulong),
        ("blue_mask", ctypes.c_ulong), ("obdata", ctypes.c_void_p),
    ]


class XShmSegmentInfo(ctypes.Structure):
    _fields_ = [
        ("shmseg", ctypes.c_ulong),
        ("shmid", ctypes.c_int),
        ("shmaddr", ctypes.c_void_p),
        ("readOnly", ctypes.c_int),
    ]


def _load_x11():
    global _libx11, _libxext, _x_error_handler
    if _libx11 is not None:
        return _libx11 is not False

    name = ctypes.util.find_library("X11")
    if not name:
        _libx11 = False
        return False

    x11 = ctypes.CDLL(name)
    ext_name = ctypes.util.find_library("Xext")
    xext = ctypes.CDLL(ext_name) if ext_name else None

    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
    x11.XDefaultRootWindow.restype = ctypes.c_ulong
    x11.XQueryTree.argtypes = [
        ctypes.c_void_p, ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)), ctypes.POINTER(ctypes.c_uint),
    ]
    x11.XQueryTree.restype = ctypes.c_int
    x11.XFree.argtypes = [ctypes.c_void_p]
    x11.XGetWindowAttributes.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(XWindowAttributes)]
    x11.XGetWindowAttributes.restype = ctypes.c_int
    x11.XGetWindowProperty.argtypes = [
        ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_long, ctypes.c_long,
        ctypes.c_int, ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
    ]
    x11.XGetWindowProperty.restype = ctypes.c_int
    x11.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    x11.XInternAtom.restype = ctypes.c_ulong
    x11.XTranslateCoordinates.argtypes = [
        ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_ulong),
    ]
    x11.XGetImage.argtypes = [
        ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_ulong, ctypes.c_int,
    ]
    x11.XGetImage.restype = ctypes.POINTER(XImage)
    x11.XDestroyImage.argtypes = [ctypes.POINTER(XImage)]
    x11.XSetErrorHandler.argtypes = [ctypes.c_void_p]
    x11.XSetErrorHandler.restype = ctypes.c_void_p
    x11.XFlush.argtypes = [ctypes.c_void_p]
    x11.XSync.argtypes = [ctypes.c_void_p, ctypes.c_int]

    if xext:
        xext.XShmQueryExtension.argtypes = [ctypes.c_void_p]
        xext.XShmQueryExtension.restype = ctypes.c_int
        xext.XShmCreateImage.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_int,
            ctypes.c_void_p, ctypes.POINTER(XShmSegmentInfo), ctypes.c_uint, ctypes.c_uint,
        ]
        xext.XShmCreateImage.restype = ctypes.POINTER(XImage)
        xext.XShmAttach.argtypes = [ctypes.c_void_p, ctypes.POINTER(XShmSegmentInfo)]
        xext.XShmAttach.restype = ctypes.c_int
        xext.XShmDetach.argtypes = [ctypes.c_void_p, ctypes.POINTER(XShmSegmentInfo)]
        xext.XShmGetImage.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(XImage),
            ctypes.c_int, ctypes.c_int, ctypes.c_ulong,
        ]
        xext.XShmGetImage.restype = ctypes.c_int

    @ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(XErrorEvent))
    def _handler(_display, _event):
        return 0

    _x_error_handler = _handler
    x11.XSetErrorHandler(_x_error_handler)
    _libx11 = x11
    _libxext = xext
    return True


def _find_pids_for_process(process_name: str):
    target = process_name.lower()
    pids = set()
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (proc.info.get("name") or "").lower()
            if name == target or target in name:
                pids.add(proc.info["pid"])
                continue
            for part in proc.info.get("cmdline") or []:
                if target in os.path.basename(part).lower():
                    pids.add(proc.info["pid"])
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return pids


def _pid_matches(pid: int, process_name: str) -> bool:
    try:
        proc = psutil.Process(pid)
        target = process_name.lower()
        if proc.name().lower() == target or target in proc.name().lower():
            return True
        for part in proc.cmdline():
            if target in os.path.basename(part).lower():
                return True
        for child in proc.children(recursive=True):
            try:
                if target in child.name().lower():
                    return True
                for part in child.cmdline():
                    if target in os.path.basename(part).lower():
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    return False


# ========================= PipeWire + Portal =========================

class PipeWirePortalCapture:
    """
    xdg-desktop-portal ScreenCast → OpenPipeWireRemote(fd) → GStreamer appsink。

    关键：必须用 portal 返回的 **fd** 连接 PipeWire（仅 node id 在会话结束后会失效）。
    会话对象需保持存活，否则流断开。
    """

    def __init__(self, token_file: str = ".pw_restore_token"):
        self.token_file = os.environ.get("DQN_PIPEWIRE_TOKEN_FILE", token_file)
        self.node_id = os.environ.get("DQN_PIPEWIRE_NODE") or None
        self.restore_token = os.environ.get("DQN_PIPEWIRE_RESTORE") or self._load_token()
        self.pipeline = None
        self.sink = None
        self.Gst = None
        self.session = None  # session_handle 字符串，须保持 portal 会话
        self._pw_fd = None  # OpenPipeWireRemote 返回的 fd
        self._bus = None
        self._lock = threading.Lock()
        self._started = False
        self._last_error = None

    def _load_token(self) -> Optional[str]:
        try:
            if os.path.isfile(self.token_file):
                with open(self.token_file, "r", encoding="utf-8") as f:
                    t = f.read().strip()
                    return t or None
        except OSError:
            pass
        return None

    def _save_token(self, token: str):
        if not token:
            return
        try:
            with open(self.token_file, "w", encoding="utf-8") as f:
                f.write(token)
            print(f">> PipeWire restore token 已保存: {self.token_file}")
        except OSError as e:
            print(f">> 无法保存 restore token: {e}")

    def _portal_start(self) -> Optional[int]:
        """通过 portal 获取 pipewire node id。返回 node id 或 None。"""
        try:
            import dbus
            from dbus.mainloop.glib import DBusGMainLoop
            from gi.repository import GLib
        except ImportError as e:
            self._last_error = f"缺少 dbus/gi: {e}"
            return None

        DBusGMainLoop(set_as_default=True)
        bus = dbus.SessionBus()
        try:
            portal = bus.get_object(
                "org.freedesktop.portal.Desktop",
                "/org/freedesktop/portal/desktop",
            )
        except dbus.exceptions.DBusException as e:
            self._last_error = f"无法连接 xdg-desktop-portal: {e}"
            return None

        screencast = dbus.Interface(portal, "org.freedesktop.portal.ScreenCast")
        loop = GLib.MainLoop()
        result_holder = {"node": None, "token": None, "error": None}

        def _call_with_request(method, *args, timeout=120.0):
            request_path_holder = {"path": None}
            done = {"v": False}

            def on_response(response, results):
                done["v"] = True
                if response != 0:
                    result_holder["error"] = f"portal 拒绝/取消 (code={response})"
                    loop.quit()
                    return
                request_path_holder["results"] = dict(results)
                loop.quit()

            # Create session / select / start 都返回 request handle
            request_handle = method(*args)
            request_path_holder["path"] = str(request_handle)
            req = bus.get_object("org.freedesktop.portal.Desktop", request_handle)
            req_if = dbus.Interface(req, "org.freedesktop.portal.Request")
            req_if.connect_to_signal("Response", on_response)

            def on_timeout():
                if not done["v"]:
                    result_holder["error"] = "portal 请求超时"
                    loop.quit()
                return False

            GLib.timeout_add(int(timeout * 1000), on_timeout)
            loop.run()
            if result_holder["error"]:
                return None
            return request_path_holder.get("results")

        sender = bus.get_unique_name()[1:].replace(".", "_")
        token = f"dqn{int(time.time())}"
        session_opts = {
            "session_handle_token": dbus.String(token),
            "handle_token": dbus.String(token + "_create"),
        }
        # CreateSession
        try:
            create_handle = screencast.CreateSession(session_opts)
        except Exception as e:
            self._last_error = f"CreateSession 失败: {e}"
            return None

        create_done = {"results": None, "err": None}

        def on_create(response, results):
            if response != 0:
                create_done["err"] = f"CreateSession 失败 code={response}"
            else:
                create_done["results"] = dict(results)
            loop.quit()

        req = bus.get_object("org.freedesktop.portal.Desktop", create_handle)
        dbus.Interface(req, "org.freedesktop.portal.Request").connect_to_signal("Response", on_create)
        GLib.timeout_add(60000, lambda: (loop.quit(), False)[1])
        loop.run()
        if create_done["err"] or not create_done["results"]:
            self._last_error = create_done["err"] or "CreateSession 无结果"
            return None

        session_handle = create_done["results"].get("session_handle")
        if not session_handle:
            self._last_error = "无 session_handle"
            return None
        self.session = str(session_handle)

        select_opts = {
            "types": dbus.UInt32(1 | 2),  # monitor | window
            "multiple": dbus.Boolean(False),
            "handle_token": dbus.String(token + "_select"),
        }
        # cursor_mode: 2 = embedded（可选）; 我们不需要鼠标光标 →  hidden=1
        try:
            select_opts["cursor_mode"] = dbus.UInt32(1)  # hidden
        except Exception:
            pass
        if self.restore_token:
            select_opts["restore_token"] = dbus.String(self.restore_token)
            # persist_mode: 2 = 永久直到撤销
            select_opts["persist_mode"] = dbus.UInt32(2)

        try:
            select_handle = screencast.SelectSources(session_handle, select_opts)
        except Exception as e:
            self._last_error = f"SelectSources 失败: {e}"
            return None

        select_done = {"ok": False, "err": None}

        def on_select(response, results):
            if response != 0:
                select_done["err"] = f"SelectSources 取消/失败 code={response}"
            else:
                select_done["ok"] = True
            loop.quit()

        req = bus.get_object("org.freedesktop.portal.Desktop", select_handle)
        dbus.Interface(req, "org.freedesktop.portal.Request").connect_to_signal("Response", on_select)
        print(">> 请在弹出的系统对话框中选择【游戏窗口】进行共享（仅首次/令牌失效时）")
        GLib.timeout_add(180000, lambda: (loop.quit(), False)[1])
        loop.run()
        if not select_done["ok"]:
            self._last_error = select_done["err"] or "SelectSources 失败"
            return None

        start_opts = {"handle_token": dbus.String(token + "_start")}
        try:
            start_handle = screencast.Start(session_handle, "", start_opts)
        except Exception as e:
            self._last_error = f"Start 失败: {e}"
            return None

        start_done = {"results": None, "err": None}

        def on_start(response, results):
            if response != 0:
                start_done["err"] = f"Start 取消/失败 code={response}"
            else:
                start_done["results"] = dict(results)
            loop.quit()

        req = bus.get_object("org.freedesktop.portal.Desktop", start_handle)
        dbus.Interface(req, "org.freedesktop.portal.Request").connect_to_signal("Response", on_start)
        GLib.timeout_add(120000, lambda: (loop.quit(), False)[1])
        loop.run()
        if start_done["err"] or not start_done["results"]:
            self._last_error = start_done["err"] or "Start 无结果"
            return None

        results = start_done["results"]
        streams = results.get("streams")
        restore = results.get("restore_token")
        if restore:
            self.restore_token = str(restore)
            self._save_token(self.restore_token)

        if not streams:
            self._last_error = "portal 未返回 streams"
            return None

        # streams: array of (node_id, props)
        try:
            first = streams[0]
            node_id = int(first[0])
            props = dict(first[1]) if len(first) > 1 else {}
            size = props.get("size")
            if size is not None:
                try:
                    sw, sh = int(size[0]), int(size[1])
                    if sw <= 1 or sh <= 1:
                        print(
                            f">> 警告: portal 报告 size={sw}x{sh}（异常小），"
                            f"请确认选的是游戏窗口而非装饰条"
                        )
                except Exception:
                    pass
        except Exception as e:
            self._last_error = f"解析 stream 失败: {e}"
            return None

        # 必须 OpenPipeWireRemote 拿到 fd，仅凭 node id 会在会话外失效
        try:
            import dbus.lowlevel as dbus_lowlevel

            msg = dbus_lowlevel.MethodCallMessage(
                "org.freedesktop.portal.Desktop",
                "/org/freedesktop/portal/desktop",
                "org.freedesktop.portal.ScreenCast",
                "OpenPipeWireRemote",
            )
            msg.append(session_handle, dbus.Dictionary({}, signature="sv"))
            reply = bus.send_message_with_reply_and_block(msg, 15000)
            fds = reply.get_unix_fds()
            if not fds:
                # 部分绑定把 UnixFd 放在 body
                body = reply.get_args_list()
                for item in body:
                    if hasattr(item, "take"):
                        self._pw_fd = int(item.take())
                        break
                    if type(item).__name__ == "UnixFd":
                        self._pw_fd = int(item)
                        break
                if self._pw_fd is None:
                    raise RuntimeError(f"OpenPipeWireRemote 无 fd, body={body!r}")
            else:
                self._pw_fd = int(fds[0])
        except Exception as e:
            self._last_error = f"OpenPipeWireRemote 失败: {e}"
            return None

        self._bus = bus
        print(
            f">> Portal ScreenCast 已授权, pipewire node={node_id} fd={self._pw_fd}"
        )
        return node_id

    def start(self) -> bool:
        with self._lock:
            if self._started:
                return True
            try:
                import gi
                gi.require_version("Gst", "1.0")
                from gi.repository import Gst
            except Exception as e:
                self._last_error = f"GStreamer 不可用: {e}"
                return False

            Gst.init(None)
            self.Gst = Gst

            node = self.node_id
            if not node or self._pw_fd is None:
                # 需要完整 portal 流程（含 fd）；仅有 DQN_PIPEWIRE_NODE 不够
                if self.node_id and self._pw_fd is None and not self.session:
                    print(
                        ">> 仅有 node id 无 portal fd，重新走 ScreenCast 授权…"
                    )
                    self.node_id = None
                node = self._portal_start()
                if node is None:
                    return False
                self.node_id = str(node)

            node = str(self.node_id)
            fd = self._pw_fd
            # 优先：fd + target-object（portal 正确用法）
            descs = []
            if fd is not None:
                descs.append(
                    f'pipewiresrc fd={fd} target-object={node} do-timestamp=true ! '
                    f'videoconvert n-threads=2 ! video/x-raw,format=BGR ! '
                    f'appsink name=sink max-buffers=1 drop=true sync=false emit-signals=false'
                )
                descs.append(
                    f'pipewiresrc fd={fd} path={node} do-timestamp=true ! '
                    f'videoconvert n-threads=2 ! video/x-raw,format=BGR ! '
                    f'appsink name=sink max-buffers=1 drop=true sync=false emit-signals=false'
                )
                descs.append(
                    f'pipewiresrc fd={fd} do-timestamp=true ! '
                    f'videoconvert n-threads=2 ! video/x-raw,format=BGR ! '
                    f'appsink name=sink max-buffers=1 drop=true sync=false emit-signals=false'
                )
            descs.append(
                f'pipewiresrc target-object={node} do-timestamp=true ! '
                f'videoconvert n-threads=2 ! video/x-raw,format=BGR ! '
                f'appsink name=sink max-buffers=1 drop=true sync=false emit-signals=false'
            )

            last_err = None
            for desc in descs:
                try:
                    self.pipeline = Gst.parse_launch(desc)
                    self.sink = self.pipeline.get_by_name("sink")
                    ret = self.pipeline.set_state(Gst.State.PLAYING)
                    if ret == Gst.StateChangeReturn.FAILURE:
                        self.pipeline.set_state(Gst.State.NULL)
                        last_err = "set_state PLAYING failure"
                        continue
                    state_ret, state, pending = self.pipeline.get_state(5 * Gst.SECOND)
                    if state_ret == Gst.StateChangeReturn.FAILURE:
                        self.pipeline.set_state(Gst.State.NULL)
                        last_err = f"preroll failed state={state}"
                        continue
                    sample = self.sink.emit("try-pull-sample", 3000 * Gst.MSECOND)
                    if sample is None:
                        self.pipeline.set_state(Gst.State.NULL)
                        last_err = "appsink 无帧"
                        continue
                    caps = sample.get_caps().get_structure(0)
                    tw = int(caps.get_value("width"))
                    th = int(caps.get_value("height"))
                    if tw <= 2 or th <= 2:
                        self.pipeline.set_state(Gst.State.NULL)
                        last_err = f"废流尺寸 {tw}x{th}（请重选游戏窗口）"
                        continue
                    self._started = True
                    print(
                        f">> PipeWire 捕获管线运行中 (node={node} fd={fd} {tw}x{th})"
                    )
                    return True
                except Exception as e:
                    last_err = str(e)
                    try:
                        if self.pipeline:
                            self.pipeline.set_state(Gst.State.NULL)
                    except Exception:
                        pass
                    self.pipeline = None
            self._last_error = f"GStreamer 管线失败: {last_err}"
            return False

    def grab(self, region=None) -> Optional[np.ndarray]:
        if not self._started and not self.start():
            return None
        Gst = self.Gst
        sample = self.sink.emit("try-pull-sample", 50 * Gst.MSECOND)
        if sample is None:
            sample = self.sink.emit("try-pull-sample", 200 * Gst.MSECOND)
        if sample is None:
            return None

        buf = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        w = int(caps.get_value("width"))
        h = int(caps.get_value("height"))
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return None
        try:
            arr = np.frombuffer(mapinfo.data, dtype=np.uint8)
            need = w * h * 3
            if arr.size < need:
                # 可能有 padding
                bpl = arr.size // max(h, 1)
                if bpl >= w * 3:
                    frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape(h, bpl)[:, : w * 3]
                    frame = frame.reshape(h, w, 3).copy()
                else:
                    return None
            else:
                frame = arr[:need].reshape(h, w, 3).copy()
        finally:
            buf.unmap(mapinfo)

        if region:
            left, top, right, bot = region
            # region 若是屏幕坐标，pipewire 流已是窗口/输出内容，按相对裁剪
            if right <= frame.shape[1] and bot <= frame.shape[0]:
                frame = frame[top:bot, left:right].copy()
        return frame

    def stop(self):
        with self._lock:
            if self.pipeline is not None and self.Gst is not None:
                try:
                    self.pipeline.set_state(self.Gst.State.NULL)
                except Exception:
                    pass
            self.pipeline = None
            self.sink = None
            self._started = False
            if self._pw_fd is not None:
                try:
                    os.close(self._pw_fd)
                except Exception:
                    pass
                self._pw_fd = None
            # session 由 portal 在进程结束时回收；置空引用
            self.session = None


# ========================= 统一 Grabber =========================

class LinuxScreenGrabber:
    def __init__(self):
        self.display = None
        self.hwnd = None
        self.process_name = None
        self.sct = mss() if HAS_MSS else None
        self._shm_cache = None
        self._lock = threading.Lock()
        self._pw = PipeWirePortalCapture()
        self._method_pref = None

        if os.environ.get("DISPLAY") and _load_x11():
            dname = os.environ.get("DISPLAY", "").encode() or None
            self.display = _libx11.XOpenDisplay(dname)
            if self.display:
                print(">> 检测到 XWayland/X11 DISPLAY，可用窗口级 XShm 回退")

    def release(self):
        with self._lock:
            self._release_shm()
            if self.display and _libx11:
                try:
                    _libx11.XCloseDisplay(self.display)
                except Exception:
                    pass
                self.display = None
            if self.sct is not None:
                try:
                    self.sct.close()
                except Exception:
                    pass
                self.sct = None
            self._pw.stop()

    def _release_shm(self):
        if not self._shm_cache:
            return
        _wid, _w, _h, ximg, shminfo, _addr = self._shm_cache
        try:
            if _libxext and shminfo is not None:
                _libxext.XShmDetach(self.display, ctypes.byref(shminfo))
            if ximg:
                _libx11.XDestroyImage(ximg)
        except Exception:
            pass
        try:
            libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
            if shminfo is not None and shminfo.shmaddr:
                libc.shmdt(shminfo.shmaddr)
                libc.shmctl(shminfo.shmid, 0, None)
        except Exception:
            pass
        self._shm_cache = None

    def _atom(self, name: str) -> int:
        return _libx11.XInternAtom(self.display, name.encode(), 0)

    def _get_property(self, window, atom, atom_type, length=1):
        actual_type = ctypes.c_ulong()
        actual_format = ctypes.c_int()
        nitems = ctypes.c_ulong()
        bytes_after = ctypes.c_ulong()
        prop = ctypes.POINTER(ctypes.c_ubyte)()
        status = _libx11.XGetWindowProperty(
            self.display, window, atom, 0, length, 0, atom_type,
            ctypes.byref(actual_type), ctypes.byref(actual_format),
            ctypes.byref(nitems), ctypes.byref(bytes_after), ctypes.byref(prop),
        )
        if status != 0 or not prop:
            return None, 0, 0
        return prop, int(nitems.value), int(actual_format.value)

    def _window_pid(self, window) -> Optional[int]:
        prop, nitems, fmt = self._get_property(window, self._atom("_NET_WM_PID"), 0)
        if not prop or nitems < 1:
            return None
        try:
            raw = ctypes.string_at(prop, max(4, nitems * (fmt // 8 or 4)))
            return int(struct.unpack("I", raw[:4])[0])
        finally:
            _libx11.XFree(prop)

    def _window_name(self, window) -> str:
        for atom_name in ("_NET_WM_NAME", "WM_NAME"):
            prop, nitems, fmt = self._get_property(
                window, self._atom(atom_name), 0, length=1024
            )
            if prop and nitems:
                try:
                    raw = ctypes.string_at(prop, nitems * max(fmt // 8, 1))
                    return raw.split(b"\x00", 1)[0].decode("utf-8", errors="ignore")
                finally:
                    _libx11.XFree(prop)
        return ""

    def _iter_windows(self, root=None):
        if root is None:
            root = _libx11.XDefaultRootWindow(self.display)
        root_return = ctypes.c_ulong()
        parent_return = ctypes.c_ulong()
        children = ctypes.POINTER(ctypes.c_ulong)()
        nchildren = ctypes.c_uint()
        if not _libx11.XQueryTree(
            self.display, root,
            ctypes.byref(root_return), ctypes.byref(parent_return),
            ctypes.byref(children), ctypes.byref(nchildren),
        ):
            return
        try:
            for i in range(nchildren.value):
                child = children[i]
                yield child
                yield from self._iter_windows(child)
        finally:
            if children:
                _libx11.XFree(children)

    def get_window_handle(self, process_name: str):
        if not self.display:
            return None
        if self.hwnd and self.process_name == process_name:
            attrs = XWindowAttributes()
            if _libx11.XGetWindowAttributes(self.display, self.hwnd, ctypes.byref(attrs)):
                if attrs.map_state != 0 and attrs.width > 0 and attrs.height > 0:
                    return self.hwnd

        pids = _find_pids_for_process(process_name)
        best = None
        best_area = 0
        target_lower = process_name.lower().replace(".exe", "")

        for win in self._iter_windows():
            attrs = XWindowAttributes()
            if not _libx11.XGetWindowAttributes(self.display, win, ctypes.byref(attrs)):
                continue
            if attrs.map_state != 2 or attrs.width < 64 or attrs.height < 64:
                continue
            pid = self._window_pid(win)
            name = self._window_name(win).lower()
            matched = False
            if pid is not None and (pid in pids or _pid_matches(pid, process_name)):
                matched = True
            elif target_lower and target_lower in name:
                matched = True
            elif "yakuza" in name and "yakuza" in target_lower:
                matched = True
            if not matched:
                continue
            area = attrs.width * attrs.height
            if area > best_area:
                best_area = area
                best = win

        if best:
            self.hwnd = best
            self.process_name = process_name
        return best

    def _client_geometry(self, hwnd) -> Optional[Tuple[int, int, int, int]]:
        attrs = XWindowAttributes()
        if not _libx11.XGetWindowAttributes(self.display, hwnd, ctypes.byref(attrs)):
            return None
        root = _libx11.XDefaultRootWindow(self.display)
        dx = ctypes.c_int()
        dy = ctypes.c_int()
        child = ctypes.c_ulong()
        _libx11.XTranslateCoordinates(
            self.display, hwnd, root, 0, 0,
            ctypes.byref(dx), ctypes.byref(dy), ctypes.byref(child),
        )
        return int(dx.value), int(dy.value), int(attrs.width), int(attrs.height)

    def _ensure_shm(self, hwnd, width, height):
        if (
            self._shm_cache
            and self._shm_cache[0] == hwnd
            and self._shm_cache[1] == width
            and self._shm_cache[2] == height
        ):
            return True
        self._release_shm()
        if not _libxext or not _libxext.XShmQueryExtension(self.display):
            return False

        attrs = XWindowAttributes()
        _libx11.XGetWindowAttributes(self.display, hwnd, ctypes.byref(attrs))
        shminfo = XShmSegmentInfo()
        ximg = _libxext.XShmCreateImage(
            self.display, attrs.visual, attrs.depth, 2, None,
            ctypes.byref(shminfo), width, height,
        )
        if not ximg:
            return False

        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        libc.shmget.argtypes = [ctypes.c_int, ctypes.c_size_t, ctypes.c_int]
        libc.shmget.restype = ctypes.c_int
        libc.shmat.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        libc.shmat.restype = ctypes.c_void_p

        size = ximg.contents.bytes_per_line * height
        shmid = libc.shmget(0, size, 0o600 | 0o1000)
        if shmid < 0:
            _libx11.XDestroyImage(ximg)
            return False
        addr = libc.shmat(shmid, None, 0)
        if addr in (-1, ctypes.c_void_p(-1).value):
            libc.shmctl(shmid, 0, None)
            _libx11.XDestroyImage(ximg)
            return False

        shminfo.shmid = shmid
        shminfo.shmaddr = addr
        shminfo.readOnly = 0
        ximg.contents.data = addr
        ximg.contents.obdata = ctypes.cast(ctypes.pointer(shminfo), ctypes.c_void_p)

        if not _libxext.XShmAttach(self.display, ctypes.byref(shminfo)):
            libc.shmdt(addr)
            libc.shmctl(shmid, 0, None)
            _libx11.XDestroyImage(ximg)
            return False
        _libx11.XSync(self.display, 0)
        self._shm_cache = (hwnd, width, height, ximg, shminfo, addr)
        return True

    def _ximage_to_bgr(self, ximg) -> Optional[np.ndarray]:
        w = ximg.contents.width
        h = ximg.contents.height
        bpl = ximg.contents.bytes_per_line
        bpp = ximg.contents.bits_per_pixel
        data_ptr = ximg.contents.data
        if not data_ptr or w <= 0 or h <= 0:
            return None
        buf = ctypes.string_at(data_ptr, bpl * h)
        little = sys.byteorder == "little"
        if bpp == 32:
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, bpl)[:, : w * 4]
            pixel = arr.reshape(h, w, 4)[:, :, :3]
            # red_mask==0xFF0000: 32bit 像素值 0x00RRGGBB，little-endian 内存字节序为 B,G,R(,A)
            #   -> 内存即 BGR，无需反转；big-endian 内存为 R,G,B，需反转成 BGR。
            if ximg.contents.red_mask == 0xFF0000:
                return pixel.copy() if little else pixel[:, :, ::-1].copy()
            # red_mask==0x0000FF: 像素值 0x00BBGGRR，little-endian 内存为 R,G,B
            #   -> 需反转成 BGR；big-endian 内存即 BGR，无需反转。
            return pixel[:, :, ::-1].copy() if little else pixel.copy()
        if bpp == 24:
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, bpl)[:, : w * 3]
            # 24bpp 每像素 3 字节，内存字节序同样按端序决定
            return arr.reshape(h, w, 3) if little else arr.reshape(h, w, 3)[:, :, ::-1].copy()
        return None

    def grab_x11shm(self, process_name, region=None):
        if not self.display:
            return None
        hwnd = self.get_window_handle(process_name)
        if not hwnd:
            return None
        geom = self._client_geometry(hwnd)
        if not geom:
            return None
        sx, sy, width, height = geom
        src_x = src_y = 0
        if region:
            left, top, right, bot = region
            src_x = max(0, left - sx)
            src_y = max(0, top - sy)
            width = min(width - src_x, right - left)
            height = min(height - src_y, bot - top)
            if width <= 0 or height <= 0:
                return None

        with self._lock:
            if self._ensure_shm(hwnd, width, height):
                _, _, _, ximg, _, _ = self._shm_cache
                ok = _libxext.XShmGetImage(
                    self.display, hwnd, ximg, src_x, src_y, 0xFFFFFFFF
                )
                _libx11.XFlush(self.display)
                if ok:
                    img = self._ximage_to_bgr(ximg)
                    if img is not None:
                        return img

            ximg = _libx11.XGetImage(
                self.display, hwnd, src_x, src_y, width, height, 0xFFFFFFFF, 2
            )
            if not ximg:
                return None
            try:
                return self._ximage_to_bgr(ximg)
            finally:
                _libx11.XDestroyImage(ximg)

    def grab_mss(self, process_name, region=None):
        if not HAS_MSS or self.sct is None:
            return None
        if region:
            monitor = {
                "top": region[1], "left": region[0],
                "width": region[2] - region[0], "height": region[3] - region[1],
            }
        else:
            if not self.display:
                return None
            hwnd = self.get_window_handle(process_name)
            if not hwnd:
                return None
            geom = self._client_geometry(hwnd)
            if not geom:
                return None
            sx, sy, w, h = geom
            monitor = {"top": sy, "left": sx, "width": w, "height": h}
        shot = self.sct.grab(monitor)
        img = np.frombuffer(shot.rgb, dtype=np.uint8).reshape(shot.height, shot.width, 3)
        return img[:, :, ::-1].copy()

    def grab_pipewire(self, process_name, region=None):
        frame = self._pw.grab(region=region)
        if frame is None and self._pw._last_error:
            # 仅首次打印
            if not getattr(self, "_pw_err_printed", False):
                print(f">> PipeWire 捕获失败: {self._pw._last_error}")
                self._pw_err_printed = True
        return frame

    def grab_auto(self, process_name, region=None):
        """
        Wayland 会话: 优先 PipeWire portal 流；
        Proton 多在 XWayland: 同步尝试窗口级 XShm（通常更快且无需弹窗）。
        """
        # 已选定的稳定路径
        if self._method_pref == "pipewire":
            frame = self.grab_pipewire(process_name, region)
            if frame is not None:
                return frame
        if self._method_pref == "x11shm":
            frame = self.grab_x11shm(process_name, region)
            if frame is not None:
                return frame

        # 探测: XWayland 窗口级（Proton 常见，零 portal）
        frame = self.grab_x11shm(process_name, region)
        if frame is not None:
            if self._method_pref != "x11shm":
                print(">> 捕获后端: X11/XWayland MIT-SHM（窗口缓冲，非截图）")
                self._method_pref = "x11shm"
            return frame

        # PipeWire portal（纯 Wayland 窗口 / 需要授权）
        frame = self.grab_pipewire(process_name, region)
        if frame is not None:
            if self._method_pref != "pipewire":
                print(">> 捕获后端: PipeWire ScreenCast（Wayland portal 流）")
                self._method_pref = "pipewire"
            return frame

        frame = self.grab_mss(process_name, region)
        if frame is not None and self._method_pref != "mss":
            print(">> 捕获后端: MSS 回退")
            self._method_pref = "mss"
        return frame


_grabber = LinuxScreenGrabber()
atexit.register(_grabber.release)


def grab_screen(process_name, method="auto", region=None):
    method = (method or "auto").lower()
    if method in ("auto", "default"):
        return _grabber.grab_auto(process_name, region)
    if method in ("pipewire", "pw", "portal", "wayland"):
        return _grabber.grab_pipewire(process_name, region)
    if method in ("x11shm", "x11", "shm", "xwayland"):
        return _grabber.grab_x11shm(process_name, region)
    if method in ("niri", "screenshot"):
        print(">> niri 截图捕获已禁用，改用 pipewire/auto")
        return _grabber.grab_auto(process_name, region)
    if method == "mss":
        return _grabber.grab_mss(process_name, region)
    return _grabber.grab_auto(process_name, region)
