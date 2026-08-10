# -*- coding: utf-8 -*-
"""
Linux 按键注入（Proton/Wine 友好）。

默认 uinput 虚拟键盘（训练用，不持续抓鼠标）。
resume 时可临时创建虚拟鼠标：移动到窗口中心并单击，帮助游戏获得焦点/捕获。

输入安全守卫（默认开启，环境变量 DQN_INPUT_GUARD=0 关闭）：
  uinput 事件会落在「当前聚焦窗口」。为确保 AI 输入只进入游戏窗口，
  每次注入前都校验游戏窗口是否聚焦；若未聚焦则立即终止进程，
  避免把按键/点击打进用户正在使用的其它应用。

需要: python-evdev，以及对 /dev/uinput 的写权限
"""
from __future__ import annotations

import atexit
import ctypes
import os
import sys
import time
from typing import Optional, Tuple

# 与 Windows 版相同的扫描码常量（对外 API 兼容）
W = 0x11
A = 0x1E
S = 0x1F
D = 0x20
E = 0x12
J = 0x24
K = 0x25
M = 0x32
L = 0x26
Q = 0x10
I = 0x17
O = 0x18
ENTER = 0x1C
ESC = 0x01

# PS/2 set-1 扫描码 → Linux KEY_*
_SCAN_TO_EVDEV = {
    W: "KEY_W",
    A: "KEY_A",
    S: "KEY_S",
    D: "KEY_D",
    E: "KEY_E",
    J: "KEY_J",
    K: "KEY_K",
    M: "KEY_M",
    L: "KEY_L",
    Q: "KEY_Q",
    I: "KEY_I",
    O: "KEY_O",
    ENTER: "KEY_ENTER",
    ESC: "KEY_ESC",
}

# 扫描码 → X11 keysym 名（XSendEvent 直接送指定窗口，不依赖焦点）
_SCAN_TO_KEYSYM = {
    W: "w",
    A: "a",
    S: "s",
    D: "d",
    E: "e",
    J: "j",
    K: "k",
    M: "m",
    L: "l",
    Q: "q",
    I: "i",
    O: "o",
    ENTER: "Return",
    ESC: "Escape",
}

# 输入注入方式: auto|uinput|xsend
#  xsend = X11 XSendEvent 直接送游戏窗口（XWayland 下不抢焦点，后台可用）
#  uinput = 虚拟键盘（Wayland 原生窗口才需要，依赖焦点）
_input_method = os.environ.get("DQN_INPUT_METHOD", "auto").lower()

_ui = None
_mouse = None
_ecodes = None
_pressed = set()

# ---- X11 XSendEvent 注入状态（不依赖焦点）----
_x11_dpy = None
_x11_hwnd = None
_x11_lib = None
_x11_root = 0
_x11_ok = False

# ---- 输入安全守卫 ----
_process_name = None  # 由 set_process_name() 设置
_guard_enabled = os.environ.get("DQN_INPUT_GUARD", "1").lower() not in ("0", "false", "no")
_guard_cache = {"t": 0.0, "ok": False}
_GUARD_TTL = 0.15  # 焦点校验结果缓存秒数
# 失焦时自动重新聚焦（niri focus-window，不注入真实键鼠）；重试 N 次仍失败才终止
_auto_resume = os.environ.get("DQN_INPUT_AUTO_RESUME", "1").lower() not in ("0", "false", "no")
_AUTO_RESUME_RETRIES = 3
_AUTO_RESUME_GAP = 0.15


def set_process_name(name: Optional[str]):
    """设置目标游戏进程名，供焦点守卫校验（None 关闭守卫）。"""
    global _process_name
    _process_name = name


def invalidate_guard_cache():
    """强制下次注入前重新校验焦点（focus 成功后调用）。"""
    _guard_cache["t"] = 0.0


def _try_auto_resume() -> bool:
    """失焦时尝试把焦点还给游戏窗口（合成器 API，不碰真实键鼠）。"""
    try:
        from backends.linux.window_focus import focus_game_window
        result = focus_game_window(_process_name)
        ok = bool(result and result.get("ok"))
        if ok:
            print(">> 守卫: 游戏窗口失焦，已自动重新聚焦 (niri focus-window)")
        else:
            print(
                f">> 守卫: 自动聚焦失败 {result.get('detail') if result else ''}"
            )
        return ok
    except Exception as e:
        print(f">> 守卫: 自动聚焦异常: {e}")
        return False


def _assert_game_focused():
    """每次注入前调用：uinput 路径必须聚焦；xsend 路径直接送窗口，无需焦点。"""
    if not _guard_enabled or not _process_name:
        return
    if _method_is_xsend():
        # XSendEvent 直接送达游戏窗口，不改变合成器焦点
        if not _x11_ensure():
            _die_focus_lost()
        return
    now = time.monotonic()
    if now - _guard_cache["t"] < _GUARD_TTL:
        if not _guard_cache["ok"]:
            _die_focus_lost()
        return
    from backends.linux.window_focus import is_game_window_focused
    ok = is_game_window_focused(_process_name)
    if not ok and _auto_resume:
        for _ in range(_AUTO_RESUME_RETRIES):
            if _try_auto_resume():
                time.sleep(_AUTO_RESUME_GAP)
                ok = is_game_window_focused(_process_name)
                if ok:
                    break
    _guard_cache["t"] = time.monotonic()
    _guard_cache["ok"] = ok
    if not ok:
        _die_focus_lost()


def _die_focus_lost():
    """自动 resume 后仍失焦：uinput 事件会落入其它窗口，立即终止进程。"""
    print("\n" + "=" * 60)
    print("!! 输入安全守卫触发：自动重新聚焦失败，游戏窗口仍未被聚焦。")
    print("!! 为避免将输入落入其它窗口，程序已立即终止。")
    print("!! 请手动聚焦游戏窗口（或在 game-node 输入 resume / focus）后重启。")
    print("=" * 60)
    try:
        shutdown()
    except Exception:
        pass
    os._exit(2)


# ---- X11 XSendEvent 直接注入（不依赖焦点，XWayland 后台可用）----

class _XKeyEvent(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int), ("serial", ctypes.c_ulong), ("send_event", ctypes.c_int),
        ("display", ctypes.c_void_p), ("window", ctypes.c_ulong), ("root", ctypes.c_ulong),
        ("subwindow", ctypes.c_ulong), ("time", ctypes.c_ulong),
        ("x", ctypes.c_int), ("y", ctypes.c_int),
        ("x_root", ctypes.c_int), ("y_root", ctypes.c_int),
        ("state", ctypes.c_uint), ("keycode", ctypes.c_uint), ("same_screen", ctypes.c_int),
    ]


class _XEvent(ctypes.Union):
    _fields_ = [("type", ctypes.c_int), ("xkey", _XKeyEvent), ("pad", ctypes.c_long * 24)]


def _x11_ensure():
    """复用 capture 的 X 连接 + 游戏窗口句柄。返回是否可用。"""
    global _x11_dpy, _x11_hwnd, _x11_lib, _x11_root, _x11_ok
    if _x11_ok:
        return True
    try:
        from backends.linux import capture as cap
        if not cap._libx11 or not cap._grabber or not cap._grabber.display:
            return False
        hwnd = cap._grabber.get_window_handle(_process_name)
        if not hwnd:
            return False
        lib = cap._libx11
        dpy = cap._grabber.display
        lib.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        lib.XKeysymToKeycode.restype = ctypes.c_uint
        lib.XStringToKeysym.argtypes = [ctypes.c_char_p]
        lib.XStringToKeysym.restype = ctypes.c_ulong
        lib.XSendEvent.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_long, ctypes.POINTER(_XEvent)]
        lib.XSendEvent.restype = ctypes.c_int
        lib.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
        lib.XDefaultRootWindow.restype = ctypes.c_ulong
        lib.XFlush.argtypes = [ctypes.c_void_p]
        _x11_dpy, _x11_hwnd, _x11_lib, _x11_root = (
            dpy, hwnd, lib, lib.XDefaultRootWindow(dpy)
        )
        _x11_ok = True
        return True
    except Exception as e:
        print(f">> XSendEvent 初始化失败: {e}")
        return False


def _x11_send_key(hex_key_code: int, press: bool):
    """通过 XSendEvent 把按键直接送到游戏窗口（不改变 X/合成器焦点）。"""
    if not _x11_ensure():
        return False
    name = _SCAN_TO_KEYSYM.get(hex_key_code)
    if not name:
        return False
    keysym = _x11_lib.XStringToKeysym(name.encode())
    kc = _x11_lib.XKeysymToKeycode(_x11_dpy, keysym)
    ev = _XEvent()
    ev.xkey.type = 2 if press else 3  # KeyPress / KeyRelease
    ev.xkey.display = _x11_dpy
    ev.xkey.window = _x11_hwnd
    ev.xkey.root = _x11_root
    ev.xkey.subwindow = 0
    ev.xkey.time = 0
    ev.xkey.same_screen = 1
    ev.xkey.keycode = kc
    _x11_lib.XSendEvent(_x11_dpy, _x11_hwnd, 1, 0x0001, ctypes.byref(ev))
    _x11_lib.XFlush(_x11_dpy)
    return True


def _method_is_xsend() -> bool:
    if _input_method == "xsend":
        return True
    if _input_method == "uinput":
        return False
    # auto: 有 X 显示且能取到窗口 → xsend（不抢焦点）；否则 uinput
    if _input_method == "auto":
        if _x11_ok:
            return True
        try:
            from backends.linux import capture as cap
            if cap._libx11 and cap._grabber and cap._grabber.display:
                if cap._grabber.get_window_handle(_process_name):
                    return True
        except Exception:
            pass
        return False
    return False


def _ensure_uinput():
    global _ui, _ecodes
    if _ui is not None:
        return True
    try:
        from evdev import UInput, ecodes
    except ImportError as e:
        raise RuntimeError(
            "Linux 输入需要 python-evdev。请: pip install evdev 或 pacman -S python-evdev"
        ) from e

    key_codes = [getattr(ecodes, name) for name in _SCAN_TO_EVDEV.values()]
    # 训练默认：仅键盘，避免长期占用鼠标
    cap = {ecodes.EV_KEY: key_codes}
    _ui = UInput(cap, name="dqn-yakuza-kb", phys="dqn/yakuza/keyboard")
    _ecodes = ecodes
    atexit.register(shutdown)
    print(">> Linux uinput 虚拟键盘已就绪")
    return True


def _ensure_mouse(screen_size: Tuple[int, int] = (1920, 1080)):
    """按需创建带绝对坐标的虚拟鼠标（仅 resume 使用）。"""
    global _mouse, _ecodes
    if _mouse is not None:
        return True
    from evdev import UInput, ecodes, AbsInfo

    if _ecodes is None:
        _ensure_uinput()
    w, h = max(int(screen_size[0]), 64), max(int(screen_size[1]), 64)
    cap = {
        ecodes.EV_KEY: [ecodes.BTN_LEFT, ecodes.BTN_RIGHT, ecodes.BTN_MIDDLE],
        ecodes.EV_ABS: [
            (ecodes.ABS_X, AbsInfo(value=0, min=0, max=w - 1, fuzz=0, flat=0, resolution=0)),
            (ecodes.ABS_Y, AbsInfo(value=0, min=0, max=h - 1, fuzz=0, flat=0, resolution=0)),
        ],
    }
    _mouse = UInput(cap, name="dqn-yakuza-mouse", phys="dqn/yakuza/mouse")
    print(f">> Linux uinput 虚拟鼠标已就绪（{w}x{h}，仅 resume 点击用）")
    return True


def shutdown():
    global _ui, _mouse
    if _ui is not None:
        try:
            for code in list(_pressed):
                _ui.write(_ecodes.EV_KEY, code, 0)
            _ui.syn()
        except Exception:
            pass
        try:
            _ui.close()
        except Exception:
            pass
        _ui = None
        _pressed.clear()
    if _mouse is not None:
        try:
            _mouse.close()
        except Exception:
            pass
        _mouse = None


def _ev_code(hex_key_code: int) -> int:
    _ensure_uinput()
    name = _SCAN_TO_EVDEV.get(hex_key_code)
    if name is None:
        raise ValueError(f"未映射的扫描码: {hex_key_code:#x}")
    return getattr(_ecodes, name)


def press_key(hex_key_code):
    _assert_game_focused()
    if _method_is_xsend():
        _x11_send_key(hex_key_code, True)
        return
    code = _ev_code(hex_key_code)
    _ui.write(_ecodes.EV_KEY, code, 1)
    _ui.syn()
    _pressed.add(code)


def release_key(hex_key_code):
    if _method_is_xsend():
        _x11_send_key(hex_key_code, False)
        return
    code = _ev_code(hex_key_code)
    _ui.write(_ecodes.EV_KEY, code, 0)
    _ui.syn()
    _pressed.discard(code)


def tap_key(hex_key_code, duration=0.05):
    press_key(hex_key_code)
    time.sleep(duration)
    release_key(hex_key_code)


def hold_key(hex_key_code, duration=0.4):
    press_key(hex_key_code)
    time.sleep(duration)
    release_key(hex_key_code)


def weak_attack():
    tap_key(J, 0.05)


def strong_attack():
    tap_key(K, 0.05)


def start_forward(): press_key(W)
def stop_forward(): release_key(W)
def start_back(): press_key(S)
def stop_back(): release_key(S)
def start_left(): press_key(A)
def stop_left(): release_key(A)
def start_right(): press_key(D)
def stop_right(): release_key(D)
def start_defense(): press_key(I)
def stop_defense(): release_key(I)


def go_forward():
    hold_key(W, 0.2)


def go_back():
    hold_key(S, 0.2)


def go_left():
    hold_key(A, 0.2)


def go_right():
    hold_key(D, 0.2)


def dodge():
    tap_key(L, 0.1)


def defense():
    hold_key(I, 0.2)


def grab():
    tap_key(E, 0.05)


def striking_pose_start():
    press_key(O)


def striking_pose_cancel():
    release_key(O)


def extrem_heat_mode():
    hold_key(Q, 0.4)


def press_enter():
    tap_key(ENTER, 0.05)


def press_esc():
    hold_key(ESC, 0.3)


def virtual_click_at(
    x: int,
    y: int,
    screen_size: Tuple[int, int] = None,
    button: str = "left",
):
    """
    仅通过 **独立 uinput 虚拟鼠标设备** 发送绝对坐标点击。
    不调用 xdotool / 不 SetCursorPos / 不移动实体鼠标。
    实体指针位置由合成器与物理设备决定，与本设备分离。
    """
    if screen_size is None:
        screen_size = (max(int(x) + 1, 1920), max(int(y) + 1, 1080))
    _assert_game_focused()
    _ensure_mouse(screen_size)
    btn = {
        "left": _ecodes.BTN_LEFT,
        "right": _ecodes.BTN_RIGHT,
        "middle": _ecodes.BTN_MIDDLE,
    }.get(button, _ecodes.BTN_LEFT)
    w, h = screen_size
    x = int(max(0, min(w - 1, int(x))))
    y = int(max(0, min(h - 1, int(y))))
    _mouse.write(_ecodes.EV_ABS, _ecodes.ABS_X, x)
    _mouse.write(_ecodes.EV_ABS, _ecodes.ABS_Y, y)
    _mouse.syn()
    time.sleep(0.02)
    _mouse.write(_ecodes.EV_KEY, btn, 1)
    _mouse.syn()
    time.sleep(0.05)
    _mouse.write(_ecodes.EV_KEY, btn, 0)
    _mouse.syn()


# 兼容旧名
click_at = virtual_click_at


def resume_game(
    process_name: str = "Yakuza6.exe",
    press_escape: bool = True,
    virtual_click: bool = True,
) -> dict:
    """
    恢复游戏焦点（全程不占用真实鼠标/键盘硬件）：

      1. niri focus-window（合成器 API，无指针）
      2. 独立 uinput 虚拟鼠标在屏幕逻辑坐标上点一下（可选，助游戏捕获）
      3. 独立 uinput 虚拟键盘发送 ESC

    真实键鼠设备不被抓取、不被重定位。
    """
    from backends.linux.window_focus import focus_game_window

    result = focus_game_window(process_name)
    print(
        f">> resume 聚焦: ok={result['ok']} via={result['method']} {result['detail']}"
    )

    if result.get("ok"):
        # 聚焦成功后清掉守卫缓存，保证后续点击/ESC 通过校验
        invalidate_guard_cache()

    clicked = False
    if virtual_click and result.get("ok"):
        try:
            sw, sh = _guess_screen_size()
            # 聚焦后点「输出中心」：虚拟 ABS 设备坐标，不搬实体光标
            # 若已知窗口客户区尺寸，用相对中心映射到输出中心仍可接受
            cx, cy = sw // 2, sh // 2
            local = result.get("local_center")
            if local is not None:
                # 无全局窗位时仍用输出中心，避免误点其它屏
                pass
            virtual_click_at(cx, cy, screen_size=(sw, sh))
            clicked = True
            print(
                f">> resume 虚拟鼠标点击 (uinput-only) @({cx},{cy}) "
                f"screen={sw}x{sh} — 不移动真实指针"
            )
            time.sleep(0.08)
        except Exception as e:
            print(f">> resume 虚拟点击失败（可忽略）: {e}")

    if press_escape:
        time.sleep(0.05)
        # uinput 虚拟键盘，非真实键盘设备
        press_esc()
        print(">> resume 已发送 ESC（uinput 虚拟键盘）")

    result["clicked"] = clicked
    result["esc"] = bool(press_escape)
    result["real_mouse"] = False
    result["real_keyboard"] = False
    return result


def _guess_screen_size() -> Tuple[int, int]:
    import json
    import shutil
    import subprocess

    if shutil.which("niri"):
        try:
            p = subprocess.run(
                ["niri", "msg", "--json", "focused-output"],
                capture_output=True, text=True, timeout=2.0,
            )
            if p.returncode == 0 and p.stdout.strip():
                data = json.loads(p.stdout)
                logical = data.get("logical") or {}
                w = logical.get("width")
                h = logical.get("height")
                if w and h:
                    return int(w), int(h)
                mode = data.get("mode") or {}
                w = mode.get("width")
                h = mode.get("height")
                if w and h:
                    return int(w), int(h)
        except Exception:
            pass
        try:
            p = subprocess.run(
                ["niri", "msg", "--json", "outputs"],
                capture_output=True, text=True, timeout=2.0,
            )
            if p.returncode == 0 and p.stdout.strip():
                data = json.loads(p.stdout)
                # dict name -> output
                if isinstance(data, dict):
                    for _name, out in data.items():
                        logical = (out or {}).get("logical") or {}
                        if logical.get("width") and logical.get("height"):
                            return int(logical["width"]), int(logical["height"])
        except Exception:
            pass
    return 1920, 1080
