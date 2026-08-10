# -*- coding: utf-8 -*-
"""Windows 按键：仅 SendInput 虚拟注入，resume 不移动实体光标。"""
import ctypes
import time

SendInput = ctypes.windll.user32.SendInput

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

PUL = ctypes.POINTER(ctypes.c_ulong)


class KeyBdInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL),
    ]


class HardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_short),
        ("wParamH", ctypes.c_ushort),
    ]


class MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL),
    ]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput), ("mi", MouseInput), ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("ii", Input_I)]


def press_key(hex_key_code):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hex_key_code, 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def release_key(hex_key_code):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hex_key_code, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


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


def resume_game(process_name: str = "Yakuza6.exe", press_escape: bool = True) -> dict:
    """
    聚焦窗口 + ESC。
    故意不调用 SetCursorPos / mouse_event，避免占用真实鼠标。
    """
    try:
        import win32gui
        import win32con
        import win32process
        import psutil
    except ImportError as e:
        press_esc()
        return {
            "ok": False,
            "method": "esc-only",
            "detail": str(e),
            "esc": press_escape,
            "real_mouse": False,
        }

    hwnds = []

    def _cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return True
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            if psutil.Process(pid).name().lower() == process_name.lower():
                hwnds.append(hwnd)
        except Exception:
            pass
        return True

    win32gui.EnumWindows(_cb, None)
    if not hwnds:
        press_esc()
        return {
            "ok": False,
            "method": "esc-only",
            "detail": "window not found",
            "esc": press_escape,
            "real_mouse": False,
            "clicked": False,
        }

    hwnd = hwnds[0]
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
    except Exception as e:
        print(f">> SetForegroundWindow: {e}")

    # 不移动真实光标、不发真实鼠标消息
    if press_escape:
        time.sleep(0.05)
        press_esc()

    return {
        "ok": True,
        "method": "win32-focus-esc",
        "clicked": False,
        "esc": press_escape,
        "real_mouse": False,
        "detail": f"hwnd={hwnd} (no real cursor move)",
    }
