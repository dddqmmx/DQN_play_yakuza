# -*- coding: utf-8 -*-
"""
DirectKeys - 游戏按键控制模块
使用底层Windows API发送按键事件
"""

import ctypes
import time

SendInput = ctypes.windll.user32.SendInput

# 按键扫描码定义
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

# C结构体定义
PUL = ctypes.POINTER(ctypes.c_ulong)

class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# 底层按键函数
def press_key(hex_key_code):
    """按下按键"""
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hex_key_code, 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def release_key(hex_key_code):
    """释放按键"""
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hex_key_code, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def tap_key(hex_key_code, duration=0.05):
    """轻击按键（按下后立即释放）"""
    press_key(hex_key_code)
    time.sleep(duration)
    release_key(hex_key_code)

def hold_key(hex_key_code, duration=0.4):
    """持续按住按键"""
    press_key(hex_key_code)
    time.sleep(duration)
    release_key(hex_key_code)

# 游戏动作函数
def weak_attack():
    """轻攻击"""
    tap_key(J, 0.05)

def strong_attack():
    """重攻击"""
    tap_key(K, 0.05)

# 持续性动作控制 (用于长按)
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
    """前进 (旧接口保持兼容，内部改为短时长)"""
    hold_key(W, 0.2)

def go_back():
    """后退"""
    hold_key(S, 0.2)

def go_left():
    """左移"""
    hold_key(A, 0.2)

def go_right():
    """右移"""
    hold_key(D, 0.2)

def dodge():
    """闪避"""
    tap_key(L, 0.1)

def defense():
    """防御"""
    hold_key(I, 0.2)

def grab():
    """抓取"""
    tap_key(E, 0.05)

def striking_pose_start():
    """开始蓄力姿势"""
    press_key(O)

def striking_pose_cancel():
    """取消蓄力姿势"""
    release_key(O)

def extrem_heat_mode():
    """极热模式"""
    hold_key(Q, 0.4)

def press_enter():
    """按回车键"""
    tap_key(ENTER, 0.05)

def press_esc():
    """按ESC键"""
    hold_key(ESC, 0.3)