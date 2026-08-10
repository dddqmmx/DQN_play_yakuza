# -*- coding: utf-8 -*-
"""兼容层：按平台导出按键 API（模块级函数，供旧代码 import）。"""
from backends import is_windows, load_platform

if is_windows():
    from backends.windows.input_keys import *  # noqa: F401,F403
else:
    from backends.linux.input_keys import *  # noqa: F401,F403
