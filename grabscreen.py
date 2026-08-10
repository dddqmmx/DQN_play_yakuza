# -*- coding: utf-8 -*-
"""兼容层：按平台委托捕获。"""
from backends import default_capture_method, load_platform

_bundle = None


def _capture():
    global _bundle
    if _bundle is None:
        _bundle = load_platform()
    return _bundle.capture


def grab_screen_by_process_name(process_name, region=None):
    return _capture().grab(process_name, region=region)


def grab_screen(process_name, method=None, region=None):
    method = method or default_capture_method()
    # 允许临时换 method：重新走底层
    from backends import is_windows
    if is_windows():
        from backends.windows.capture import grab_screen as _gs
    else:
        from backends.linux.capture import grab_screen as _gs
    return _gs(process_name, method=method, region=region)
