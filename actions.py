# -*- coding: utf-8 -*-
"""兼容层：ActionController 自动绑定当前平台 InputBackend。"""
from core.actions import ActionController as _CoreActionController
from backends import load_platform

_bundle = None


def _default_input():
    global _bundle
    if _bundle is None:
        _bundle = load_platform()
    return _bundle.input


class ActionController(_CoreActionController):
    def __init__(self, input_backend=None):
        if input_backend is None:
            input_backend = _default_input()
        super().__init__(input_backend=input_backend)
