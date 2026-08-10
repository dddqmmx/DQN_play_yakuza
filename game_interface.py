# -*- coding: utf-8 -*-
"""兼容层：GameInterface 使用当前平台 CaptureBackend。"""
from core.observation import GameInterface, GameObservation  # noqa: F401
from backends import load_platform

_bundle = None


def _default_capture():
    global _bundle
    if _bundle is None:
        _bundle = load_platform()
    return _bundle.capture


class GameInterface(GameObservation):
    def __init__(self, process_name=None, capture=None):
        if capture is None:
            capture = _default_capture()
        super().__init__(capture=capture, process_name=process_name)
