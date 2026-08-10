# -*- coding: utf-8 -*-
"""兼容层：转发到 platform 包。"""
from backends import (  # noqa: F401
    default_capture_method,
    is_linux,
    is_wayland,
    is_windows,
    load_platform,
)
