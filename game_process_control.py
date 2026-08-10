# -*- coding: utf-8 -*-
"""兼容层：进程冻结。"""
from backends import is_windows

if is_windows():
    from backends.windows.process import (  # noqa: F401
        GameProcessFreezer,
        find_process_id,
        iter_process_thread_ids,
    )
else:
    from backends.linux.process import (  # noqa: F401
        GameProcessFreezer,
        find_process_ids,
    )

    def find_process_id(process_name):
        pids = find_process_ids(process_name)
        return pids[0] if pids else None
