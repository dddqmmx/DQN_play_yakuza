import ctypes
import time
from ctypes import wintypes


PROCESS_SUSPEND_RESUME = 0x0800
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
TH32CS_SNAPPROCESS = 0x00000002
TH32CS_SNAPTHREAD = 0x00000004
THREAD_SUSPEND_RESUME = 0x0002
INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.POINTER(wintypes.ULONG)),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", wintypes.WCHAR * 260),
    ]


class THREADENTRY32(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ThreadID", wintypes.DWORD),
        ("th32OwnerProcessID", wintypes.DWORD),
        ("tpBasePri", wintypes.LONG),
        ("tpDeltaPri", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
    ]


kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
ntdll = ctypes.WinDLL("ntdll", use_last_error=True)

kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
kernel32.Process32FirstW.restype = wintypes.BOOL
kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
kernel32.Process32NextW.restype = wintypes.BOOL
kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.Thread32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(THREADENTRY32)]
kernel32.Thread32First.restype = wintypes.BOOL
kernel32.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(THREADENTRY32)]
kernel32.Thread32Next.restype = wintypes.BOOL
kernel32.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.OpenThread.restype = wintypes.HANDLE
kernel32.SuspendThread.argtypes = [wintypes.HANDLE]
kernel32.SuspendThread.restype = wintypes.DWORD
kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
kernel32.ResumeThread.restype = wintypes.DWORD
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.CloseHandle.restype = wintypes.BOOL
ntdll.NtSuspendProcess.argtypes = [wintypes.HANDLE]
ntdll.NtSuspendProcess.restype = wintypes.LONG
ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
ntdll.NtResumeProcess.restype = wintypes.LONG


def find_process_id(process_name):
    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snapshot == INVALID_HANDLE_VALUE:
        raise ctypes.WinError(ctypes.get_last_error())

    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        ok = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        target = process_name.lower()
        while ok:
            if entry.szExeFile.lower() == target:
                return int(entry.th32ProcessID)
            ok = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return None


def iter_process_thread_ids(pid):
    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
    if snapshot == INVALID_HANDLE_VALUE:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        entry = THREADENTRY32()
        entry.dwSize = ctypes.sizeof(THREADENTRY32)
        ok = kernel32.Thread32First(snapshot, ctypes.byref(entry))
        while ok:
            if int(entry.th32OwnerProcessID) == int(pid):
                yield int(entry.th32ThreadID)
            ok = kernel32.Thread32Next(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)


class GameProcessFreezer:
    def __init__(self, process_name, target_fps=20.0, strict_passes=2):
        self.process_name = process_name
        self.frame_seconds = 1.0 / max(float(target_fps), 1.0)
        self.strict_passes = max(int(strict_passes), 1)
        self.pid = None
        self.handle = None
        self.is_suspended = False
        self.thread_handles = []

    def open(self):
        pid = find_process_id(self.process_name)
        if pid is None:
            raise RuntimeError(f"找不到游戏进程: {self.process_name}")
        if self.pid == pid and self.handle:
            return
        self.close(resume=True)
        access = PROCESS_SUSPEND_RESUME | PROCESS_QUERY_LIMITED_INFORMATION
        handle = kernel32.OpenProcess(access, False, pid)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self.pid = pid
        self.handle = handle

    def suspend(self):
        self.open()
        if self.is_suspended:
            return
        status = ntdll.NtSuspendProcess(self.handle)
        if status != 0:
            raise RuntimeError(f"NtSuspendProcess failed: {status:#x}")
        self._suspend_threads_strict()
        self.is_suspended = True

    def resume(self):
        if not self.handle or not self.is_suspended:
            return
        self._resume_strict_threads()
        status = ntdll.NtResumeProcess(self.handle)
        if status != 0:
            raise RuntimeError(f"NtResumeProcess failed: {status:#x}")
        self.is_suspended = False

    def _suspend_threads_strict(self):
        self._resume_strict_threads()
        suspended_tids = set()
        for _ in range(self.strict_passes):
            for tid in iter_process_thread_ids(self.pid):
                if tid in suspended_tids:
                    continue
                handle = kernel32.OpenThread(THREAD_SUSPEND_RESUME, False, tid)
                if not handle:
                    continue
                previous_count = kernel32.SuspendThread(handle)
                if previous_count == 0xFFFFFFFF:
                    kernel32.CloseHandle(handle)
                    continue
                suspended_tids.add(tid)
                self.thread_handles.append(handle)
            time.sleep(0.001)

    def _resume_strict_threads(self):
        while self.thread_handles:
            handle = self.thread_handles.pop()
            try:
                kernel32.ResumeThread(handle)
            finally:
                kernel32.CloseHandle(handle)

    def run_one_frame(self):
        self.resume()
        time.sleep(self.frame_seconds)
        self.suspend()

    def close(self, resume=True):
        if self.handle:
            if resume and self.is_suspended:
                try:
                    self.resume()
                except Exception:
                    pass
            self._resume_strict_threads()
            kernel32.CloseHandle(self.handle)
        self.pid = None
        self.handle = None
        self.is_suspended = False
