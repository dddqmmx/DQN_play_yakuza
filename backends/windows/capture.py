# -*- coding: utf-8 -*-
"""Windows 窗口捕获（原 grabscreen_pro 实现）。"""
import win32gui
import win32ui
import win32con
import win32process
import psutil
import numpy as np
import atexit

try:
    from mss import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

try:
    import dxcam
    HAS_DXCAM = True
except ImportError:
    HAS_DXCAM = False


class ScreenGrabberPro:
    def __init__(self):
        self.hwnd = None
        self.process_name = None
        self.sct = mss() if HAS_MSS else None
        self.camera = None
        self.last_rect = None

    def release(self):
        try:
            if self.camera is not None:
                if hasattr(self.camera, "is_capturing") and self.camera.is_capturing:
                    self.camera.stop()
                if hasattr(self.camera, "release"):
                    self.camera.release()
                self.camera = None
                print("DXCAM 资源已释放")
        except Exception as e:
            print(f"释放 DXCAM 资源时出错: {e}")

        try:
            if self.sct is not None:
                self.sct.close()
                self.sct = None
                print("MSS 资源已释放")
        except Exception as e:
            print(f"释放 MSS 资源时出错: {e}")

    def get_window_handle(self, process_name):
        if self.hwnd and self.process_name == process_name:
            if win32gui.IsWindow(self.hwnd):
                return self.hwnd

        def callback(hwnd, hwnds):
            if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
                try:
                    _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
                    if psutil.Process(found_pid).name() == process_name:
                        hwnds.append(hwnd)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return True

        hwnds = []
        win32gui.EnumWindows(callback, hwnds)
        if hwnds:
            self.hwnd = hwnds[0]
            self.process_name = process_name
            return self.hwnd
        return None

    def grab_bitblt(self, process_name, region=None):
        hwnd = self.get_window_handle(process_name)
        if not hwnd:
            return None

        if region:
            left, top, right, bot = region
            w, h = right - left, bot - top
            client_left, client_top = win32gui.ScreenToClient(hwnd, (left, top))
        else:
            left, top, right, bot = win32gui.GetClientRect(hwnd)
            w, h = right - left, bot - top
            client_left, client_top = 0, 0

        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()

        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
        saveDC.SelectObject(saveBitMap)

        saveDC.BitBlt((0, 0), (w, h), mfcDC, (client_left, client_top), win32con.SRCCOPY)

        signedIntsArray = saveBitMap.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype="uint8")
        img.shape = (h, w, 4)

        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)

        return img[:, :, :3].copy()

    def grab_mss_optimized(self, process_name, region=None):
        if not HAS_MSS:
            return None
        hwnd = self.get_window_handle(process_name)
        if not hwnd:
            return None

        if region:
            monitor = {
                "top": region[1],
                "left": region[0],
                "width": region[2] - region[0],
                "height": region[3] - region[1],
            }
        else:
            rect = win32gui.GetClientRect(hwnd)
            left, top = win32gui.ClientToScreen(hwnd, (rect[0], rect[1]))
            w, h = rect[2] - rect[0], rect[3] - rect[1]
            monitor = {"top": top, "left": left, "width": w, "height": h}

        screenshot = self.sct.grab(monitor)
        img = np.frombuffer(screenshot.rgb, dtype="uint8").reshape(
            screenshot.height, screenshot.width, 3
        )
        return img[:, :, ::-1].copy()

    def grab_dxcam(self, process_name, region=None):
        if not HAS_DXCAM:
            return self.grab_mss_optimized(process_name, region)

        hwnd = self.get_window_handle(process_name)
        if not hwnd:
            return None

        if region:
            dxcam_region = region
        else:
            rect = win32gui.GetClientRect(hwnd)
            left, top = win32gui.ClientToScreen(hwnd, (rect[0], rect[1]))
            w, h = rect[2] - rect[0], rect[3] - rect[1]
            dxcam_region = (left, top, left + w, top + h)

        try:
            if self.camera is None:
                self.camera = dxcam.create(output_color="BGR")

            if not self.camera.is_capturing:
                self.camera.start(region=dxcam_region, target_fps=60)

            frame = self.camera.get_latest_frame()
            if frame is None:
                return self.grab_mss_optimized(process_name, region)
            return frame
        except Exception as e:
            print(f"DXCAM 运行时错误: {e}，自动切换到 MSS 模式")
            return self.grab_mss_optimized(process_name, region)


_grabber = ScreenGrabberPro()
atexit.register(_grabber.release)


def grab_screen(process_name, method="bitblt", region=None):
    if method == "bitblt":
        return _grabber.grab_bitblt(process_name, region)
    if method == "mss":
        return _grabber.grab_mss_optimized(process_name, region)
    if method == "dxcam":
        return _grabber.grab_dxcam(process_name, region)
    return None
