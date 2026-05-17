import win32gui
import win32ui
import win32con
import win32process
import psutil
import numpy as np
import time
import atexit

# Try to import mss and dxcam
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
        # Keep DXCAM lazy: creating it during import can break some game
        # renderers or fail with desktop-duplication permission errors.
        return
        
        # 预先初始化 dxcam，防止与后续 PyTorch 的 CUDA 上下文冲突
        if HAS_DXCAM:
            try:
                self.camera = dxcam.create(output_color="BGR")
                print("DXCAM 预初始化成功")
            except Exception as e:
                print(f"DXCAM 预初始化失败: {e}，将尝试在运行时初始化或使用回退方案")

    def release(self):
        """释放资源，特别是 DXCAM 和 MSS"""
        try:
            if self.camera is not None:
                if hasattr(self.camera, 'is_capturing') and self.camera.is_capturing:
                    self.camera.stop()
                # dxcam 并没有显式的 release() 方法在某些版本中，
                # 但将其设为 None 有助于垃圾回收。
                # 如果有 release 属性则调用它
                if hasattr(self.camera, 'release'):
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
        """
        方案 1: Win32 BitBlt (不需要额外库，速度快)
        注意: 窗口不能最小化。如果窗口被遮挡，可能会截到上层窗口。
        """
        hwnd = self.get_window_handle(process_name)
        if not hwnd: return None

        if region:
            # region: (left, top, right, bottom) in screen coordinates
            left, top, right, bot = region
            w, h = right - left, bot - top
            # Convert screen coordinates to client coordinates for BitBlt
            client_left, client_top = win32gui.ScreenToClient(hwnd, (left, top))
        else:
            left, top, right, bot = win32gui.GetClientRect(hwnd)
            w, h = right - left, bot - top
            client_left, client_top = 0, 0
        
        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()
        
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
        saveDC.SelectObject(saveBitMap)
        
        saveDC.BitBlt((0, 0), (w, h), mfcDC, (client_left, client_top), win32con.SRCCOPY)
        
        signedIntsArray = saveBitMap.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype='uint8')
        img.shape = (h, w, 4)
        
        # Cleanup
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)
        
        return img[:, :, :3].copy() # 返回 BGR

    def grab_mss_optimized(self, process_name, region=None):
        """
        方案 2: MSS 优化版
        直接从 buffer 读取，减少内存拷贝。
        """
        if not HAS_MSS: return None
        hwnd = self.get_window_handle(process_name)
        if not hwnd: return None

        if region:
            monitor = {"top": region[1], "left": region[0], "width": region[2]-region[0], "height": region[3]-region[1]}
        else:
            rect = win32gui.GetClientRect(hwnd)
            left, top = win32gui.ClientToScreen(hwnd, (rect[0], rect[1]))
            w, h = rect[2]-rect[0], rect[3]-rect[1]
            monitor = {"top": top, "left": left, "width": w, "height": h}

        screenshot = self.sct.grab(monitor)
        
        img = np.frombuffer(screenshot.rgb, dtype='uint8').reshape(screenshot.height, screenshot.width, 3)
        return img[:, :, ::-1].copy() # RGB to BGR

    def grab_dxcam(self, process_name, region=None):
        """
        方案 3: DXCAM (后台线程连续截屏模式)
        """
        if not HAS_DXCAM: 
            return self.grab_mss_optimized(process_name, region)
            
        hwnd = self.get_window_handle(process_name)
        if not hwnd: return None

        # Determine region
        if region:
            dxcam_region = region
        else:
            rect = win32gui.GetClientRect(hwnd)
            left, top = win32gui.ClientToScreen(hwnd, (rect[0], rect[1]))
            w, h = rect[2]-rect[0], rect[3]-rect[1]
            dxcam_region = (left, top, left+w, top+h)

        try:
            if self.camera is None:
                self.camera = dxcam.create(output_color="BGR")

            if not self.camera.is_capturing:
                self.camera.start(region=dxcam_region, target_fps=60)
            
            frame = self.camera.get_latest_frame()
            if frame is None:
                # 如果 dxcam 获取失败，退而求其次使用 mss
                return self.grab_mss_optimized(process_name, region)
            return frame
        except Exception as e:
            print(f"DXCAM 运行时错误: {e}，自动切换到 MSS 模式")
            return self.grab_mss_optimized(process_name, region)

# 统一导出接口
_grabber = ScreenGrabberPro()
atexit.register(_grabber.release)

def grab_screen(process_name, method='bitblt', region=None):
    if method == 'bitblt':
        return _grabber.grab_bitblt(process_name, region)
    elif method == 'mss':
        return _grabber.grab_mss_optimized(process_name, region)
    elif method == 'dxcam':
        return _grabber.grab_dxcam(process_name, region)
    return None
