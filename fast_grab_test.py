import win32gui
import win32ui
import win32con
import win32process
import psutil
import numpy as np
import cv2

class FastScreenGrabber:
    def __init__(self):
        self.hwnd = None
        self.process_name = None

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

    def grab_bitblt(self, process_name):
        hwnd = self.get_window_handle(process_name)
        if not hwnd:
            return None

        # Get the client area size
        left, top, right, bot = win32gui.GetClientRect(hwnd)
        w = right - left
        h = bot - top

        # Get the device context
        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()

        # Create a bitmap object
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)

        saveDC.SelectObject(saveBitMap)

        # BitBlt from the window DC to the memory DC
        # Note: If the window is obscured, this might capture what's on top of it
        # unless you use PrintWindow with PW_RENDERFULLCONTENT (which is slower)
        saveDC.BitBlt((0, 0), (w, h), mfcDC, (0, 0), win32con.SRCCOPY)

        # Convert to numpy array
        signedIntsArray = saveBitMap.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype='uint8')
        img.shape = (h, w, 4)

        # Cleanup
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)

        # Return BGR (remove alpha)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

# For testing
if __name__ == "__main__":
    import time
    grabber = FastScreenGrabber()
    pname = "Yakuza6.exe" # Change to your process
    
    print("Starting BitBlt benchmark...")
    start = time.time()
    for i in range(100):
        frame = grabber.grab_bitblt(pname)
        if frame is None:
            print("Window not found!")
            break
    end = time.time()
    if frame is not None:
        print(f"BitBlt average FPS: {100/(end-start):.2f}")
        print(f"Frame shape: {frame.shape}")
