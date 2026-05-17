import time
import numpy as np
from grabscreen_pro import grab_screen

def benchmark_method(process_name, method='bitblt', iterations=100):
    print(f"Benchmarking {method}...")
    start_time = time.time()
    
    count = 0
    for i in range(iterations):
        img = grab_screen(process_name, method=method)
        if img is not None:
            count += 1
            if i == 0:
                print(f"  {method} Image shape: {img.shape}")
    
    end_time = time.time()
    duration = end_time - start_time
    if count > 0:
        fps = count / duration
        print(f"  {method} Total time for {count} frames: {duration:.4f}s")
        print(f"  {method} Average FPS: {fps:.2f}")
    else:
        print(f"  {method} Failed to capture any frames. (Check if lib is installed or window is open)")

if __name__ == "__main__":
    from config import GAME_CONFIG
    process_name = GAME_CONFIG['process_name']
    
    # Check BitBlt (Fastest built-in)
    benchmark_method(process_name, method='bitblt')
    
    # Check MSS (Current)
    benchmark_method(process_name, method='mss')
    
    # Check DXCAM (Needs install)
    benchmark_method(process_name, method='dxcam')
