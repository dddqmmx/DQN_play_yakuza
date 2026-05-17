import time
import numpy as np
from grabscreen_pro import grab_screen

def benchmark_method(process_name, method='bitblt', iterations=100):
    print(f"Benchmarking {method}...")
    
    # Warm up
    for _ in range(5):
        grab_screen(process_name, method=method)
        
    start_time = time.time()
    count = 0
    null_count = 0
    for i in range(iterations):
        img = grab_screen(process_name, method=method)
        if img is not None:
            count += 1
            if count == 1:
                print(f"  {method} Image shape: {img.shape}")
        else:
            null_count += 1
    
    end_time = time.time()
    duration = end_time - start_time
    if count > 0:
        fps = count / duration
        print(f"  {method} Success: {count}, Null: {null_count}, Total time: {duration:.4f}s")
        print(f"  {method} Average FPS: {fps:.2f}")
    else:
        print(f"  {method} Failed to capture any frames. Null: {null_count}")

if __name__ == "__main__":
    from config import GAME_CONFIG
    process_name = GAME_CONFIG['process_name']
    
    benchmark_method(process_name, method='bitblt', iterations=100)
    benchmark_method(process_name, method='dxcam', iterations=100)
