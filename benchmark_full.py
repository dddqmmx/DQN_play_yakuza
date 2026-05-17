import time
import numpy as np
import cv2
from grabscreen_pro import grab_screen

def preprocess_frame(image, frame_size=(160, 160)):
    # BGR -> Gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize
    resized = cv2.resize(gray, frame_size, interpolation=cv2.INTER_AREA)
    # Normalize
    return (resized / 255.0).astype(np.float32)

def benchmark_full(process_name, method='bitblt', iterations=100):
    print(f"Benchmarking {method} WITH preprocessing...")
    start_time = time.time()
    
    count = 0
    for i in range(iterations):
        img = grab_screen(process_name, method=method)
        if img is not None:
            processed = preprocess_frame(img)
            count += 1
    
    end_time = time.time()
    duration = end_time - start_time
    if count > 0:
        fps = count / duration
        print(f"  {method} Average FPS (Capture + Preprocess): {fps:.2f}")
    else:
        print(f"  {method} Failed.")

if __name__ == "__main__":
    from config import GAME_CONFIG
    process_name = GAME_CONFIG['process_name']
    
    benchmark_full(process_name, method='bitblt')
    benchmark_full(process_name, method='mss')
