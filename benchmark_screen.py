import time
import numpy as np
from grabscreen import grab_screen_by_process_name
import cv2

def benchmark(process_name, iterations=100):
    print(f"Benchmarking screen capture for {process_name}...")
    start_time = time.time()
    
    for i in range(iterations):
        img = grab_screen_by_process_name(process_name)
        if i == 0:
            print(f"Image shape: {img.shape}")
    
    end_time = time.time()
    duration = end_time - start_time
    fps = iterations / duration
    print(f"Total time for {iterations} frames: {duration:.4f}s")
    print(f"Average FPS: {fps:.2f}")

if __name__ == "__main__":
    # You might need to change this to your game process name
    # Looking at main.py, it uses GAME_CONFIG['process_name']
    # I'll check config.py for the process name
    try:
        from config import GAME_CONFIG
        process_name = GAME_CONFIG['process_name']
        benchmark(process_name)
    except Exception as e:
        print(f"Error: {e}")
