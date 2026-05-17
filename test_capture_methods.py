import cv2
import numpy as np
import os
from grabscreen_pro import grab_screen
from config import GAME_CONFIG

def test_capture_methods():
    methods = ['bitblt', 'mss', 'dxcam']
    process_name = GAME_CONFIG['process_name']
    
    output_dir = "capture_test"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for method in methods:
        print(f"Testing method: {method}")
        try:
            screen = grab_screen(process_name, method=method)
            if screen is not None:
                print(f"  Success! Shape: {screen.shape}")
                print(f"  Mean brightness: {np.mean(screen)}")
                filename = os.path.join(output_dir, f"capture_{method}.png")
                cv2.imwrite(filename, screen)
                print(f"  Saved to {filename}")
            else:
                print(f"  Failed to capture using {method}")
        except Exception as e:
            print(f"  Error with {method}: {e}")

if __name__ == "__main__":
    test_capture_methods()
