import cv2
import numpy as np
import json
import os
from game_interface import GameInterface
from config import HEALTH_CONFIG, GAME_CONFIG

def test_detection():
    print("Initializing GameInterface...")
    game = GameInterface()
    
    print("Capturing screen...")
    raw_screen = game.get_raw_screen()
    
    if raw_screen is None:
        print("Failed to capture screen. Is the game running?")
        return
    
    print(f"Captured screen size: {raw_screen.shape[1]}x{raw_screen.shape[0]}")
    print(f"Target window size from config: {GAME_CONFIG['window_size'][0]}x{GAME_CONFIG['window_size'][1]}")

    # Create output directory
    output_dir = "debug_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save full screen
    cv2.imwrite(os.path.join(output_dir, "full_screen.png"), raw_screen)
    print(f"Saved full screen to {output_dir}/full_screen.png")

    # Get regions
    player_region, boss_region = game.get_health_regions(raw_screen)
    
    # Process Player HP
    if player_region is not None and player_region.size > 0:
        print(f"Player region coordinates: {game.locations['player']}")
        print(f"Player region shape: {player_region.shape}")
        print(f"Player region mean BGR: {np.mean(player_region, axis=(0,1))}")
        cv2.imwrite(os.path.join(output_dir, "player_region.png"), player_region)
        rgb_player = cv2.cvtColor(player_region, cv2.COLOR_BGR2RGB)
        player_mask = cv2.inRange(rgb_player, game.player_low, game.player_high)
        print(f"Player mask white pixels: {np.sum(player_mask > 0)}")
        cv2.imwrite(os.path.join(output_dir, "player_mask.png"), player_mask)
        hp = game.detect_health(player_region, is_boss=False)
        print(f"Player HP detected: {hp:.3f}")
    else:
        print("Player region is empty or invalid.")

    # Process Boss HP
    if boss_region is not None and boss_region.size > 0:
        print(f"Boss region mean BGR: {np.mean(boss_region, axis=(0,1))}")
        cv2.imwrite(os.path.join(output_dir, "boss_region.png"), boss_region)
        rgb_boss = cv2.cvtColor(boss_region, cv2.COLOR_BGR2RGB)
        boss_mask = cv2.inRange(rgb_boss, game.boss_low, game.boss_high)
        print(f"Boss mask white pixels: {np.sum(boss_mask > 0)}")
        cv2.imwrite(os.path.join(output_dir, "boss_mask.png"), boss_mask)
        hp = game.detect_health(boss_region, is_boss=True)
        print(f"Boss HP detected: {hp:.3f}")
    else:
        print("Boss region is empty or invalid.")

    # Save a combined debug image with rectangles
    debug_img = raw_screen.copy()
    with game.location_lock:
        p1, p2 = game.locations['player']
        b1, b2 = game.locations['boss']
        cv2.rectangle(debug_img, tuple(p1), tuple(p2), (0, 255, 0), 2)
        cv2.rectangle(debug_img, tuple(b1), tuple(b2), (0, 0, 255), 2)
    cv2.imwrite(os.path.join(output_dir, "debug_locations.png"), debug_img)
    print(f"Saved debug view to {output_dir}/debug_locations.png")

if __name__ == "__main__":
    test_detection()
