from grabscreen_pro import grab_screen

def grab_screen_by_process_name(process_name, region=None):
    # 异步训练模式下，dxcam 容易产生冲突，改用 mss 方案
    return grab_screen(process_name, method='mss', region=region)
