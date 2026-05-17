try:
    import torch
except ModuleNotFoundError:
    torch = None

# 设备配置
DEVICE = torch.device("cuda" if torch and torch.cuda.is_available() else "cpu") if torch else "cpu"

# 游戏相关配置
GAME_CONFIG = {
    'process_name': "Yakuza6.exe",
    'window_size': (1360, 768),
    'frame_size': (160, 160),  # 稍微缩小尺寸以提高速度，160x160 足够识别动作
    'frame_stack_size': 4,
    'num_actions': 18,         # 13 基础动作 + 5 连招槽位
    'max_combo_slots': 5
}

# 训练超参数
TRAINING_CONFIG = {
    'batch_size': 64,           # 针对 6GB 显存和大模型优化，减小 batch 以防 OOM
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_decay': 0.9999,
    'epsilon_min': 0.05,
    'learning_rate': 1e-4,     # 模型变大，降低学习率以确保收敛稳定性
    'weight_decay': 1e-4,
    'tau': 0.005,               # 软更新系数 (替代原先的 target_update_freq)
    'reward_scale': 0.1,        # 奖励缩放因子，防止奖励过大导致梯度爆炸
    'save_freq': 2000,
    'memory_capacity': 20000,
    'n_step': 3,                # N-step return，加速长期奖励回传
}

# 网络架构配置 (重型增强版/Pro版)
NETWORK_CONFIG = {
    'se_reduction': 4,          # 进一步增强 SE 模块的注意力精度
    'dueling_hidden_size': 1024, # 显著提升逻辑推理能力
    'dueling_dropout': 0.2,
    'use_recurrent': True,
    'recurrent_hidden_size': 512, # 显著增强长时记忆，识别敌人招式前摇
    'use_noisy': True,
    'noisy_std': 0.5,
    'num_quantiles': 51,        # QR-DQN的分位数数量 (Distributional RL)
    'use_fpn': True             # 启用特征金字塔(FPN)进行多尺度特征融合
}

MODEL_PROFILES = {
    'small': {
        'dueling_hidden_size': 256,
        'recurrent_hidden_size': 128,
        'num_quantiles': 21,
        'use_recurrent': False,
        'use_noisy': True,
        'use_fpn': False,
        'se_reduction': 8,
        'noisy_std': 0.4,
    },
    'medium': {
        'dueling_hidden_size': 512,
        'recurrent_hidden_size': 256,
        'num_quantiles': 35,
        'use_recurrent': True,
        'use_noisy': True,
        'use_fpn': True,
        'se_reduction': 8,
        'noisy_std': 0.5,
    },
    'large': NETWORK_CONFIG.copy(),
}

# 优先级经验回放配置
REPLAY_CONFIG = {
    'alpha': 0.6,
    'beta': 0.4
}

# 血条检测配置
HEALTH_CONFIG = {
    'boss_rgb_thresholds': {
        'red_range': (180, 255),
        'green_range': (0, 60),
        'blue_range': (0, 60)
    },
    'player_rgb_thresholds': {
        'red_range': (140, 255),
        'green_range': (0, 60),
        'blue_range': (0, 110)
    }
}

# 奖励函数配置
REWARD_CONFIG = {
    'damage_dealt_multiplier': 250.0,      # 攻击伤害是主考核指标
    'damage_taken_multiplier': 60.0,      # 受伤仍扣分，但不压过输出导向
    'time_penalty': -0.01,               # 降低时间惩罚，减少对积极性的干扰
    'combo_base_reward': 0.05,           # 连击只作为特别加分
    'combo_timeout': 3.0,
    'enable_combo_reward': True,
    'low_health_penalty': -2.0,          # 降低低血量惩罚
    'low_health_threshold': 0.15,
    'boss_defeat_reward': 30.0,          # 击杀作为结算特别加分，主要分数仍来自累计伤害
    'special_bonus_cap_without_damage': 0.05, # 未造成伤害时，其他加分的单步上限
    'special_bonus_damage_ratio': 0.15,  # 造成伤害时，其他加分最多约为伤害分的一小部分
    'special_bonus_step_cap': 1.0,       # 其他加分的绝对单步上限
    
    # 针对“怂”行为的惩罚 (如一直后退或不动)
    'passive_action_penalty_base': -0.01, # 降低基础消极动作惩罚
    'cowardice_threshold': 10,           # 增加容忍度
    
    # 抓取动作引导 (用户需求：防止长期刷闪避，惩罚不使用抓投)
    'grab_action_bonus': 0.0,           # 抓取本身不加分，只让命中伤害说话
    'grab_frequency_threshold': 100,     # 增加不抓取的容忍时间
    'grab_absence_penalty_base': -0.01,  # 大幅降低不抓取惩罚，防止其主导奖励
    'grab_combo_bonus': 0.02,            # 抓取连招只给极小特别加分
    
    # 引导奖励
    'attack_action_bonus': 0.005,        # 尝试攻击只给极小引导分
    'combo_variety_bonus': 0.05,         # 多样性只作为特别加分
    'defense_success_reward': 0.0        # 防御不再单独加分
}


# 文件路径配置
FILE_PATHS = {
    'checkpoint': 'dqn_optimized.pth',
    'locations': 'locations.json',
    'tensorboard_logs': './runs_optimized'
}
