import os as _os

try:
    import torch
except ModuleNotFoundError:
    torch = None

# 设备配置
DEVICE = torch.device("cuda" if torch and torch.cuda.is_available() else "cpu") if torch else "cpu"

# 游戏相关配置
# Linux/Proton 下进程名通常仍是 Yakuza6.exe；可用 DQN_PROCESS_NAME 覆盖
GAME_CONFIG = {
    'process_name': _os.environ.get("DQN_PROCESS_NAME", "Yakuza6.exe"),
    'window_size': (1360, 768),
    'frame_size': (160, 160),  # 稍微缩小尺寸以提高速度，160x160 足够识别动作
    'frame_stack_size': 4,
    # 只有 13 个原子动作。预设连招槽位（原先的 13~17）已删除：
    # 连招现在由模型自己规划成动作序列，不再是"从 combos.json 里挑一个录好的宏"。
    'num_actions': 13,
    # 计划执行途中挨打超过这个血量比例，立刻丢弃剩余计划重新规划
    'plan_interrupt_damage': 0.01,
    # 连续这么多帧都判"血量回升"，就认定失真的是基线而不是当前帧，强制重新同步。
    # 守卫只拦上升、不拦下降，是个单向棘轮：偏低的异常读数会被当成"打出伤害"
    # 照单全收并成为新基线，此后每一帧正常读数都比它高、永远被拒。
    # 实测一次死锁 176 帧、期间一条样本都不产生（见 core/game_loop.py 主循环）。
    'health_regen_resync_after': 3,
    # 血条自动定位：默认开；DQN_AUTO_LOCATE=0 可关
    'auto_locate': _os.environ.get("DQN_AUTO_LOCATE", "1").lower() not in ("0", "false", "no"),
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
    # 每个环境步最多做几次梯度更新（按本次会话计）。在线 RL 里更新数远超环境步就是
    # 在小 buffer 上反复过拟合，还会占满 GPU 让决策排队。
    # 0.25 = Rainbow 的标准比值（每 4 个环境步一次更新）。在这台 6GB 卡上 CTM 一次
    # 更新约 103ms，环境约 9.5 步/s，0.25 对应 ~25% GPU 占用，决策能维持在 ~15ms；
    # 设成 1.0 就又把 GPU 占满了。设 0 表示不限速。
    'replay_ratio': 0.25,
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

# ---------------------------------------------------------------- CTM 计划模型
# 借鉴 Continuous Thought Machines (arXiv:2505.05522)：
#   - 内部 tick 递归：一次决策里"想" iterations 次，而不是一次前向出答案
#   - 神经元级模型（NLM）：每个神经元用自己的小 MLP 处理自己的前激活历史
#   - 同步性作为表征：拿神经元两两之间的时序同步度去查询画面 / 输出动作
#   - certainty：每个 tick 自评"想清楚没有"，用来挑 tick、也用来决定提交多少步
CTM_CONFIG = {
    # --- CTM 核心 ---
    'iterations': 8,              # 内部思考 tick 数 T
    'd_model': 512,               # 内部神经元数 D
    'd_input': 256,               # 注意力/视觉 token 维度
    'heads': 4,
    'n_synch_out': 256,           # 输出同步性用的神经元数
    'n_synch_action': 128,        # 查询画面用的同步性神经元数
    'synapse_depth': 4,           # >1 走 U-Net synapse
    'memory_length': 12,          # NLM 看多长的前激活历史 M
    'memory_hidden_dims': 16,
    'deep_nlms': True,
    'do_layernorm_nlm': False,    # 上游从不开；开了动力学会变得很怪
    'neuron_select_type': 'random-pairing',
    'n_random_pairing_self': 8,
    'dropout': 0.0,
    'token_grid': 10,             # 视觉 token 网格 10x10 = 100 个 token

    # --- 计划头 ---
    'plan_length': 6,             # 一次输出多长的动作序列 L
    'num_quantiles': 21,          # QR-DQN 分位数
    'use_noisy': True,
    'noisy_std': 0.4,
    # 每个槽位以此概率替换成随机动作。NoisyNets 的 sigma 是可学习的，训练久了会自己
    # 收缩到几乎没有探索；这条是兜底下限，保证任何时候都还在试别的招式。
    'explore_epsilon': 0.08,

    # --- 计划训练 ---
    # batch 比 TRAINING_CONFIG 小一半：CTM 一次更新要跑 3 遍前向、每遍 iterations 个
    # 内部 tick，batch 64 时实测单步 238ms，决策会被挤到 250ms（离线基准才 8ms）。
    'batch_size': 32,
    'plan_loss_weight': 0.5,           # 槽位一致性 loss 权重
    'plan_curriculum_lookahead': 2,    # 只训练"已对齐前缀 + 这么多步"的槽位；<0 关闭
    'plan_memory_capacity': 4096,      # 1 步 transition 环形 buffer（喂一致性 loss）
    'plan_batch_size': 16,

    # --- 计划回报锚（让槽位 j 对真实的 j 步回报负责）---
    # 纯自举的一致性 loss 只保证"槽位之间自洽"，没有任何奖励信号进得去，实测
    # 6145 步后槽位 1~3 在真实画面上的决断度只有 0.116/0.111/0.083，低于"毫无
    # 偏好"的 0.145 —— 被回归成了一摊糊。所以额外拿**真的连续执行掉的那几步**
    # 的折扣回报去锚：槽位 j 的目标 = 从第 j 步起的实际回报 + 尾部自举。
    #
    # 只有 commit >= 2 才会产生这种样本，所以它天然是个自举课程：
    # 槽位 0 稳 -> commit 2 -> 槽位 1 拿到真实回报 -> commit 3 -> ……逐格向外长。
    # 训练早期这个 buffer 是空的，loss 为 0，不影响原有两项。
    'plan_run_capacity': 2048,
    'plan_run_batch_size': 8,
    'plan_return_weight': 0.5,         # 回报锚 loss 的权重

    # --- 提交长度（计划的决断程度 -> 提交几步）---
    # 阈值**逐槽位**作用在 `CTMPlannerNet.plan_slot_confidence` 的相对动作间隔上，
    # 尺度无关。单槽位标定（13 个动作）：无偏好 ≈0.145、最优高出 2σ ≈0.199、
    # 3σ ≈0.298、10σ ≈0.719。取 0.20 表示"这一步确实有偏好了才往下连"。
    # commit = 从槽位 0 起连续过线的槽位个数，详见 CTMPlannerNet.commit_length。
    #
    # 原来还有个 `commit_confidence_hi: 0.60` 配合线性映射用，已删除：0.60 对应
    # 单槽位约 7σ 的动作间隔，实际训练根本到不了，配上 round() 后连 2 步都要
    # 0.267，结果 commit 恒为 1、整条计划只有第 0 格被执行。
    'commit_confidence_lo': 0.20,  # 逐槽位：低于此就在这一步截断，不再往下连
}

CTM_PROFILES = {
    'small': {
        **CTM_CONFIG,
        'iterations': 4,
        'd_model': 256,
        'd_input': 128,
        'heads': 4,
        'n_synch_out': 128,
        'n_synch_action': 64,
        'synapse_depth': 2,
        'memory_length': 8,
        'plan_length': 4,
        'token_grid': 10,
    },
    'medium': {
        **CTM_CONFIG,
        'iterations': 6,
        'd_model': 384,
        'd_input': 192,
        'n_synch_out': 192,
        'n_synch_action': 96,
        'synapse_depth': 3,
        'memory_length': 10,
        'plan_length': 5,
    },
    'large': CTM_CONFIG.copy(),
}

# 优先级经验回放配置
REPLAY_CONFIG = {
    'alpha': 0.6,
    'beta': 0.4
}

# 血条检测配置
# 颜色/几何门限都在 core/health_bar.py（都是对着实际像素量出来的，不建议乱调）。
# 这里只留跨模块用得到的开关。
HEALTH_CONFIG = {
    # 连续读到 0 多少帧才认定死亡/击杀，防单帧误判触发回合结束
    'zero_confirm_frames': int(_os.environ.get("DQN_ZERO_CONFIRM", "2")),
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
    'health_regen_tolerance': 0.02,      # 血量不应回升；超过该容差视为垃圾帧并跳过（0.02 容检测噪声）
    'special_bonus_cap_without_damage': 0.05, # 未造成伤害时，其他加分的单步上限
    'special_bonus_damage_ratio': 0.15,  # 造成伤害时，其他加分最多约为伤害分的一小部分
    'special_bonus_step_cap': 1.0,       # 其他加分的绝对单步上限

    # 针对"怂"行为的惩罚 (如一直后退或不动)
    'passive_action_penalty_base': -0.01, # 降低基础消极动作惩罚
    'cowardice_threshold': 10,           # 增加容忍度
    
    # 抓取动作引导
    # 注意：`grab_absence_penalty_base` 曾经是全局最强的动作塑形项 ——
    # 超过 5s 不抓取后惩罚随时间线性增长（15s 未抓时已达 -2.0/秒），
    # 远超 `attack_action_bonus`(0.005) 和伤害奖励的实际量级，
    # 于是最优策略退化成"每隔几秒按一次 E"，也就是只按抓取不干别的。
    # 与 "damage 是唯一主奖励" 的设计相悖，故关闭；抓取是否值得由命中伤害说话。
    'grab_action_bonus': 0.0,           # 抓取本身不加分
    'grab_frequency_threshold': 100,     # 保留配置项，惩罚关闭后不生效
    'grab_absence_penalty_base': 0.0,    # 关闭"不抓取就扣分"的强制引导
    'grab_success_bonus': 0.02,          # 抓取命中造成伤害才给极小特别加分
    'grab_combo_bonus': 0.0,             # 抓取连招不再额外加分，避免误导
    
    # 引导奖励
    'attack_action_bonus': 0.005,        # 尝试攻击只给极小引导分
    'combo_variety_bonus': 0.05,         # 多样性只作为特别加分
    'defense_success_reward': 0.0,       # 防御不再单独加分

    # 针对"一成不变的招式"的惩罚。
    # 关键是**只罚无效的重复**：同一招连按但打不出伤害才扣分，一旦命中立刻清零。
    # 不这么门控的话，如果重复某招本来就是最优解（格斗游戏里很常见），
    # 惩罚会把策略推离最优 —— 那就和"damage 是唯一主奖励"的设计冲突了。
    'repeat_no_damage_seconds': 0.4,       # 同一进攻动作连续无伤害超过这么久才开始罚
    'repeat_no_damage_penalty_per_sec': -1.0,  # 超出部分按秒累加（随连按时长升级）
    'monotony_window': 20,                 # 统计最近多少步的招式多样性
    'monotony_min_variety': 3,             # 窗口内少于几种动作算"一成不变"
    'monotony_penalty_per_sec': -0.6,      # 每缺一种动作、每秒扣多少
    'monotony_damage_floor': 0.01,         # 窗口内累计伤害超过这个值就不算"无效"，不罚
}


# 文件路径配置
FILE_PATHS = {
    'checkpoint': 'dqn_optimized.pth',       # --arch pro（旧 ProNet）
    'ctm_checkpoint': 'ctm_planner.pth',     # --arch ctm（新计划模型）
    'locations': 'locations.json',
    'tensorboard_logs': './runs_optimized'
}
