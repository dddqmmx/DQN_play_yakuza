# -*- coding: utf-8 -*-
"""
CTM 计划模型的冒烟 / 回归测试。不需要游戏，CPU 也能跑：

    python test_ctm_planner.py

覆盖：
  1. 三档 profile 的 forward 形状
  2. select_plan 输出合法（长度、取值范围、commit 范围）
  3. 提交长度确实随计划的决断程度变化，且该度量对 Q 的绝对尺度免疫
  3b. **回归**：整条计划真的会被连着执行，不是只有第 0 格
      （修复前 commit 恒为 1，训练日志 32/32 条 Step 行都是"提交1"）
  3c. commit>1 才浮现的耦合：ε 探索截断 + 计划 buffer 只收第 0 格样本
  4. update_model 单步：loss 有限、参数更新、PER 优先级被写回
  5. 槽位自举一致性真的在收敛：Q_j(s) 的 argmax 逐渐对上 Q_{j-1}^tgt(s')
"""
from __future__ import annotations

import shutil
import tempfile

import numpy as np
import torch

from config import CTM_PROFILES, GAME_CONFIG
from ctm_agent import CTMPlannerAgent
from ctm_planner import CTMPlannerNet
from replay_buffer import PlanRun

NUM_ACTIONS = GAME_CONFIG["num_actions"]

# 测试用的迷你配置：结构和真配置同构，只是把每个维度都压小，CPU 上秒级跑完
TINY = {
    **CTM_PROFILES["small"],
    "iterations": 3,
    "d_model": 64,
    "d_input": 32,
    "heads": 2,
    "n_synch_out": 32,
    "n_synch_action": 16,
    "synapse_depth": 2,
    "memory_length": 4,
    "n_random_pairing_self": 2,
    "num_quantiles": 5,
    "plan_length": 4,
    "token_grid": 4,
    "plan_memory_capacity": 128,
    "plan_batch_size": 8,
}

PASSED = []
FAILED = []


def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    print(f"  [{'ok ' if condition else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")


def fake_state(seed=None):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(4, 160, 160), dtype=np.uint8)


def test_forward_shapes():
    print("\n1. forward 形状")
    for prof in ("small", "medium", "large"):
        cfg = CTM_PROFILES[prof]
        net = CTMPlannerNet(NUM_ACTIONS, config=cfg)
        x = torch.from_numpy(np.stack([fake_state(0), fake_state(1)]))
        q, cert = net(x, torch.rand(2, 1), torch.rand(2, 1))
        expect_q = (2, cfg["plan_length"], NUM_ACTIONS, cfg["num_quantiles"], cfg["iterations"])
        expect_c = (2, 2, cfg["iterations"])
        n_params = sum(p.numel() for p in net.parameters())
        check(f"{prof}: q_dist {tuple(q.shape)}", tuple(q.shape) == expect_q, f"期望 {expect_q}")
        check(f"{prof}: certainties {tuple(cert.shape)}", tuple(cert.shape) == expect_c)
        check(
            f"{prof}: certainty 落在 [0,1]",
            bool(((cert >= -1e-4) & (cert <= 1 + 1e-4)).all()),
            f"min={cert.min():.4f} max={cert.max():.4f}",
        )
        check(f"{prof}: 无 NaN", bool(torch.isfinite(q).all()))
        print(f"       参数量 {n_params / 1e6:.2f}M")

        picked, certainty, tick = CTMPlannerNet.pick_most_certain(q, cert)
        check(
            f"{prof}: pick_most_certain 形状",
            tuple(picked.shape) == expect_q[:4] and tuple(certainty.shape) == (2,),
        )
        # 挑出来的确实是 certainty 最大的那个 tick
        manual = torch.stack([q[i, ..., tick[i]] for i in range(2)])
        check(f"{prof}: 挑的是最确定的 tick", torch.allclose(picked, manual))


def _q_with_margins(margins, sigma=0.3, n=256, seed=0, quantiles=1):
    """
    造 (n, L, A, Q) 的 Q 分布，让槽位 j 的最优动作比同槽其余动作高出
    margins[j] 个 σ。这是"训练到一定程度的计划头"的样子。

    单槽位标定（13 个动作，蒙特卡洛实测）：0σ -> 0.145、2σ -> 0.199、
    3σ -> 0.298、10σ -> 0.719。真实 checkpoint 落在 2σ 附近，**不是** 10σ。
    """
    g = torch.Generator().manual_seed(seed)
    L = len(margins)
    q = torch.randn(n, L, NUM_ACTIONS, quantiles, generator=g) * sigma
    for j, m in enumerate(margins):
        q[:, j, 0, :] += m * sigma
    return q


# 6145 步的 ctm_planner_small.pth 在 378 帧真实游戏画面（samples/，与训练同一条
# 预处理）上重放，量到的**逐槽位**决断度。全量统计：
#   逐槽位均值 0.287 / 0.116 / 0.111 / 0.083   —— 后三格低于"毫无偏好"的 0.145
#   旧门 378/378 都只提交 1 步（0% 连招，与训练日志 32/32 条"提交1"一致）
#   新门 355/16/7/0（6.1% 连招）
# 下面这 40 行是**刻意加权**的子集：新门会连招的 23 行全取 + 只走 1 步的每 22 行
# 取 1，目的是让两种门的差别在少量数据上就能判死，不是全量分布的无偏采样。
# 重新生成：把 CTMPlannerNet.plan_slot_confidence 在 samples/ 上跑一遍即可。
REAL_SLOT_CONF = [
    (0.090, 0.168, 0.234, 0.195), (0.302, 0.258, 0.254, 0.142),
    (0.282, 0.245, 0.183, 0.128), (0.074, 0.115, 0.169, 0.133),
    (0.322, 0.227, 0.169, 0.094), (0.318, 0.205, 0.086, 0.276),
    (0.333, 0.277, 0.146, 0.003), (0.393, 0.101, 0.087, 0.218),
    (0.254, 0.258, 0.244, 0.008), (0.272, 0.207, 0.185, 0.308),
    (0.244, 0.275, 0.289, 0.025), (0.250, 0.295, 0.292, 0.052),
    (0.233, 0.264, 0.295, 0.028), (0.370, 0.201, 0.047, 0.233),
    (0.141, 0.159, 0.190, 0.188), (0.383, 0.212, 0.071, 0.076),
    (0.354, 0.205, 0.155, 0.145), (0.340, 0.029, 0.220, 0.067),
    (0.570, 0.220, 0.189, 0.074), (0.470, 0.228, 0.178, 0.008),
    (0.418, 0.206, 0.113, 0.313), (0.192, 0.020, 0.002, 0.054),
    (0.133, 0.017, 0.029, 0.157), (0.455, 0.118, 0.218, 0.094),
    (0.399, 0.027, 0.136, 0.011), (0.475, 0.120, 0.165, 0.024),
    (0.500, 0.044, 0.237, 0.048), (0.489, 0.215, 0.112, 0.013),
    (0.370, 0.015, 0.192, 0.066), (0.464, 0.086, 0.081, 0.005),
    (0.117, 0.131, 0.106, 0.015), (0.277, 0.215, 0.130, 0.033),
    (0.327, 0.269, 0.184, 0.005), (0.466, 0.208, 0.225, 0.032),
    (0.308, 0.214, 0.231, 0.030), (0.363, 0.207, 0.182, 0.053),
    (0.048, 0.157, 0.079, 0.019), (0.037, 0.157, 0.040, 0.016),
    (0.201, 0.271, 0.176, 0.064), (0.122, 0.169, 0.077, 0.052),
]

# 上表里的一行真机数据：前 3 格都过线、末格没过。旧门把它判成 1 步。
MEASURED_SLOT_CONF = [0.302, 0.258, 0.254, 0.142]


def _old_commit_length(mean_conf, L, lo=0.20, hi=0.60):
    """
    修复前的实现，只在测试里作对照用：先把 L 个槽位的决断度平均成一个数，
    再线性映射到 [1, L]。
    """
    ratio = min(max((mean_conf - lo) / (hi - lo), 0.0), 1.0)
    return int(round(1 + ratio * (L - 1)))


def test_commit_length():
    print("\n2/3. 提交长度随计划的决断程度变化")
    net = CTMPlannerNet(NUM_ACTIONS, config=TINY)
    lo = TINY["commit_confidence_lo"]
    L = TINY["plan_length"]

    # 逐槽位阈值：全过 -> 提交满；全不过 -> 只提交 1 步
    k_all = int(net.commit_length(torch.full((1, L), lo + 0.1))[0])
    k_none = int(net.commit_length(torch.full((1, L), lo - 0.1))[0])
    check(f"每格都够决断 -> 提交满 {L} 步", k_all == L, f"k={k_all}")
    check("每格都不决断 -> 只提交 1 步", k_none == 1, f"k={k_none}")

    # 前缀单调：过线的前缀越长，提交越多
    ks = [
        int(net.commit_length(
            torch.tensor([[lo + 0.1] * p + [lo - 0.1] * (L - p)])
        )[0])
        for p in range(L + 1)
    ]
    check("commit 随过线前缀长度单调不减", all(b >= a for a, b in zip(ks, ks[1:])), f"k={ks}")
    check("commit 始终在 [1, L]", all(1 <= k <= L for k in ks), f"k={ks}")
    print(f"       过线前缀长度 0..{L} -> commit {ks}")

    # plan_slot_confidence 必须对 Q 的绝对尺度免疫 —— 熵版 certainty 就是栽在这里：
    # 尺度一大，纯随机的 Q 也会被判成"很确定"，模型会凭噪声连招。
    flat_small = _q_with_margins([0] * L, sigma=0.01, seed=1)
    flat_big = _q_with_margins([0] * L, sigma=5.0, seed=1)
    decisive = _q_with_margins([10] * L, sigma=0.3, seed=1)
    c_small = float(CTMPlannerNet.plan_confidence(flat_small).mean())
    c_big = float(CTMPlannerNet.plan_confidence(flat_big).mean())
    c_dec = float(CTMPlannerNet.plan_confidence(decisive).mean())
    check("决断度对 Q 尺度免疫", abs(c_small - c_big) < 0.02,
          f"σ=0.01 -> {c_small:.3f}, σ=5.0 -> {c_big:.3f}")
    check("无偏好时低于 lo（只提交 1 步）", c_small < lo, f"{c_small:.3f} < {lo}")
    check("有明确最优动作时显著更高", c_dec > c_small + 0.2, f"{c_small:.3f} -> {c_dec:.3f}")
    print(f"       无偏好 {c_small:.3f} / 无偏好但 Q 很大 {c_big:.3f} / 有明确最优 {c_dec:.3f}")

    agent = _make_agent()
    plan, commit, confidence = agent.select_plan(fake_state(7), 0.8, 0.6)
    check("select_plan 计划长度 == plan_length", len(plan) == L, f"len={len(plan)}")
    check(
        "select_plan 动作都是合法 id",
        all(isinstance(a, int) and 0 <= a < NUM_ACTIONS for a in plan),
        f"plan={plan}",
    )
    check("select_plan commit 在 [1, L]", 1 <= commit <= L, f"commit={commit}")
    print(f"       plan={plan} commit={commit} 决断度={confidence:.4f}")
    agent.close()


def test_plan_is_actually_executed():
    """
    回归测试：整条计划不能只有第 0 格被执行。

    修复前 `commit_length` 先把 L 个槽位的决断度**平均**再线性映射到 [1, L]，
    三个毛病叠起来让 commit 恒等于 1（1600 步训练日志里 32/32 条 Step 行都是
    "提交1 均提交1.0 中断0"）：平均让最没学好的后段槽位一票同权；阈值按单槽位
    尺度标定却作用在均值上；round() 让标称 lo 形同虚设（L=4 时实际要 0.267）。

    下面用的都是**真机重放量到的**决断度（见 REAL_SLOT_CONF），不是原测试里
    那种 10σ 的理想情况 —— 原测试只探了 10σ（conf≈0.72，刚好越过 hi），
    真实训练落在 2σ 附近，中间整段从没被测过，所以这个错标漏了过去。
    """
    print("\n3b. 回归：计划真的会被连着执行（不只第 0 格）")
    net = CTMPlannerNet(NUM_ACTIONS, config=TINY)
    lo = TINY["commit_confidence_lo"]
    L = TINY["plan_length"]

    # --- 现场复原：拿真实 checkpoint 量到的逐槽位决断度直接判 ---
    measured = torch.tensor([MEASURED_SLOT_CONF])
    new_k = int(net.commit_length(measured)[0])
    old_k = _old_commit_length(sum(MEASURED_SLOT_CONF) / L, L)
    check(
        "真实 checkpoint 的决断度下，修复前只提交 1 步（复现 bug）",
        old_k == 1, f"旧实现 commit={old_k}",
    )
    check(
        "同样的决断度，修复后提交多步",
        new_k > 1, f"新实现 commit={new_k}",
    )
    print(f"       逐槽位 {MEASURED_SLOT_CONF} (均值 {sum(MEASURED_SLOT_CONF)/L:.3f})"
          f" -> 旧 {old_k} 步 / 新 {new_k} 步")

    # --- 后段槽位含糊，不该否决已经很清楚的前缀 ---
    strong_prefix = [0.30, 0.30, 0.30, 0.10]
    k_new = int(net.commit_length(torch.tensor([strong_prefix]))[0])
    k_old = _old_commit_length(sum(strong_prefix) / L, L)
    check(
        "前 3 格清楚、末格含糊 -> 提交 3 步",
        k_new == 3, f"commit={k_new}",
    )
    check(
        "同一条计划修复前只提交 1 步（末格把前缀一起拖下水）",
        k_old == 1, f"旧实现 commit={k_old}",
    )
    print(f"       逐槽位 {strong_prefix} -> 旧 {k_old} 步 / 新 {k_new} 步")

    # --- 没有 round 死区：刚好过线就该多提交一步 ---
    at_lo = int(net.commit_length(torch.tensor([[lo, lo, lo - 0.01, 0.0]]))[0])
    check("前两格刚好达到 lo -> 提交 2 步（旧实现有 round 死区）", at_lo == 2, f"commit={at_lo}")
    dead_zone_conf = lo + (0.60 - lo) / (2 * (L - 1)) - 1e-6      # 旧实现的死区上沿
    check(
        "旧实现在 lo 之上仍有一整段死区",
        _old_commit_length(lo, L) == 1 and _old_commit_length(dead_zone_conf, L) == 1,
        f"旧实现 conf={lo} 和 conf={dead_zone_conf:.4f} 都只给 1 步",
    )

    # --- 前缀语义：commit 必须精确等于"从第 0 格起连续过线的格数" ---
    exact_ok = True
    for pattern in ([1,1,1,1],[1,1,1,0],[1,1,0,1],[1,0,1,1],[0,1,1,1],[1,0,0,0],[0,0,0,0]):
        conf = torch.tensor([[lo + 0.05 if b else lo - 0.05 for b in pattern]])
        want = 0
        for b in pattern:
            if b: want += 1
            else: break
        want = max(1, want)
        got = int(net.commit_length(conf)[0])
        if got != want:
            exact_ok = False
            print(f"       !! pattern={pattern} 期望 {want} 实得 {got}")
    check("commit == 从第 0 格起连续过线的格数（下限 1）", exact_ok)

    # --- 真实数据回放：修复前在真机数据上 100% 只走 1 步 ---
    old_ks = [_old_commit_length(sum(row) / L, L) for row in REAL_SLOT_CONF]
    new_ks = [int(net.commit_length(torch.tensor([list(row)]))[0]) for row in REAL_SLOT_CONF]
    # 参照实现：commit 应当等于"从第 0 格起连续过线的格数"，下限 1
    ref_ks = []
    for row in REAL_SLOT_CONF:
        k = 0
        for v in row:
            if v >= lo: k += 1
            else: break
        ref_ks.append(max(1, k))
    check("真机数据上修复前 100% 只提交 1 步（复现 bug）",
          set(old_ks) == {1}, f"旧门取值 {sorted(set(old_ks))}")
    check("commit_length 与参照前缀实现逐行一致", new_ks == ref_ks,
          f"不一致 {sum(1 for a, b in zip(new_ks, ref_ks) if a != b)} 行")
    n_multi = sum(1 for k in new_ks if k > 1)
    check("修复后这批数据里有大量决策提交多步", n_multi >= 20, f"{n_multi}/{len(REAL_SLOT_CONF)} 行")
    check("提交步数不再是常数", len(set(new_ks)) >= 3, f"取到的值 {sorted(set(new_ks))}")
    print(f"       {len(REAL_SLOT_CONF)} 行真机决断度: 旧门全是 1 步 | "
          f"新门 {n_multi} 行连招，取值 {sorted(set(new_ks))}")
    print(f"       （全量 378 帧上：旧门 0% 连招，新门 6.1%；这批是加权子集，见 REAL_SLOT_CONF 注释）")

    # --- 防止有人把均值又传回来（那正是这个 bug 的成因）---
    raised = False
    try:
        net.commit_length(torch.tensor([0.25, 0.30]))       # (B,) 而不是 (B, L)
    except ValueError:
        raised = True
    check("传整条计划的均值进来会直接报错", raised)


def test_commit_respects_epsilon_and_plan_memory():
    """
    commit > 1 之后才会浮现的两处耦合。commit 恒为 1 的年代它们都是死的。
    """
    print("\n3c. commit>1 带出来的耦合：ε 探索 与 计划 buffer 纯度")
    L = TINY["plan_length"]

    # --- ε 改写过的槽位，其决断度不再作数：提交长度必须截到那一步 ---
    agent = _make_agent()
    dev = next(agent.policy_net.parameters()).device
    # 让网络吐一条"每格都极决断"的计划，这样 commit 本来会是满 L 步
    q = _q_with_margins([10] * L, sigma=0.3, n=1, seed=5,
                        quantiles=TINY["num_quantiles"]).unsqueeze(-1)
    q = q.expand(-1, -1, -1, -1, TINY["iterations"]).contiguous().to(dev)
    cert = torch.zeros(1, 2, TINY["iterations"], device=dev)
    cert[:, 1, 0] = 1.0
    agent.policy_net.forward = lambda *a, **k: (q, cert)

    agent.explore_epsilon = 0.0
    _, commit_no_eps, _ = agent.select_plan(fake_state(1), 0.9, 0.9)
    check(f"极决断且无 ε -> 提交满 {L} 步", commit_no_eps == L, f"commit={commit_no_eps}")

    agent.explore_epsilon = 1.0            # 每格都被改成随机动作
    _, commit_all_eps, _ = agent.select_plan(fake_state(1), 0.9, 0.9)
    check("ε 改写了第 0 格 -> 只提交 1 步（不拿随机动作开环连招）",
          commit_all_eps == 1, f"commit={commit_all_eps}")
    print(f"       同一条极决断计划: ε=0 提交 {commit_no_eps} 步, ε=1 提交 {commit_all_eps} 步")
    agent.close()

    # --- 计划一致性 loss 只能吃"第 0 格"的样本 ---
    agent = _make_agent()
    rng = np.random.default_rng(0)
    for slot in (0, 0, 0, 1, 2, 3, 1):
        agent.store_transition(
            fake_state(int(rng.integers(0, 999))), 1.0, 1.0, 1,
            0.0, fake_state(int(rng.integers(0, 999))), 0.99, 0.99, False,
            plan_slot=slot,
        )
    check("只有 plan_slot==0 的样本进计划 buffer",
          len(agent.plan_memory) == 3, f"n={len(agent.plan_memory)}")
    check("所有样本都仍然被计入（n-step/PER 不筛）",
          agent.transitions_seen == 7, f"seen={agent.transitions_seen}")
    print(f"       喂了 7 条（3 条来自第 0 格）-> 计划 buffer {len(agent.plan_memory)} 条, "
          f"总计 {agent.transitions_seen} 条")
    agent.close()



def test_plan_run_assembly():
    """
    回报锚的原料：把一次 commit 里连续执行掉的那几步拼回一条 run。
    拼错比不拼更糟 —— 会给槽位 j 安上别的槽位的回报。
    """
    print("\n3d. run 拼装：只收真的连着执行过的那几步")
    agent = _make_agent()
    L = TINY["plan_length"]
    rng = np.random.default_rng(0)

    def feed(slots, done_at=None):
        for i, sl in enumerate(slots):
            agent.store_transition(
                fake_state(int(rng.integers(0, 9999))), 1.0, 1.0, sl + 1, float(sl),
                fake_state(int(rng.integers(0, 9999))), 0.9, 0.9,
                done_at is not None and i == done_at, plan_slot=sl,
            )

    feed([0, 1, 2])                      # commit=3
    feed([0])                            # commit=1，随后的 slot0 会把上一条冲出来
    n_after_commit3 = len(agent.plan_run_memory)
    check("commit=3 拼出 1 条 run", n_after_commit3 == 1, f"n={n_after_commit3}")

    feed([0])                            # 又一条 commit=1
    check("commit=1 不产生 run（没有 j>0 被执行过）",
          len(agent.plan_run_memory) == 1, f"n={len(agent.plan_run_memory)}")

    feed([0, 1])
    feed([0])
    check("commit=2 产生 run", len(agent.plan_run_memory) == 2, f"n={len(agent.plan_run_memory)}")

    def stored_runs():
        m = agent.plan_run_memory
        return [r for r in m.data if r is not None]

    def last_run():
        m = agent.plan_run_memory
        return m.data[(m.write - 1) % m.capacity]

    run = last_run()
    check("run 记下了实际执行的动作序列", run.actions == (1, 2), f"actions={run.actions}")
    check("run 记下了对应的奖励序列", run.rewards == (0.0, 1.0), f"rewards={run.rewards}")

    before = len(agent.plan_run_memory)
    feed([0, 2, 3])                      # 槽位对不上（缺 1）
    feed([0])
    check("槽位错位的样本被丢弃，不会拼成错的 run",
          len(agent.plan_run_memory) == before, f"n={len(agent.plan_run_memory)}")

    before = len(agent.plan_run_memory)
    feed([0, 1, 2, 3])                   # 满 L 步应当立刻收口，不必等下一条计划
    check(f"连续执行满 {L} 步立刻收口", len(agent.plan_run_memory) == before + 1,
          f"n={len(agent.plan_run_memory)}")
    check("run 长度不超过 plan_length",
          all(len(r.actions) <= L for r in stored_runs()))
    print(f"       buffer 里 {len(agent.plan_run_memory)} 条 run，长度 "
          f"{[len(r.actions) for r in stored_runs()]}")
    agent.close()


def test_plan_return_targets():
    """回报锚的目标值：必须是"从第 j 步起的折扣回报"，逐位对得上。"""
    print("\n3e. 回报锚的目标：折扣后缀回报 + 掩码")
    agent = _make_agent()
    agent.reward_scale = 1.0
    L = TINY["plan_length"]
    g = agent.gamma

    run = PlanRun(fake_state(1), 1.0, 1.0, (5, 6, 7), (1.0, 2.0, 4.0),
                  fake_state(2), 0.5, 0.5, False)
    ret, act, steps, mask = agent._plan_run_targets([run])
    ret, act, steps, mask = ret[0].tolist(), act[0].tolist(), steps[0].tolist(), mask[0].tolist()

    want = [1.0 + g * (2.0 + g * 4.0), 2.0 + g * 4.0, 4.0, 0.0]
    check("逐槽位折扣后缀回报正确",
          all(abs(a - b) < 1e-4 for a, b in zip(ret[:3], want[:3])),
          f"得到 {[round(v, 4) for v in ret]}，期望 {[round(v, 4) for v in want]}")
    check("动作对得上", act[:3] == [5, 6, 7], f"act={act}")
    check("到 tail 的步数对得上", steps[:3] == [3.0, 2.0, 1.0], f"steps={steps}")
    check("槽位 0 不参与（它有 PER 的 n-step TD）", mask[0] == 0.0, f"mask={mask}")
    check("执行过的槽位 1..k-1 参与", mask[1] == 1.0 and mask[2] == 1.0, f"mask={mask}")
    check("没执行到的槽位不参与（交给一致性 loss）", mask[3] == 0.0, f"mask={mask}")
    print(f"       回报 {[round(v, 3) for v in ret]} 掩码 {mask}")
    agent.close()


def test_reward_reaches_plan_slots():
    """
    这一节是整件事的目的：**真实奖励要能到达槽位 1**。

    修复前槽位 1..L-1 只有自举一致性 loss，没有任何奖励信号进得去，实测 6145 步
    后它们在真实画面上的决断度只有 0.116/0.111/0.083 —— 低于"毫无偏好"的 0.145。

    这里造一批同头、同尾的 run：第 1 格做 GOOD 拿 +1，做 BAD 拿 -1。两者共用同一个
    尾部自举，所以 Q_1(s,GOOD) - Q_1(s,BAD) 的理论收敛值就是 1-(-1)=2.0
    （实测 1200 步后是 2.016，Q 分别是 +1.009 / -1.007，这一项是准的）。
    只开回报锚（plan_loss_weight=0），关掉 NoisyNets 和目标网络软更新，
    否则测的就不是这条机制了（同 5 号测试）。

    注意不能断言"argmax 是全动作里的 GOOD"太早：槽位 1 上从没被执行过的那 11 个
    动作还停在随机初始化上，得等 GOOD/BAD 拉开差距才会盖过它们。
    """
    print("\n3f. 真实奖励能到达槽位 1（回报锚的目的）")
    tmp = tempfile.mkdtemp()
    agent = CTMPlannerAgent(
        NUM_ACTIONS,
        network_config={**TINY, "use_noisy": False},
        checkpoint_file=f"{tmp}/ctm_test.pth",
        log_dir=tmp,
    )
    agent.batch_size = 8
    agent.plan_batch_size = 8
    agent.plan_run_batch_size = 8
    agent.plan_loss_weight = 0.0          # 只留回报锚
    agent.plan_return_weight = 1.0
    agent.reward_scale = 1.0
    agent.tau = 0.0                       # 冻住目标网络，靶子别跑
    GOOD, BAD = 3, 9

    head, mid, tail = fake_state(11), fake_state(12), fake_state(13)
    for i in range(60):
        good = i % 2 == 0
        a1, r1 = (GOOD, 1.0) if good else (BAD, -1.0)
        agent.store_transition(head, 1.0, 1.0, 0, 0.0, mid, 1.0, 1.0, False, plan_slot=0)
        agent.store_transition(mid, 1.0, 1.0, a1, r1, tail, 1.0, 1.0, False, plan_slot=1)
    agent.store_transition(head, 1.0, 1.0, 0, 0.0, mid, 1.0, 1.0, False, plan_slot=0)
    check("run buffer 攒到了样本", len(agent.plan_run_memory) >= 50,
          f"n={len(agent.plan_run_memory)}")

    dev = next(agent.policy_net.parameters()).device
    h = agent._state_tensor([head])
    one = torch.ones(1, 1, dtype=torch.float32, device=dev)

    def slot1():
        with torch.no_grad():
            q, c = agent.policy_net(h, one, one)
            picked, _, _ = CTMPlannerNet.pick_most_certain(q, c)
            qs = picked[0, 1].mean(-1)                    # 槽位 1 的逐动作 Q
            return float(qs[GOOD] - qs[BAD]), int(qs.argmax())

    gap0, _ = slot1()
    while agent.steps < 600:
        agent.update_model()
    gap1, arg1 = slot1()

    check("回报锚 loss 非零（确实在训）", agent.last_losses["plan_return"] > 0,
          f"{agent.last_losses['plan_return']:.4f}")
    check("训练后槽位 1 明显偏好拿到 +1 的那个动作", gap1 > gap0 + 0.5,
          f"Q1[GOOD]-Q1[BAD]: {gap0:.4f} -> {gap1:.4f}")
    check("差距朝理论值 2.0 走，且没有跑飞", 0.8 < gap1 < 2.5, f"gap={gap1:.4f}")
    check("槽位 1 的 argmax 已经指向 GOOD", arg1 == GOOD, f"argmax={arg1}, GOOD={GOOD}")
    print(f"       {agent.steps} 步后 Q1[GOOD]-Q1[BAD] {gap0:+.4f} -> {gap1:+.4f}"
          f"（理论 2.0），argmax={arg1}")
    agent.close()
    shutil.rmtree(tmp, ignore_errors=True)


def test_return_anchor_is_inert_without_runs():
    """commit 恒为 1 的旧世界里，回报锚必须完全不产生影响（向后兼容）。"""
    print("\n3g. 没有 run 时回报锚静默")
    agent = _make_agent()
    agent.batch_size = 8
    for i in range(40):                    # 全是 plan_slot=0，等于 commit 恒为 1
        agent.store_transition(
            fake_state(i), 1.0, 1.0, 1, 0.1, fake_state(i + 1), 0.99, 0.99, False,
            plan_slot=0,
        )
    check("commit 恒为 1 时 run buffer 为空", len(agent.plan_run_memory) == 0,
          f"n={len(agent.plan_run_memory)}")
    agent.update_model()
    check("回报锚 loss 为 0", agent.last_losses["plan_return"] == 0.0,
          f"{agent.last_losses['plan_return']}")
    check("其余两项照常训练", agent.last_losses["td"] > 0)
    check("参数无 NaN/Inf",
          all(torch.isfinite(p).all() for p in agent.policy_net.parameters()))
    print(f"       losses={ {k: round(v, 4) for k, v in agent.last_losses.items()} }")
    agent.close()


def _make_agent(log_dir=None):
    return CTMPlannerAgent(
        NUM_ACTIONS,
        network_config=TINY,
        checkpoint_file=f"{log_dir or tempfile.mkdtemp()}/ctm_test.pth",
        log_dir=log_dir or tempfile.mkdtemp(),
    )


def _fill(agent, n, seed=0):
    rng = np.random.default_rng(seed)
    s = fake_state(seed)
    boss = self_hp = 1.0
    for i in range(n):
        nxt = fake_state(seed + i + 1)
        next_boss = max(0.0, boss - rng.random() * 0.01)
        next_self = max(0.0, self_hp - rng.random() * 0.01)
        agent.store_transition(
            s, boss, self_hp, int(rng.integers(0, NUM_ACTIONS)),
            float(rng.normal()), nxt, next_boss, next_self, False,
        )
        s, boss, self_hp = nxt, next_boss, next_self


def test_update_step():
    print("\n4. update_model 单步")
    tmp = tempfile.mkdtemp()
    agent = _make_agent(tmp)
    agent.batch_size = 8

    _fill(agent, 40, seed=1)
    check("PER buffer 有样本", len(agent.memory) > 0, f"n={len(agent.memory)}")
    check("计划 buffer 有样本", len(agent.plan_memory) == 40, f"n={len(agent.plan_memory)}")

    before = [p.detach().clone() for p in agent.policy_net.parameters()]
    prio_before = agent.memory.tree.tree[-agent.memory.tree.capacity:].copy()

    agent.update_model()

    check("steps 前进", agent.steps == 1, f"steps={agent.steps}")
    changed = sum(
        1 for a, b in zip(before, agent.policy_net.parameters())
        if not torch.equal(a, b.detach())
    )
    check("参数确实被更新", changed > 0, f"{changed}/{len(before)} 个张量变了")
    finite = all(torch.isfinite(p).all() for p in agent.policy_net.parameters())
    check("参数无 NaN/Inf", bool(finite))

    prio_after = agent.memory.tree.tree[-agent.memory.tree.capacity:]
    check("PER 优先级被写回", not np.array_equal(prio_before, prio_after))

    # 存档 / 读档往返
    agent.save_checkpoint()
    agent.save_queue.join()
    fresh = _make_agent(tmp)
    fresh.checkpoint_file = agent.checkpoint_file
    ok = fresh.load_checkpoint()
    check("checkpoint 往返加载成功", ok)
    if ok:
        same = all(
            torch.equal(a.detach().cpu(), b.detach().cpu())
            for a, b in zip(agent.policy_net.parameters(), fresh.policy_net.parameters())
        )
        check("读回来的权重一致", same)
        check("steps 一起恢复", fresh.steps == agent.steps, f"{fresh.steps} vs {agent.steps}")
    fresh.close()
    agent.close()
    shutil.rmtree(tmp, ignore_errors=True)


def test_plan_consistency_converges():
    """
    验证槽位自举一致性这个**回归目标**本身是对的：Q_j(s) 在往
    Q_{j-1}^target(s') 上收敛。

    这里刻意把三个混淆项摘掉，否则测的就不是这条机制了：
      - `use_noisy=False`：NoisyLinear 每次前向重采噪声，量级和早期学到的信号
        相当，argmax 会随机跳（诊断时实测噪声下 400 步内看不出任何趋势）；
      - `tau=0`：默认 EMA 目标是会动的，回归的靶子一直在跑；
      - `plan_curriculum_lookahead=-1`：关课程掩码，让所有槽位都参与。
    这三项都是生产设置里该有的东西，只是不该在"验证回归目标"的测试里出现。
    生产路径（噪声 + EMA 目标）由下面的 test_losses_are_finite 覆盖。
    """
    print("\n5. 槽位自举一致性：Q_j(s) -> Q_{j-1}^target(s')")
    tmp = tempfile.mkdtemp()
    agent = CTMPlannerAgent(
        NUM_ACTIONS,
        network_config={**TINY, "use_noisy": False},
        checkpoint_file=f"{tmp}/ctm_test.pth",
        log_dir=tmp,
    )
    agent.batch_size = 8
    agent.plan_batch_size = 8
    agent.plan_curriculum_lookahead = -1
    agent.tau = 0.0
    _fill(agent, 24, seed=3)

    batch = agent.plan_memory.sample(16)
    dev = next(agent.policy_net.parameters()).device
    s = agent._state_tensor([t.state for t in batch])
    ns = agent._state_tensor([t.next_state for t in batch])
    boss = torch.tensor([[t.boss_health] for t in batch], dtype=torch.float32, device=dev)
    selfh = torch.tensor([[t.self_health] for t in batch], dtype=torch.float32, device=dev)
    nboss = torch.tensor([[t.next_boss_health] for t in batch], dtype=torch.float32, device=dev)
    nself = torch.tensor([[t.next_self_health] for t in batch], dtype=torch.float32, device=dev)

    def probe():
        """返回 (槽位间 Q 距离, argmax 一致率)。距离才是被优化的量。"""
        with torch.no_grad():
            pq, pc = agent.policy_net(s, boss, selfh)
            tq, tc = agent.target_net(ns, nboss, nself)
            p, _, _ = CTMPlannerNet.pick_most_certain(pq, pc)
            t, _, _ = CTMPlannerNet.pick_most_certain(tq, tc)
            dist = float((p[:, 1:] - t[:, :-1]).abs().mean())
            agree = float((p.mean(-1).argmax(-1)[:, 1:] == t.mean(-1).argmax(-1)[:, :-1]).float().mean())
            return dist, agree

    d0, a0 = probe()
    for _ in range(400):
        agent.update_model()
    d1, a1 = probe()

    print(f"       |Q_j(s) - Q_{{j-1}}^tgt(s')| {d0:.5f} -> {d1:.5f}")
    print(f"       argmax 一致率              {a0:.3f} -> {a1:.3f}")
    check("槽位间 Q 距离显著下降", d1 < d0 * 0.95, f"{d0:.5f} -> {d1:.5f}")
    check("argmax 一致率不降反升", a1 > a0, f"{a0:.3f} -> {a1:.3f}")
    check("训练后参数仍有限", all(torch.isfinite(p).all() for p in agent.policy_net.parameters()))
    agent.close()
    shutil.rmtree(tmp, ignore_errors=True)


def test_losses_are_finite():
    """生产设置（NoisyLinear 探索 + EMA 目标 + 课程掩码）下两个 loss 都要是有限值。"""
    print("\n6. 生产设置下的 loss 健康度")
    tmp = tempfile.mkdtemp()
    agent = _make_agent(tmp)
    agent.batch_size = 8
    agent.plan_batch_size = 8
    _fill(agent, 40, seed=5)

    td_hist, plan_hist = [], []
    for _ in range(60):
        agent.update_model()
        td_hist.append(agent.last_losses["td"])
        plan_hist.append(agent.last_losses["plan"])

    check("TD loss 全程有限", all(np.isfinite(td_hist)), f"最后 {td_hist[-1]:.4f}")
    check("计划 loss 全程有限", all(np.isfinite(plan_hist)), f"最后 {plan_hist[-1]:.4f}")
    check("计划 loss 非零（确实在训）", plan_hist[-1] > 0, f"{plan_hist[-1]:.4f}")
    check("参数无 NaN/Inf", all(torch.isfinite(p).all() for p in agent.policy_net.parameters()))
    print(f"       TD {td_hist[0]:.4f} -> {td_hist[-1]:.4f} | "
          f"Plan {plan_hist[0]:.4f} -> {plan_hist[-1]:.4f}")
    agent.close()
    shutil.rmtree(tmp, ignore_errors=True)


def test_exploration_is_alive():
    """
    回归测试这次训练翻车的根因：探索噪声被分位数平均抵消，策略锁死在单一动作。

    动作排序看的是 `q.mean(-1)`。如果把 NoisyLinear 铺在 (A × quantiles) 上，
    各分位数的噪声互相独立，一平均就按 sqrt(num_quantiles) 衰减 —— 线上实测
    噪声 std 只剩 0.011，而最优/次优动作的 Q 差距是 0.213，差 19 倍，
    重采 60 次噪声计划一次都不变。修法是让噪声只作用在 (L, A) 上再广播到分位数。
    """
    print("\n7. 探索噪声没有被分位数平均抵消")
    net = CTMPlannerNet(NUM_ACTIONS, config=TINY)
    x = torch.from_numpy(fake_state(11)[None])
    hp = torch.full((1, 1), 0.8)

    per_q, avg_q, plans = [], [], []
    for _ in range(120):
        net.reset_noise()
        with torch.no_grad():
            q, c = net(x, hp, hp)
            p, _, _ = CTMPlannerNet.pick_most_certain(q, c)
        per_q.append(p[0, 0, :, 0].numpy())
        avg_q.append(p[0, 0].mean(-1).numpy())
        plans.append(tuple(p.mean(-1).argmax(-1)[0].tolist()))

    s_single = float(np.array(per_q).std(0).mean())
    s_avg = float(np.array(avg_q).std(0).mean())
    ratio = s_single / max(s_avg, 1e-9)
    n_plans = len(set(plans))
    # 噪声要有意义，得和"最优 vs 次优"的差距同量级 —— 绝对值多大取决于网络训练
    # 到什么程度，所以这里比的是比值，不是硬阈值。
    mean_q = np.array(avg_q).mean(0)
    gap = float(np.sort(mean_q)[-1] - np.sort(mean_q)[-2])
    print(f"       单分位数 std={s_single:.5f}  平均后 std={s_avg:.5f}  衰减 {ratio:.2f}x "
          f"(sqrt({TINY['num_quantiles']})={TINY['num_quantiles'] ** 0.5:.1f})")
    print(f"       噪声/动作差距 = {s_avg:.5f}/{gap:.5f} = {s_avg / max(gap, 1e-9):.2f}")
    print(f"       同一状态 120 次决策产生 {n_plans} 种不同计划")

    check("噪声不随分位数平均衰减", ratio < 1.5, f"衰减 {ratio:.2f}x，应接近 1.0")
    check("噪声真的改变了动作排序", n_plans > 1, f"只有 {n_plans} 种计划")
    check("噪声与动作差距同量级", s_avg / max(gap, 1e-9) > 0.3,
          f"比值 {s_avg / max(gap, 1e-9):.2f}（线上翻车时是 0.05）")

    # 关掉 noisy 时必须是确定性的，否则没法做可复现的评测
    quiet = CTMPlannerNet(NUM_ACTIONS, config={**TINY, "use_noisy": False})
    outs = []
    for _ in range(5):
        quiet.reset_noise()
        with torch.no_grad():
            q, c = quiet(x, hp, hp)
        outs.append(q)
    check("use_noisy=False 时输出确定", all(torch.equal(outs[0], o) for o in outs[1:]))


def test_epsilon_floor():
    """ε 兜底：sigma 学没了也还得有探索。"""
    print("\n8. ε 兜底")
    tmp = tempfile.mkdtemp()
    agent = CTMPlannerAgent(
        NUM_ACTIONS,
        network_config={**TINY, "use_noisy": False, "explore_epsilon": 0.5},
        checkpoint_file=f"{tmp}/x.pth", log_dir=tmp,
    )
    s = fake_state(3)
    plans = [tuple(agent.select_plan(s, 0.9, 0.8)[0]) for _ in range(60)]
    flat = [a for p in plans for a in p]
    check("ε>0 时同一状态会产生不同计划", len(set(plans)) > 1, f"{len(set(plans))} 种")
    check("随机动作覆盖了多种 id", len(set(flat)) > 3, f"出现 {len(set(flat))} 种动作")
    agent.close()

    agent = CTMPlannerAgent(
        NUM_ACTIONS,
        network_config={**TINY, "use_noisy": False, "explore_epsilon": 0.0},
        checkpoint_file=f"{tmp}/y.pth", log_dir=tmp,
    )
    plans = [tuple(agent.select_plan(s, 0.9, 0.8)[0]) for _ in range(20)]
    check("ε=0 且无噪声时计划确定", len(set(plans)) == 1, f"{len(set(plans))} 种")
    agent.close()
    shutil.rmtree(tmp, ignore_errors=True)


def test_monotony_penalty():
    """
    一成不变的招式要扣分，但**打得出伤害的重复不能扣** —— 否则会把策略推离
    本来就最优的连按。
    """
    print("\n9. 一成不变招式的惩罚")
    from core.observation import GameObservation

    class _Cap:
        def grab(self, *a, **k): return None
        def release(self): pass

    def run(actions, damage_per_step, steps=40):
        obs = GameObservation.__new__(GameObservation)   # 只测奖励，不碰截屏
        obs.frame_size = (160, 160)
        obs.reset_reward_stats()
        obs.last_action = -1
        total = 0.0
        boss = self_hp = 1.0
        for i in range(steps):
            nb = max(0.0, boss - damage_per_step)
            r = obs.calculate_reward(self_hp, boss, self_hp, nb, actions[i % len(actions)])
            obs.last_step_time -= 0.1          # 模拟每步 0.1s
            total += r if r is not None else 0.0
            boss = nb
        return total

    same_no_dmg = run([2], 0.0)
    varied_no_dmg = run([1, 2, 7, 9, 5], 0.0)
    same_with_dmg = run([2], 0.004)
    varied_with_dmg = run([1, 2, 7, 9, 5], 0.004)

    print(f"       无伤害: 一成不变 {same_no_dmg:7.2f} vs 多样 {varied_no_dmg:7.2f}")
    print(f"       有伤害: 一成不变 {same_with_dmg:7.2f} vs 多样 {varied_with_dmg:7.2f}")
    check("无伤害时一成不变被罚得更狠", same_no_dmg < varied_no_dmg - 0.5,
          f"{same_no_dmg:.2f} vs {varied_no_dmg:.2f}")
    check("打得出伤害时重复不被惩罚", same_with_dmg > varied_with_dmg - 0.5,
          f"{same_with_dmg:.2f} vs {varied_with_dmg:.2f}")
    check("伤害仍是主导项", same_with_dmg > same_no_dmg + 5,
          f"{same_no_dmg:.2f} -> {same_with_dmg:.2f}")


if __name__ == "__main__":
    print("=" * 60)
    print(f"CTM 计划模型测试 | num_actions={NUM_ACTIONS}")
    print("=" * 60)

    # 每个用例跑之前都重新播种。以前只在这里播一次，于是每个用例的网络初始化
    # 都取决于**它前面跑过哪些用例** —— 新增几个用例就把 5 号的随机流挪走了，
    # 那条本来 6/6 个种子都稳过的断言凭空变红。测试之间不该有这种耦合。
    for fn in (
        test_forward_shapes,
        test_commit_length,
        test_plan_is_actually_executed,
        test_commit_respects_epsilon_and_plan_memory,
        test_plan_run_assembly,
        test_plan_return_targets,
        test_reward_reaches_plan_slots,
        test_return_anchor_is_inert_without_runs,
        test_update_step,
        test_plan_consistency_converges,
        test_losses_are_finite,
        test_exploration_is_alive,
        test_epsilon_floor,
        test_monotony_penalty,
    ):
        torch.manual_seed(0)
        np.random.seed(0)
        fn()

    print("\n" + "=" * 60)
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        for name in FAILED:
            print(f"  FAIL: {name}")
        raise SystemExit(1)
    print("全部通过")
