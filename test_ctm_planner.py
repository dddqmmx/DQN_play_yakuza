# -*- coding: utf-8 -*-
"""
CTM 计划模型的冒烟 / 回归测试。不需要游戏，CPU 也能跑：

    python test_ctm_planner.py

覆盖：
  1. 三档 profile 的 forward 形状
  2. select_plan 输出合法（长度、取值范围、commit 范围）
  3. 提交长度确实随计划的决断程度变化，且该度量对 Q 的绝对尺度免疫
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


def test_commit_length():
    print("\n2/3. 提交长度随计划的决断程度变化")
    net = CTMPlannerNet(NUM_ACTIONS, config=TINY)
    lo = TINY["commit_confidence_lo"]
    hi = TINY["commit_confidence_hi"]
    L = TINY["plan_length"]

    probe = torch.tensor([0.0, lo, (lo + hi) / 2, hi, 1.0])
    k = net.commit_length(probe)
    check("决断度=0 -> 只提交 1 步", int(k[0]) == 1, f"k={k.tolist()}")
    check(f"决断度=1 -> 提交满 {L} 步", int(k[-1]) == L, f"k={k.tolist()}")
    check("commit 单调不减", bool((k[1:] >= k[:-1]).all()), f"k={k.tolist()}")
    check("commit 始终在 [1, L]", bool(((k >= 1) & (k <= L)).all()))
    print(f"       决断度 {[round(v, 2) for v in probe.tolist()]} -> commit {k.tolist()}")

    # plan_confidence 必须对 Q 的绝对尺度免疫 —— 熵版 certainty 就是栽在这里：
    # 尺度一大，纯随机的 Q 也会被判成"很确定"，模型会凭噪声连招。
    torch.manual_seed(1)
    flat_small = torch.randn(512, L, NUM_ACTIONS, 1) * 0.01
    flat_big = torch.randn(512, L, NUM_ACTIONS, 1) * 5.0
    decisive = torch.randn(512, L, NUM_ACTIONS, 1) * 0.3
    decisive[:, :, 0] += 10 * 0.3
    c_small = float(CTMPlannerNet.plan_confidence(flat_small).mean())
    c_big = float(CTMPlannerNet.plan_confidence(flat_big).mean())
    c_dec = float(CTMPlannerNet.plan_confidence(decisive).mean())
    check("决断度对 Q 尺度免疫", abs(c_small - c_big) < 0.02, f"σ=0.01 -> {c_small:.3f}, σ=5.0 -> {c_big:.3f}")
    check("无偏好时低于 lo（只提交 1 步）", c_small < lo, f"{c_small:.3f} < {lo}")
    check("有明确最优动作时显著更高", c_dec > c_small + 0.2, f"{c_small:.3f} -> {c_dec:.3f}")
    check(
        "决断的计划确实提交多步",
        int(net.commit_length(torch.tensor([c_dec]))[0]) > 1,
        f"commit={int(net.commit_length(torch.tensor([c_dec]))[0])}",
    )
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
    torch.manual_seed(0)
    np.random.seed(0)

    print("=" * 60)
    print(f"CTM 计划模型测试 | num_actions={NUM_ACTIONS}")
    print("=" * 60)

    test_forward_shapes()
    test_commit_length()
    test_update_step()
    test_plan_consistency_converges()
    test_losses_are_finite()
    test_exploration_is_alive()
    test_epsilon_floor()
    test_monotony_penalty()

    print("\n" + "=" * 60)
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        for name in FAILED:
            print(f"  FAIL: {name}")
        raise SystemExit(1)
    print("全部通过")
