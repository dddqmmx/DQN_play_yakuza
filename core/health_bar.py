# -*- coding: utf-8 -*-
"""
血条检测（Yakuza 6 HUD）。

# 旧实现为什么不好用

1. **`detect_health` 用"最右侧红像素"当血量。**
   画面里任何偏红的像素（红色任务文字、招牌、木地板、血迹）都会把血量顶到 1.0。
2. **掩码全空时返回 `0.0`，而不是"读不到"。**
   训练循环把 0.0 当成"boss 被打死 / 玩家死亡"，于是产生
   `damage_dealt=1.0 → reward≈250` 的垃圾样本，直接污染 replay buffer。
3. **`locate_health_bars` 在全屏找"细长红带"。**
   菜单分隔线、任务列表红字都会被当成血条锁定
   （locations.json 里那组 `[[49,13],[1280,40]]` 就是这么来的）。
4. **定位到的框是"当前填充范围"，不是"血条轨道"。**
   于是 `血量 = 填充宽度 / 框宽度 ≡ 1.0`，永远满血。

# 实测的 HUD 结构（1280x720）

    玩家条  轨道 x=138..500  y=40..48   填充 BGR≈(19,19,209)
    Boss条  轨道 x=320..965  y=665..675 两端金色端帽
    两者"已损失"部分都是**纯黑 (0,0,0)**
    前沿有一段高亮火花/渐变（Boss 条上宽达 ~70px）

低血量时**已损失部分改为紫色脉动**（BGR 在 (0,0,0)↔(73,12,105) 之间来回），
所以"量黑色长度"在低血闪烁时会失效；但**红色填充本身始终稳定**。

# 本实现

- **纯黑是轨道的指纹**：轨道右端 = 填充右侧那段连续纯黑的末尾。
  游戏背景几乎不会恰好是 (0,0,0)，所以这比"找红带"稳得多。
- **轨道只锁一次**（连续多帧一致 + 右端取历史最大），之后每帧只量
  "从左端起的连续红色长度"，低血闪烁不受影响。
- **结构化校验**代替颜色阈值：轨道内必须能被
  (填充 | 已损失 | 前沿高亮) 解释，且轨道左右紧邻的列**不是**这些颜色
  （即轨道是"有边界"的）。淡出到全黑时边界检查失败 → 返回 `None`，
  而不是误报"血量 0"。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------- 颜色门限
# 填充红：H≈0（跨 180 环回），高饱和、够亮
# 下界从 170 放宽到 165：热血模式下玩家条前段实测 H=166~169、S≈233，
# 明显还是填充红，只是被染了一点品红，卡在旧阈值外会被整列漏判。
FILL_HUE_BANDS: Tuple[Tuple[int, int], ...] = ((0, 8), (165, 180))
FILL_SAT_MIN = 110
FILL_VAL_MIN = 80

# 极限热血模式（按 Q）会给血条盖一层蓝->紫的动画，填充色从 H≈0 整体搬走。
# 实测 1280x720（tools/record_heat_effect.py 采的 samples_heat/）：
#   特效期 boss 条左端 HSV(103, 95, 78)，右端一路渐变到 HSV(165, 98, 52)；
#   同一时刻玩家条基本不受影响，仍是 H≈0 的红。
# 这套门限**只在已锁定的轨道内测量时**启用；定位轨道仍然只认红，
# 否则满屏的蓝色 UI 都会被当成候选血条。
HEAT_HUE_BAND: Tuple[int, int] = (95, 168)
HEAT_SAT_MIN = 70
HEAT_VAL_MIN = 40
# 轨道内有这么大比例的列落在热血色带里，才认定"特效正在放"
HEAT_MIN_FRAC = 0.15

# 轨道就地修正的约束（见 BarTracker._repair）
REPAIR_UNTIL_GOOD_READS = 60   # 连续读出这么多次有效血量后冻结轨道，不再修
REPAIR_MARGIN_RATIO = 0.08     # 单侧单次最多外扩轨道宽度的这个比例

# 已损失部分（正常）：纯黑。留余量给压缩/缩放噪声和紫色脉动最暗的一相。
BLACK_MAX = 14

# 已损失部分（低血脉动）：紫色，G 是最小通道。
# 脉动会从 (0,0,0) 一路亮到 (73,12,105)，最暗的几档靠 BLACK_MAX 兜底。
PULSE_VAL_MAX = 140
PULSE_G_MARGIN_R = 6
PULSE_G_MARGIN_B = 4

# 前沿高亮火花：暖色调 + 够饱和（V 下限放低以覆盖火花衰减尾巴）
GLOW_VAL_MIN = 40
GLOW_SAT_MIN = 60
GLOW_HUE_MAX = 38


def _hsv(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]


def fill_mask(bgr: np.ndarray) -> np.ndarray:
    """血条填充色（饱和红）。"""
    if bgr.size == 0:
        return np.zeros(bgr.shape[:2], dtype=bool)
    h, s, v = _hsv(bgr)
    hue_ok = np.zeros(h.shape, dtype=bool)
    for lo, hi in FILL_HUE_BANDS:
        hue_ok |= (h >= lo) & (h <= hi)
    return hue_ok & (s >= FILL_SAT_MIN) & (v >= FILL_VAL_MIN)


def black_mask(bgr: np.ndarray) -> np.ndarray:
    """纯黑：空轨道的指纹。"""
    if bgr.size == 0:
        return np.zeros(bgr.shape[:2], dtype=bool)
    return bgr.max(axis=2) <= BLACK_MAX


def pulse_mask(bgr: np.ndarray) -> np.ndarray:
    """低血量时已损失部分的紫色脉动（G 明显低于 R 和 B）。"""
    if bgr.size == 0:
        return np.zeros(bgr.shape[:2], dtype=bool)
    b = bgr[:, :, 0].astype(np.int16)
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)
    return (
        (bgr.max(axis=2) <= PULSE_VAL_MAX)
        & (r >= g + PULSE_G_MARGIN_R)
        & (b >= g + PULSE_G_MARGIN_B)
    )


def empty_mask(bgr: np.ndarray) -> np.ndarray:
    """已损失部分：纯黑或低血紫色脉动。"""
    return black_mask(bgr) | pulse_mask(bgr)


def heat_mask(bgr: np.ndarray) -> np.ndarray:
    """
    极限热血模式的蓝->紫覆盖色。只在已锁定轨道内当作"填充"，不参与定位。
    """
    if bgr.size == 0:
        return np.zeros(bgr.shape[:2], dtype=bool)
    h, s, v = _hsv(bgr)
    lo, hi = HEAT_HUE_BAND
    return (h >= lo) & (h <= hi) & (s >= HEAT_SAT_MIN) & (v >= HEAT_VAL_MIN)


def glow_mask(bgr: np.ndarray) -> np.ndarray:
    """血条前沿的高亮火花/渐变。"""
    if bgr.size == 0:
        return np.zeros(bgr.shape[:2], dtype=bool)
    h, s, v = _hsv(bgr)
    warm = (h <= GLOW_HUE_MAX) | (h >= 170)
    return warm & (s >= GLOW_SAT_MIN) & (v >= GLOW_VAL_MIN)


# ---------------------------------------------------------------- 全黑画面
# 玩家死亡/结算/读盘时整屏几乎全黑，此时血条读不到，需要按回车推进。
# 实测：死亡黑屏 f(V<=40)=0.993，"Do you want to continue?" = 0.992，
#       而最暗的一帧战斗画面只有 0.285 —— 阈值取 0.85 有 3 倍余量。
BLACK_SCREEN_VAL_MAX = 40
BLACK_SCREEN_MIN_FRAC = 0.85


def dark_fraction(frame: np.ndarray, val_max: int = BLACK_SCREEN_VAL_MAX) -> float:
    """画面中"暗像素"的占比。降采样后计算，够快也够稳。"""
    if frame is None or frame.size == 0:
        return 0.0
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    v = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 2]
    return float((v <= val_max).mean())


def is_black_screen(
    frame: np.ndarray,
    val_max: int = BLACK_SCREEN_VAL_MAX,
    min_frac: float = BLACK_SCREEN_MIN_FRAC,
) -> bool:
    """画面是否几乎全黑（死亡/结算/读盘）。"""
    return dark_fraction(frame, val_max) >= min_frac


# ---------------------------------------------------------------- 数据结构
@dataclass(frozen=True)
class BarBox:
    """血条**轨道**（填充区 + 已损失区，不含描边），x2/y2 为开区间。"""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1

    def valid(self) -> bool:
        return self.w > 8 and self.h > 0

    def as_pairs(self) -> List[List[int]]:
        return [[self.x1, self.y1], [self.x2, self.y2]]

    @staticmethod
    def from_pairs(pairs) -> "BarBox":
        (x1, y1), (x2, y2) = pairs
        return BarBox(int(x1), int(y1), int(x2), int(y2))

    def scaled(self, src: Tuple[int, int], dst: Tuple[int, int]) -> "BarBox":
        sw, sh = src
        dw, dh = dst
        if sw <= 0 or sh <= 0:
            return self
        fx, fy = dw / sw, dh / sh
        return BarBox(
            int(round(self.x1 * fx)),
            int(round(self.y1 * fy)),
            int(round(self.x2 * fx)),
            int(round(self.y2 * fy)),
        )

    def iou(self, other: "BarBox") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = self.w * self.h + other.w * other.h - inter
        return inter / union if union > 0 else 0.0


@dataclass
class BarReading:
    """一次读数。`value is None` 表示本帧不可信，调用方**必须跳过**。"""

    value: Optional[float]
    confidence: float = 0.0
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.value is not None


@dataclass
class BarTarget:
    """搜索先验。roi 用归一化坐标，适配任意分辨率。"""

    name: str
    roi: Tuple[float, float, float, float]     # x1,y1,x2,y2 归一化
    min_track_ratio: float = 0.10              # 轨道宽 / 帧宽 下限
    max_track_ratio: float = 0.90
    min_height: int = 4                        # @720p，内部按帧高缩放
    max_height: int = 20
    expect_left: Optional[float] = None        # 期望左端（归一化），仅用于打分


PLAYER_TARGET = BarTarget(
    name="player",
    roi=(0.04, 0.02, 0.60, 0.16),
    min_track_ratio=0.15,
    max_track_ratio=0.50,
    min_height=4,
    max_height=16,
    expect_left=0.108,
)

BOSS_TARGET = BarTarget(
    name="boss",
    roi=(0.10, 0.58, 0.99, 0.99),
    min_track_ratio=0.25,
    max_track_ratio=0.85,
    min_height=4,
    max_height=20,
    expect_left=0.250,
)


# ---------------------------------------------------------------- 列分类
def _core_rows(frame: np.ndarray, box: BarBox) -> np.ndarray:
    """取轨道竖直中间段，躲开顶部高光渐变和描边渗色。"""
    region = frame[box.y1:box.y2, box.x1:box.x2]
    h = region.shape[0]
    if h < 3:
        return region
    r0 = int(h * 0.35)
    r1 = max(r0 + 1, int(round(h * 0.95)))
    return region[r0:r1]


def _col_classes(core: np.ndarray, include_heat: bool = False) -> Dict[str, np.ndarray]:
    """每列是否为 fill / empty / glow（列内过半行满足）。"""
    out = {
        "fill": fill_mask(core).mean(axis=0) >= 0.5,
        "empty": empty_mask(core).mean(axis=0) >= 0.5,
        "glow": glow_mask(core).mean(axis=0) >= 0.5,
    }
    if include_heat:
        out["heat"] = heat_mask(core).mean(axis=0) >= 0.5
    return out


def _leading_run(flags: np.ndarray, gap_tol: int = 4) -> int:
    """从左端起的连续 True 长度（允许 <=gap_tol 空隙），返回最后 True 列+1。"""
    n = len(flags)
    if n == 0:
        return 0
    start = 0
    while start < min(3, n) and not flags[start]:   # 容忍左侧被描边吃掉
        start += 1
    if start >= min(3, n):
        return 0
    end = 0
    gap = 0
    for i in range(start, n):
        if flags[i]:
            end = i + 1
            gap = 0
        else:
            gap += 1
            if gap > gap_tol:
                break
    return end


def _trailing_run(flags: np.ndarray, gap_tol: int = 2) -> int:
    """从右端起的连续 True 长度。"""
    n = len(flags)
    if n == 0:
        return 0
    start = n - 1
    while start > max(n - 4, -1) and not flags[start]:
        start -= 1
    if start <= max(n - 4, -1):
        return 0
    count = 0
    gap = 0
    for i in range(start, -1, -1):
        if flags[i]:
            count = n - i
            gap = 0
        else:
            gap += 1
            if gap > gap_tol:
                break
    return count


# ---------------------------------------------------------------- 读数
def _bounded(frame: np.ndarray, box: BarBox) -> Tuple[bool, bool]:
    """
    轨道是否"有边界"：紧邻的像素**不属于**血条（不是 fill 也不是 empty）。

    返回 (right_ok, vertical_ok)。

    只查右侧和上下，**不查左侧** —— 血条左端内侧常有一段纯黑内边距
    （Boss 条 x=315..319 就是纯黑），查左侧会误判成"无边界"。

    淡出到全黑 / 全屏纯色时这两项都会失败，于是"全黑轨道"被判为
    "HUD 不在"而不是"血量 0"。
    """
    fh, fw = frame.shape[:2]

    def not_belong(region: np.ndarray) -> float:
        if region.size == 0:
            return 0.0
        return 1.0 - float((fill_mask(region) | empty_mask(region)).mean())

    x_lo, x_hi = min(fw, box.x2), min(fw, box.x2 + 3)
    right = not_belong(frame[box.y1:box.y2, x_lo:x_hi]) >= 0.5 if x_hi > x_lo else False

    top = frame[max(0, box.y1 - 3):box.y1, box.x1:box.x2]
    bottom = frame[min(fh, box.y2):min(fh, box.y2 + 3), box.x1:box.x2]
    vertical = max(not_belong(top), not_belong(bottom)) >= 0.5

    return right, vertical


def measure_fill(frame: np.ndarray, box: BarBox) -> BarReading:
    """
    在**已锁定的轨道** box 内量血量。

    血量边界取"最后一个填充列"与"右侧已损失区起点"的中点
    —— 两者之间是前沿高亮，取中点最接近真实值。
    """
    if frame is None or not box.valid():
        return BarReading(None, 0.0, "invalid-box")
    fh, fw = frame.shape[:2]
    if box.x1 < 0 or box.y1 < 0 or box.x2 > fw or box.y2 > fh:
        return BarReading(None, 0.0, "box-out-of-frame")

    core = _core_rows(frame, box)
    cls = _col_classes(core, include_heat=True)
    width = len(cls["fill"])
    if width <= 8:
        return BarReading(None, 0.0, "too-narrow")

    # 极限热血模式：填充色被蓝->紫动画整体染掉，此时把热血色也算作"填充"。
    #
    # 判据是**结构**而不是颜色占比：
    #   - 覆盖动画从左端往右铺，所以轨道左端"一点红都没有"（run_plain == 0）；
    #   - 低血量时的紫色脉动虽然颜色和热血色高度重叠（脉动最亮档 BGR(73,12,105)
    #     正落在热血色带里），但它只出现在右侧已损失区，左端仍是红 —— 只看颜色
    #     会把脉动帧全判成热血，实测让 378 张常态样本里 59 张被误拒。
    # gap_tol 放大是为了跨过 boss 条两端的金色端帽（约 10 列，固定不参与填充）。
    merged = cls["fill"] | cls["heat"]
    # 热血覆盖色和红色填充之间有一段谁都不属于的过渡带（boss 条实测约 31/645 列，
    # 约 5%），gap 容忍度不够就会让 run_merged 断在过渡带前面。
    gap_tol = max(4, width // 40)
    heat_gap_tol = max(8, width // 16)
    run_plain = _leading_run(cls["fill"])
    run_merged = _leading_run(merged, gap_tol=heat_gap_tol)
    heat_frac = float(cls["heat"].mean())
    heat_on = (
        heat_frac >= HEAT_MIN_FRAC
        and run_plain == 0
        and run_merged >= max(8, width // 20)
        # 领先段若本身就被判为"已损失"（低血脉动），那是脉动不是覆盖动画
        and float(cls["empty"][:run_merged].mean()) < 0.5
    )
    filled = merged if heat_on else cls["fill"]

    covered = float((filled | cls["empty"] | cls["glow"]).mean())
    right_ok, vert_ok = _bounded(frame, box)

    fill_run = _leading_run(filled, gap_tol=heat_gap_tol if heat_on else gap_tol)
    empty_run = _trailing_run(cls["empty"])

    # --- 空血条 ---
    if fill_run == 0:
        empty_frac = float(cls["empty"].mean())
        if empty_frac < 0.75:
            return BarReading(None, empty_frac, f"no-fill-no-empty({empty_frac:.2f})")
        # 报"血量 0"是危险操作（会被当成死亡/击杀），边界必须双向确认
        if not (right_ok and vert_ok):
            return BarReading(None, empty_frac, "no-fill-unbounded")
        return BarReading(0.0, empty_frac, "empty")

    # --- 有填充 ---
    if covered < 0.85:
        return BarReading(None, covered, f"uncovered({covered:.2f})")
    if not (right_ok or vert_ok):
        return BarReading(None, covered, "unbounded")

    head_purity = float(filled[:fill_run].mean())
    if head_purity < 0.75:
        return BarReading(None, head_purity, f"head-impure({head_purity:.2f})")

    if heat_on:
        # 覆盖动画到底只盖"已填充"那段、还是连空轨道一起盖，样本里没拍到
        # （采样时 boss 恰好都接近满血）。所以特效期只在能被空区独立佐证时才报数：
        # 空区仍是黑的 -> 两个方向互相印证；整条都被染蓝而看不到空区 -> 无法区分
        # "真满血"和"动画盖住一切"，宁可丢帧也不能报出虚高的血量（那会变成
        # 血量回升的垃圾样本）。
        if empty_run == 0:
            return BarReading(None, heat_frac, f"heat-saturated({heat_frac:.2f})")
        empty_start = width - empty_run
        # 容差按边界渐变的实测宽度给：热血色会往已损失区糊进去十来列
        # （362 宽的玩家条上实测重叠 11 列 ≈ 3%），那是渐变不是矛盾。
        if fill_run > empty_start + max(8, width // 25):
            return BarReading(None, heat_frac, "heat-inconsistent")

    edge = float(fill_run)
    if empty_run > 0:
        empty_start = width - empty_run
        if empty_start >= fill_run:
            edge = 0.5 * (fill_run + empty_start)
    ratio = float(np.clip(edge / width, 0.0, 1.0))
    conf = float(np.clip(0.6 * head_purity + 0.4 * covered, 0.0, 1.0))
    return BarReading(ratio, conf, "heat-ok" if heat_on else "ok")


# ---------------------------------------------------------------- 定位
def _row_runs(row: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    if not row.any():
        return []
    d = np.diff(row.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if row[0]:
        starts.insert(0, 0)
    if row[-1]:
        ends.append(len(row))
    return [(s, e) for s, e in zip(starts, ends) if e - s >= min_len]


def _track_right_edge(
    empty_cols: np.ndarray, fill_x2: int, max_gap: int
) -> int:
    """
    轨道右端 = 填充右侧那段连续"已损失"区的末尾。
    填充和已损失之间允许 max_gap 列的前沿高亮。
    """
    n = len(empty_cols)
    x = fill_x2
    # 跳过前沿高亮，找到已损失区起点
    while x < n and not empty_cols[x] and x - fill_x2 <= max_gap:
        x += 1
    if x >= n or not empty_cols[x]:
        return fill_x2          # 满血：没有已损失区
    last = x
    gap = 0
    while x < n:
        if empty_cols[x]:
            last = x + 1
            gap = 0
        else:
            gap += 1
            if gap > 2:
                break
        x += 1
    return last


def find_tracks(frame: np.ndarray, target: BarTarget) -> List[Tuple[float, BarBox, str]]:
    """在 ROI 内找候选轨道，返回 [(score, box, note)]，分数降序。"""
    fh, fw = frame.shape[:2]
    rx1, ry1 = max(0, int(target.roi[0] * fw)), max(0, int(target.roi[1] * fh))
    rx2, ry2 = min(fw, int(target.roi[2] * fw)), min(fh, int(target.roi[3] * fh))
    if rx2 - rx1 < 24 or ry2 - ry1 < 4:
        return []

    roi = frame[ry1:ry2, rx1:rx2]
    fm = fill_mask(roi)
    em = empty_mask(roi)
    scale = fh / 720.0
    min_h = max(2, int(round(target.min_height * scale)))
    max_h = max(min_h + 1, int(round(target.max_height * scale)))
    min_track = max(24, int(round(target.min_track_ratio * fw)))
    max_track = int(round(target.max_track_ratio * fw))
    # 填充段最短：低血时红色很短，但定位阶段要求它够长才稳
    min_fill = max(8, int(round(min_track * 0.10)))
    max_gap = max(8, int(round(max_track * 0.15)))

    per_row: Dict[int, Tuple[int, int]] = {}
    for y in range(fm.shape[0]):
        runs = _row_runs(fm[y], min_fill)
        if runs:
            per_row[y] = max(runs, key=lambda r: r[1] - r[0])

    # 左端对齐的相邻行合并成带
    bands: List[Tuple[int, int, int, int]] = []
    ys = sorted(per_row)
    i = 0
    while i < len(ys):
        y = ys[i]
        x1, x2 = per_row[y]
        acc_x1, acc_x2 = [x1], [x2]
        j = i + 1
        while j < len(ys) and ys[j] == ys[j - 1] + 1:
            nx1, nx2 = per_row[ys[j]]
            if abs(nx1 - x1) > 3:
                break
            acc_x1.append(nx1)
            acc_x2.append(nx2)
            j += 1
        bands.append((y, ys[j - 1] + 1, int(np.median(acc_x1)), int(np.max(acc_x2))))
        i = j

    out: List[Tuple[float, BarBox, str]] = []
    for by1, by2, bx1, bx2 in bands:
        bh = by2 - by1
        if bh < min_h or bh > max_h:
            continue
        empty_cols = em[by1:by2].mean(axis=0) >= 0.5
        tx2 = _track_right_edge(empty_cols, bx2, max_gap)
        if tx2 - bx1 < min_track or tx2 - bx1 > max_track:
            continue
        box = BarBox(bx1 + rx1, by1 + ry1, tx2 + rx1, by2 + ry1)
        score, note = _score(frame, box, target, bx2 - bx1)
        if score > 0:
            out.append((score, box, note))

    out.sort(key=lambda t: t[0], reverse=True)
    return out


def _score(
    frame: np.ndarray, box: BarBox, target: BarTarget, fill_w: int
) -> Tuple[float, str]:
    fh, fw = frame.shape[:2]
    aspect = box.w / max(box.h, 1)
    if aspect < 10.0:
        return 0.0, "too-square"

    cls = _col_classes(_core_rows(frame, box))
    covered = float((cls["fill"] | cls["empty"] | cls["glow"]).mean())
    if covered < 0.9:
        return 0.0, f"uncovered({covered:.2f})"

    right_ok, vert_ok = _bounded(frame, box)
    if not (right_ok and vert_ok):
        return 0.0, f"unbounded(r={right_ok} v={vert_ok})"

    score = 0.0
    score += min(aspect, 70.0) * 0.5
    score += covered * 40.0
    score += (box.w / fw) * 20.0
    if target.expect_left is not None:
        err = abs(box.x1 / fw - target.expect_left)
        score += max(0.0, 1.0 - err * 6.0) * 30.0
    return score, f"a={aspect:.0f} cov={covered:.2f} fill={fill_w}"


# ---------------------------------------------------------------- 跟踪
class BarTracker:
    """
    一个血条目标的定位 + 读数。

    - 候选必须连续 `confirm_frames` 帧一致才锁定轨道
    - 轨道右端取历史最大：低血闪烁时看不到完整轨道，靠高血帧补全
    - 锁定后每帧只量填充长度；连续读不到超过 `relocate_after` 秒才解锁
    """

    def __init__(
        self,
        target: BarTarget,
        confirm_frames: int = 3,
        iou_thresh: float = 0.55,
        relocate_after: float = 20.0,
    ):
        self.target = target
        self.confirm_frames = confirm_frames
        self.iou_thresh = iou_thresh
        self.relocate_after = relocate_after

        self.box: Optional[BarBox] = None
        self.frame_size: Optional[Tuple[int, int]] = None
        self._pending: Optional[BarBox] = None
        self._pending_hits = 0
        self._repair_pending: Optional[Tuple[int, int]] = None
        self._good_reads = 0
        self._last_good = 0.0

    @property
    def locked(self) -> bool:
        return self.box is not None

    def reset(self):
        self.box = None
        self._pending = None
        self._pending_hits = 0
        self._repair_pending = None
        self._good_reads = 0

    def lock(self, frame: np.ndarray, box: BarBox):
        self.box = box
        self.frame_size = (frame.shape[1], frame.shape[0])
        self._last_good = time.time()
        self._good_reads = 0

    def rescale(self, new_size: Tuple[int, int]):
        if self.frame_size is None:
            self.frame_size = new_size
            return
        if self.frame_size == new_size:
            return
        if self.box is not None:
            self.box = self.box.scaled(self.frame_size, new_size)
        self.frame_size = new_size

    def _observe(self, box: BarBox) -> Optional[BarBox]:
        if self._pending is not None and self._pending.iou(box) >= self.iou_thresh:
            self._pending_hits += 1
            # 取并集：左端取最小、右端取最大。低血/受击帧看到的填充段是残缺的，
            # 只认第一帧会把轨道锁歪（实测锁成 x195..501，真值 x138..500）。
            self._pending = BarBox(
                min(self._pending.x1, box.x1),
                min(self._pending.y1, box.y1),
                max(self._pending.x2, box.x2),
                max(self._pending.y2, box.y2),
            )
        else:
            self._pending = box
            self._pending_hits = 1
        if self._pending_hits >= self.confirm_frames:
            confirmed = self._pending
            self._pending = None
            self._pending_hits = 0
            return confirmed
        return None

    def locate(self, frame: np.ndarray) -> Optional[BarBox]:
        cands = find_tracks(frame, self.target)
        if not cands:
            return None
        confirmed = self._observe(cands[0][1])
        if confirmed is not None:
            self.lock(frame, confirmed)
            return self.box
        return None

    def read(self, frame: np.ndarray, allow_locate: bool = True) -> BarReading:
        if frame is None:
            return BarReading(None, 0.0, "no-frame")
        self.rescale((frame.shape[1], frame.shape[0]))

        if self.box is None:
            if allow_locate:
                # 全黑画面（死亡/读盘）上没有 HUD，此时定位只会锁到噪声
                if not is_black_screen(frame):
                    self.locate(frame)
            if self.box is None:
                return BarReading(None, 0.0, "not-located")

        r = measure_fill(frame, self.box)
        if r.ok:
            self._last_good = time.time()
            self._good_reads += 1
            if allow_locate:
                self._repair(frame)
            return r

        # 解锁重搜是高风险操作：一旦在残缺画面上重锁，轨道就歪了
        # （实测锁成 x195..501）。死亡黑屏会连续几十秒读不到，
        # 那段时间**不能**计入重定位倒计时。
        if (
            allow_locate
            and not is_black_screen(frame)
            and time.time() - self._last_good > self.relocate_after
        ):
            print(f">> 血条轨道解锁重搜 [{self.target.name}]（已 {self.relocate_after:.0f}s 读不到）")
            self.reset()
        return r

    def _repair(self, frame: np.ndarray):
        """
        轨道锁窄了就地修正，直到撞上真正的边界。

        方向性很关键：
          - **左端只穿过填充色**。轨道左端就是填充的起点；若连"已损失"的
            纯黑也穿，会退到容器的黑色内边距里（Boss 条 x=315..319），
            于是填充不再从第 0 列开始，读数直接崩掉。
          - **右端只穿过已损失区**。锁窄时缺的那截必然是已损失部分。

        必须在探测范围内**收住**（`run < margin`）才采纳，否则说明那侧没有
        边界、只是背景偏暗。提议连续两帧一致才生效，避免单帧噪声污染轨道。

        只在轨道**尚未稳定**时修（`_good_reads < REPAIR_UNTIL_GOOD_READS`）。
        这条限制是必须的：`read()` 每次成功读数都会调用本方法，而 Boss 条右侧
        就是游戏背景，暗一点就会被判成"已损失"，于是右端一路外扩 ——
        实测 x965 → 970 → 1001 → 1041。轨道一变宽，同样的填充长度算出的血量
        就变小，纠回来又跳上去，直接被上层当成"血量回升"垃圾帧丢弃，
        实测 200 行日志里 86 次，吞吐从 7.4 步/s 掉到 2.3。
        另外单次修正的幅度也要收住，免得一步跨到背景里。
        """
        box = self.box
        if box is None or self._good_reads >= REPAIR_UNTIL_GOOD_READS:
            return
        fw = frame.shape[1]
        margin = max(8, int(box.w * REPAIR_MARGIN_RATIO))
        new_x1, new_x2 = box.x1, box.x2

        lo = max(0, box.x1 - margin)
        if lo < box.x1:
            cls = _col_classes(_core_rows(frame, BarBox(lo, box.y1, box.x1, box.y2)))
            run = _trailing_run(cls["fill"], gap_tol=1)
            if 0 < run < (box.x1 - lo):
                new_x1 = box.x1 - run

        hi = min(fw, box.x2 + margin)
        if hi > box.x2:
            cls = _col_classes(_core_rows(frame, BarBox(box.x2, box.y1, hi, box.y2)))
            run = _leading_run(cls["empty"], gap_tol=1)
            if 0 < run < (hi - box.x2):
                new_x2 = box.x2 + run

        if (new_x1, new_x2) == (box.x1, box.x2):
            self._repair_pending = None
            return
        proposal = (new_x1, new_x2)
        if self._repair_pending != proposal:
            self._repair_pending = proposal
            return
        self._repair_pending = None
        self.box = BarBox(new_x1, box.y1, new_x2, box.y2)
        print(
            f">> 血条轨道已修正 [{self.target.name}]: "
            f"x{box.x1}..{box.x2} → x{new_x1}..{new_x2}"
        )

    # -- 持久化 --
    def to_dict(self) -> Optional[Dict]:
        if self.box is None:
            return None
        return {"box": self.box.as_pairs(), "frame_size": list(self.frame_size or ())}

    def load_dict(self, data: Optional[Dict], frame_size: Tuple[int, int]):
        if not data:
            return
        try:
            box = BarBox.from_pairs(data["box"])
            fs = tuple(int(v) for v in data.get("frame_size", frame_size))
        except Exception:
            return
        if not box.valid():
            return
        self.frame_size = fs
        self.box = box
        self.rescale(frame_size)
        self._last_good = time.time()


# ---------------------------------------------------------------- 调试
def draw_debug(
    frame: np.ndarray,
    trackers: Dict[str, "BarTracker"],
    readings: Dict[str, BarReading],
) -> np.ndarray:
    img = frame.copy()
    colors = {"player": (0, 255, 0), "boss": (0, 128, 255)}
    for i, (name, tr) in enumerate(trackers.items()):
        col = colors.get(name, (255, 255, 255))
        r = readings.get(name)
        if tr.box is not None:
            b = tr.box
            cv2.rectangle(img, (b.x1 - 1, b.y1 - 1), (b.x2, b.y2), col, 1)
            if r is not None and r.ok:
                x = b.x1 + int(b.w * r.value)
                cv2.line(img, (x, b.y1 - 4), (x, b.y2 + 4), (0, 255, 255), 1)
        txt = f"{name}: " + ("None" if r is None or not r.ok else f"{r.value:.3f}")
        if r is not None:
            txt += f" ({r.reason})"
        cv2.putText(img, txt, (8, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, col, 1, cv2.LINE_AA)
    return img
