"""扩增子顾问 (Amplicon Advisor) —— 滑动窗口扫描，自动推荐最优扩增区域

将整条目标序列拆分为滑动窗口，对每个候选扩增子从以下四个维度打分：

  1. GC均匀度 (gc_score)
     理想GC%: 45–55%；GC偏高/偏低 → 扣分；局部GC极端段 → 额外扣分。

  2. 引物结合区结构简单性 (struct_score)
     对窗口两端各30 bp用primer3计算发夹ΔG：结构越稳定 → 引物越难结合 → 扣分。
     这直接预测了该区域能否形成良好的引物结合位点。

  3. 特异性 (spec_score)
     窗口两端种子序列在非目标库中的最短Hamming距离：越相似 → 扣分。
     复用 offtarget_finder 的现有逻辑，无额外依赖。

  4. 复杂度/重复序列惩罚 (complexity_score)
     检测连续同碱基延伸(≥5)、简单重复(dinucleotide repeat ≥ 4次)；
     这类区域引物难以准确结合，且易引发滑动错配。

综合得分 = 0.35×gc + 0.35×struct + 0.20×spec + 0.10×complexity

返回 AmpliconCandidate 列表，供 GUI 展示热图 + 可排序候选列表。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import primer3


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class AmpliconCandidate:
    """单个候选扩增子窗口的评分结果。"""
    start: int            # 0-based，扩增子在模板中的起始位置
    end: int              # 0-based exclusive，结束位置
    size: int             # 扩增子长度 (bp)
    gc_pct: float         # 窗口整体GC%
    gc_score: float       # [0, 1] GC均匀度得分
    struct_score: float   # [0, 1] 引物结合区结构简单性
    spec_score: float     # [0, 1] 相对于非目标库的特异性
    complexity_score: float  # [0, 1] 无重复/低复杂度序列惩罚
    composite: float      # [0, 1] 综合得分
    window_seq: str       # 扩增子序列本身

    @property
    def rank_str(self) -> str:
        return f"{self.composite:.3f}"


# ---------------------------------------------------------------------------
# 评分函数
# ---------------------------------------------------------------------------

def _gc(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(1 for c in seq.upper() if c in "GC") / len(seq)


def _gc_score(seq: str) -> float:
    """GC均匀度得分：整体GC%接近50%且各段均匀 → 高分。"""
    n = len(seq)
    if n == 0:
        return 0.0

    gc_total = _gc(seq)
    # 整体偏离理想值 (45–55%) 的惩罚
    ideal_center = 0.50
    deviation = abs(gc_total - ideal_center)
    base_score = max(0.0, 1.0 - deviation / 0.30)  # 偏差30%时得0分

    # 局部均匀性：将窗口分4段，各段GC标准差越大扣分越多
    k = min(4, max(1, n // 30))
    seg_len = n // k
    gc_segs = [_gc(seq[i * seg_len: (i + 1) * seg_len]) for i in range(k)]
    import statistics
    std = statistics.pstdev(gc_segs) if len(gc_segs) > 1 else 0.0
    uniformity = max(0.0, 1.0 - std / 0.20)  # 标准差20%时得0分

    return 0.6 * base_score + 0.4 * uniformity


def _struct_score(seq: str, temp_c: float = 60.0) -> float:
    """引物结合区结构简单性得分：两端30 bp的发夹ΔG越稳定越扣分。"""
    probe = 30
    regions = []
    if len(seq) >= probe:
        regions.append(seq[:probe])        # 5' 端（正向引物结合区）
        regions.append(seq[-probe:])       # 3' 端（反向引物结合区）
    else:
        regions.append(seq)

    penalties = []
    for r in regions:
        try:
            res = primer3.bindings.calc_hairpin(r, temp_c=temp_c)
            dg = res.dg / 1000.0  # kcal/mol
            # dg < 0 = stable hairpin = bad; dg > 0 = no hairpin = good
            # penalty: 0 when dg >= 0, scales to 1.0 when dg = -8 kcal/mol
            p = max(0.0, min(1.0, -dg / 8.0))
            penalties.append(p)
        except Exception:
            penalties.append(0.0)

    avg_penalty = sum(penalties) / len(penalties)
    return max(0.0, 1.0 - avg_penalty)


def _spec_score(
    seq: str,
    offtarget_records: list[tuple[str, str]],
    seed_len: int = 10,
) -> float:
    """特异性得分：窗口两端种子与非目标库的最短Hamming距离。

    距离越短（越相似）→ 特异性越差 → 得分越低。
    无非目标库时返回1.0（视为特异性满分）。
    """
    if not offtarget_records:
        return 1.0

    from .offtarget_finder import find_offtarget_window

    # 用两端种子检测特异性
    seeds = []
    if len(seq) >= seed_len:
        seeds.append(seq[:20])   # 5'端20nt作为正向引物代表
        seeds.append(seq[-20:])  # 3'端20nt作为反向引物代表

    min_dist = seed_len  # 初始化为最大值（完全不匹配）
    for seed in seeds:
        window = find_offtarget_window(seed, offtarget_records, max_mismatch=3, seed_len=seed_len)
        if window is not None:
            # 找到匹配 → 特异性下降（我们不精算距离，有匹配就记为低特异性）
            min_dist = min(min_dist, 0)
        else:
            min_dist = min(min_dist, seed_len)  # 未找到匹配 → 视为最大距离

    # 没有任何offtarget命中时 min_dist仍是seed_len → 满分
    return min(1.0, min_dist / seed_len)


def _complexity_score(seq: str) -> float:
    """复杂度得分：惩罚连续同碱基(≥5) 和简单二核苷酸重复(≥4次)。"""
    seq = seq.upper()
    penalty = 0.0

    # 连续同碱基延伸
    for match in re.finditer(r"(A{5,}|T{5,}|G{5,}|C{5,})", seq):
        run_len = len(match.group())
        penalty += (run_len - 4) * 0.05  # 超出4个每多1个扣5%

    # 简单二核苷酸重复 (AT, GC, AC, GT…)
    for dinuc in ["AT", "TA", "GC", "CG", "AC", "CA", "GT", "TG"]:
        pattern = f"({dinuc}){{4,}}"
        for match in re.finditer(pattern, seq):
            repeats = len(match.group()) // 2
            penalty += (repeats - 3) * 0.08

    return max(0.0, 1.0 - min(penalty, 1.0))


# ---------------------------------------------------------------------------
# 主扫描函数
# ---------------------------------------------------------------------------

def scan_amplicons(
    template: str,
    product_min: int = 100,
    product_max: int = 300,
    step: int = 10,
    temp_c: float = 60.0,
    offtarget_records: Optional[list[tuple[str, str]]] = None,
    top_n: int = 50,
    progress_callback=None,
) -> list[AmpliconCandidate]:
    """扫描模板序列，对所有候选扩增子窗口打分，返回按综合得分降序排列的列表。

    参数:
      template:           目标模板序列（纯碱基字符串）
      product_min/max:    扩增子大小范围 (bp)
      step:               滑动步长 (bp)，越小越精细但越慢
      temp_c:             退火温度，用于结构评分
      offtarget_records:  [(id, seq), ...] 非目标模板库
      top_n:              返回前N个候选
      progress_callback:  可选回调函数(done, total)，用于GUI进度更新
    """
    offtarget_records = offtarget_records or []
    template = template.upper()
    n = len(template)

    # 以中等窗口大小扫描（产物大小取中间值，减少计算量）
    target_size = (product_min + product_max) // 2

    candidates: list[AmpliconCandidate] = []
    positions = range(0, n - target_size + 1, step)
    total = len(positions)

    for done, start in enumerate(positions):
        end = start + target_size
        if end > n:
            break
        window = template[start:end]

        gc_s = _gc_score(window)
        st_s = _struct_score(window, temp_c)
        sp_s = _spec_score(window, offtarget_records)
        cx_s = _complexity_score(window)

        composite = 0.35 * gc_s + 0.35 * st_s + 0.20 * sp_s + 0.10 * cx_s

        candidates.append(AmpliconCandidate(
            start=start,
            end=end,
            size=target_size,
            gc_pct=_gc(window),
            gc_score=gc_s,
            struct_score=st_s,
            spec_score=sp_s,
            complexity_score=cx_s,
            composite=composite,
            window_seq=window,
        ))

        if progress_callback and (done % 20 == 0 or done == total - 1):
            progress_callback(done + 1, total)

    candidates.sort(key=lambda c: c.composite, reverse=True)
    return candidates[:top_n]


def per_position_scores(candidates: list[AmpliconCandidate], seq_len: int) -> list[float]:
    """将候选窗口得分映射到每个序列位置（取覆盖该位置的所有窗口中的最高分）。

    用于GUI热图：输出长度=seq_len的列表，方便按位置着色。
    """
    scores = [0.0] * seq_len
    for c in candidates:
        for i in range(c.start, min(c.end, seq_len)):
            if c.composite > scores[i]:
                scores[i] = c.composite
    return scores
