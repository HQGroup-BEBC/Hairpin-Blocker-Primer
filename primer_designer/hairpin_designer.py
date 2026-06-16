"""Module 2: 发夹阻断域设计 —— 茎长(4-6bp)/环长(3-5nt)/环序列搜索

对每条引物，在其5'端拼接 `stem_comp + loop_seq`（stem_comp与引物自身3'端
互补，详见thermo_model模块说明），遍历茎长/环长/环序列组合，选出最能
区分目标与脱靶结合的发夹阻断设计。

design_hairpin_blocker_ai 是该搜索的"AI for Science"加速版本：用
gnn_model中的结构感知GNN代理模型（训练时使用NNN数据集TargetStruct点括号作为
氢键边，推断时由stem_len/loop_len参数构造点括号，零NUPACK运行时依赖）
在不设环序列数量上限的扩展候选空间中先筛选，再由primer3精确复算
筛选出的候选，详见该函数docstring。
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from Bio.Seq import Seq

from .thermo_model import (
    TARGET_HAIRPIN_MARGIN_KCAL,
    compute_dg_hairpin,
    compute_dg_homodimer,
    compute_dg_offtarget,
    compute_dg_target,
    compute_si_ei,
)

STEM_LEN_RANGE = range(4, 7)  # 4-6 bp（用户建议从4-6 bp开始；茎过长→发夹过稳→on-target扩增效率风险）
LOOP_LEN_RANGE = range(3, 6)  # 3-5 nt
MAX_LOOP_CANDIDATES = 10
AI_TOP_K = 20

_BASES = "ACGT"


def _revcomp(seq: str) -> str:
    return str(Seq(seq).reverse_complement())


def _is_valid_loop(seq: str) -> bool:
    """过滤含GGG/CCC（可能形成G-四链体/额外结构）及自身为回文
    （可能在环内自我配对）的环序列"""
    if "GGG" in seq or "CCC" in seq:
        return False
    if seq == _revcomp(seq):
        return False
    return True


def generate_loop_sequences(length: int, max_candidates: int = MAX_LOOP_CANDIDATES) -> list[str]:
    """枚举长度为length的环序列，过滤规则见_is_valid_loop，最多返回
    max_candidates个"""
    candidates: list[str] = []
    for combo in product(_BASES, repeat=length):
        seq = "".join(combo)
        if not _is_valid_loop(seq):
            continue
        candidates.append(seq)
        if len(candidates) >= max_candidates:
            break
    return candidates


def generate_loop_sequences_all(length: int) -> list[str]:
    """枚举长度为length的全部环序列（不设数量上限），过滤规则与
    generate_loop_sequences相同。供design_hairpin_blocker_ai的扩展搜索使用"""
    return [
        "".join(combo)
        for combo in product(_BASES, repeat=length)
        if _is_valid_loop("".join(combo))
    ]


_HOMODIMER_WARN = -5.0   # kcal/mol，两份发夹引物互聚体警戒线（primer-dimer风险）
_HOMODIMER_HIGH = -8.0   # kcal/mol，强二聚体高风险线

_ON_TARGET_RISK_MEDIUM_DG = -5.0  # 发夹ΔG低于此值进入中等on-target风险
_ON_TARGET_RISK_HIGH_DG = -8.0    # 发夹ΔG低于此值进入高on-target风险（发夹难以被目标模板打开）

_SCORE_PENALTY_HOMODIMER = 2.0   # 强二聚体设计评分惩罚
_SCORE_PENALTY_ONTARGET_HIGH = 2.0  # 高on-target风险设计评分惩罚


@dataclass
class HairpinDesign:
    stem_len: int
    loop_len: int
    loop_seq: str
    hairpin_primer_seq: str
    dg_target: float
    dg_hairpin: float
    dg_offtarget: float
    si: float
    ei: float
    verdict_si: str
    verdict_ei: str
    overall_verdict: str
    ascii_hairpin: str
    ascii_target: str
    ascii_offtarget: str
    # 新增：自延伸/primer-dimer风险字段
    dg_homodimer: float = 0.0      # 两份发夹引物互聚体ΔG（Cordaro 2021 PCR效率预测特征）
    on_target_risk: str = "low"    # 'low'/'medium'/'high'——发夹过稳可能阻碍目标模板打开


def _on_target_risk(dg_hairpin: float) -> str:
    """分类on-target扩增效率风险：发夹过稳时目标模板难以置换发夹（动力学捕获），
    导致Ct延后/斜率变差/低拷贝检出失败/标准曲线效率<90%。
    分级阈值参考Cordaro et al. 2021 (PCR_ML_model) PCR效率预测分析。"""
    if dg_hairpin > _ON_TARGET_RISK_MEDIUM_DG:
        return "low"
    if dg_hairpin > _ON_TARGET_RISK_HIGH_DG:
        return "medium"
    return "high"


def _evaluate_candidate(
    stem_len: int, loop_len: int, loop_seq: str, primer_seq: str, offtarget_window: str | None
) -> HairpinDesign:
    stem_comp = _revcomp(primer_seq[-stem_len:])
    hairpin_primer_seq = stem_comp + loop_seq + primer_seq

    # 5'端无单链悬挂（self-extension防护）：stem_comp从位置0开始，3'末端与5'末端配对
    # 聚合酶无法沿5'→3'方向在5'端末端之外读取模板，故自延伸被结构本身封堵。

    dg_target, ascii_target = compute_dg_target(hairpin_primer_seq, primer_seq)
    dg_hairpin, ascii_hairpin = compute_dg_hairpin(hairpin_primer_seq)
    dg_offtarget, ascii_offtarget = compute_dg_offtarget(hairpin_primer_seq, offtarget_window)
    si_ei = compute_si_ei(dg_target, dg_hairpin, dg_offtarget)
    dg_homo = compute_dg_homodimer(hairpin_primer_seq)
    risk = _on_target_risk(dg_hairpin)

    return HairpinDesign(
        stem_len=stem_len,
        loop_len=loop_len,
        loop_seq=loop_seq,
        hairpin_primer_seq=hairpin_primer_seq,
        dg_target=dg_target,
        dg_hairpin=dg_hairpin,
        dg_offtarget=dg_offtarget,
        si=si_ei.si,
        ei=si_ei.ei,
        verdict_si=si_ei.verdict_si,
        verdict_ei=si_ei.verdict_ei,
        overall_verdict=si_ei.overall_verdict,
        ascii_hairpin=ascii_hairpin,
        ascii_target=ascii_target,
        ascii_offtarget=ascii_offtarget,
        dg_homodimer=dg_homo,
        on_target_risk=risk,
    )


def _select_best(designs: list[HairpinDesign]) -> HairpinDesign:
    """从候选设计列表中选择最优者。

    主打分：满足约束 ΔG_target < ΔG_hairpin - 2.0 时，score = ΔG_offtarget - ΔG_hairpin
    （脱靶相对发夹结合越弱越好）。

    扣分惩罚（均为减分，不淘汰候选）：
      - primer-dimer风险：dg_homodimer < -5 kcal/mol → 扣 _SCORE_PENALTY_HOMODIMER
        （两份引物互聚体 → PCR效率下降，Cordaro 2021）
      - on-target高风险：dg_hairpin < -8 kcal/mol → 扣 _SCORE_PENALTY_ONTARGET_HIGH
        （发夹过稳 → 目标模板难以动力学置换 → Ct延后/低拷贝检出失败）

    若没有组合满足约束，回退为SI最高的组合（overall_verdict='reject'）。
    """
    best: HairpinDesign | None = None
    best_score = float("-inf")
    fallback: HairpinDesign | None = None
    fallback_si = float("-inf")

    for design in designs:
        if design.si > fallback_si:
            fallback_si = design.si
            fallback = design

        constraint_ok = design.dg_target < design.dg_hairpin - TARGET_HAIRPIN_MARGIN_KCAL
        score = design.dg_offtarget - design.dg_hairpin

        # Penalty: primer-dimer risk
        if design.dg_homodimer < _HOMODIMER_WARN:
            score -= _SCORE_PENALTY_HOMODIMER
        # Penalty: on-target risk (over-stable hairpin resists template opening)
        if design.dg_hairpin < _ON_TARGET_RISK_HIGH_DG:
            score -= _SCORE_PENALTY_ONTARGET_HIGH

        if constraint_ok and score > best_score:
            best_score = score
            best = design

    result = best if best is not None else fallback
    assert result is not None  # 至少有一个候选
    return result


def design_hairpin_blocker(primer_seq: str, offtarget_window: str | None) -> HairpinDesign:
    """遍历茎长/环长/环序列组合（环序列截断至MAX_LOOP_CANDIDATES个），
    返回_select_best选出的最优发夹阻断设计"""
    designs: list[HairpinDesign] = []
    for stem_len in STEM_LEN_RANGE:
        if stem_len >= len(primer_seq):
            continue
        for loop_len in LOOP_LEN_RANGE:
            for loop_seq in generate_loop_sequences(loop_len):
                designs.append(_evaluate_candidate(stem_len, loop_len, loop_seq, primer_seq, offtarget_window))
    return _select_best(designs)


def design_hairpin_blocker_ai(
    primer_seq: str,
    offtarget_window: str | None,
    surrogate,
    top_k: int = AI_TOP_K,
) -> HairpinDesign:
    """AI加速搜索（"AI for Science"组件）：用StructureFreeGNN代理模型
    (gnn_model.GNNSurrogate)在完整的stem(4-8bp)/loop(3-5nt)候选空间——
    环序列不设MAX_LOOP_CANDIDATES数量上限，候选总数约扩大30倍——中按
    预测ΔG_hairpin(60°C)批量筛选出top_k个候选，再由primer3精确复算这些
    候选的ΔG_target/ΔG_hairpin/ΔG_offtarget/SI/EI，最终用_select_best
    规则选出最优设计。GNN仅作预筛选，primer3给出最终热力学数值。

    surrogate为None时回退到design_hairpin_blocker（精确穷举）。
    """
    if surrogate is None:
        return design_hairpin_blocker(primer_seq, offtarget_window)

    from .thermo_model import ANNEAL_TEMP_C

    candidates: list[tuple[int, int, str, str]] = []
    for stem_len in STEM_LEN_RANGE:
        if stem_len >= len(primer_seq):
            continue
        stem_comp = _revcomp(primer_seq[-stem_len:])
        for loop_len in LOOP_LEN_RANGE:
            for loop_seq in generate_loop_sequences_all(loop_len):
                candidates.append((stem_len, loop_len, loop_seq, stem_comp + loop_seq + primer_seq))

    if not candidates:
        return design_hairpin_blocker(primer_seq, offtarget_window)

    from .gnn_model import design_dotbracket

    pred_dg = np.array([
        surrogate.predict_dg(
            c[3],
            temp_c=ANNEAL_TEMP_C,
            dot_bracket=design_dotbracket(len(c[3]), c[0], c[1]),
        )
        for c in candidates
    ])

    ref_hairpin_seq = candidates[0][3]
    dg_target_est, _ = compute_dg_target(ref_hairpin_seq, primer_seq)
    threshold = dg_target_est + TARGET_HAIRPIN_MARGIN_KCAL

    order = np.argsort(pred_dg)
    qualifying = [i for i in order if pred_dg[i] > threshold]
    non_qualifying = [i for i in order[::-1] if pred_dg[i] <= threshold]
    shortlist = (qualifying + non_qualifying)[:top_k]

    designs = [_evaluate_candidate(*candidates[i][:3], primer_seq, offtarget_window) for i in shortlist]
    return _select_best(designs)
