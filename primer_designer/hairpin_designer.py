"""Module 2: 发夹阻断域设计 —— 茎长(4-6bp)/环长(3-5nt)/环序列搜索（含HEG阻断基团）

引物结构（含HEG）：
    5'-[stem_comp]-[loop]-[HEG]-[primer_body]-3'

其中：
  stem_comp = revcomp(primer_body[-stem_len:])   拼接在5'端
  loop      = 3-5 nt环序列（过滤GGG/CCC/回文）
  HEG       = 六乙二醇非核苷酸连接子（阻断聚合酶read-through）
  primer_body = 原始引物主体序列（不含发夹域）

HEG的作用：
  1. 阻断聚合酶从5'端发夹域read-through进入引物主体（消除self-priming）
  2. 对目标模板结合引入熵罚（+1.8 kcal/mol，见thermo_model.HEG_ENTROPY_PENALTY）
  3. 对错配脱靶结合无惩罚（脱靶不需跨越HEG）

热力学分量（均来自thermo_model）：
  dg_hairpin   = calc_hairpin(dna_seq)         dna_seq = stem_comp+loop+primer_body
  dg_target    = calc_heterodimer(body, revcomp(body)) + HEG_PENALTY
  dg_offtarget = calc_heterodimer(body, offtarget_window)    （无HEG罚）
  dg_homodimer = calc_heterodimer(dna_seq, dna_seq)

hairpin_primer_seq 字段为展示字符串：
    stem_comp + loop + "[HEG]" + primer_body

design_hairpin_blocker_ai 是该搜索的"AI for Science"加速版本：用
gnn_model中的结构感知GNN代理模型在不设环序列数量上限的扩展候选空间中
先筛选ΔG_hairpin，再由primer3精确复算，详见该函数docstring。
"""
from __future__ import annotations

from dataclasses import dataclass, field
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

STEM_LEN_RANGE = range(4, 7)   # 4-6 bp
LOOP_LEN_RANGE = range(3, 6)   # 3-5 nt
MAX_LOOP_CANDIDATES = 10
AI_TOP_K = 20

_BASES = "ACGT"


def _revcomp(seq: str) -> str:
    return str(Seq(seq).reverse_complement())


def _is_valid_loop(seq: str) -> bool:
    """过滤含GGG/CCC（G四链体风险）及回文（环内自配对风险）的环序列"""
    if "GGG" in seq or "CCC" in seq:
        return False
    if seq == _revcomp(seq):
        return False
    return True


def generate_loop_sequences(length: int, max_candidates: int = MAX_LOOP_CANDIDATES) -> list[str]:
    """枚举长度为length的环序列，过滤后最多返回max_candidates个"""
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
    """枚举长度为length的全部环序列（不设数量上限），供AI搜索使用"""
    return [
        "".join(combo)
        for combo in product(_BASES, repeat=length)
        if _is_valid_loop("".join(combo))
    ]


_HOMODIMER_WARN = -5.0
_HOMODIMER_HIGH = -8.0
_ON_TARGET_RISK_MEDIUM_DG = -5.0
_ON_TARGET_RISK_HIGH_DG = -8.0
_SCORE_PENALTY_HOMODIMER = 2.0
_SCORE_PENALTY_ONTARGET_HIGH = 2.0


@dataclass
class HairpinDesign:
    stem_len: int
    loop_len: int
    loop_seq: str
    # 序列字段
    primer_body_seq: str       # 原始引物主体序列（不含发夹域）
    hairpin_domain_seq: str    # stem_comp + loop_seq（发夹阻断域，纯核苷酸）
    dna_seq: str               # stem_comp + loop + primer_body（全DNA，供primer3计算发夹/同源二聚体）
    hairpin_primer_seq: str    # 展示字符串：stem_comp+loop+"[HEG]"+primer_body
    # 热力学字段
    dg_target: float           # primer_body与目标杂交ΔG（含HEG熵罚）
    dg_hairpin: float          # 完整DNA序列发夹折叠ΔG（不含HEG罚）
    dg_offtarget: float        # primer_body与错配模板杂交ΔG（无HEG罚）
    si: float
    ei: float
    verdict_si: str
    verdict_ei: str
    overall_verdict: str
    ascii_hairpin: str
    ascii_target: str
    ascii_offtarget: str
    dg_homodimer: float = 0.0
    on_target_risk: str = "low"


def _on_target_risk(dg_hairpin: float) -> str:
    if dg_hairpin > _ON_TARGET_RISK_MEDIUM_DG:
        return "low"
    if dg_hairpin > _ON_TARGET_RISK_HIGH_DG:
        return "medium"
    return "high"


def _evaluate_candidate(
    stem_len: int, loop_len: int, loop_seq: str, primer_seq: str, offtarget_window: str | None
) -> HairpinDesign:
    stem_comp = _revcomp(primer_seq[-stem_len:])
    hairpin_domain = stem_comp + loop_seq                    # 纯发夹阻断域（HEG前的部分）
    dna_seq = hairpin_domain + primer_seq                    # 全DNA序列（无HEG）供primer3使用
    hairpin_primer_seq = stem_comp + loop_seq + "[HEG]" + primer_seq  # 展示字符串

    # 各分量热力学计算（函数签名已更新至HEG修正版）
    dg_target, ascii_target = compute_dg_target(primer_seq)              # body only + HEG penalty
    dg_hairpin, ascii_hairpin = compute_dg_hairpin(dna_seq)              # full DNA, no HEG penalty
    dg_offtarget, ascii_offtarget = compute_dg_offtarget(primer_seq, offtarget_window)  # body only
    si_ei = compute_si_ei(dg_target, dg_hairpin, dg_offtarget)
    dg_homo = compute_dg_homodimer(dna_seq)                              # full DNA
    risk = _on_target_risk(dg_hairpin)

    return HairpinDesign(
        stem_len=stem_len,
        loop_len=loop_len,
        loop_seq=loop_seq,
        primer_body_seq=primer_seq,
        hairpin_domain_seq=hairpin_domain,
        dna_seq=dna_seq,
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
    """从候选设计中选最优。

    主打分（满足约束 dg_target < dg_hairpin - 2.0 时）：score = dg_offtarget - dg_hairpin
    扣分：primer-dimer风险（dg_homodimer < -5）、高on-target风险（dg_hairpin < -8）
    无满足约束的设计时，回退为SI最高的候选。
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

        if design.dg_homodimer < _HOMODIMER_WARN:
            score -= _SCORE_PENALTY_HOMODIMER
        if design.dg_hairpin < _ON_TARGET_RISK_HIGH_DG:
            score -= _SCORE_PENALTY_ONTARGET_HIGH

        if constraint_ok and score > best_score:
            best_score = score
            best = design

    result = best if best is not None else fallback
    assert result is not None
    return result


def design_hairpin_blocker(primer_seq: str, offtarget_window: str | None) -> HairpinDesign:
    """遍历茎长/环长/环序列组合，返回_select_best选出的最优发夹阻断设计"""
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
    """AI加速搜索：GNN代理模型预筛选dg_hairpin，primer3精确复算Top-K候选。

    GNN输入为全DNA序列（stem_comp+loop+primer_body），dot_bracket由设计参数构造。
    surrogate为None时回退到精确穷举。
    """
    if surrogate is None:
        return design_hairpin_blocker(primer_seq, offtarget_window)

    from .thermo_model import ANNEAL_TEMP_C
    from .gnn_model import design_dotbracket

    # 候选列表: (stem_len, loop_len, loop_seq, dna_seq)
    candidates: list[tuple[int, int, str, str]] = []
    for stem_len in STEM_LEN_RANGE:
        if stem_len >= len(primer_seq):
            continue
        stem_comp = _revcomp(primer_seq[-stem_len:])
        for loop_len in LOOP_LEN_RANGE:
            for loop_seq in generate_loop_sequences_all(loop_len):
                dna_seq = stem_comp + loop_seq + primer_seq
                candidates.append((stem_len, loop_len, loop_seq, dna_seq))

    if not candidates:
        return design_hairpin_blocker(primer_seq, offtarget_window)

    # GNN批量预测 dg_hairpin（使用全DNA序列和对应dot_bracket）
    pred_dg = np.array([
        surrogate.predict_dg(
            c[3],
            temp_c=ANNEAL_TEMP_C,
            dot_bracket=design_dotbracket(len(c[3]), c[0], c[1]),
        )
        for c in candidates
    ])

    # 阈值：dg_target（body only + HEG penalty）+ 2.0 kcal/mol 裕量
    dg_target_est, _ = compute_dg_target(primer_seq)
    threshold = dg_target_est + TARGET_HAIRPIN_MARGIN_KCAL

    order = np.argsort(pred_dg)
    qualifying = [i for i in order if pred_dg[i] > threshold]
    non_qualifying = [i for i in order[::-1] if pred_dg[i] <= threshold]
    shortlist = (qualifying + non_qualifying)[:top_k]

    designs = [_evaluate_candidate(*candidates[i][:3], primer_seq, offtarget_window) for i in shortlist]
    return _select_best(designs)
