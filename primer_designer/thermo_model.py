"""三方竞争热力学模型：ΔG_target / ΔG_hairpin / ΔG_offtarget 计算与SI/EI评分

HEG修正版（含非核苷酸阻断基团熵罚）
============================================

三方竞争不等式:
    ΔG_target + HEG_PENALTY  ≪  ΔG_hairpin  <  ΔG_offtarget

各项定义：
  ΔG_target    = calc_heterodimer(primer_body, revcomp(primer_body)) + HEG_ENTROPY_PENALTY
                 仅对 primer_body 计算与目标模板的杂交能，再叠加 HEG 解折叠熵罚。
                 物理意义：目标模板需置换发夹茎部并跨越HEG柔性链才能与引物主体结合。

  ΔG_hairpin   = calc_hairpin(dna_seq)   其中 dna_seq = stem_comp + loop + primer_body
                 计算完整发夹引物（去除HEG占位符后的纯DNA序列）的分子内发夹折叠能，
                 不叠加HEG惩罚（HEG仅影响目标模板结合能，不影响自身折叠）。

  ΔG_offtarget = calc_heterodimer(primer_body, offtarget_window)
                 仅对 primer_body 与错配模板计算杂交能，无HEG惩罚
                 （错配结合不需要跨越HEG）。

SI统一版公式（三方竞争比值）:
    若 ΔG_offtarget ≥ 0：              SI = 5.0   (无脱靶竞争，自动 excellent)
    若 ΔG_hairpin  ≥ 0：              SI = 0.0   (发夹不稳定，reject)
    否则：
        SI = (ΔG_offtarget − ΔG_hairpin) / (ΔG_hairpin − ΔG_target)

  分子 (ΔG_offtarget − ΔG_hairpin) > 0：发夹对脱靶的"竞争优势"
  分母 (ΔG_hairpin  − ΔG_target)  > 0：目标模板对发夹的"置换优势"
  SI > 1：发夹阻断脱靶的能力 > 目标模板打开发夹的代价 → 特异性可靠
  SI > 1.5：优秀；0.8–1.5：可用；< 0.8：淘汰

  示例（典型值）：
    dg_target = -12.5 (20nt完美匹配 + HEG罚 1.8)
    dg_hairpin = -8   (6bp茎+4nt环 发夹域)
    dg_offtarget = -3 (2bp错配)
    分子 = -3 − (−8) = 5.0
    分母 = −8 − (−12.5) = 4.5
    SI = 5.0/4.5 ≈ 1.11 → acceptable

EI公式（不变）:
    EI = (ΔG_hairpin − ΔG_target) / |ΔG_target|
    EI > 1.0：优秀；0.5–1.0：可用；< 0.5：淘汰

硬约束：ΔG_target < ΔG_hairpin − TARGET_HAIRPIN_MARGIN_KCAL（=2.0 kcal/mol）
"""
from __future__ import annotations

from dataclasses import dataclass

import primer3
from Bio.Seq import Seq

ANNEAL_TEMP_C = 60.0               # 退火温度 (°C)
TARGET_HAIRPIN_MARGIN_KCAL = 2.0   # ΔG_target必须比ΔG_hairpin低至少2 kcal/mol
HEG_ENTROPY_PENALTY = 1.8          # HEG解折叠熵罚 (kcal/mol)，基于Scorpion文献

_SI_EXCELLENT, _SI_ACCEPTABLE = 1.5, 0.8
_EI_EXCELLENT, _EI_ACCEPTABLE = 1.0, 0.5
_SI_NO_OFFTARGET_CAP = 5.0
_VERDICT_RANK = {"reject": 0, "acceptable": 1, "excellent": 2}


def _revcomp(seq: str) -> str:
    return str(Seq(seq).reverse_complement())


def compute_dg_target(primer_body_seq: str) -> tuple[float, str]:
    """ΔG_target: primer_body与目标模板（完全互补链）的杂交自由能，叠加HEG熵罚。

    物理意义：目标模板需跨越HEG柔性链才能与primer_body结合，HEG引入额外熵代价。
    返回 (dg_kcal, ascii_structure)。
    """
    target_seq = _revcomp(primer_body_seq)
    r = primer3.bindings.calc_heterodimer(
        primer_body_seq, target_seq, temp_c=ANNEAL_TEMP_C, output_structure=True
    )
    dg = r.dg / 1000.0 + HEG_ENTROPY_PENALTY
    return dg, r.ascii_structure


def compute_dg_hairpin(dna_seq: str) -> tuple[float, str]:
    """ΔG_hairpin: 完整DNA序列（stem_comp+loop+primer_body）的分子内发夹折叠自由能。

    传入 dna_seq（不含HEG占位符的纯核苷酸串），由primer3计算最稳定发夹结构。
    不叠加HEG惩罚（HEG仅在目标模板结合时起熵代价作用）。
    返回 (dg_kcal, ascii_structure)。
    """
    r = primer3.bindings.calc_hairpin(dna_seq, temp_c=ANNEAL_TEMP_C, output_structure=True)
    return r.dg / 1000.0, r.ascii_structure


def compute_dg_homodimer(dna_seq: str) -> float:
    """ΔG of two copies of the primer (DNA-only seq) annealing to each other.

    Values more negative than -5 kcal/mol → meaningful primer-dimer risk.
    Values < -8 kcal/mol → high risk.
    """
    r = primer3.bindings.calc_heterodimer(dna_seq, dna_seq, temp_c=ANNEAL_TEMP_C)
    return r.dg / 1000.0


def compute_dg_offtarget(primer_body_seq: str, offtarget_window: str | None) -> tuple[float, str]:
    """ΔG_offtarget: primer_body与非目标模板窗口（含1-3bp错配）的杂交自由能。

    错配结合无需跨越HEG，不叠加HEG惩罚。
    无显著脱靶位点时返回 (0.0, "")。
    返回 (dg_kcal, ascii_structure)。
    """
    if offtarget_window is None:
        return 0.0, ""
    r = primer3.bindings.calc_heterodimer(
        primer_body_seq, offtarget_window, temp_c=ANNEAL_TEMP_C, output_structure=True
    )
    return r.dg / 1000.0, r.ascii_structure


@dataclass
class SiEiResult:
    si: float
    ei: float
    verdict_si: str
    verdict_ei: str
    overall_verdict: str


def compute_si_ei(dg_target: float, dg_hairpin: float, dg_offtarget: float) -> SiEiResult:
    """三方竞争SI/EI计算（统一版公式，含HEG修正后的dg_target）。

    SI = (ΔG_offtarget − ΔG_hairpin) / (ΔG_hairpin − ΔG_target)
    EI = (ΔG_hairpin − ΔG_target) / |ΔG_target|
    """
    if dg_offtarget >= 0:
        si = _SI_NO_OFFTARGET_CAP
    elif dg_hairpin >= 0:
        si = 0.0
    else:
        denom = dg_hairpin - dg_target
        if denom <= 0:
            # target更稳定或与hairpin相当，EI有效但SI分母异常，取封顶
            si = _SI_NO_OFFTARGET_CAP
        else:
            si = (dg_offtarget - dg_hairpin) / denom

    if dg_target == 0:
        ei = -999.0
    else:
        ei = (dg_hairpin - dg_target) / abs(dg_target)

    verdict_si = (
        "excellent" if si > _SI_EXCELLENT else "acceptable" if si > _SI_ACCEPTABLE else "reject"
    )
    verdict_ei = (
        "excellent" if ei > _EI_EXCELLENT else "acceptable" if ei > _EI_ACCEPTABLE else "reject"
    )
    overall = min(verdict_si, verdict_ei, key=lambda v: _VERDICT_RANK[v])

    return SiEiResult(si=si, ei=ei, verdict_si=verdict_si, verdict_ei=verdict_ei, overall_verdict=overall)
