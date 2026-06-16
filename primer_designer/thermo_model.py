"""三方竞争热力学模型：ΔG_target / ΔG_hairpin / ΔG_offtarget 计算与SI/EI评分

ΔG计算全部基于primer3-py（calc_hairpin / calc_heterodimer），退火温度取
ANNEAL_TEMP_C（℃），单位统一换算为 kcal/mol（primer3-py原始输出单位为cal/mol）。

SI/EI公式说明（与用户确认后的最终版本）
------------------------------------
原始研究方案中的不等式 ΔG_target ≪ ΔG_hairpin < ΔG_offtarget 采用标准ΔG
约定（数值越负，结合越稳定）。围绕该不等式直接构造的"差值比"公式，在用
真实primer3-py数据验证时发现规模不匹配：ΔG_target（完整20bp双链，约
-9~-15 kcal/mol）的量级是发夹在该拓扑下可达到的ΔG_hairpin（约
-1~-3.4 kcal/mol，因5'拼接的茎-环结构必然带有一个跨越整条引物主体的大环，
熵罚分几乎抵消了茎的配对自由能）的4-6倍，导致任何把ΔG_target直接放进SI
分母/分子的公式都会把SI压缩到远低于0.8的范围，使"acceptable/excellent"
阈值永远不可达。

最终采用的公式改为"发夹 vs 脱靶"的直接竞争比值，不再把ΔG_target放入SI：
    SI = ΔG_hairpin / ΔG_offtarget          （两者均为负值时，比值为正）
    EI = (ΔG_hairpin - ΔG_target) / |ΔG_target|   （不变）
ΔG_target仍通过独立约束 "ΔG_target < ΔG_hairpin - 2.0" 起作用（确保目标
模板始终能打开发夹），并通过EI衡量该约束的"富余量"。

SI的含义：
- ΔG_offtarget >= 0（库中无显著脱靶位点，或脱靶结合本身不稳定）：视为"无
  竞争"，SI取一个大于excellent阈值的封顶值 `_SI_NO_OFFTARGET_CAP`
- ΔG_hairpin >= 0（该stem/loop组合在该温度下不形成稳定发夹）且存在真实
  脱靶位点：发夹不提供任何竞争性保护，SI <= 0 → reject
- 两者均为负值：SI = |ΔG_hairpin| / |ΔG_offtarget|，>1表示发夹结合强于
  脱靶结合（脱靶被发夹竞争掉），数值越大特异性越好

阈值方向（SI>1.5优秀、0.8~1.5可用、<0.8淘汰；EI>1.0优秀、0.5~1.0可用、
<0.5淘汰）与原文档保持一致。
"""
from __future__ import annotations

from dataclasses import dataclass

import primer3
from Bio.Seq import Seq

ANNEAL_TEMP_C = 60.0  # 退火温度，发夹竞争发生的温度（文档建议55-65°C）
TARGET_HAIRPIN_MARGIN_KCAL = 2.0  # ΔG_target必须比ΔG_hairpin强至少2 kcal/mol

_SI_EXCELLENT, _SI_ACCEPTABLE = 1.5, 0.8
_EI_EXCELLENT, _EI_ACCEPTABLE = 1.0, 0.5
_SI_NO_OFFTARGET_CAP = 5.0  # 无显著脱靶竞争时SI的封顶值（明确高于excellent阈值）
_VERDICT_RANK = {"reject": 0, "acceptable": 1, "excellent": 2}


def _revcomp(seq: str) -> str:
    return str(Seq(seq).reverse_complement())


def compute_dg_target(hairpin_primer_seq: str, primer_seq: str) -> tuple[float, str]:
    """ΔG_target: 发夹阻断引物整体与目标模板（primer_seq的完全互补链）的杂交自由能，
    附带ASCII结构图（"目标结合态"，用于三态竞争结构示意图）"""
    target_seq = _revcomp(primer_seq)
    r = primer3.bindings.calc_heterodimer(
        hairpin_primer_seq, target_seq, temp_c=ANNEAL_TEMP_C, output_structure=True
    )
    return r.dg / 1000.0, r.ascii_structure


def compute_dg_hairpin(hairpin_primer_seq: str) -> tuple[float, str]:
    """ΔG_hairpin: 发夹阻断引物的分子内发夹折叠自由能，附带ASCII结构图（"发夹折叠态"）"""
    r = primer3.bindings.calc_hairpin(hairpin_primer_seq, temp_c=ANNEAL_TEMP_C, output_structure=True)
    return r.dg / 1000.0, r.ascii_structure


def compute_dg_homodimer(seq: str) -> float:
    """ΔG of two copies of the hairpin primer annealing to each other (primer-dimer risk).

    calc_heterodimer(seq, seq) models bimolecular self-annealing.  Values more
    negative than -5 kcal/mol indicate meaningful primer-dimer formation risk at
    the annealing temperature; values < -8 kcal/mol are flagged as high risk.
    Concept adapted from Cordaro et al. 2021 (PCR_ML_model, bioRxiv) which showed
    that primer self-complementarity is a primary predictor of PCR efficiency loss.
    """
    r = primer3.bindings.calc_heterodimer(seq, seq, temp_c=ANNEAL_TEMP_C)
    return r.dg / 1000.0


def compute_dg_offtarget(hairpin_primer_seq: str, offtarget_window: str | None) -> tuple[float, str]:
    """ΔG_offtarget: 发夹阻断引物与非目标模板（含1-3bp错配）的杂交自由能，
    附带ASCII结构图（"脱靶结合态"）；无显著脱靶位点时视为不结合，
    返回 (0.0, "")"""
    if offtarget_window is None:
        return 0.0, ""
    r = primer3.bindings.calc_heterodimer(
        hairpin_primer_seq, offtarget_window, temp_c=ANNEAL_TEMP_C, output_structure=True
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
    if dg_offtarget >= 0:
        # 无显著脱靶位点（或脱靶结合本身不稳定）：无竞争压力，SI封顶
        si = _SI_NO_OFFTARGET_CAP
    elif dg_hairpin >= 0:
        # 该stem/loop组合不形成稳定发夹，对真实脱靶位点无竞争性保护
        si = 0.0
    else:
        si = dg_hairpin / dg_offtarget

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
