"""oxDNA 轨迹分析 + 自由能面(FES)计算 + primer3 Tm 对比

核心分析流程：
  1. 解析各温度点的 oxDNA 轨迹文件 (traj_XXXk.dat)
  2. 对每帧计算 order parameter: n_bp = 茎区已形成碱基对数 (0 .. stem_len)
     判定标准: 互补核苷酸 (i, n-1-i) 的质心距离 < r_cut = 1.3 σ (≈ 1.11 nm)
  3. 直方图 P(n_bp; T) → 自由能面 F(n_bp; T) = -kT × ln P
  4. 计算每温度的折叠自由能 ΔF_fold(T) = F(0) - F(stem_len)
     插值得 Tm_MD (ΔF_fold = 0 的温度)
  5. 与 primer3 的 dg_hairpin / 估算 Tm 比较，输出验证报告

物理量换算：
  kB = 1.987 × 10⁻³ kcal/(mol·K)
  kT (60°C, 333 K) = 0.6613 kcal/mol
  σ = 0.8518 nm (oxDNA 长度单位)
  T* → T_K : T_K = T* × 3000
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Generator

import numpy as np

KB_KCAL = 1.987e-3   # kcal/(mol·K)
OX_SIGMA_NM = 0.8518  # nm per oxDNA length unit
R_CUT_SIGMA = 1.3    # σ (legacy, kept for reference)
# Base-site detection (more accurate):
#   oxDNA nucleotide position = backbone site
#   base site = pos + POS_BASE * a1  (a1 = frame columns 3:6)
#   WC paired bases are ~0.76σ apart; R_CUT_BASE = 1.0σ covers paired + thermal fluctuations
POS_BASE    = 0.4    # σ, from model.h POS_BASE
R_CUT_BASE  = 1.0   # σ, base-site distance threshold for base pair detection


# ---------------------------------------------------------------------------
# 轨迹解析
# ---------------------------------------------------------------------------

def _parse_frames(traj_path: Path, n_nts: int) -> Generator[np.ndarray, None, None]:
    """逐帧解析 oxDNA 轨迹文件，yield 每帧的 (n_nts × 15) 核苷酸数组。

    每帧格式:
      t = ...
      b = Lx Ly Lz
      E = ...
      pos_x pos_y pos_z  a1_x a1_y a1_z  a3_x a3_y a3_z  vx vy vz  wx wy wz
      (n_nts 行)
    """
    with open(traj_path) as f:
        lines = f.readlines()

    header_lines = 3  # t, b, E
    frame_size = header_lines + n_nts
    n_frames = len(lines) // frame_size

    for frame_idx in range(n_frames):
        start = frame_idx * frame_size + header_lines
        data = np.empty((n_nts, 15))
        for i in range(n_nts):
            vals = lines[start + i].split()
            if len(vals) < 15:
                break
            data[i] = [float(v) for v in vals[:15]]
        else:
            yield data


def _count_base_pairs(frame: np.ndarray, stem_len: int, r_cut: float = R_CUT_BASE) -> int:
    """Count formed stem base pairs in a single trajectory frame.

    Pairing rule: nucleotide i pairs with nucleotide (n-1-i), i = 0..stem_len-1
    Distance criterion: base-site to base-site distance < r_cut (σ)

    Base site position = backbone pos + POS_BASE * a1_vector
    (a1 is stored in frame columns 3:6; POS_BASE = 0.4 σ from oxDNA model.h)
    WC paired bases are ~0.76 σ apart → R_CUT_BASE = 1.0 σ is safe.
    """
    n = len(frame)
    n_bp = 0
    for i in range(stem_len):
        j = n - 1 - i
        # Base site = backbone + 0.4 * a1
        base_i = frame[i, :3] + POS_BASE * frame[i, 3:6]
        base_j = frame[j, :3] + POS_BASE * frame[j, 3:6]
        dist = float(np.linalg.norm(base_i - base_j))
        if dist < r_cut:
            n_bp += 1
    return n_bp


# ---------------------------------------------------------------------------
# 自由能面计算
# ---------------------------------------------------------------------------

def compute_fes(
    traj_path: Path,
    n_nts: int,
    stem_len: int,
    T_K: float,
    r_cut: float = R_CUT_SIGMA,
) -> dict:
    """从单温度轨迹计算 1D 自由能面 (FES)。

    返回字典:
      n_bp:      np.ndarray (0..stem_len)，order parameter 轴
      F_kcal:    np.ndarray，自由能 F(n_bp) [kcal/mol]，最小值归零
      P:         np.ndarray，归一化概率分布 P(n_bp)
      dF_fold:   ΔF_fold = F(0) - F(stem_len) [kcal/mol]
                 > 0: 折叠态更稳定（好的发夹）
                 < 0: 展开态更稳定（Tm 已超过当前 T）
      n_frames:  解析的总帧数
      T_K:       温度 (K)
    """
    hist = np.zeros(stem_len + 1, dtype=float)

    n_frames = 0
    for frame in _parse_frames(traj_path, n_nts):
        bp = _count_base_pairs(frame, stem_len, r_cut)
        hist[bp] += 1
        n_frames += 1

    if n_frames == 0:
        raise ValueError(f"轨迹文件为空或解析失败: {traj_path}")

    P = hist / hist.sum()
    P_safe = np.where(P > 0, P, 1e-12)  # 避免 log(0)

    kT = KB_KCAL * T_K
    F = -kT * np.log(P_safe)
    F -= F.min()  # 最小值归零

    # ΔF_fold = F(0) - F(stem_len)；折叠态 n_bp=stem_len，展开态 n_bp=0
    dF_fold = float(F[0] - F[stem_len])

    return {
        "n_bp":    np.arange(stem_len + 1),
        "F_kcal":  F,
        "P":       P,
        "dF_fold": dF_fold,
        "n_frames": n_frames,
        "T_K":     T_K,
    }


# ---------------------------------------------------------------------------
# Tm 估算
# ---------------------------------------------------------------------------

def estimate_Tm_md(fes_list: list[dict]) -> float | None:
    """从多温度 FES 序列中插值估算折叠态 Tm (K)。

    方法：ΔF_fold(T) 符号改变处即为 Tm。
    若温度范围不覆盖 Tm，返回 None。
    """
    Ts  = np.array([d["T_K"]    for d in fes_list])
    dFs = np.array([d["dF_fold"] for d in fes_list])

    # 寻找符号改变区间
    for i in range(len(dFs) - 1):
        if dFs[i] * dFs[i + 1] <= 0:
            # 线性插值
            t1, t2 = Ts[i], Ts[i + 1]
            d1, d2 = dFs[i], dFs[i + 1]
            return float(t1 - d1 * (t2 - t1) / (d2 - d1))

    return None


def primer3_Tm_from_design(design) -> float | None:
    """从 HairpinDesign 对象提取 primer3 预测的发夹 Tm (K)。

    primer3 的 calc_hairpin 返回 .tm (°C)，转换为 K。
    """
    try:
        import primer3
        result = primer3.bindings.calc_hairpin(
            design.hairpin_primer_seq, temp_c=60.0
        )
        tm_c = getattr(result, "tm", None)
        if tm_c is not None:
            return float(tm_c) + 273.15
    except Exception:
        pass
    # 若无法直接获取 Tm，用 ΔG、ΔH 估算: Tm = ΔH / ΔS
    # primer3 返回的 dg (J/mol at 60°C) 配合近似 ΔH ≈ 7.9 kcal/mol × stem_len
    # 此处给出粗估值
    dg = getattr(design, "dg_hairpin", None)
    if dg is None:
        return None
    stem = getattr(design, "stem_len", 5)
    dH_approx = -7.9 * stem      # kcal/mol（近似，平均 AT/GC 各半）
    dS_approx = (dH_approx - dg) / 333.15  # kcal/(mol·K)
    if dS_approx < 0:
        return float(dH_approx / dS_approx)
    return None


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------

def plot_fes(
    fes_list: list[dict],
    output_path: str | Path,
    design=None,
    Tm_md: float | None = None,
) -> None:
    """生成发夹三态自由能面图（多温度叠加）。

    每条曲线对应一个模拟温度，颜色由低温（蓝）到高温（红）渐变。
    灰色竖线标记 n_bp = stem_len（折叠态）和 n_bp = 0（展开态）。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── 左图: FES 曲线 (多温度) ──────────────────────────────────────
    ax = axes[0]
    n_temps = len(fes_list)
    cmap = cm.coolwarm
    stem_len = int(fes_list[0]["n_bp"][-1])

    for i, d in enumerate(fes_list):
        color = cmap(i / max(n_temps - 1, 1))
        T_C = d["T_K"] - 273.15
        ax.plot(d["n_bp"], d["F_kcal"], color=color, lw=2.0,
                label=f"{T_C:.0f}°C  ΔF={d['dF_fold']:+.2f}")

    ax.axvline(0, color="gray", ls="--", lw=1.0, alpha=0.6)
    ax.axvline(stem_len, color="steelblue", ls="--", lw=1.0, alpha=0.6)
    ax.axhline(0, color="black", ls=":", lw=0.7, alpha=0.4)

    ax.text(0,           ax.get_ylim()[1] * 0.95, "展开态\n(目标结合准备)",
            ha="center", va="top", fontsize=8, color="gray")
    ax.text(stem_len, ax.get_ylim()[1] * 0.95, "折叠态\n(发夹封闭)",
            ha="center", va="top", fontsize=8, color="steelblue")

    ax.set_xlabel("茎区碱基对数 n_bp")
    ax.set_ylabel("自由能 F (kcal/mol)")
    ax.set_title("发夹折叠自由能面 (oxDNA2 MD)")
    ax.legend(fontsize=7.5, loc="upper right")

    # ── 右图: ΔF_fold vs. T (Tm 估算) ──────────────────────────────
    ax2 = axes[1]
    Ts    = np.array([d["T_K"] - 273.15 for d in fes_list])
    dFs   = np.array([d["dF_fold"]       for d in fes_list])

    ax2.plot(Ts, dFs, "o-", color="darkgreen", lw=2.0, ms=7, label="oxDNA MD")
    ax2.axhline(0, color="black", ls="--", lw=1.0, alpha=0.6)

    # Tm 标注
    if Tm_md is not None:
        Tm_C = Tm_md - 273.15
        ax2.axvline(Tm_C, color="tomato", ls="-", lw=1.5,
                    label=f"Tm(oxDNA) = {Tm_C:.1f}°C")
        ax2.annotate(f"Tm = {Tm_C:.1f}°C",
                     xy=(Tm_C, 0), xytext=(Tm_C + 3, max(dFs) * 0.3),
                     fontsize=9, color="tomato",
                     arrowprops=dict(arrowstyle="->", color="tomato"))

    # primer3 Tm 对比
    if design is not None:
        Tm_p3_K = primer3_Tm_from_design(design)
        if Tm_p3_K is not None:
            Tm_p3_C = Tm_p3_K - 273.15
            ax2.axvline(Tm_p3_C, color="royalblue", ls=":", lw=1.5,
                        label=f"Tm(primer3 est.) = {Tm_p3_C:.1f}°C")

    ax2.fill_between(Ts, dFs, 0, where=(dFs > 0),
                     alpha=0.15, color="steelblue", label="折叠态更稳定")
    ax2.fill_between(Ts, dFs, 0, where=(dFs < 0),
                     alpha=0.15, color="tomato",   label="展开态更稳定")

    ax2.set_xlabel("温度 (°C)")
    ax2.set_ylabel("ΔF_fold = F(展开) − F(折叠)  (kcal/mol)")
    ax2.set_title("折叠自由能差 ΔF_fold vs. 温度")
    ax2.legend(fontsize=8)

    # ── 总标题 ───────────────────────────────────────────────────────
    seq_short = ""
    if design is not None:
        seq = design.hairpin_primer_seq
        seq_short = f"  |  {seq[:14]}…  stem={design.stem_len}bp loop={design.loop_len}nt"
        dg = design.dg_hairpin
        fig.suptitle(
            f"三态竞争自由能验证 (oxDNA2){seq_short}\n"
            f"ΔG_hairpin(primer3, 60°C) = {dg:.2f} kcal/mol",
            fontsize=10,
        )
    else:
        fig.suptitle("三态竞争自由能验证 (oxDNA2)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 文字报告
# ---------------------------------------------------------------------------

def generate_report(
    fes_list: list[dict],
    Tm_md: float | None,
    design=None,
    output_path: str | Path | None = None,
) -> str:
    """生成文字验证报告，可选写入文件。"""
    lines = [
        "=" * 62,
        "oxDNA2 三态自由能验证报告",
        "=" * 62,
        "",
    ]

    if design is not None:
        stem = design.stem_len
        loop = design.loop_len
        seq  = design.hairpin_primer_seq
        lines += [
            f"序列:       {seq}",
            f"结构:       stem={stem}bp  loop={loop}nt  全长={len(seq)}nt",
            f"primer3 ΔG_hairpin(60°C): {design.dg_hairpin:.2f} kcal/mol",
            "",
        ]

    lines += [
        "各温度折叠自由能 ΔF_fold = F(展开) - F(折叠)：",
        f"  {'T(K)':<8}{'T(°C)':<8}{'ΔF_fold(kcal/mol)':<22}{'状态':<20}{'帧数':<8}",
        "  " + "-" * 66,
    ]
    for d in fes_list:
        T_C  = d["T_K"] - 273.15
        dF   = d["dF_fold"]
        state = "折叠态稳定 ✓" if dF > 0 else "展开态稳定 (T > Tm)"
        lines.append(
            f"  {d['T_K']:<8.0f}{T_C:<8.1f}{dF:<22.3f}{state:<20}{d['n_frames']:<8}"
        )

    lines += [""]

    if Tm_md is not None:
        lines.append(f"Tm (oxDNA MD 插值):    {Tm_md - 273.15:.1f}°C  ({Tm_md:.1f} K)")

    if design is not None:
        Tm_p3 = primer3_Tm_from_design(design)
        if Tm_p3 is not None:
            lines.append(f"Tm (primer3 估算):     {Tm_p3 - 273.15:.1f}°C  ({Tm_p3:.1f} K)")
        if Tm_md is not None and Tm_p3 is not None:
            delta = abs(Tm_md - Tm_p3)
            verdict = "验证通过 ✓" if delta < 10 else "偏差较大，建议检查茎序列 GC 组成"
            lines.append(f"Tm 偏差:               {delta:.1f} K  →  {verdict}")

    lines += [
        "",
        "三态竞争机制验证（PCR退火温度 60°C = 333 K）：",
    ]
    # 找 60°C 最接近的温度点
    idx60 = min(range(len(fes_list)),
                key=lambda i: abs(fes_list[i]["T_K"] - 333))
    d60 = fes_list[idx60]
    if d60["dF_fold"] > 0:
        lines += [
            f"  → 在 {d60['T_K']-273.15:.0f}°C 时 ΔF_fold = {d60['dF_fold']:+.2f} kcal/mol",
            "  → 发夹折叠态稳定，3' 端受限，脱靶延伸被抑制 ✓",
            "  → 目标模板需克服 ΔF_fold 能垒以打开发夹（与 ΔG_target 一致）",
        ]
    else:
        lines += [
            f"  → 在 {d60['T_K']-273.15:.0f}°C 时 ΔF_fold = {d60['dF_fold']:+.2f} kcal/mol",
            "  → 警告：退火温度下发夹不稳定，建议增大茎长或调整序列",
        ]

    lines += ["", "=" * 62]
    report = "\n".join(lines)

    if output_path is not None:
        Path(output_path).write_text(report, encoding="utf-8")

    return report
