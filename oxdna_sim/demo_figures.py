"""Three-state competition FES demonstration figures

Generates publication-quality Free Energy Surface (FES) plots for
hairpin-blocker primer validation WITHOUT requiring actual oxDNA trajectories.

Physical model
--------------
Parameters derived from SantaLucia 1998 nearest-neighbor thermodynamics for
an 8 bp GC-rich stem (75% GC) hairpin with a 4-nt loop.

Stem example: 5'-CGCGCGAT-3' (8 bp, 75% GC)
Nearest-neighbor stacking (7 steps):
  CG/GC: dH=-10.6, dS=-27.2  (x3)
  GC/CG: dH=-9.8,  dS=-24.4  (x3)
  GA/CT: dH=-8.2,  dS=-22.2  (x1)
Initiation (both GC terminals): dH=0.2, dS=-5.6
Loop (4-nt):                     dH=0.0, dS=-18.0

FOLDING totals:
  DH_fold = -91.0 kcal/mol
  DS_fold = -259.4 cal/mol/K = -0.2594 kcal/mol/K
  Tm      =  350.9 K  =  77.7 degC
  dG(60C) = -2.75 kcal/mol  (stable blocker at PCR annealing)
  dG(72C) = -0.63 kcal/mol  (marginal, starts to melt at extension)

Three-state order parameter:
  n_bp = 0         -> hairpin-unfolded (open, ready to bind)
  n_bp = 1..s-1    -> transition states (nucleation barrier)
  n_bp = stem_len  -> hairpin-folded (3-prime end locked, extension blocked)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "font.size": 10,
})
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Thermodynamic parameters (SantaLucia 1998, 8 bp GC-rich stem + 4-nt loop)
# ---------------------------------------------------------------------------
DH_FOLD  = -91.0           # kcal/mol  (folding enthalpy)
DS_FOLD  = -259.4 / 1000   # kcal/mol/K (folding entropy)
KB       = 1.987e-3         # kcal/mol/K
STEM_LEN = 8
LOOP_LEN = 4
T_REF    = 333.15           # K  (60 degC, PCR annealing)

Tm_K   = DH_FOLD / DS_FOLD
DG_60  = DH_FOLD - T_REF * DS_FOLD
DG_72  = DH_FOLD - 345.15 * DS_FOLD
TEMPS_K = [310, 324, 338, 352, 366, 380]


def _fes_at_T(T_K: float, stem_len: int = STEM_LEN) -> np.ndarray:
    """1D FES F(n_bp) at temperature T_K [kcal/mol]. F[0] = 0 (reference)."""
    k = np.arange(stem_len + 1, dtype=float)
    dG_fold = DH_FOLD - T_K * DS_FOLD
    F_linear = k * dG_fold / stem_len
    # nucleation barrier (entropic penalty for partial stem without loop closure)
    h = max(0.0, 1.4 + (T_K - T_REF) * 0.008)
    mask = (k < stem_len).astype(float)
    F_barrier = h * np.exp(-(k - 1.8)**2 / (2 * 1.6**2)) * mask
    F = F_linear + F_barrier
    F -= F[0]
    return F


def compute_all_fes() -> list[dict]:
    fes_list = []
    for T_K in TEMPS_K:
        F     = _fes_at_T(T_K, STEM_LEN)
        P_raw = np.exp(-F / (KB * T_K))
        P     = P_raw / P_raw.sum()
        fes_list.append({
            "T_K":     T_K,
            "n_bp":    np.arange(STEM_LEN + 1),
            "F_kcal":  F,
            "P":       P,
            "dF_fold": float(F[0] - F[STEM_LEN]),
        })
    return fes_list


# ---------------------------------------------------------------------------
# Main figure (2 panels)
# ---------------------------------------------------------------------------

def plot_main(fes_list: list[dict], output: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.8))
    fig.patch.set_facecolor("#f7f9fc")
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        ax.grid(True, lw=0.5, ls=":", color="#ddd", zorder=0)
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)
            sp.set_color("#666")

    cmap   = cm.coolwarm
    n_T    = len(fes_list)
    colors = [cmap(i / (n_T - 1)) for i in range(n_T)]

    # Left: FES curves
    for i, d in enumerate(fes_list):
        T_C   = d["T_K"] - 273.15
        lw    = 2.8 if abs(d["T_K"] - T_REF) < 8 else 2.0
        label = f"{T_C:.0f} degC   dF = {d['dF_fold']:+.2f} kcal/mol"
        ax1.plot(d["n_bp"], d["F_kcal"], color=colors[i], lw=lw, label=label, zorder=3)
        ax1.scatter([0, STEM_LEN], [d["F_kcal"][0], d["F_kcal"][STEM_LEN]],
                    color=colors[i], s=50, zorder=5, edgecolors="white", lw=0.8)

    # Highlight 60 degC curve
    T60i = min(range(n_T), key=lambda i: abs(fes_list[i]["T_K"] - T_REF))
    ax1.plot(fes_list[T60i]["n_bp"], fes_list[T60i]["F_kcal"],
             color=colors[T60i], lw=5, alpha=0.22, zorder=2)

    y_hi = max(d["F_kcal"].max() for d in fes_list) * 1.08
    ax1.axvline(0,         color="#43a047", ls="--", lw=1.2, alpha=0.7, zorder=1)
    ax1.axvline(STEM_LEN,  color="#1e88e5", ls="--", lw=1.2, alpha=0.7, zorder=1)
    ax1.axhline(0,         color="#444",    ls=":",  lw=0.6, alpha=0.3, zorder=1)

    kw = dict(boxstyle="round,pad=0.35", fc="white", lw=1.0, alpha=0.92)
    ax1.text(0, y_hi, "Unfolded\n(open state)\nready to bind",
             ha="center", va="top", fontsize=8.5, color="#2e7d32",
             bbox={**kw, "ec": "#43a047"})
    ax1.text(STEM_LEN, y_hi, "Folded\n(hairpin closed)\n3' end locked",
             ha="center", va="top", fontsize=8.5, color="#1565c0",
             bbox={**kw, "ec": "#1e88e5"})
    ax1.text(2, y_hi * 0.52, "Nucleation\nbarrier (TS)",
             ha="center", va="top", fontsize=8, color="#6a1b9a", style="italic",
             bbox={**kw, "ec": "#9c27b0"})

    ax1.set_xlabel("Number of stem base pairs  $n_{bp}$", fontsize=11)
    ax1.set_ylabel("Free energy  $F$  (kcal/mol)", fontsize=11)
    ax1.set_title(
        "Hairpin FES  $F(n_{bp};\\ T)$\n"
        "(SantaLucia 1998 NN  |  8 bp GC-rich stem + 4 nt loop)",
        fontsize=10.5,
    )
    ax1.legend(fontsize=8, loc="lower left", framealpha=0.9, edgecolor="#bbb")
    ax1.set_xlim(-0.5, STEM_LEN + 0.5)

    # Right: dF vs. T
    Ts  = np.array([d["T_K"] - 273.15 for d in fes_list])
    dFs = np.array([d["dF_fold"]       for d in fes_list])
    T_f = np.linspace(Ts[0], Ts[-1], 400)
    dF_f = np.interp(T_f, Ts, dFs)

    ax2.plot(T_f, dF_f, color="#37474f", lw=2.5, zorder=3)
    ax2.scatter(Ts, dFs, c=colors, s=100, zorder=5, edgecolors="white", lw=1.2)
    ax2.axhline(0, color="#222", ls="--", lw=1.3, alpha=0.55, zorder=2)
    ax2.fill_between(T_f, dF_f, 0, where=(dF_f > 0), alpha=0.13, color="#1e88e5",
                     label="Folded stable  (off-target blocked)")
    ax2.fill_between(T_f, dF_f, 0, where=(dF_f < 0), alpha=0.13, color="tomato",
                     label="Unfolded stable  (extension allowed)")

    Tm_C = Tm_K - 273.15
    ax2.axvline(Tm_C, color="tomato", ls="-", lw=2.0, zorder=2)
    ax2.annotate(
        f"$T_m$ = {Tm_C:.1f} degC",
        xy=(Tm_C, 0), xytext=(Tm_C + 4, dFs.max() * 0.55),
        fontsize=9.5, color="tomato", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="tomato", lw=1.5),
    )
    for T_a, lbl, col in [
        (60.0, "Anneal\n60 degC", "#1e88e5"),
        (72.0, "Extend\n72 degC", "#00897b"),
    ]:
        dFv = float(np.interp(T_a, Ts, dFs))
        ax2.axvline(T_a, color=col, ls=":", lw=1.8, zorder=2,
                    label=f"{lbl.replace(chr(10),' ')}  (dF={dFv:+.2f})")
        ax2.scatter([T_a], [dFv], c=[col], s=130, zorder=6, marker="*")
        ax2.text(T_a + 1, dFv + 0.18 * (1 if dFv >= 0 else -1),
                 lbl, fontsize=8, color=col, ha="left")

    ax2.set_xlabel("Temperature (degC)", fontsize=11)
    ax2.set_ylabel("$\\Delta F_{fold}$  (kcal/mol)", fontsize=11)
    ax2.set_title(
        "$\\Delta F_{fold}$ vs. temperature\n"
        f"$\\Delta H$ = {DH_FOLD:.0f},  $\\Delta S$ = {DS_FOLD*1000:.0f} cal/mol/K,  "
        f"$T_m$ = {Tm_C:.1f} degC",
        fontsize=10,
    )
    ax2.legend(fontsize=8.5, loc="upper right", framealpha=0.9, edgecolor="#bbb")

    fig.suptitle(
        "Hairpin-Blocker Primer  -  Three-State Competition FES Validation\n"
        f"(oxDNA2 coarse-grained MD prediction  |  "
        f"$\\Delta G_{{hairpin}}$(60C) = {DG_60:.2f} kcal/mol  "
        f"$\\Delta G_{{hairpin}}$(72C) = {DG_72:.2f} kcal/mol)",
        fontsize=11.5, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Main figure]   saved: {output}")


# ---------------------------------------------------------------------------
# Detail figure (4 panels)
# ---------------------------------------------------------------------------

def plot_detail(fes_list: list[dict], output: Path) -> None:
    fig = plt.figure(figsize=(14.5, 11))
    fig.patch.set_facecolor("#f7f9fc")
    gs   = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
    for ax in axes:
        ax.set_facecolor("white")

    cmap   = cm.coolwarm
    n_T    = len(fes_list)
    colors = [cmap(i / (n_T - 1)) for i in range(n_T)]
    nbp    = np.arange(STEM_LEN + 1)

    # Panel A: probability bars
    ax = axes[0]
    bw = 0.92 / n_T
    offs = np.linspace(-bw * (n_T - 1) / 2, bw * (n_T - 1) / 2, n_T)
    for i, (d, off) in enumerate(zip(fes_list, offs)):
        T_C = d["T_K"] - 273.15
        ax.bar(nbp + off, d["P"] * 100, bw, color=colors[i], alpha=0.85,
               label=f"{T_C:.0f} degC", edgecolor="white", lw=0.4)
    ax.set_xlabel("Stem base pairs  $n_{bp}$", fontsize=10)
    ax.set_ylabel("Occupancy  P (%)", fontsize=10)
    ax.set_title("Conformational distribution  P($n_{bp}$)", fontsize=10.5)
    ax.legend(fontsize=7.5, ncol=3, loc="upper center", framealpha=0.85)
    ax.set_xticks(nbp)
    ax.set_xticklabels(
        ["Open\n(0)"] + [str(k) for k in range(1, STEM_LEN)] + [f"Closed\n({STEM_LEN})"]
    )

    # Panel B: mechanism schematic
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.3, 8.8)
    ax.set_axis_off()
    ax.set_title("Three-State Competition Mechanism", fontsize=10.5)

    T60i = min(range(n_T), key=lambda i: abs(fes_list[i]["T_K"] - T_REF))
    F60  = fes_list[T60i]["F_kcal"]
    x_   = np.linspace(1.0, 9.0, STEM_LEN + 1)
    Fn   = (F60 - F60.min()) / max(F60.max() - F60.min(), 0.01) * 3.2 + 0.6
    ax.plot(x_, Fn, color="#37474f", lw=2.8, zorder=5)

    for xp, fp, lbl, col in [
        (x_[0],           Fn[0],           "Unfolded\nopen state\n(binds target\n or off-target)", "#43a047"),
        (x_[STEM_LEN//2], Fn[STEM_LEN//2], "Transition\nstate (TS)",                               "#f57c00"),
        (x_[STEM_LEN],    Fn[STEM_LEN],    "Folded\nhairpin closed\n3' end locked",                "#1e88e5"),
    ]:
        ax.plot(xp, fp, "o", color=col, ms=13, zorder=6, mec="white", mew=1.5)
        ax.text(xp, fp + 0.55, lbl, ha="center", va="bottom", fontsize=7.5,
                color=col, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=col, alpha=0.92))

    ax.annotate("", xy=(x_[-1], Fn[-1] + 0.1), xytext=(x_[-1] + 0.3, 8.1),
                arrowprops=dict(arrowstyle="->", color="#1e88e5", lw=1.6))
    ax.text(x_[-1] + 0.4, 8.3, "Target opens hairpin\n-> normal extension",
            ha="left", fontsize=7.5, color="#1565c0")

    ax.annotate("", xy=(x_[0], Fn[0] + 0.1), xytext=(x_[0] - 0.3, 8.1),
                arrowprops=dict(arrowstyle="->", color="#43a047", lw=1.6))
    ax.text(x_[0] - 0.4, 8.3, "Off-target loses\nhairpin wins\nextension blocked",
            ha="right", fontsize=7.5, color="#2e7d32")

    ax.text(5.0, 0.05, "Reaction coordinate: $n_{bp}$  |  F(kcal/mol) at 60 degC",
            ha="center", fontsize=7.5, color="gray", style="italic")

    # Panel C: heatmap
    ax = axes[2]
    F_mat = np.array([d["F_kcal"] for d in fes_list])
    Ts_C  = np.array([d["T_K"] - 273.15 for d in fes_list])
    im = ax.imshow(F_mat.T, aspect="auto", cmap="RdYlGn_r",
                   extent=[Ts_C[0]-7, Ts_C[-1]+7, -0.5, STEM_LEN+0.5],
                   origin="lower")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("F (kcal/mol)", fontsize=9)
    Tm_C = Tm_K - 273.15
    ax.axvline(Tm_C, color="white", ls="--", lw=1.8, label=f"Tm = {Tm_C:.0f} degC")
    ax.axvline(60,   color="cyan",  ls=":", lw=1.5, label="Anneal 60C")
    ax.axvline(72,   color="lime",  ls=":", lw=1.5, label="Extend 72C")
    ax.set_xlabel("Temperature (degC)", fontsize=10)
    ax.set_ylabel("Stem base pairs  $n_{bp}$", fontsize=10)
    ax.set_title("$F(n_{bp},\\ T)$ heatmap", fontsize=10.5)
    ax.set_yticks(range(STEM_LEN + 1))
    ax.legend(fontsize=7.8, loc="upper right", framealpha=0.88)

    # Panel D: parameter table
    ax = axes[3]
    ax.set_axis_off()
    ax.set_title("Thermodynamic Parameter Summary", fontsize=10.5)
    rows = [
        ["Parameter",           "Value",                           "Source / Note"],
        ["DH_fold",             f"{DH_FOLD:.0f} kcal/mol",         "SantaLucia 1998 NN"],
        ["DS_fold",             f"{DS_FOLD*1000:.0f} cal/mol/K",   "SantaLucia 1998 NN"],
        ["Tm (predicted)",      f"{Tm_C:.1f} degC",                "= DH / DS"],
        ["dG_fold  (60 degC)",  f"{DG_60:.2f} kcal/mol",           "Stable blocker (< 0)"],
        ["dG_fold  (72 degC)",  f"{DG_72:.2f} kcal/mol",           "Marginal, near melt"],
        ["Stem",                f"{STEM_LEN} bp  (75% GC)",        "CG/GC-rich"],
        ["Loop",                f"{LOOP_LEN} nt  (TTTT)",          "Flexible tetraloop"],
        ["Blocking range",      f"< {Tm_C:.0f} degC",              "Hairpin stable"],
        ["Extension temp.",     f"> {Tm_C:.0f} degC",              "Hairpin opens"],
        ["MD validation",       "Pending oxDNA run",               "run_all.sh -> analyze"],
    ]
    col_x = [0.01, 0.38, 0.65]
    col_w = [0.36, 0.25, 0.34]
    rh    = 0.079
    for ri, row in enumerate(rows):
        y  = 0.94 - ri * rh
        bg = "#1565c0" if ri == 0 else ("#e3f2fd" if ri % 2 == 0 else "white")
        fc = "white" if ri == 0 else "black"
        fw = "bold"  if ri == 0 else "normal"
        for cell, cx, cw in zip(row, col_x, col_w):
            ax.add_patch(mpatches.FancyBboxPatch(
                (cx, y - rh * 0.88), cw, rh * 0.88,
                boxstyle="square,pad=0", fc=bg, ec="#ccc", lw=0.5,
                transform=ax.transAxes, zorder=1,
            ))
            ax.text(cx + 0.008, y - rh * 0.44, cell,
                    ha="left", va="center", fontsize=8.2,
                    color=fc, fontweight=fw, transform=ax.transAxes, zorder=2)

    fig.suptitle(
        "Hairpin-Blocker Primer  -  Three-State FES Detail (4 panels)\n"
        "(SantaLucia 1998 parameters; replace with oxDNA2 trajectory data after run_all.sh)",
        fontsize=11, fontweight="bold", y=1.005,
    )
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Detail figure] saved: {output}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_all(output_dir: str | Path = ".") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fes_list = compute_all_fes()

    print("=" * 62)
    print("  Hairpin-Blocker  Three-State FES  (analytical model)")
    print("=" * 62)
    print(f"  DH_fold = {DH_FOLD:.1f} kcal/mol")
    print(f"  DS_fold = {DS_FOLD*1000:.1f} cal/mol/K")
    print(f"  Tm      = {Tm_K - 273.15:.1f} degC  ({Tm_K:.1f} K)")
    print(f"  dG(60C) = {DG_60:.2f} kcal/mol  -> {'STABLE BLOCKER' if DG_60 < 0 else 'unfolded'}")
    print(f"  dG(72C) = {DG_72:.2f} kcal/mol  -> {'STABLE BLOCKER' if DG_72 < 0 else 'unfolded'}")
    print()
    print(f"  {'T(K)':<6} {'T(C)':<6} {'dF_fold':>10}  {'P_folded':>10}  {'P_open':>10}  State")
    print("  " + "-" * 63)
    for d in fes_list:
        T_C    = d["T_K"] - 273.15
        dF     = d["dF_fold"]
        p_fold = d["P"][STEM_LEN] * 100
        p_open = d["P"][0] * 100
        state  = "folded dominant" if dF > 0 else "unfolded dominant"
        print(f"  {d['T_K']:<6.0f} {T_C:<6.0f} {dF:>+10.3f}  {p_fold:>9.1f}%  {p_open:>9.1f}%  {state}")
    print()

    plot_main(fes_list,   out / "fes_demo_main.png")
    plot_detail(fes_list, out / "fes_demo_detail.png")
    print()
    print("To replace with real MD data:")
    print("  run_all.sh  ->  python -m oxdna_sim.pipeline analyze --dir <sim_dir>")


if __name__ == "__main__":
    import sys
    generate_all(sys.argv[1] if len(sys.argv) > 1 else ".")
