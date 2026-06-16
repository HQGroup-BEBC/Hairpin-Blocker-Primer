"""Parse oxDNA production VMMC histogram and generate real FES figures.

oxDNA hist_prod.dat column layout (with extrapolate_hist):
  col 0    : n_bp (order parameter)
  col 1    : biased histogram count at simulation T
  col 2    : unbiased count at sim T (= biased / umbrella_weight)
  col 3..  : unbiased count at each extrapolation temperature
             (already Boltzmann-reweighted AND umbrella-weight-corrected)

So P(n_bp; T_k) ∝ erdata[k][n_bp]  →  FES = -kT_k × ln P
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

KB_KCAL = 1.987e-3   # kcal/(mol·K)

# ── Temperature labels (must match extrapolate_hist order in input_prod.conf) ──
EXTRAP_TEMPS_C = [37, 51, 65, 79, 93, 107]   # °C
EXTRAP_TEMPS_K = [T + 273.15 for T in EXTRAP_TEMPS_C]


def parse_last_hist(hist_path: Path) -> tuple[list[str], np.ndarray]:
    """Read the last histogram block from hist_prod.dat.

    Returns (header_temps_str, data) where data is shape (n_states, ncols).
    ncols = 2 + n_extrap_temps  (biased, unbiased_simT, unbiased_T1, ...)
    """
    with open(hist_path) as f:
        text = f.read()

    blocks = re.split(r"(?=#t =)", text)
    last_block = blocks[-1].strip()

    lines = last_block.split("\n")
    header = lines[0]  # "#t = ...; extr. Ts: ..."

    # extract extra-T labels from header
    m = re.search(r"extr\. Ts:\s*(.*)", header)
    extr_labels = m.group(1).strip().split() if m else []

    rows = []
    for line in lines[1:]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append([float(v) for v in line.split()])

    data = np.array(rows)
    return extr_labels, data


def compute_fes_from_hist(
    hist_path: Path,
    temps_K: list[float],
    stem_len: int,
) -> list[dict]:
    """Compute FES at each extrapolation temperature from prod histogram.

    Args:
        hist_path:  path to hist_prod.dat
        temps_K:    list of temperatures in K, matching extrapolate_hist order
        stem_len:   number of stem base pairs (max n_bp)

    Returns list of dicts with keys: T_K, n_bp, F_kcal, P, dF_fold
    """
    extr_labels, data = parse_last_hist(hist_path)

    # data[:,0] = n_bp index
    # data[:,1] = biased count (skip)
    # data[:,2] = unbiased at sim T
    # data[:,3..] = unbiased at each extrapolation T

    n_states = stem_len + 1
    fes_list = []

    for ti, T_K in enumerate(temps_K):
        col = 3 + ti   # column index for this extrapolation T
        if col >= data.shape[1]:
            print(f"  WARNING: no column {col} for T={T_K}K, skipping")
            continue

        counts = np.zeros(n_states)
        for row in data:
            n_bp = int(row[0])
            if 0 <= n_bp < n_states:
                counts[n_bp] = row[col]

        total = counts.sum()
        if total == 0:
            print(f"  WARNING: zero counts at T={T_K}K")
            continue

        P = counts / total
        P_safe = np.where(P > 0, P, 1e-30)
        kT = KB_KCAL * T_K
        F = -kT * np.log(P_safe)
        F -= F.min()

        dF_fold = float(F[0] - F[stem_len])

        fes_list.append({
            "T_K": T_K,
            "n_bp": np.arange(n_states),
            "F_kcal": F,
            "P": P,
            "dF_fold": dF_fold,
            "counts": counts,
        })

    return fes_list


def estimate_Tm(fes_list: list[dict]) -> float | None:
    """Interpolate Tm from ΔF_fold sign change."""
    Ts  = np.array([d["T_K"]    for d in fes_list])
    dFs = np.array([d["dF_fold"] for d in fes_list])
    for i in range(len(dFs) - 1):
        if dFs[i] * dFs[i + 1] <= 0:
            t1, t2 = Ts[i], Ts[i + 1]
            d1, d2 = dFs[i], dFs[i + 1]
            return float(t1 - d1 * (t2 - t1) / (d2 - d1))
    return None


def plot_real_fes(fes_list: list[dict], out_main: Path, out_detail: Path,
                  meta: dict | None = None) -> None:
    """Generate publication-quality FES figures from real simulation data."""

    stem_len = int(fes_list[0]["n_bp"][-1])
    Tm_K = estimate_Tm(fes_list)
    Tm_C = Tm_K - 273.15 if Tm_K else None

    n_temps = len(fes_list)
    cmap = cm.coolwarm

    # ─────────────────────────────────────────────────
    # Figure 1: Main overview (2 panels)
    # ─────────────────────────────────────────────────
    fig1, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig1.patch.set_facecolor("white")

    # Left: FES curves
    ax = axes[0]
    ax.set_facecolor("#fafafa")
    for i, d in enumerate(fes_list):
        color = cmap(i / max(n_temps - 1, 1))
        T_C = d["T_K"] - 273.15
        ax.plot(d["n_bp"], d["F_kcal"], color=color, lw=2.2,
                label=f"{T_C:.0f}°C  ΔF={d['dF_fold']:+.2f}")
        ax.scatter(d["n_bp"], d["F_kcal"], color=color, s=28, zorder=3)

    ax.axvline(0, color="#999", ls="--", lw=1.0, alpha=0.7)
    ax.axvline(stem_len, color="steelblue", ls="--", lw=1.0, alpha=0.7)
    ax.axhline(0, color="black", ls=":", lw=0.7, alpha=0.4)

    ylim = ax.get_ylim()
    ax.text(0,     ylim[1] * 0.97, "Open\n(target-ready)",
            ha="center", va="top", fontsize=8, color="#888")
    ax.text(stem_len, ylim[1] * 0.97, "Folded\n(hairpin closed)",
            ha="center", va="top", fontsize=8, color="steelblue")

    ax.set_xlabel("Stem base pairs formed  n_bp", fontsize=11)
    ax.set_ylabel("Free energy  F (kcal/mol)", fontsize=11)
    ax.set_title("Hairpin FES — oxDNA2 VMMC + Umbrella Sampling", fontsize=11)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(alpha=0.3)

    # Right: ΔF_fold vs T
    ax2 = axes[1]
    ax2.set_facecolor("#fafafa")
    Ts  = np.array([d["T_K"] - 273.15 for d in fes_list])
    dFs = np.array([d["dF_fold"]       for d in fes_list])
    ax2.plot(Ts, dFs, "o-", color="darkgreen", lw=2.2, ms=7,
             label="oxDNA2 VMMC")
    ax2.axhline(0, color="black", ls="--", lw=1.0, alpha=0.6)
    ax2.fill_between(Ts, dFs, 0, where=(dFs > 0),
                     alpha=0.15, color="steelblue", label="Folded favored")
    ax2.fill_between(Ts, dFs, 0, where=(dFs < 0),
                     alpha=0.15, color="tomato",   label="Open favored")

    if Tm_C is not None:
        ax2.axvline(Tm_C, color="tomato", ls="-", lw=2.0,
                    label=f"Tm(oxDNA) = {Tm_C:.1f}°C")
        ax2.annotate(f"Tm = {Tm_C:.1f}°C",
                     xy=(Tm_C, 0), xytext=(Tm_C + 4, max(abs(dFs)) * 0.4),
                     fontsize=9, color="tomato",
                     arrowprops=dict(arrowstyle="->", color="tomato", lw=1.5))

    if meta and "primer3_tm" in meta:
        p3_tm = float(meta["primer3_tm"])
        ax2.axvline(p3_tm, color="royalblue", ls=":", lw=1.8,
                    label=f"Tm(primer3 NN) = {p3_tm:.1f}°C")
        ax2.annotate(f"primer3\n{p3_tm:.1f}°C",
                     xy=(p3_tm, min(dFs) * 0.4), xytext=(p3_tm - 20, min(dFs) * 0.55),
                     fontsize=8, color="royalblue",
                     arrowprops=dict(arrowstyle="->", color="royalblue", lw=1.2))

    ax2.set_xlabel("Temperature (°C)", fontsize=11)
    ax2.set_ylabel(r"$\Delta F_{fold}$ = F(open) − F(folded)  (kcal/mol)", fontsize=10)
    ax2.set_title("Folding Free Energy vs. Temperature", fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # Suptitle
    seq_info = ""
    note = ""
    if meta:
        seq = meta.get("seq", "")
        sl = meta.get("stem_len", "?")
        ll = meta.get("loop_len", "?")
        seq_info = f"  {seq}  (stem={sl}bp, loop={ll}nt)"
        p3_tm = meta.get("primer3_tm", "")
        note = f"   [primer3 NN Tm={p3_tm}°C; oxDNA2 systematic offset ~25°C typical]"
    fig1.suptitle(
        f"Three-State Competition FES — oxDNA2 VMMC + Umbrella Sampling\n{seq_info}{note}",
        fontsize=10, y=1.02
    )
    fig1.tight_layout()
    fig1.savefig(out_main, dpi=200, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved main figure: {out_main}")

    # ─────────────────────────────────────────────────
    # Figure 2: Detailed 6-panel (one per temperature)
    # ─────────────────────────────────────────────────
    n_row, n_col = 2, 3
    fig2, axes2 = plt.subplots(n_row, n_col, figsize=(14, 9))
    fig2.patch.set_facecolor("white")

    for idx, d in enumerate(fes_list):
        r, c = divmod(idx, n_col)
        ax = axes2[r, c]
        ax.set_facecolor("#fafafa")
        color = cmap(idx / max(n_temps - 1, 1))
        T_C = d["T_K"] - 273.15

        # FES as bar
        n_arr = d["n_bp"]
        F_arr = d["F_kcal"]
        ax.bar(n_arr, F_arr, color=color, alpha=0.55, width=0.7, zorder=2)
        ax.plot(n_arr, F_arr, "o-", color=color, lw=2.0, ms=6, zorder=3)
        ax.axhline(0, color="black", ls=":", lw=0.7, alpha=0.4)
        ax.axvline(stem_len, color="steelblue", ls="--", lw=1.0, alpha=0.6)
        ax.set_xlim(-0.5, stem_len + 0.5)
        ax.set_xlabel("n_bp", fontsize=9)
        ax.set_ylabel("F (kcal/mol)", fontsize=9)
        ax.set_title(f"{T_C:.0f}°C  |  ΔF={d['dF_fold']:+.2f} kcal/mol",
                     fontsize=9)
        ax.grid(alpha=0.3)

        # State label
        state = "Folded ✓" if d["dF_fold"] > 0 else "Open (T>Tm)"
        ax.text(0.05, 0.95, state, transform=ax.transAxes,
                fontsize=8, va="top",
                color="steelblue" if d["dF_fold"] > 0 else "tomato")

    fig2.suptitle(
        "Hairpin FES at Each Temperature — oxDNA2 VMMC Umbrella Sampling",
        fontsize=12
    )
    fig2.tight_layout()
    fig2.savefig(out_detail, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved detail figure: {out_detail}")


def main():
    BASE = Path(__file__).parent.parent
    SIM  = BASE / "sim_out"
    FIG  = BASE / "figures"
    FIG.mkdir(exist_ok=True)

    hist_path = SIM / "hist_prod.dat"
    if not hist_path.exists():
        print(f"ERROR: {hist_path} not found. Production run still in progress?")
        return

    # Load meta
    meta = {}
    meta_path = SIM / "meta.txt"
    if meta_path.exists():
        for line in meta_path.read_text().split("\n"):
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip()

    stem_len = int(meta.get("stem_len", 5))

    print(f"Parsing {hist_path} ...")
    extr_labels, data = parse_last_hist(hist_path)
    print(f"  Extrapolation labels: {extr_labels}")
    print(f"  Data shape: {data.shape}")
    print(f"  n_bp states: {data[:,0].astype(int).tolist()}")

    fes_list = compute_fes_from_hist(hist_path, EXTRAP_TEMPS_K, stem_len)

    if not fes_list:
        print("ERROR: no FES data computed. Check histogram file.")
        return

    print("\nFES Summary:")
    print(f"{'T(°C)':>8}  {'ΔF_fold':>10}  {'P_fold%':>9}  {'P_open%':>9}")
    print("-" * 42)
    for d in fes_list:
        T_C  = d["T_K"] - 273.15
        pf   = d["P"][stem_len] * 100
        po   = d["P"][0] * 100
        print(f"{T_C:>8.0f}  {d['dF_fold']:>10.3f}  {pf:>9.2f}  {po:>9.2f}")

    Tm_K = estimate_Tm(fes_list)
    if Tm_K:
        print(f"\nTm (oxDNA2 VMMC) = {Tm_K - 273.15:.1f}°C")
    else:
        print("\nTm outside simulation temperature range")

    # Generate figures — replace the analytical demo figures
    plot_real_fes(
        fes_list,
        out_main   = FIG / "fes_demo_main.png",
        out_detail = FIG / "fes_demo_detail.png",
        meta       = meta,
    )
    print("\nReal FES figures generated successfully.")


if __name__ == "__main__":
    main()
