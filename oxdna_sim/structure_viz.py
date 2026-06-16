"""
Extract representative hairpin conformations from oxDNA trajectory,
write coarse-grained PDB files, and generate 3D structural figures.

oxDNA coordinate system:
  pos    : backbone site (simulation units σ = 0.8518 nm = 8.518 Å)
  a1     : backbone→base direction (unit vector)
  a3     : stacking direction (unit vector)
  base site = pos + POS_BASE * a1   (POS_BASE = 0.4 σ)

PDB output: two pseudo-atoms per nucleotide
  P   at backbone site  (represents phosphate/sugar)
  N1  at base site      (represents base center)
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D           # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection

sys.path.insert(0, str(Path(__file__).parent.parent))
from oxdna_sim.analyze import _parse_frames, POS_BASE, R_CUT_BASE

# ── Constants ────────────────────────────────────────────────────────────────
SIM2ANG  = 8.518        # σ → Ångström
N_NTS    = 14
STEM_LEN = 5
SEQ      = "GCGCGTTTTCGCGC"

BASE_COLORS = {"G": "#2ca02c", "C": "#1f77b4", "T": "#ff7f0e", "A": "#d62728"}
PAIR_COLOR  = "#999999"
BB_COLOR    = "#444444"

# Temperature labels for display (chosen frames: n_bp=0→65+°C, n_bp=5→37°C)
NBP_TEMP_LABEL = {
    0: "107°C  (open / unfolded)",
    1: "~93°C  (1/5 bp)",
    2: "~79°C  (2/5 bp)",
    3: "~65°C  (3/5 bp)",
    4: "~51°C  (4/5 bp)",
    5: "37°C   (closed / folded)",
}


# ── Frame extraction ─────────────────────────────────────────────────────────

def collect_representative_frames(traj_path: Path) -> dict[int, np.ndarray]:
    """Return one representative frame per n_bp state (0..STEM_LEN)."""
    best: dict[int, np.ndarray] = {}
    for frame in _parse_frames(traj_path, N_NTS):
        n_bp = sum(
            1 for i in range(STEM_LEN)
            if np.linalg.norm(
                (frame[i, :3] + POS_BASE * frame[i, 3:6]) -
                (frame[N_NTS-1-i, :3] + POS_BASE * frame[N_NTS-1-i, 3:6])
            ) < R_CUT_BASE
        )
        if n_bp not in best:
            best[n_bp] = frame.copy()
        if len(best) == STEM_LEN + 1:
            break
    return best


# ── PDB writer ───────────────────────────────────────────────────────────────

def _pdb_atom(serial: int, name: str, resname: str, chain: str,
              resseq: int, x: float, y: float, z: float) -> str:
    return (
        f"ATOM  {serial:5d}  {name:<3s} {resname:>3s} {chain}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {name[0]:>2s}\n"
    )


def write_pdb(frame: np.ndarray, n_bp: int, out_path: Path) -> None:
    """Write coarse-grained 2-bead-per-nucleotide PDB for one conformation."""
    # Center at origin
    center = frame[:, :3].mean(axis=0)
    frame_c = frame.copy()
    frame_c[:, :3] -= center

    lines = [
        f"REMARK oxDNA2 hairpin conformation: {n_bp}/{STEM_LEN} stem base pairs\n",
        f"REMARK sequence: {SEQ}\n",
        f"REMARK approx. temperature state: {NBP_TEMP_LABEL.get(n_bp,'?')}\n",
        f"REMARK scale: 1 oxDNA σ = {SIM2ANG:.3f} Å  (centered at origin)\n",
        f"REMARK beads: P = backbone/sugar site, N1 = base center site\n",
    ]

    serial = 1
    for i in range(N_NTS):
        base   = SEQ[i]
        resseq = i + 1
        chain  = "A"
        resname = f"D{base}"

        # backbone bead
        bb = frame_c[i, :3] * SIM2ANG
        lines.append(_pdb_atom(serial, "P", resname, chain, resseq,
                                bb[0], bb[1], bb[2]))
        serial += 1

        # base bead
        bs = (frame_c[i, :3] + POS_BASE * frame_c[i, 3:6]) * SIM2ANG
        lines.append(_pdb_atom(serial, "N1", resname, chain, resseq,
                                bs[0], bs[1], bs[2]))
        serial += 1

    # CONECT records for backbone chain
    for i in range(N_NTS - 1):
        a1 = 2 * i + 1      # P of nt i
        a2 = 2 * (i + 1) + 1  # P of nt i+1
        lines.append(f"CONECT{a1:5d}{a2:5d}\n")

    # CONECT records for base pairs
    for i in range(n_bp):
        j = N_NTS - 1 - i
        b1 = 2 * i + 2       # N1 of nt i
        b2 = 2 * j + 2       # N1 of nt j
        lines.append(f"CONECT{b1:5d}{b2:5d}\n")

    lines.append("END\n")
    out_path.write_text("".join(lines))
    print(f"  Wrote {out_path.name}  (n_bp={n_bp})")


# ── 3D structural figure ─────────────────────────────────────────────────────

def _draw_frame_3d(ax: "Axes3D", frame: np.ndarray, n_bp: int,
                   scale: float = SIM2ANG) -> None:
    """Draw one hairpin conformation on a 3D axes."""
    center = frame[:, :3].mean(axis=0)
    fc = frame.copy(); fc[:, :3] -= center
    frame = fc
    bb = frame[:, :3] * scale
    bs = (frame[:, :3] + POS_BASE * frame[:, 3:6]) * scale

    # — backbone tube ——————————————————————————————————————
    ax.plot(bb[:, 0], bb[:, 1], bb[:, 2],
            "-", color=BB_COLOR, lw=1.5, alpha=0.7, zorder=1)

    # — nucleotide sticks (backbone→base) ——————————————————
    for i in range(N_NTS):
        ax.plot([bb[i, 0], bs[i, 0]],
                [bb[i, 1], bs[i, 1]],
                [bb[i, 2], bs[i, 2]],
                "-", color=BASE_COLORS.get(SEQ[i], "#888"), lw=2.5,
                alpha=0.9, zorder=2)
        # base sphere
        ax.scatter(bs[i, 0], bs[i, 1], bs[i, 2],
                   s=80, c=BASE_COLORS.get(SEQ[i], "#888"),
                   edgecolors="white", linewidths=0.5, depthshade=True, zorder=3)
        # backbone dot
        ax.scatter(bb[i, 0], bb[i, 1], bb[i, 2],
                   s=18, c=BB_COLOR, alpha=0.5, depthshade=True, zorder=2)

    # — hydrogen-bond lines for formed pairs ———————————————
    for i in range(n_bp):
        j = N_NTS - 1 - i
        ax.plot([bs[i, 0], bs[j, 0]],
                [bs[i, 1], bs[j, 1]],
                [bs[i, 2], bs[j, 2]],
                "--", color=PAIR_COLOR, lw=1.5, alpha=0.8, zorder=4)

    # — residue labels for stem ends ————————————————————————
    for i in [0, STEM_LEN - 1, N_NTS - STEM_LEN, N_NTS - 1]:
        ax.text(bs[i, 0], bs[i, 1], bs[i, 2],
                f" {SEQ[i]}{i+1}", fontsize=6, color=BASE_COLORS.get(SEQ[i], "#888"))


def plot_structures_panel(frames: dict[int, np.ndarray],
                          out_path: Path) -> None:
    """6-panel 3D structural figure, one conformation per n_bp state."""
    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor("white")

    states = sorted(frames.keys())
    for idx, nbp in enumerate(states):
        ax = fig.add_subplot(2, 3, idx + 1, projection="3d")
        _draw_frame_3d(ax, frames[nbp], nbp)

        T_label = NBP_TEMP_LABEL.get(nbp, f"n_bp={nbp}")
        ax.set_title(f"n_bp = {nbp}/{STEM_LEN}\n{T_label}", fontsize=9)
        ax.set_xlabel("x (Å)", fontsize=7)
        ax.set_ylabel("y (Å)", fontsize=7)
        ax.set_zlabel("z (Å)", fontsize=7)
        ax.tick_params(labelsize=6)

        # equal aspect ratio trick
        bb = frames[nbp][:, :3] * SIM2ANG
        bs = (frames[nbp][:, :3] + POS_BASE * frames[nbp][:, 3:6]) * SIM2ANG
        all_pts = np.vstack([bb, bs])
        mid = all_pts.mean(axis=0)
        span = np.ptp(all_pts, axis=0).max() / 2 * 1.2
        ax.set_xlim(mid[0] - span, mid[0] + span)
        ax.set_ylim(mid[1] - span, mid[1] + span)
        ax.set_zlim(mid[2] - span, mid[2] + span)

    # legend
    import matplotlib.patches as mpatches
    legend_els = [mpatches.Patch(color=c, label=b)
                  for b, c in BASE_COLORS.items()]
    legend_els += [
        plt.Line2D([0], [0], color=BB_COLOR,   lw=1.5, label="Backbone"),
        plt.Line2D([0], [0], color=PAIR_COLOR, lw=1.5, ls="--", label="H-bond pair"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=6,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        f"GCGCGTTTTCGCGC Hairpin Conformations — oxDNA2 VMMC\n"
        f"(backbone P=gray line; base N1=color sphere; dashes=H-bond)",
        fontsize=11
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 3D panel: {out_path}")


def plot_single_pair(frames: dict[int, np.ndarray],
                     nbp_list: list[int],
                     out_path: Path,
                     elev: float = 25, azim: float = -50) -> None:
    """Side-by-side comparison: fully open (n_bp=0) vs fully closed (n_bp=5)."""
    sel = [n for n in nbp_list if n in frames]
    ncols = len(sel)
    fig = plt.figure(figsize=(ncols * 5.5, 5.5))
    fig.patch.set_facecolor("white")

    for idx, nbp in enumerate(sel):
        ax = fig.add_subplot(1, ncols, idx + 1, projection="3d")
        _draw_frame_3d(ax, frames[nbp], nbp)
        ax.view_init(elev=elev, azim=azim + idx * 5)
        T_label = NBP_TEMP_LABEL.get(nbp, "")
        ax.set_title(f"n_bp = {nbp}/{STEM_LEN}\n{T_label}", fontsize=10)
        ax.set_xlabel("x (Å)", fontsize=8)
        ax.set_ylabel("y (Å)", fontsize=8)
        ax.set_zlabel("z (Å)", fontsize=8)

        bb = frames[nbp][:, :3] * SIM2ANG
        bs = (frames[nbp][:, :3] + POS_BASE * frames[nbp][:, 3:6]) * SIM2ANG
        all_pts = np.vstack([bb, bs])
        mid = all_pts.mean(axis=0)
        span = np.ptp(all_pts, axis=0).max() / 2 * 1.3
        ax.set_xlim(mid[0] - span, mid[0] + span)
        ax.set_ylim(mid[1] - span, mid[1] + span)
        ax.set_zlim(mid[2] - span, mid[2] + span)

    fig.suptitle(
        "GCGCGTTTTCGCGC — Hairpin Open vs Closed Conformations",
        fontsize=12
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison figure: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    BASE = Path(__file__).parent.parent
    SIM  = BASE / "sim_out"
    FIG  = BASE / "figures"
    PDB  = BASE / "pdb_structures"
    FIG.mkdir(exist_ok=True)
    PDB.mkdir(exist_ok=True)

    traj = SIM / "traj_prod.dat"
    print(f"Scanning {traj} for representative frames...")
    frames = collect_representative_frames(traj)
    print(f"Found states: {sorted(frames.keys())}\n")

    # Write PDB files
    print("Writing PDB files:")
    for nbp, frame in sorted(frames.items()):
        pdb_path = PDB / f"hairpin_nbp{nbp}.pdb"
        write_pdb(frame, nbp, pdb_path)

    # 3D panel figure (all 6 states)
    print("\nGenerating 3D structure figures:")
    plot_structures_panel(frames, FIG / "structures_all.png")

    # Open vs closed comparison
    plot_single_pair(frames, [0, 3, 5],
                     FIG / "structures_open_vs_closed.png")

    print("\nDone. Files:")
    for p in sorted(PDB.glob("*.pdb")):
        print(f"  {p}")
    for p in sorted(FIG.glob("structures*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
