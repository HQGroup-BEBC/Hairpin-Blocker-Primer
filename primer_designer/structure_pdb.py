"""DNA发夹阻断引物 PDB 结构文件生成器

几何参数来源: Arnott & Hukins (1972) Biochem Biophys Res Commun 47, 1504-1509
                (标准 B-form DNA 纤维衍射模型)

B-form DNA 关键参数:
  · 螺旋上升 (rise): 3.38 Å / bp
  · 螺旋扭转 (twist): 36.0° / bp  (10 bp / 圈)
  · 螺旋直径: ~20 Å (磷酸基外径 ~18 Å)
  · 糖苷角 (glycosidic): anti 构型
  · 沟: 大沟 22 Å / 小沟 12 Å (Å宽度)

输出原子 (每个核苷酸):
  P   — 磷原子 (骨架核心，用于定位螺旋轴)
  O5' — 5'氧
  C5' — 5'碳
  C4' — 脱氧核糖 C4' (连接骨架与碱基)
  C3' — 3'碳
  O3' — 3'氧 (连接到下一个磷酸基)

PDB 链设计:
  Chain A  — stem_comp(茎) + loop + primer_body(引物主体)  [5'→3']
  Chain B  — primer_3'stem(茎的另一条链)                  [3'→5', 反向平行]
  茎区配对: A residue i ↔ B residue i (对应 hairpin 中 pos i 和 pos n-1-i)

可视化建议 (py3Dmol):
  茎-stem: cartoon / sphere
  单链区:  line / stick
  碱基对:  CONECT records 形成横档连线
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# B-DNA Arnott 骨架原子极坐标 (r [Å], θ [°], Δz [Å])
# θ 以右手螺旋轴 z 为中心，相对于大沟中线定义
# Δz 以碱基对平面 z=0 为基准
# ---------------------------------------------------------------------------
_BACKBONE: list[tuple[str, float, float, float]] = [
    # name      r(Å)   θ(°)    Δz(Å)   — 相对于该碱基对中心
    ("P",      8.91, -139.0,  0.00),
    ("O5'",    8.06, -139.0,  0.72),
    ("C5'",    7.11, -131.0,  1.25),
    ("C4'",    6.17, -113.0,  1.83),
    ("C3'",    5.31,  -68.0,  2.88),
    ("O3'",    6.21,  -50.0,  3.38),   # O3' 位于下一个 bp 的 z=0 处
]

# 元素符号映射
_ELEM: dict[str, str] = {
    "P": "P", "O5'": "O", "C5'": "C",
    "C4'": "C", "C3'": "C", "O3'": "O",
}

# 碱基 → 三字符残基名 (DNA)
_RES: dict[str, str] = {"A": " DA", "T": " DT", "G": " DG", "C": " DC"}

# 互补碱基
_COMP: dict[str, str] = {"A": "T", "T": "A", "G": "C", "C": "G"}


@dataclass
class _Atom:
    serial: int
    name: str
    res_name: str
    chain: str
    res_seq: int
    x: float
    y: float
    z: float
    element: str

    def pdb_line(self) -> str:
        name_field = f" {self.name:<3s}" if len(self.name) < 4 else self.name[:4]
        return (
            f"ATOM  {self.serial:5d} {name_field} {self.res_name:3s} "
            f"{self.chain}{self.res_seq:4d}    "
            f"{self.x:8.3f}{self.y:8.3f}{self.z:8.3f}"
            f"  1.00  0.00          {self.element:>2s}\n"
        )


def _helix_xyz(
    r: float, theta_deg: float, dz: float,
    bp_idx: int,
    start_phase_deg: float = 0.0,
    twist_deg: float = 36.0,
    rise: float = 3.38,
) -> tuple[float, float, float]:
    """将 B-DNA 极坐标转换为全局笛卡尔坐标。

    bp_idx:          碱基对编号 (0-based)，决定 z 高度和螺旋相位
    start_phase_deg: 起始相位（chain A 用 0°，chain B 用 180°）
    """
    phi = math.radians(theta_deg + start_phase_deg + bp_idx * twist_deg)
    x = r * math.cos(phi)
    y = r * math.sin(phi)
    z = bp_idx * rise + dz
    return x, y, z


def _single_strand_arc_xyz(
    k: int,
    n_ss: int,
    z_top: float,
    phi_top: float,
    r_helix: float = 6.17,
) -> tuple[float, float, float]:
    """单链弧区 (loop + primer_body) 的骨架坐标。

    k:        弧区内的核苷酸编号 (0-based)
    n_ss:     弧区总核苷酸数
    z_top:    茎区最内侧碱基对的 z 坐标 (弧的起始 z)
    phi_top:  茎区最内侧碱基对的螺旋相位 (弧的起始角)
    r_helix:  茎区 C4' 的螺旋半径 (弧的起始半径)

    弧路径: 同时在 φ 方向旋转 π, 向外展开 arc_r, 上升 arc_h
    使弧的首尾两端恰好与双链茎顶部两链的 C4' 位置平滑衔接。
    """
    t = (k + 1) / (n_ss + 1)          # t ∈ (0, 1), 两端不含
    arc_r = max(4.0, n_ss * 0.9)      # Å, 向外伸展
    arc_h = max(7.0, n_ss * 1.8)      # Å, 向上抬升

    phi = phi_top + math.pi * t
    r   = r_helix + arc_r * math.sin(math.pi * t)
    z   = z_top   + arc_h * math.sin(math.pi * t)
    x   = r * math.cos(phi)
    y   = r * math.sin(phi)
    return x, y, z


def hairpin_to_pdb(design) -> str:
    """将 HairpinDesign 转换为 PDB 格式字符串。

    返回可直接写入 .pdb 文件或传给 py3Dmol 的字符串。
    Chain A: 全长引物序列 (5'→3')，茎区骨架为 B-form 双螺旋构型
    Chain B: 引物 3' 茎的配对链 (反向平行)，残基编号与 Chain A 茎区一一对应
    """
    seq  = design.hairpin_primer_seq
    n    = len(seq)
    s    = design.stem_len
    l    = design.loop_len
    n_pb = n - 2 * s - l
    n_ss = l + n_pb

    RISE  = 3.38
    TWIST = 36.0
    PHI_A = 0.0       # Chain A 起始相位
    PHI_B = 180.0     # Chain B 起始相位 (反向平行)

    atoms: list[_Atom] = []
    serial = 1

    # ----- CONECT 记录收集 (骨架键 + 碱基对横档键) -----
    connect: list[tuple[int, int]] = []

    # ── Chain A: stem_comp (B-form 茎区, 5'→3' 向上) ──────────────────
    a_stem_serials: list[list[int]] = []   # 每个 bp 的原子 serial 列表
    for i in range(s):
        bp_serials: list[int] = []
        for aname, r, theta, dz in _BACKBONE:
            x, y, z = _helix_xyz(r, theta, dz, i, PHI_A, TWIST, RISE)
            res_name = _RES.get(seq[i], " DN")
            elem     = _ELEM[aname]
            atoms.append(_Atom(serial, aname, res_name, "A", i + 1, x, y, z, elem))
            bp_serials.append(serial)
            serial += 1
        # 骨架内键 (同残基内顺序连接)
        for j in range(len(bp_serials) - 1):
            connect.append((bp_serials[j], bp_serials[j + 1]))
        # 与上一残基的 O3'→P 跨残基键
        if a_stem_serials:
            connect.append((a_stem_serials[-1][-1], bp_serials[0]))
        a_stem_serials.append(bp_serials)

    # ── Chain B: primer_3stem (B-form 茎区, 反向平行) ─────────────────
    b_stem_serials: list[list[int]] = []
    for i in range(s):
        bp_serials: list[int] = []
        seq_idx = n - 1 - i      # primer_3stem 中 bp i 对应序列位置 n-1-i
        for aname, r, theta, dz in _BACKBONE:
            x, y, z = _helix_xyz(r, theta, dz, i, PHI_B, TWIST, RISE)
            res_name = _RES.get(seq[seq_idx], " DN")
            elem     = _ELEM[aname]
            atoms.append(_Atom(serial, aname, res_name, "B", i + 1, x, y, z, elem))
            bp_serials.append(serial)
            serial += 1
        for j in range(len(bp_serials) - 1):
            connect.append((bp_serials[j], bp_serials[j + 1]))
        if b_stem_serials:
            connect.append((b_stem_serials[-1][-1], bp_serials[0]))
        b_stem_serials.append(bp_serials)

    # ── Chain A 碱基对横档: A bp_i ↔ B bp_i (C4' ↔ C4') ──────────────
    # C4' 是 _BACKBONE 中第4个 (index 3)
    C4_IDX = 3
    for i in range(s):
        s1_c4 = a_stem_serials[i][C4_IDX]
        s2_c4 = b_stem_serials[i][C4_IDX]
        connect.append((s1_c4, s2_c4))

    # ── Chain A: loop + primer_body (单链弧区) ─────────────────────────
    # _BACKBONE 元组格式: (name, r, theta, dz)
    z_top   = (s - 1) * RISE + _BACKBONE[C4_IDX][3]   # 最内层 C4' 的 z (dz 分量)
    phi_top = math.radians(PHI_A + (s - 1) * TWIST + _BACKBONE[C4_IDX][2])  # 最内层 C4' 的 φ (theta)
    r_c4    = _BACKBONE[C4_IDX][1]   # C4' 的螺旋半径 (r 分量)

    arc_serials: list[int] = []
    for k in range(n_ss):
        x, y, z = _single_strand_arc_xyz(k, n_ss, z_top, phi_top, r_c4)
        seq_idx = s + k
        res_name = _RES.get(seq[seq_idx], " DN")
        # 单链区只放置 C4' 和 P (两个代表性原子)
        for aname, dx, dy, dz in [("C4'", 0, 0, 0), ("P", 1.5, 0, -1.5)]:
            elem = _ELEM[aname]
            atoms.append(_Atom(serial, aname, res_name, "A", s + k + 1,
                               x + dx, y + dy, z + dz, elem))
            arc_serials.append(serial)
            serial += 1
        # 连接到前一个残基
        if k == 0 and a_stem_serials:
            connect.append((a_stem_serials[-1][-1], arc_serials[-2]))
        elif k > 0:
            connect.append((arc_serials[-4], arc_serials[-2]))  # C4'→C4'

    # ── PDB 文件头 ─────────────────────────────────────────────────────
    lines: list[str] = [
        "REMARK  Hairpin-Blocker Primer — Generated by Hairpin-Blocker Designer\n",
        f"REMARK  Sequence (5'->3'): {seq}\n",
        f"REMARK  stem={s}bp  loop={l}nt  primer_body={n_pb}nt\n",
        f"REMARK  dG_hairpin={design.dg_hairpin:.2f} kcal/mol  "
        f"SI={design.si:.2f}  EI={design.ei:.2f}\n",
        f"REMARK  Geometry: B-form DNA, Arnott & Hukins (1972)\n",
        f"REMARK  Chain A: full hairpin primer (5'->3'); "
        f"Chain B: complementary 3'stem (antiparallel)\n",
        "REMARK  Visualization: py3Dmol / PyMOL / UCSF Chimera\n",
    ]

    # SEQRES
    def seqres(chain, length, bases):
        atoms_str = "  ".join(f"D{b}" for b in bases)
        return f"SEQRES   1 {chain}  {length:3d}  {atoms_str}\n"

    lines.append(seqres("A", n, seq))
    lines.append(seqres("B", s, [_COMP.get(seq[n-1-i], "N") for i in range(s)]))

    # ATOM records
    for atom in atoms:
        lines.append(atom.pdb_line())

    # CONECT records (骨架 + 碱基对横档)
    for s1, s2 in connect:
        lines.append(f"CONECT{s1:5d}{s2:5d}\n")

    lines.append("END\n")
    return "".join(lines)


def save_pdb(design, path: str) -> str:
    """将发夹设计输出为 PDB 文件，返回 PDB 字符串。"""
    content = hairpin_to_pdb(design)
    with open(path, "w") as f:
        f.write(content)
    return content
