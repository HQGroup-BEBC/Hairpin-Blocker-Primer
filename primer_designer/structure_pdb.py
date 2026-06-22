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

HEG 非核苷酸连接子 (Hexaethylene Glycol):
  以 HETATM 记录、残基名 XHE 表示单个代表原子 O1
  位于 loop 和 primer_body 弧区之间，以虚线标记连接关系

PDB 链设计:
  Chain A  — stem_comp(茎) + loop + [HEG] + primer_body(引物主体)  [5'→3']
  Chain B  — primer_3'stem(茎的另一条链)                           [3'→5', 反向平行]
  茎区配对: A residue i ↔ B residue i (对应 hairpin 中 pos i 和 pos n-1-i)

可视化建议 (py3Dmol / PyMOL):
  茎-stem: cartoon / sphere (蓝色)
  loop:    line / stick (绿色)
  HEG:     sphere, 橙色, HETATM XHE
  primer_body: stick / ball (红色)
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
    record_type: str = "ATOM"

    def pdb_line(self) -> str:
        record = f"{self.record_type:<6s}"
        name_field = f" {self.name:<3s}" if len(self.name) < 4 else self.name[:4]
        return (
            f"{record}{self.serial:5d} {name_field} {self.res_name:3s} "
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
    """单链弧区 (loop + HEG + primer_body) 的骨架坐标。

    k:        弧区内的位置编号 (0-based)
    n_ss:     弧区总位置数 (loop_len + 1_HEG + n_pb)
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

    支持含 HEG 非核苷酸连接子的发夹阻断引物：
      hairpin_domain = stem_comp + loop  (Chain A 茎区 + 弧区前段)
      HEG            = 1个 HETATM XHE 原子 (弧区中段)
      primer_body    = 原始引物序列        (弧区后段)
    Chain B: 引物 3' 茎配对链 (反向平行)

    优先使用 design.hairpin_domain_seq / design.primer_body_seq；
    若不存在则回退到解析 design.hairpin_primer_seq 中的 [HEG] 标记。
    """
    # --- 提取各区段序列 ---
    if hasattr(design, "hairpin_domain_seq") and hasattr(design, "primer_body_seq"):
        hairpin_domain = design.hairpin_domain_seq  # stem_comp + loop
        primer_body    = design.primer_body_seq
    else:
        # 兼容旧版：解析 [HEG] 标记
        raw = design.hairpin_primer_seq
        if "[HEG]" in raw:
            hairpin_domain, primer_body = raw.split("[HEG]", 1)
        else:
            hairpin_domain = raw
            primer_body = ""

    s    = design.stem_len
    l    = design.loop_len
    n_pb = len(primer_body)
    # 弧区总位置数：loop(l) + HEG(1) + primer_body(n_pb)
    n_ss = l + 1 + n_pb

    # 整体序列长度（用于 Chain B 定位和 SEQRES）
    full_dna = hairpin_domain + primer_body   # DNA only, no HEG marker

    RISE  = 3.38
    TWIST = 36.0
    PHI_A = 0.0
    PHI_B = 180.0

    atoms: list[_Atom] = []
    serial = 1
    connect: list[tuple[int, int]] = []

    # ── Chain A: stem_comp (B-form 茎区, 5'→3') ───────────────────────
    a_stem_serials: list[list[int]] = []
    for i in range(s):
        bp_serials: list[int] = []
        for aname, r, theta, dz in _BACKBONE:
            x, y, z = _helix_xyz(r, theta, dz, i, PHI_A, TWIST, RISE)
            res_name = _RES.get(hairpin_domain[i], " DN")
            elem     = _ELEM[aname]
            atoms.append(_Atom(serial, aname, res_name, "A", i + 1, x, y, z, elem))
            bp_serials.append(serial)
            serial += 1
        for j in range(len(bp_serials) - 1):
            connect.append((bp_serials[j], bp_serials[j + 1]))
        if a_stem_serials:
            connect.append((a_stem_serials[-1][-1], bp_serials[0]))
        a_stem_serials.append(bp_serials)

    # ── Chain B: primer_3'stem (B-form 茎区, 反向平行) ────────────────
    b_stem_serials: list[list[int]] = []
    for i in range(s):
        bp_serials: list[int] = []
        # primer_body 的 3' 端 stem 配对: 3'端最后s个碱基
        seq_idx_in_body = n_pb - 1 - i
        base = primer_body[seq_idx_in_body] if 0 <= seq_idx_in_body < n_pb else "N"
        for aname, r, theta, dz in _BACKBONE:
            x, y, z = _helix_xyz(r, theta, dz, i, PHI_B, TWIST, RISE)
            res_name = _RES.get(base, " DN")
            elem     = _ELEM[aname]
            atoms.append(_Atom(serial, aname, res_name, "B", i + 1, x, y, z, elem))
            bp_serials.append(serial)
            serial += 1
        for j in range(len(bp_serials) - 1):
            connect.append((bp_serials[j], bp_serials[j + 1]))
        if b_stem_serials:
            connect.append((b_stem_serials[-1][-1], bp_serials[0]))
        b_stem_serials.append(bp_serials)

    # ── Chain A 碱基对横档 (C4' ↔ C4') ──────────────────────────────
    C4_IDX = 3
    for i in range(s):
        connect.append((a_stem_serials[i][C4_IDX], b_stem_serials[i][C4_IDX]))

    # ── Chain A: 单链弧区 (loop + HEG + primer_body) ──────────────────
    z_top   = (s - 1) * RISE + _BACKBONE[C4_IDX][3]
    phi_top = math.radians(PHI_A + (s - 1) * TWIST + _BACKBONE[C4_IDX][2])
    r_c4    = _BACKBONE[C4_IDX][1]

    arc_serials: list[int] = []  # C4' serial of each arc position
    res_offset = s + 1           # residue numbering offset for arc (1-based)

    for k in range(n_ss):
        x, y, z = _single_strand_arc_xyz(k, n_ss, z_top, phi_top, r_c4)
        res_num = res_offset + k

        if k < l:
            # Loop region (nucleotides)
            base = hairpin_domain[s + k]
            res_name = _RES.get(base, " DN")
            for aname, dx, dy, dz_off in [("C4'", 0, 0, 0), ("P", 1.5, 0, -1.5)]:
                elem = _ELEM[aname]
                atoms.append(_Atom(serial, aname, res_name, "A", res_num,
                                   x + dx, y + dy, z + dz_off, elem, "ATOM"))
                if aname == "C4'":
                    arc_serials.append(serial)
                serial += 1

        elif k == l:
            # HEG residue — single HETATM O1 atom
            atoms.append(_Atom(serial, "O1", "XHE", "A", res_num,
                               x, y, z, "O", "HETATM"))
            arc_serials.append(serial)
            serial += 1

        else:
            # Primer body region (nucleotides), k > l
            body_idx = k - l - 1
            base = primer_body[body_idx] if body_idx < n_pb else "N"
            res_name = _RES.get(base, " DN")
            for aname, dx, dy, dz_off in [("C4'", 0, 0, 0), ("P", 1.5, 0, -1.5)]:
                elem = _ELEM[aname]
                atoms.append(_Atom(serial, aname, res_name, "A", res_num,
                                   x + dx, y + dy, z + dz_off, elem, "ATOM"))
                if aname == "C4'":
                    arc_serials.append(serial)
                serial += 1

        # Connect arc positions
        if k == 0 and a_stem_serials:
            connect.append((a_stem_serials[-1][-1], arc_serials[-1]))
        elif k > 0:
            connect.append((arc_serials[-2], arc_serials[-1]))

    # ── PDB 文件头 ─────────────────────────────────────────────────────
    heg_display = design.hairpin_primer_seq if hasattr(design, "hairpin_primer_seq") else (
        hairpin_domain + "[HEG]" + primer_body
    )
    lines: list[str] = [
        "REMARK  Hairpin-Blocker Primer — Generated by Hairpin-Blocker Designer\n",
        f"REMARK  Sequence (5'->3'): {heg_display}\n",
        f"REMARK  stem={s}bp  loop={l}nt  HEG=1  primer_body={n_pb}nt\n",
        f"REMARK  dG_hairpin={design.dg_hairpin:.2f} kcal/mol  "
        f"SI={design.si:.2f}  EI={design.ei:.2f}\n",
        f"REMARK  Geometry: B-form DNA, Arnott & Hukins (1972)\n",
        f"REMARK  HEG: HETATM residue XHE between loop and primer_body\n",
        f"REMARK  Chain A: hairpin_domain+HEG+primer_body (5'->3'); "
        f"Chain B: complementary 3'stem (antiparallel)\n",
        "REMARK  Visualization: py3Dmol / PyMOL / UCSF Chimera\n",
        "REMARK  Color scheme: stem=blue, loop=green, HEG=orange, primer_body=red\n",
    ]

    # SEQRES (DNA bases only, HEG excluded per PDB convention)
    def seqres(chain: str, length: int, bases: list[str]) -> str:
        atoms_str = "  ".join(f"D{b}" for b in bases)
        return f"SEQRES   1 {chain}  {length:3d}  {atoms_str}\n"

    chain_a_bases = list(hairpin_domain) + list(primer_body)
    lines.append(seqres("A", len(chain_a_bases), chain_a_bases))
    comp_bases = [_COMP.get(primer_body[n_pb - 1 - i], "N") for i in range(s)]
    lines.append(seqres("B", s, comp_bases))

    # ATOM / HETATM records
    for atom in atoms:
        lines.append(atom.pdb_line())

    # CONECT records
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
