"""oxDNA 拓扑文件生成器 (.top)

发夹阻断引物是一条单链 DNA，在溶液中自折叠形成茎环结构。
拓扑文件描述各核苷酸之间的共价连接关系（3'/5' 邻居），
不包含坐标信息（坐标在 .dat 配置文件中）。

oxDNA2 拓扑文件格式：
  第 1 行:  N_nts  N_strands
  后 N 行:  strand_id  base  3'_neighbor  5'_neighbor
            (索引 0-based，-1 表示链端无邻居)
"""
from __future__ import annotations

from pathlib import Path


_VALID_BASES = frozenset("ACGTacgt")


def write_topology(seq: str, path: str | Path) -> None:
    """将发夹引物序列写入 oxDNA 拓扑文件。

    参数:
      seq:  全长发夹引物序列 (5'→3')
      path: 输出 .top 文件路径

    拓扑约定：
      · 单条链 (strand_id = 1)，5' 端为 nt 0，3' 端为 nt n-1
      · 折叠态下 nt i 与 nt (n-1-i) 形成 Watson-Crick 碱基对 (i = 0..stem-1)
        但拓扑文件只记录共价连接，不记录氢键——氢键由 oxDNA 势能自动建模
    """
    seq = seq.upper()
    bad = set(seq) - _VALID_BASES
    if bad:
        raise ValueError(f"序列含非法字符: {bad}")

    n = len(seq)
    lines: list[str] = [f"{n} 1\n"]  # N_nts N_strands

    for i, base in enumerate(seq):
        three_prime = i + 1 if i < n - 1 else -1
        five_prime  = i - 1 if i > 0     else -1
        lines.append(f"1 {base} {three_prime} {five_prime}\n")

    with open(path, "w") as f:
        f.writelines(lines)


def read_topology(path: str | Path) -> tuple[int, str]:
    """读取 oxDNA 拓扑文件，返回 (n_nts, sequence_5to3)。"""
    lines = Path(path).read_text().splitlines()
    n_nts = int(lines[0].split()[0])
    seq = "".join(line.split()[1] for line in lines[1: n_nts + 1])
    return n_nts, seq
