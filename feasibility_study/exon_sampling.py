"""GENCODE外显子随机采样工具，为基因组可行性验证生成均衡样本"""
from __future__ import annotations

import random
from Bio import SeqIO


def sample_exons(
    gencode_fasta: str,
    n_samples: int = 1000,
    min_len: int = 100,
    max_len: int = 300,
    random_seed: int = 42,
) -> list:
    """从GENCODE外显子FASTA文件随机采样。

    参数:
      gencode_fasta: GENCODE外显子FASTA文件路径
      n_samples: 采样数量
      min_len: 最小外显子长度 (bp)
      max_len: 最大外显子长度 (bp)
      random_seed: 随机种子（保证可重复性）

    返回: SeqRecord列表
    """
    records = [
        r for r in SeqIO.parse(gencode_fasta, "fasta")
        if min_len <= len(r.seq) <= max_len
        and all(c in "ACGTNacgtn" for c in str(r.seq))
    ]
    if not records:
        raise ValueError(f"No qualifying exons found in {gencode_fasta}")

    random.seed(random_seed)
    n = min(n_samples, len(records))
    return random.sample(records, n)


def gc_content(seq: str) -> float:
    seq = seq.upper()
    return sum(1 for c in seq if c in "GC") / max(len(seq), 1)


def bin_by_gc(
    records: list,
    bins: list[tuple[float, float]] | None = None,
) -> dict[str, list]:
    """按GC含量分箱，返回分层样本字典。

    默认三档：低GC(<45%)、中GC(45-55%)、高GC(>55%)
    """
    if bins is None:
        bins = [(0.0, 0.45), (0.45, 0.55), (0.55, 1.0)]
    labels = ["low_gc", "mid_gc", "high_gc"]
    result: dict[str, list] = {label: [] for label in labels}
    for r in records:
        gc = gc_content(str(r.seq))
        for (lo, hi), label in zip(bins, labels):
            if lo <= gc < hi:
                result[label].append(r)
                break
    return result
