"""非目标模板库中的脱靶结合位点搜索（种子精确匹配 + 3'端位置加权错配评分）"""
from __future__ import annotations

import re

from Bio.Seq import Seq

# varVAMP风格位置加权：在种子区之外的"可变区"中，越靠近种子边界（即越
# 接近引物3'锚定区域）的位置错配权重越高，向引物5'尾部递减，参考varVAMP
# 对3'端最后5个位置施加32/16/8/4/2倍错配惩罚的设计
_3PRIME_WEIGHTS = (32, 16, 8, 4, 2)


def _revcomp(seq: str) -> str:
    return str(Seq(seq).reverse_complement())


def _hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def _position_weights(plen: int, seed_len: int) -> list[int]:
    """构造长度为plen的逐位置错配权重（primer_rc坐标，position 0 = 引物
    3'最末端）。种子区[0, seed_len)由精确匹配保证不会产生错配；种子区之外
    紧邻种子边界的`len(_3PRIME_WEIGHTS)`个位置按_3PRIME_WEIGHTS递减加权，
    更远的位置权重为1。"""
    weights = [1] * plen
    for offset, w in enumerate(_3PRIME_WEIGHTS):
        idx = seed_len + offset
        if idx < plen:
            weights[idx] = w
    return weights


def _mismatch_score(window: str, ref: str, weights: list[int]) -> tuple[int, int]:
    """返回 (原始错配数, 位置加权错配得分)。

    错配集中在远离3'锚定种子的引物5'尾部（加权得分低）意味着该脱靶位点
    3'端附近仍能与引物较好配对——聚合酶可能在此错误延伸，是发夹阻断设计
    更需要针对的"危险"脱靶窗口；错配靠近3'锚定种子边界（加权得分高）的
    位点对该核心区域干扰更大，优先级较低。"""
    raw = 0
    weighted = 0
    for i, (a, b) in enumerate(zip(window, ref)):
        if a != b:
            raw += 1
            weighted += weights[i]
    return raw, weighted


def find_offtarget_window(
    primer_seq: str,
    offtarget_records: list[tuple[str, str]],
    max_mismatch: int = 3,
    seed_len: int = 10,
) -> str | None:
    """在非目标库中搜索primer_seq最可能的脱靶结合位点。

    返回与primer_seq等长的"模板侧"序列窗口，使得
    calc_heterodimer(primer_seq, window) 直接给出该脱靶结合的ΔG
    （与完全匹配时 window == reverse_complement(primer_seq) 同一约定）。
    找不到错配数<=max_mismatch的位点时返回None（视为无显著脱靶）。

    在错配数<=max_mismatch的候选窗口中，按位置加权错配得分
    （`_mismatch_score`）取得分最低者——而非简单的最小Hamming距离——
    优先选出错配集中在引物5'尾部、3'端附近完整的"危险"脱靶位点。
    """
    plen = len(primer_seq)
    seed_len = min(seed_len, plen)
    primer_rc = _revcomp(primer_seq)
    weights = _position_weights(plen, seed_len)

    # 引物3'端是特异性最关键的区域，以它作为搜索种子
    seed_for_rc_match = _revcomp(primer_seq[-seed_len:])  # window起点 = 种子命中位置
    seed_for_direct_match = primer_seq[-seed_len:]  # window终点 = 种子命中位置 + seed_len

    best_weighted: int | None = None
    best_window: str | None = None

    for _record_id, ref_seq in offtarget_records:
        ref_len = len(ref_seq)
        if ref_len < plen:
            continue

        # 情形1：参考序列上的窗口 ~ reverse_complement(primer_seq)（同链结合）
        # window与primer_rc同向对齐：position 0 = 引物3'最末端碱基
        for m in re.finditer(f"(?={re.escape(seed_for_rc_match)})", ref_seq):
            start = m.start()
            end = start + plen
            if end > ref_len:
                continue
            window = ref_seq[start:end]
            raw, weighted = _mismatch_score(window, primer_rc, weights)
            if raw <= max_mismatch and (best_weighted is None or weighted < best_weighted):
                best_weighted, best_window = weighted, window

        # 情形2：参考序列上的窗口 ~ primer_seq（引物结合到参考序列的互补链）
        # window与primer_seq同向对齐（position 0 = 引物5'端）；将其反向互补后
        # 与primer_rc同向对齐（position 0 = 引物3'最末端碱基），与情形1一致
        for m in re.finditer(f"(?={re.escape(seed_for_direct_match)})", ref_seq):
            seed_start = m.start()
            start = seed_start + seed_len - plen
            end = seed_start + seed_len
            if start < 0 or end > ref_len:
                continue
            window = ref_seq[start:end]
            window_rc = _revcomp(window)
            raw, weighted = _mismatch_score(window_rc, primer_rc, weights)
            if raw <= max_mismatch and (best_weighted is None or weighted < best_weighted):
                best_weighted, best_window = weighted, window_rc

    return best_window
