"""Module 4: 引物对组装与SI-EI Pareto前沿筛选"""
from __future__ import annotations

from dataclasses import dataclass

from .candidate_generator import PrimerCandidate
from .hairpin_designer import HairpinDesign


@dataclass
class HairpinBlockerPair:
    candidate: PrimerCandidate
    left: HairpinDesign
    right: HairpinDesign
    pair_si: float
    pair_ei: float


def make_pair(candidate: PrimerCandidate, left: HairpinDesign, right: HairpinDesign) -> HairpinBlockerPair:
    """以左右引物中较差的一侧（min）代表整对引物的SI/EI"""
    return HairpinBlockerPair(
        candidate=candidate,
        left=left,
        right=right,
        pair_si=min(left.si, right.si),
        pair_ei=min(left.ei, right.ei),
    )


def pareto_front(pairs: list[HairpinBlockerPair]) -> list[HairpinBlockerPair]:
    """SI-EI二维非支配解筛选（两个指标均越大越好），按pair_si降序返回"""
    front = []
    for p in pairs:
        dominated = any(
            q.pair_si >= p.pair_si
            and q.pair_ei >= p.pair_ei
            and (q.pair_si > p.pair_si or q.pair_ei > p.pair_ei)
            for q in pairs
            if q is not p
        )
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda p: p.pair_si, reverse=True)
