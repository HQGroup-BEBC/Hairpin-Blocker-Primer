"""Module 1: 基于primer3-py生成候选引物对"""
from __future__ import annotations

from dataclasses import dataclass

import primer3

DEFAULT_PARAMS: dict = {
    "PRIMER_OPT_SIZE": 20,
    "PRIMER_MIN_SIZE": 18,
    "PRIMER_MAX_SIZE": 25,
    "PRIMER_OPT_TM": 60.0,
    "PRIMER_MIN_TM": 57.0,
    "PRIMER_MAX_TM": 63.0,
    "PRIMER_MIN_GC": 40.0,
    "PRIMER_MAX_GC": 60.0,
    "PRIMER_MAX_POLY_X": 4,
    "PRIMER_MAX_NS_ACCEPTED": 0,
    "PRIMER_SALT_MONOVALENT": 50.0,
    "PRIMER_DNA_CONC": 50.0,
    "PRIMER_MAX_SELF_ANY_TH": 45.0,
    "PRIMER_MAX_SELF_END_TH": 35.0,
    "PRIMER_MAX_HAIRPIN_TH": 24.0,
    "PRIMER_PAIR_MAX_COMPL_ANY_TH": 45.0,
    "PRIMER_PAIR_MAX_COMPL_END_TH": 35.0,
    "PRIMER_PRODUCT_SIZE_RANGE": [[100, 300]],
    "PRIMER_NUM_RETURN": 20,
}


@dataclass
class PrimerCandidate:
    index: int
    left_seq: str
    right_seq: str
    left_start: int
    left_len: int
    right_start: int
    right_len: int
    left_tm: float
    right_tm: float
    left_gc: float
    right_gc: float
    product_size: int
    pair_penalty: float


def generate_candidates(
    template: str,
    target_region: tuple[int, int] | None = None,
    params: dict | None = None,
) -> list[PrimerCandidate]:
    """调用primer3-py，返回候选引物对列表（按primer3默认的penalty排序）"""
    global_args = dict(DEFAULT_PARAMS)
    if params:
        global_args.update(params)

    seq_args: dict = {
        "SEQUENCE_ID": "target",
        "SEQUENCE_TEMPLATE": template,
    }
    if target_region is not None:
        start, length = target_region
        seq_args["SEQUENCE_TARGET"] = [[start, length]]

    result = primer3.bindings.design_primers(seq_args=seq_args, global_args=global_args)

    num_returned = result.get("PRIMER_PAIR_NUM_RETURNED", 0)
    candidates: list[PrimerCandidate] = []
    for i in range(num_returned):
        left_start, left_len = result[f"PRIMER_LEFT_{i}"]
        right_start, right_len = result[f"PRIMER_RIGHT_{i}"]
        candidates.append(
            PrimerCandidate(
                index=i,
                left_seq=result[f"PRIMER_LEFT_{i}_SEQUENCE"],
                right_seq=result[f"PRIMER_RIGHT_{i}_SEQUENCE"],
                left_start=left_start,
                left_len=left_len,
                right_start=right_start,
                right_len=right_len,
                left_tm=result[f"PRIMER_LEFT_{i}_TM"],
                right_tm=result[f"PRIMER_RIGHT_{i}_TM"],
                left_gc=result[f"PRIMER_LEFT_{i}_GC_PERCENT"],
                right_gc=result[f"PRIMER_RIGHT_{i}_GC_PERCENT"],
                product_size=result[f"PRIMER_PAIR_{i}_PRODUCT_SIZE"],
                pair_penalty=result[f"PRIMER_PAIR_{i}_PENALTY"],
            )
        )
    return candidates
