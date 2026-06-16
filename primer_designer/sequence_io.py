"""FASTA序列读取与校验"""
from __future__ import annotations

import re

from Bio import SeqIO

_VALID_BASES = set("ACGTN")


def validate_sequence(seq: str) -> str:
    """清理并校验DNA序列，仅允许 A/C/G/T/N（不区分大小写）"""
    cleaned = re.sub(r"\s+", "", seq).upper()
    if not cleaned:
        raise ValueError("序列为空")
    invalid = set(cleaned) - _VALID_BASES
    if invalid:
        raise ValueError(
            f"序列中含有非法字符: {''.join(sorted(invalid))}（仅支持 A/C/G/T/N）"
        )
    return cleaned


def parse_fasta_text(text: str) -> list[tuple[str, str]]:
    """解析粘贴的FASTA或纯序列文本，返回 [(record_id, sequence), ...]"""
    text = text.strip()
    if not text:
        return []

    if not text.startswith(">"):
        return [("input_sequence", validate_sequence(text))]

    records: list[tuple[str, str]] = []
    current_id: str | None = None
    current_seq: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id is not None:
                records.append((current_id, validate_sequence("".join(current_seq))))
            header = line[1:].strip().split()
            current_id = header[0] if header else f"seq{len(records) + 1}"
            current_seq = []
        else:
            current_seq.append(line)
    if current_id is not None:
        records.append((current_id, validate_sequence("".join(current_seq))))
    return records


def load_fasta_file(path: str) -> list[tuple[str, str]]:
    """从文件加载FASTA，返回 [(record_id, sequence), ...]

    显式以UTF-8打开文件再交给SeqIO.parse，避免Windows系统默认按
    本地编码(如gbk)打开含中文描述的FASTA文件时报UnicodeDecodeError。
    """
    with open(path, encoding="utf-8") as handle:
        records = [(rec.id, validate_sequence(str(rec.seq))) for rec in SeqIO.parse(handle, "fasta")]
    if not records:
        raise ValueError("文件中未找到有效的FASTA序列")
    return records
