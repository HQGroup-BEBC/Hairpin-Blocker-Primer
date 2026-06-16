"""端到端自测脚本：加载示例数据，运行完整设计流程，打印每对引物的
ΔG_target/ΔG_hairpin/ΔG_offtarget/SI/EI/判定，并验证阈值方向：
- 无脱靶命中的引物对 SI 应被封顶为 excellent
- 含1bp错配脱靶的引物 SI 应明显低于含2bp错配脱靶的引物（错配越少，脱靶
  结合越强，发夹越难以胜出）
"""
from __future__ import annotations

from primer_designer.pipeline import run_design
from primer_designer.sequence_io import load_fasta_file

TARGET_FASTA = "sample_data/target_example.fasta"
OFFTARGET_FASTA = "sample_data/offtarget_example.fasta"


def main() -> None:
    target_records = load_fasta_file(TARGET_FASTA)
    offtarget_records = load_fasta_file(OFFTARGET_FASTA)
    _, template = target_records[0]

    pairs, front = run_design(template, offtarget_records=offtarget_records)

    print(f"候选引物对总数: {len(pairs)}")
    print(f"Pareto前沿数量: {len(front)}")
    print()
    header = (
        f"{'idx':>3} {'left_seq':<22} {'right_seq':<22} "
        f"{'L_dgT':>7} {'L_dgH':>7} {'L_dgO':>7} {'L_SI':>6} "
        f"{'R_dgT':>7} {'R_dgH':>7} {'R_dgO':>7} {'R_SI':>6} "
        f"{'pair_SI':>7} {'pair_EI':>7} {'verdict':>10}"
    )
    print(header)
    for p in pairs:
        c, l, r = p.candidate, p.left, p.right
        print(
            f"{c.index:>3} {c.left_seq:<22} {c.right_seq:<22} "
            f"{l.dg_target:7.2f} {l.dg_hairpin:7.2f} {l.dg_offtarget:7.2f} {l.si:6.2f} "
            f"{r.dg_target:7.2f} {r.dg_hairpin:7.2f} {r.dg_offtarget:7.2f} {r.si:6.2f} "
            f"{p.pair_si:7.2f} {p.pair_ei:7.2f} "
            f"{min(l.overall_verdict, r.overall_verdict, key=lambda v: ['reject','acceptable','excellent'].index(v)):>10}"
        )

    print()
    print("Pareto前沿 (按pair_si降序):")
    for p in front:
        c = p.candidate
        print(f"  idx={c.index} left={c.left_seq} right={c.right_seq} pair_SI={p.pair_si:.2f} pair_EI={p.pair_ei:.2f}")

    # --- 校验 ---
    by_idx = {p.candidate.index: p for p in pairs}

    # 候选0-3的左引物 GCTCTATCCCGGCGGTATTC 在脱靶库中无命中 -> SI应被封顶为5.0(excellent)
    no_off_pairs = [p for p in pairs if p.candidate.left_seq == "GCTCTATCCCGGCGGTATTC"]
    assert no_off_pairs, "未找到预期的无脱靶候选(0-3)"
    for p in no_off_pairs:
        assert p.left.si == 5.0, f"idx={p.candidate.index} 无脱靶时左引物SI应封顶为5.0, 实际={p.left.si}"
        assert p.left.verdict_si == "excellent"
    print("\n[OK] 无脱靶命中的引物SI被正确封顶为excellent")

    # 候选4: 左引物含1bp错配脱靶, 右引物含2bp错配脱靶 -> 右引物SI应明显高于左引物SI
    p4 = by_idx.get(4)
    assert p4 is not None, "未找到候选4"
    assert p4.left.dg_offtarget < 0, "候选4左引物应命中1bp错配脱靶"
    assert p4.right.dg_offtarget < 0, "候选4右引物应命中2bp错配脱靶"
    assert p4.left.si < p4.right.si, (
        f"1bp错配脱靶(SI={p4.left.si:.2f})应比2bp错配脱靶(SI={p4.right.si:.2f})更难被发夹胜出(SI更低)"
    )
    print("[OK] 1bp错配脱靶的SI低于2bp错配脱靶的SI（脱靶结合越强，越难被发夹竞争掉）")

    print("\n全部校验通过")


if __name__ == "__main__":
    main()
