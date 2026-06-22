"""基因组尺度可行性验证脚本

在人类基因组真实外显子序列中验证Hairpin-Blocker设计的普适性。
回答核心问题：ΔG_target ≪ ΔG_hairpin < ΔG_offtarget 的约束
在真实基因组序列空间中有多大比例的靶点可以满足？

统计结果作为论文 Supplementary Figure 1，直接回应审稿人对普适性的质疑。

用法:
  python -m feasibility_study.genome_feasibility --gencode gencode_v38_exons.fa --n 1000
  python feasibility_study/genome_feasibility.py --gencode /path/to/exons.fa
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from primer_designer import pipeline
from feasibility_study.exon_sampling import sample_exons, bin_by_gc


def run_feasibility_study(
    gencode_fasta: str,
    n_samples: int = 1000,
    random_seed: int = 42,
    use_ai_search: bool = False,
    verbose: bool = True,
) -> dict:
    """在GENCODE外显子随机样本上运行可行性评估。

    参数:
      gencode_fasta: GENCODE外显子FASTA路径 (100-300 bp外显子)
      n_samples: 采样外显子数量
      random_seed: 随机种子
      use_ai_search: 是否启用GNN加速（较慢但更全面）
      verbose: 是否打印进度

    返回字典:
      feasibility_rate:  可行靶点比例 (SI>0.8 且 EI>0.5)
      optimal_rate:      最优靶点比例 (SI>1.5 且 EI>1.0)
      total:             总评估靶点数
      feasible:          可行靶点数
      optimal:           最优靶点数
      failed:            无法设计的靶点数
      gc_breakdown:      分GC含量的可行率 {low_gc, mid_gc, high_gc}
    """
    records = sample_exons(gencode_fasta, n_samples, random_seed=random_seed)

    if verbose:
        print(f"[Feasibility] 加载 {len(records)} 条外显子序列", flush=True)

    gc_bins = bin_by_gc(records)

    feasible = optimal = total = failed = 0
    gc_feasible: dict[str, int] = {k: 0 for k in gc_bins}
    gc_total: dict[str, int] = {k: len(v) for k, v in gc_bins.items()}

    # 建立 record id → gc_label 映射（用 record.id 而非 id()，避免内存地址失效）
    gc_label_map: dict[str, str] = {}
    for label, recs in gc_bins.items():
        for r in recs:
            gc_label_map[r.id] = label

    for i, record in enumerate(records):
        total += 1
        if verbose and i % 100 == 0:
            print(
                f"[Feasibility] 进度: {i}/{len(records)}  "
                f"feasible={feasible}  optimal={optimal}",
                flush=True,
            )

        try:
            pairs, _ = pipeline.run_design(
                template=str(record.seq).upper().replace("N", "A"),
                target_region=None,
                offtarget_records=[],
                params={"PRIMER_NUM_RETURN": 5},
                use_ai_search=use_ai_search,
            )
            if not pairs:
                failed += 1
                continue

            best = max(pairs, key=lambda x: x.pair_si)
            label = gc_label_map.get(record.id)

            if best.pair_si > 1.5 and best.pair_ei > 1.0:
                optimal += 1
                feasible += 1
                if label:
                    gc_feasible[label] += 1
            elif best.pair_si > 0.8 and best.pair_ei > 0.5:
                feasible += 1
                if label:
                    gc_feasible[label] += 1

        except Exception as e:
            failed += 1
            if verbose:
                print(f"[Feasibility] 跳过 {record.id}: {e}", flush=True)
            continue

    result = {
        "feasibility_rate": feasible / total if total > 0 else 0.0,
        "optimal_rate": optimal / total if total > 0 else 0.0,
        "total": total,
        "feasible": feasible,
        "optimal": optimal,
        "failed": failed,
        "gc_breakdown": {
            k: gc_feasible[k] / gc_total[k] if gc_total[k] > 0 else 0.0
            for k in gc_bins
        },
    }

    if verbose:
        print("\n=== 可行性验证结果 ===")
        print(f"总靶点数:       {result['total']}")
        print(f"可行率:         {result['feasibility_rate']:.1%}  (SI>0.8, EI>0.5)")
        print(f"最优率:         {result['optimal_rate']:.1%}  (SI>1.5, EI>1.0)")
        print(f"设计失败数:     {result['failed']}")
        print(
            f"GC分组可行率:  "
            f"低GC={result['gc_breakdown']['low_gc']:.1%}  "
            f"中GC={result['gc_breakdown']['mid_gc']:.1%}  "
            f"高GC={result['gc_breakdown']['high_gc']:.1%}"
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="基因组尺度Hairpin-Blocker设计可行性验证"
    )
    parser.add_argument("--gencode", required=True, help="GENCODE外显子FASTA文件路径")
    parser.add_argument("--n", type=int, default=1000, help="采样外显子数量 (默认1000)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认42)")
    parser.add_argument("--ai", action="store_true", help="启用GNN加速搜索")
    parser.add_argument("--quiet", action="store_true", help="静默模式")
    args = parser.parse_args()

    result = run_feasibility_study(
        gencode_fasta=args.gencode,
        n_samples=args.n,
        random_seed=args.seed,
        use_ai_search=args.ai,
        verbose=not args.quiet,
    )

    # 合格线检查（论文 Supplementary Figure 1 标准）
    print("\n=== 合格线检查 ===")
    print(
        f"Feasibility Rate > 50%: "
        f"{'✓' if result['feasibility_rate'] > 0.5 else '✗'} "
        f"({result['feasibility_rate']:.1%})"
    )
    print(
        f"Optimal Rate > 30%:     "
        f"{'✓' if result['optimal_rate'] > 0.3 else '✗'} "
        f"({result['optimal_rate']:.1%})"
    )
    gc_vals = list(result["gc_breakdown"].values())
    gc_diff = max(gc_vals) - min(gc_vals) if gc_vals else 0.0
    print(
        f"GC敏感性 < 20%:         "
        f"{'✓' if gc_diff < 0.2 else '✗'} "
        f"(max_diff={gc_diff:.1%})"
    )


if __name__ == "__main__":
    main()
