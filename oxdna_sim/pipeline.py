"""oxDNA 三态自由能验证管道 — 主入口

用法（命令行）:
  # 从发夹引物序列准备模拟文件
  python -m oxdna_sim.pipeline --prepare --seq CGCGATCAAAATCGATCGCG --stem 5 --loop 4 --dir ./sim_kras

  # 运行模拟（需已安装 oxDNA）
  cd ./sim_kras && bash run_all.sh

  # 分析轨迹，生成 FES 图和报告
  python -m oxdna_sim.pipeline --analyze --dir ./sim_kras --stem 5

用法（Python API）:
  from oxdna_sim.pipeline import prepare_simulation, analyze_simulation

  design = ...  # HairpinDesign 对象
  sim_dir = prepare_simulation(design, output_dir="./sim_out")
  # (运行 sim_dir/run_all.sh 后)
  results = analyze_simulation(sim_dir, design=design)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .topology import write_topology
from .config   import write_config
from .inputs   import write_inputs, write_run_script, TEMPS_K
from .analyze  import compute_fes, estimate_Tm_md, plot_fes, generate_report


# ---------------------------------------------------------------------------
# 准备模拟目录
# ---------------------------------------------------------------------------

def prepare_simulation(
    design,
    output_dir: str | Path = "oxdna_sim_out",
    oxdna_bin: str = "oxDNA",
) -> Path:
    """根据 HairpinDesign 对象准备完整的 oxDNA 模拟目录。

    生成文件:
      hairpin.top    — 拓扑文件
      init.dat       — 初始构型（线性展开）
      input_XXXK.conf — 各温度输入文件 (6 个)
      run_all.sh     — 运行脚本

    参数:
      design:     HairpinDesign 对象（含 hairpin_primer_seq / stem_len / loop_len）
      output_dir: 输出目录（自动创建）
      oxdna_bin:  oxDNA 可执行文件名（默认 "oxDNA"）

    返回: output_dir 的 Path 对象
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    seq = design.hairpin_primer_seq

    # 写入拓扑和初始构型
    write_topology(seq, out / "hairpin.top")
    write_config(seq, out / "init.dat")

    # 写入各温度输入文件和运行脚本
    write_inputs(out)
    run_script = write_run_script(out, oxdna_bin=oxdna_bin)

    # 保存序列元数据（供后续分析用）
    meta_lines = [
        f"seq={seq}",
        f"stem_len={design.stem_len}",
        f"loop_len={design.loop_len}",
        f"dg_hairpin={design.dg_hairpin:.4f}",
        f"si={design.si:.4f}",
        f"ei={design.ei:.4f}",
    ]
    (out / "meta.txt").write_text("\n".join(meta_lines))

    print(f"[oxDNA 准备完成] 目录: {out.resolve()}")
    print(f"  核苷酸数: {len(seq)}")
    print(f"  茎区: {design.stem_len} bp   环区: {design.loop_len} nt")
    print(f"  温度点: {[f'{T-273}°C' for T in TEMPS_K]}")
    print(f"  运行模拟: cd {out} && bash run_all.sh")

    return out


# ---------------------------------------------------------------------------
# 分析结果
# ---------------------------------------------------------------------------

def analyze_simulation(
    sim_dir: str | Path,
    stem_len: int | None = None,
    design=None,
) -> dict:
    """分析 oxDNA 模拟结果，生成 FES 图和文字报告。

    参数:
      sim_dir:  prepare_simulation 生成的目录（含轨迹文件）
      stem_len: 茎区长度（若 design 为 None 时必须提供）
      design:   HairpinDesign 对象（可选，用于 primer3 Tm 对比）

    返回:
      {
        "fes_list":  [FES 字典列表，每温度一个],
        "Tm_md":     float | None (K),
        "report":    str,
        "plot_path": Path,
      }
    """
    sim_dir = Path(sim_dir)

    # 读取元数据
    if design is None and (sim_dir / "meta.txt").exists():
        meta = dict(
            line.split("=", 1)
            for line in (sim_dir / "meta.txt").read_text().splitlines()
            if "=" in line
        )
        seq = meta.get("seq", "")
        n_nts = len(seq)
        if stem_len is None:
            stem_len = int(meta.get("stem_len", 5))
    else:
        if design is not None:
            n_nts    = len(design.hairpin_primer_seq)
            stem_len = stem_len or design.stem_len
        else:
            raise ValueError("需提供 stem_len 参数或 design 对象")

    # 计算各温度 FES
    fes_list = []
    for T_K in TEMPS_K:
        traj = sim_dir / f"traj_{T_K}K.dat"
        if not traj.exists():
            print(f"  [跳过] 未找到轨迹: {traj.name}")
            continue
        print(f"  分析 {T_K} K ({T_K-273:.0f}°C) 轨迹...", end=" ", flush=True)
        try:
            fes = compute_fes(traj, n_nts, stem_len, T_K)
            fes_list.append(fes)
            print(f"{fes['n_frames']} 帧  ΔF_fold={fes['dF_fold']:+.2f} kcal/mol")
        except Exception as e:
            print(f"错误: {e}")

    if not fes_list:
        raise RuntimeError(f"未找到任何有效轨迹文件。请先运行: cd {sim_dir} && bash run_all.sh")

    # 估算 Tm
    Tm_md = estimate_Tm_md(fes_list) if len(fes_list) >= 2 else None

    # 生成图和报告
    plot_path = sim_dir / "fes_three_state.png"
    plot_fes(fes_list, plot_path, design=design, Tm_md=Tm_md)

    report_path = sim_dir / "validation_report.txt"
    report = generate_report(fes_list, Tm_md, design=design, output_path=report_path)

    print()
    print(report)
    print(f"\n图表已保存: {plot_path}")

    return {
        "fes_list":  fes_list,
        "Tm_md":     Tm_md,
        "report":    report,
        "plot_path": plot_path,
    }


# ---------------------------------------------------------------------------
# 命令行接口
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="oxDNA 发夹阻断引物三态自由能验证管道",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd")

    # -- prepare --
    p_prep = sub.add_parser("prepare", help="生成模拟输入文件")
    p_prep.add_argument("--seq",    required=True,  help="发夹引物序列 (5'→3')")
    p_prep.add_argument("--stem",   type=int, default=5, help="茎区长度 bp (默认 5)")
    p_prep.add_argument("--loop",   type=int, default=4, help="环区长度 nt (默认 4)")
    p_prep.add_argument("--dg",     type=float, default=-3.0, help="ΔG_hairpin kcal/mol")
    p_prep.add_argument("--si",     type=float, default=1.5,  help="SI 值")
    p_prep.add_argument("--ei",     type=float, default=1.0,  help="EI 值")
    p_prep.add_argument("--dir",    default="oxdna_sim_out",  help="输出目录")
    p_prep.add_argument("--oxdna",  default="oxDNA",          help="oxDNA 可执行文件")

    # -- analyze --
    p_ana = sub.add_parser("analyze", help="分析轨迹，生成 FES 图和报告")
    p_ana.add_argument("--dir",  required=True, help="模拟目录")
    p_ana.add_argument("--stem", type=int, default=None, help="茎区长度（若无 meta.txt 时必填）")

    args = parser.parse_args()

    if args.cmd == "prepare":
        # 构造一个轻量级 design 对象
        from types import SimpleNamespace
        design = SimpleNamespace(
            hairpin_primer_seq=args.seq,
            stem_len=args.stem,
            loop_len=args.loop,
            dg_hairpin=args.dg,
            si=args.si,
            ei=args.ei,
        )
        prepare_simulation(design, output_dir=args.dir, oxdna_bin=args.oxdna)

    elif args.cmd == "analyze":
        analyze_simulation(args.dir, stem_len=args.stem)

    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
