"""oxDNA 模拟输入文件生成器

为 6 个温度点各生成一个 oxDNA2 输入文件，覆盖典型发夹 Tm 范围。
每个温度点为独立 MD 模拟（而非 REMD），便于在有限计算资源下运行。

温度设置依据：
  · 4-6 bp 发夹的 Tm 通常在 45–75°C (318–348 K)
  · 温度范围覆盖 310 K (37°C) ~ 380 K (107°C)，步长约 14 K
  · 每温度点独立采样 → 合并用于 FES 和 Tm 估算

时间步参数（oxDNA2 标准推荐）：
  dt = 0.003  (oxDNA 时间单位)
  步数 = 20,000,000 ≈ 20 ns 等效
  轨迹保存间隔 = 10,000 步 → 每温度 2000 帧

模拟条件：
  DNA2 力场 (oxDNA2, Snodin et al. 2015)
  NaCl = 0.1 M (生理盐浓度近似)
  Langevin 恒温器 (γ = 2.5)
"""
from __future__ import annotations

from pathlib import Path

# ── 温度序列 ────────────────────────────────────────────────────────
# oxDNA 温度单位: T* = T_K / 3000
# 参考: Sulc et al. (2012) J. Chem. Phys. 137, 135101
TEMPS_K: list[int] = [310, 324, 338, 352, 366, 380]
TEMPS_OX: list[float] = [round(T / 3000.0, 6) for T in TEMPS_K]

# ── 模拟参数 ────────────────────────────────────────────────────────
STEPS           = 20_000_000   # 每温度步数
DT              = 0.003        # 时间步长 (oxDNA 无量纲单位)
PRINT_INTERVAL  = 10_000       # 轨迹保存间隔 (步)
ENERGY_INTERVAL = 1_000        # 能量输出间隔 (步)
LANGEVIN_GAMMA  = 2.50         # Langevin 摩擦系数
SALT            = 0.1          # NaCl 浓度 (M)

_INPUT_TEMPLATE = """\
# oxDNA2 模拟参数 — 发夹阻断引物三态自由能验证
# 温度: {T_K} K ({T_C:.1f}°C)  |  T* = {T_ox}
# 生成工具: Hairpin-Blocker Designer oxdna_sim/inputs.py

interaction_type = DNA2
salt_concentration = {salt}

thermostat = langevin
newtonian_steps = 103
diff_coeff = {gamma}

dt = {dt}
verlet_skin = 0.05
steps = {steps}
seed = {seed}

T = {T_ox}

topology = hairpin.top
conf_file = init.dat
trajectory_file = traj_{T_K}K.dat
energy_file = energy_{T_K}K.dat
last_conf_file = last_{T_K}K.dat

print_conf_interval = {print_interval}
print_energy_every = {energy_interval}

no_stdout_energy = 0
restart_step_counter = 1
refresh_vel = 1
time_scale = linear
backend = CPU
"""


def write_inputs(output_dir: str | Path, seed_base: int = 12345) -> list[Path]:
    """为每个温度点写入 oxDNA 输入文件，返回文件路径列表。

    参数:
      output_dir: 输出目录（需已存在 hairpin.top 和 init.dat）
      seed_base:  随机种子基数，各温度点依次 +1

    返回:
      [input_310K.conf, input_324K.conf, ...]
    """
    out = Path(output_dir)
    paths: list[Path] = []

    for i, (T_K, T_ox) in enumerate(zip(TEMPS_K, TEMPS_OX)):
        content = _INPUT_TEMPLATE.format(
            T_K=T_K,
            T_C=T_K - 273.15,
            T_ox=T_ox,
            salt=SALT,
            gamma=LANGEVIN_GAMMA,
            dt=DT,
            steps=STEPS,
            seed=seed_base + i,
            print_interval=PRINT_INTERVAL,
            energy_interval=ENERGY_INTERVAL,
        )
        p = out / f"input_{T_K}K.conf"
        p.write_text(content)
        paths.append(p)

    return paths


def write_run_script(output_dir: str | Path, oxdna_bin: str = "oxDNA") -> Path:
    """生成 Shell 运行脚本，顺序执行所有温度点的模拟。

    参数:
      output_dir: 与 write_inputs 相同的目录
      oxdna_bin:  oxDNA 可执行文件名或完整路径

    用法：
      bash run_all.sh              # 顺序运行（单机）
      bash run_all.sh &            # 后台运行
    """
    out = Path(output_dir)
    lines = [
        "#!/bin/bash",
        "# 发夹阻断引物三态自由能验证 — oxDNA2 模拟",
        f"# oxDNA 可执行文件: {oxdna_bin}",
        "",
        "set -e",
        f"OXDNA={oxdna_bin}",
        f"DIR={out.resolve()}",
        "",
        'echo "开始多温度 MD 模拟..."',
        "",
    ]
    for T_K in TEMPS_K:
        lines += [
            f'echo "  运行 {T_K} K ({T_K - 273:.0f}°C)..."',
            f'$OXDNA $DIR/input_{T_K}K.conf > $DIR/log_{T_K}K.txt 2>&1',
        ]
    lines += [
        "",
        'echo "所有模拟完成。运行分析:"',
        "python -m oxdna_sim.pipeline --analyze --dir $DIR",
    ]
    script = out / "run_all.sh"
    script.write_text("\n".join(lines) + "\n")
    script.chmod(0o755)
    return script
