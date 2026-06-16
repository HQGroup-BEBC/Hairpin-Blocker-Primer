"""oxDNA 初始构型文件生成器 (.dat)

初始构型采用线性展开的单链，核苷酸沿 z 轴排列。
这是最保守的起始构型：
  · 避免人工引入折叠偏置
  · REMD 高温副本（> Tm）可在皮秒内退折叠，低温副本通过交换获取折叠构型
  · 有效避免 B-form 初始构型的坐标生成误差

oxDNA 物理单位换算：
  长度单位 σ = 0.8518 nm (Debye 长度，1 M NaCl)
  温度单位 T* = T_K / 3000   (T* = 0.1 ≈ 300 K)
  能量单位 ε  (参见 Sulc et al. 2012)

.dat 文件格式 (每帧)：
  t = <时间步>
  b = <Lx> <Ly> <Lz>           (模拟盒子边长，单位 σ)
  E = <Etot> <Epot> <Ekin>     (初始时均设 0)
  <pos_x> <pos_y> <pos_z>  <a1_x> <a1_y> <a1_z>  <a3_x> <a3_y> <a3_z>
  <vel_x> <vel_y> <vel_z>  <ang_vel_x> <ang_vel_y> <ang_vel_z>
  (每核苷酸一行)

核苷酸坐标约定：
  pos: 质心坐标
  a1:  "碱基向量"——从糖磷酸骨架指向碱基（Watson-Crick 面方向）
  a3:  "叠加向量"——沿 3' 方向的堆叠法向量
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# 单链 DNA 每核苷酸的轴向间距 (σ)
# 实验值 ~0.6 nm，转换为 σ: 0.6 / 0.8518 ≈ 0.704 σ
_SS_RISE = 0.704

# 模拟盒子大小 = max(链长, 最小盒子) 的倍数
_BOX_MULT = 5.0
_BOX_MIN  = 25.0   # σ (~21 nm)，确保孤立链不与镜像相互作用


def write_config(seq: str, path: str | Path, jitter: float = 0.01) -> None:
    """将发夹引物写入线性展开的 oxDNA 初始构型文件。

    参数:
      seq:    全长发夹引物序列 (5'→3')
      path:   输出 .dat 文件路径
      jitter: 位置随机扰动幅度 (σ)，防止对称能量简并
    """
    n = len(seq)
    rng = np.random.default_rng(42)

    # ── 位置：沿 z 轴线性排列，中心在原点 ──────────────────────────
    z_center = (n - 1) * _SS_RISE / 2.0
    positions = np.zeros((n, 3))
    for i in range(n):
        positions[i] = [0.0, 0.0, i * _SS_RISE - z_center]

    # 加入微小随机扰动，避免完全对称构型
    positions += rng.uniform(-jitter, jitter, (n, 3))

    # ── 朝向向量 ────────────────────────────────────────────────────
    # a1: 碱基向量，初始指向 +x 方向
    #     (对单链而言碱基方向任意，但 oxDNA 势能会自动建立正确方向)
    # a3: 叠加法向量，沿 5'→3' 即 +z 方向
    a1 = np.array([1.0, 0.0, 0.0])
    a3 = np.array([0.0, 0.0, 1.0])   # 5'→3' = +z

    box_L = max(n * _SS_RISE * _BOX_MULT, _BOX_MIN)

    # ── 写入文件 ─────────────────────────────────────────────────────
    with open(path, "w") as f:
        f.write("t = 0\n")
        f.write(f"b = {box_L:.4f} {box_L:.4f} {box_L:.4f}\n")
        f.write("E = 0.0 0.0 0.0\n")
        for i in range(n):
            p = positions[i]
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}  "
                f"{a1[0]:.6f} {a1[1]:.6f} {a1[2]:.6f}  "
                f"{a3[0]:.6f} {a3[1]:.6f} {a3[2]:.6f}  "
                f"0.000000 0.000000 0.000000  "
                f"0.000000 0.000000 0.000000\n"
            )
