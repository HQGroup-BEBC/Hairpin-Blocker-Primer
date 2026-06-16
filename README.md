# Hairpin-Blocker Designer — 发夹阻断引物设计系统

> 通过在引物 5' 端主动设计发夹阻断结构，以热力学竞争机制抑制非特异性扩增的 PCR 引物设计工具。  
> 集成 TargetStruct-GNN 加速搜索、MPIGN 多重引物优化、3D 分子结构可视化与扩增子顾问。

---

## 目录

1. [核心设计原理](#1-核心设计原理)
2. [项目结构](#2-项目结构)
3. [模块说明](#3-模块说明)
4. [AI for Science 组件](#4-ai-for-science-组件)
5. [多重PCR优化（MPIGN）](#5-多重pcr优化mpign)
6. [3D 分子结构可视化](#6-3d-分子结构可视化)
7. [扩增子顾问](#7-扩增子顾问)
8. [GUI 界面说明](#8-gui-界面说明)
9. [安装与运行](#9-安装与运行)
10. [数据格式](#10-数据格式)
11. [参数说明](#11-参数说明)
12. [参考文献](#12-参考文献)

---

## 1. 核心设计原理

### 1.1 三方竞争热力学模型

传统引物设计依赖序列特异性被动筛选；本系统改为在引物 5' 端**主动工程化**一个发夹阻断结构，令三种结合态之间形成热力学竞争：

```
ΔG_target  ≪  ΔG_hairpin  <  ΔG_offtarget
（约定：数值越负=结合越稳定）
```

| 结合态 | 含义 | 结果 |
|--------|------|------|
| **目标结合** ΔG_target | 发夹引物与完全互补目标模板杂交 | 目标模板打开发夹 → 3' 端暴露 → 正常延伸 ✓ |
| **发夹折叠** ΔG_hairpin | 分子内自折叠，3' 端被锁入双链茎部 | 脱靶弱结合时发夹竞争胜出 → 3' 端被封 → 无延伸 ✓ |
| **脱靶结合** ΔG_offtarget | 与错配的非目标序列杂交 | 弱于发夹 → 被发夹阻断 → 无延伸 ✓ |

### 1.2 发夹引物结构

```
5'─[stem_comp]─[loop]─[primer_body]─[primer_3'stem]─3'
    ←─── s bp ──→  l nt  ←── (n-2s-l) nt ──→ ← s bp →

折叠后：
        loop
     ╭──────╮
5'  ─┤ stem ├─ primer_body ─ 3'
     ╰──────╯
     (3'端被封)
```

- `stem_comp = reverse_complement(primer_seq[-s:])` — 拼接在 5' 端
- `stem_len`：4–6 bp（过长→发夹过稳→ on-target 效率风险）
- `loop_len`：3–5 nt（GTTTT 等低二级结构 loop 序列）

### 1.3 SI / EI 评分

#### 特异性指数（SI）

```
若 ΔG_offtarget ≥ 0：              SI = SI_CAP = 5.0   (无脱靶竞争，自动 excellent)
若 ΔG_hairpin  ≥ 0：              SI ≤ 0              (发夹不稳定，reject)
否则：                             SI = ΔG_hairpin / ΔG_offtarget
```

SI > 1 意味着发夹稳定性强于脱靶结合 → 脱靶被成功阻断。

#### 效率指数（EI）

```
EI = (ΔG_hairpin − ΔG_target) / |ΔG_target|
```

EI > 0 确保目标结合能克服发夹，数值越大效率富余量越大。

#### 阈值

| 指标 | 优秀 | 可用 | 淘汰 |
|------|------|------|------|
| SI   | > 1.5 | 0.8–1.5 | < 0.8 |
| EI   | > 1.0 | 0.5–1.0 | < 0.5 |

目标结合约束（硬过滤）：`ΔG_target < ΔG_hairpin − 2.0 kcal/mol`

---

## 2. 项目结构

```
pcr_primer_designer/
├── main.py                          # 启动入口
├── requirements.txt
├── train_gnn.py                     # GNN 独立训练脚本
├── test_pipeline.py                 # 管道集成测试
├── gui_smoke_test.py                # GUI 冒烟测试
│
├── primer_designer/                 # 核心引擎
│   ├── sequence_io.py               # FASTA 解析与序列校验
│   ├── candidate_generator.py       # primer3 候选引物生成
│   ├── offtarget_finder.py          # 非目标库种子匹配
│   ├── thermo_model.py              # 三方竞争 ΔG + SI/EI 计算
│   ├── hairpin_designer.py          # 发夹设计搜索（经典 + AI 加速）
│   ├── pareto.py                    # 引物对组装 + Pareto 前沿
│   ├── pipeline.py                  # 单模板 / 多重PCR 主流程
│   ├── gnn_model.py                 # TargetStruct-GNN 代理模型
│   ├── multiplex_gnn.py             # MPIGN 多重引物优化
│   ├── amplicon_advisor.py          # 扩增子顾问（滑动窗口评分）
│   └── structure_pdb.py             # B-form DNA PDB 原子坐标生成
│
├── gui/
│   ├── main_window.py               # Tkinter 主窗口
│   └── viewer_3d.py                 # GPU WebGL 3D 分子查看器
│
└── sample_data/
    ├── target_example.fasta         # 单模板示例（400 bp）
    ├── offtarget_example.fasta      # 非目标库示例（3 条诱饵序列）
    └── multiplex_example.fasta      # 多重PCR示例（3条癌症热点扩增子）
```

---

## 3. 模块说明

### `sequence_io.py`

| 函数 | 功能 |
|------|------|
| `validate_sequence(seq)` | 去空白转大写，允许 ACGTN，否则报错 |
| `parse_fasta_text(text)` | 解析粘贴文本（含/不含 FASTA 头） |
| `load_fasta_file(path)` | BioPython SeqIO 读取文件 |

### `candidate_generator.py`

调用 `primer3.bindings.design_primers` 生成候选引物。

**默认参数（`DEFAULT_PARAMS`）：**

| 参数 | 值 | 说明 |
|------|----|------|
| 引物长度 | 18–25（最优 20） | nt |
| Tm | 57–63℃（最优 60℃） | 退火温度 |
| GC% | 40–60% | — |
| 产物大小 | 100–300 bp | — |
| 候选数量 | 20 | `PRIMER_NUM_RETURN` |
| PolyX | ≤ 4 | 同碱基延伸限制 |

`PrimerCandidate` 数据类额外携带 `left_target_seq` / `right_target_seq`（引物在模板上的对应窗口序列），供 `thermo_model` 计算 ΔG_target。

### `offtarget_finder.py`

```python
find_offtarget_window(primer_seq, offtarget_records, max_mismatch=3, seed_len=10)
```

1. 取引物 3' 端 10 nt 种子，在非目标库双链中精确匹配
2. 从命中位置扩展引物等长窗口，计算 Hamming 距离
3. 返回错配数 ≤ `max_mismatch` 中最相似的窗口（供 ΔG_offtarget 计算）

### `thermo_model.py`

所有 ΔG 计算基于 `primer3-py`，退火温度 60℃，单位 kcal/mol。

| 函数 | 计算内容 |
|------|----------|
| `compute_dg_target(hp_seq, primer_seq)` | `calc_heterodimer(hp_seq, revcomp(primer_seq))` |
| `compute_dg_hairpin(hp_seq)` | `calc_hairpin(hp_seq)` |
| `compute_dg_offtarget(hp_seq, window)` | `calc_heterodimer(hp_seq, revcomp(window))` |
| `compute_dg_homodimer(hp_seq)` | `calc_homodimer(hp_seq)`（自身二聚体风险） |
| `compute_si_ei(...)` | SI/EI 及 verdict 判定 |

on-target 风险分级（影响 EI 显示颜色）：
- **low**：ΔG_hairpin > −5 kcal/mol
- **medium**：−8 ≤ ΔG_hairpin ≤ −5 kcal/mol
- **high**：ΔG_hairpin < −8 kcal/mol（可能导致 Ct 延后 / 扩增效率 < 90%）

### `hairpin_designer.py`

**经典搜索** `design_hairpin_blocker`：
- 遍历 `stem_len ∈ [4,5,6]`，`loop_len ∈ [3,4,5]`，`loop_seq`（过滤 GGG/CCC/高自互补）
- 评分函数：`score = ΔG_offtarget − ΔG_hairpin`（越大=脱靶被阻断越彻底）
- 硬约束：`ΔG_target < ΔG_hairpin − 2.0`

**AI 加速搜索** `design_hairpin_blocker_ai`（`use_ai_search=True`）：
1. 扩展 loop_seq 候选到无上限（`itertools.product("ACGT", repeat=loop_len)`）
2. GNN 代理模型批量预测 ΔG_hairpin，保留 Top-K=20
3. 对 Top-K 调用 primer3 精确复算，选出最优

### `pareto.py`

`pareto_front(pairs)` 返回 (pair_SI, pair_EI) 二维非支配解集合，按 pair_SI 降序。

`pair_SI = min(left.SI, right.SI)`，`pair_EI = min(left.EI, right.EI)`

### `pipeline.py`

```python
# 单模板
pairs, front = run_design(template, target_region, offtarget_records, params, use_ai_search)

# 多重PCR
result = run_multiplex_design(templates, ..., n_select, cross_talk_threshold, physics_ctx)
```

---

## 4. AI for Science 组件

### TargetStruct-GNN（`gnn_model.py`）

**训练数据：** Ke et al., Nat. Commun. 2025，NNN 数据集  
文件：`external_data/nnn_dna_thermo/fitted_variant_arr.csv`  
筛选条件：`two_state=True`，共 **19,738** 条发夹热力学数据（ΔG at 37℃）

**图结构（4 类边关系）：**

| 关系类型 | 来源 | 作用 |
|----------|------|------|
| ① 正向骨架（5'→3'） | 序列拓扑 | 描述核苷酸连接顺序 |
| ② 反向骨架（3'→5'） | 序列拓扑 | 反向信息传播 |
| ③ 软配对边 | PairAttention 子网学习 | WC 相容性先验 + 最小环长掩码 |
| ④ 氢键边（结构感知） | 点括号解析 | **训练时**：NNN 数据集的 TargetStruct（NUPACK 预算）；**推断时**：`design_dotbracket(stem, loop)` 零依赖构造 |

**推断时零 NUPACK 依赖：**
```python
def design_dotbracket(total_len, stem_len, loop_len) -> str:
    # stem=4, total=26 → "((((...................))))"
```

**性能对比：**

| 模型 | R²（10 epoch） | R²（80 epoch 预期） |
|------|---------------|-------------------|
| StructureFreeGNN（旧） | 0.801 | 0.857 |
| **TargetStruct-GNN（当前）** | **0.857** | **~0.924** |

**训练：**
```bash
python train_gnn.py          # 在 NNN 数据集上训练，自动使用 GPU（若可用）
```

**推断（pipeline 内部调用）：**
```python
surrogate = GNNSurrogate(device="cuda")
dg_pred = surrogate.predict_dg(seq, temp_c=60.0, dot_bracket="(((...)))")
```

---

## 5. 多重PCR优化（MPIGN）

**MPIGN = Multiplex Primer Interaction Graph Network**，`multiplex_gnn.py`

### 5.1 图建模

- **节点**：每条候选引物，10 维特征向量（长度、GC%、ΔG_hairpin、SI、EI、ΔG_homodimer、茎/环参数）
- **边**：任意两条引物间的异源二聚体 ΔG（`primer3.calc_heterodimer`），代表串扰危险度

### 5.2 物理环境修正

盐浓度修正（基于 SantaLucia 2004）：

```
salt_factor = 1 + 0.05 × log₁₀(Mg²⁺/2.0)        (Mg²⁺ 主导时)
            = 1 + 0.04 × log₁₀(Na⁺/50.0)           (Na⁺ 修正)
ΔG_corr = ΔG_p3 × salt_factor
```

### 5.3 图注意力消息传递

```python
class MPIGNAttentionLayer(nn.Module):
    # 物理偏置直接加入注意力 logit：
    phys_bias = ΔG_corr_ij.clamp(max=0.0) / kT
    scores = scores + phys_bias.unsqueeze(0)
```

**高阶相互作用捕获：** 即使 A-B、B-C 的直接 ΔG 均满足阈值，若 B 的邻域整体张力高，MPIGN 通过消息传递降低 A 和 C 的评分，避免它们与 B 共池。

### 5.4 结果

`MultiplexResult` 包含：
- `selected_pairs`：MPIGN 优选的引物对子集
- `dg_matrix`：N×N 串扰 ΔG 矩阵（GUI 热图原始数据）
- `node_scores`：每条引物的图注意力综合评分
- `cross_talk_warnings`：违反阈值的引物对列表

### 5.5 示例数据（`multiplex_example.fasta`）

模拟三靶液体活检多重 PCR 面板（癌症突变热点）：

| 靶标 | 长度 | GC% | 热点 | 应用场景 |
|------|------|-----|------|----------|
| KRAS exon2 | 396 bp | 43.7% | G12C / G12V / G13D | 结直肠癌、肺腺癌、胰腺癌 |
| BRAF exon15 | 388 bp | 44.1% | V600E / V600K | 黑色素瘤、甲状腺癌、结直肠癌 |
| EGFR exon21 | 385 bp | 56.1% | L858R | 肺腺癌（靶向治疗伴随诊断）|

---

## 6. 3D 分子结构可视化

### 6.1 PDB 生成（`structure_pdb.py`）

基于 **Arnott & Hukins (1972)** B-form DNA 纤维衍射模型，每个核苷酸放置 6 个骨架原子：

| 原子 | 半径 r (Å) | 角度 θ (°) | 相对高度 Δz (Å) |
|------|-----------|-----------|----------------|
| P    | 8.91 | −139.0 | 0.00 |
| O5'  | 8.06 | −139.0 | 0.72 |
| C5'  | 7.11 | −131.0 | 1.25 |
| C4'  | 6.17 | −113.0 | 1.83 |
| C3'  | 5.31 | −68.0  | 2.88 |
| O3'  | 6.21 | −50.0  | 3.38 |

B-form 参数：rise = 3.38 Å/bp，twist = 36°/bp（10 bp/圈）

**双链区（茎）：** Chain A（5'→3'）与 Chain B（反向平行，相位差 180°）

**单链弧区（loop + primer body）：** 弧路径参数化公式，保证与茎顶平滑衔接：
```
r(t) = R_c4 + arc_r × sin(πt)
φ(t) = φ_top + π × t
z(t) = z_top + arc_h × sin(πt)        t ∈ (0, 1)
```

输出 PDB 兼容 PyMOL / UCSF Chimera / RCSB 所有标准软件。

### 6.2 GPU 加速查看器（`viewer_3d.py`）

- 渲染引擎：**3Dmol.js**（WebGL），等效 OpenGL 硬件加速，帧率 > 60 fps
- 嵌入方式：**pywebview**（优先，Edge WebView2 GPU 渲染）→ fallback 到系统浏览器
- 交互：鼠标左键旋转 / 滚轮缩放 / 右键平移
- 表示法：棒状、球状、卡通、线框、混合（按区段着色）
- 附加功能：VDW 分子表面、PNG 导出

```python
from gui.viewer_3d import launch_viewer
launch_viewer(pdb_string, stem_len=5, loop_len=4, total_len=30)
```

---

## 7. 扩增子顾问

`amplicon_advisor.py`：滑动窗口扫描目标序列，从四个维度打分推荐扩增区域。

| 维度 | 权重 | 方法 |
|------|------|------|
| GC 均匀度 | 35% | 整体偏离 50% 的惩罚 + 分段标准差 |
| 结构简单性 | 35% | 两端各 30 bp 发夹 ΔG（primer3），越稳定扣分越多 |
| 特异性 | 20% | 两端 20 nt 种子在非目标库中的命中情况 |
| 低复杂度惩罚 | 10% | 同碱基延伸(≥5)、二核苷酸重复(≥4次) |

综合得分 = 0.35×GC + 0.35×struct + 0.20×spec + 0.10×complexity

结果以**热图 + 可排序候选列表**展示，点击候选一键填入目标区域输入框。

---

## 8. GUI 界面说明

### 8.1 布局

```
┌─────────────────────┬────────────────────────────────────────────┐
│ 输入面板（左）       │ 结果面板（右）                              │
│                     │                                            │
│ · 目标序列          │ ┌── 候选引物表（Treeview）───────────────┐ │
│ · 目标区域          │ │ 排名 | 正向引物 | 反向引物 | SI | EI    │ │
│ · 非目标库          │ └──────────────────────────────────────── ┘ │
│ · 设计参数          │                                            │
│ · AI 搜索开关       │ ┌── 详情面板 ──┬── 图表标签页 ──────────┐ │
│ · 多重PCR框架(MPIGN)│ │ ΔG 数值      │ Tab1: SI-EI 散点图     │ │
│   ├ 启用/禁用       │ │ ASCII结构    │ Tab2: 发夹 2D 结构     │ │
│   ├ 加载多重示例    │ │ 3D查看器按钮 │ Tab3: 发夹 3D B-form   │ │
│   └ 参数配置        │ │ 导出PDB按钮  │ Tab4: 多重PCR串扰图    │ │
│                     │ └──────────────┴────────────────────────┘ │
│ · 开始设计          │                                            │
│ · 导出 CSV          │                                            │
└─────────────────────┴────────────────────────────────────────────┘
```

### 8.2 图表标签页

| 标签页 | 内容 | 交互 |
|--------|------|------|
| SI-EI 散点图 | 所有候选灰点，Pareto 前沿红点，阈值辅助线 | 点击数据点联动选中表格行 |
| 发夹 2D 结构 | 阶梯状发夹拓扑图，碱基对横档，区段着色 | 自动随选中行更新 |
| 发夹 3D B-form | matplotlib 3D 双螺旋（可拖拽旋转）| 与 2D 同步 |
| 多重PCR串扰图 | 左：N×N ΔG 热图；右：引物交互网络 | 多重PCR完成后自动切换 |

### 8.3 详情面板按钮

| 按钮 | 功能 |
|------|------|
| 3D分子结构 (GPU WebGL) | 生成 PDB → 在 pywebview / 浏览器中打开 GPU 渲染查看器 |
| 导出PDB文件 | 保存 .pdb 文件，可直接用 PyMOL / UCSF Chimera 打开 |

### 8.4 多重PCR框架

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 每靶候选数 | 3 | 每条目标序列生成的引物对数量 |
| 最终选出对数 | （空=靶数） | MPIGN 输出的引物对总数 |
| 串扰阈值 | −5.0 kcal/mol | 安全阈值，超过则触发警告 |

---

## 9. 安装与运行

### 9.1 依赖

```bash
pip install biopython>=1.87 primer3-py>=2.3.0 numpy>=1.24 scipy>=1.10 pandas>=2.0 torch>=2.0
pip install pywebview          # 可选：3D 查看器 GPU 嵌入模式（Windows Edge WebView2）
```

GPU 训练（PyTorch 自动检测 CUDA）：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121    # CUDA 12.1
```

### 9.2 运行

```bash
cd pcr_primer_designer
python main.py
```

### 9.3 GNN 训练（可选）

1. 下载 NNN 数据集：[GreenleafLab/nnn_paper](https://github.com/GreenleafLab/nnn_paper)
2. 放置：`external_data/nnn_dna_thermo/fitted_variant_arr.csv`
3. 训练：`python train_gnn.py`（自动 GPU；批大小 512；预期 R²≈0.924 at 80 epoch）

---

## 10. 数据格式

### 目标序列（FASTA）

```fasta
>sequence_id description
ATCGATCGATCG...
```

多条目标序列（多重PCR模式，每条对应一个扩增子靶）：

```fasta
>KRAS_exon2
GCCTGCTGAAA...
>BRAF_exon15
GGTGCTTTTGG...
>EGFR_exon21
GCCTGCTGGGC...
```

### 非目标模板库（FASTA）

包含与目标序列可能存在同源性的背景序列（基因组背景、同族基因、旁系同源物等）。未提供时 ΔG_offtarget = 0，SI 自动取上限值 5.0。

### CSV 导出字段

```
rank, pareto, product_size, pair_si, pair_ei,
left_primer, left_hairpin_primer, left_stem_len, left_loop_len, left_loop_seq,
left_dg_target, left_dg_hairpin, left_dg_offtarget, left_dg_homodimer, left_on_target_risk,
left_si, left_ei, left_verdict,
right_primer, right_hairpin_primer, right_stem_len, right_loop_len, right_loop_seq,
right_dg_target, right_dg_hairpin, right_dg_offtarget, right_dg_homodimer, right_on_target_risk,
right_si, right_ei, right_verdict
```

### PDB 输出

标准 PDB 格式，ATOM 记录（P/O/C 元素），含 SEQRES 和 CONECT，兼容：
- PyMOL（开源版 `pymol-open-source`）
- UCSF Chimera / ChimeraX
- RCSB 3D Viewer（网页版）
- VESTA / VMD（其他分子可视化软件）

---

## 11. 参数说明

### primer3 关键参数

| 参数键 | 含义 | 默认值 |
|--------|------|--------|
| `PRIMER_MIN/OPT/MAX_SIZE` | 引物长度范围 | 18 / 20 / 25 nt |
| `PRIMER_MIN/OPT/MAX_TM` | 解链温度范围 | 57 / 60 / 63 ℃ |
| `PRIMER_MIN/MAX_GC` | GC% 范围 | 40 / 60 % |
| `PRIMER_PRODUCT_SIZE_RANGE` | 产物大小范围 | [[100, 300]] bp |
| `PRIMER_NUM_RETURN` | 候选对数量 | 20 |
| `PRIMER_MAX_POLY_X` | 最大同碱基延伸 | 4 |

### 发夹设计参数

| 参数 | 范围 | 说明 |
|------|------|------|
| `stem_len` | 4–6 bp | 茎区长度；越长发夹越稳，on-target 风险越高 |
| `loop_len` | 3–5 nt | 环区长度；3 nt 最小稳定 loop |
| `ANNEAL_TEMP_C` | 60℃（固定） | 所有 ΔG 计算温度 |
| `TARGET_HAIRPIN_MARGIN_KCAL` | 2.0 kcal/mol | ΔG_target 必须比 ΔG_hairpin 强的最小裕量 |

### PhysicsContext（MPIGN 物理参数）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `temp_c` | 60.0℃ | 退火温度 |
| `mg_mm` | 2.0 mM | Mg²⁺ 浓度（0 → 改用 Na⁺ 修正） |
| `na_mm` | 50.0 mM | Na⁺ / K⁺ 等效浓度 |
| `primer_conc_nm` | 200.0 nM | 引物浓度 |

---

## 12. 参考文献

1. **Ke et al. (2025)** "Array-based thermodynamic measurements reveal design principles for hairpin blocker sequences." *Nat. Commun.* — NNN 数据集（TargetStruct-GNN 训练来源）

2. **Arnott & Hukins (1972)** "Optimised parameters for A-DNA and B-DNA." *Biochem. Biophys. Res. Commun.* 47, 1504–1509 — B-form DNA 原子坐标参数

3. **SantaLucia & Hicks (2004)** "The thermodynamics of DNA structural motifs." *Annu. Rev. Biophys. Biomol. Struct.* 33, 415–440 — 盐浓度对 DNA 热力学的修正模型

4. **primer3 (Untergasser et al. 2012)** *Nucleic Acids Res.* 40, e115 — 引物设计与热力学计算库（`primer3-py` Python 绑定）

5. **3Dmol.js (Rego & Koes 2015)** "3Dmol.js: molecular visualization with WebGL." *Bioinformatics* 31, 1322–1324 — GPU WebGL 分子查看器
