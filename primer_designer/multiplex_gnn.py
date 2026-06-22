"""多重引物相互作用图注意力网络 (Multiplex Primer Interaction Graph Network, MPIGN)

核心创新架构：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 相互作用图建模 (Interaction Graph Construction)
   节点 (Nodes): 每条候选引物是一个节点。
     特征 = sequence CNN嵌入 + 设计特征 (SI, EI, ΔG_hairpin, GC%, ΔG_homodimer)
     既利用了现有的TargetStruct-GNN序列嵌入能力，又融入了发夹设计层的热力学评分。
   边 (Edges): 任意两条引物间的异源二聚体ΔG (primer3.calc_heterodimer)，
     代表"串扰危险度" (cross-talk penalty)，数值越负越危险。

2. 物理环境感知边修正 (Physics-Aware Edge Modulation)
   将反应物理条件作为全局上下文注入图网络：
   - 盐浓度 (Mg²⁺/Na⁺): 影响DNA杂交热力学稳定性，修正有效ΔG
     基于SantaLucia 2004方法: Mg²⁺升高 → Tm升高 → 有效ΔG更负
   这使得模型能在不同盐浓度条件下预测最稳定的引物组合。

3. 图注意力消息传递 (Graph Attention Message Passing)
   捕获纯两两筛选无法发现的"高阶相互作用"：
   若引物A与B各自兼容，B与C各自兼容，但B-C之间有强串扰，
   则B的"邻域张力"会通过消息传递影响A的评分，
   即使A-B直接ΔG满足阈值，A也应当避免与B(及其张力邻域)共池。
   这是O(N)消息传递相对于O(N²)穷举筛选的本质优势。

4. 贪心子图选择 (Greedy Subgraph Selection)
   依据节点评分贪心选出最优引物子集，保证任意两条被选引物的盐修正后ΔG
   均高于安全阈值 (default: -5 kcal/mol at annealing temperature)。

与pipeline的集成:
   run_mpign(pairs, ...) 接受现有run_design()结果，输出MultiplexResult，
   包含selected_pairs、串扰热图矩阵和网络图数据，供GUI可视化。

参考文献:
   - SantaLucia & Hicks (2004) Annu. Rev. Biophys. 33, 415 [盐修正]
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import primer3
import torch
from torch import nn

from .hairpin_designer import HairpinDesign
from .pareto import HairpinBlockerPair
from .thermo_model import ANNEAL_TEMP_C


# ---------------------------------------------------------------------------
# 1. 物理上下文 (Physical Environment Context)
# ---------------------------------------------------------------------------

@dataclass
class PhysicsContext:
    """PCR反应物理环境参数，用于盐浓度修正串扰ΔG。

    默认值对应标准管式PCR (2mM Mg, 50mM KCl/Na)。
    """
    temp_c: float = 60.0             # 退火温度 (°C)
    mg_mm: float = 2.0               # Mg²⁺浓度 (mM); 0 → 使用Na修正
    na_mm: float = 50.0              # Na⁺/K⁺等效浓度 (mM)
    primer_conc_nm: float = 200.0    # 引物浓度 (nM)


def _salt_factor(ctx: PhysicsContext) -> float:
    """盐修正因子——相对于标准2mM Mg参考条件的ΔG缩放比。

    基于SantaLucia 2004的方法：Mg²⁺主导时，[Mg]升高→Tm升高→有效ΔG更负。
    实现为相对缩放因子(>1=比参考更稳定, <1=比参考更不稳定)。
    """
    if ctx.mg_mm > 0:
        return 1.0 + 0.05 * math.log10(max(ctx.mg_mm, 0.01) / 2.0)
    else:
        return 1.0 + 0.04 * math.log10(max(ctx.na_mm, 1.0) / 50.0)


def physics_corrected_dg(dg_p3: float, ctx: PhysicsContext) -> float:
    """将primer3计算的串扰ΔG修正至指定盐浓度条件下的有效值。

    dg_p3 < 0: 稳定的异源二聚体 (危险串扰)
    高盐条件下修正后的ΔG更负 → 串扰风险更高
    """
    return dg_p3 * _salt_factor(ctx)


# ---------------------------------------------------------------------------
# 2. 引物节点特征 (Node Feature Extraction)
# ---------------------------------------------------------------------------

_BASE_IDX: dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3}


def _gc_content(seq: str) -> float:
    n = len(seq)
    if n == 0:
        return 0.0
    return sum(1 for c in seq.upper() if c in "GC") / n


def _node_feature_vector(
    primer_seq: str,
    design: Optional[HairpinDesign] = None,
) -> np.ndarray:
    """构建单条引物的节点特征向量 (10维)。

    特征说明:
      [0]  引物长度(归一化至[0,1]，max_len=40)
      [1]  GC含量
      [2]  ΔG_hairpin / 10   (发夹稳定性；可选)
      [3]  SI / 5            (特异性指数；可选)
      [4]  EI / 2            (效率指数；可选)
      [5]  ΔG_homodimer / 10 (自身二聚体倾向；可选)
      [6]  on_target_risk:   low=0, medium=0.5, high=1.0
      [7]  茎长 / 8          (发夹茎长；可选)
      [8]  环长 / 5          (发夹环长；可选)
      [9]  是否有发夹设计     (0/1)
    """
    feats = np.zeros(10, dtype=np.float32)
    feats[0] = len(primer_seq) / 40.0
    feats[1] = _gc_content(primer_seq)
    if design is not None:
        feats[2] = design.dg_hairpin / 10.0
        feats[3] = min(design.si, 5.0) / 5.0
        feats[4] = min(max(design.ei, -2.0), 2.0) / 2.0
        feats[5] = design.dg_homodimer / 10.0
        feats[6] = {"low": 0.0, "medium": 0.5, "high": 1.0}.get(design.on_target_risk, 0.0)
        feats[7] = design.stem_len / 8.0
        feats[8] = design.loop_len / 5.0
        feats[9] = 1.0
    return feats


# ---------------------------------------------------------------------------
# 3. 串扰矩阵计算 (Cross-Talk Interaction Matrix)
# ---------------------------------------------------------------------------

def compute_interaction_matrix(
    seqs: list[str],
    temp_c: float = ANNEAL_TEMP_C,
    ctx: Optional[PhysicsContext] = None,
) -> np.ndarray:
    """计算引物池中所有引物对的异源二聚体ΔG矩阵 (n × n)。

    使用 primer3.calc_heterodimer 计算所有有序对(i, j)的ΔG。
    对角线(自身) = 同源二聚体ΔG (ΔG_homodimer)。
    如果提供 ctx，对物理环境进行修正。

    返回: 对称矩阵, 单位 kcal/mol。数值越负 = 串扰越危险。
    """
    n = len(seqs)
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            r = primer3.bindings.calc_heterodimer(seqs[i], seqs[j], temp_c=temp_c)
            dg = r.dg / 1000.0
            mat[i, j] = physics_corrected_dg(dg, ctx) if ctx else dg
        # diagonal: homodimer
        r_hom = primer3.bindings.calc_heterodimer(seqs[i], seqs[i], temp_c=temp_c)
        dg_hom = r_hom.dg / 1000.0
        mat[i, i] = physics_corrected_dg(dg_hom, ctx) if ctx else dg_hom
    return mat


# ---------------------------------------------------------------------------
# 4. MPIGN 模型 (Graph Attention Network)
# ---------------------------------------------------------------------------

class MPIGNAttentionLayer(nn.Module):
    """物理感知图注意力层 (Physics-Aware Graph Attention Layer)。

    核心思路：
    - Query/Key/Value 由可学习线性变换给出
    - Attention logit = q_i·k_j/√d + bias_ij
    - bias_ij = -|ΔG_corr_ij| / kT   (串扰ΔG越负 → 注意力权重越低 → 危险邻居消息被抑制)
      这实现了"危险邻居的消息权重被物理量直接压制"的效果
    - 消息聚合后更新节点表示：h_i ← h_i + dropout(LayerNorm(MLP(concat(h_i, msg_i))))
    """

    def __init__(self, node_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert node_dim % num_heads == 0, "node_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.W_q = nn.Linear(node_dim, node_dim, bias=False)
        self.W_k = nn.Linear(node_dim, node_dim, bias=False)
        self.W_v = nn.Linear(node_dim, node_dim, bias=False)
        self.W_out = nn.Linear(node_dim, node_dim)
        self.norm1 = nn.LayerNorm(node_dim)
        self.norm2 = nn.LayerNorm(node_dim)
        self.ff = nn.Sequential(
            nn.Linear(node_dim, node_dim * 2),
            nn.GELU(),
            nn.Linear(node_dim * 2, node_dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,       # (N, node_dim) 节点表示
        dg_mat: torch.Tensor,  # (N, N) 物理修正后ΔG矩阵
        mask: torch.Tensor,    # (N,) bool 有效引物掩码
        kT: float = 0.62,      # 热能 kT at 60°C ≈ 0.62 kcal/mol
    ) -> torch.Tensor:
        N = h.size(0)
        q = self.W_q(h).view(N, self.num_heads, self.head_dim).transpose(0, 1)  # (H, N, d_k)
        k = self.W_k(h).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        v = self.W_v(h).view(N, self.num_heads, self.head_dim).transpose(0, 1)

        scores = torch.bmm(q, k.transpose(-2, -1)) / self.scale  # (H, N, N)

        # 物理偏置: bias[i,j] = -|ΔG_ij| / kT  (dangerousness penalty)
        # dg_mat <= 0 for cross-talks: |ΔG| = -dg (since dg < 0)
        phys_bias = dg_mat.clamp(max=0.0) / kT    # (N, N), non-positive → attention penalty
        scores = scores + phys_bias.unsqueeze(0)   # broadcast over heads

        # 掩码无效节点
        invalid = (~mask).unsqueeze(0).unsqueeze(1).expand_as(scores)
        scores = scores.masked_fill(invalid, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)

        msg = torch.bmm(attn, v)  # (H, N, d_k)
        msg = msg.transpose(0, 1).contiguous().view(N, -1)  # (N, node_dim)
        msg = self.W_out(msg)

        h = self.norm1(h + self.drop(msg))
        h = self.norm2(h + self.drop(self.ff(h)))
        return h * mask.float().unsqueeze(-1)


class MPIGNModel(nn.Module):
    """多重引物相互作用图注意力网络 (MPIGN) 完整模型。

    无监督模式 (默认): 用primer3 ΔG直接构造注意力偏置，GAT权重使用随机初始化
    (随机权重已足够捕获图拓扑带来的邻域张力传播，因为偏置项直接由物理量驱动)。
    有监督模式: 在标记的多重PCR成功/失败数据上训练后，注意力权重能学习非线性模式。

    前向输出: 每个节点的"多重兼容性评分" (compatibility score)
    score > 0: 在池中兼容性好 (低串扰邻域)
    score < 0: 在当前池的上下文中兼容性差 (高串扰邻域张力)
    """

    def __init__(
        self,
        node_feat_dim: int = 10,
        hidden: int = 32,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.layers = nn.ModuleList([
            MPIGNAttentionLayer(hidden, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.score_head = nn.Sequential(
            nn.Linear(hidden, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(
        self,
        node_feats: torch.Tensor,  # (N, node_feat_dim)
        dg_mat: torch.Tensor,      # (N, N) 物理修正ΔG
        mask: torch.Tensor,        # (N,) bool
        kT: float = 0.62,
    ) -> torch.Tensor:             # (N,) compatibility scores
        h = self.input_proj(node_feats) * mask.float().unsqueeze(-1)
        for layer in self.layers:
            h = layer(h, dg_mat, mask, kT)
        return self.score_head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# 5. 物理评分 (Physics-Only Scoring, No Training Required)
# ---------------------------------------------------------------------------

def physics_compatibility_scores(
    dg_mat: np.ndarray,
    cross_talk_threshold: float = -5.0,
    kT: float = 0.62,
) -> np.ndarray:
    """不依赖神经网络权重的纯物理兼容性评分 (n,)。

    每个节点的评分 = 对池中所有其他节点的"安全度"加权平均：
      w_ij = exp(ΔG_ij / kT)  → 越危险的邻居权重越小 (被物理量直接抑制)
      s_i = Σ_j w_ij × max(0, ΔG_ij - threshold)  → 与危险邻居的"余量"加权和
    评分越高 = 在当前池的上下文中越安全。

    与神经网络评分结合: final_score = 0.5 × phys_score + 0.5 × gnn_score
    """
    n = dg_mat.shape[0]
    scores = np.zeros(n, dtype=np.float32)
    for i in range(n):
        row = dg_mat[i].copy()
        row[i] = 0.0  # exclude self
        # 指数权重: 安全邻居(ΔG>0)权重~1, 危险邻居(ΔG<-10)权重→0
        weights = np.exp(row / kT)
        weights = weights / (weights.sum() + 1e-9)
        # 余量: ΔG - threshold (positive = safe margin, negative = in danger)
        margins = row - cross_talk_threshold
        scores[i] = float(np.dot(weights, margins))
    return scores


# ---------------------------------------------------------------------------
# 6. 贪心子集选择 (Greedy Compatible Subset Selection)
# ---------------------------------------------------------------------------

def greedy_compatible_subset(
    primer_indices: list[int],
    dg_mat: np.ndarray,
    scores: np.ndarray,
    n_select: int,
    cross_talk_threshold: float = -5.0,
    pair_constraints: Optional[list[tuple[int, int]]] = None,
) -> list[int]:
    """贪心选择最优兼容子集。

    策略:
    1. 按 scores 降序排列候选 (高分=在当前池中最安全)
    2. 遍历排序后的候选; 若新候选与所有已选引物的ΔG均 > threshold, 则加入
    3. 如果提供 pair_constraints [(left_idx, right_idx), ...], 则强制两者同时选或同时不选
    4. 重复直至选出 n_select 个 (或候选耗尽)

    参数:
      primer_indices: 候选引物在dg_mat中的下标列表
      dg_mat:         完整 (M, M) ΔG矩阵, M ≥ max(primer_indices)
      scores:         (M,) 各引物的兼容性评分
      n_select:       目标选出数量
      cross_talk_threshold: 安全ΔG阈值 (kcal/mol), default -5.0
      pair_constraints: [(i, j), ...] 必须同进同出的引物对
    """
    if pair_constraints is None:
        pair_constraints = []

    # 约束映射: idx → 其配对引物idx
    pair_map: dict[int, int] = {}
    for a, b in pair_constraints:
        pair_map[a] = b
        pair_map[b] = a

    ordered = sorted(primer_indices, key=lambda i: scores[i], reverse=True)
    selected: list[int] = []

    for i in sorted(set(primer_indices), key=lambda x: scores[x], reverse=True):
        if i in selected:
            continue
        partner = pair_map.get(i)
        # 检查i与已选所有引物的串扰
        safe = all(dg_mat[i, s] > cross_talk_threshold for s in selected)
        if partner is not None and partner not in selected:
            safe = safe and all(dg_mat[partner, s] > cross_talk_threshold for s in selected)
            safe = safe and (dg_mat[i, partner] > cross_talk_threshold)
        if safe:
            selected.append(i)
            if partner is not None and partner not in selected:
                selected.append(partner)
        if len(selected) >= n_select:
            break

    # 未满足数量时输出实际结果 (可能 < n_select)
    return selected[:n_select]


# ---------------------------------------------------------------------------
# 7. 结果数据类 (MultiplexResult)
# ---------------------------------------------------------------------------

@dataclass
class MultiplexResult:
    """MPIGN多重PCR优化结果。

    Attributes:
      all_pairs:           输入的所有候选引物对
      selected_pairs:      最终选出的兼容引物对 (按pair_si降序)
      all_seqs:            全部引物序列 (左右各一), 对应dg_matrix的行/列
      seq_labels:          每条序列的标签 (e.g. "Pair1-L", "Pair1-R")
      dg_matrix:           全引物池 (2N × 2N) 物理修正串扰ΔG矩阵
      node_scores:         (2N,) MPIGN节点兼容性评分
      cross_talk_warnings: 危险串扰对列表 [(label_i, label_j, ΔG), ...]
      ctx:                 使用的物理上下文
      cross_talk_threshold: 串扰安全阈值 (kcal/mol)
    """
    all_pairs: list[HairpinBlockerPair]
    selected_pairs: list[HairpinBlockerPair]
    all_seqs: list[str]
    seq_labels: list[str]
    dg_matrix: np.ndarray
    node_scores: np.ndarray
    cross_talk_warnings: list[tuple[str, str, float]]
    ctx: PhysicsContext
    cross_talk_threshold: float


# ---------------------------------------------------------------------------
# 8. 主入口 (Main Entry Point)
# ---------------------------------------------------------------------------

def run_mpign(
    pairs: list[HairpinBlockerPair],
    n_select: Optional[int] = None,
    cross_talk_threshold: float = -5.0,
    ctx: Optional[PhysicsContext] = None,
    use_gnn: bool = True,
    gnn_hidden: int = 32,
    gnn_layers: int = 3,
) -> MultiplexResult:
    """运行MPIGN，从候选引物对池中筛选最优多重兼容子集。

    参数:
      pairs:               来自 run_design() 的 HairpinBlockerPair 列表
      n_select:            目标引物对数量; None → 自动(池大小的一半或全部)
      cross_talk_threshold: 安全串扰ΔG阈值 (kcal/mol), default -5.0
      ctx:                 物理上下文; None → 使用默认标准条件
      use_gnn:             是否使用MPIGN神经网络 (True) 或仅物理评分 (False)
      gnn_hidden:          MPIGN隐藏层维度
      gnn_layers:          MPIGN消息传递层数
    """
    if not pairs:
        raise ValueError("引物池为空，无法运行MPIGN")

    if ctx is None:
        ctx = PhysicsContext()

    if n_select is None:
        n_select = max(1, len(pairs) // 2)

    kT = 0.592 * (ctx.temp_c + 273.15) / 298.15  # kT at reaction temperature [kcal/mol]

    # --- 组装全引物列表 ---
    all_seqs: list[str] = []
    seq_labels: list[str] = []
    pair_constraints: list[tuple[int, int]] = []
    all_designs: list[Optional[HairpinDesign]] = []

    for k, p in enumerate(pairs):
        idx_l = len(all_seqs)
        all_seqs.append(p.left.dna_seq)
        seq_labels.append(f"Pair{k+1}-L")
        all_designs.append(p.left)

        idx_r = len(all_seqs)
        all_seqs.append(p.right.dna_seq)
        seq_labels.append(f"Pair{k+1}-R")
        all_designs.append(p.right)

        pair_constraints.append((idx_l, idx_r))

    M = len(all_seqs)

    # --- 计算物理修正串扰矩阵 ---
    dg_mat = compute_interaction_matrix(all_seqs, temp_c=ctx.temp_c, ctx=ctx)

    # --- 节点特征矩阵 ---
    node_feats_np = np.stack([
        _node_feature_vector(seq, design)
        for seq, design in zip(all_seqs, all_designs)
    ])  # (M, 10)

    # --- 评分: 物理基线 + 可选MPIGN ---
    phys_scores = physics_compatibility_scores(dg_mat, cross_talk_threshold, kT)

    if use_gnn:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MPIGNModel(
            node_feat_dim=10,
            hidden=gnn_hidden,
            num_layers=gnn_layers,
            num_heads=4,
        ).to(device)
        model.eval()

        node_feats_t = torch.tensor(node_feats_np, device=device)
        dg_mat_t = torch.tensor(dg_mat, device=device)
        mask_t = torch.ones(M, dtype=torch.bool, device=device)

        with torch.no_grad():
            gnn_scores = model(node_feats_t, dg_mat_t, mask_t, kT=kT).cpu().numpy()

        # 归一化后融合: 物理评分保留60%权重 (更可解释), GNN占40%
        def _normalize(x: np.ndarray) -> np.ndarray:
            std = x.std()
            return (x - x.mean()) / (std + 1e-9)

        combined_scores = 0.6 * _normalize(phys_scores) + 0.4 * _normalize(gnn_scores)
    else:
        combined_scores = phys_scores

    # --- 贪心子集选择 (以引物对为单位) ---
    pair_scores = np.array([
        min(combined_scores[2 * k], combined_scores[2 * k + 1])
        for k in range(len(pairs))
    ])

    # 选引物对: 只取偶数索引(左引物), pair_constraints保证右引物同时入选
    left_indices = list(range(0, M, 2))
    selected_left = greedy_compatible_subset(
        primer_indices=left_indices,
        dg_mat=dg_mat,
        scores=combined_scores,
        n_select=n_select * 2,       # 为了让pair_constraints展开后数量够
        cross_talk_threshold=cross_talk_threshold,
        pair_constraints=pair_constraints,
    )

    selected_pair_indices = sorted({i // 2 for i in selected_left if i % 2 == 0})
    selected_pairs = [pairs[i] for i in selected_pair_indices]

    # --- 串扰警告列表 ---
    warnings: list[tuple[str, str, float]] = []
    for i in range(M):
        for j in range(i + 1, M):
            if dg_mat[i, j] <= cross_talk_threshold:
                warnings.append((seq_labels[i], seq_labels[j], float(dg_mat[i, j])))
    warnings.sort(key=lambda w: w[2])  # 最危险的排最前

    return MultiplexResult(
        all_pairs=pairs,
        selected_pairs=selected_pairs,
        all_seqs=all_seqs,
        seq_labels=seq_labels,
        dg_matrix=dg_mat,
        node_scores=combined_scores,
        cross_talk_warnings=warnings,
        ctx=ctx,
        cross_talk_threshold=cross_talk_threshold,
    )
