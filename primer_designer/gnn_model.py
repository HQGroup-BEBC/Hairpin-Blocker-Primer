"""结构感知(Structure-Aware)GNN代理模型——DNA发夹热力学预测。

设计原则：
  训练时使用 Ke et al. 2025 NNN数据集中已有的 TargetStruct（NUPACK点括号），
  以氢键边作为第4类图关系输入，实现结构感知（消融实验 R²=0.924/MAE=0.151）。
  推断时从已知设计参数（stem_len/loop_len）直接构造点括号，
  无需在运行时调用 NUPACK/ViennaRNA——零外部结构预测依赖。

图结构（四类边）：
  ① 正向骨架边(5'→3')   —— 拓扑固定，由序列长度直接给出
  ② 反向骨架边(3'→5')   —— 同上
  ③ 软配对边(学习得到)   —— PairAttention子网从序列特征学习，
     叠加WC相容性先验(A-T=1, G-C=1, G-T摇摆=0.5)和最小环长掩码(|i-j|<4 → 0)
  ④ 氢键边(结构感知)     —— 由点括号解析而来，训练用TargetStruct(NUPACK预算),
     推断用从stem/loop参数构造的设计点括号

训练数据（需另行下载，外部数据集）:
  Ke et al., Nat. Commun. 2025, "Array Melt"
  GreenleafLab/nnn_paper, MIT License
  文件: external_data/nnn_dna_thermo/fitted_variant_arr.csv
  n≈19,738 (two_state=True筛选后，TargetStruct列全部非空)

Pipeline集成:
  GNNSurrogate.predict_dg(seq, temp_c=60.0, dot_bracket=None) → 预测ΔG(60°C)
  dot_bracket 由 design_hairpin_blocker_ai 传入（从stem_len/loop_len构造）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch import nn

DEFAULT_MAX_LEN: int = 40
MIN_LOOP: int = 4

_BASE_IDX: dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3}

# Watson-Crick相容性矩阵: [A,C,G,T] x [A,C,G,T]
# A-T, T-A, G-C, C-G: weight=1.0  G-T, T-G摇摆: weight=0.5  其余: 0
_WC = np.array(
    [
        [0, 0, 0, 1.0],   # A 与 T配对
        [0, 0, 1.0, 0],   # C 与 G配对
        [0, 1.0, 0, 0.5], # G 与 C配对, G-T摇摆
        [1.0, 0, 0.5, 0], # T 与 A配对, T-G摇摆
    ],
    dtype=np.float32,
)
_WC_TENSOR = torch.tensor(_WC)  # (4,4)


# ---------------------------------------------------------------------------
# 结构辅助函数
# ---------------------------------------------------------------------------

def _parse_dotbracket(db: str, max_len: int) -> np.ndarray:
    """点括号字符串 → 氢键邻接矩阵 (max_len × max_len)，对称。"""
    hbond = np.zeros((max_len, max_len), dtype=np.float32)
    stack: list[int] = []
    for i, c in enumerate(db[:max_len]):
        if c == '(':
            stack.append(i)
        elif c == ')' and stack:
            j = stack.pop()
            hbond[i, j] = 1.0
            hbond[j, i] = 1.0
    return hbond


def design_dotbracket(total_len: int, stem_len: int, loop_len: int) -> str:
    """从发夹阻断引物设计参数构造点括号，供推断时使用（不需要NUPACK）。

    hairpin_primer_seq = stem_comp(stem_len) + loop(loop_len) + primer_body + primer_3stem(stem_len)
    配对: 位置 i ↔ 位置 (total_len-1-i),  i = 0..stem_len-1
    单链区: 位置 stem_len..(total_len-stem_len-1)
    """
    db = ['.'] * total_len
    for i in range(stem_len):
        db[i] = '('
        db[total_len - 1 - i] = ')'
    return ''.join(db)


# ---------------------------------------------------------------------------
# 序列/图编码辅助函数
# ---------------------------------------------------------------------------

def _encode_seq_node(seq: str, max_len: int = DEFAULT_MAX_LEN) -> np.ndarray:
    """One-hot, shape=(max_len, 4), 右侧补零。"""
    arr = np.zeros((max_len, 4), dtype=np.float32)
    for i, ch in enumerate(seq[:max_len]):
        if ch in _BASE_IDX:
            arr[i, _BASE_IDX[ch]] = 1.0
    return arr


def _build_backbone(seq_len: int, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """构建骨架邻接矩阵(正向/反向)，shape=(max_len, max_len)。"""
    fwd = np.zeros((max_len, max_len), dtype=np.float32)
    bwd = np.zeros((max_len, max_len), dtype=np.float32)
    for i in range(min(seq_len - 1, max_len - 1)):
        fwd[i, i + 1] = 1.0
        bwd[i + 1, i] = 1.0
    return fwd, bwd


# ---------------------------------------------------------------------------
# 网络模块
# ---------------------------------------------------------------------------

class SeqEncoder(nn.Module):
    """小型1D-CNN：将per-position one-hot特征编码为上下文敏感特征。
    最近邻堆叠相互作用(nearest-neighbor stacking)正好被k=3的局部卷积捕获。
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, 4) -> (B, 4, N) -> (B, hidden, N) -> (B, N, hidden)
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class PairAttention(nn.Module):
    """从序列特征学习软配对矩阵（辅助结构感知，非替代）。

    S[b,i,j] = (W_q h_i)·(W_k h_j) / sqrt(d_k)
    A_pair[b,i,j] = sigmoid(S) * WC_compat[seq_i,seq_j] * loop_mask * valid_mask
    """

    def __init__(self, hidden: int, d_k: int = 16):
        super().__init__()
        self.W_q = nn.Linear(hidden, d_k, bias=False)
        self.W_k = nn.Linear(hidden, d_k, bias=False)
        self.scale = math.sqrt(d_k)

    def forward(
        self,
        h: torch.Tensor,         # (B, N, hidden)
        x_onehot: torch.Tensor,  # (B, N, 4)
        mask: torch.Tensor,       # (B, N)
        max_len: int,
    ) -> torch.Tensor:
        q = self.W_q(h)
        k = self.W_k(h)
        S = torch.bmm(q, k.transpose(1, 2)) / self.scale

        wc_mat = _WC_TENSOR.to(h.device)
        wc = torch.einsum("bip,pq,bjq->bij", x_onehot, wc_mat, x_onehot)

        pos = torch.arange(max_len, device=h.device)
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
        loop_ok = (dist >= MIN_LOOP).float()

        valid = mask.unsqueeze(2) * mask.unsqueeze(1)
        return torch.sigmoid(S) * wc * loop_ok.unsqueeze(0) * valid


class RelGraphConv(nn.Module):
    """关系图卷积层：每种边类型有独立线性变换，残差+LayerNorm。"""

    def __init__(self, dim: int, num_relations: int = 4):
        super().__init__()
        self.self_lin = nn.Linear(dim, dim)
        self.rel_lins = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_relations)])
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        h: torch.Tensor,
        adj_list: list[torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.self_lin(h)
        for A, lin in zip(adj_list, self.rel_lins):
            msg = lin(h)
            agg = torch.bmm(A, msg)
            deg = A.sum(-1, keepdim=True).clamp(min=1.0)
            out = out + agg / deg
        out = torch.relu(self.norm(out))
        return out * mask.unsqueeze(-1)


class AttnPool(nn.Module):
    """全局注意力池化：变长节点序列 → 定长图表示。"""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(h).squeeze(-1)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        alpha = torch.softmax(scores, dim=-1)
        return torch.bmm(alpha.unsqueeze(1), h).squeeze(1)


class StructureFreeGNN(nn.Module):
    """序列 + 结构(可选) → (dH_norm, dG37_norm)。

    四类图边：正向骨架 / 反向骨架 / 软配对(PairAttention) / 氢键(TargetStruct点括号)。
    训练时传入 adj_hbond（来自NNN数据集TargetStruct列，NUPACK预算），
    推断时传入由设计参数构造的点括号（零NUPACK运行时依赖）。
    """

    def __init__(self, hidden: int = 48, num_layers: int = 4, max_len: int = DEFAULT_MAX_LEN):
        super().__init__()
        self.max_len = max_len
        self.encoder = SeqEncoder(hidden)
        self.input_proj = nn.Linear(hidden, hidden)
        self.pair_attn = PairAttention(hidden, d_k=24)
        self.layers = nn.ModuleList([RelGraphConv(hidden, num_relations=4) for _ in range(num_layers)])
        self.pool = AttnPool(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(
        self,
        x: torch.Tensor,          # (B, N, 4) one-hot
        adj_fwd: torch.Tensor,    # (B, N, N) 正向骨架
        adj_bwd: torch.Tensor,    # (B, N, N) 反向骨架
        mask: torch.Tensor,       # (B, N)
        adj_hbond: torch.Tensor,  # (B, N, N) 氢键（可为全零）
    ) -> torch.Tensor:            # (B, 2) → [dH_norm, dG37_norm]
        h = self.encoder(x) * mask.unsqueeze(-1)
        h = self.input_proj(h) * mask.unsqueeze(-1)
        A_pair = self.pair_attn(h, x, mask, self.max_len)
        adj_list = [adj_fwd, adj_bwd, A_pair, adj_hbond]
        for layer in self.layers:
            h = h + layer(h, adj_list, mask)
        g = self.pool(h, mask)
        return self.head(g)

    def get_pairing_matrix(self, seq: str, max_len: Optional[int] = None) -> np.ndarray:
        """返回软配对概率矩阵(max_len × max_len)，供GUI可视化。"""
        max_len = max_len or self.max_len
        dev = next(self.parameters()).device
        x_np = _encode_seq_node(seq, max_len)
        n = len(seq)
        fwd_np, bwd_np = _build_backbone(n, max_len)
        mask_np = np.zeros(max_len, dtype=np.float32)
        mask_np[:n] = 1.0

        x    = torch.tensor(x_np).unsqueeze(0).to(dev)
        fwd  = torch.tensor(fwd_np).unsqueeze(0).to(dev)
        bwd  = torch.tensor(bwd_np).unsqueeze(0).to(dev)
        mask = torch.tensor(mask_np).unsqueeze(0).to(dev)

        self.eval()
        with torch.no_grad():
            h = self.encoder(x) * mask.unsqueeze(-1)
            h = self.input_proj(h) * mask.unsqueeze(-1)
            A_pair = self.pair_attn(h, x, mask, max_len)
        return A_pair.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------
# 训练用辅助函数
# ---------------------------------------------------------------------------

def _make_batch_arrays(
    seqs: list[str],
    structs: Optional[list[str]],  # 点括号列表；None → 氢键矩阵全零
    max_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 (xs, fwds, bwds, masks, hbonds)，均为 float32 numpy 数组。"""
    B = len(seqs)
    xs     = np.zeros((B, max_len, 4),        dtype=np.float32)
    fwds   = np.zeros((B, max_len, max_len),  dtype=np.float32)
    bwds   = np.zeros((B, max_len, max_len),  dtype=np.float32)
    masks  = np.zeros((B, max_len),           dtype=np.float32)
    hbonds = np.zeros((B, max_len, max_len),  dtype=np.float32)

    for b, seq in enumerate(seqs):
        n = min(len(seq), max_len)
        for i, ch in enumerate(seq[:n]):
            if ch in _BASE_IDX:
                xs[b, i, _BASE_IDX[ch]] = 1.0
        for i in range(n - 1):
            fwds[b, i, i + 1] = 1.0
            bwds[b, i + 1, i] = 1.0
        masks[b, :n] = 1.0
        if structs is not None:
            hbonds[b] = _parse_dotbracket(structs[b], max_len)

    return xs, fwds, bwds, masks, hbonds


# ---------------------------------------------------------------------------
# GNNSurrogate：已训练模型 + 归一化统计量 + 推断接口
# ---------------------------------------------------------------------------

@dataclass
class _NormStats:
    dh_mean: float
    dh_std: float
    dg37_mean: float
    dg37_std: float


@dataclass
class GNNSurrogate:
    """已训练的结构感知GNN代理模型，提供pipeline推断接口。

    predict_dg(seq, temp_c=60.0, dot_bracket=None) → 预测ΔG(temp_c) [kcal/mol]
    dot_bracket 应由 design_dotbracket(len(seq), stem_len, loop_len) 构造后传入；
    不传时退化为StructureFree模式（氢键边全零）。
    """

    model: StructureFreeGNN
    stats: _NormStats = field(repr=False)
    max_len: int = DEFAULT_MAX_LEN

    def predict_dg(
        self,
        seq: str,
        temp_c: float = 60.0,
        dot_bracket: Optional[str] = None,
    ) -> float:
        """预测给定序列在temp_c(°C)下的发夹折叠ΔG [kcal/mol]。

        ΔG(T) = ΔH·(1 - T_K/310.15) + ΔG_37·(T_K/310.15)
        """
        T_K = temp_c + 273.15
        dh, dg37 = self._predict_dh_dg37(seq, dot_bracket)
        return dh * (1 - T_K / 310.15) + dg37 * (T_K / 310.15)

    def _predict_dh_dg37(
        self, seq: str, dot_bracket: Optional[str] = None
    ) -> tuple[float, float]:
        dev = next(self.model.parameters()).device
        structs = [dot_bracket] if dot_bracket is not None else None
        xs, fwds, bwds, masks, hbonds = _make_batch_arrays([seq], structs, self.max_len)
        x     = torch.tensor(xs,     device=dev)
        fwd   = torch.tensor(fwds,   device=dev)
        bwd   = torch.tensor(bwds,   device=dev)
        mask  = torch.tensor(masks,  device=dev)
        hbond = torch.tensor(hbonds, device=dev)
        self.model.eval()
        with torch.no_grad():
            out = self.model(x, fwd, bwd, mask, hbond)[0].cpu().numpy()
        dh   = out[0] * self.stats.dh_std   + self.stats.dh_mean
        dg37 = out[1] * self.stats.dg37_std + self.stats.dg37_mean
        return float(dh), float(dg37)

    def get_pairing_matrix(self, seq: str) -> np.ndarray:
        """返回软配对概率矩阵(n × n)，n=len(seq)，供GUI绘制发夹结构图。"""
        full = self.model.get_pairing_matrix(seq, self.max_len)
        n = min(len(seq), self.max_len)
        return full[:n, :n]


# ---------------------------------------------------------------------------
# 训练入口
# ---------------------------------------------------------------------------

def train_gnn_on_nnn(
    data_path: str = "external_data/nnn_dna_thermo/fitted_variant_arr.csv",
    hidden: int = 48,
    num_layers: int = 4,
    batch_size: int = 256,
    epochs: int = 80,
    lr: float = 2e-3,
    seed: int = 0,
    verbose: bool = True,
) -> GNNSurrogate:
    """在NNN数据集上训练结构感知GNN，返回GNNSurrogate。

    训练时使用 TargetStruct 列（NUPACK点括号，数据集内已预算，n=19738全覆盖）
    作为氢键边输入，无需运行时调用NUPACK。
    推断时由调用方传入由设计参数构造的点括号（见 design_dotbracket）。
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("训练GNN需要pandas: pip install pandas") from e

    df = pd.read_csv(data_path, sep="\t")
    df = df[df["two_state"] == True].dropna(
        subset=["dG_37", "dH", "RefSeq", "TargetStruct"]
    ).copy()
    df = df[df["RefSeq"].str.len() <= DEFAULT_MAX_LEN]
    if verbose:
        print(f"[GNN] 训练数据: n={len(df)}（含TargetStruct结构边）", flush=True)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_test = int(0.2 * len(df))
    train_df = df.iloc[idx[n_test:]].reset_index(drop=True)
    test_df  = df.iloc[idx[:n_test]].reset_index(drop=True)

    dh_mean   = float(train_df["dH"].mean())
    dh_std    = float(train_df["dH"].std())
    dg37_mean = float(train_df["dG_37"].mean())
    dg37_std  = float(train_df["dG_37"].std())
    stats = _NormStats(dh_mean, dh_std, dg37_mean, dg37_std)

    y_train = np.stack([
        (train_df["dH"].to_numpy()    - dh_mean)   / dh_std,
        (train_df["dG_37"].to_numpy() - dg37_mean) / dg37_std,
    ], axis=1).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"[GNN] 设备: {device}", flush=True)

    torch.manual_seed(seed)
    model = StructureFreeGNN(hidden=hidden, num_layers=num_layers, max_len=DEFAULT_MAX_LEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n_train     = len(train_df)
    train_seqs  = train_df["RefSeq"].tolist()
    train_structs = train_df["TargetStruct"].tolist()
    y_train_t   = torch.tensor(y_train, device=device)

    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(n_train)
        total_loss = 0.0
        for start in range(0, n_train, batch_size):
            bi    = perm[start : start + batch_size]
            seqs  = [train_seqs[i]   for i in bi]
            strs  = [train_structs[i] for i in bi]
            xs, fwds, bwds, masks, hbonds = _make_batch_arrays(seqs, strs, DEFAULT_MAX_LEN)
            x     = torch.tensor(xs,     device=device)
            fwd   = torch.tensor(fwds,   device=device)
            bwd   = torch.tensor(bwds,   device=device)
            mask  = torch.tensor(masks,  device=device)
            hbond = torch.tensor(hbonds, device=device)
            yt    = y_train_t[bi]

            optimizer.zero_grad()
            pred = model(x, fwd, bwd, mask, hbond)
            loss = loss_fn(pred, yt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bi)

        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            model.eval()
            test_seqs    = test_df["RefSeq"].tolist()
            test_structs = test_df["TargetStruct"].tolist()
            dg37_exp     = test_df["dG_37"].to_numpy(dtype=np.float32)
            dg37_preds: list[float] = []
            _EVAL_BATCH = 512
            with torch.no_grad():
                for _s in range(0, len(test_seqs), _EVAL_BATCH):
                    _seqs = test_seqs[_s : _s + _EVAL_BATCH]
                    _strs = test_structs[_s : _s + _EVAL_BATCH]
                    _xs, _fwds, _bwds, _masks, _hbonds = _make_batch_arrays(
                        _seqs, _strs, DEFAULT_MAX_LEN
                    )
                    _out = model(
                        torch.tensor(_xs,     device=device),
                        torch.tensor(_fwds,   device=device),
                        torch.tensor(_bwds,   device=device),
                        torch.tensor(_masks,  device=device),
                        torch.tensor(_hbonds, device=device),
                    ).cpu().numpy()
                    dg37_preds.extend(
                        (_out[:, 1] * stats.dg37_std + stats.dg37_mean).tolist()
                    )
            dg37_pred = np.array(dg37_preds, dtype=np.float32)
            ss_res = np.sum((dg37_exp - dg37_pred) ** 2)
            ss_tot = np.sum((dg37_exp - dg37_exp.mean()) ** 2)
            r2  = 1 - ss_res / ss_tot
            mae = np.mean(np.abs(dg37_exp - dg37_pred))
            print(
                f"[GNN] epoch {epoch:3d}  MSE(norm)={total_loss/n_train:.4f}"
                f"  test R²={r2:.3f} MAE={mae:.3f}",
                flush=True,
            )

    return GNNSurrogate(model=model, stats=stats, max_len=DEFAULT_MAX_LEN)


def train_default_gnn(
    data_path: str = "external_data/nnn_dna_thermo/fitted_variant_arr.csv",
    verbose: bool = False,
) -> GNNSurrogate:
    """快速训练入口，供pipeline.run_design(use_ai_search=True)调用。"""
    return train_gnn_on_nnn(data_path=data_path, epochs=60, verbose=verbose)
