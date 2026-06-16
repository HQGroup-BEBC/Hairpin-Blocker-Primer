"""主流程：模板序列 -> 候选引物 -> 发夹阻断设计 -> 排序与Pareto前沿
多重PCR模式：多模板 -> 各自设计候选 -> MPIGN多重兼容性优化"""
from __future__ import annotations

from .candidate_generator import generate_candidates
from .hairpin_designer import design_hairpin_blocker, design_hairpin_blocker_ai
from .offtarget_finder import find_offtarget_window
from .pareto import HairpinBlockerPair, make_pair, pareto_front


def run_design(
    template: str,
    target_region: tuple[int, int] | None = None,
    offtarget_records: list[tuple[str, str]] | None = None,
    params: dict | None = None,
    use_ai_search: bool = False,
) -> tuple[list[HairpinBlockerPair], list[HairpinBlockerPair]]:
    """运行完整设计流程（单模板）。

    use_ai_search=True时，用结构感知TargetStruct-GNN代理模型加速发夹搜索空间筛选：
    推断时从stem_len/loop_len直接构造点括号（零NUPACK运行时依赖），
    训练时使用NNN数据集TargetStruct列（NUPACK预算）；预期R²≈0.924。

    返回 (全部结果按pair_si降序排序, Pareto前沿子集)。
    """
    offtarget_records = offtarget_records or []

    surrogate = None
    if use_ai_search:
        from .gnn_model import train_default_gnn

        surrogate = train_default_gnn()

    candidates = generate_candidates(template, target_region, params)

    pairs: list[HairpinBlockerPair] = []
    for candidate in candidates:
        left_window = find_offtarget_window(candidate.left_seq, offtarget_records)
        right_window = find_offtarget_window(candidate.right_seq, offtarget_records)

        if use_ai_search:
            left_design = design_hairpin_blocker_ai(candidate.left_seq, left_window, surrogate)
            right_design = design_hairpin_blocker_ai(candidate.right_seq, right_window, surrogate)
        else:
            left_design = design_hairpin_blocker(candidate.left_seq, left_window)
            right_design = design_hairpin_blocker(candidate.right_seq, right_window)

        pairs.append(make_pair(candidate, left_design, right_design))

    pairs.sort(key=lambda p: p.pair_si, reverse=True)
    front = pareto_front(pairs)
    return pairs, front


def run_multiplex_design(
    templates: list[tuple[str, str]],
    offtarget_records: list[tuple[str, str]] | None = None,
    params: dict | None = None,
    n_pairs_per_target: int = 3,
    n_select: int | None = None,
    cross_talk_threshold: float = -5.0,
    physics_ctx=None,
    use_ai_search: bool = False,
    use_gnn_scoring: bool = True,
):
    """多模板多重PCR设计流程。

    对每个目标序列独立运行 run_design() 获取候选引物对，
    然后将所有候选汇总至 MPIGN，通过图注意力网络+物理环境修正
    优化选出在多重PCR中串扰最少的引物组合。

    参数:
      templates:          [(名称, 序列), ...] 多个目标序列
      offtarget_records:  非目标模板库 (供 find_offtarget_window 使用)
      params:             primer3 参数 (各目标共用)
      n_pairs_per_target: 每个目标生成的候选引物对数量
      n_select:           最终选出的引物对数量 (None → len(templates))
      cross_talk_threshold: 安全串扰ΔG阈值 (kcal/mol)
      physics_ctx:        PhysicsContext 物理条件 (None → 标准管式PCR)
      use_ai_search:      是否用TargetStruct-GNN加速发夹候选筛选
      use_gnn_scoring:    是否启用MPIGN神经网络评分

    返回: MultiplexResult (含selected_pairs / dg_matrix / node_scores / warnings)
    """
    from .multiplex_gnn import run_mpign, PhysicsContext

    if physics_ctx is None:
        physics_ctx = PhysicsContext()

    offtarget_records = offtarget_records or []

    surrogate = None
    if use_ai_search:
        from .gnn_model import train_default_gnn

        surrogate = train_default_gnn(verbose=False)

    if n_select is None:
        n_select = len(templates)

    # --- 为每个目标序列设计候选引物对 ---
    local_params = dict(params) if params else {}
    local_params.setdefault("PRIMER_NUM_RETURN", n_pairs_per_target)

    all_pairs: list[HairpinBlockerPair] = []
    for name, template in templates:
        candidates = generate_candidates(template, None, local_params)
        for candidate in candidates:
            left_window = find_offtarget_window(candidate.left_seq, offtarget_records)
            right_window = find_offtarget_window(candidate.right_seq, offtarget_records)
            if use_ai_search and surrogate is not None:
                left_d = design_hairpin_blocker_ai(candidate.left_seq, left_window, surrogate)
                right_d = design_hairpin_blocker_ai(candidate.right_seq, right_window, surrogate)
            else:
                left_d = design_hairpin_blocker(candidate.left_seq, left_window)
                right_d = design_hairpin_blocker(candidate.right_seq, right_window)
            all_pairs.append(make_pair(candidate, left_d, right_d))

    if not all_pairs:
        raise ValueError("所有目标序列均未能生成候选引物对")

    # --- MPIGN 多重兼容性优化 ---
    return run_mpign(
        pairs=all_pairs,
        n_select=n_select,
        cross_talk_threshold=cross_talk_threshold,
        ctx=physics_ctx,
        use_gnn=use_gnn_scoring,
    )
