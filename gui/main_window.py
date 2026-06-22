"""Hairpin-Blocker Designer 主窗口（Tkinter + matplotlib）"""
from __future__ import annotations

import csv
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

import matplotlib
# 配置中文字体——必须在导入 matplotlib 子模块之前设置
# Windows 优先使用微软雅黑；若不存在则依次 fallback 至 SimHei / Arial Unicode MS
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Arial Unicode MS",
    "WenQuanYi Micro Hei", "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False   # 防止负号显示为方块

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from primer_designer.candidate_generator import DEFAULT_PARAMS
from primer_designer.pipeline import run_design
from primer_designer.sequence_io import load_fasta_file, parse_fasta_text

SAMPLE_DATA_DIR = Path(__file__).resolve().parent.parent / "sample_data"

VERDICT_LABELS = {"reject": "淘汰", "acceptable": "可用", "excellent": "优秀"}
VERDICT_TAGS = {"reject": "verdict_reject", "acceptable": "verdict_acceptable", "excellent": "verdict_excellent"}
VERDICT_RANK = {"reject": 0, "acceptable": 1, "excellent": 2}


class MainWindow(ttk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master)
        self.master = master
        master.title("Hairpin-Blocker Designer —— 发夹阻断引物设计")
        master.geometry("1360x860")

        self.pairs: list = []
        self.front_ids: set[str] = set()
        self._queue: queue.Queue = queue.Queue()
        self._scatter = None
        self._highlight_artist = None
        self._mplex_result = None    # 多重PCR优化结果 (MultiplexResult)
        self._current_pair = None    # 当前选中的候选对 (供3D查看器使用)

        self._build_widgets()
        self.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # 构建界面
    # ------------------------------------------------------------------
    def _build_widgets(self) -> None:
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned, width=380)
        paned.add(left, weight=0)

        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        self._build_input_panel(left)
        self._build_result_panel(right)

    def _build_input_panel(self, parent: ttk.Frame) -> None:
        # 目标序列
        frm_target = ttk.LabelFrame(parent, text="目标序列 (FASTA 或纯序列)")
        frm_target.pack(fill=tk.X, padx=8, pady=(8, 4))

        self.txt_target = scrolledtext.ScrolledText(frm_target, height=6, wrap=tk.WORD)
        self.txt_target.pack(fill=tk.X, padx=6, pady=(6, 2))
        self.txt_target.bind("<KeyRelease>", lambda e: self._update_target_len())

        row = ttk.Frame(frm_target)
        row.pack(fill=tk.X, padx=6, pady=(0, 6))
        ttk.Button(row, text="加载文件...", command=self._load_target_file).pack(side=tk.LEFT)
        ttk.Button(row, text="加载示例", command=self._load_sample).pack(side=tk.LEFT, padx=(6, 0))
        self.lbl_target_len = ttk.Label(row, text="长度: 0 bp")
        self.lbl_target_len.pack(side=tk.RIGHT)

        # 目标区域
        frm_region = ttk.LabelFrame(parent, text="目标区域 (可选, 0-based 起始位置+长度)")
        frm_region.pack(fill=tk.X, padx=8, pady=4)
        row = ttk.Frame(frm_region)
        row.pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(row, text="起始位置:").pack(side=tk.LEFT)
        self.var_region_start = tk.StringVar()
        ttk.Entry(row, textvariable=self.var_region_start, width=8).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(row, text="长度:").pack(side=tk.LEFT)
        self.var_region_len = tk.StringVar()
        ttk.Entry(row, textvariable=self.var_region_len, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            row, text="扩增子顾问...", command=self._open_amplicon_advisor,
        ).pack(side=tk.RIGHT, padx=(0, 4))

        # 非目标模板库
        frm_off = ttk.LabelFrame(parent, text="非目标模板库 (FASTA, 可选)")
        frm_off.pack(fill=tk.X, padx=8, pady=4)
        self.txt_offtarget = scrolledtext.ScrolledText(frm_off, height=6, wrap=tk.WORD)
        self.txt_offtarget.pack(fill=tk.X, padx=6, pady=(6, 2))
        row = ttk.Frame(frm_off)
        row.pack(fill=tk.X, padx=6, pady=(0, 6))
        ttk.Button(row, text="加载文件...", command=self._load_offtarget_file).pack(side=tk.LEFT)
        ttk.Label(
            row,
            text="未提供时 ΔG_offtarget=0，SI按无脱靶竞争封顶为excellent",
            foreground="gray",
        ).pack(side=tk.LEFT, padx=(8, 0))

        # 设计参数
        frm_params = ttk.LabelFrame(parent, text="设计参数")
        frm_params.pack(fill=tk.X, padx=8, pady=4)

        self.param_vars: dict[str, tk.StringVar] = {}

        def add_row(label, specs):
            row = ttk.Frame(frm_params)
            row.pack(fill=tk.X, padx=6, pady=2)
            ttk.Label(row, text=label, width=10).pack(side=tk.LEFT)
            for sub_label, key, default in specs:
                ttk.Label(row, text=sub_label).pack(side=tk.LEFT)
                var = tk.StringVar(value=str(default))
                self.param_vars[key] = var
                ttk.Entry(row, textvariable=var, width=6).pack(side=tk.LEFT, padx=(2, 8))

        add_row("引物长度", [
            ("最小", "PRIMER_MIN_SIZE", DEFAULT_PARAMS["PRIMER_MIN_SIZE"]),
            ("最优", "PRIMER_OPT_SIZE", DEFAULT_PARAMS["PRIMER_OPT_SIZE"]),
            ("最大", "PRIMER_MAX_SIZE", DEFAULT_PARAMS["PRIMER_MAX_SIZE"]),
        ])
        add_row("Tm(℃)", [
            ("最小", "PRIMER_MIN_TM", DEFAULT_PARAMS["PRIMER_MIN_TM"]),
            ("最优", "PRIMER_OPT_TM", DEFAULT_PARAMS["PRIMER_OPT_TM"]),
            ("最大", "PRIMER_MAX_TM", DEFAULT_PARAMS["PRIMER_MAX_TM"]),
        ])
        add_row("GC%", [
            ("最小", "PRIMER_MIN_GC", DEFAULT_PARAMS["PRIMER_MIN_GC"]),
            ("最大", "PRIMER_MAX_GC", DEFAULT_PARAMS["PRIMER_MAX_GC"]),
        ])
        add_row("产物大小", [
            ("最小", "PRODUCT_MIN", DEFAULT_PARAMS["PRIMER_PRODUCT_SIZE_RANGE"][0][0]),
            ("最大", "PRODUCT_MAX", DEFAULT_PARAMS["PRIMER_PRODUCT_SIZE_RANGE"][0][1]),
        ])
        add_row("候选数量", [
            ("数量", "PRIMER_NUM_RETURN", DEFAULT_PARAMS["PRIMER_NUM_RETURN"]),
        ])

        # AI加速搜索
        self.var_use_ai_search = tk.BooleanVar(value=False)
        frm_ai = ttk.Frame(parent)
        frm_ai.pack(fill=tk.X, padx=8, pady=(4, 0))
        ttk.Checkbutton(
            frm_ai,
            text="启用TargetStruct-GNN加速搜索 (结构感知，训练约1-2min)",
            variable=self.var_use_ai_search,
        ).pack(side=tk.LEFT)

        # 多重PCR模式 (MPIGN)
        frm_mplex = ttk.LabelFrame(parent, text="多重PCR优化 (MPIGN — 多重引物相互作用图注意力网络 + 盐修正)")
        frm_mplex.pack(fill=tk.X, padx=8, pady=(6, 0))

        row_mplex_top = ttk.Frame(frm_mplex)
        row_mplex_top.pack(fill=tk.X, padx=6, pady=(4, 0))
        self.var_mplex_mode = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row_mplex_top,
            text="启用多重PCR模式 (目标序列框中可粘贴多条FASTA)",
            variable=self.var_mplex_mode,
            command=self._on_mplex_toggle,
        ).pack(side=tk.LEFT)
        ttk.Button(
            row_mplex_top,
            text="加载多重PCR示例",
            command=self._load_multiplex_sample,
        ).pack(side=tk.RIGHT)

        row_mp1 = ttk.Frame(frm_mplex)
        row_mp1.pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(row_mp1, text="每靶候选数:").pack(side=tk.LEFT)
        self.var_mplex_n_per = tk.StringVar(value="3")
        ttk.Entry(row_mp1, textvariable=self.var_mplex_n_per, width=5).pack(side=tk.LEFT, padx=(2, 12))
        ttk.Label(row_mp1, text="最终选出对数:").pack(side=tk.LEFT)
        self.var_mplex_n_sel = tk.StringVar(value="")
        ttk.Entry(row_mp1, textvariable=self.var_mplex_n_sel, width=5).pack(side=tk.LEFT, padx=(2, 12))
        ttk.Label(row_mp1, text="串扰阈值(kcal):").pack(side=tk.LEFT)
        self.var_mplex_xt = tk.StringVar(value="-5.0")
        ttk.Entry(row_mp1, textvariable=self.var_mplex_xt, width=6).pack(side=tk.LEFT, padx=2)

        self.lbl_mplex_hint = ttk.Label(
            frm_mplex,
            text="提示: 禁用时仅对当前候选池做单模板设计",
            foreground="gray", font=("", 8),
        )
        self.lbl_mplex_hint.pack(anchor=tk.W, padx=6, pady=(0, 4))

        # 运行控制
        frm_run = ttk.Frame(parent)
        frm_run.pack(fill=tk.X, padx=8, pady=8)
        self.btn_run = ttk.Button(frm_run, text="开始设计", command=self._on_run)
        self.btn_run.pack(side=tk.LEFT)
        self.btn_export = ttk.Button(frm_run, text="导出CSV", command=self._on_export, state=tk.DISABLED)
        self.btn_export.pack(side=tk.LEFT, padx=(6, 0))

        self.progress = ttk.Progressbar(parent, mode="indeterminate")
        self.progress.pack(fill=tk.X, padx=8, pady=(0, 4))

        self.lbl_status = ttk.Label(parent, text="就绪")
        self.lbl_status.pack(fill=tk.X, padx=8, pady=(0, 8))

    def _build_result_panel(self, parent: ttk.Frame) -> None:
        columns = ("rank", "left_hp", "right_hp", "product", "si", "ei", "verdict", "pareto")
        headings = {
            "rank": "#",
            "left_hp": "正向发夹阻断引物 (5'->3')",
            "right_hp": "反向发夹阻断引物 (5'->3')",
            "product": "产物(bp)",
            "si": "SI",
            "ei": "EI",
            "verdict": "综合判定",
            "pareto": "Pareto",
        }
        widths = {
            "rank": 36, "left_hp": 260, "right_hp": 260,
            "product": 60, "si": 60, "ei": 60, "verdict": 70, "pareto": 60,
        }

        frm_table = ttk.Frame(parent)
        frm_table.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))

        self.tree = ttk.Treeview(frm_table, columns=columns, show="headings", height=10)
        for c in columns:
            self.tree.heading(c, text=headings[c])
            anchor = tk.W if c in ("left_hp", "right_hp") else tk.CENTER
            self.tree.column(c, width=widths[c], anchor=anchor)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        vsb = ttk.Scrollbar(frm_table, orient=tk.VERTICAL, command=self.tree.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=vsb.set)

        self.tree.tag_configure("verdict_excellent", background="#d6f5d6")
        self.tree.tag_configure("verdict_acceptable", background="#fff7cc")
        self.tree.tag_configure("verdict_reject", background="#f8d7da")

        self.tree.bind("<<TreeviewSelect>>", self._on_select_row)

        bottom = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        bottom.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))

        frm_detail = ttk.LabelFrame(bottom, text="详情")
        bottom.add(frm_detail, weight=1)

        # 详情面板工具栏
        frm_detail_tb = ttk.Frame(frm_detail)
        frm_detail_tb.pack(fill=tk.X, padx=4, pady=(4, 2))
        self.btn_3d_viewer = ttk.Button(
            frm_detail_tb,
            text="3D分子结构 (GPU WebGL)",
            command=self._on_open_3d_viewer,
            state=tk.DISABLED,
        )
        self.btn_3d_viewer.pack(side=tk.LEFT)
        self.btn_export_pdb = ttk.Button(
            frm_detail_tb,
            text="导出PDB文件",
            command=self._on_export_pdb,
            state=tk.DISABLED,
        )
        self.btn_export_pdb.pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(
            frm_detail_tb,
            text="GPU渲染 · 鼠标旋转/缩放 · 类PyMOL交互",
            foreground="gray", font=("", 8),
        ).pack(side=tk.LEFT, padx=(10, 0))

        self.txt_detail = scrolledtext.ScrolledText(frm_detail, wrap=tk.NONE, font=("Courier New", 10))
        self.txt_detail.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # 右侧：标签页切换 SI-EI散点图 / 发夹二级结构图
        frm_right = ttk.Frame(bottom)
        bottom.add(frm_right, weight=1)
        nb = ttk.Notebook(frm_right)
        nb.pack(fill=tk.BOTH, expand=True)

        # Tab 1: SI-EI散点图
        tab_scatter = ttk.Frame(nb)
        nb.add(tab_scatter, text="SI-EI散点图")
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=tab_scatter)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("pick_event", self._on_pick)
        self._update_plot()

        # Tab 2: 发夹二级结构图 (2D)
        tab_struct = ttk.Frame(nb)
        nb.add(tab_struct, text="发夹2D结构")
        self.fig_struct = Figure(figsize=(5, 4), dpi=100)
        self.ax_struct = self.fig_struct.add_subplot(111)
        self.canvas_struct = FigureCanvasTkAgg(self.fig_struct, master=tab_struct)
        self.canvas_struct.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_hairpin_placeholder()

        # Tab 3: 发夹3D B-form双螺旋结构图
        tab_3d = ttk.Frame(nb)
        nb.add(tab_3d, text="发夹3D结构")
        self.fig_3d = Figure(figsize=(5, 4), dpi=100)
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=tab_3d)
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_3d_placeholder()

        # Tab 4: 多重PCR串扰热图 + 引物交互网络
        tab_mplex = ttk.Frame(nb)
        nb.add(tab_mplex, text="多重PCR串扰图")
        self.fig_mplex = Figure(figsize=(5, 4), dpi=100)
        self.canvas_mplex = FigureCanvasTkAgg(self.fig_mplex, master=tab_mplex)
        self.canvas_mplex.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_mplex_placeholder()

        self._nb = nb  # 保存引用，供自动切换标签页使用

    # ------------------------------------------------------------------
    # 输入处理
    # ------------------------------------------------------------------
    def _update_target_len(self) -> None:
        text = self.txt_target.get("1.0", tk.END)
        try:
            records = parse_fasta_text(text)
        except ValueError:
            self.lbl_target_len.config(text="长度: -")
            return
        if records:
            self.lbl_target_len.config(text=f"长度: {len(records[0][1])} bp")
        else:
            self.lbl_target_len.config(text="长度: 0 bp")

    def _load_target_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("FASTA", "*.fasta *.fa *.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            records = load_fasta_file(path)
        except (ValueError, OSError) as exc:
            messagebox.showerror("加载失败", str(exc))
            return
        self.txt_target.delete("1.0", tk.END)
        for rid, seq in records:
            self.txt_target.insert(tk.END, f">{rid}\n{seq}\n")
        self._update_target_len()

    def _load_offtarget_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("FASTA", "*.fasta *.fa *.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            records = load_fasta_file(path)
        except (ValueError, OSError) as exc:
            messagebox.showerror("加载失败", str(exc))
            return
        self.txt_offtarget.delete("1.0", tk.END)
        for rid, seq in records:
            self.txt_offtarget.insert(tk.END, f">{rid}\n{seq}\n")

    def _load_sample(self) -> None:
        try:
            target_records = load_fasta_file(str(SAMPLE_DATA_DIR / "target_example.fasta"))
            offtarget_records = load_fasta_file(str(SAMPLE_DATA_DIR / "offtarget_example.fasta"))
        except (ValueError, OSError) as exc:
            messagebox.showerror("加载示例失败", str(exc))
            return
        self.txt_target.delete("1.0", tk.END)
        for rid, seq in target_records:
            self.txt_target.insert(tk.END, f">{rid}\n{seq}\n")
        self.txt_offtarget.delete("1.0", tk.END)
        for rid, seq in offtarget_records:
            self.txt_offtarget.insert(tk.END, f">{rid}\n{seq}\n")
        self._update_target_len()

    def _load_multiplex_sample(self) -> None:
        """加载多重PCR示例数据并自动切换到多重模式。

        示例: 3条癌症突变热点扩增子的多重液体活检PCR面板
          · KRAS exon2  — 结直肠癌/肺腺癌  G12C/G12V/G13D 热点  396 bp
          · BRAF exon15 — 黑色素瘤/结直肠癌 V600E/V600K 热点   388 bp
          · EGFR exon21 — 肺腺癌            L858R 热点          385 bp
        MPIGN 将优化三组引物对以确保多重PCR中相互串扰最小。
        """
        mplex_path = SAMPLE_DATA_DIR / "multiplex_example.fasta"
        offtarget_path = SAMPLE_DATA_DIR / "offtarget_example.fasta"

        try:
            records = load_fasta_file(str(mplex_path))
        except (ValueError, OSError) as exc:
            messagebox.showerror("加载失败", str(exc))
            return

        self.txt_target.delete("1.0", tk.END)
        for rid, seq in records:
            self.txt_target.insert(tk.END, f">{rid}\n{seq}\n")
        self._update_target_len()

        # 加载非目标库 (可选)
        try:
            ot_records = load_fasta_file(str(offtarget_path))
            self.txt_offtarget.delete("1.0", tk.END)
            for rid, seq in ot_records:
                self.txt_offtarget.insert(tk.END, f">{rid}\n{seq}\n")
        except (ValueError, OSError):
            pass

        # 自动启用多重PCR模式并设置推荐参数
        self.var_mplex_mode.set(True)
        self.var_mplex_n_per.set("3")
        self.var_mplex_n_sel.set("3")
        self.var_mplex_xt.set("-5.0")
        self._on_mplex_toggle()

        n = len(records)
        self.lbl_status.config(
            text=f"已加载多重PCR示例: {n} 条靶序列 "
                 f"({', '.join(r[0] for r in records)}) — 点击「开始设计」运行MPIGN优化"
        )

    def _collect_params(self) -> dict:
        try:
            product_min = int(self.param_vars["PRODUCT_MIN"].get())
            product_max = int(self.param_vars["PRODUCT_MAX"].get())
            return {
                "PRIMER_MIN_SIZE": int(self.param_vars["PRIMER_MIN_SIZE"].get()),
                "PRIMER_OPT_SIZE": int(self.param_vars["PRIMER_OPT_SIZE"].get()),
                "PRIMER_MAX_SIZE": int(self.param_vars["PRIMER_MAX_SIZE"].get()),
                "PRIMER_MIN_TM": float(self.param_vars["PRIMER_MIN_TM"].get()),
                "PRIMER_OPT_TM": float(self.param_vars["PRIMER_OPT_TM"].get()),
                "PRIMER_MAX_TM": float(self.param_vars["PRIMER_MAX_TM"].get()),
                "PRIMER_MIN_GC": float(self.param_vars["PRIMER_MIN_GC"].get()),
                "PRIMER_MAX_GC": float(self.param_vars["PRIMER_MAX_GC"].get()),
                "PRIMER_PRODUCT_SIZE_RANGE": [[product_min, product_max]],
                "PRIMER_NUM_RETURN": int(self.param_vars["PRIMER_NUM_RETURN"].get()),
            }
        except ValueError as exc:
            raise ValueError(f"参数格式错误: {exc}") from exc

    # ------------------------------------------------------------------
    # 运行设计 (后台线程) —— _on_run 由多重PCR区域的 _on_mplex_toggle 根据模式路由
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 结果展示
    # ------------------------------------------------------------------
    def _populate_results(self) -> None:
        self.tree.delete(*self.tree.get_children())
        for i, p in enumerate(self.pairs):
            c, l, r = p.candidate, p.left, p.right
            overall = min(l.overall_verdict, r.overall_verdict, key=lambda v: VERDICT_RANK[v])
            pareto_mark = "★" if str(i) in self.front_ids else ""
            self.tree.insert(
                "", tk.END, iid=str(i),
                values=(
                    i + 1,
                    l.hairpin_primer_seq,
                    r.hairpin_primer_seq,
                    c.product_size,
                    f"{p.pair_si:.2f}",
                    f"{p.pair_ei:.2f}",
                    VERDICT_LABELS[overall],
                    pareto_mark,
                ),
                tags=(VERDICT_TAGS[overall],),
            )
        self._update_plot()
        if self.pairs:
            self.tree.selection_set("0")
            self.tree.see("0")

    def _on_select_row(self, event=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        self._show_detail(idx)
        self._highlight_plot_point(idx)

    def _show_detail(self, idx: int) -> None:
        p = self.pairs[idx]
        c, l, r = p.candidate, p.left, p.right
        self._current_pair = p
        self.btn_3d_viewer.config(state=tk.NORMAL)
        self.btn_export_pdb.config(state=tk.NORMAL)

        lines = [
            f"候选 #{idx + 1}   产物大小: {c.product_size} bp   "
            f"pair_SI={p.pair_si:.2f}   pair_EI={p.pair_ei:.2f}",
            "",
        ]
        for label, primer, design in (("正向引物 (Left)", c.left_seq, l), ("反向引物 (Right)", c.right_seq, r)):
            lines.append(f"=== {label} ===")
            lines.append(f"原始引物 (5'->3'):     {primer}")
            lines.append(f"发夹阻断引物 (5'->3'): {design.hairpin_primer_seq}")
            lines.append(f"  茎长: {design.stem_len} bp   环长: {design.loop_len} nt   环序列: {design.loop_seq}")
            lines.append(f"  ΔG_target    = {design.dg_target:8.2f} kcal/mol")
            lines.append(f"  ΔG_hairpin   = {design.dg_hairpin:8.2f} kcal/mol")
            lines.append(f"  ΔG_offtarget = {design.dg_offtarget:8.2f} kcal/mol")
            dg_homo = getattr(design, "dg_homodimer", None)
            if dg_homo is not None:
                homo_flag = "  ⚠ primer-dimer风险" if dg_homo < -5.0 else ""
                lines.append(f"  ΔG_homodimer = {dg_homo:8.2f} kcal/mol{homo_flag}")
            risk = getattr(design, "on_target_risk", None)
            if risk is not None:
                risk_note = {
                    "low": "发夹稳定性适中，目标模板可正常置换",
                    "medium": "中等风险，建议验证低拷贝扩增效率",
                    "high": "⚠ 发夹过稳（ΔG<-8），可能导致Ct延后/标准曲线效率<90%",
                }.get(risk, risk)
                lines.append(f"  on-target风险 = {risk}  ({risk_note})")
            lines.append(
                f"  SI = {design.si:6.2f} ({VERDICT_LABELS[design.verdict_si]})   "
                f"EI = {design.ei:6.2f} ({VERDICT_LABELS[design.verdict_ei]})"
            )
            lines.append("  --- 三态竞争结构图 (越靠下=结合越弱, 发夹应胜出脱靶) ---")
            lines.append("  [1] 发夹折叠态 (分子内, calc_hairpin):")
            for s_line in design.ascii_hairpin.splitlines():
                lines.append("    " + s_line)
            lines.append("  [2] 目标结合态 (与目标模板完全互补链, calc_heterodimer):")
            for s_line in design.ascii_target.splitlines():
                lines.append("    " + s_line)
            lines.append("  [3] 脱靶结合态 (与脱靶模板窗口, calc_heterodimer):")
            if design.ascii_offtarget:
                for s_line in design.ascii_offtarget.splitlines():
                    lines.append("    " + s_line)
            else:
                lines.append("    (库中无显著脱靶位点, 视为不结合)")
            lines.append("")

        self.txt_detail.delete("1.0", tk.END)
        self.txt_detail.insert(tk.END, "\n".join(lines))
        self._draw_hairpin_design(p.left)
        self._draw_hairpin_3d(p.left)

    def _draw_hairpin_placeholder(self) -> None:
        self.ax_struct.clear()
        self.ax_struct.set_axis_off()
        self.ax_struct.text(0.5, 0.5, "选中一条候选引物后显示发夹结构",
                            ha="center", va="center", transform=self.ax_struct.transAxes,
                            color="gray", fontsize=11)
        self.fig_struct.tight_layout()
        self.canvas_struct.draw_idle()

    def _draw_hairpin_design(self, design) -> None:
        """2D ladder hairpin diagram with proper stem-loop topology.

        Layout: 5' stem_comp rises on the left; single-stranded region (loop +
        primer body) arcs over the top in a half-ellipse; 3' stem descends on
        the right — giving the classic hairpin lollipop appearance.
        Pairing: position i ↔ position n-1-i (i=0..stem-1).
        """
        import numpy as _np
        import matplotlib.patches as mpatches

        ax = self.ax_struct
        ax.clear()
        ax.set_aspect("equal")
        ax.set_axis_off()

        # 使用 dna_seq（纯核苷酸，不含[HEG]占位符）计算布局
        seq = getattr(design, "dna_seq", design.hairpin_primer_seq.replace("[HEG]", ""))
        n = len(seq)
        s = design.stem_len
        l = design.loop_len
        n_pb = n - 2 * s - l   # primer body (ss) length
        n_ss = l + n_pb         # total single-stranded in the arc

        # ----- Layout parameters -----
        h = 1.0                              # vertical spacing between base pairs
        d = max(1.4, n_ss * 0.14)           # stem half-width; grows with arc size
        b_arc = max(2.8, n_ss * 0.32)       # arc height above stem top
        base_r = max(0.20, min(0.38, 7.0 / n))
        Y_top = (s - 1) * h                 # y of innermost stem pair

        xs = _np.zeros(n)
        ys = _np.zeros(n)

        # Left stem (stem_comp): pos 0 at bottom (5' end), pos s-1 at top
        for i in range(s):
            xs[i] = -d
            ys[i] = i * h

        # Single-stranded arc (loop + primer body): half-ellipse from (-d, Y_top) to (+d, Y_top)
        for k in range(n_ss):
            t = (k + 1) / (n_ss + 1)
            xs[s + k] = -d * _np.cos(_np.pi * t)
            ys[s + k] = Y_top + b_arc * _np.sin(_np.pi * t)

        # Right stem (primer 3' region): pos n-s at top, pos n-1 at bottom (3' end)
        for j in range(s):
            xs[n - s + j] = +d
            ys[n - s + j] = Y_top - j * h

        # ----- Colors -----
        CLR = {"sc": "#1565C0", "lp": "#E65100", "pb": "#607D8B", "s3": "#C62828"}
        colors = []
        for i in range(n):
            if i < s:
                colors.append(CLR["sc"])
            elif i < s + l:
                colors.append(CLR["lp"])
            elif i < s + l + n_pb:
                colors.append(CLR["pb"])
            else:
                colors.append(CLR["s3"])

        # ----- Backbone -----
        for i in range(n - 1):
            ax.plot([xs[i], xs[i + 1]], [ys[i], ys[i + 1]],
                    color="#BDBDBD", lw=1.2, zorder=1, solid_capstyle="round")

        # ----- Base-pair bridges -----
        for i in range(s):
            j = n - 1 - i
            ax.plot([xs[i] + base_r, xs[j] - base_r], [ys[i], ys[j]],
                    color="#555555", lw=1.8, zorder=2)

        # ----- Nucleotide circles + letters -----
        fsize = max(5, min(9, 130 // max(n, 1)))
        for i in range(n):
            circ = mpatches.Circle((xs[i], ys[i]), base_r, color=colors[i], zorder=3)
            ax.add_patch(circ)
            ax.text(xs[i], ys[i], seq[i], ha="center", va="center",
                    fontsize=fsize, color="white", fontweight="bold",
                    fontfamily="monospace", zorder=4)

        # HEG非核苷酸连接子标注（位于loop末和primer body之间的弧上）
        if n_ss > 0 and l > 0 and n_pb > 0:
            t_heg = (l + 0.5) / (n_ss + 1)
            x_heg = -d * _np.cos(_np.pi * t_heg)
            y_heg = Y_top + b_arc * _np.sin(_np.pi * t_heg)
            ax.annotate(
                "HEG", xy=(x_heg, y_heg), fontsize=6.5, color="#E65100",
                ha="center", va="center", fontweight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=0.25", fc="#FFF3E0", ec="#E65100", lw=1.0),
            )

        # 5' / 3' end labels
        ax.text(xs[0] - base_r - 0.25, ys[0], "5'",
                ha="right", va="center", fontsize=8, color=CLR["sc"], fontweight="bold")
        ax.text(xs[n - 1] + base_r + 0.25, ys[n - 1], "3'",
                ha="left", va="center", fontsize=8, color=CLR["s3"], fontweight="bold")

        # ----- Legend -----
        legend_elems = [
            mpatches.Patch(facecolor=CLR["sc"], label=f"5' stem_comp  {s} bp"),
            mpatches.Patch(facecolor=CLR["lp"], label=f"loop  {l} nt  ({design.loop_seq})"),
            mpatches.Patch(facecolor=CLR["pb"], label=f"引物主体 (ss)  {n_pb} nt"),
            mpatches.Patch(facecolor=CLR["s3"], label=f"3' stem  {s} bp"),
        ]
        ax.legend(handles=legend_elems, loc="lower center", fontsize=7,
                  ncol=2, framealpha=0.85, bbox_to_anchor=(0.5, 0.0))

        # ----- Title with risk indicators -----
        risk = getattr(design, "on_target_risk", "?")
        risk_clr = {"low": "#2E7D32", "medium": "#E65100", "high": "#B71C1C"}.get(risk, "black")
        homo = getattr(design, "dg_homodimer", None)
        homo_txt = f"  ΔG_dimer={homo:.1f}" if homo is not None else ""
        ax.set_title(
            f"发夹二级结构   stem={s}bp  loop={l}nt\n"
            f"ΔG_hp={design.dg_hairpin:.2f}  EI={design.ei:.2f}{homo_txt}  "
            f"on-target风险: {risk}",
            fontsize=8.5, color="black",
        )

        margin = 1.8
        ax.set_xlim(xs.min() - margin, xs.max() + margin)
        ax.set_ylim(ys.min() - margin - 1.5, ys.max() + margin)
        self.fig_struct.tight_layout()
        self.canvas_struct.draw_idle()

    # ------------------------------------------------------------------
    # 散点图
    # ------------------------------------------------------------------
    def _update_plot(self) -> None:
        self.ax.clear()
        self._highlight_artist = None

        si_all = [p.pair_si for p in self.pairs]
        ei_all = [p.pair_ei for p in self.pairs]
        self._scatter = self.ax.scatter(si_all, ei_all, c="gray", s=30, picker=5, label="All candidates")

        front_idx = [int(i) for i in self.front_ids]
        if front_idx:
            front_x = [self.pairs[i].pair_si for i in front_idx]
            front_y = [self.pairs[i].pair_ei for i in front_idx]
            self.ax.scatter(front_x, front_y, c="red", s=50, label="Pareto front", zorder=3)

        self.ax.axvline(1.5, color="green", linestyle="--", linewidth=0.8)
        self.ax.axvline(0.8, color="orange", linestyle="--", linewidth=0.8)
        self.ax.axhline(1.0, color="green", linestyle=":", linewidth=0.8)
        self.ax.axhline(0.5, color="orange", linestyle=":", linewidth=0.8)

        self.ax.set_xlabel("SI (Specificity Index)")
        self.ax.set_ylabel("EI (Efficiency Index)")
        self.ax.legend(loc="best", fontsize=8)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _highlight_plot_point(self, idx: int) -> None:
        if self._highlight_artist is not None:
            self._highlight_artist.remove()
            self._highlight_artist = None
        p = self.pairs[idx]
        self._highlight_artist = self.ax.scatter(
            [p.pair_si], [p.pair_ei], facecolors="none", edgecolors="blue", s=140, linewidths=2, zorder=4
        )
        self.canvas.draw_idle()

    def _on_pick(self, event) -> None:
        if event.artist is not self._scatter or not len(event.ind):
            return
        idx = int(event.ind[0])
        self.tree.selection_set(str(idx))
        self.tree.see(str(idx))

    # ------------------------------------------------------------------
    # 导出CSV
    # ------------------------------------------------------------------
    def _on_export(self) -> None:
        if not self.pairs:
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return

        fieldnames = [
            "rank", "pareto", "product_size", "pair_si", "pair_ei",
            "left_primer", "left_hairpin_primer", "left_stem_len", "left_loop_len", "left_loop_seq",
            "left_dg_target", "left_dg_hairpin", "left_dg_offtarget",
            "left_dg_homodimer", "left_on_target_risk",
            "left_si", "left_ei", "left_verdict",
            "right_primer", "right_hairpin_primer", "right_stem_len", "right_loop_len", "right_loop_seq",
            "right_dg_target", "right_dg_hairpin", "right_dg_offtarget",
            "right_dg_homodimer", "right_on_target_risk",
            "right_si", "right_ei", "right_verdict",
        ]
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for i, p in enumerate(self.pairs):
                    c, l, r = p.candidate, p.left, p.right
                    writer.writerow({
                        "rank": i + 1,
                        "pareto": "yes" if str(i) in self.front_ids else "no",
                        "product_size": c.product_size,
                        "pair_si": p.pair_si,
                        "pair_ei": p.pair_ei,
                        "left_primer": c.left_seq,
                        "left_hairpin_primer": l.hairpin_primer_seq,
                        "left_stem_len": l.stem_len,
                        "left_loop_len": l.loop_len,
                        "left_loop_seq": l.loop_seq,
                        "left_dg_target": l.dg_target,
                        "left_dg_hairpin": l.dg_hairpin,
                        "left_dg_offtarget": l.dg_offtarget,
                        "left_dg_homodimer": getattr(l, "dg_homodimer", ""),
                        "left_on_target_risk": getattr(l, "on_target_risk", ""),
                        "left_si": l.si,
                        "left_ei": l.ei,
                        "left_verdict": l.overall_verdict,
                        "right_primer": c.right_seq,
                        "right_hairpin_primer": r.hairpin_primer_seq,
                        "right_stem_len": r.stem_len,
                        "right_loop_len": r.loop_len,
                        "right_loop_seq": r.loop_seq,
                        "right_dg_target": r.dg_target,
                        "right_dg_hairpin": r.dg_hairpin,
                        "right_dg_offtarget": r.dg_offtarget,
                        "right_dg_homodimer": getattr(r, "dg_homodimer", ""),
                        "right_on_target_risk": getattr(r, "on_target_risk", ""),
                        "right_si": r.si,
                        "right_ei": r.ei,
                        "right_verdict": r.overall_verdict,
                    })
        except OSError as exc:
            messagebox.showerror("导出失败", str(exc))
            return

        self.lbl_status.config(text=f"已导出: {path}")

    # ------------------------------------------------------------------
    # 多重PCR 控制与可视化
    # ------------------------------------------------------------------
    def _on_mplex_toggle(self) -> None:
        enabled = self.var_mplex_mode.get()
        hint = "多重模式: 目标序列框中粘贴多条FASTA，每条对应一个扩增子靶" if enabled else "提示: 禁用时仅对当前候选池做单模板设计"
        self.lbl_mplex_hint.config(text=hint)

    def _on_run(self) -> None:
        if self.var_mplex_mode.get():
            self._on_run_multiplex()
        else:
            self._on_run_single()

    def _on_run_single(self) -> None:
        """原单模板设计流程。"""
        try:
            records = parse_fasta_text(self.txt_target.get("1.0", tk.END))
        except ValueError as exc:
            messagebox.showerror("目标序列错误", str(exc))
            return
        if not records:
            messagebox.showerror("目标序列错误", "请输入目标序列")
            return
        template = records[0][1]

        target_region = None
        start_s = self.var_region_start.get().strip()
        len_s = self.var_region_len.get().strip()
        if start_s or len_s:
            try:
                target_region = (int(start_s), int(len_s))
            except ValueError:
                messagebox.showerror("目标区域错误", "起始位置和长度必须为整数")
                return

        try:
            offtarget_records = parse_fasta_text(self.txt_offtarget.get("1.0", tk.END))
        except ValueError as exc:
            messagebox.showerror("非目标模板库错误", str(exc))
            return

        try:
            params = self._collect_params()
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc))
            return

        use_ai_search = self.var_use_ai_search.get()
        self._set_running(True, "运行中..." if not use_ai_search else "运行中 (训练AI代理模型)...")

        thread = threading.Thread(
            target=self._worker_single,
            args=(template, target_region, offtarget_records, params, use_ai_search),
            daemon=True,
        )
        thread.start()
        self.after(100, self._poll_queue)

    def _on_run_multiplex(self) -> None:
        """多重PCR模式: 多条FASTA → MPIGN优化。"""
        try:
            records = parse_fasta_text(self.txt_target.get("1.0", tk.END))
        except ValueError as exc:
            messagebox.showerror("目标序列错误", str(exc))
            return
        if not records:
            messagebox.showerror("目标序列错误", "请在目标序列框中粘贴多条FASTA")
            return
        if len(records) < 2:
            messagebox.showwarning("多重PCR", "仅检测到1条序列，建议输入至少2条；继续以单模板候选池做多重优化")

        try:
            offtarget_records = parse_fasta_text(self.txt_offtarget.get("1.0", tk.END))
        except ValueError as exc:
            messagebox.showerror("非目标模板库错误", str(exc))
            return

        try:
            params = self._collect_params()
            n_per = int(self.var_mplex_n_per.get())
            n_sel_s = self.var_mplex_n_sel.get().strip()
            n_sel = int(n_sel_s) if n_sel_s else None
            xt = float(self.var_mplex_xt.get())
        except ValueError as exc:
            messagebox.showerror("参数格式错误", str(exc))
            return

        from primer_designer.multiplex_gnn import PhysicsContext

        ctx = PhysicsContext(
            temp_c=float(self.param_vars.get("PRIMER_OPT_TM", tk.StringVar(value="60")).get()),
        )
        use_ai = self.var_use_ai_search.get()
        self._set_running(True, f"多重PCR优化 ({len(records)} 条靶序列)...")

        thread = threading.Thread(
            target=self._worker_multiplex,
            args=(records, offtarget_records, params, n_per, n_sel, xt, ctx, use_ai),
            daemon=True,
        )
        thread.start()
        self.after(100, self._poll_queue)

    def _set_running(self, running: bool, status: str = "") -> None:
        if running:
            self.btn_run.config(state=tk.DISABLED)
            self.btn_export.config(state=tk.DISABLED)
            self.progress.start(10)
            self.lbl_status.config(text=status)
        else:
            self.progress.stop()
            self.btn_run.config(state=tk.NORMAL)

    def _worker_single(self, template, target_region, offtarget_records, params, use_ai) -> None:
        try:
            pairs, front = run_design(template, target_region, offtarget_records, params, use_ai_search=use_ai)
            self._queue.put(("ok_single", pairs, front))
        except Exception as exc:
            self._queue.put(("error", exc, None))

    def _worker_multiplex(self, records, offtarget_records, params, n_per, n_sel, xt, ctx, use_ai) -> None:
        try:
            from primer_designer.pipeline import run_multiplex_design

            result = run_multiplex_design(
                templates=records,
                offtarget_records=offtarget_records,
                params=params,
                n_pairs_per_target=n_per,
                n_select=n_sel,
                cross_talk_threshold=xt,
                physics_ctx=ctx,
                use_ai_search=use_ai,
                use_gnn_scoring=True,
            )
            self._queue.put(("ok_multiplex", result, None))
        except Exception as exc:
            self._queue.put(("error", exc, None))

    def _poll_queue(self) -> None:
        try:
            kind, a, b = self._queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_queue)
            return

        self._set_running(False)

        if kind == "error":
            self.lbl_status.config(text=f"出错: {a}")
            messagebox.showerror("设计出错", str(a))
            return

        if kind == "ok_single":
            pairs, front = a, b
            front_id_set = {id(p) for p in front}
            self.pairs = pairs
            self._mplex_result = None
            self.front_ids = {str(i) for i, p in enumerate(pairs) if id(p) in front_id_set}
            self._populate_results()
            self.lbl_status.config(text=f"完成: 共{len(pairs)}个候选, Pareto前沿{len(front)}个")
            self.btn_export.config(state=tk.NORMAL if pairs else tk.DISABLED)

        elif kind == "ok_multiplex":
            result = a
            self._mplex_result = result
            self.pairs = result.all_pairs
            self.front_ids = {str(i) for i, p in enumerate(self.pairs) if p in result.selected_pairs}
            self._populate_results()
            self._draw_mplex_result(result)
            n_warn = len(result.cross_talk_warnings)
            n_sel = len(result.selected_pairs)
            self.lbl_status.config(
                text=f"多重PCR完成: 候选{len(self.pairs)}对 → 优选{n_sel}对, {n_warn}处串扰警告"
            )
            self.btn_export.config(state=tk.NORMAL if self.pairs else tk.DISABLED)
            # 自动切换到多重PCR串扰图标签页
            self._nb.select(3)

    # ------------------------------------------------------------------
    # 多重PCR 可视化
    # ------------------------------------------------------------------
    def _draw_mplex_placeholder(self) -> None:
        self.fig_mplex.clear()
        ax = self.fig_mplex.add_subplot(111)
        ax.set_axis_off()
        ax.text(0.5, 0.5, "启用多重PCR模式并运行后显示串扰热图",
                ha="center", va="center", transform=ax.transAxes, color="gray", fontsize=10)
        self.fig_mplex.tight_layout()
        self.canvas_mplex.draw_idle()

    def _draw_mplex_result(self, result) -> None:
        """绘制双子图: 左=串扰热图, 右=引物交互网络。"""
        import matplotlib.colors as mcolors
        import matplotlib.patches as mpatches

        self.fig_mplex.clear()
        labels = result.seq_labels
        mat = result.dg_matrix
        scores = result.node_scores
        n = len(labels)
        selected_seqs = {p.left.hairpin_primer_seq for p in result.selected_pairs} | \
                        {p.right.hairpin_primer_seq for p in result.selected_pairs}
        selected_mask = np.array([s in selected_seqs for s in result.all_seqs])
        thr = result.cross_talk_threshold

        # ----- 子图1: 串扰热图 -----
        ax1 = self.fig_mplex.add_subplot(121)
        vmax = 0.0
        vmin = min(thr * 2.5, mat.min())
        cmap = "RdYlGn"
        im = ax1.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        self.fig_mplex.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="ΔG (kcal/mol)")

        tick_labels = [lbl.replace("Pair", "P").replace("-", "\n") for lbl in labels]
        ax1.set_xticks(range(n))
        ax1.set_yticks(range(n))
        ax1.set_xticklabels(tick_labels, fontsize=max(4, 7 - n // 4), rotation=45, ha="right")
        ax1.set_yticklabels(tick_labels, fontsize=max(4, 7 - n // 4))

        # 标记被选中的引物行/列
        for i, sel in enumerate(selected_mask):
            if sel:
                ax1.axhline(i, color="gold", lw=2.5, alpha=0.7)
                ax1.axvline(i, color="gold", lw=2.5, alpha=0.7)

        # 危险格子加星号
        for i in range(n):
            for j in range(n):
                if i != j and mat[i, j] <= thr:
                    ax1.text(j, i, "✕", ha="center", va="center",
                             fontsize=7, color="black", fontweight="bold")

        ax1.set_title(f"引物串扰热图 (ΔG, kcal/mol)\n阈值={thr:.1f}  金色=已选", fontsize=8)

        # ----- 子图2: 引物交互网络 -----
        ax2 = self.fig_mplex.add_subplot(122)
        ax2.set_aspect("equal")
        ax2.set_axis_off()

        # 节点布局: 圆形排列
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        node_x = np.cos(angles)
        node_y = np.sin(angles)

        # 绘制危险边 (ΔG ≤ 阈值)
        for i in range(n):
            for j in range(i + 1, n):
                if mat[i, j] <= thr:
                    danger_level = min(1.0, (thr - mat[i, j]) / max(abs(thr), 1.0))
                    edge_color = mcolors.to_rgba("red", alpha=0.3 + 0.5 * danger_level)
                    lw = 0.8 + 2.0 * danger_level
                    ax2.plot([node_x[i], node_x[j]], [node_y[i], node_y[j]],
                             color=edge_color, lw=lw, zorder=1)

        # 绘制节点 (分左/右引物，已选/未选)
        node_r = max(0.06, 0.4 / max(n, 1))
        for i in range(n):
            is_sel = selected_mask[i]
            is_left = (i % 2 == 0)
            base_color = "#1565C0" if is_left else "#C62828"
            edge_c = "gold" if is_sel else "white"
            edge_lw = 3 if is_sel else 1
            circ = mpatches.Circle(
                (node_x[i], node_y[i]), node_r,
                color=base_color, ec=edge_c, linewidth=edge_lw, zorder=3,
            )
            ax2.add_patch(circ)
            score_norm = max(0.0, min(1.0, (scores[i] + 5) / 10.0))
            ax2.text(
                node_x[i] * 1.22, node_y[i] * 1.22,
                tick_labels[i], ha="center", va="center",
                fontsize=max(4, 6 - n // 6),
                color="gold" if is_sel else "black",
            )

        legend_elems = [
            mpatches.Patch(facecolor="#1565C0", label="正向(L)引物"),
            mpatches.Patch(facecolor="#C62828", label="反向(R)引物"),
            mpatches.Patch(facecolor="gray", ec="gold", linewidth=2, label="已选优化子集"),
        ]
        ax2.legend(handles=legend_elems, loc="lower center", fontsize=6,
                   ncol=3, bbox_to_anchor=(0.5, -0.05))
        ax2.set_title(
            f"引物相互作用网络  红线=危险串扰(ΔG≤{thr:.1f})\n"
            f"金边=MPIGN优选 ({len(result.selected_pairs)}对)",
            fontsize=8,
        )
        ax2.set_xlim(-1.6, 1.6)
        ax2.set_ylim(-1.6, 1.6)

        self.fig_mplex.tight_layout()
        self.canvas_mplex.draw_idle()

    # ------------------------------------------------------------------
    # 发夹 3D B-form 双螺旋可视化
    # ------------------------------------------------------------------
    def _draw_3d_placeholder(self) -> None:
        self.fig_3d.clear()
        ax = self.fig_3d.add_subplot(111, projection="3d")
        ax.set_axis_off()
        ax.text2D(0.5, 0.5, "选中一条候选引物后显示3D双螺旋",
                  ha="center", va="center", transform=ax.transAxes,
                  color="gray", fontsize=10)
        self.fig_3d.tight_layout()
        self.canvas_3d.draw_idle()

    def _draw_hairpin_3d(self, design) -> None:
        """发夹阻断引物的 3D B-form 双螺旋渲染。

        茎区按照真实 B-form DNA 几何参数构造：
          · 螺旋半径 R = 0.9 Å (归一化单位)
          · 每碱基对上升 rise = 0.38 (对应 3.4–3.8 Å，略放大以增强可视性)
          · 每碱基对扭转 twist = 36° (B-form 标准值，10 bp / 圈)
          · 两条链相差 180° (反向平行)

        环区和引物主体以向外展开的 3D 螺旋弧线渲染：
          · 弧线同时沿角向 (φ) 和径向 (r) 及高度 (z) 变化，形成自然的单链形态
          · 颜色：5'茎(蓝) / 环(橙) / 引物主体(灰) / 3'茎(红)

        支持鼠标左键拖拽旋转、右键拖拽缩放。
        """
        import numpy as _np
        import matplotlib.patches as mpatches

        self.fig_3d.clear()
        ax = self.fig_3d.add_subplot(111, projection="3d")
        ax.set_axis_off()

        # 使用 dna_seq（纯核苷酸，不含[HEG]占位符）计算布局
        seq  = getattr(design, "dna_seq", design.hairpin_primer_seq.replace("[HEG]", ""))
        n    = len(seq)
        s    = design.stem_len
        lp   = design.loop_len
        n_pb = n - 2 * s - lp      # primer body 单链长度
        n_ss = lp + n_pb            # 总单链数目（环 + 引物主体）

        # ── B-form 参数 ──────────────────────────────────────────────
        R     = 0.90                          # 螺旋半径
        rise  = 0.38                          # 每 bp 上升高度（放大以增强可视性）
        omega = _np.radians(36.0)             # 每 bp 扭转角 (B-form)
        phi0  = 0.0                           # 5'端起始相位

        # ── 颜色 ─────────────────────────────────────────────────────
        C = {"sc": "#1565C0", "lp": "#E65100", "pb": "#607D8B", "s3": "#C62828"}

        # ── 茎区：两条反向平行链 ─────────────────────────────────────
        # strand1 (stem_comp, 蓝, 5'→3' 向上)
        # strand2 (primer 3'茎, 红, 3'→5' 向上即 antiparallel)
        s1 = _np.array([
            [R * _np.cos(phi0 + omega * i),
             R * _np.sin(phi0 + omega * i),
             i * rise]
            for i in range(s)
        ])
        s2 = _np.array([
            [R * _np.cos(phi0 + omega * i + _np.pi),
             R * _np.sin(phi0 + omega * i + _np.pi),
             i * rise]
            for i in range(s)
        ])

        # 骨架连线
        ax.plot(s1[:, 0], s1[:, 1], s1[:, 2], color=C["sc"], lw=2.5, zorder=5)
        ax.plot(s2[:, 0], s2[:, 1], s2[:, 2], color=C["s3"], lw=2.5, zorder=5)

        # 碱基对横档 (rungs)
        for i in range(s):
            ax.plot([s1[i, 0], s2[i, 0]],
                    [s1[i, 1], s2[i, 1]],
                    [s1[i, 2], s2[i, 2]],
                    color="#9E9E9E", lw=1.0, alpha=0.75)

        # 核苷酸球体 + 碱基字母
        sph_sz = max(60, 160 - n * 3)
        fs = max(5, min(8, 120 // max(n, 1)))
        for i in range(s):
            ax.scatter(*s1[i], color=C["sc"], s=sph_sz, depthshade=True, zorder=6)
            ax.scatter(*s2[i], color=C["s3"], s=sph_sz, depthshade=True, zorder=6)
            # 茎_comp 字母标注在左侧
            ax.text(s1[i, 0] - 0.18, s1[i, 1], s1[i, 2],
                    seq[i], color="white", fontsize=fs,
                    ha="center", va="center", fontweight="bold", zorder=7)
            # primer 3'茎字母标注（取倒序位置）
            j = n - 1 - i
            ax.text(s2[i, 0] + 0.18, s2[i, 1], s2[i, 2],
                    seq[j], color="white", fontsize=fs,
                    ha="center", va="center", fontweight="bold", zorder=7)

        # ── 单链弧区 ─────────────────────────────────────────────────
        # 弧线起点：s1[-1]，终点：s2[-1]
        # 同时扩大半径(向外展开) + 上升高度，形成自然单链形态
        z_top  = s1[-1, 2]
        phi_t  = phi0 + omega * (s - 1)        # 最内层碱基对的相位
        arc_r  = max(0.50, n_ss * 0.12)        # 弧向外伸出的额外半径
        arc_h  = max(1.00, n_ss * 0.22)        # 弧顶高于茎顶的高度

        arc_pts = _np.array([
            (
                (R + arc_r * _np.sin(_np.pi * t)) * _np.cos(phi_t + _np.pi * t),
                (R + arc_r * _np.sin(_np.pi * t)) * _np.sin(phi_t + _np.pi * t),
                z_top + arc_h * _np.sin(_np.pi * t),
            )
            for t in _np.linspace(1 / (n_ss + 1), n_ss / (n_ss + 1), n_ss)
        ]) if n_ss > 0 else _np.zeros((0, 3))

        def _draw_arc_segment(pts_list, color):
            if len(pts_list) < 2:
                return
            arr = _np.array(pts_list)
            ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], color=color, lw=2.0, zorder=5)
            for row in arr[1:-1]:   # 两端端点已在茎里画过
                ax.scatter(*row, color=color, s=sph_sz * 0.8, depthshade=True, zorder=6)

        if n_ss > 0:
            # 环段 (loop)
            loop_pts = [s1[-1]] + list(arc_pts[:lp])
            if len(loop_pts) > 1:
                _draw_arc_segment(loop_pts, C["lp"])
                for k, pt in enumerate(arc_pts[:lp]):
                    ax.text(pt[0], pt[1], pt[2], seq[s + k],
                            color="white", fontsize=fs,
                            ha="center", va="center", fontweight="bold", zorder=7)

            # 引物主体段 (primer body)
            if n_pb > 0:
                seg_start = arc_pts[lp - 1] if lp > 0 else s1[-1]
                pb_pts = [seg_start] + list(arc_pts[lp:]) + [s2[-1]]
                if len(pb_pts) > 1:
                    _draw_arc_segment(pb_pts, C["pb"])
                    for k, pt in enumerate(arc_pts[lp:]):
                        ax.text(pt[0], pt[1], pt[2], seq[s + lp + k],
                                color="white", fontsize=fs,
                                ha="center", va="center", fontweight="bold", zorder=7)

        # ── 5' / 3' 端标签 ───────────────────────────────────────────
        ax.text(s1[0, 0], s1[0, 1], s1[0, 2] - rise * 0.9,
                "5'", color=C["sc"], fontsize=9, fontweight="bold", ha="center")
        ax.text(s2[0, 0], s2[0, 1], s2[0, 2] - rise * 0.9,
                "3'", color=C["s3"], fontsize=9, fontweight="bold", ha="center")

        # ── 图例 ─────────────────────────────────────────────────────
        legend_elems = [
            mpatches.Patch(facecolor=C["sc"], label=f"5' stem_comp  {s}bp"),
            mpatches.Patch(facecolor=C["lp"], label=f"loop  {lp}nt  ({design.loop_seq})"),
            mpatches.Patch(facecolor=C["pb"], label=f"引物主体(ss)  {n_pb}nt"),
            mpatches.Patch(facecolor=C["s3"], label=f"3' stem  {s}bp"),
        ]
        ax.legend(handles=legend_elems, loc="lower left", fontsize=6.5,
                  ncol=2, framealpha=0.85)

        # ── 标题与视角 ────────────────────────────────────────────────
        risk     = getattr(design, "on_target_risk", "?")
        risk_clr = {"low": "#2E7D32", "medium": "#E65100", "high": "#B71C1C"}.get(risk, "black")
        ax.set_title(
            f"3D B-form 双螺旋  stem={s}bp  loop={lp}nt  "
            f"twist=36°/bp  rise=3.4Å/bp\n"
            f"ΔG_hp={design.dg_hairpin:.2f} kcal/mol  "
            f"on-target风险: {risk}",
            fontsize=8, color="black",
        )

        # 初始视角：斜45°俯视，能同时看到螺旋扭转和环的形态
        ax.view_init(elev=22, azim=-55)

        # 自动对齐坐标范围使图形居中
        all_x = list(s1[:, 0]) + list(s2[:, 0]) + (list(arc_pts[:, 0]) if len(arc_pts) else [])
        all_y = list(s1[:, 1]) + list(s2[:, 1]) + (list(arc_pts[:, 1]) if len(arc_pts) else [])
        all_z = list(s1[:, 2]) + list(s2[:, 2]) + (list(arc_pts[:, 2]) if len(arc_pts) else [])
        pad = 0.6
        ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
        ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
        ax.set_zlim(min(all_z) - pad * 2, max(all_z) + pad * 2)

        self.fig_3d.tight_layout()
        self.canvas_3d.draw_idle()

    # ------------------------------------------------------------------
    # 扩增子顾问弹窗 (Amplicon Advisor popup)
    # ------------------------------------------------------------------
    def _open_amplicon_advisor(self) -> None:
        """打开扩增子顾问弹窗：滑动窗口扫描目标序列，推荐最优扩增区域。"""
        try:
            records = parse_fasta_text(self.txt_target.get("1.0", tk.END))
        except ValueError as exc:
            messagebox.showerror("目标序列错误", str(exc))
            return
        if not records:
            messagebox.showerror("目标序列错误", "请先在目标序列框中输入序列")
            return
        template = records[0][1]
        if len(template) < 60:
            messagebox.showerror("序列太短", "目标序列长度不足60 bp，无法扫描扩增子")
            return

        try:
            offtarget_records = parse_fasta_text(self.txt_offtarget.get("1.0", tk.END))
        except ValueError:
            offtarget_records = []

        try:
            product_min = int(self.param_vars["PRODUCT_MIN"].get())
            product_max = int(self.param_vars["PRODUCT_MAX"].get())
        except (ValueError, KeyError):
            product_min, product_max = 100, 300

        AdvisorWindow(
            parent=self,
            template=template,
            offtarget_records=offtarget_records,
            product_min=product_min,
            product_max=product_max,
            on_select=self._apply_amplicon_region,
        )

    def _apply_amplicon_region(self, start: int, length: int) -> None:
        """将顾问选中的扩增子区域填入目标区域输入框。"""
        self.var_region_start.set(str(start))
        self.var_region_len.set(str(length))
        self.lbl_status.config(text=f"已选定扩增子区域: 起始={start}, 长度={length} bp")

    # ------------------------------------------------------------------
    # GPU 加速 3D 分子结构查看器
    # ------------------------------------------------------------------
    def _on_open_3d_viewer(self) -> None:
        """生成 PDB 原子坐标并在 GPU WebGL 查看器中展示 (py3Dmol + Edge/Chrome)。

        查看器特性:
          · 硬件 GPU 渲染 (WebGL via Edge WebView2 / Chrome)，等效 OpenGL 加速
          · 鼠标左键旋转 / 滚轮缩放 / 右键平移
          · 棒状/球状/卡通/混合等多种表示法可切换
          · 半透明分子表面 (VDW) 渲染
          · 一键导出高分辨率 PNG
          · PDB 格式兼容 PyMOL / UCSF Chimera (可独立打开)
        """
        if self._current_pair is None:
            messagebox.showwarning("提示", "请先选择一条候选引物")
            return

        from primer_designer.structure_pdb import hairpin_to_pdb
        from gui.viewer_3d import launch_viewer

        p = self._current_pair
        # 展示正向引物发夹结构 (left design)
        design = p.left
        try:
            pdb_str = hairpin_to_pdb(design)
        except Exception as exc:
            messagebox.showerror("PDB生成失败", str(exc))
            return

        seq = design.hairpin_primer_seq  # 含[HEG]的展示字符串，仅用于标题
        dna_total = len(getattr(design, "dna_seq", seq.replace("[HEG]", "")))
        self.lbl_status.config(text=f"正在启动3D查看器 — {seq[:16]}…")
        launch_viewer(
            pdb_string=pdb_str,
            title=f"发夹阻断引物 3D — {seq[:14]}… | stem={design.stem_len}bp loop={design.loop_len}nt",
            stem_len=design.stem_len,
            loop_len=design.loop_len,
            total_len=dna_total,
            use_webview=True,
        )
        self.lbl_status.config(
            text=f"3D查看器已启动 (WebGL GPU渲染)  "
                 f"— 若未弹窗请安装 pywebview: pip install pywebview"
        )

    def _on_export_pdb(self) -> None:
        """将当前选中引物的发夹结构导出为 PDB 文件。"""
        if self._current_pair is None:
            messagebox.showwarning("提示", "请先选择一条候选引物")
            return

        from primer_designer.structure_pdb import save_pdb

        design = self._current_pair.left
        seq_short = design.hairpin_primer_seq[:10]
        path = filedialog.asksaveasfilename(
            defaultextension=".pdb",
            initialfile=f"hairpin_{seq_short}.pdb",
            filetypes=[("PDB 结构文件", "*.pdb"), ("All files", "*.*")],
            title="导出 PDB 分子结构文件",
        )
        if not path:
            return
        try:
            save_pdb(design, path)
        except OSError as exc:
            messagebox.showerror("导出失败", str(exc))
            return
        self.lbl_status.config(text=f"PDB已导出: {path}  (可直接用PyMOL/UCSF Chimera打开)")


# ===========================================================================
# 扩增子顾问独立弹窗
# ===========================================================================

class AdvisorWindow(tk.Toplevel):
    """扩增子顾问弹窗：热图 + 可排序候选列表，点击候选一键填回主窗口。"""

    _COL_DEFS = [
        ("rank",  "排名",     40),
        ("start", "起始位置",  70),
        ("end",   "终止位置",  70),
        ("size",  "大小(bp)",  65),
        ("gc",    "GC%",       55),
        ("gc_s",  "GC均匀",    65),
        ("st_s",  "结构简单",  65),
        ("sp_s",  "特异性",    60),
        ("cx_s",  "低复杂度",  65),
        ("comp",  "综合得分",  70),
    ]

    _ATTR_MAP = {
        "gc_s": "gc_score", "st_s": "struct_score",
        "sp_s": "spec_score", "cx_s": "complexity_score",
        "comp": "composite", "gc": "gc_pct",
        "start": "start", "end": "end", "size": "size", "rank": "composite",
    }

    def __init__(self, parent, template, offtarget_records, product_min, product_max, on_select):
        super().__init__(parent)
        self.title("扩增子顾问 — 滑动窗口评分")
        self.geometry("900x640")
        self.resizable(True, True)
        self.grab_set()

        self._template = template
        self._offtarget = offtarget_records
        self._pmin = product_min
        self._pmax = product_max
        self._on_select = on_select
        self._candidates: list = []
        self._pos_scores: list[float] = []
        self._sort_col = "comp"
        self._sort_desc = True
        self._selected_candidate = None
        self._heat_cursor: list = []
        import queue as _q
        self._scan_queue = _q.Queue()

        self._build_ui()
        self._start_scan()

    def _build_ui(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=(8, 0))
        self.lbl_status = ttk.Label(top, text="正在扫描...")
        self.lbl_status.pack(side=tk.LEFT)
        self.progress = ttk.Progressbar(top, mode="determinate", length=200)
        self.progress.pack(side=tk.LEFT, padx=(12, 0))

        frm_heat = ttk.LabelFrame(self, text="序列适合度热图  (绿=高分/推荐  红=低分/避免  点击定位)")
        frm_heat.pack(fill=tk.X, padx=8, pady=(6, 4))
        self.fig_heat = Figure(figsize=(8.5, 1.3), dpi=96)
        self.ax_heat = self.fig_heat.add_subplot(111)
        self.canvas_heat = FigureCanvasTkAgg(self.fig_heat, master=frm_heat)
        self.canvas_heat.get_tk_widget().pack(fill=tk.X, padx=4, pady=4)
        self.canvas_heat.mpl_connect("button_press_event", self._on_heat_click)

        frm_list = ttk.LabelFrame(
            self, text="候选扩增子  (单击行 → 热图定位高亮  双击/点按钮 → 填入主窗口)"
        )
        frm_list.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))
        cols = [c[0] for c in self._COL_DEFS]
        self.tree = ttk.Treeview(frm_list, columns=cols, show="headings", height=14)
        for cid, label, width in self._COL_DEFS:
            self.tree.heading(cid, text=label, command=lambda c=cid: self._sort_by(c))
            self.tree.column(cid, width=width, anchor=tk.CENTER)
        sb = ttk.Scrollbar(frm_list, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.tag_configure("good", background="#d6f5d6")
        self.tree.tag_configure("mid",  background="#fff7cc")
        self.tree.tag_configure("poor", background="#f8d7da")
        self.tree.tag_configure("sel",  background="#cce5ff")
        self.tree.bind("<<TreeviewSelect>>", self._on_list_select)
        self.tree.bind("<Double-1>", lambda e: self._apply_selection())

        bot = ttk.Frame(self)
        bot.pack(fill=tk.X, padx=8, pady=(0, 8))
        self.lbl_sel = ttk.Label(bot, text="未选中任何候选")
        self.lbl_sel.pack(side=tk.LEFT)
        ttk.Button(bot, text="使用此扩增子区域", command=self._apply_selection).pack(side=tk.RIGHT)
        ttk.Button(bot, text="关闭", command=self.destroy).pack(side=tk.RIGHT, padx=(0, 6))

    def _start_scan(self) -> None:
        import threading
        threading.Thread(target=self._scan_worker, daemon=True).start()
        self.after(100, self._poll_scan)

    def _scan_worker(self) -> None:
        from primer_designer.amplicon_advisor import scan_amplicons, per_position_scores

        def cb(done, total):
            self._scan_queue.put(("progress", done, total))

        try:
            cands = scan_amplicons(
                self._template,
                product_min=self._pmin,
                product_max=self._pmax,
                step=10,
                offtarget_records=self._offtarget,
                top_n=50,
                progress_callback=cb,
            )
            pos = per_position_scores(cands, len(self._template))
            self._scan_queue.put(("done", cands, pos))
        except Exception as exc:
            self._scan_queue.put(("error", exc, None))

    def _poll_scan(self) -> None:
        try:
            msg = self._scan_queue.get_nowait()
        except Exception:
            self.after(100, self._poll_scan)
            return

        kind = msg[0]
        if kind == "progress":
            _, done, total = msg
            self.progress["maximum"] = total
            self.progress["value"] = done
            self.lbl_status.config(text=f"扫描中... {done}/{total}")
            self.after(100, self._poll_scan)
        elif kind == "done":
            _, candidates, pos_scores = msg
            self._candidates = candidates
            self._pos_scores = pos_scores
            self.progress["value"] = self.progress["maximum"]
            self.lbl_status.config(text=f"扫描完成: {len(candidates)} 个候选 (展示前50)")
            self._draw_heatmap()
            self._populate_list()
        else:
            self.lbl_status.config(text=f"扫描出错: {msg[1]}")

    def _draw_heatmap(self, highlight_start: int = -1) -> None:
        if not self._pos_scores:
            return
        import matplotlib.cm as cm, matplotlib.colors as mcolors

        ax = self.ax_heat
        ax.clear()
        ax.set_axis_off()
        n = len(self._pos_scores)
        scores = np.array(self._pos_scores)
        cmap = cm.get_cmap("RdYlGn")
        ax.imshow(cmap(scores)[np.newaxis, :, :], aspect="auto",
                  extent=[0, n, 0, 1], interpolation="nearest")

        step = max(50, n // 10)
        for pos in range(0, n + 1, step):
            ax.text(pos, -0.15, str(pos), ha="center", va="top",
                    fontsize=7, color="gray", transform=ax.transData)

        if highlight_start >= 0:
            size = (self._pmin + self._pmax) // 2
            ax.axvspan(highlight_start, highlight_start + size,
                       ymin=0, ymax=1, color="blue", alpha=0.25, lw=0)
            ax.axvline(highlight_start, color="blue", lw=1.5)
            ax.axvline(highlight_start + size, color="blue", lw=1.5)

        ax.set_xlim(0, n)
        ax.set_ylim(-0.4, 1)
        ax.set_title(
            f"序列适合度热图  总长 {n} bp  产物 {self._pmin}–{self._pmax} bp",
            fontsize=9,
        )
        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
        sm.set_array([])
        cb = self.fig_heat.colorbar(sm, ax=ax, orientation="horizontal",
                                     fraction=0.04, pad=0.38, aspect=40)
        cb.ax.tick_params(labelsize=7)
        cb.set_label("适合度得分", fontsize=7)
        self.fig_heat.tight_layout()
        self.canvas_heat.draw_idle()

    def _on_heat_click(self, event) -> None:
        if event.xdata is None or not self._candidates:
            return
        clicked = int(event.xdata)
        best = min(self._candidates, key=lambda c: abs(c.start - clicked))
        iid = str(best.start)
        if iid in self.tree.get_children():
            self.tree.selection_set(iid)
            self.tree.see(iid)

    def _populate_list(self) -> None:
        self.tree.delete(*self.tree.get_children())
        attr = self._ATTR_MAP.get(self._sort_col, "composite")
        sorted_cands = sorted(
            self._candidates,
            key=lambda c: getattr(c, attr, 0),
            reverse=self._sort_desc,
        )
        for rank, c in enumerate(sorted_cands, 1):
            tag = "good" if c.composite >= 0.70 else "mid" if c.composite >= 0.50 else "poor"
            self.tree.insert("", tk.END, iid=str(c.start), tags=(tag,), values=(
                rank, c.start, c.end, c.size,
                f"{c.gc_pct*100:.1f}%",
                f"{c.gc_score:.2f}", f"{c.struct_score:.2f}",
                f"{c.spec_score:.2f}", f"{c.complexity_score:.2f}",
                f"{c.composite:.3f}",
            ))

    def _sort_by(self, col: str) -> None:
        self._sort_desc = not self._sort_desc if self._sort_col == col else True
        self._sort_col = col
        self._populate_list()

    def _on_list_select(self, event=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        start = int(sel[0])
        cand = next((c for c in self._candidates if c.start == start), None)
        if cand is None:
            return
        self._selected_candidate = cand
        self.lbl_sel.config(
            text=f"选中: 起始={cand.start}  大小={cand.size} bp  "
                 f"GC={cand.gc_pct*100:.1f}%  "
                 f"GC均匀={cand.gc_score:.2f}  结构={cand.struct_score:.2f}  "
                 f"综合={cand.composite:.3f}"
        )
        # 重绘热图并高亮选中位置
        self._draw_heatmap(highlight_start=cand.start)
        # 列表行标色
        for item in self.tree.get_children():
            tags = [t for t in self.tree.item(item, "tags") if t != "sel"]
            self.tree.item(item, tags=tags)
        cur = list(self.tree.item(sel[0], "tags"))
        cur.append("sel")
        self.tree.item(sel[0], tags=cur)

    def _apply_selection(self) -> None:
        if self._selected_candidate is None:
            messagebox.showinfo("提示", "请先在列表中单击选中一个候选扩增子")
            return
        c = self._selected_candidate
        self._on_select(c.start, c.size)
        self.destroy()
