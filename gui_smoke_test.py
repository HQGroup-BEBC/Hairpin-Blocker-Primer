"""GUI逻辑自测脚本（非可视化）：实例化主窗口，模拟"加载示例->开始设计->
选中行->导出CSV"完整流程，验证各回调函数与matplotlib嵌入是否正常工作。"""
from __future__ import annotations

import os
import tempfile
import time
import tkinter as tk

os.environ.setdefault("DISPLAY", ":0")

import gui.main_window as mw  # noqa: E402


def main() -> None:
    root = tk.Tk()
    win = mw.MainWindow(root)
    root.update()

    win._load_sample()
    root.update()
    print("target len label:", win.lbl_target_len.cget("text"))
    print("target text starts with:", win.txt_target.get("1.0", "1.50"))
    print("offtarget text starts with:", win.txt_offtarget.get("1.0", "1.50"))

    win._on_run()
    for _ in range(200):
        root.update()
        if not win._queue.empty():
            break
        time.sleep(0.02)
    win._poll_queue()
    root.update()

    print("status:", win.lbl_status.cget("text"))
    print("num pairs:", len(win.pairs))
    print("tree children:", len(win.tree.get_children()))
    print("front_ids:", win.front_ids)

    win.tree.selection_set("0")
    win._on_select_row()
    root.update()
    detail_text = win.txt_detail.get("1.0", tk.END)
    print("--- detail (first 600 chars) ---")
    print(detail_text[:600])

    # 测试导出CSV（绕过文件选择对话框）
    tmp_csv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
    orig_dialog = mw.filedialog.asksaveasfilename
    mw.filedialog.asksaveasfilename = lambda **kw: tmp_csv
    win._on_export()
    mw.filedialog.asksaveasfilename = orig_dialog
    size = os.path.getsize(tmp_csv)
    print(f"csv exported: {tmp_csv} size={size}")
    with open(tmp_csv, encoding="utf-8-sig") as f:
        lines = f.readlines()
    print(f"csv lines: {len(lines)}")
    print("csv header:", lines[0].strip())
    print("csv row1  :", lines[1].strip()[:200])

    # 测试选中另一行（用于触发散点高亮逻辑）
    if len(win.tree.get_children()) > 1:
        win.tree.selection_set("1")
        win._on_select_row()
        root.update()
        print("selected row 1 OK, highlight artist:", win._highlight_artist is not None)

    root.destroy()
    print("DONE")


if __name__ == "__main__":
    main()
