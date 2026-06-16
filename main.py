"""Hairpin-Blocker Designer 启动入口"""
from __future__ import annotations

import tkinter as tk

from gui.main_window import MainWindow


def main() -> None:
    root = tk.Tk()
    MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()


