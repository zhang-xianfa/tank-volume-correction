# -*- coding: utf-8 -*-
"""
程序入口
========

用法：::

    python src/main.py

默认读取 ``data/20250819新增八辆车装卸数据.xlsx``，
结果输出到 ``output/结果.xlsx``。

也可作为模块调用：::

    from src.volume_pipeline import run_pipeline
    run_pipeline("数据.xlsx", "结果.xlsx")

Excel 数据格式约定见 ``src/data_loader.py`` 模块说明。
"""

from __future__ import annotations

from pathlib import Path

from .volume_pipeline import run_pipeline

# 项目根目录（src/ 的上一级）
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "20250819新增八辆车装卸数据.xlsx"
RESULT_FILE = BASE_DIR / "output" / "结果.xlsx"


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"找不到数据文件：{DATA_FILE}\n"
            "请把符合格式要求的装卸数据 Excel 放入 data/ 目录。"
        )

    print(f"读取数据：{DATA_FILE}")
    print(f"结果输出：{RESULT_FILE}")
    run_pipeline(str(DATA_FILE), str(RESULT_FILE))


if __name__ == "__main__":
    main()
