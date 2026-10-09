# -*- coding: utf-8 -*-
"""
数据加载模块
============

负责解析油罐车装卸数据 Excel，产出两部分数据：

1. **油罐静态参数** ``tank_params``：每辆车每个仓的几何参数
2. **装卸记录** ``all_raw_data``：每辆车一张表，包含装/卸时的油高与倾角

Excel 文件格式约定（必须与示例数据保持一致）：

- 一个名为 ``说明`` 的工作表，列为：
  ``序号、编号（车辆编号）、仓号、dc（探针距罐头）、L_total（罐长）、
  height（截面高）、width（截面宽）、-X（探针横向偏移）``
- 每个车辆一张工作表（表名 = 车辆编号），列为：
  ``时间、仓号、装车油高、车头车尾连线方向倾角x、倾角y、
  卸车油高、车头车尾连线方向倾角x.1、倾角y.1``
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

#: 「说明」工作表中读取的参数列
_DESC_COLUMNS = ["序号", "编号", "仓号", "dc", "L_total", "height", "width", "-X"]

#: 车辆数据工作表必须包含的列
_REQUIRED_RECORD_COLUMNS = [
    "时间", "仓号",
    "装车油高", "车头车尾连线方向倾角x", "倾角y",
    "卸车油高", "车头车尾连线方向倾角x.1", "倾角y.1",
]

#: 重命名后统一使用的列名（x 为纵向倾角，y 为横向倾角；1 装车、2 卸车）
_COLUMN_RENAME = {
    "车头车尾连线方向倾角x": "倾角x1",
    "倾角y": "倾角y1",
    "车头车尾连线方向倾角x.1": "倾角x2",
    "倾角y.1": "倾角y2",
}


class ExcelDataLoader:
    """油罐车装卸数据加载器。"""

    def __init__(self, excel_file: str) -> None:
        self.excel_file = excel_file

    def load(self) -> tuple[dict[str, dict[str, dict[str, float]]], list[dict[str, Any]]]:
        """读取整个 Excel 文件。

        Returns:
            (tank_params, all_raw_data)

            - ``tank_params``: ``{车辆编号: {仓号: {参数名: 值}}}``
              参数包括 dc / d_y(横向偏移) / width / height / L
            - ``all_raw_data``: 列表，每项为 ``{"车辆编号": str, "原始数据": DataFrame}``
        """
        tank_params = self._load_tank_params()
        all_raw_data = self._load_car_records()
        return tank_params, all_raw_data

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _load_tank_params(self) -> dict[str, dict[str, dict[str, float]]]:
        """解析「说明」工作表，得到每辆车每个仓的油罐几何参数。"""
        desc_df = pd.read_excel(self.excel_file, sheet_name="说明", usecols=_DESC_COLUMNS)

        tank_params: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
        for _, row in desc_df.iterrows():
            car_id, tank_id = row["编号"], row["仓号"]

            # 跳过车辆编号或仓号缺失的无效行
            if pd.isna(car_id) or pd.isna(tank_id):
                continue

            tank_params[car_id][tank_id] = {
                "d_c": row["dc"],          # 探针到罐头的距离
                "d_y": row["-X"],          # 探针横向偏移
                "width": row["width"],     # 截面宽度
                "height": row["height"],   # 截面高度
                "L": row["L_total"],       # 罐体长度
            }
        return tank_params

    def _load_car_records(self) -> list[dict[str, Any]]:
        """解析每辆车的工作表，得到清洗后的装卸记录。"""
        excel = pd.ExcelFile(self.excel_file)
        car_sheets = [s for s in excel.sheet_names if s != "说明"]

        all_raw_data: list[dict[str, Any]] = []
        for car_id in car_sheets:
            try:
                df = pd.read_excel(excel, sheet_name=car_id)
            except Exception as exc:  # 个别车辆表可能损坏，跳过并提示
                print(f"读取车辆 {car_id} 报错：{exc}，跳过该车")
                continue

            df_clean = self._clean_record_df(df)
            all_raw_data.append({"车辆编号": car_id, "原始数据": df_clean})
        return all_raw_data

    @staticmethod
    def _clean_record_df(df: pd.DataFrame) -> pd.DataFrame:
        """清洗单张车辆工作表。

        1. 删除装车油高 / 卸车油高为空的行（无油位读数无法计算体积）
        2. 其余缺失值填充 0（倾角缺失视为未倾斜）
        3. 统一列名为简短形式（倾角x1/x2/y1/y2）
        """
        existing = [col for col in _REQUIRED_RECORD_COLUMNS if col in df.columns]
        height_cols = [c for c in ("装车油高", "卸车油高") if c in existing]

        df_clean = df.dropna(subset=height_cols).copy() if height_cols else df.copy()
        df_clean[existing] = df_clean[existing].fillna(0)
        df_clean.rename(columns=_COLUMN_RENAME, inplace=True)
        return df_clean
