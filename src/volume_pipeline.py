# -*- coding: utf-8 -*-
"""
体积计算与校正流水线
====================

对每辆车执行两步处理：

1. **仓级校正系数优化**：装卸同一条油时，装车体积 V1 与卸车体积 V2 理论上相等，
   但因油罐实际几何与图纸存在偏差，V1 与 V2 存在系统性偏差。
   对每个仓拟合一个体积校正系数 k，使 ``|V1 - k*V2| / V1`` 最小；
2. **逐条记录计算**：用校正后的 k 计算每条装卸记录的体积与相对误差，
   汇总成结果表导出 Excel。

误差指标：相对误差 = ``|装车体积 - 校正后卸车体积| / 装车体积``
"""

from __future__ import annotations

import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .data_loader import ExcelDataLoader
from .tank_geometry import TankGeometry

warnings.filterwarnings("ignore")

#: mm³ → m³ 的单位换算系数
_MM3_TO_M3 = 1e-9

#: 校正系数优化的约束与阈值
_CORRECTION_BOUNDS = (0.90, 1.10)   # k 的允许范围（偏离 1 过大说明模型本身有问题）
_MIN_SAMPLES = 20                    # 参与优化的最少有效记录数
_MIN_IMPROVEMENT_PCT = 0.1           # 优化改进超过该百分比才采用新系数
_REGULARIZATION = 0.01               # 目标函数的正则化强度（|k-1| 线性惩罚）


class TankVolumePipeline:
    """单辆车的体积计算与校正流水线。"""

    def __init__(
        self,
        car_id: str,
        records: pd.DataFrame,
        tank_params: dict[Any, dict[str, float]],
        result_file: str,
    ) -> None:
        self.car_id = car_id
        self.records = records
        self.tank_params = tank_params
        self.result_file = result_file

        #: 各仓的体积校正系数 {仓号: k}
        self.correction_factors: dict[Any, float] = {}

    # ------------------------------------------------------------------
    # 校正系数优化
    # ------------------------------------------------------------------

    def fit_tank_correction_factor(self, tank_id: Any) -> float:
        """为单个仓优化体积校正系数 k。

        用该仓全部有效记录做一维搜索：
        目标函数 = 相对误差中位数 + 0.01*|k-1|（正则化使 k 尽量接近 1）。

        Returns:
            优化后的校正系数（改进不足时返回 1.0，即不校正）
        """
        samples = self._collect_optimization_samples(tank_id)
        if len(samples) < _MIN_SAMPLES:
            print(f"    仓号 {tank_id}：有效数据不足（{len(samples)} 条），使用默认系数 1.0")
            return 1.0

        original_error = self._evaluate(samples, k=1.0)

        start = time.time()
        result = minimize_scalar(
            lambda k: self._objective(samples, k),
            bounds=_CORRECTION_BOUNDS,
            method="bounded",
            options={"xatol": 1e-5, "maxiter": 100},
        )

        if not result.success:
            print(f"    仓号 {tank_id}：优化未收敛，使用默认系数 1.0")
            return 1.0

        optimized_error = self._evaluate(samples, k=result.x)
        improvement = (original_error - optimized_error) / original_error * 100

        if improvement > _MIN_IMPROVEMENT_PCT:
            print(
                f"    仓号 {tank_id}：优化成功 | 系数 {result.x:.6f} | "
                f"改进 {improvement:.2f}% | 耗时 {time.time() - start:.1f}s"
            )
            return result.x

        print(f"    仓号 {tank_id}：优化改进不足（{improvement:.2f}%），使用默认系数 1.0")
        return 1.0

    def _collect_optimization_samples(self, tank_id: Any) -> list[dict[str, float]]:
        """收集某仓可用于优化的有效记录（装/卸油高均大于 0）。"""
        tank_info = self.tank_params.get(tank_id, {})
        samples: list[dict[str, float]] = []

        tank_rows = self.records[self.records["仓号"] == tank_id]
        for _, row in tank_rows.iterrows():
            h1, h2 = row["装车油高"], row["卸车油高"]
            if pd.notna(h1) and pd.notna(h2) and h1 > 0 and h2 > 0:
                samples.append({
                    "h1": h1, "x1": row["倾角x1"], "y1": row["倾角y1"],
                    "h2": h2, "x2": row["倾角x2"], "y2": row["倾角y2"],
                    "tank_info": tank_info,
                })
        return samples

    @staticmethod
    def _compute_volumes(sample: dict[str, float]) -> tuple[float, float]:
        """计算一条记录（同一采样时刻）的装车与卸车体积。"""
        info = sample["tank_info"]
        v1 = _volume_of(info, sample["h1"], sample["x1"], sample["y1"])
        v2 = _volume_of(info, sample["h2"], sample["x2"], sample["y2"])
        return v1, v2

    @staticmethod
    def _evaluate(samples: list[dict[str, float]], k: float) -> float:
        """评估系数 k 的效果：全部样本相对误差的平均值。"""
        errors = []
        for sample in samples:
            v1, v2 = TankVolumePipeline._compute_volumes(sample)
            if v1 > 1e-9 and v2 > 1e-9:
                errors.append(abs(v1 - k * v2) / v1)
        return float(np.mean(errors)) if errors else 1e9

    @staticmethod
    def _objective(samples: list[dict[str, float]], k: float) -> float:
        """优化目标函数：误差中位数 + 正则化项。

        中位数比均值更抗离群记录；正则化让 k 尽量接近 1，
        避免在个别异常数据上过拟合出极端系数。
        """
        errors = []
        for sample in samples:
            v1, v2 = TankVolumePipeline._compute_volumes(sample)
            if v1 > 1e-9 and v2 > 1e-9:
                errors.append(abs(v1 - k * v2) / v1)
        if not errors:
            return 1e9
        return float(np.median(errors) + _REGULARIZATION * abs(k - 1.0))

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def run(self) -> list[dict[str, Any]]:
        """处理该车全部记录，返回结果列表并导出 Excel。"""
        tank_ids = list(self.tank_params.keys())
        if not tank_ids:
            print(f"  警告：未找到车辆 {self.car_id} 的油罐参数，跳过该车")
            return []

        print(f"=== 开始处理车辆：{self.car_id}，共 {len(self.records)} 条记录，"
              f"{len(tank_ids)} 个仓 {tank_ids} ===")

        # 第一步：为每个仓拟合校正系数
        print("  正在为每个仓优化体积校正系数...")
        for tank_id in tank_ids:
            self.correction_factors[tank_id] = self.fit_tank_correction_factor(tank_id)

        corrected_count = sum(1 for k in self.correction_factors.values() if abs(k - 1.0) > 1e-6)
        print(f"  使用校正的仓数量：{corrected_count}/{len(tank_ids)}")

        # 第二步：逐条记录计算体积与误差
        results = self._process_all_records(tank_ids)

        # 第三步：打印统计并导出
        self._print_car_statistics(results)
        if results:
            self._export_to_excel(results)
        return results

    def _process_all_records(self, tank_ids: list[Any]) -> list[dict[str, Any]]:
        """逐条计算该车所有记录，返回结果字典列表。"""
        results: list[dict[str, Any]] = []
        car_errors: list[float] = []

        for idx, row in self.records.iterrows():
            tank_id = row["仓号"]

            # 跳过无参数仓号或油高无效的记录
            if tank_id not in tank_ids:
                continue
            h1, h2 = row["装车油高"], row["卸车油高"]
            if pd.isna(h1) or pd.isna(h2) or h1 <= 0 or h2 <= 0:
                continue

            info = self.tank_params[tank_id]
            k = self.correction_factors.get(tank_id, 1.0)

            v1 = _volume_of(info, h1, row["倾角x1"], row["倾角y1"])
            v2 = _volume_of(info, h2, row["倾角x2"], row["倾角y2"])
            v2_corrected = k * v2

            rel_err = abs(v1 - v2_corrected) / v1 if v1 > 1e-9 else None
            if rel_err is not None:
                car_errors.append(rel_err)

            results.append({
                "车辆编号": self.car_id,
                "时间": row["时间"],
                "仓号": tank_id,
                "装车油高(mm)": h1,
                "装车纵向倾角(°)": row["倾角x1"],
                "装车横向倾角(°)": row["倾角y1"],
                "卸车油高(mm)": h2,
                "卸车纵向倾角(°)": row["倾角x2"],
                "卸车横向倾角(°)": row["倾角y2"],
                "装车体积(m³)": round(v1, 6),
                "卸车体积(m³)": round(v2, 6),
                "校正后卸车体积(m³)": round(v2_corrected, 6),
                "体积校正系数": k,
                "相对误差": round(rel_err, 6) if rel_err is not None else "无",
                "相对误差(%)": round(rel_err * 100, 4) if rel_err is not None else "无",
            })

            if (idx + 1) % 100 == 0 or (idx + 1) == len(self.records):
                print(f"  已处理 {idx + 1}/{len(self.records)} 条记录，有效 {len(results)} 条")

        self.car_errors = car_errors
        return results

    def _print_car_statistics(self, results: list[dict[str, Any]]) -> None:
        """打印该车的误差统计。"""
        valid = [r for r in results if isinstance(r["相对误差"], float)]
        if not valid:
            return
        avg = float(np.mean([r["相对误差"] for r in valid]))
        print(f"  车辆 {self.car_id} 误差统计：")
        print(f"    有效记录总数：{len(valid)} 条")
        print(f"    平均相对误差：{avg:.6f}（{avg * 100:.4f}%）")

    def _export_to_excel(self, results: list[dict[str, Any]]) -> None:
        """把该车结果写入结果 Excel（每辆车一个工作表）。"""
        import os

        result_df = pd.DataFrame(results)
        if not os.path.exists(self.result_file):
            with pd.ExcelWriter(self.result_file, engine="openpyxl") as writer:
                result_df.to_excel(writer, sheet_name=self.car_id, index=False)
        else:
            with pd.ExcelWriter(
                self.result_file, mode="a", engine="openpyxl", if_sheet_exists="replace"
            ) as writer:
                result_df.to_excel(writer, sheet_name=self.car_id, index=False)
        print(f"  车辆 {self.car_id} 结果已导出到 {self.result_file}，有效记录 {len(results)} 条")


# ----------------------------------------------------------------------
# 模块级工具函数
# ----------------------------------------------------------------------

def _volume_of(
    info: dict[str, float], height: float, tilt_x: float, tilt_y: float
) -> float:
    """按油罐参数与油位读数计算体积（m³）。

    Args:
        info: 该仓几何参数字典（d_c/d_y/width/height/L）
        height: 油高读数 (mm)
        tilt_x: 纵向倾角 (度)
        tilt_y: 横向倾角 (度)
    """
    geometry = TankGeometry(
        probe_distance=info["d_c"],
        probe_offset=info["d_y"],
        width=info["width"],
        height=info["height"],
        length=info["L"],
    )
    return geometry.volume(height, tilt_x, tilt_y) * _MM3_TO_M3


def run_pipeline(excel_file: str, result_file: str) -> dict[str, list[dict[str, Any]]]:
    """整体流水线：读取数据 → 逐车处理 → 汇总统计。

    Args:
        excel_file: 装卸数据 Excel 路径（格式见 data_loader 模块说明）
        result_file: 结果 Excel 输出路径（每辆车一个工作表）

    Returns:
        {车辆编号: 该车结果记录列表}
    """
    tank_params, all_raw_data = ExcelDataLoader(excel_file).load()

    all_results: dict[str, list[dict[str, Any]]] = {}
    all_errors: dict[str, list[float]] = defaultdict(list)

    for car in all_raw_data:
        car_id = car["车辆编号"]
        pipeline = TankVolumePipeline(
            car_id=car_id,
            records=car["原始数据"],
            tank_params=tank_params.get(car_id, {}),
            result_file=result_file,
        )
        results = pipeline.run()
        all_results[car_id] = results
        all_errors[car_id] = pipeline.car_errors

    _print_overall_statistics(all_errors, all_results)
    _export_summary_csv(all_errors, result_file)
    return all_results


#: 汇总表的输出文件名
SUMMARY_FILENAME = "summary.csv"

#: 仓库内保存结果的目录（随仓库提交，便于复核 README 里的数字）
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _export_summary_csv(
    all_errors: dict[str, list[float]], result_file: str
) -> None:
    """把逐车误差与整体误差落盘成 CSV，方便引用与复现。

    会写两份：结果 Excel 同目录一份（本次运行产物），
    仓库 ``results/`` 下一份（随代码提交，供他人复核）。
    最后一行是全部车辆的整体统计。
    """
    rows = []
    for car_id, errors in all_errors.items():
        if not errors:
            continue
        avg = float(np.mean(errors))
        rows.append({
            "车辆编号": car_id,
            "有效记录数": len(errors),
            "平均相对误差": avg,
            "平均相对误差(%)": round(avg * 100, 4),
        })

    if rows:
        total = [e for errors in all_errors.values() for e in errors]
        overall = float(np.mean(total))
        rows.append({
            "车辆编号": "全部",
            "有效记录数": len(total),
            "平均相对误差": overall,
            "平均相对误差(%)": round(overall * 100, 4),
        })

    df_summary = pd.DataFrame(rows)
    summary_paths = [
        Path(result_file).resolve().parent / SUMMARY_FILENAME,
        RESULTS_DIR / SUMMARY_FILENAME,
    ]
    for summary_path in summary_paths:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"\n汇总指标已保存至：{summary_path}")


def _print_overall_statistics(
    all_errors: dict[str, list[float]],
    all_results: dict[str, list[dict[str, Any]]],
) -> None:
    """打印全部车辆的整体误差统计。"""
    print("\n" + "=" * 60)
    print("所有车辆整体误差统计")
    print("=" * 60)

    total_errors: list[float] = []
    for car_id, errors in all_errors.items():
        if errors:
            total_errors.extend(errors)
            print(
                f"车辆 {car_id}：{len(errors)} 条记录，"
                f"平均相对误差 {np.mean(errors):.6f}（{np.mean(errors) * 100:.4f}%）"
            )

    if total_errors:
        overall = np.mean(total_errors)
        print(f"\n总计算记录数：{len(total_errors)}")
        print(f"整体平均相对误差：{overall:.6f}（{overall * 100:.4f}%）")
    print("=" * 60)
