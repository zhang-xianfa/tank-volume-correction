# -*- coding: utf-8 -*-
"""
椭圆截面卧式油罐的液体体积几何模型
========================================

这是整个项目的数学核心：给定油罐尺寸、油位探针读数与罐体倾角，
计算罐内液体体积。所有下游计算（体积校正、误差统计）都建立在本模块之上。

几何模型推导
------------

油罐为**卧式椭圆截面柱体**（截面是长轴水平的椭圆）：

- 椭圆截面半轴：``a = width / 2``（水平方向）、``b = height / 2``（竖直方向）
- 罐体长度 ``L``，油位探针距罐头 ``d_c``，探针偏离截面中心的横向距离 ``d_y``

坐标系（见图 README.md 中的示意图）：

- x 轴：沿罐体长度方向，探针位置为原点，向罐尾为正
  （因此 x 的积分范围是 ``[-d_c, L - d_c]``）
- y 轴：沿截面宽度方向，椭圆中心为原点
- z 轴：竖直方向，椭圆中心为原点

在截面位置 y 处，椭圆上下边界分别为：:

    z_top(y)    = +b * sqrt(1 - (y/a)^2)
    z_bottom(y) = -b * sqrt(1 - (y/a)^2)

罐体倾斜时，液面（世界坐标系中的水平面）在罐体坐标系里是一个**平面**：
纵向倾角 alpha 使液面沿 x 方向线性变化，横向倾角 beta 使液面沿 y 方向线性变化。

探针读数是"液面到探针处罐底的高度 h"，而探针并不在截面中心，需要折算：

    H = h + d_y * tan(beta) + b - |C_z|        (1)

其中 ``C_z = b * sqrt(1 - (d_y/a)^2)`` 是探针正下方椭圆边界的 z 坐标。
式 (1) 的含义：把探针处的液位，换算成"过 x=0 截面中心线上的液面高度 H"。

于是任意位置 (x, y) 处的液面高度（相对椭圆中心）为：:

    z_surface(x, y) = (H - b) - x * tan(alpha) - y * tan(beta)

该处的液柱高度 = 液面与椭圆上边界的较小者 - 椭圆下边界，且不小于 0：:

    zA(x, y) = max( min(z_surface, z_top(y)) - z_bottom(y), 0 )

最后在椭圆投影区域上做二重积分（梯形法），即得液体体积：:

    V = ∬ zA(x, y) dx dy

数值实现
--------

- 使用 ``numpy.meshgrid`` 一次性生成 (x, y) 网格，整张网格**向量化**计算 zA，
  再用两次 ``numpy.trapz`` 完成二重积分，避免逐点 Python 循环；
- 网格密度 ``nx``（长度方向）与 ``ny``（宽度方向）可调，
  默认 200×100 在精度与速度间取得平衡；
- 输入长度单位统一为 mm，输出体积单位为 mm³，调用方乘以 1e-9 换算为 m³。
"""

from __future__ import annotations

import numpy as np


class TankGeometry:
    """椭圆截面卧式油罐的体积计算器。

    Attributes:
        probe_distance: 油位探针到罐头的距离 d_c (mm)
        probe_offset:   探针偏离截面中心线的横向距离 d_y (mm)，
                        即数据表中 "-X" 列的原值（从罐侧壁起算）
        width:          椭圆截面宽度（长轴）(mm)
        height:         椭圆截面高度（短轴）(mm)
        length:         罐体长度 L (mm)
    """

    def __init__(
        self,
        probe_distance: float,
        probe_offset: float,
        width: float,
        height: float,
        length: float,
    ) -> None:
        self.probe_distance = probe_distance
        self.width = width
        self.height = height
        self.length = length

        # 截面半轴
        self.a = width / 2.0
        self.b = height / 2.0

        # 探针横向偏移换算为"相对椭圆中心"的坐标
        self.d_y = probe_offset - self.a

        # 沿长度方向的积分区间：探针处为原点，[-d_c, L - d_c]
        self.m_head = probe_distance          # 原点左侧（罐头侧）长度
        self.m_tail = length - probe_distance  # 原点右侧（罐尾侧）长度

        # 探针正下方椭圆边界的 z 坐标（式(1)中的 C_z）
        self.c_z = self.b * np.sqrt(1.0 - (self.d_y / self.a) ** 2)

    # ------------------------------------------------------------------
    # 核心计算
    # ------------------------------------------------------------------

    def volume(
        self,
        oil_height: float,
        longitudinal_tilt: float,
        lateral_tilt: float,
        nx: int = 200,
        ny: int = 100,
    ) -> float:
        """计算给定油位与倾角下的液体体积。

        Args:
            oil_height: 探针实测油高 h (mm)
            longitudinal_tilt: 纵向倾角 alpha (度)，罐头-罐尾连线方向的倾角
            lateral_tilt:      横向倾角 beta (度)，截面宽度方向的倾角
            nx, ny: 长度/宽度方向的网格划分数，越大越精确但越慢

        Returns:
            液体体积 (mm³)
        """
        alpha = np.radians(longitudinal_tilt)
        beta = np.radians(lateral_tilt)

        # 式(1)：探针液位 → 截面中心线液面高度 H
        center_level = (
            oil_height
            + self.d_y * np.tan(beta)
            + self.b
            - abs(self.c_z)
        )

        # 积分网格
        x_points = np.linspace(-self.m_head, self.m_tail, nx)
        y_points = np.linspace(-self.a, self.a, ny)
        x_grid, y_grid = np.meshgrid(x_points, y_points)

        # 网格上逐点的液柱高度 zA(x, y)
        column_heights = self._column_heights(
            x_grid, y_grid, center_level, alpha, beta
        )

        # 梯形法二重积分（先沿 y 后沿 x）
        return float(np.trapz(np.trapz(column_heights, y_points, axis=0), x_points))

    def _column_heights(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        center_level: float,
        alpha: float,
        beta: float,
    ) -> np.ndarray:
        """向量化计算网格上每一点 (x, y) 的液柱高度 zA。

        Args:
            x_grid, y_grid: 与 meshgrid 生成的积分网格
            center_level:   式(1) 得到的截面中心线液面高度 H
            alpha, beta:    纵向/横向倾角（弧度）

        Returns:
            与网格同形状的液柱高度数组
        """
        # 液面高度沿 x 方向线性变化，并截断到罐体高度范围 [0, 2b]
        surface_x = np.clip(center_level - x_grid * np.tan(alpha), 0.0, 2.0 * self.b)

        # 液面相对椭圆中心的 z 坐标（再叠加横向倾角的影响）
        surface_z = surface_x - self.b - y_grid * np.tan(beta)

        # 椭圆上下边界
        z_top = self.b * np.sqrt(1.0 - (y_grid / self.a) ** 2)
        z_bottom = -z_top

        # 液柱高度 = min(液面, 上边界) - 下边界，且不低于 0
        return np.maximum(np.minimum(surface_z, z_top) - z_bottom, 0.0)
