"""
DataCollectorAndChartPrinter.py

面向 DGCC / CCDE / SaNSDE 批量实验的数据记录与绘图模块。

主要特性：
- 使用 Matplotlib Agg 非交互式后端，不弹出绘图窗口，不会因 plt.show() 阻塞批量实验。
- 每次绘图自动保存 PNG；默认同时保存对应 CSV。
- 每个 DataCollector 自动生成唯一 run_id，避免 30 次实验相互覆盖。
- 保存后关闭 Figure，释放 Matplotlib 资源。
- 保留原有 DataCollector / ChartPrinter 的主要调用接口。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from itertools import count
import re
from typing import Optional, Sequence

import numpy as np
import pandas as pd

# 必须在导入 pyplot 之前设置后端。
# Agg 只负责生成图片文件，不创建 GUI 窗口。
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


_RUN_COUNTER = count(1)


def _safe_name(value: str) -> str:
    """将函数名等内容转换为适合文件名的字符串。"""
    value = str(value).strip()
    value = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE)
    return value.strip("._") or "unnamed"


def _new_run_id() -> str:
    """时间戳加进程内序号，降低文件名冲突的可能性。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{timestamp}_{next(_RUN_COUNTER):03d}"


class DataCollector:
    """存储一次算法运行过程中的 FES、最佳适应度和误差。"""

    def __init__(
        self,
        func_name: str,
        dim: int,
        algorithm: str = "DE/rand/1/bin",
        run_id: Optional[str] = None,
        output_dir: str | Path = "results",
    ):
        self.func_name = str(func_name)
        self.dim = int(dim)
        self.algorithm = str(algorithm)
        self.history: list[dict] = []

        # 每次构造 DataCollector 默认对应一次独立实验。
        self.run_id = _safe_name(run_id or _new_run_id())
        self.output_dir = Path(output_dir)

    def record(self, fes: int, best_fitness: float, f_global: float):
        """记录一次算法状态。"""
        error = float(best_fitness) - float(f_global)
        self.history.append(
            {
                "fes": int(fes),
                "best_fitness": float(best_fitness),
                "error": float(error),
            }
        )

    def get_dataframe(self) -> pd.DataFrame:
        """导出为 pandas DataFrame。"""
        return pd.DataFrame(
            self.history,
            columns=["fes", "best_fitness", "error"],
        )

    def save_csv(self, file_path: str | Path | None = None) -> Path:
        """保存本次实验的数据；未指定路径时自动生成唯一文件名。"""
        if file_path is None:
            folder = self.output_dir / "csv"
            filename = (
                f"{_safe_name(self.func_name)}_D{self.dim}_"
                f"{_safe_name(self.algorithm)}_{self.run_id}.csv"
            )
            path = folder / filename
        else:
            path = Path(file_path)

        path.parent.mkdir(parents=True, exist_ok=True)
        self.get_dataframe().to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[DataCollector] CSV 已保存：{path.resolve()}")
        return path


class ChartPrinter:
    """自动保存收敛曲线的绘图工具（默认不显示 GUI 窗口）。"""

    def __init__(
        self,
        figsize=(10, 6),
        dpi=100,
        output_dir: str | Path = "results",
        auto_save: bool = True,
        save_csv: bool = True,
    ):
        self.figsize = figsize
        self.dpi = dpi
        self.output_dir = Path(output_dir)
        self.auto_save = bool(auto_save)
        self.save_csv = bool(save_csv)

        # 中文字体按系统可用字体回退；找不到 SimHei 时 Matplotlib 会使用默认字体。
        plt.rcParams["font.sans-serif"] = [
            "SimHei",
            "Microsoft YaHei",
            "Noto Sans CJK SC",
            "DejaVu Sans",
        ]
        plt.rcParams["axes.unicode_minus"] = False

    def _make_filename(self, collector: DataCollector, suffix: str = "convergence") -> str:
        return (
            f"{_safe_name(collector.func_name)}_D{collector.dim}_"
            f"{_safe_name(collector.algorithm)}_{collector.run_id}_{suffix}.png"
        )

    @staticmethod
    def _prepare_y(values, log_y: bool):
        """对数坐标不能直接绘制 0 或负值；仅对绘图数据做下限处理。"""
        y = np.asarray(values, dtype=float)
        if log_y:
            positive = y[np.isfinite(y) & (y > 0)]
            floor = max(float(np.min(positive)) * 0.1, np.finfo(float).tiny) if positive.size else 1e-300
            y = np.where(np.isfinite(y), np.maximum(y, floor), np.nan)
        return y

    def plot(
        self,
        data_collector: DataCollector,
        plot_error: bool = True,
        log_y: bool = True,
        show: bool = False,
        file_path: str | Path | None = None,
        save_csv: bool | None = None,
    ) -> Path | None:
        """
        绘制并默认保存单条收敛曲线。

        show 参数仅为兼容旧代码而保留。当前模块固定使用 Agg 后端，
        不会弹出 GUI 窗口；因此 show=True 也不会等待用户关闭图片。
        """
        df = data_collector.get_dataframe()
        if df.empty:
            raise ValueError("当前 DataCollector 没有记录数据，无法绘图。")

        y_col = "error" if plot_error else "best_fitness"
        y_label = "误差值 (f(x) - f_global)" if plot_error else "适应度值 f(x)"

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        try:
            y_values = self._prepare_y(df[y_col].to_numpy(), log_y)
            ax.plot(
                df["fes"].to_numpy(),
                y_values,
                linewidth=1.5,
                label=data_collector.algorithm,
            )
            ax.set_xlabel("函数评估次数 (FES)", fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            ax.set_title(
                f"{data_collector.func_name} (D={data_collector.dim}) 收敛曲线",
                fontsize=14,
            )
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11)

            if log_y:
                ax.set_yscale("log")

            saved_path = None
            if self.auto_save or file_path is not None:
                if file_path is None:
                    folder = self.output_dir / "figures"
                    path = folder / self._make_filename(data_collector)
                else:
                    path = Path(file_path)

                path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(path, dpi=300, bbox_inches="tight")
                saved_path = path
                print(f"[ChartPrinter] 图片已保存：{path.resolve()}")

            should_save_csv = self.save_csv if save_csv is None else bool(save_csv)
            if should_save_csv:
                # 默认放在 results/csv；如果 DataCollector 自己指定了 output_dir，则使用它。
                data_collector.save_csv()

            return saved_path
        finally:
            # 无论保存是否成功，都关闭 Figure，防止长时间批量运行时累积内存。
            plt.close(fig)

    def plot_multi(
        self,
        data_collectors: Sequence[DataCollector],
        plot_error: bool = True,
        log_y: bool = True,
        title: str = "算法对比收敛曲线",
        show: bool = False,
        file_path: str | Path | None = None,
    ) -> Path | None:
        """绘制多条收敛曲线并自动保存。"""
        if not data_collectors:
            raise ValueError("data_collectors 为空，无法绘制对比曲线。")

        y_col = "error" if plot_error else "best_fitness"
        y_label = "误差值 (f(x) - f_global)" if plot_error else "适应度值 f(x)"

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        try:
            for dc in data_collectors:
                df = dc.get_dataframe()
                if df.empty:
                    continue
                y_values = self._prepare_y(df[y_col].to_numpy(), log_y)
                ax.plot(
                    df["fes"].to_numpy(),
                    y_values,
                    linewidth=1.5,
                    label=dc.algorithm,
                )

            ax.set_xlabel("函数评估次数 (FES)", fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            ax.set_title(title, fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11)

            if log_y:
                ax.set_yscale("log")

            if file_path is None:
                folder = self.output_dir / "figures"
                filename = f"comparison_{_new_run_id()}.png"
                path = folder / filename
            else:
                path = Path(file_path)

            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"[ChartPrinter] 对比图片已保存：{path.resolve()}")
            return path
        finally:
            plt.close(fig)

    def save(self, file_path: str | Path):
        """
        兼容旧接口：保存当前 pyplot 图。
        新代码优先使用 plot(..., file_path=...)，因为它会明确管理 Figure 生命周期。
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[ChartPrinter] 图片已保存：{path.resolve()}")
        plt.close("all")
        return path
