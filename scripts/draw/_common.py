"""绘图共享模块：样式、工具函数。

从旧 draw_fig*.py 中提取的公共逻辑，避免三个脚本重复。
"""

from __future__ import annotations

import os
import json

import numpy as np
import matplotlib.pyplot as plt


POLICY_STYLES = {
    "FCFS": {"color": "#1F77FF", "marker": "o", "linestyle": "--"},
    "PSF": {"color": "#FF8C1A", "marker": "D", "linestyle": "--"},
    "PSFA": {"color": "#00B050", "marker": "^", "linestyle": "-"},
    "PSFAQ": {"color": "#8F3FD6", "marker": "o", "linestyle": "-"},
    "RR": {"color": "#E63946", "marker": "s", "linestyle": "--"},
}


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "#FAFAFA"
    plt.rcParams["savefig.facecolor"] = "white"


def ensure_dir(d: str) -> str:
    d = os.path.abspath(d)
    os.makedirs(d, exist_ok=True)
    return d


def infer_prefix(data_path: str) -> str:
    base = os.path.basename(data_path)
    if base.endswith("_data.npz"):
        return base[:-len("_data.npz")]
    if base.endswith(".npz"):
        return base[:-len(".npz")]
    return base


def apply_axes_style(
    ax,
    *,
    xlabel: str = "",
    ylabel: str = "",
    xlim=None,
    ylim=None,
    show_legend: bool = True,
):
    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_ylabel(ylabel, fontsize=24)
    ax.tick_params(axis="both", labelsize=21)
    ax.grid(color="#D8D8D8", linestyle="--", linewidth=0.8, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(1.3)
    if xlim is not None:
        x0, x1 = float(xlim[0]), float(xlim[1])
        if x0 == x1:
            pad = 1.0 if x0 == 0.0 else 0.05 * abs(x0)
            x0 -= pad
            x1 += pad
        ax.set_xlim(x0, x1)
    if ylim is not None:
        y0, y1 = float(ylim[0]), float(ylim[1])
        if y0 == y1:
            pad = 0.1 if y0 == 0.0 else 0.05 * abs(y0)
            y0 -= pad
            y1 += pad
        ax.set_ylim(y0, y1)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            legend = ax.legend(
                loc="upper left",
                fontsize=20,
                frameon=True,
                fancybox=True,
                framealpha=0.92,
                ncol=1,
                handlelength=2.2,
            )
            legend.get_frame().set_edgecolor((0, 0, 0, 0.2))
            legend.get_frame().set_linewidth(0.8)


def auto_ylim(series_list: list[np.ndarray]):
    ys = np.concatenate([
        np.asarray(s, dtype=np.float64).reshape(-1)
        for s in series_list
        if s is not None and np.size(s) > 0
    ])
    ys = ys[np.isfinite(ys)]
    if ys.size == 0:
        return None
    y_min = float(np.min(ys))
    y_max = float(np.max(ys))
    if y_max == y_min:
        pad = 0.1 if y_max == 0.0 else 0.05 * abs(y_max)
    else:
        pad = 0.05 * (y_max - y_min)
    return (y_min - pad, y_max + pad)


def mean_ci(runs: np.ndarray, z: float = 1.96):
    """计算均值与置信区间。当 seed 数 < 2 时返回 (均值, None, None)。"""
    x = np.asarray(runs, dtype=np.float64)
    if x.ndim == 0:
        x = x.reshape(1, 1)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    m = np.nanmean(x, axis=0)
    n_axis1 = x.shape[0]
    if n_axis1 < 2:
        return m, None, None  # seed-count=1 时无 CI
    n = np.sum(np.isfinite(x), axis=0).astype(np.float64)
    s = np.nanstd(x, axis=0, ddof=1)
    se = np.where(n > 0, s / np.sqrt(np.maximum(n, 1.0)), np.nan)
    lo = m - float(z) * se
    hi = m + float(z) * se
    return m, lo, hi


def display_policy_name(name: str) -> str:
    return {"PSFA": "AA", "PSFAQ": "QBA"}.get(name, name)


def get_policy_style(name: str) -> dict:
    return POLICY_STYLES.get(name, {"color": "#6F6F6F", "marker": "o", "linestyle": "-"})


def meta_summary(meta: dict) -> dict:
    return {
        "cfg_sim_snapshot": meta.get("cfg_sim_snapshot"),
        "cfg_formula_snapshot": meta.get("cfg_formula_snapshot"),
        "batch_time_source": meta.get("batch_time_source"),
        "args": meta.get("args"),
        "git_commit": (meta.get("git") or {}).get("git_commit"),
        "git_dirty": (meta.get("git") or {}).get("git_dirty"),
        "generated_at_utc": meta.get("generated_at_utc"),
        "config_path": meta.get("config_path"),
        "npz_sha256": meta.get("npz_sha256"),
    }


def fill_ci(ax, x: np.ndarray, lo: np.ndarray | None, hi: np.ndarray | None, color: str, alpha: float = 0.18):
    """仅在 CI 数据有效时填充置信区间。"""
    if lo is None or hi is None:
        return
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0)


# 默认数据/输出目录：scripts/plot/
_DEFAULT_PLOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "plot")
)
