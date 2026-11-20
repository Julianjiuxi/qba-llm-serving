"""fig9 绘图：p–k 交互机制验证 — QBA 在 (标签阈值 p, quota k) 网格上的等待时间与提升比例。

只绘图 QBA（本实验仅扫 QBA；FCFS/PSF/AA 与 k 无关）。
- 左侧一列: 以 p 为横轴，不同 k 各一条曲线。
- 右侧一列: 以 χ = p / (k*(1-p)) 为横轴重新排列，并标注 χ=1 参考线。
  若 quota 计数结构主导，不同 k 的曲线应在 χ 轴上更一致。

输出（7张图）:
  fig9_qba_mean_vs_p.pdf / fig9_qba_p99_vs_p.pdf / fig9_qba_tl_p99_vs_p.pdf
  fig9_qba_mean_vs_chi.pdf / fig9_qba_p99_vs_chi.pdf
  fig9_qba_promotion_fraction_vs_p.pdf / fig9_qba_promotion_fraction_vs_chi.pdf
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from ._common import (
    setup_matplotlib,
    ensure_dir,
    apply_axes_style,
    auto_ylim,
    mean_ci,
    meta_summary,
    fill_ci,
    _DEFAULT_PLOT_DIR,
)

# k 曲线样式（本实验无策略对比，故不使用 POLICY_STYLES）
_K_STYLES = {
    3: {"color": "#1F77FF", "marker": "o", "linestyle": "-"},
    5: {"color": "#FF8C1A", "marker": "D", "linestyle": "-"},
    9: {"color": "#8F3FD6", "marker": "s", "linestyle": "-"},
}


def _chi(p: np.ndarray, k: int) -> np.ndarray:
    """quota 计数结构的特征比 χ = p / (k*(1-p))。"""
    return p / (float(k) * (1.0 - p))


def _panel(
    out_dir: str,
    fname: str,
    xlabel: str,
    ylabel: str,
    series: dict,
    *,
    chi1_line: bool = False,
):
    """通用面板：每个 k 一条曲线（含 95% CI），可选 χ=1 参考线。"""
    setup_matplotlib()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ylim_series = []
    xs_all = []
    for k, (xs, m, lo, hi) in series.items():
        st = _K_STYLES.get(int(k), {"color": "#6F6F6F", "marker": "o", "linestyle": "-"})
        ax.plot(xs, m, color=st["color"], marker=st["marker"],
                markersize=9, linestyle=st["linestyle"], linewidth=3.0,
                label=f"k={int(k)}")
        fill_ci(ax, xs, lo, hi, st["color"], alpha=0.16)
        ylim_series.append(m)
        xs_all.append(np.asarray(xs, dtype=np.float64))

    xs_cat = np.concatenate(xs_all)
    x0, x1 = float(np.min(xs_cat)), float(np.max(xs_cat))
    pad = 0.05 * (x1 - x0) if x1 > x0 else 0.05
    xlim = (x0 - pad, x1 + pad)
    if chi1_line:
        ax.axvline(1.0, color="#888888", linestyle=":", linewidth=1.6, zorder=0)
        xlim = (min(xlim[0], 1.0 - pad), max(xlim[1], 1.0 + pad))

    apply_axes_style(
        ax, xlabel=xlabel, ylabel=ylabel,
        xlim=xlim, ylim=auto_ylim(ylim_series), show_legend=True,
    )
    fig.tight_layout()
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="fig9: p–k 交互机制验证绘图")
    parser.add_argument("--data", type=str, default=os.path.join(_DEFAULT_PLOT_DIR, "fig9_data.npz"))
    parser.add_argument("--out-dir", type=str, default=_DEFAULT_PLOT_DIR)
    parser.add_argument("--prefix", type=str, default="fig9")
    parser.add_argument("--verbose-meta", action="store_true", default=False)
    args = parser.parse_args()

    data_path = os.path.abspath(args.data)
    out_dir = ensure_dir(os.path.abspath(args.out_dir))
    prefix = args.prefix

    z = np.load(data_path, allow_pickle=False)
    quota_ks = [int(k) for k in np.asarray(z["quota_ks"]).tolist()]
    percentiles = np.asarray(z["percentiles"], dtype=np.float64)

    def _metric_series(metric: str) -> dict:
        """取 QBA 某指标在 (seed, k, p) 上的逐 seed 均值/CI，按 k 组织。"""
        runs = np.asarray(z[f"psfaq_{metric}_runs"], dtype=np.float64)  # (n_seeds, n_k, n_p)
        out = {}
        for ki, k in enumerate(quota_ks):
            out[k] = mean_ci(runs[:, ki, :])  # 每项形如 (n_p,)
        return out

    meta_path = os.path.join(os.path.dirname(data_path), f"{prefix}_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print("meta" if args.verbose_meta else "meta_summary",
              meta if args.verbose_meta else meta_summary(meta))

    def _vs_p(metric: str) -> dict:
        return {k: (percentiles, m, lo, hi) for k, (m, lo, hi) in _metric_series(metric).items()}

    def _vs_chi(metric: str) -> dict:
        out = {}
        for k, (m, lo, hi) in _metric_series(metric).items():
            out[k] = (_chi(percentiles, k), m, lo, hi)
        return out

    # ── 以 p 为横轴 ──
    for metric, fname, ylab in [
        ("mean", f"{prefix}_qba_mean_vs_p.pdf", "seconds"),
        ("p99", f"{prefix}_qba_p99_vs_p.pdf", "seconds"),
        ("tl_p99", f"{prefix}_qba_tl_p99_vs_p.pdf", "seconds"),
        ("promotion_fraction", f"{prefix}_qba_promotion_fraction_vs_p.pdf", "promotion fraction"),
    ]:
        path = _panel(out_dir, fname, "p", ylab, _vs_p(metric))
        print(f"saved: {path}")

    # ── 以 χ 为横轴（标注 χ=1）──
    for metric, fname, ylab in [
        ("mean", f"{prefix}_qba_mean_vs_chi.pdf", "seconds"),
        ("p99", f"{prefix}_qba_p99_vs_chi.pdf", "seconds"),
        ("promotion_fraction", f"{prefix}_qba_promotion_fraction_vs_chi.pdf", "promotion fraction"),
    ]:
        path = _panel(out_dir, fname, "chi = p / (k*(1-p))", ylab, _vs_chi(metric), chi1_line=True)
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
