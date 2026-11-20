"""fig7 绘图：label_threshold_percentail（p）扫参 — FCFS/PSF/AA/QBA 在不同长短划分阈值下的 waiting time。

p 是 answer_len 的分位数阈值，决定真值短/长标签的划分。
注意: 与 fig5/fig6 不同，p 会改变真值短/长划分本身，因此 FCFS 的 per-class 指标也随 p
      变化，此处 FCFS 作为曲线绘制而非水平参考线。
RR 为内部策略，不绘图。

输出（9张图）：
  整体: fig7_sim_waiting_{mean,p95,p99}_vs_p.pdf
  per-class: fig7_true_short_{mean,p99}_vs_p.pdf
             fig7_true_long_{mean,p99}_vs_p.pdf
  相对等待:  fig7_true_short_relative_wait_vs_p.pdf
             fig7_true_long_relative_wait_vs_p.pdf
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
    display_policy_name,
    get_policy_style,
    meta_summary,
    fill_ci,
    _DEFAULT_PLOT_DIR,
)

# (npz 策略键, 样式键)；RR 为内部策略，不绘图
_POLICIES = [("FCFS", "FCFS"), ("PSF", "PSF"), ("AA", "PSFA"), ("QBA", "PSFAQ")]


def _plot_panel(
    out_dir: str,
    prefix: str,
    x_vals: np.ndarray,
    sweep_runs: dict,
    fname: str,
    *,
    xlabel: str = "p",
    ylabel: str = "seconds",
    is_relative_metric: bool = False,
):
    """通用面板：4条扫参曲线（FCFS/PSF/AA/QBA），无水平参考线。"""
    setup_matplotlib()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ylim_series = []

    for policy, style_key in _POLICIES:
        st = get_policy_style(style_key)
        m, lo, hi = mean_ci(sweep_runs[policy])
        ax.plot(x_vals, m, color=st["color"], marker=st["marker"],
                markersize=9, linestyle=st["linestyle"], linewidth=3.0,
                label=display_policy_name(style_key))
        fill_ci(ax, x_vals, lo, hi, st["color"],
                alpha=0.16 if style_key == "PSFA" else 0.18)
        ylim_series.append(m)

    ylim = auto_ylim(ylim_series)
    if is_relative_metric:
        # relative metric 的 y 轴从 0.5 附近开始，留一点下界
        ylim = (max(0.0, ylim[0] - 0.1), ylim[1])
    apply_axes_style(
        ax, xlabel=xlabel, ylabel=ylabel,
        xlim=(float(np.min(x_vals)), float(np.max(x_vals))),
        ylim=ylim, show_legend=True,
    )
    fig.tight_layout()
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="fig7: label_threshold_percentail 扫参绘图")
    parser.add_argument("--data", type=str, default=os.path.join(_DEFAULT_PLOT_DIR, "fig7_data.npz"))
    parser.add_argument("--out-dir", type=str, default=_DEFAULT_PLOT_DIR)
    parser.add_argument("--prefix", type=str, default="fig7")
    parser.add_argument("--verbose-meta", action="store_true", default=False)
    args = parser.parse_args()

    data_path = os.path.abspath(args.data)
    out_dir = ensure_dir(os.path.abspath(args.out_dir))
    prefix = args.prefix

    z = np.load(data_path, allow_pickle=False)
    percentiles = np.asarray(z["percentiles"], dtype=np.float64)

    def _read_runs(policy_prefix):
        d = {}
        for mk in ["mean", "p95", "p99", "ts_mean", "ts_p99", "tl_mean", "tl_p99"]:
            key = f"{policy_prefix}_{mk}_runs"
            if key in z.files:
                d[mk] = np.asarray(z[key], dtype=np.float64)
        return d

    sweep_data = {}
    for p, pfx in [("FCFS", "fcfs"), ("PSF", "psf"), ("AA", "psfa"), ("QBA", "psfaq")]:
        sweep_data[p] = _read_runs(pfx)

    meta_path = os.path.join(os.path.dirname(data_path), f"{prefix}_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if args.verbose_meta:
            print("meta", meta)
        else:
            print("meta_summary", meta_summary(meta))

    # ── 1-3. 整体 waiting time (mean / p95 / p99) ──
    for mk, key in [("mean", "mean"), ("p95", "p95"), ("p99", "p99")]:
        sweep_sub = {p: sweep_data[p][key] for p in sweep_data}
        path = _plot_panel(
            out_dir, prefix, percentiles, sweep_sub,
            f"{prefix}_sim_waiting_{mk}_vs_p.pdf",
        )
        print(f"saved: {path}")

    # ── 4. true-short mean ──
    sweep_sub = {p: sweep_data[p]["ts_mean"] for p in sweep_data}
    path = _plot_panel(
        out_dir, prefix, percentiles, sweep_sub,
        f"{prefix}_true_short_mean_vs_p.pdf",
        ylabel="seconds",
    )
    print(f"saved: {path}")

    # ── 5. true-short p99 ──
    sweep_sub = {p: sweep_data[p]["ts_p99"] for p in sweep_data}
    path = _plot_panel(
        out_dir, prefix, percentiles, sweep_sub,
        f"{prefix}_true_short_p99_vs_p.pdf",
        ylabel="seconds",
    )
    print(f"saved: {path}")

    # ── 6. true-long mean ──
    sweep_sub = {p: sweep_data[p]["tl_mean"] for p in sweep_data}
    path = _plot_panel(
        out_dir, prefix, percentiles, sweep_sub,
        f"{prefix}_true_long_mean_vs_p.pdf",
        ylabel="seconds",
    )
    print(f"saved: {path}")

    # ── 7. true-long p99 ──
    sweep_sub = {p: sweep_data[p]["tl_p99"] for p in sweep_data}
    path = _plot_panel(
        out_dir, prefix, percentiles, sweep_sub,
        f"{prefix}_true_long_p99_vs_p.pdf",
        ylabel="seconds",
    )
    print(f"saved: {path}")

    # ── 8. true-short relative mean wait ──
    fcfs_ts = sweep_data["FCFS"]["ts_mean"]  # (n_seeds, n_points)
    sweep_sub = {}
    for p in sweep_data:
        sweep_sub[p] = sweep_data[p]["ts_mean"] / fcfs_ts
    path = _plot_panel(
        out_dir, prefix, percentiles, sweep_sub,
        f"{prefix}_true_short_relative_wait_vs_p.pdf",
        ylabel="relative mean wait", is_relative_metric=True,
    )
    print(f"saved: {path}")

    # ── 9. true-long relative mean wait ──
    fcfs_tl = sweep_data["FCFS"]["tl_mean"]  # (n_seeds, n_points)
    sweep_sub = {}
    for p in sweep_data:
        sweep_sub[p] = sweep_data[p]["tl_mean"] / fcfs_tl
    path = _plot_panel(
        out_dir, prefix, percentiles, sweep_sub,
        f"{prefix}_true_long_relative_wait_vs_p.pdf",
        ylabel="relative mean wait", is_relative_metric=True,
    )
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
