"""fig8 绘图：prompt_scale（prompt 长度倍率）扫参 — FCFS/PSF/AA/QBA 在不同 prompt 尺度下的 waiting time。

横轴为 prompt scale a（以 2 为底的对数刻度，使 0.25/0.5/1/2/4 等距）。
注意: prompt 缩放会改变整体 workload，因此 FCFS 的整体指标随倍率变化，
此处 FCFS 作为曲线绘制而非水平参考线；相对指标以"相同倍率、相同 seed 的 FCFS"为基准。
RR 为内部策略，不绘图。

输出（9张图）：
  整体: fig8_sim_waiting_{mean,p95,p99}_vs_a.pdf
  per-class: fig8_true_short_{mean,p99}_vs_a.pdf
             fig8_true_long_{mean,p99}_vs_a.pdf
  相对等待:  fig8_true_short_relative_wait_vs_a.pdf
             fig8_true_long_relative_wait_vs_a.pdf
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


def _set_log2_ticks(ax, x_vals: np.ndarray) -> None:
    """在 log2 横轴上显式标注真实倍率刻度，并关闭次刻度。"""
    ticks = [float(v) for v in x_vals]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{v:g}" for v in ticks])
    ax.minorticks_off()


def _plot_panel(
    out_dir: str,
    prefix: str,
    x_vals: np.ndarray,
    sweep_runs: dict,
    fname: str,
    *,
    xlabel: str = "Prompt scale",
    ylabel: str = "seconds",
    is_relative_metric: bool = False,
):
    """通用面板：4条扫参曲线（FCFS/PSF/AA/QBA），log2 横轴，无水平参考线。"""
    setup_matplotlib()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_xscale("log", base=2)

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
    _set_log2_ticks(ax, x_vals)
    fig.tight_layout()
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="fig8: prompt_scale 扫参绘图")
    parser.add_argument("--data", type=str, default=os.path.join(_DEFAULT_PLOT_DIR, "fig8_data.npz"))
    parser.add_argument("--out-dir", type=str, default=_DEFAULT_PLOT_DIR)
    parser.add_argument("--prefix", type=str, default="fig8")
    parser.add_argument("--verbose-meta", action="store_true", default=False)
    args = parser.parse_args()

    data_path = os.path.abspath(args.data)
    out_dir = ensure_dir(os.path.abspath(args.out_dir))
    prefix = args.prefix

    z = np.load(data_path, allow_pickle=False)
    prompt_scales = np.asarray(z["prompt_scales"], dtype=np.float64)

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
            out_dir, prefix, prompt_scales, sweep_sub,
            f"{prefix}_sim_waiting_{mk}_vs_a.pdf",
        )
        print(f"saved: {path}")

    # ── 4. true-short mean ──
    sweep_sub = {p: sweep_data[p]["ts_mean"] for p in sweep_data}
    path = _plot_panel(
        out_dir, prefix, prompt_scales, sweep_sub,
        f"{prefix}_true_short_mean_vs_a.pdf",
        ylabel="seconds",
    )
    print(f"saved: {path}")

    # ── 5. true-short p99 ──
    sweep_sub = {p: sweep_data[p]["ts_p99"] for p in sweep_data}
    path = _plot_panel(
        out_dir, prefix, prompt_scales, sweep_sub,
        f"{prefix}_true_short_p99_vs_a.pdf",
        ylabel="seconds",
    )
    print(f"saved: {path}")

    # ── 6. true-long mean ──
    sweep_sub = {p: sweep_data[p]["tl_mean"] for p in sweep_data}
    path = _plot_panel(
        out_dir, prefix, prompt_scales, sweep_sub,
        f"{prefix}_true_long_mean_vs_a.pdf",
        ylabel="seconds",
    )
    print(f"saved: {path}")

    # ── 7. true-long p99 ──
    sweep_sub = {p: sweep_data[p]["tl_p99"] for p in sweep_data}
    path = _plot_panel(
        out_dir, prefix, prompt_scales, sweep_sub,
        f"{prefix}_true_long_p99_vs_a.pdf",
        ylabel="seconds",
    )
    print(f"saved: {path}")

    # ── 8. true-short relative mean wait ──
    fcfs_ts = sweep_data["FCFS"]["ts_mean"]  # (n_seeds, n_scales)
    sweep_sub = {}
    for p in sweep_data:
        sweep_sub[p] = sweep_data[p]["ts_mean"] / fcfs_ts
    path = _plot_panel(
        out_dir, prefix, prompt_scales, sweep_sub,
        f"{prefix}_true_short_relative_wait_vs_a.pdf",
        ylabel="relative mean wait", is_relative_metric=True,
    )
    print(f"saved: {path}")

    # ── 9. true-long relative mean wait ──
    fcfs_tl = sweep_data["FCFS"]["tl_mean"]  # (n_seeds, n_scales)
    sweep_sub = {}
    for p in sweep_data:
        sweep_sub[p] = sweep_data[p]["tl_mean"] / fcfs_tl
    path = _plot_panel(
        out_dir, prefix, prompt_scales, sweep_sub,
        f"{prefix}_true_long_relative_wait_vs_a.pdf",
        ylabel="relative mean wait", is_relative_metric=True,
    )
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
