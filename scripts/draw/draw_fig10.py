"""fig10 绘图：prefill/decode 构成扫描（总 token 需求归一化）。

FCFS/PSF/AA/QBA 在"匹配平均总 token 需求"下、不同 prefill/decode 构成下的 waiting time。
横轴为构成倍率 a（以 2 为底的对数刻度，使 0.25/0.5/1/2/4 等距）。
与 fig8 的关键区别：总 token 需求被归一化固定，因此这里看到的是"纯构成变化"，
而 fig8 是"构成变化 + 总需求膨胀"的对照。
注意: 构成变化会改变整体 workload，故 FCFS 作为曲线绘制；相对指标以同倍率同 seed 的 FCFS 为基准。
RR 为内部策略，不绘图。

输出（11张图）：
  整体: fig10_sim_waiting_{mean,p95,p99}_vs_a.pdf
  per-class: fig10_true_short_{mean,p99}_vs_a.pdf
             fig10_true_long_{mean,p99}_vs_a.pdf
  相对等待:  fig10_true_short_relative_wait_vs_a.pdf
             fig10_true_long_relative_wait_vs_a.pdf
  诊断:     fig10_total_token_mean_vs_a.pdf   （应近似水平线，验证总需求守恒）
            fig10_token_ratio_vs_a.pdf        （应随 a 单调上升，验证构成改变）
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


def _safe_relative(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """逐元素相对比值；分母为 0 处标记为 NaN（不可定义），不引入 epsilon 改变结果。

    低负载/小样本下 FCFS 某类 mean wait 可能为 0，直接相除会得到 inf/NaN 并污染 CI。
    这里把分母为 0 的点显式置为 NaN（绘图时呈现断点），其余点保留真实比值。
    """
    numer = np.asarray(numer, dtype=np.float64)
    denom = np.asarray(denom, dtype=np.float64)
    out = np.full(numer.shape, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(numer, denom, out=out, where=(denom != 0.0))
    return out


def _plot_panel(
    out_dir: str,
    prefix: str,
    x_vals: np.ndarray,
    sweep_runs: dict,
    fname: str,
    *,
    xlabel: str = "Composition scale a",
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


def _plot_work_diag_panel(
    out_dir: str,
    x_vals: np.ndarray,
    runs: np.ndarray,
    fname: str,
    *,
    xlabel: str = "Composition scale a",
    ylabel: str,
):
    """workload 诊断面板：单条曲线（跨 seed 均值 + 95% CI），log2 横轴。"""
    setup_matplotlib()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_xscale("log", base=2)

    m, lo, hi = mean_ci(runs)
    ax.plot(x_vals, m, color="#333333", marker="o", markersize=9,
            linestyle="-", linewidth=3.0, label=ylabel)
    fill_ci(ax, x_vals, lo, hi, "#333333", alpha=0.15)

    apply_axes_style(
        ax, xlabel=xlabel, ylabel=ylabel,
        xlim=(float(np.min(x_vals)), float(np.max(x_vals))),
        ylim=auto_ylim([m]), show_legend=True,
    )
    _set_log2_ticks(ax, x_vals)
    fig.tight_layout()
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="fig10: prefill/decode 构成扫描绘图")
    parser.add_argument("--data", type=str, default=os.path.join(_DEFAULT_PLOT_DIR, "fig10_data.npz"))
    parser.add_argument("--out-dir", type=str, default=_DEFAULT_PLOT_DIR)
    parser.add_argument("--prefix", type=str, default="fig10")
    parser.add_argument("--verbose-meta", action="store_true", default=False)
    args = parser.parse_args()

    data_path = os.path.abspath(args.data)
    out_dir = ensure_dir(os.path.abspath(args.out_dir))
    prefix = args.prefix

    z = np.load(data_path, allow_pickle=False)
    scales = np.asarray(z["composition_scales"], dtype=np.float64)

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
    for mk in ["mean", "p95", "p99"]:
        sweep_sub = {p: sweep_data[p][mk] for p in sweep_data}
        path = _plot_panel(
            out_dir, prefix, scales, sweep_sub,
            f"{prefix}_sim_waiting_{mk}_vs_a.pdf",
        )
        print(f"saved: {path}")

    # ── 4. true-short mean ──
    sweep_sub = {p: sweep_data[p]["ts_mean"] for p in sweep_data}
    print(f"saved: {_plot_panel(out_dir, prefix, scales, sweep_sub, f'{prefix}_true_short_mean_vs_a.pdf')}")

    # ── 5. true-short p99 ──
    sweep_sub = {p: sweep_data[p]["ts_p99"] for p in sweep_data}
    print(f"saved: {_plot_panel(out_dir, prefix, scales, sweep_sub, f'{prefix}_true_short_p99_vs_a.pdf')}")

    # ── 6. true-long mean ──
    sweep_sub = {p: sweep_data[p]["tl_mean"] for p in sweep_data}
    print(f"saved: {_plot_panel(out_dir, prefix, scales, sweep_sub, f'{prefix}_true_long_mean_vs_a.pdf')}")

    # ── 7. true-long p99 ──
    sweep_sub = {p: sweep_data[p]["tl_p99"] for p in sweep_data}
    print(f"saved: {_plot_panel(out_dir, prefix, scales, sweep_sub, f'{prefix}_true_long_p99_vs_a.pdf')}")

    # ── 8. true-short relative mean wait ──
    fcfs_ts = sweep_data["FCFS"]["ts_mean"]
    sweep_sub = {p: _safe_relative(sweep_data[p]["ts_mean"], fcfs_ts) for p in sweep_data}
    print(f"saved: {_plot_panel(out_dir, prefix, scales, sweep_sub, f'{prefix}_true_short_relative_wait_vs_a.pdf', ylabel='relative mean wait', is_relative_metric=True)}")

    # ── 9. true-long relative mean wait ──
    fcfs_tl = sweep_data["FCFS"]["tl_mean"]
    sweep_sub = {p: _safe_relative(sweep_data[p]["tl_mean"], fcfs_tl) for p in sweep_data}
    print(f"saved: {_plot_panel(out_dir, prefix, scales, sweep_sub, f'{prefix}_true_long_relative_wait_vs_a.pdf', ylabel='relative mean wait', is_relative_metric=True)}")

    # ── 10. 总 token 需求（应近似水平，验证归一化） ──
    if "total_mean_runs" in z.files:
        runs = np.asarray(z["total_mean_runs"], dtype=np.float64)
        print(f"saved: {_plot_work_diag_panel(out_dir, scales, runs, f'{prefix}_total_token_mean_vs_a.pdf', ylabel='mean total tokens E[P+D]')}")

    # ── 11. token_ratio = E[P]/E[D]（应随 a 上升，验证构成改变） ──
    if "token_ratio_runs" in z.files:
        runs = np.asarray(z["token_ratio_runs"], dtype=np.float64)
        print(f"saved: {_plot_work_diag_panel(out_dir, scales, runs, f'{prefix}_token_ratio_vs_a.pdf', ylabel='token ratio E[P]/E[D]')}")


if __name__ == "__main__":
    main()
