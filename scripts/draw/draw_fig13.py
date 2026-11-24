"""fig13 绘图：跨负载的构成扫描（fig11 的小型迁移验证）。

FCFS/PSF/AA/QBA 在"逐请求总 token 严格守恒"下、不同 prefill/decode 构成（a）与不同
到达率（λ）下的 waiting time。每个 λ 一个子图，子图内为 4 条策略曲线，横轴为构成倍率 a
（以 2 为底的对数刻度）。用于检查 fig11 的构成效应与非单调趋势在其它负载下是否仍存在。
RR 为内部策略，不绘图。

输出：
  整体:   fig13_sim_waiting_{mean,p99}_vs_a.pdf
  per-class: fig13_true_short_mean_vs_a.pdf
             fig13_true_long_mean_vs_a.pdf
  资源:   fig13_oom_vs_a.pdf
  提升:   fig13_qba_promoted_vs_a.pdf（仅 QBA）
  诊断:   fig13_token_ratio_vs_a.pdf        （应随 a 单调上升）
          fig13_total_token_mean_vs_a.pdf   （应跨 a 水平，验证逐请求总 token 守恒）
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


def _policy_panels(
    out_dir: str,
    x_vals: np.ndarray,
    lambdas: np.ndarray,
    cells: dict,
    fname: str,
    *,
    ylabel: str,
    title_fmt: str = "λ = {v:g}",
    policies: list[tuple[str, str]] | None = None,
):
    """每 λ 一个子图，子图内为若干策略曲线 vs a。

    cells[policy]: [n_seeds, n_a, n_lambda] 的指标数组。
    policies: (npz 策略键, 样式键) 列表，默认全部四个主策略。
    """
    policies = policies if policies is not None else _POLICIES
    setup_matplotlib()
    n_l = len(lambdas)
    fig, axes = plt.subplots(1, n_l, figsize=(7.5 * n_l, 6), squeeze=False)
    axes = axes[0]
    ylim_series = []
    for li in range(n_l):
        ax = axes[li]
        ax.set_xscale("log", base=2)
        for policy, style_key in policies:
            st = get_policy_style(style_key)
            m, lo, hi = mean_ci(np.asarray(cells[policy], dtype=np.float64)[:, :, li])
            ax.plot(x_vals, m, color=st["color"], marker=st["marker"],
                    markersize=9, linestyle=st["linestyle"], linewidth=3.0,
                    label=display_policy_name(style_key))
            fill_ci(ax, x_vals, lo, hi, st["color"],
                    alpha=0.16 if style_key == "PSFA" else 0.18)
            ylim_series.extend([m, lo, hi])  # 纳入 CI 上下界，lo/hi 为 None 时由 auto_ylim 跳过
        apply_axes_style(
            ax, xlabel="Composition scale a", ylabel=ylabel,
            xlim=(float(np.min(x_vals)), float(np.max(x_vals))),
            ylim=None, show_legend=True,
        )
        ax.set_title(title_fmt.format(v=float(lambdas[li])), fontsize=22)
        _set_log2_ticks(ax, x_vals)
    ylim = auto_ylim(ylim_series)
    if ylim is not None:  # 各子图共享同一 y 范围，便于跨负载对比
        for ax in axes:
            ax.set_ylim(ylim)
    fig.tight_layout()
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def _work_panels(
    out_dir: str,
    x_vals: np.ndarray,
    lambdas: np.ndarray,
    runs: np.ndarray,
    fname: str,
    *,
    ylabel: str,
    title_fmt: str = "λ = {v:g}",
):
    """每 λ 一个子图，单条曲线（跨 seed 均值 + 95% CI）vs a。runs: [n_seeds, n_a, n_l]."""
    setup_matplotlib()
    n_l = len(lambdas)
    fig, axes = plt.subplots(1, n_l, figsize=(7.5 * n_l, 6), squeeze=False)
    axes = axes[0]
    ylim_series = []
    for li in range(n_l):
        ax = axes[li]
        ax.set_xscale("log", base=2)
        m, lo, hi = mean_ci(np.asarray(runs, dtype=np.float64)[:, :, li])
        ax.plot(x_vals, m, color="#333333", marker="o", markersize=9,
                linestyle="-", linewidth=3.0, label=ylabel)
        fill_ci(ax, x_vals, lo, hi, "#333333", alpha=0.15)
        ylim_series.extend([m, lo, hi])  # 纳入 CI 上下界，lo/hi 为 None 时由 auto_ylim 跳过
        apply_axes_style(
            ax, xlabel="Composition scale a", ylabel=ylabel,
            xlim=(float(np.min(x_vals)), float(np.max(x_vals))),
            ylim=None, show_legend=True,
        )
        ax.set_title(title_fmt.format(v=float(lambdas[li])), fontsize=22)
        _set_log2_ticks(ax, x_vals)
    ylim = auto_ylim(ylim_series)
    if ylim is not None:
        for ax in axes:
            ax.set_ylim(ylim)
    fig.tight_layout()
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="fig13: 跨负载构成扫描绘图")
    parser.add_argument("--data", type=str, default=os.path.join(_DEFAULT_PLOT_DIR, "fig13_data.npz"))
    parser.add_argument("--out-dir", type=str, default=_DEFAULT_PLOT_DIR)
    parser.add_argument("--prefix", type=str, default="fig13")
    parser.add_argument("--verbose-meta", action="store_true", default=False)
    args = parser.parse_args()

    data_path = os.path.abspath(args.data)
    out_dir = ensure_dir(os.path.abspath(args.out_dir))
    prefix = args.prefix

    z = np.load(data_path, allow_pickle=False)
    scales = np.asarray(z["composition_scales"], dtype=np.float64)
    lambdas = np.asarray(z["lambdas"], dtype=np.float64)

    def _read_runs(policy_prefix):
        d = {}
        for mk in ["mean", "p95", "p99", "ts_mean", "ts_p99", "tl_mean", "tl_p99",
                   "oom_prob", "throughput", "batch_util", "promoted_total"]:
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

    # ── 1-2. 整体 waiting time (mean / p99) ──
    for mk, tag in [("mean", "mean"), ("p99", "p99")]:
        cells = {p: sweep_data[p][mk] for p in sweep_data}
        print(f"saved: {_policy_panels(out_dir, scales, lambdas, cells, f'{prefix}_sim_waiting_{tag}_vs_a.pdf', ylabel='seconds')}")

    # ── 3-4. per-class mean waiting ──
    for mk, tag in [("ts_mean", "true_short_mean"), ("tl_mean", "true_long_mean")]:
        cells = {p: sweep_data[p][mk] for p in sweep_data}
        print(f"saved: {_policy_panels(out_dir, scales, lambdas, cells, f'{prefix}_{tag}_vs_a.pdf', ylabel='seconds')}")

    # ── 5. OOM 概率 ──
    if all("oom_prob" in sweep_data[p] for p in sweep_data):
        cells = {p: sweep_data[p]["oom_prob"] for p in sweep_data}
        print(f"saved: {_policy_panels(out_dir, scales, lambdas, cells, f'{prefix}_oom_vs_a.pdf', ylabel='OOM probability')}")

    # ── 6. QBA promotion 数量 ──
    if "promoted_total" in sweep_data["QBA"]:
        cells = {"QBA": sweep_data["QBA"]["promoted_total"]}
        print(f"saved: {_policy_panels(out_dir, scales, lambdas, cells, f'{prefix}_qba_promoted_vs_a.pdf', ylabel='QBA promoted (total)', policies=[('QBA', 'PSFAQ')])}")

    # ── 7. token_ratio = E[P]/E[D]（应随 a 上升） ──
    if "token_ratio_runs" in z.files:
        runs = np.asarray(z["token_ratio_runs"], dtype=np.float64)
        print(f"saved: {_work_panels(out_dir, scales, lambdas, runs, f'{prefix}_token_ratio_vs_a.pdf', ylabel='token ratio E[P]/E[D]')}")

    # ── 8. 平均总 token（应跨 a 水平） ──
    if "total_mean_runs" in z.files:
        runs = np.asarray(z["total_mean_runs"], dtype=np.float64)
        print(f"saved: {_work_panels(out_dir, scales, lambdas, runs, f'{prefix}_total_token_mean_vs_a.pdf', ylabel='mean total tokens E[P+D]')}")


if __name__ == "__main__":
    main()
