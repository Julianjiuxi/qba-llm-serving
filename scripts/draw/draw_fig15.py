"""fig15 绘图：固定安全比例下的 KV 容量敏感性与调度折中实验。

数据形状（fig15_short_data.npz / fig15_long_data.npz）:
    各策略的指标 / 诊断 / 串量 / KV / 执行 / 机制数组均为
        [n_seeds, n_a, n_lambda, n_K, n_N]；
    窗口数组为 [n_seeds, n_a, n_lambda, n_K, n_N, n_windows]（边界多一维）。
    本实验的每个阶段只有一个 N（短流 5000 或长流 100000），绘图时取该维。

图布局约定:
    网格 rows = 构成倍率 a（短流 {0.5,1,2,4} / 长流 {1,2}），cols = 到达率 λ；
    横轴 = KV 容量 K（log2）。用于回答「结论是否依赖 KV 资源约束」。

三组正文图:
    组 1 容量—等待折中:
        fig15_wait_{mean,p99}_vs_K.pdf        （rows=a, cols=λ, lines=policies, x=K）
        fig15_tl_{mean,p99}_vs_K.pdf          （true-long 等待）
        fig15_tradeoff_vs_K.pdf               （C_mean=W_QBA−W_PSF 与 G_tail=P99L,PSF−P99L,QBA）
    组 2 容量—KV 门控 / promotion 诊断:
        fig15_kv_gate_closed_rate_vs_K.pdf    （批次起始 KV ≥ 安全门的批次占比）
        fig15_kv_hard_touch_rate_vs_K.pdf     （硬容量越界批次占比，仅记录）
        fig15_kv_peak_over_capacity_vs_K.pdf  （KV 峰值 / 容量）
        fig15_promotion_ratio_vs_K.pdf        （N_prom / N_all）
        fig15_phi_Q_vs_K.pdf                  （N_prom / N_pred_long）
        fig15_long_admit_normal_vs_K.pdf      （唯一普通准入 predicted-long 数）
    组 3 长流窗口积压对照（长流数据）:
        fig15_backlog_time_vs_K.pdf           （时间口径累计积压，rows=a, cols=λ, lines=K）
        fig15_queue_len_time_vs_K.pdf
        fig15_unfinished_at_arrival_stop_vs_K.pdf
        fig15_drain_time_vs_K.pdf
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
    auto_ylim,
    mean_ci,
    display_policy_name,
    get_policy_style,
    meta_summary,
    fill_ci,
    _DEFAULT_PLOT_DIR,
)
from .draw_fig14 import _style_panel, _set_log2_ticks

_POLICY_PREFIX = {"FCFS": "fcfs", "PSF": "psf", "AA": "psfa", "QBA": "psfaq"}
# (npz 策略键, 样式键)
_POLICY_STYLE = [("FCFS", "FCFS"), ("PSF", "PSF"), ("AA", "PSFA"), ("QBA", "PSFAQ")]

_TITLE_FS = 16


# ── 读取 ────────────────────────────────────────────────────

def _read_5d(z, pfx: str, key: str):
    """读取 [n_seeds, n_a, n_l, n_K, n_N]，去掉唯一的 N 维 → [n_seeds, n_a, n_l, n_K]。"""
    k = f"{pfx}_{key}_runs"
    if k not in z.files:
        return None
    arr = np.asarray(z[k], dtype=np.float64)
    if arr.ndim == 5 and arr.shape[-1] == 1:
        return arr[..., 0]
    return arr


def _read_window_5d(z, pfx: str, mode: str, wk: str):
    """读取窗口 [n_seeds, n_a, n_l, n_K, n_N, n_w] → [n_seeds, n_a, n_l, n_K, n_w]。"""
    k = f"{pfx}_{mode}_{wk}_runs"
    if k not in z.files:
        return None
    arr = np.asarray(z[k], dtype=np.float64)
    if arr.ndim == 6 and arr.shape[4] == 1:
        return arr[:, :, :, :, 0, :]
    return arr


# ── 绘图：x=K ───────────────────────────────────────────────

def _grid_vs_K_policies(out_dir: str, a_vals, lambdas, caps, cells: dict,
                        fname: str, *, ylabel: str, ylim=None):
    """rows=a, cols=λ；横轴 K（log2）；每条策略一条曲线。cells[pol]=[n_seeds,n_a,n_l,n_K]。"""
    n_a, n_l = len(a_vals), len(lambdas)
    setup_matplotlib()
    fig, axes = plt.subplots(n_a, n_l, figsize=(3.7 * n_l, 3.4 * n_a),
                             squeeze=False, sharex=True)
    ylim_series = []
    for ai in range(n_a):
        for li in range(n_l):
            ax = axes[ai][li]
            ax.set_xscale("log", base=2)
            for pol, style_key in _POLICY_STYLE:
                arr = cells.get(pol)
                if arr is None:
                    continue
                m, lo, hi = mean_ci(arr[:, ai, li, :])  # [n_seeds, n_K]
                st = get_policy_style(style_key)
                ax.plot(caps, m, color=st["color"], marker=st["marker"],
                        markersize=6, linestyle=st["linestyle"], linewidth=2.2,
                        label=display_policy_name(style_key))
                fill_ci(ax, caps, lo, hi, st["color"], alpha=0.16)
                ylim_series.extend([m, lo, hi])
            _style_panel(ax, xlabel=("KV capacity K (tokens)" if ai == n_a - 1 else ""),
                         ylabel=(f"a={float(a_vals[ai]):g}\n{ylabel}" if li == 0 else ""),
                         legend=(ai == 0 and li == 0), ylim=ylim)
            _set_log2_ticks(ax, caps, as_int=True)
            if ai == 0:
                ax.set_title(f"λ={float(lambdas[li]):g}", fontsize=_TITLE_FS)
    if ylim is None:
        ylim = auto_ylim(ylim_series)
        if ylim is not None:
            for ai in range(n_a):
                for li in range(n_l):
                    axes[ai][li].set_ylim(ylim)
    fig.tight_layout()
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def _grid_vs_K_series(out_dir: str, a_vals, lambdas, caps, series: dict,
                      fname: str, *, ylabel: str, ylim=None):
    """rows=a, cols=λ；横轴 K（log2）；`series[name]=(color, marker, label, 4D数组)`。"""
    n_a, n_l = len(a_vals), len(lambdas)
    setup_matplotlib()
    fig, axes = plt.subplots(n_a, n_l, figsize=(3.7 * n_l, 3.4 * n_a),
                             squeeze=False, sharex=True)
    ylim_series = []
    for ai in range(n_a):
        for li in range(n_l):
            ax = axes[ai][li]
            ax.set_xscale("log", base=2)
            for name, (color, marker, label, arr) in series.items():
                arr = np.asarray(arr, dtype=np.float64)
                m, lo, hi = mean_ci(arr[:, ai, li, :])  # [n_seeds, n_K]
                ax.plot(caps, m, color=color, marker=marker, markersize=6,
                        linestyle="-", linewidth=2.2, label=label)
                fill_ci(ax, caps, lo, hi, color, alpha=0.16)
                ylim_series.extend([m, lo, hi])
            _style_panel(ax, xlabel=("KV capacity K (tokens)" if ai == n_a - 1 else ""),
                         ylabel=(f"a={float(a_vals[ai]):g}\n{ylabel}" if li == 0 else ""),
                         legend=(ai == 0 and li == 0), ylim=ylim)
            _set_log2_ticks(ax, caps, as_int=True)
            if ai == 0:
                ax.set_title(f"λ={float(lambdas[li]):g}", fontsize=_TITLE_FS)
    if ylim is None:
        ylim = auto_ylim(ylim_series)
        if ylim is not None:
            for ai in range(n_a):
                for li in range(n_l):
                    axes[ai][li].set_ylim(ylim)
    fig.tight_layout()
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def _window_by_K(out_dir: str, a_vals, lambdas, caps, arr5, fname: str, *,
                 ylabel: str):
    """rows=a, cols=λ；横轴窗口序号；每条 K 一条曲线。arr5=[n_seeds,n_a,n_l,n_K,n_w]。"""
    n_a, n_l = len(a_vals), len(lambdas)
    arr5 = np.asarray(arr5, dtype=np.float64)
    n_K, n_w = int(arr5.shape[3]), int(arr5.shape[4])
    setup_matplotlib()
    fig, axes = plt.subplots(n_a, n_l, figsize=(3.7 * n_l, 3.4 * n_a),
                             squeeze=False, sharex=True)
    x = np.arange(n_w)
    cmap = plt.get_cmap("viridis")
    ylim_series = []
    for ai in range(n_a):
        for li in range(n_l):
            ax = axes[ai][li]
            for ki in range(n_K):
                m, lo, hi = mean_ci(arr5[:, ai, li, ki, :])  # [n_seeds, n_w]
                color = cmap(0.15 + 0.7 * ki / max(1, n_K - 1))
                ax.plot(x, m, color=color, marker="o", markersize=5, linewidth=2.2,
                        label=f"K={int(caps[ki])}")
                fill_ci(ax, x, lo, hi, color, alpha=0.14)
                ylim_series.extend([m, lo, hi])
            _style_panel(ax, xlabel=("window index (equal time)" if ai == n_a - 1 else ""),
                         ylabel=(f"a={float(a_vals[ai]):g}\n{ylabel}" if li == 0 else ""),
                         legend=(ai == 0 and li == 0))
            if ai == 0:
                ax.set_title(f"λ={float(lambdas[li]):g}", fontsize=_TITLE_FS)
    ylim = auto_ylim(ylim_series)
    if ylim is not None:
        for ai in range(n_a):
            for li in range(n_l):
                axes[ai][li].set_ylim(ylim)
    fig.tight_layout()
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── 三组图 ──────────────────────────────────────────────────

def _group1_tradeoff(out_dir, prefix, scales, lambdas, caps, runs):
    """组 1：容量—等待折中曲线（每策略 + QBA−PSF 折中）。"""
    for mk, tag, ylabel in [
        ("mean", "wait_mean", "mean waiting (s)"),
        ("p99", "wait_p99", "p99 waiting (s)"),
        ("tl_mean", "tl_mean", "true-long mean waiting (s)"),
        ("tl_p99", "tl_p99", "true-long p99 waiting (s)"),
    ]:
        cells = {p: runs[p].get(mk) for p in runs
                 if runs[p].get(mk) is not None}
        if cells:
            print(f"saved: {_grid_vs_K_policies(out_dir, scales, lambdas, caps, cells, f'{prefix}_{tag}_vs_K.pdf', ylabel=ylabel)}")

    qba_mean, psf_mean = runs["QBA"].get("mean"), runs["PSF"].get("mean")
    qba_tlp99, psf_tlp99 = runs["QBA"].get("tl_p99"), runs["PSF"].get("tl_p99")
    if qba_mean is not None and psf_mean is not None:
        c_mean = qba_mean - psf_mean  # 平均等待代价 W_QBA − W_PSF
        series = {"C_mean": ("#8F3FD6", "o", "C_mean = W_QBA − W_PSF (s)", c_mean)}
        if qba_tlp99 is not None and psf_tlp99 is not None:
            g_tail = psf_tlp99 - qba_tlp99  # 长尾收益 P99L,PSF − P99L,QBA
            series["G_tail"] = ("#00B050", "^", "G_tail = P99L,PSF − P99L,QBA (s)", g_tail)
        print(f"saved: {_grid_vs_K_series(out_dir, scales, lambdas, caps, series, f'{prefix}_tradeoff_vs_K.pdf', ylabel='QBA vs PSF tradeoff (s)')}")


def _group2_kv_promotion(out_dir, prefix, scales, lambdas, caps, runs):
    """组 2：容量—KV 门控 / promotion 诊断。"""
    for key, tag, ylabel, ylim in [
        ("kv_gate_closed_rate", "kv_gate_closed_rate",
         "KV safety-gate closed batch fraction", (0.0, 1.0)),
        ("kv_hard_touch_rate", "kv_hard_touch_rate",
         "hard-capacity touched batch fraction (recorded only)", (0.0, 1.0)),
        ("kv_peak_over_capacity_mean", "kv_peak_over_capacity",
         "KV peak / capacity (mean)", None),
        ("promotion_ratio", "promotion_ratio",
         "promotion ratio (N_prom / N_all)", (0.0, 1.0)),
        ("phi_Q", "phi_Q", "phi_Q (N_prom / N_pred_long)", (0.0, 1.0)),
        ("n_pred_long_normal", "long_admit_normal",
         "unique normal admits of predicted-long", None),
    ]:
        cells = {p: runs[p].get(key) for p in runs if runs[p].get(key) is not None}
        if cells:
            print(f"saved: {_grid_vs_K_policies(out_dir, scales, lambdas, caps, cells, f'{prefix}_{tag}_vs_K.pdf', ylabel=ylabel, ylim=ylim)}")


def _group3_long_windows(out_dir, prefix, z, scales, lambdas, caps, runs):
    """组 3：长流窗口积压对照（时间口径），rows=a, cols=λ, lines=K。"""
    for wk, tag, ylabel in [
        ("backlog", "backlog_time", "cumulative backlog (arrivals − completions)"),
        ("queue_len_mean", "queue_len_time", "mean wait queue length"),
    ]:
        arr = _read_window_5d(z, "psfaq", "time", wk)
        if arr is not None:
            print(f"saved: {_window_by_K(out_dir, scales, lambdas, caps, arr, f'{prefix}_{tag}_vs_K.pdf', ylabel=ylabel)}")

    # 长流标量（到达停止时未完成数 / drain time）——rows=a, cols=λ, lines=policies
    for key, tag, ylabel in [
        ("unfinished_at_arrival_stop", "unfinished_at_arrival_stop",
         "unfinished requests at arrival stop"),
        ("drain_time", "drain_time", "drain time (s)"),
    ]:
        cells = {p: runs[p].get(key) for p in runs if runs[p].get(key) is not None}
        if cells:
            print(f"saved: {_grid_vs_K_policies(out_dir, scales, lambdas, caps, cells, f'{prefix}_{tag}_vs_K.pdf', ylabel=ylabel)}")


def _load(path: str):
    if not path or not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=False)


def _read_runs(z) -> dict:
    runs = {}
    for p, pfx in _POLICY_PREFIX.items():
        pk = "psfaq" if p == "QBA" else pfx
        entry: dict = {}
        keys = [
            "mean", "p99", "tl_mean", "tl_p99",
            "kv_gate_closed_rate", "kv_hard_touch_rate",
            "kv_peak_over_capacity_mean",
            "promotion_ratio", "phi_Q", "n_pred_long_normal",
            "unfinished_at_arrival_stop", "drain_time",
        ]
        for k in keys:
            arr = _read_5d(z, pk, k)
            if arr is not None:
                entry[k] = arr
        runs[p] = entry
    return runs


def main():
    parser = argparse.ArgumentParser(description="fig15: KV 容量敏感性绘图")
    parser.add_argument("--data-short", type=str,
                        default=os.path.join(_DEFAULT_PLOT_DIR, "fig15_short_data.npz"))
    parser.add_argument("--data-long", type=str,
                        default=os.path.join(_DEFAULT_PLOT_DIR, "fig15_long_data.npz"))
    parser.add_argument("--out-dir", type=str, default=_DEFAULT_PLOT_DIR)
    parser.add_argument("--prefix", type=str, default="fig15")
    parser.add_argument("--verbose-meta", action="store_true", default=False)
    args = parser.parse_args()

    out_dir = ensure_dir(os.path.abspath(args.out_dir))
    prefix = args.prefix

    z_short = _load(os.path.abspath(args.data_short))
    z_long = _load(os.path.abspath(args.data_long))

    # 组 1 / 组 2 以短流为主（四档容量），缺失时回退长流
    z_main = z_short if z_short is not None else z_long
    if z_main is not None:
        src_path = (os.path.abspath(args.data_short) if z_short is not None
                    else os.path.abspath(args.data_long))
        meta_path = os.path.join(os.path.dirname(src_path), f"{prefix}_{'short' if z_short is not None else 'long'}_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            print("meta_summary", meta if args.verbose_meta else meta_summary(meta))
        scales = np.asarray(z_main["composition_scales"], dtype=np.float64)
        lambdas = np.asarray(z_main["lambdas"], dtype=np.float64)
        caps = np.asarray(z_main["capacities"], dtype=np.float64)
        runs = _read_runs(z_main)
        _group1_tradeoff(out_dir, prefix, scales, lambdas, caps, runs)
        _group2_kv_promotion(out_dir, prefix, scales, lambdas, caps, runs)
    else:
        print("skip group 1-2: no short/long data found")

    # 组 3 用长流数据
    if z_long is not None:
        scales_l = np.asarray(z_long["composition_scales"], dtype=np.float64)
        lambdas_l = np.asarray(z_long["lambdas"], dtype=np.float64)
        caps_l = np.asarray(z_long["capacities"], dtype=np.float64)
        runs_l = _read_runs(z_long)
        _group3_long_windows(out_dir, prefix, z_long, scales_l, lambdas_l, caps_l, runs_l)
    else:
        print("skip group 3: no long data found")


if __name__ == "__main__":
    main()
