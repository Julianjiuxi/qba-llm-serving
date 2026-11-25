"""fig14 绘图：长请求流下的负载—构成实验（Tier1）+ 机制分解（Tier2）。

数据形状（fig14_data.npz）:
    每个策略的指标/诊断/串量/机制数组均为 [n_seeds, n_a, n_lambda, n_N]；
    双口径窗口数组为 [n_seeds, n_a, n_lambda, n_N, n_windows]；
    变换诊断（transform_*）为 [n_seeds, n_a, n_lambda]；前缀诊断（prefix_*）为 4D。

图布局约定:
    网格 rows = 构成倍率 a，cols = 到达率 λ；横轴 = 请求流长度 N（log2）。
    用于直接回答「结论是否依赖运行长度 N」。

主要输出:
    等待 / 资源:
        fig14_wait_{mean,p99}_vs_N.pdf        （rows=a, cols=λ, lines=policies, x=N）
        fig14_tl_mean_vs_N.pdf                （true-long 平均等待）
    串量标量 vs N（积压/提升双口径）:
        fig14_drain_time_vs_N.pdf             （t_last_completion - t_last_arrival）
        fig14_unfinished_at_arrival_stop_vs_N.pdf  （未完成数，与窗口 backlog 一致，主图）
        fig14_unfinished_fraction_vs_N.pdf
        fig14_waiting_at_arrival_stop_vs_N.pdf     （未开始服务数）
        fig14_promotion_ratio_vs_N.pdf        （N_prom / N_all）
        fig14_phi_Q_vs_N.pdf                  （N_prom / N_pred_long，同 fig12 口径）
    窗口（时间口径覆盖 [0, t_last_arrival]，不含排空；关键：稳定 vs 积压增长）:
        fig14_queue_len_time.pdf              （rows=a, cols=λ, lines=N, x=窗口序号）
        fig14_backlog_time.pdf
    机制分解（QBA，固定最大 N）:
        fig14_mech_debt_buckets_vs_a.pdf      （no_eligible/candidate_blocked/success/kv_gate_closed 四桶占比）
        fig14_mech_eta_vs_a.pdf               （η_eligibility / η_feasibility）
        fig14_mech_long_admit_vs_a.pdf        （normal / promoted / eligible_at_entry）
        fig14_mech_breaks_vs_a.pdf            （budget / kv / concurrency 阻塞）
    机制 k 网格（QBA，固定 N=mech_n，横轴 quota_k，rows=a, cols=λ）:
        fig14_mechk_eta_vs_k.pdf              （η_eligibility / η_feasibility）
        fig14_mechk_debt_buckets_vs_k.pdf     （debt 批四桶占比分解）
        fig14_mechk_wait_vs_k.pdf             （mean / p99 等待）
        fig14_mechk_promotion_vs_k.pdf        （promotion_ratio 与 phi_Q）
    工作量诊断:
        fig14_token_ratio_vs_a.pdf            （应随 a 上升）
        fig14_total_token_mean_vs_a.pdf       （应跨 a 水平，验证逐请求守恒）
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

# (npz 策略键, 样式键)；RR 为内部策略，不绘图
_POLICIES = [("FCFS", "FCFS"), ("PSF", "PSF"), ("AA", "PSFA"), ("QBA", "PSFAQ")]
_POLICY_PREFIX = {"FCFS": "fcfs", "PSF": "psf", "AA": "psfa", "QBA": "psfaq"}

_METRIC_KEYS = ["mean", "p95", "p99", "ts_mean", "ts_p99", "tl_mean", "tl_p99",
                "oom_prob", "throughput", "batch_util", "promoted_total"]
_STREAM_KEYS = ["t_last_arrival", "t_last_completion", "drain_time",
                "waiting_at_arrival_stop", "unfinished_at_arrival_stop",
                "unfinished_fraction", "promotion_ratio", "phi_Q"]
_MECH_KEYS = [
    "mech_debt_episode_count", "mech_debt_episode_mean_duration",
    "mech_debt_episode_p99_duration", "mech_debt_episode_max_duration",
    "mech_debt_unresolved_at_end", "mech_debt_unresolved_duration",
    "mech_promotion_success_count",
    "mech_debt_batch_no_eligible", "mech_debt_batch_candidate_blocked",
    "mech_debt_batch_success", "mech_debt_batch_kv_gate_closed",
    "mech_eta_eligibility", "mech_eta_feasibility",
    "mech_long_admit_normal", "mech_long_admit_promoted",
    "mech_long_admit_eligible_at_entry",
    "mech_budget_break_count", "mech_kv_break_count",
    "mech_concurrency_break_count", "mech_batch_count",
]
_WINDOW_KEYS = ["wait_mean", "wait_p99", "count", "arrivals", "completions",
                "backlog", "queue_len_mean", "promotion_ratio", "phi_Q"]

_PANEL_FS = 15      # 面板轴标签字号
_TICK_FS = 13       # 面板刻度字号
_TITLE_FS = 16      # 面板标题字号


# ── 读取 ────────────────────────────────────────────────────

def _read_policy(z, pfx: str) -> dict:
    """读取某策略的所有 4D 数组（指标 + 串量 + 机制）。"""
    out: dict = {}
    for mk in _METRIC_KEYS + _STREAM_KEYS + _MECH_KEYS:
        key = f"{pfx}_{mk}_runs"
        if key in z.files:
            out[mk] = np.asarray(z[key], dtype=np.float64)
    return out


def _read_window(z, pfx: str, mode: str) -> dict:
    """读取某策略某口径的所有窗口数组 [n_seeds, n_a, n_l, n_N, n_w]。"""
    out: dict = {}
    for wk in _WINDOW_KEYS:
        key = f"{pfx}_{mode}_{wk}_runs"
        if key in z.files:
            out[wk] = np.asarray(z[key], dtype=np.float64)
    return out


# ── 面板样式 ────────────────────────────────────────────────

def _style_panel(ax, *, xlabel: str = "", ylabel: str = "",
                 xlim=None, ylim=None, legend: bool = False):
    ax.set_xlabel(xlabel, fontsize=_PANEL_FS)
    ax.set_ylabel(ylabel, fontsize=_PANEL_FS)
    ax.tick_params(axis="both", labelsize=_TICK_FS)
    ax.grid(color="#D8D8D8", linestyle="--", linewidth=0.7, alpha=0.85)
    for spine in ax.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(1.1)
    if xlim is not None:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            legend_obj = ax.legend(loc="best", fontsize=_TICK_FS, frameon=True,
                                   fancybox=True, framealpha=0.92, ncol=1,
                                   handlelength=1.8)
            legend_obj.get_frame().set_edgecolor((0, 0, 0, 0.2))
            legend_obj.get_frame().set_linewidth(0.8)


def _set_log2_ticks(ax, vals, as_int: bool = False):
    ticks = [float(v) for v in vals]
    ax.set_xticks(ticks)
    labels = [f"{int(round(v))}" if as_int else f"{v:g}" for v in ticks]
    ax.set_xticklabels(labels)
    ax.minorticks_off()


def _grid_vs_n(out_dir: str, a_vals, lambdas, lengths, cells: dict, fname: str, *,
               ylabel: str, policies=None, xlabel: str = "stream length N",
               title_by_col: bool = True):
    """rows=a, cols=λ；横轴 N（log2）；每条策略一条曲线。cells[pol]: 4D 数组。"""
    policies = policies if policies is not None else _POLICIES
    n_a, n_l = len(a_vals), len(lambdas)
    setup_matplotlib()
    fig, axes = plt.subplots(n_a, n_l, figsize=(3.7 * n_l, 3.4 * n_a),
                             squeeze=False, sharex=True)
    ylim_series = []
    for ai in range(n_a):
        for li in range(n_l):
            ax = axes[ai][li]
            ax.set_xscale("log", base=2)
            for pol, style_key in policies:
                arr = np.asarray(cells[pol], dtype=np.float64)
                if arr.ndim != 4:
                    continue
                m, lo, hi = mean_ci(arr[:, ai, li, :])  # [n_seeds, n_N]
                st = get_policy_style(style_key)
                ax.plot(lengths, m, color=st["color"], marker=st["marker"],
                        markersize=6, linestyle=st["linestyle"], linewidth=2.2,
                        label=display_policy_name(style_key))
                fill_ci(ax, lengths, lo, hi, st["color"], alpha=0.16)
                ylim_series.extend([m, lo, hi])
            _style_panel(ax, xlabel=(xlabel if ai == n_a - 1 else ""),
                         ylabel=(ylabel if li == 0 else ""),
                         legend=(ai == 0 and li == 0))
            _set_log2_ticks(ax, lengths, as_int=True)
            if title_by_col and ai == 0:
                ax.set_title(f"λ={float(lambdas[li]):g}", fontsize=_TITLE_FS)
            if (not title_by_col) and li == n_l - 1:
                ax.text(1.03, 0.5, f"a={float(a_vals[ai]):g}", transform=ax.transAxes,
                        rotation=270, va="center", fontsize=_PANEL_FS)
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


def _grid_vs_a(out_dir: str, a_vals, lambdas, series: dict, fname: str, *,
               ylabel: str, xlabel: str = "composition scale a", ylim=None):
    """rows=1, cols=λ；横轴 a（log2）；`series[name]` = (color, marker, label, 4D数组)。"""
    n_l = len(lambdas)
    setup_matplotlib()
    fig, axes = plt.subplots(1, n_l, figsize=(3.7 * n_l, 3.6), squeeze=False)
    axes = axes[0]
    ylim_series = []
    for li in range(n_l):
        ax = axes[li]
        ax.set_xscale("log", base=2)
        for name, (color, marker, label, arr) in series.items():
            arr = np.asarray(arr, dtype=np.float64)
            m, lo, hi = mean_ci(arr[:, :, li])  # [n_seeds, n_a]
            ax.plot(a_vals, m, color=color, marker=marker, markersize=7,
                    linestyle="-", linewidth=2.4, label=label)
            fill_ci(ax, a_vals, lo, hi, color, alpha=0.16)
            ylim_series.extend([m, lo, hi])
        _style_panel(ax, xlabel=xlabel, ylabel=(ylabel if li == 0 else ""),
                     legend=(li == 0), ylim=ylim)
        _set_log2_ticks(ax, a_vals)
        ax.set_title(f"λ={float(lambdas[li]):g}", fontsize=_TITLE_FS)
    if ylim is None:
        ylim = auto_ylim(ylim_series)
        if ylim is not None:
            for ax in axes:
                ax.set_ylim(ylim)
    fig.tight_layout()
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def _grid_vs_k(out_dir: str, a_vals, lambdas, ks, series: dict, fname: str, *,
               ylabel: str, ylim=None):
    """rows=a, cols=λ；横轴 quota_k（log2）；series[name]=4D [n_seeds,n_a,n_l,n_k]。"""
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
                m, lo, hi = mean_ci(arr[:, ai, li, :])  # [n_seeds, n_k]
                ax.plot(ks, m, color=color, marker=marker, markersize=6,
                        linestyle="-", linewidth=2.2, label=label)
                fill_ci(ax, ks, lo, hi, color, alpha=0.16)
                ylim_series.extend([m, lo, hi])
            _style_panel(ax, xlabel=("quota_k" if ai == n_a - 1 else ""),
                         ylabel=(ylabel if li == 0 else ""),
                         legend=(ai == 0 and li == 0), ylim=ylim)
            _set_log2_ticks(ax, ks, as_int=True)
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


def _window_panels(out_dir: str, a_vals, lambdas, lengths, arr5, fname: str, *,
                   ylabel: str):
    """rows=a, cols=λ；横轴窗口序号；每条 N 一条曲线（时间口径窗口）。

    arr5: [n_seeds, n_a, n_l, n_N, n_w]。
    """
    n_a, n_l = len(a_vals), len(lambdas)
    arr5 = np.asarray(arr5, dtype=np.float64)
    n_w = int(arr5.shape[-1])
    setup_matplotlib()
    fig, axes = plt.subplots(n_a, n_l, figsize=(3.7 * n_l, 3.4 * n_a),
                             squeeze=False, sharex=True)
    x = np.arange(n_w)
    cmap = plt.get_cmap("viridis")
    ylim_series = []
    for ai in range(n_a):
        for li in range(n_l):
            ax = axes[ai][li]
            for ni in range(len(lengths)):
                m, lo, hi = mean_ci(arr5[:, ai, li, ni, :])  # [n_seeds, n_w]
                color = cmap(0.15 + 0.7 * ni / max(1, len(lengths) - 1))
                ax.plot(x, m, color=color, marker="o", markersize=5, linewidth=2.2,
                        label=f"N={int(lengths[ni])}")
                fill_ci(ax, x, lo, hi, color, alpha=0.14)
                ylim_series.extend([m, lo, hi])
            _style_panel(ax, xlabel=("window index (equal time)" if ai == n_a - 1 else ""),
                         ylabel=(ylabel if li == 0 else ""),
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


# ── 主流程 ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="fig14: 长流负载—构成绘图 + 机制分解")
    parser.add_argument("--data", type=str,
                        default=os.path.join(_DEFAULT_PLOT_DIR, "fig14_data.npz"))
    parser.add_argument("--out-dir", type=str, default=_DEFAULT_PLOT_DIR)
    parser.add_argument("--prefix", type=str, default="fig14")
    parser.add_argument("--verbose-meta", action="store_true", default=False)
    args = parser.parse_args()

    data_path = os.path.abspath(args.data)
    out_dir = ensure_dir(os.path.abspath(args.out_dir))
    prefix = args.prefix

    z = np.load(data_path, allow_pickle=False)
    scales = np.asarray(z["composition_scales"], dtype=np.float64)
    lambdas = np.asarray(z["lambdas"], dtype=np.float64)
    lengths = np.asarray(z["lengths"], dtype=np.float64)

    runs = {p: _read_policy(z, pfx) for p, pfx in _POLICY_PREFIX.items()}

    meta_path = os.path.join(os.path.dirname(data_path), f"{prefix}_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print("meta_summary", meta if args.verbose_meta else meta_summary(meta))

    # ── 1-3. 等待 / 资源 vs N ──
    for mk, tag, ylabel in [
        ("mean", "wait_mean", "mean waiting (s)"),
        ("p99", "wait_p99", "p99 waiting (s)"),
        ("tl_mean", "tl_mean", "true-long mean waiting (s)"),
    ]:
        cells = {p: runs[p][mk] for p in runs if mk in runs[p]}
        if cells:
            print(f"saved: {_grid_vs_n(out_dir, scales, lambdas, lengths, cells, f'{prefix}_{tag}_vs_N.pdf', ylabel=ylabel)}")

    # ── 4-9. 串量标量 vs N ──
    # 积压双口径：unfinished_* 与时间窗口 backlog 一致（主图）；waiting_* 仅「未开始服务」
    # 提升双口径：promotion_ratio=N_prom/N_all；phi_Q=N_prom/N_pred_long（同 fig12）
    for sk, tag, ylabel in [
        ("drain_time", "drain_time", "drain time (s)"),
        ("unfinished_at_arrival_stop", "unfinished_at_arrival_stop",
         "unfinished requests at arrival stop"),
        ("unfinished_fraction", "unfinished_fraction",
         "unfinished fraction at arrival stop (matches window backlog)"),
        ("waiting_at_arrival_stop", "waiting_at_arrival_stop",
         "waiting (not started) requests at arrival stop"),
        ("promotion_ratio", "promotion_ratio", "promotion ratio (N_prom / N_all)"),
        ("phi_Q", "phi_Q", "phi_Q (N_prom / N_pred_long, fig12 scale)"),
    ]:
        cells = {p: runs[p][sk] for p in runs if sk in runs[p]}
        if cells:
            print(f"saved: {_grid_vs_n(out_dir, scales, lambdas, lengths, cells, f'{prefix}_{tag}_vs_N.pdf', ylabel=ylabel)}")

    # ── 7. 资源诊断 vs N（OOM）──
    cells = {p: runs[p]["oom_prob"] for p in runs if "oom_prob" in runs[p]}
    if cells:
        print(f"saved: {_grid_vs_n(out_dir, scales, lambdas, lengths, cells, f'{prefix}_oom_vs_N.pdf', ylabel='OOM probability')}")

    # ── 8-9. 窗口时间口径（稳定 vs 积压增长）──
    for pfx in ["psfaq"]:
        win = _read_window(z, pfx, "time")
        if "queue_len_mean" in win:
            print(f"saved: {_window_panels(out_dir, scales, lambdas, lengths, win['queue_len_mean'], f'{prefix}_queue_len_time.pdf', ylabel='mean wait queue length')}")
        if "backlog" in win:
            print(f"saved: {_window_panels(out_dir, scales, lambdas, lengths, win['backlog'], f'{prefix}_backlog_time.pdf', ylabel='cumulative backlog (arrivals - completions)')}")

    # ── 10-13. 机制分解（QBA，固定最大 N）──
    qba = runs["QBA"]
    ni_fix = len(lengths) - 1
    if "mech_debt_batch_no_eligible" in qba:
        ne = qba["mech_debt_batch_no_eligible"][:, :, :, ni_fix]
        cb = qba["mech_debt_batch_candidate_blocked"][:, :, :, ni_fix]
        sc = qba["mech_debt_batch_success"][:, :, :, ni_fix]
        kv = qba["mech_debt_batch_kv_gate_closed"][:, :, :, ni_fix]
        # 完整债务批分解：四类互斥（no_eligible/candidate_blocked/success 仅在 KV 门开启
        # 且本批调度出 batch 时计数；kv_gate_closed 属门关闭分支，无候选集检查）
        tot = ne + cb + sc + kv
        frac = lambda x: np.divide(x, tot, out=np.full_like(x, np.nan), where=tot > 0)
        series = {
            "no_eligible": ("#1F77FF", "o", "no aged-long candidate", frac(ne)),
            "candidate_blocked": ("#FF8C1A", "D", "candidate blocked", frac(cb)),
            "success": ("#00B050", "^", "promotion success", frac(sc)),
            "kv_gate_closed": ("#E63946", "s", "KV gate closed", frac(kv)),
        }
        print(f"saved: {_grid_vs_a(out_dir, scales, lambdas, series, f'{prefix}_mech_debt_buckets_vs_a.pdf', ylabel='fraction of debt batches', ylim=(0.0, 1.0))}")

    if "mech_eta_eligibility" in qba:
        eta_e = qba["mech_eta_eligibility"][:, :, :, ni_fix]
        eta_f = qba["mech_eta_feasibility"][:, :, :, ni_fix]
        series = {
            "eta_e": ("#8F3FD6", "o", "η_eligibility", eta_e),
            "eta_f": ("#E63946", "s", "η_feasibility", eta_f),
        }
        print(f"saved: {_grid_vs_a(out_dir, scales, lambdas, series, f'{prefix}_mech_eta_vs_a.pdf', ylabel='η', ylim=(0.0, 1.0))}")

    if "mech_long_admit_normal" in qba:
        ln = qba["mech_long_admit_normal"][:, :, :, ni_fix]
        lp = qba["mech_long_admit_promoted"][:, :, :, ni_fix]
        le = qba["mech_long_admit_eligible_at_entry"][:, :, :, ni_fix]
        series = {
            "normal": ("#1F77FF", "o", "normal admit", ln),
            "promoted": ("#00B050", "^", "promoted admit", lp),
            "eligible_at_entry": ("#FF8C1A", "D", "normal admit & eligible", le),
        }
        print(f"saved: {_grid_vs_a(out_dir, scales, lambdas, series, f'{prefix}_mech_long_admit_vs_a.pdf', ylabel='predicted-long count')}")

    if "mech_budget_break_count" in qba:
        budget = qba["mech_budget_break_count"][:, :, :, ni_fix]
        kv = qba["mech_kv_break_count"][:, :, :, ni_fix]
        conc = qba["mech_concurrency_break_count"][:, :, :, ni_fix]
        batches = qba["mech_batch_count"][:, :, :, ni_fix]
        per_batch = lambda x: np.divide(x, batches, out=np.full_like(x, np.nan),
                                        where=batches > 0)
        series = {
            "budget": ("#1F77FF", "o", "no prefill budget", per_batch(budget)),
            "kv": ("#FF8C1A", "D", "KV gate", per_batch(kv)),
            "concurrency": ("#E63946", "s", "max concurrency", per_batch(conc)),
        }
        print(f"saved: {_grid_vs_a(out_dir, scales, lambdas, series, f'{prefix}_mech_breaks_vs_a.pdf', ylabel='breaks per batch')}")

    # ── 14. 机制 k 网格（QBA，固定 N=mech_n，扫 quota_k）──
    if "mech_ks" in z.files and "mechk_mech_eta_eligibility" in z.files:
        ks = np.asarray(z["mech_ks"], dtype=np.float64)
        mech_n_val = int(np.asarray(z["mech_n"]).reshape(-1)[0])

        def g(name: str):
            return np.asarray(z[f"mechk_{name}"], dtype=np.float64)

        print(f"mechk grid: N={mech_n_val} ks={ks.astype(int).tolist()}")

        series = {
            "eta_e": ("#8F3FD6", "o", "η_eligibility", g("mech_eta_eligibility")),
            "eta_f": ("#E63946", "s", "η_feasibility", g("mech_eta_feasibility")),
        }
        print(f"saved: {_grid_vs_k(out_dir, scales, lambdas, ks, series, f'{prefix}_mechk_eta_vs_k.pdf', ylabel='η', ylim=(0.0, 1.0))}")

        ne = g("mech_debt_batch_no_eligible")
        cb = g("mech_debt_batch_candidate_blocked")
        sc = g("mech_debt_batch_success")
        kv = g("mech_debt_batch_kv_gate_closed")
        tot = ne + cb + sc + kv
        frac = lambda x: np.divide(x, tot, out=np.full_like(x, np.nan), where=tot > 0)
        series = {
            "no_eligible": ("#1F77FF", "o", "no aged-long candidate", frac(ne)),
            "candidate_blocked": ("#FF8C1A", "D", "candidate blocked", frac(cb)),
            "success": ("#00B050", "^", "promotion success", frac(sc)),
            "kv_gate_closed": ("#E63946", "s", "KV gate closed", frac(kv)),
        }
        print(f"saved: {_grid_vs_k(out_dir, scales, lambdas, ks, series, f'{prefix}_mechk_debt_buckets_vs_k.pdf', ylabel='fraction of debt batches', ylim=(0.0, 1.0))}")

        series = {
            "mean": ("#1F77FF", "o", "mean waiting", g("mean")),
            "p99": ("#FF8C1A", "D", "p99 waiting", g("p99")),
        }
        print(f"saved: {_grid_vs_k(out_dir, scales, lambdas, ks, series, f'{prefix}_mechk_wait_vs_k.pdf', ylabel='waiting (s)')}")

        series = {
            "promotion_ratio": ("#00B050", "^", "promotion ratio (N_prom/N_all)", g("promotion_ratio")),
            "phi_Q": ("#1F77FF", "o", "phi_Q (N_prom/N_pred_long)", g("phi_Q")),
        }
        print(f"saved: {_grid_vs_k(out_dir, scales, lambdas, ks, series, f'{prefix}_mechk_promotion_vs_k.pdf', ylabel='promotion ratio', ylim=(0.0, 1.0))}")

    # ── 15-16. 工作量诊断 vs a（守恒验证）──
    if "transform_token_ratio" in z.files:
        runs_tr = np.asarray(z["transform_token_ratio"], dtype=np.float64)
        series = {"token_ratio": ("#333333", "o", "E[P]/E[D]", runs_tr)}
        print(f"saved: {_grid_vs_a(out_dir, scales, lambdas, series, f'{prefix}_token_ratio_vs_a.pdf', ylabel='token ratio E[P]/E[D]')}")
    if "transform_total_mean" in z.files:
        runs_tr = np.asarray(z["transform_total_mean"], dtype=np.float64)
        series = {"total": ("#333333", "o", "E[P+D]", runs_tr)}
        print(f"saved: {_grid_vs_a(out_dir, scales, lambdas, series, f'{prefix}_total_token_mean_vs_a.pdf', ylabel='mean total tokens E[P+D]')}")


if __name__ == "__main__":
    main()
