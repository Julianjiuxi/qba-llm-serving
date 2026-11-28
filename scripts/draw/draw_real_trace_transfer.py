"""scaled real-trace replay 绘图：真实 Azure 轨迹上的理论迁移验证（模块 A–E）。

与 ``draw_fig15.py`` 结构对齐，但**不进入主系列命名**。数据来自
``scripts/real_trace_transfer.py`` 输出的 ``rt_{A..E}_data.npz``（默认位于
``scripts/real_trace_output``）。

数据形状约定（与主脚本 store 一致）:
    所有数组的**第 0 维都是窗口维（n_windows）**——100 个短窗口 / 32 个长窗口作为
    **配对重复单位**；绘图时对第 0 维求均值与置信区间（``mean_ci``）。
    A: 策略数组 [n_windows, n_a, n_K, n_l]；窗口数组 [n_windows, n_a, n_K, n_l, n_w]。
    B: QBA [n_windows, n_K, n_p, n_k]；PSF/AA [n_windows, n_K, n_p]；FCFS [n_windows, n_K]。
    C: 策略数组 [n_windows, n_K, n_err]；四群体 quad_{pfx} [n_windows, n_K, n_err, 4, 7]。
    D: 策略数组 [n_windows, n_N, n_a, n_K, n_l]；窗口 [..., n_w]。
    E: 策略数组 [n_windows, n_K]；e_lambda_eff [n_windows]。

正文三组对照（见设计稿）:
    组 1  原生真实构成下的策略折中及 KV 响应              → 模块 A
    组 2  实际 quota pressure 与 promotion、误判和等待      → 模块 B / C
    组 3  真实连续长流中的积压演化，以及资源放宽后的变化    → 模块 D / E
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
)
from .draw_fig14 import _style_panel, _set_log2_ticks

# (npz 策略键, 样式键)
_POLICIES = [("FCFS", "FCFS"), ("PSF", "PSF"), ("AA", "PSFA"), ("QBA", "PSFAQ")]
_PFX = {"FCFS": "fcfs", "PSF": "psf", "AA": "psfa", "QBA": "psfaq"}

# 模块 B：FCFS 整体指标跨 p 复用（[n_w, n_K] 常数线），但 **true-long 群体随 p 改变**，
# 需从按各 p 阈值重新分组统计的专用数组读取（[n_w, n_K, n_p]）。
_FCFS_METRIC_KEY = {"tl_p99": "tl_group_wait_p99", "tl_mean": "tl_group_wait_mean"}

_QUAD_NAMES = ["true-short /\npred-short", "true-short /\npred-long",
               "true-long /\npred-short", "true-long /\npred-long"]
_QUAD_STAT_NAMES = ["n", "wait_mean", "wait_p99", "promo_n", "P_mean", "D_mean", "T_mean"]

# 模块 A 关心的标量指标 / 固定评价群体键（不含策略前缀）
_A_KEYS = [
    "mean", "p99", "tl_mean", "tl_p99",
    "kv_gate_closed_rate", "kv_hard_touch_rate", "kv_peak_over_capacity_mean",
    "promotion_ratio", "phi_Q", "n_pred_long_normal",
    "unfinished_at_arrival_stop", "drain_time",
    "fixed_fx_ans_long_wait_mean", "fixed_fx_ans_long_wait_p99", "fixed_fx_ans_long_n",
    "fixed_fx_tot_long_wait_mean", "fixed_fx_tot_long_wait_p99", "fixed_fx_tot_long_n",
]

_RT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "real_trace_output"))
_TITLE_FS = 12


# ── 读取 ────────────────────────────────────────────────────

def _arr(z, key: str):
    if key not in z.files:
        return None
    return np.asarray(z[key], dtype=np.float64)


def _read_policy(z, pfx: str, keys: list[str]) -> dict:
    out: dict = {}
    for k in keys:
        a = _arr(z, f"{pfx}_{k}")
        if a is not None:
            out[k] = a
    return out


def _read_runs_a(z) -> dict:
    return {pol: _read_policy(z, _PFX[pol], _A_KEYS) for pol, _sk in _POLICIES}


def _load(path: str):
    if not path or not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=False)


def _meta_of(data_path: str, prefix: str, module: str, verbose: bool):
    meta_path = os.path.join(os.path.dirname(data_path), f"{prefix}_{module}_meta.json")
    if not os.path.exists(meta_path):
        return
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    print(f"[{module}] meta:", meta if verbose else meta_summary(meta))


def _maybe_print_pre(z, module: str, n_windows: int) -> None:
    keys = ["pre_raw_lambda", "pre_scaled_E_P", "pre_scaled_E_D", "pre_scaled_E_PD",
            "pre_true_long_fraction", "pre_inter_arrival_Ca2"]
    vals = {k: _arr(z, k) for k in keys}
    if all(v is None for v in vals.values()):
        return
    print(f"[{module}] window pre-stats (mean over {n_windows} windows):")
    for k, v in vals.items():
        if v is None:
            continue
        flat = v.reshape(-1)
        flat = flat[np.isfinite(flat)]
        if flat.size:
            print(f"    {k[4:]:<22s} {np.nanmean(flat):.4f}")


# ── 通用网格面板 ────────────────────────────────────────────

def _save(fig, out_dir: str, fname: str) -> str:
    out_path = os.path.join(out_dir, fname)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")
    return out_path


def _finish_ylim(axes, series, n_rows, n_cols, ylim):
    if ylim is not None:
        for r in range(n_rows):
            for c in range(n_cols):
                axes[r][c].set_ylim(float(ylim[0]), float(ylim[1]))
        return
    ylim = auto_ylim(series)
    if ylim is not None:
        for r in range(n_rows):
            for c in range(n_cols):
                axes[r][c].set_ylim(ylim)


# ── 组 1：模块 A（策略折中 + KV 响应）──────────────────────

def _a_policy_vs_K(out_dir, prefix, a_vals, lambdas, caps, runs, metric, fname, *,
                   ylabel, ylim=None):
    """rows=a, cols=λ；x=K(log2)；lines=policies。runs[pol][metric]=[n_w,n_a,n_K,n_l]。"""
    n_a, n_l = len(a_vals), len(lambdas)
    setup_matplotlib()
    fig, axes = plt.subplots(n_a, n_l, figsize=(3.5 * n_l, 3.2 * n_a),
                             squeeze=False, sharex=True)
    series = []
    for ai in range(n_a):
        for li in range(n_l):
            ax = axes[ai][li]
            ax.set_xscale("log", base=2)
            for pol, sk in _POLICIES:
                arr = runs.get(pol, {}).get(metric)
                if arr is None:
                    continue
                m, lo, hi = mean_ci(arr[:, ai, :, li])          # [n_w, n_K] → [n_K]
                st = get_policy_style(sk)
                ax.plot(caps, m, color=st["color"], marker=st["marker"],
                        markersize=5, linestyle=st["linestyle"], linewidth=2.0,
                        label=display_policy_name(sk))
                fill_ci(ax, caps, lo, hi, st["color"], alpha=0.16)
                series.extend([m, lo, hi])
            _style_panel(ax, xlabel=("KV capacity K (tokens)" if ai == n_a - 1 else ""),
                         ylabel=(f"a={a_vals[ai]:g}\n{ylabel}" if li == 0 else ""),
                         legend=(ai == 0 and li == 0))
            _set_log2_ticks(ax, caps, as_int=True)
            if ai == 0:
                ax.set_title(f"λ={lambdas[li]:g}", fontsize=_TITLE_FS)
    _finish_ylim(axes, series, n_a, n_l, ylim)
    return _save(fig, out_dir, fname)


def _a_series_vs_K(out_dir, prefix, a_vals, lambdas, caps, series_dict, fname, *,
                   ylabel, ylim=None):
    """rows=a, cols=λ；x=K(log2)；series_dict[name]=(color, marker, label, [n_w,n_a,n_K,n_l])。"""
    n_a, n_l = len(a_vals), len(lambdas)
    setup_matplotlib()
    fig, axes = plt.subplots(n_a, n_l, figsize=(3.5 * n_l, 3.2 * n_a),
                             squeeze=False, sharex=True)
    series = []
    for ai in range(n_a):
        for li in range(n_l):
            ax = axes[ai][li]
            ax.set_xscale("log", base=2)
            for name, (color, marker, label, arr) in series_dict.items():
                arr = np.asarray(arr, dtype=np.float64)
                m, lo, hi = mean_ci(arr[:, ai, :, li])
                ax.plot(caps, m, color=color, marker=marker, markersize=5,
                        linestyle="-", linewidth=2.0, label=label)
                fill_ci(ax, caps, lo, hi, color, alpha=0.16)
                series.extend([m, lo, hi])
            _style_panel(ax, xlabel=("KV capacity K (tokens)" if ai == n_a - 1 else ""),
                         ylabel=(f"a={a_vals[ai]:g}\n{ylabel}" if li == 0 else ""),
                         legend=(ai == 0 and li == 0))
            _set_log2_ticks(ax, caps, as_int=True)
            if ai == 0:
                ax.set_title(f"λ={lambdas[li]:g}", fontsize=_TITLE_FS)
    _finish_ylim(axes, series, n_a, n_l, ylim)
    return _save(fig, out_dir, fname)


def _draw_module_a(out_dir, prefix, z, verbose_meta):
    scales = _arr(z, "composition_scales")
    lambdas = _arr(z, "lambdas")
    caps = _arr(z, "capacities")
    if scales is None or lambdas is None or caps is None:
        print("skip A: 缺少轴数组")
        return
    runs = _read_runs_a(z)
    n_w = int(_arr(z, "n_windows").reshape(-1)[0]) if _arr(z, "n_windows") is not None else 0
    _maybe_print_pre(z, "A", n_w)
    for metric, tag, ylabel, ylim in [
        ("mean", "wait_mean", "mean waiting (s)", None),
        ("p99", "wait_p99", "p99 waiting (s)", None),
        ("tl_mean", "tl_mean", "true-long mean waiting (s)", None),
        ("tl_p99", "tl_p99", "true-long p99 waiting (s)", None),
    ]:
        if any(runs[p].get(metric) is not None for p, _ in _POLICIES):
            _a_policy_vs_K(out_dir, prefix, scales, lambdas, caps, runs, metric,
                           f"{prefix}_A_{tag}_vs_K.pdf", ylabel=ylabel, ylim=ylim)
    # 固定评价群体（跨策略共享请求 ID）
    for metric, tag, ylabel in [
        ("fixed_fx_ans_long_wait_mean", "fixed_anslong_wait_mean",
         "fixed eval group (answer-top) mean wait (s)"),
        ("fixed_fx_ans_long_wait_p99", "fixed_anslong_wait_p99",
         "fixed eval group (answer-top) p99 wait (s)"),
        ("fixed_fx_tot_long_wait_mean", "fixed_totlong_wait_mean",
         "fixed eval group (total-top) mean wait (s)"),
        ("fixed_fx_tot_long_wait_p99", "fixed_totlong_wait_p99",
         "fixed eval group (total-top) p99 wait (s)"),
    ]:
        if any(runs[p].get(metric) is not None for p, _ in _POLICIES):
            _a_policy_vs_K(out_dir, prefix, scales, lambdas, caps, runs, metric,
                           f"{prefix}_A_{tag}_vs_K.pdf", ylabel=ylabel)
    # KV 门控 / promotion
    for metric, tag, ylabel, ylim in [
        ("kv_gate_closed_rate", "kv_gate_closed_rate",
         "KV safety-gate closed batch fraction", (0.0, 1.0)),
        ("kv_hard_touch_rate", "kv_hard_touch_rate",
         "hard-capacity touched batch fraction", (0.0, 1.0)),
        ("kv_peak_over_capacity_mean", "kv_peak_over_capacity",
         "KV peak / capacity (mean)", None),
        ("promotion_ratio", "promotion_ratio",
         "promotion ratio (N_prom / N_all)", (0.0, 1.0)),
        ("phi_Q", "phi_Q", "phi_Q (N_prom / N_pred_long)", (0.0, 1.0)),
        ("n_pred_long_normal", "long_admit_normal",
         "unique normal admits of predicted-long", None),
    ]:
        if any(runs[p].get(metric) is not None for p, _ in _POLICIES):
            _a_policy_vs_K(out_dir, prefix, scales, lambdas, caps, runs, metric,
                           f"{prefix}_A_{tag}_vs_K.pdf", ylabel=ylabel, ylim=ylim)
    # QBA vs PSF 折中
    qba, psf = runs.get("QBA", {}), runs.get("PSF", {})
    if qba.get("mean") is not None and psf.get("mean") is not None:
        series = {"C_mean": ("#8F3FD6", "o", "C_mean = W_QBA − W_PSF (s)",
                             qba["mean"] - psf["mean"])}
        if qba.get("tl_p99") is not None and psf.get("tl_p99") is not None:
            series["G_tail"] = ("#00B050", "^", "G_tail = P99L,PSF − P99L,QBA (s)",
                                psf["tl_p99"] - qba["tl_p99"])
        _a_series_vs_K(out_dir, prefix, scales, lambdas, caps, series,
                       f"{prefix}_A_tradeoff_vs_K.pdf", ylabel="QBA vs PSF tradeoff (s)")


# ── 组 2a：模块 B（实际 quota pressure 与 promotion）────────

def _b_chi_vs_phi(out_dir, prefix, z, caps):
    """rows=K；x=实际 χ_pred；lines=k；y=phi_Q。"""
    chi = _arr(z, "b_chi_pred")                 # [n_w, n_p, n_k]
    phi = _arr(z, "psfaq_phi_Q")                # [n_w, n_K, n_p, n_k]
    ks = _arr(z, "ks")
    if chi is None or phi is None or ks is None:
        print("skip B: χ/phi 缺失")
        return
    n_K, n_k = phi.shape[1], phi.shape[3]
    x = np.nanmean(chi, axis=0)                  # [n_p, n_k]
    setup_matplotlib()
    fig, axes = plt.subplots(1, n_K, figsize=(4.2 * n_K, 3.4), squeeze=False)
    cmap = plt.get_cmap("viridis")
    for ki in range(n_K):
        ax = axes[0][ki]
        for kk in range(n_k):
            y = np.nanmean(phi[:, ki, :, kk], axis=0)   # [n_p]
            color = cmap(0.15 + 0.7 * kk / max(1, n_k - 1))
            ax.plot(x[:, kk], y, color=color, marker="o", markersize=6,
                    linewidth=2.0, label=f"k={int(ks[kk])}")
        _style_panel(ax, xlabel="actual χ_pred = N_pred_short / (k·N_pred_long)",
                     ylabel=("promotion coverage φ_Q" if ki == 0 else ""),
                     legend=(ki == 0))
        ax.set_title(f"K={int(caps[ki])}", fontsize=_TITLE_FS)
    _save(fig, out_dir, f"{prefix}_B_chi_vs_phi.pdf")


def _b_vs_p(out_dir, prefix, z, caps, metric, fname, ylabel, *, style_for_policy=True):
    """rows=K；x=p；lines=QBA(k) + PSF/AA/FCFS 参考。metric 可为 {pfx}_ 前缀键。"""
    ps = _arr(z, "ps")
    if ps is None:
        print("skip B: p 轴缺失")
        return
    arrays = {}
    for pol, _sk in _POLICIES:
        key = _FCFS_METRIC_KEY.get(metric, metric) if pol == "FCFS" else metric
        arrays[pol] = _arr(z, f"{_PFX[pol]}_{key}")
    if all(a is None for a in arrays.values()):
        print(f"skip B[{metric}]: 无可绘数组")
        return
    n_K = len(caps)
    ks = _arr(z, "ks")
    n_k = int(ks.shape[0]) if ks is not None else 1
    setup_matplotlib()
    fig, axes = plt.subplots(1, n_K, figsize=(4.2 * n_K, 3.4), squeeze=False)
    series = []
    for ki in range(n_K):
        ax = axes[0][ki]
        # 非 QBA 参考线
        for pol, sk in _POLICIES:
            if pol == "QBA":
                continue
            arr = arrays[pol]
            if arr is None:
                continue
            st = get_policy_style(sk)
            if arr.ndim == 2:                     # FCFS [n_w, n_K] → 常数
                m = np.nanmean(arr[:, ki])
                ax.axhline(m, color=st["color"], linestyle=st["linestyle"],
                           linewidth=2.0, label=display_policy_name(sk))
            else:                                  # PSF/AA [n_w, n_K, n_p]
                m, lo, hi = mean_ci(arr[:, ki, :])
                ax.plot(ps, m, color=st["color"], marker=st["marker"], markersize=5,
                        linestyle=st["linestyle"], linewidth=2.0,
                        label=display_policy_name(sk))
                fill_ci(ax, ps, lo, hi, st["color"], alpha=0.14)
                series.extend([m, lo, hi])
        # QBA k 曲线
        q = arrays["QBA"]
        cmap = plt.get_cmap("viridis")
        if q is not None and q.ndim == 4:          # [n_w, n_K, n_p, n_k]
            for kk in range(q.shape[3]):
                m, lo, hi = mean_ci(q[:, ki, :, kk])
                color = cmap(0.15 + 0.7 * kk / max(1, n_k - 1))
                ax.plot(ps, m, color=color, marker="o", markersize=5, linewidth=2.0,
                        label=f"QBA k={int(ks[kk])}" if ks is not None else f"QBA k[{kk}]")
                fill_ci(ax, ps, lo, hi, color, alpha=0.14)
                series.extend([m, lo, hi])
        _style_panel(ax, xlabel="label quantile p",
                     ylabel=(ylabel if ki == 0 else ""), legend=(ki == 0))
        ax.set_title(f"K={int(caps[ki])}", fontsize=_TITLE_FS)
    ylim = auto_ylim(series)
    if ylim is not None:
        for ki in range(n_K):
            axes[0][ki].set_ylim(ylim[0], ylim[1])
    _save(fig, out_dir, fname)


def _draw_module_b(out_dir, prefix, z, verbose_meta):
    caps = _arr(z, "capacities")
    if caps is None:
        print("skip B: 缺少 capacities")
        return
    _b_chi_vs_phi(out_dir, prefix, z, caps)
    for metric, tag, ylabel in [
        ("mean", "wait_mean", "mean waiting (s)"),
        ("p99", "wait_p99", "p99 waiting (s)"),
        ("tl_p99", "tl_p99", "true-long p99 waiting (s)"),
        ("fixed_fx_tot_long_wait_p99", "fixed_totlong_wait_p99",
         "fixed eval group (total-top) p99 wait (s)"),
        ("promotion_ratio", "promotion_ratio", "promotion ratio (N_prom / N_all)"),
        ("phi_Q", "phi_Q", "phi_Q (N_prom / N_pred_long)"),
    ]:
        _b_vs_p(out_dir, prefix, z, caps, metric, f"{prefix}_B_{tag}_vs_p.pdf", ylabel)


# ── 组 2b：模块 C（方向性误判）──────────────────────────────

def _c_vs_err(out_dir, prefix, z, caps, metric, fname, ylabel, *, ylim=None):
    """rows=K；x=err 配置；lines=policies。"""
    err_names = [str(x) for x in np.asarray(z["err_names"]).reshape(-1)]
    n_err = len(err_names)
    arrays = {pol: _arr(z, f"{_PFX[pol]}_{metric}") for pol, _sk in _POLICIES}
    if all(a is None for a in arrays.values()):
        print(f"skip C[{metric}]")
        return
    n_K = len(caps)
    x = np.arange(n_err)
    setup_matplotlib()
    fig, axes = plt.subplots(1, n_K, figsize=(4.4 * n_K, 3.4), squeeze=False)
    series = []
    for ki in range(n_K):
        ax = axes[0][ki]
        for pol, sk in _POLICIES:
            arr = arrays[pol]
            if arr is None:
                continue
            m, lo, hi = mean_ci(arr[:, ki, :])      # [n_w, n_err] → [n_err]
            st = get_policy_style(sk)
            ax.plot(x, m, color=st["color"], marker=st["marker"], markersize=5,
                    linestyle=st["linestyle"], linewidth=2.0,
                    label=display_policy_name(sk))
            fill_ci(ax, x, lo, hi, st["color"], alpha=0.16)
            series.extend([m, lo, hi])
        _style_panel(ax, xlabel="error configuration",
                     ylabel=(ylabel if ki == 0 else ""), legend=(ki == 0))
        ax.set_xticks(x)
        ax.set_xticklabels(err_names, rotation=0)
        ax.set_title(f"K={int(caps[ki])}", fontsize=_TITLE_FS)
    _finish_ylim(axes, series, 1, n_K, ylim)
    _save(fig, out_dir, fname)


def _c_quad(out_dir, prefix, z, caps, stat_idx, fname, ylabel):
    """rows=K；cols=四群体；x=err；lines=policies；y=某群体统计量。"""
    err_names = [str(x) for x in np.asarray(z["err_names"]).reshape(-1)]
    n_err = len(err_names)
    quads = {pol: _arr(z, f"quad_{_PFX[pol]}") for pol, _sk in _POLICIES}
    if all(q is None for q in quads.values()):
        print(f"skip C[quad {fname}]")
        return
    n_K = len(caps)
    x = np.arange(n_err)
    setup_matplotlib()
    fig, axes = plt.subplots(n_K, 4, figsize=(4.0 * 4, 3.2 * n_K), squeeze=False)
    series = []
    for ki in range(n_K):
        for qi in range(4):
            ax = axes[ki][qi]
            for pol, sk in _POLICIES:
                q = quads[pol]
                if q is None:
                    continue
                sub = q[:, ki, :, qi, stat_idx]      # [n_w, n_err]
                m, lo, hi = mean_ci(sub)
                st = get_policy_style(sk)
                ax.plot(x, m, color=st["color"], marker=st["marker"], markersize=4,
                        linestyle=st["linestyle"], linewidth=1.8,
                        label=display_policy_name(sk))
                fill_ci(ax, x, lo, hi, st["color"], alpha=0.16)
                series.extend([m, lo, hi])
            _style_panel(ax, xlabel=("error configuration" if ki == n_K - 1 else ""),
                         ylabel=(f"K={int(caps[ki])}\n{ylabel}" if qi == 0 else ""),
                         legend=(ki == 0 and qi == 0))
            ax.set_xticks(x)
            ax.set_xticklabels(err_names)
            if ki == 0:
                ax.set_title(_QUAD_NAMES[qi], fontsize=_TITLE_FS)
    _finish_ylim(axes, series, n_K, 4, None)
    _save(fig, out_dir, fname)


def _draw_module_c(out_dir, prefix, z, verbose_meta):
    caps = _arr(z, "capacities")
    if caps is None:
        print("skip C: 缺少 capacities")
        return
    for metric, tag, ylabel, ylim in [
        ("promotion_ratio", "promotion_ratio", "promotion ratio (N_prom / N_all)", (0.0, 1.0)),
        ("phi_Q", "phi_Q", "phi_Q (N_prom / N_pred_long)", (0.0, 1.0)),
        ("mean", "wait_mean", "mean waiting (s)", None),
        ("tl_p99", "tl_p99", "true-long p99 waiting (s)", None),
        ("fixed_fx_tot_long_wait_p99", "fixed_totlong_wait_p99",
         "fixed eval group (total-top) p99 wait (s)", None),
    ]:
        _c_vs_err(out_dir, prefix, z, caps, metric,
                  f"{prefix}_C_{tag}_vs_err.pdf", ylabel, ylim=ylim)
    _c_quad(out_dir, prefix, z, caps, 1, f"{prefix}_C_quad_wait_mean.pdf",
            "mean waiting (s)")
    _c_quad(out_dir, prefix, z, caps, 0, f"{prefix}_C_quad_count.pdf", "population")


# ── 组 3：模块 D（长流积压演化）─────────────────────────────

def _d_stream_vs_N(out_dir, prefix, z, caps, lengths, scales, lambdas, metric,
                   fname, ylabel, *, cap_index=-1):
    """rows=a, cols=λ；x=N(log2)；lines=policies；固定某一个 K。arr=[n_w,n_N,n_a,n_K,n_l]。"""
    arrays = {pol: _arr(z, f"{_PFX[pol]}_{metric}") for pol, _sk in _POLICIES}
    if all(a is None for a in arrays.values()):
        print(f"skip D[{metric}]")
        return
    n_a, n_l = len(scales), len(lambdas)
    setup_matplotlib()
    fig, axes = plt.subplots(n_a, n_l, figsize=(3.5 * n_l, 3.2 * n_a),
                             squeeze=False, sharex=True)
    series = []
    for ai in range(n_a):
        for li in range(n_l):
            ax = axes[ai][li]
            ax.set_xscale("log", base=2)
            for pol, sk in _POLICIES:
                arr = arrays[pol]
                if arr is None:
                    continue
                m, lo, hi = mean_ci(arr[:, :, ai, cap_index, li])   # [n_w, n_N]
                st = get_policy_style(sk)
                ax.plot(lengths, m, color=st["color"], marker=st["marker"], markersize=5,
                        linestyle=st["linestyle"], linewidth=2.0,
                        label=display_policy_name(sk))
                fill_ci(ax, lengths, lo, hi, st["color"], alpha=0.16)
                series.extend([m, lo, hi])
            _style_panel(ax, xlabel=("stream length N" if ai == n_a - 1 else ""),
                         ylabel=(f"a={scales[ai]:g}\n{ylabel}" if li == 0 else ""),
                         legend=(ai == 0 and li == 0))
            _set_log2_ticks(ax, lengths, as_int=True)
            if ai == 0:
                ax.set_title(f"λ={lambdas[li]:g}", fontsize=_TITLE_FS)
    _finish_ylim(axes, series, n_a, n_l, None)
    _save(fig, out_dir, fname)


def _d_backlog_windows(out_dir, prefix, z, caps, scales, lambdas, fname, ylabel):
    """rows=a, cols=λ；x=窗口序号；lines=K（取完整流 N）。QBA 时间口径窗口。"""
    arr = _arr(z, "psfaq_time_backlog")          # [n_w, n_N, n_a, n_K, n_l, n_win]
    if arr is None:
        print("skip D: psfaq_time_backlog 缺失")
        return
    n_N = arr.shape[1]
    n_a, n_l, n_K, n_win = arr.shape[2], arr.shape[4], arr.shape[3], arr.shape[5]
    setup_matplotlib()
    fig, axes = plt.subplots(n_a, n_l, figsize=(3.5 * n_l, 3.2 * n_a),
                             squeeze=False, sharex=True)
    x = np.arange(n_win)
    cmap = plt.get_cmap("viridis")
    series = []
    for ai in range(n_a):
        for li in range(n_l):
            ax = axes[ai][li]
            for ki in range(n_K):
                sub = arr[:, n_N - 1, ai, ki, li, :]      # [n_w, n_win]
                m, lo, hi = mean_ci(sub)
                color = cmap(0.15 + 0.7 * ki / max(1, n_K - 1))
                ax.plot(x, m, color=color, marker="o", markersize=4, linewidth=2.0,
                        label=f"K={int(caps[ki])}")
                fill_ci(ax, x, lo, hi, color, alpha=0.14)
                series.extend([m, lo, hi])
            _style_panel(ax, xlabel=("window index (equal time)" if ai == n_a - 1 else ""),
                         ylabel=(f"a={scales[ai]:g}\n{ylabel}" if li == 0 else ""),
                         legend=(ai == 0 and li == 0))
            if ai == 0:
                ax.set_title(f"λ={lambdas[li]:g}", fontsize=_TITLE_FS)
    _finish_ylim(axes, series, n_a, n_l, None)
    _save(fig, out_dir, fname)


def _draw_module_d(out_dir, prefix, z, verbose_meta):
    lengths = _arr(z, "lengths")
    scales = _arr(z, "composition_scales")
    caps = _arr(z, "capacities")
    lambdas = _arr(z, "lambdas")
    if any(v is None for v in (lengths, scales, caps, lambdas)):
        print("skip D: 缺少轴数组")
        return
    _maybe_print_pre(z, "D", int(_arr(z, "n_windows").reshape(-1)[0]))
    for metric, tag, ylabel in [
        ("unfinished_at_arrival_stop", "unfinished_at_arrival_stop",
         "unfinished requests at arrival stop"),
        ("drain_time", "drain_time", "drain time (s)"),
        ("promotion_ratio", "promotion_ratio", "promotion ratio (N_prom / N_all)"),
        ("phi_Q", "phi_Q", "phi_Q (N_prom / N_pred_long)"),
        ("n_pred_long_normal", "long_admit_normal",
         "unique normal admits of predicted-long"),
        ("mean", "wait_mean", "mean waiting (s)"),
    ]:
        _d_stream_vs_N(out_dir, prefix, z, caps, lengths, scales, lambdas, metric,
                       f"{prefix}_D_{tag}_vs_N.pdf", ylabel)
    _d_backlog_windows(out_dir, prefix, z, caps, scales, lambdas,
                       f"{prefix}_D_backlog_time.pdf",
                       "cumulative backlog (arrivals − completions)")


# ── 组 3：模块 E（统一时间缩放对照）─────────────────────────

def _draw_module_e(out_dir, prefix, z, verbose_meta):
    caps = _arr(z, "capacities")
    if caps is None:
        print("skip E: 缺少 capacities")
        return
    lam_eff = _arr(z, "e_lambda_eff")
    if lam_eff is not None:
        v = lam_eff.reshape(-1)
        v = v[np.isfinite(v)]
        print(f"[E] per-window effective λ: min={np.min(v):.3f} "
              f"mean={np.mean(v):.3f} max={np.max(v):.3f} (shared time scale, "
              f"must be reported per window)")
    for metric, tag, ylabel in [
        ("mean", "wait_mean", "mean waiting (s)"),
        ("p99", "wait_p99", "p99 waiting (s)"),
        ("tl_p99", "tl_p99", "true-long p99 waiting (s)"),
        ("fixed_fx_tot_long_wait_p99", "fixed_totlong_wait_p99",
         "fixed eval group (total-top) p99 wait (s)"),
    ]:
        arrays = {pol: _arr(z, f"{_PFX[pol]}_{metric}") for pol, _sk in _POLICIES}
        if all(a is None for a in arrays.values()):
            continue
        setup_matplotlib()
        fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.6))
        series = []
        for pol, sk in _POLICIES:
            arr = arrays[pol]
            if arr is None:
                continue
            m, lo, hi = mean_ci(arr)                  # [n_w, n_K] → [n_K]
            st = get_policy_style(sk)
            ax.plot(caps, m, color=st["color"], marker=st["marker"], markersize=6,
                    linestyle=st["linestyle"], linewidth=2.0,
                    label=display_policy_name(sk))
            fill_ci(ax, caps, lo, hi, st["color"], alpha=0.16)
            series.extend([m, lo, hi])
        _style_panel(ax, xlabel="KV capacity K (tokens)", ylabel=ylabel, legend=True)
        _set_log2_ticks(ax, caps, as_int=True)
        ylim = auto_ylim(series)
        if ylim is not None:
            ax.set_ylim(ylim)
        _save(fig, out_dir, f"{prefix}_E_{tag}_vs_K.pdf")


# ── main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="scaled real-trace replay 绘图（模块 A–E，不进入主系列命名）")
    parser.add_argument("--data-dir", type=str, default=_RT_DIR)
    parser.add_argument("--out-dir", type=str, default=_RT_DIR)
    parser.add_argument("--prefix", type=str, default="rt")
    parser.add_argument("--modules", type=str, default="A,B,C,D,E")
    parser.add_argument("--verbose-meta", action="store_true", default=False)
    args = parser.parse_args()

    out_dir = ensure_dir(os.path.abspath(args.out_dir))
    data_dir = os.path.abspath(args.data_dir)
    prefix = args.prefix
    modules = [m.strip().upper() for m in args.modules.split(",") if m.strip()]

    dispatchers = {
        "A": _draw_module_a,
        "B": _draw_module_b,
        "C": _draw_module_c,
        "D": _draw_module_d,
        "E": _draw_module_e,
    }
    for m in modules:
        path = os.path.join(data_dir, f"{prefix}_{m}_data.npz")
        z = _load(path)
        if z is None:
            print(f"skip module {m}: 未找到 {path}")
            continue
        print(f"=== module {m} ({path}) ===")
        _meta_of(path, prefix, m, args.verbose_meta)
        dispatchers[m](out_dir, prefix, z, args.verbose_meta)


if __name__ == "__main__":
    main()
