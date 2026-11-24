"""fig12 绘图：p × 预测误差 × 构成 联合分析。

数据形状统一为 [n_seeds, n_a, n_p, n_err]（policies 各自前缀）。误判设置为分类横轴
（oracle, sl0.1, sl0.3, ls0.1, ls0.3），曲线按 p 或按策略区分，子图按构成倍率 a 或按
场景区分。

输出（沿用可阅读的命名）：
  QBA 误差效应（子图=a，曲线=p）:
    fig12_qba_{mean,p99,ts_mean,tl_mean}_vs_error.pdf
  φ_Q 提升兑现率（QBA，子图=a，曲线=p）:
    fig12_qba_phi_q_vs_error.pdf
  策略对比（固定 a、p，曲线=4 策略）:
    fig12_policy_{mean,ts_mean,tl_mean}_vs_error.pdf
  工作负载诊断（子图=a，曲线=p）:
    fig12_{pi_s_pred,chi_pred,n_pred_long}_vs_error.pdf
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

# (显示名, npz 前缀, 样式键)
_POLICIES = [
    ("FCFS", "fcfs", "FCFS"),
    ("PSF", "psf", "PSF"),
    ("AA", "psfa", "PSFA"),
    ("QBA", "psfaq", "PSFAQ"),
]
# 按 p 区分曲线时的样式（最多 3 条）
_P_STYLES = [
    {"color": "#1F77FF", "marker": "o", "linestyle": "-"},
    {"color": "#00B050", "marker": "s", "linestyle": "--"},
    {"color": "#C43E1C", "marker": "^", "linestyle": "-."},
]


def _err_series(runs: np.ndarray, ai: int, pj: int):
    """取 [n_seeds, n_a, n_p, n_err] 中 (a=ai, p=pj) 的误差轴曲线，返回 (m, lo, hi)。"""
    sub = np.asarray(runs, dtype=np.float64)[:, ai, pj, :]  # [n_seeds, n_err]
    return mean_ci(sub)


def _draw_error_panels(out_dir: str, fname: str, err_labels: list[str],
                       panel_titles: list[str], series_by_panel: list[list],
                       ylabel: str, *, xlabel: str = "Prediction error setting"):
    """通用误差面板：x=误差设置（分类），每个子图若干曲线。

    series_by_panel[i] = [(label, style_dict, m, lo, hi), ...]
    """
    setup_matplotlib()
    n = len(panel_titles)
    fig, axes = plt.subplots(1, n, figsize=(7.0 * n, 6.0), squeeze=False)
    x = np.arange(len(err_labels), dtype=np.float64)

    # 共享 y 轴范围，便于跨子图比较（同时纳入均值与置信区间上下界，避免 CI 被裁切）
    flat = []
    for panel in series_by_panel:
        for (_, _, m, lo, hi) in panel:
            flat.extend([m, lo, hi])  # lo/hi 为 None 时由 auto_ylim 跳过
    ylim = auto_ylim(flat)

    for ci, (title, panel) in enumerate(zip(panel_titles, series_by_panel)):
        ax = axes[0][ci]
        ax.set_title(title, fontsize=20)
        for (label, st, m, lo, hi) in panel:
            ax.plot(x, m, color=st["color"], marker=st["marker"], markersize=8,
                    linestyle=st["linestyle"], linewidth=2.6, label=label)
            fill_ci(ax, x, lo, hi, st["color"], alpha=0.15)
        ax.set_xticks(x)
        ax.set_xticklabels(err_labels, fontsize=15)
        ax.minorticks_off()
        apply_axes_style(
            ax, xlabel=xlabel, ylabel=ylabel,
            xlim=(-0.4, len(err_labels) - 0.6), ylim=ylim,
            show_legend=(ci == 0),
        )
    fig.tight_layout()
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="fig12: p × 误差 × 构成 联合分析绘图")
    parser.add_argument("--data", type=str, default=os.path.join(_DEFAULT_PLOT_DIR, "fig12_data.npz"))
    parser.add_argument("--out-dir", type=str, default=_DEFAULT_PLOT_DIR)
    parser.add_argument("--prefix", type=str, default="fig12")
    parser.add_argument("--verbose-meta", action="store_true", default=False)
    args = parser.parse_args()

    data_path = os.path.abspath(args.data)
    out_dir = ensure_dir(os.path.abspath(args.out_dir))
    prefix = args.prefix

    z = np.load(data_path, allow_pickle=False)
    scales = np.asarray(z["composition_scales"], dtype=np.float64)
    percentiles = np.asarray(z["percentiles"], dtype=np.float64)
    err_labels = [str(s) for s in np.asarray(z["error_names"]).tolist()]
    n_a, n_p = len(scales), len(percentiles)

    def _policy(prefix_key, mk):
        key = f"{prefix_key}_{mk}_runs"
        return np.asarray(z[key], dtype=np.float64) if key in z.files else None

    def _work(dk):
        key = f"{dk}_runs"
        return np.asarray(z[key], dtype=np.float64) if key in z.files else None

    meta_path = os.path.join(os.path.dirname(data_path), f"{prefix}_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print("meta", meta if args.verbose_meta else meta_summary(meta))

    a_titles = [f"a = {v:g}" for v in scales.tolist()]

    # ── A. QBA 误差效应：子图=a，曲线=p ──────────────────────────
    qba_prefix = dict((disp, pfx) for disp, pfx, _ in _POLICIES)["QBA"]
    for mk, ylabel in [("mean", "mean wait (s)"), ("p99", "p99 wait (s)"),
                       ("ts_mean", "true-short mean wait (s)"),
                       ("tl_mean", "true-long mean wait (s)")]:
        runs = _policy(qba_prefix, mk)
        if runs is None:
            continue
        panels = []
        for ai in range(n_a):
            panel = []
            for pj in range(n_p):
                m, lo, hi = _err_series(runs, ai, pj)
                st = _P_STYLES[pj % len(_P_STYLES)]
                panel.append((f"p = {percentiles[pj]:g}", st, m, lo, hi))
            panels.append(panel)
        path = _draw_error_panels(
            out_dir, f"{prefix}_qba_{mk}_vs_error.pdf", err_labels, a_titles, panels, ylabel)
        print(f"saved: {path}")

    # ── B. φ_Q（QBA 提升兑现率）：子图=a，曲线=p ─────────────────
    phi = _policy(qba_prefix, "phi_Q")
    if phi is not None:
        panels = []
        for ai in range(n_a):
            panel = []
            for pj in range(n_p):
                m, lo, hi = _err_series(phi, ai, pj)
                st = _P_STYLES[pj % len(_P_STYLES)]
                panel.append((f"p = {percentiles[pj]:g}", st, m, lo, hi))
            panels.append(panel)
        path = _draw_error_panels(
            out_dir, f"{prefix}_qba_phi_q_vs_error.pdf", err_labels, a_titles, panels,
            "phi_Q = N_prom / N_pred_long")
        print(f"saved: {path}")

    # ── C. 策略对比：固定 a=1（若存在，否则中间档）、p 中间档 ─────
    ai_fix = int(np.argmin(np.abs(scales - 1.0))) if np.any(np.isclose(scales, 1.0)) else n_a // 2
    pj_fix = n_p // 2
    for mk, ylabel in [("mean", "mean wait (s)"), ("ts_mean", "true-short mean wait (s)"),
                       ("tl_mean", "true-long mean wait (s)")]:
        panels = [[]]
        any_curve = False
        for disp, pfx, style_key in _POLICIES:
            runs = _policy(pfx, mk)
            if runs is None:
                continue
            m, lo, hi = _err_series(runs, ai_fix, pj_fix)
            st = get_policy_style(style_key)
            panels[0].append((display_policy_name(style_key), st, m, lo, hi))
            any_curve = True
        if not any_curve:
            continue
        title = [f"a = {scales[ai_fix]:g}, p = {percentiles[pj_fix]:g}"]
        path = _draw_error_panels(
            out_dir, f"{prefix}_policy_{mk}_vs_error.pdf", err_labels, title, panels, ylabel)
        print(f"saved: {path}")

    # ── D. 工作负载诊断：子图=a，曲线=p ─────────────────────────
    for dk, ylabel in [("pi_s_pred", "predicted-short fraction pi_S_pred"),
                       ("chi_pred", "chi_pred = pi_S_pred/(k(1-pi_S_pred))"),
                       ("n_pred_long", "N_predicted_long")]:
        runs = _work(dk)
        if runs is None:
            continue
        panels = []
        for ai in range(n_a):
            panel = []
            for pj in range(n_p):
                m, lo, hi = _err_series(runs, ai, pj)
                st = _P_STYLES[pj % len(_P_STYLES)]
                panel.append((f"p = {percentiles[pj]:g}", st, m, lo, hi))
            panels.append(panel)
        path = _draw_error_panels(
            out_dir, f"{prefix}_{dk}_vs_error.pdf", err_labels, a_titles, panels, ylabel)
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
