"""fig2 绘图：lambda 扫参 — SIM 层 + 公式层 + phi 曲线。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ._common import (
    setup_matplotlib,
    ensure_dir,
    infer_prefix,
    apply_axes_style,
    auto_ylim,
    mean_ci,
    display_policy_name,
    get_policy_style,
    meta_summary,
    fill_ci,
    _DEFAULT_PLOT_DIR,
)


def _plot_sim(out_dir: str, prefix: str, sim_lambdas: np.ndarray, sim_rows: dict):
    setup_matplotlib()
    panels = [
        ("mean", f"{prefix}_sim_waiting_mean_vs_lambda.png"),
        ("p95", f"{prefix}_sim_waiting_p95_vs_lambda.png"),
        ("p99", f"{prefix}_sim_waiting_p99_vs_lambda.png"),
    ]
    out = {}
    for key, fname in panels:
        fig, ax = plt.subplots(1, 1, figsize=(9, 7))
        for policy, series in sim_rows.items():
            st = get_policy_style(policy)
            if f"{key}_runs" in series and series[f"{key}_runs"] is not None:
                m, lo, hi = mean_ci(series[f"{key}_runs"])
                ax.plot(sim_lambdas, m, color=st["color"], marker=st["marker"],
                        linestyle=st["linestyle"], markersize=9, linewidth=3.0,
                        label=display_policy_name(policy))
                fill_ci(ax, sim_lambdas, lo, hi, st["color"])
            else:
                ax.plot(sim_lambdas, series[key], color=st["color"],
                        marker=st["marker"], linestyle=st["linestyle"],
                        markersize=9, linewidth=3.0,
                        label=display_policy_name(policy))

        ylim_series = []
        for v in sim_rows.values():
            if f"{key}_runs" in v and v[f"{key}_runs"] is not None:
                m, _, _ = mean_ci(v[f"{key}_runs"])
                ylim_series.append(m)
            else:
                ylim_series.append(v[key])
        ylim = auto_ylim(ylim_series)
        apply_axes_style(
            ax, xlabel="\u03bb", ylabel="seconds",
            xlim=(float(np.min(sim_lambdas)), float(np.max(sim_lambdas))),
            ylim=ylim, show_legend=True,
        )
        fig.tight_layout()
        out_path = os.path.join(out_dir, fname)
        fig.savefig(out_path, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        out[key] = out_path
    return out


def _plot_formula(
    out_dir: str,
    prefix: str,
    formula_lambdas: np.ndarray,
    curves: dict,
    reference_load: float,
):
    setup_matplotlib()
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    st_fcfs = get_policy_style("FCFS")
    st_psf = get_policy_style("PSF")
    st_aa = get_policy_style("PSFA")
    st_qba = get_policy_style("PSFAQ")

    ax.plot(formula_lambdas, curves["E[Wq]_FCFS"], color=st_fcfs["color"],
            marker=st_fcfs["marker"], linestyle=st_fcfs["linestyle"],
            markersize=9, linewidth=3.0, label="FCFS")
    ax.plot(formula_lambdas, curves["E[Wq]_PSF"], color=st_psf["color"],
            marker=st_psf["marker"], linestyle=st_psf["linestyle"],
            markersize=9, linewidth=3.0, label="PSF")
    ax.plot(formula_lambdas, curves["E[Wq]_PSFA"], color=st_aa["color"],
            marker=st_aa["marker"], linestyle=st_aa["linestyle"],
            markersize=9, linewidth=3.0, label="AA")
    if "E[Wq]_PSFAQ" in curves:
        ax.plot(formula_lambdas, curves["E[Wq]_PSFAQ"],
                color=st_qba["color"], marker=st_qba["marker"],
                linestyle=st_qba["linestyle"], markersize=9,
                linewidth=3.0, label="QBA")
    ax.axvline(float(reference_load), linestyle="--", linewidth=1.8,
               color="#7E7E7E", alpha=0.85)

    ylim = auto_ylim([
        curves.get("E[Wq]_FCFS"), curves.get("E[Wq]_PSF"),
        curves.get("E[Wq]_PSFA"), curves.get("E[Wq]_PSFAQ"),
    ])
    apply_axes_style(
        ax, xlabel="\u03bb", ylabel="seconds",
        xlim=(float(np.min(formula_lambdas)), float(np.max(formula_lambdas))),
        ylim=ylim, show_legend=True,
    )
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{prefix}_formula_mean_waiting_vs_lambda.png")
    fig.savefig(out_path, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_formula_phi(
    out_dir: str,
    prefix: str,
    formula_lambdas: np.ndarray,
    phi: np.ndarray,
    phi_psfaq: np.ndarray | None,
    reference_load: float,
):
    setup_matplotlib()
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    st_aa = get_policy_style("PSFA")
    st_qba = get_policy_style("PSFAQ")

    ax.plot(formula_lambdas, phi, color=st_aa["color"], marker=st_aa["marker"],
            linestyle=st_aa["linestyle"], markersize=9, linewidth=3.0,
            label="AA")
    if phi_psfaq is not None:
        ax.plot(formula_lambdas, phi_psfaq, color=st_qba["color"],
                marker=st_qba["marker"], linestyle=st_qba["linestyle"],
                markersize=9, linewidth=3.0, label="QBA")
    ax.axvline(float(reference_load), linestyle="--", linewidth=1.8,
               color="#7E7E7E", alpha=0.85)
    apply_axes_style(
        ax, xlabel="\u03bb", ylabel="\u03c6",
        xlim=(float(np.min(formula_lambdas)), float(np.max(formula_lambdas))),
        ylim=(0.0, 1.0), show_legend=True,
    )
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{prefix}_formula_phi_vs_lambda.png")
    fig.savefig(out_path, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="fig2: lambda 扫参绘图 (SIM + formula + phi)")
    parser.add_argument("--data", type=str, default=os.path.join(_DEFAULT_PLOT_DIR, "fig2_data.npz"))
    parser.add_argument("--out-dir", type=str, default=_DEFAULT_PLOT_DIR)
    parser.add_argument("--prefix", type=str, default="fig2")
    parser.add_argument("--verbose-meta", action="store_true", default=False)
    args = parser.parse_args()

    data_path = os.path.abspath(args.data)
    out_dir = ensure_dir(os.path.abspath(args.out_dir))
    prefix = args.prefix

    z = np.load(data_path, allow_pickle=False)
    sim_lambdas = np.asarray(z["sim_lambdas"], dtype=np.float64)
    formula_lambdas = np.asarray(z["formula_lambdas"], dtype=np.float64)

    policies = ["FCFS", "PSF", "PSFA"]
    if "sim_mean_PSFAQ" in z.files or "sim_mean_PSFAQ_runs" in z.files:
        policies.append("PSFAQ")
    # RR removed from plots — temporary baseline, not part of theoretical analysis
    # if "sim_mean_RR" in z.files or "sim_mean_RR_runs" in z.files:
    #     policies.append("RR")
    sim_rows = {}
    for p in policies:
        row = {
            "mean": np.asarray(z[f"sim_mean_{p}"], dtype=np.float64) if f"sim_mean_{p}" in z.files else None,
            "p95": np.asarray(z[f"sim_p95_{p}"], dtype=np.float64) if f"sim_p95_{p}" in z.files else None,
            "p99": np.asarray(z[f"sim_p99_{p}"], dtype=np.float64) if f"sim_p99_{p}" in z.files else None,
        }
        if f"sim_mean_{p}_runs" in z.files:
            row["mean_runs"] = np.asarray(z[f"sim_mean_{p}_runs"], dtype=np.float64)
        if f"sim_p95_{p}_runs" in z.files:
            row["p95_runs"] = np.asarray(z[f"sim_p95_{p}_runs"], dtype=np.float64)
        if f"sim_p99_{p}_runs" in z.files:
            row["p99_runs"] = np.asarray(z[f"sim_p99_{p}_runs"], dtype=np.float64)
        sim_rows[p] = row

    curves = {
        "E[Wq]_FCFS": np.asarray(z["formula_EWq_FCFS"], dtype=np.float64),
        "E[Wq]_PSF": np.asarray(z["formula_EWq_PSF"], dtype=np.float64),
        "E[Wq]_PSFA": np.asarray(z["formula_EWq_PSFA"], dtype=np.float64),
    }
    if "formula_EWq_PSFAQ" in z.files:
        curves["E[Wq]_PSFAQ"] = np.asarray(z["formula_EWq_PSFAQ"], dtype=np.float64)
    phi = np.asarray(z["formula_phi_PSFA"], dtype=np.float64)
    phi_psfaq = np.asarray(z["formula_phi_PSFAQ"], dtype=np.float64) if "formula_phi_PSFAQ" in z.files else None
    reference_load = float(np.asarray(z["formula_reference_load"], dtype=np.float64).reshape(-1)[0])

    meta_path = os.path.join(os.path.dirname(data_path), f"{prefix}_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if args.verbose_meta:
            print("meta", meta)
        else:
            print("meta_summary", meta_summary(meta))

    sim_paths = _plot_sim(out_dir, prefix, sim_lambdas, sim_rows)
    formula_path = _plot_formula(out_dir, prefix, formula_lambdas, curves, reference_load)
    phi_path = _plot_formula_phi(out_dir, prefix, formula_lambdas, phi, phi_psfaq, reference_load)

    print(f"saved: {sim_paths['mean']}")
    print(f"saved: {sim_paths['p95']}")
    print(f"saved: {sim_paths['p99']}")
    print(f"saved: {formula_path}")
    print(f"saved: {phi_path}")


if __name__ == "__main__":
    main()
