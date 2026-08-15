"""fig4 绘图：gamma 扫参 — AA/QBA 在不同老化阈值下的 waiting time。"""

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


def _plot_panel(
    out_dir: str,
    prefix: str,
    gammas: np.ndarray,
    psfa_runs: np.ndarray,
    psfaq_runs: np.ndarray,
    refs_runs: dict,
    key: str,
    fname: str,
):
    setup_matplotlib()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    psfa_m, psfa_lo, psfa_hi = mean_ci(psfa_runs)
    psfaq_m, psfaq_lo, psfaq_hi = mean_ci(psfaq_runs)
    st_aa = get_policy_style("PSFA")
    st_qba = get_policy_style("PSFAQ")
    st_fcfs = get_policy_style("FCFS")
    st_psf = get_policy_style("PSF")

    ax.plot(gammas, psfa_m, color=st_aa["color"], marker=st_aa["marker"],
            markersize=9, linestyle=st_aa["linestyle"], linewidth=3.0,
            label=display_policy_name("PSFA"))
    fill_ci(ax, gammas, psfa_lo, psfa_hi, st_aa["color"], alpha=0.16)
    ax.plot(gammas, psfaq_m, color=st_qba["color"], marker=st_qba["marker"],
            markersize=9, linestyle=st_qba["linestyle"], linewidth=3.0,
            label=display_policy_name("PSFAQ"))
    fill_ci(ax, gammas, psfaq_lo, psfaq_hi, st_qba["color"])

    fcfs_m, _, _ = mean_ci(refs_runs["FCFS"][key])
    psf_m, _, _ = mean_ci(refs_runs["PSF"][key])

    ax.axhline(float(fcfs_m.reshape(-1)[0]), color=st_fcfs["color"],
               linestyle=st_fcfs["linestyle"], linewidth=3.0, label="FCFS")
    ax.axhline(float(psf_m.reshape(-1)[0]), color=st_psf["color"],
               linestyle=st_psf["linestyle"], linewidth=3.0, label="PSF")

    ylim_ref_vals = [fcfs_m[0], psf_m[0]]
    # RR removed from plots — temporary baseline, not part of theoretical analysis
    # if "RR" in refs_runs:
    #     st_rr = get_policy_style("RR")
    #     rr_m, _, _ = mean_ci(refs_runs["RR"][key])
    #     rr_val = float(rr_m.reshape(-1)[0])
    #     ylim_ref_vals.append(rr_val)
    #     ax.axhline(rr_val, color=st_rr["color"],
    #                linestyle=st_rr["linestyle"], linewidth=3.0,
    #                label="RR")

    ylim = auto_ylim([psfa_m, psfaq_m,
                      psfa_lo if psfa_lo is not None else psfa_m,
                      psfaq_lo if psfaq_lo is not None else psfaq_m,
                      np.asarray(ylim_ref_vals)])
    apply_axes_style(
        ax, xlabel="\u03b3", ylabel="seconds",
        xlim=(float(np.min(gammas)), float(np.max(gammas))),
        ylim=ylim, show_legend=True,
    )
    fig.tight_layout()
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="fig4: gamma 扫参绘图")
    parser.add_argument("--data", type=str, default=os.path.join(_DEFAULT_PLOT_DIR, "fig4_data.npz"))
    parser.add_argument("--out-dir", type=str, default=_DEFAULT_PLOT_DIR)
    parser.add_argument("--prefix", type=str, default="fig4")
    parser.add_argument("--verbose-meta", action="store_true", default=False)
    args = parser.parse_args()

    data_path = os.path.abspath(args.data)
    out_dir = ensure_dir(os.path.abspath(args.out_dir))
    prefix = args.prefix

    z = np.load(data_path, allow_pickle=False)
    gammas = np.asarray(z["gammas"], dtype=np.float64)

    psfa_mean_runs = np.asarray(z["psfa_mean_runs"], dtype=np.float64)
    psfa_p95_runs = np.asarray(z["psfa_p95_runs"], dtype=np.float64)
    psfa_p99_runs = np.asarray(z["psfa_p99_runs"], dtype=np.float64)

    psfaq_mean_runs = np.asarray(z["psfaq_mean_runs"], dtype=np.float64)
    psfaq_p95_runs = np.asarray(z["psfaq_p95_runs"], dtype=np.float64)
    psfaq_p99_runs = np.asarray(z["psfaq_p99_runs"], dtype=np.float64)

    refs_runs = {
        "FCFS": {
            "mean": np.asarray(z["ref_fcfs_mean_runs"], dtype=np.float64),
            "p95": np.asarray(z["ref_fcfs_p95_runs"], dtype=np.float64),
            "p99": np.asarray(z["ref_fcfs_p99_runs"], dtype=np.float64),
        },
        "PSF": {
            "mean": np.asarray(z["ref_psf_mean_runs"], dtype=np.float64),
            "p95": np.asarray(z["ref_psf_p95_runs"], dtype=np.float64),
            "p99": np.asarray(z["ref_psf_p99_runs"], dtype=np.float64),
        },
    }
    # RR removed from plots — temporary baseline, not part of theoretical analysis
    # has_rr = "ref_rr_mean_runs" in z.files
    # if has_rr:
    #     refs_runs["RR"] = {
    #         "mean": np.asarray(z["ref_rr_mean_runs"], dtype=np.float64),
    #         "p95": np.asarray(z["ref_rr_p95_runs"], dtype=np.float64),
    #         "p99": np.asarray(z["ref_rr_p99_runs"], dtype=np.float64),
    #     }

    meta_path = os.path.join(os.path.dirname(data_path), f"{prefix}_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if args.verbose_meta:
            print("meta", meta)
        else:
            print("meta_summary", meta_summary(meta))

    mean_path = _plot_panel(out_dir, prefix, gammas, psfa_mean_runs, psfaq_mean_runs,
                            refs_runs, "mean", f"{prefix}_sim_waiting_mean_vs_gamma.png")
    p95_path = _plot_panel(out_dir, prefix, gammas, psfa_p95_runs, psfaq_p95_runs,
                           refs_runs, "p95", f"{prefix}_sim_waiting_p95_vs_gamma.png")
    p99_path = _plot_panel(out_dir, prefix, gammas, psfa_p99_runs, psfaq_p99_runs,
                           refs_runs, "p99", f"{prefix}_sim_waiting_p99_vs_gamma.png")

    print(f"saved: {mean_path}")
    print(f"saved: {p95_path}")
    print(f"saved: {p99_path}")


if __name__ == "__main__":
    main()
