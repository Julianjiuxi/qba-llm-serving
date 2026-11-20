"""fig9: p–k 交互机制验证实验 — QBA 的标签阈值 p 与 quota k 的联合敏感性。

背景（由 fig7 引出的机制问题）:
    fig7 在固定 k=5 下发现 QBA 对 p 的响应存在结构性转折（mean–tail 折中）。
    本实验在同一 workload 上广播不同 k，验证该转折是否随 k 增大向更大的 p 移动，
    并按 quota 计数结构导出的 χ = p / (k*(1-p)) 重新排列横轴，检查不同 k 的曲线
    是否落在更一致的位置。

设计:
- 只扫 QBA：FCFS/PSF/AA 不使用 quota，与 k 无关；true-long 归属只由 (seed, p)
  决定、与 k 无关。因此无需为每个 k 重跑其他策略。
- k ∈ {3,5,9}，p ∈ {0.70,...,0.95}，各 k 使用同一批种子，便于配对比较。
- 种子 42–141（100 个，默认），与 fig7/fig8 对齐，便于 k=5 与 fig7 正式结果对账。
- 额外记录引擎已输出的 ``PSFAQ_promoted_total``，据此计算 promotion fraction =
  N_prom / N_L（Oracle 无误判时 N_L = n_true_long）。

运行时校验（由 _run_pk_sweep 强制）:
- 无误判：actual_epsilon_SL == actual_epsilon_LS == 0；
- n_true_short + n_true_long == admission_number 且 n_true_long > 0；
- promotion fraction ∈ [0, 1]；
- 同一 seed、同一 p 下真值 short/long 计数跨 k 一致（真值标签与 k 无关）。

注意: 本实验不改引擎/调度器/workload。PSFAQ_promoted_total 未由 ``_common.run_sim_stats``
转发，故在本地适配层直接调用 ``run_one_sim`` 提取，不改动旧实验接口。

用法:
    python -m scripts.experiment.sweep_pk
    python -m scripts.experiment.sweep_pk --quota-ks "3,5,9" --seed-count 100
    python -m scripts.experiment.sweep_pk --seed-count 2 --sim-admission-number 300  # smoke

输出:
    plot/fig9_data.npz + plot/fig9_meta.json
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from llm_in_queuing.config import load_config
from ._common import (
    progress_update,
    progress_done,
    run_one_sim,
    save_results,
)

# QBA 指标键（与 run_sim_stats 对齐，便于结果可比）
_METRICS_KEYS = [
    "mean", "p95", "p99",
    "ts_mean", "ts_p99",
    "tl_mean", "tl_p99",
    "actual_epsilon_SL", "actual_epsilon_LS",
]


def _parse_int_list(raw: str, name: str) -> np.ndarray:
    """解析正整数列表：> 0、无重复，按升序返回。

    只接受整数文本（``int(s)``）；不用 ``int(float(s))``，以免 "3.9" 被静默截断成 3。
    """
    parts = [s.strip() for s in str(raw).split(",") if s.strip()]
    if not parts:
        raise ValueError(f"{name} 不能为空")
    vals = []
    for s in parts:
        try:
            v = int(s)
        except ValueError as exc:
            raise ValueError(f"{name} 含非整数项: {s!r}") from exc
        if v <= 0:
            raise ValueError(f"{name} 必须为正整数，收到: {v}")
        vals.append(v)
    if len(set(vals)) != len(vals):
        raise ValueError(f"{name} 存在重复值: {parts}")
    return np.asarray(sorted(vals), dtype=np.int64)


def _parse_p_values(raw: str) -> np.ndarray:
    """解析 p 列表：有限、位于 (0, 1)、无重复，按升序返回。"""
    parts = [s.strip() for s in str(raw).split(",") if s.strip()]
    if not parts:
        raise ValueError("--p-values 不能为空")
    vals = [float(s) for s in parts]
    for v in vals:
        if not np.isfinite(v) or not (0.0 < v < 1.0):
            raise ValueError(f"非法 p: {v}（必须为有限且位于 (0, 1)）")
    if len(set(vals)) != len(vals):
        raise ValueError(f"--p-values 存在重复值: {parts}")
    return np.asarray(sorted(vals), dtype=np.float64)


def _extract_metrics(metrics: dict) -> dict:
    """从引擎 metrics 中提取 waiting 统计与 promotion 计数（适配层，不改旧接口）。"""
    return {
        "mean": float(metrics.get("waiting_time平均", float("nan"))),
        "p95": float(metrics.get("waiting_time p95", float("nan"))),
        "p99": float(metrics.get("waiting_time p99", float("nan"))),
        "ts_mean": float(metrics.get("waiting_time_true_short_mean", float("nan"))),
        "ts_p99": float(metrics.get("waiting_time_true_short_p99", float("nan"))),
        "tl_mean": float(metrics.get("waiting_time_true_long_mean", float("nan"))),
        "tl_p99": float(metrics.get("waiting_time_true_long_p99", float("nan"))),
        "actual_epsilon_SL": float(metrics.get("actual_epsilon_SL", 0.0)),
        "actual_epsilon_LS": float(metrics.get("actual_epsilon_LS", 0.0)),
        "promoted_total": float(metrics.get("PSFAQ_promoted_total", float("nan"))),
        "n_true_short": int(metrics.get("n_true_short", 0)),
        "n_true_long": int(metrics.get("n_true_long", 0)),
    }


def _run_pk_sweep(
    cfg: dict,
    lam: float,
    ks: np.ndarray,
    percentiles: np.ndarray,
    seeds: list[int],
) -> tuple[dict, np.ndarray]:
    """在 (k, p) 二维网格上扫参 QBA。

    Returns:
        (out, pi_s_runs)
        out: {key: np.ndarray[n_seeds, n_k, n_p]}，key 含 9 项 waiting 指标与
             promoted_total / promotion_fraction
        pi_s_runs: np.ndarray[n_seeds, n_p]，真值短类占比 π_S（仅随 p 变化）
    """
    n_seeds = len(seeds)
    n_k = len(ks)
    n_p = len(percentiles)
    shape = (n_seeds, n_k, n_p)
    admission_number = cfg.get("admission_number", None)

    out = {key: np.full(shape, np.nan, dtype=np.float64)
           for key in _METRICS_KEYS + ["promoted_total", "promotion_fraction"]}
    pi_s_runs = np.full((n_seeds, n_p), np.nan, dtype=np.float64)

    total = n_seeds * n_k * n_p
    done = 0
    print(f"stage: SIM p-k sweep runs={total} seeds={n_seeds} "
          f"ks={n_k} points={n_p}")
    for si, seed in enumerate(seeds):
        # 真值标签只由 (seed, p) 决定，与 k 无关：记录该 seed 下每个 p 的首个 k 的
        # short/long 计数，后续 k 必须一致（见下方校验）。
        ref_counts: dict[int, tuple[int, int]] = {}
        for ki, k in enumerate(ks.tolist()):
            for li, pct in enumerate(percentiles.tolist()):
                local = dict(cfg)
                local.update({
                    "lambda_arrival": float(lam),
                    "schedule_method": "QBA",
                    "seed": int(seed),
                    "quota_k": int(k),
                    "label_threshold_percentail": float(pct),
                })
                s = _extract_metrics(run_one_sim(local))

                n_ts = int(s["n_true_short"])
                n_l = int(s["n_true_long"])

                # --- Oracle 前提校验（本实验配置无误判，promotion fraction 的分母才有意义） ---
                if s["actual_epsilon_SL"] != 0.0 or s["actual_epsilon_LS"] != 0.0:
                    raise RuntimeError(
                        f"配置预期无误判，但出现预测误差: seed={seed} k={k} p={pct} "
                        f"eps_SL={s['actual_epsilon_SL']} eps_LS={s['actual_epsilon_LS']}"
                    )
                if admission_number is not None and (n_ts + n_l) != int(admission_number):
                    raise RuntimeError(
                        f"n_true_short + n_true_long != admission_number: seed={seed} "
                        f"k={k} p={pct} n_ts={n_ts} n_tl={n_l} adm={admission_number}"
                    )
                if n_l <= 0:
                    raise RuntimeError(
                        f"n_true_long 必须 > 0 才能定义 promotion fraction: "
                        f"seed={seed} k={k} p={pct} n_tl={n_l}"
                    )

                # --- 跨 k 计数一致性校验（同 seed、同 p 下真值标签不应随 k 变化） ---
                if li not in ref_counts:
                    ref_counts[li] = (n_ts, n_l)
                elif ref_counts[li] != (n_ts, n_l):
                    raise RuntimeError(
                        f"跨 k 真值计数不一致: seed={seed} p={pct} k={k} 得到 "
                        f"(n_ts,n_tl)=({n_ts},{n_l})，而首个 k 为 {ref_counts[li]}"
                    )

                for mk in _METRICS_KEYS:
                    out[mk][si, ki, li] = float(s[mk])
                out["promoted_total"][si, ki, li] = float(s["promoted_total"])

                frac = float(s["promoted_total"]) / float(n_l)
                if not (0.0 <= frac <= 1.0):
                    raise RuntimeError(
                        f"promotion_fraction 越界 [0,1]: seed={seed} k={k} p={pct} "
                        f"frac={frac} (promoted={s['promoted_total']} n_tl={n_l})"
                    )
                out["promotion_fraction"][si, ki, li] = frac

                pi_s_runs[si, li] = n_ts / float(n_ts + n_l)
                done += 1
                progress_update(done, total, "SIM p-k sweep")
    progress_done("SIM p-k sweep", total)
    return out, pi_s_runs


def main():
    parser = argparse.ArgumentParser(description="fig9: p–k 交互机制验证（QBA）")
    parser.add_argument("--quota-ks", type=str, default="3,5,9",
                        help="逗号分隔的 quota_k 列表，例如 \"3,5,9\"")
    parser.add_argument("--p-values", type=str, default="0.70,0.75,0.80,0.85,0.90,0.95",
                        help="逗号分隔的 p 列表，显式指定以避免浮点步长误差")
    parser.add_argument("--reference-load", type=float, default=32.0)
    parser.add_argument("--gamma", type=float, default=0.80)
    parser.add_argument("--plot-dir", type=str, default=os.path.join(
        os.path.dirname(__file__), "..", "plot"))
    parser.add_argument("--out-prefix", type=str, default="fig9")
    parser.add_argument("--sim-admission-number", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--seed-count", type=int, default=100,
                        help="种子数量，默认 100（42-141，与 fig7/fig8 对齐）；smoke 时用较小值")
    args = parser.parse_args()

    # 加载基础配置
    cfg_path = Path(__file__).resolve().parent / "configs" / "sweep_pk_base.yaml"
    cfg = load_config(cfg_path)
    if args.sim_admission_number is not None:
        cfg["admission_number"] = int(args.sim_admission_number)
    # 固定老化参数，仅让 (k, p) 变化（控制变量）
    cfg["gamma"] = float(args.gamma)

    plot_dir = os.path.abspath(args.plot_dir)

    reference_lambda = float(args.reference_load)
    quota_ks = _parse_int_list(args.quota_ks, "--quota-ks")
    percentiles = _parse_p_values(args.p_values)
    seeds = [int(args.seed_start) + i for i in range(int(args.seed_count))]

    print(f"stage: SIM p-k sweep quota_ks={quota_ks.tolist()} "
          f"p={percentiles.tolist()} lambda={reference_lambda} "
          f"gamma={cfg['gamma']} repeats={len(seeds)}")
    out, pi_s_runs = _run_pk_sweep(cfg, reference_lambda, quota_ks, percentiles, seeds)

    # 保存输出（QBA 各指标统一为 [n_seeds, n_k, n_p]）
    arrays = {
        "reference_lambda": np.asarray([float(reference_lambda)], dtype=np.float64),
        "quota_ks": np.asarray(quota_ks, dtype=np.int64),
        "percentiles": np.asarray(percentiles, dtype=np.float64),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "pi_s_runs": pi_s_runs,
        "psfaq_promoted_total_runs": out["promoted_total"],
        "psfaq_promotion_fraction_runs": out["promotion_fraction"],
    }
    for mk in _METRICS_KEYS:
        arrays[f"psfaq_{mk}_runs"] = out[mk]

    extra_meta = {
        "experiment": "p_k_interaction",
        "sweep_policies": ["QBA"],
        "quota_ks": [int(k) for k in quota_ks.tolist()],
        "percentiles": [float(v) for v in percentiles.tolist()],
        "chi_formula": "chi = p / (k * (1 - p))",
        "promotion_fraction_definition": "PSFAQ_promoted_total / n_true_long (Oracle 无误判时等于 /N_pred_long)",
        "answers_shared_across_k": True,
    }

    npz_path, meta_path = save_results(
        plot_dir, str(args.out_prefix), arrays, cfg,
        vars(args), config_path=str(cfg_path), extra_meta=extra_meta,
    )
    print(f"saved data: {npz_path}")
    print(f"saved meta: {meta_path}")
    print(f"next: python -m scripts.draw.draw_fig9 --data \"{npz_path}\"")


if __name__ == "__main__":
    main()
