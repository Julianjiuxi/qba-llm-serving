"""fig13: 跨负载的构成扫描（fig11 的小型迁移验证）。

背景（由 fig11 引出）:
    fig11 在固定到达率 λ = 32 下，仅改变 prefill/decode 构成（逐请求总 token 守恒），
    就已观察到显著性能差异，甚至出现随倍率 a 非单调的趋势。但单负载不能说明该现象是否
    只是该工作点的偶然。本实验把 fig11 的生成方式固定，额外扫描到达率 λ，做一个**小型
    迁移验证**：

        a ∈ {0.5, 1, 2, 4} × λ ∈ {26, 32, 38}（p = 0.85，oracle 标签）。

研究问题:
    fig11 的构成效应与非单调趋势，在不同负载下是否仍然存在？策略排序是否稳定？
    注意：fig11 观察到的等待时间"回落"发生在 a=2→4，因此默认倍率需覆盖到 a=4，
    否则跨负载只验证了构成效应是否存在，无法判断回落段是否保留。

设计:
- 网格：构成倍率 a × 到达率 λ × 策略 × 种子；p / 误差不参与扫描（oracle）。
  默认 a ∈ {0.5, 1, 2, 4}，λ ∈ {26, 32, 38}。
- 主数组形状统一为 [n_seeds, n_a, n_lambda]。
- a=1 且 λ=32 的单元应与 fig11 的 a=1、λ=32 逐位一致（锚点）。
- 只改请求的 P/D 与标签，不改到达时间与 id；守恒变换不消耗随机数（由单测保证）。
- 实现方式与 fig11 一致：模块级临时替换 ``simulator.generate_heapq``，每个 (seed, λ, a)
  单元替换一次、四个策略共用同一份变换，跑完在 ``finally`` 恢复原函数。注意这是模块级
  替换，同一进程内不要并行跑不同单元（云端通过独立子进程按 seed 并行）。

不变量声明:
    逐请求总 token 相同并不保证实际负载相同。跨负载对照只用于验证构成效应是否迁移，
    不用于分离"构成效应"与"拥塞效应"的定量贡献。

用法:
    python -m scripts.experiment.sweep_composition_load
    python -m scripts.experiment.sweep_composition_load \
        --composition-scales "0.5,1,2,4" --lambdas "26,32,38" --seed-count 2 \
        --sim-admission-number 300 --plot-dir scripts/plot/smoke_fig13

输出:
    plot/fig13_data.npz + plot/fig13_meta.json
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from llm_in_queuing import simulator
from llm_in_queuing.config import load_config
from ._common import (
    progress_update,
    progress_done,
    run_one_sim,
    save_results,
)
from .sweep_request_composition import (
    _extract_metrics,
    _make_request_composition_generator,
    _parse_composition_scales,
    _METRICS_KEYS,
    _POLICY_DIAG_KEYS,
    _WORK_DIAG_KEYS,
)


def _parse_lambdas(raw: str) -> np.ndarray:
    """解析到达率列表：有限、严格大于 0、无重复，按升序返回。"""
    parts = [s.strip() for s in str(raw).split(",") if s.strip()]
    if not parts:
        raise ValueError("--lambdas 不能为空")
    vals = [float(s) for s in parts]
    for lam in vals:
        if not np.isfinite(lam) or lam <= 0.0:
            raise ValueError(f"非法 lambda: {lam}（必须为有限且 > 0）")
    if len(set(vals)) != len(vals):
        raise ValueError(f"--lambdas 存在重复值: {parts}")
    return np.asarray(sorted(vals), dtype=np.float64)


def _run_composition_load_sweep(
    cfg: dict,
    scales: np.ndarray,
    loads: np.ndarray,
    policies: list[str],
    seeds: list[int],
) -> tuple[dict, dict, dict]:
    """在 (a, λ) 网格上扫参多个策略（逐请求总 token 守恒）。

    Returns:
        (out, policy_diag, work_diag)
        out: {policy: {metric: np.ndarray[n_seeds, n_a, n_lambda]}}
        policy_diag: {policy: {diag_key: np.ndarray[...]}}
        work_diag: {diag_key: np.ndarray[...]}（与策略无关）
    """
    n_seeds = len(seeds)
    n_a = len(scales)
    n_l = len(loads)
    shape = (n_seeds, n_a, n_l)

    out = {
        p: {k: np.full(shape, np.nan, dtype=np.float64) for k in _METRICS_KEYS}
        for p in policies
    }
    policy_diag = {
        p: {k: np.full(shape, np.nan, dtype=np.float64) for k in _POLICY_DIAG_KEYS}
        for p in policies
    }
    work_diag = {k: np.full(shape, np.nan, dtype=np.float64)
                 for k in _WORK_DIAG_KEYS}

    original_generate = simulator.generate_heapq
    total = n_seeds * n_a * n_l * len(policies)
    done = 0
    print(f"stage: SIM composition-load sweep runs={total} seeds={n_seeds} "
          f"a={n_a} lambda={n_l} policies={len(policies)}")
    for si, seed in enumerate(seeds):
        for li, lam in enumerate(loads.tolist()):
            # 该 (seed, λ) 下的基准真值标签参照（a=1 原始生成器）
            base_cfg = dict(cfg)
            base_cfg.update({"lambda_arrival": float(lam), "seed": int(seed)})
            base_heap = original_generate(base_cfg)
            base_labels = {r.id: r.true_label for _, _, r in base_heap}

            for ai, a in enumerate(scales.tolist()):
                cell_diag: dict = {}
                simulator.generate_heapq = _make_request_composition_generator(
                    original_generate, float(a), cell_diag, base_labels,
                )
                try:
                    for pol in policies:
                        local = dict(cfg)
                        local.update({
                            "lambda_arrival": float(lam),
                            "schedule_method": pol,
                            "seed": int(seed),
                        })
                        try:
                            s = _extract_metrics(run_one_sim(local))
                        except Exception as exc:  # 保留原异常，补充实验上下文
                            raise RuntimeError(
                                f"Experiment failed: seed={seed} a={a} "
                                f"lambda={lam} policy={pol}"
                            ) from exc
                        for mk in _METRICS_KEYS:
                            out[pol][mk][si, ai, li] = float(s[mk])
                        for dk in _POLICY_DIAG_KEYS:
                            policy_diag[pol][dk][si, ai, li] = float(s[dk])
                        done += 1
                        progress_update(done, total, "SIM composition-load")
                finally:
                    simulator.generate_heapq = original_generate
                # workload 诊断与策略无关，单元跑完后记录一次
                for dk in _WORK_DIAG_KEYS:
                    work_diag[dk][si, ai, li] = float(
                        cell_diag.get(dk, float("nan")))
    progress_done("SIM composition-load", total)
    return out, policy_diag, work_diag


def main():
    parser = argparse.ArgumentParser(
        description="fig13: 跨负载的构成扫描（fig11 迁移验证）")
    parser.add_argument("--composition-scales", type=str, default="0.5,1,2,4",
                        help="逗号分隔的构成倍率列表（默认覆盖到 a=4 以观察回落段）")
    parser.add_argument("--lambdas", type=str, default="26,32,38",
                        help="逗号分隔的到达率列表")
    parser.add_argument("--gamma", type=float, default=0.80)
    parser.add_argument("--quota-k", type=int, default=5)
    parser.add_argument("--plot-dir", type=str, default=os.path.join(
        os.path.dirname(__file__), "..", "plot"))
    parser.add_argument("--out-prefix", type=str, default="fig13")
    parser.add_argument("--sim-admission-number", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--seed-count", type=int, default=100)
    args = parser.parse_args()

    cfg_path = Path(__file__).resolve().parent / "configs" / "sweep_composition_load_base.yaml"
    cfg = load_config(cfg_path)
    if args.sim_admission_number is not None:
        cfg["admission_number"] = int(args.sim_admission_number)
    # 固定老化相关参数，仅让 (a, λ) 变化（控制变量）
    cfg["gamma"] = float(args.gamma)
    cfg["quota_k"] = int(args.quota_k)

    plot_dir = os.path.abspath(args.plot_dir)

    scales = _parse_composition_scales(args.composition_scales)
    loads = _parse_lambdas(args.lambdas)
    seeds = [int(args.seed_start) + i for i in range(int(args.seed_count))]

    # 构成与负载都会改变整体 workload，FCFS 一并扫参作为相对基线
    sweep_policies = ["FCFS", "PSF", "AA", "QBA"]

    print(f"stage: SIM composition-load sweep scales={scales.tolist()} "
          f"lambdas={loads.tolist()} gamma={cfg['gamma']} quota_k={cfg['quota_k']} "
          f"repeats={len(seeds)}")
    rows, policy_diag, work_diag = _run_composition_load_sweep(
        cfg, scales, loads, sweep_policies, seeds,
    )

    arrays = {
        "composition_scales": np.asarray(scales, dtype=np.float64),
        "lambdas": np.asarray(loads, dtype=np.float64),
        "seeds": np.asarray(seeds, dtype=np.int64),
    }
    for dk in _WORK_DIAG_KEYS:
        arrays[f"{dk}_runs"] = work_diag[dk]

    for p, prefix in [("FCFS", "fcfs"), ("PSF", "psf"), ("AA", "psfa"), ("QBA", "psfaq")]:
        for mk in _METRICS_KEYS:
            arrays[f"{prefix}_{mk}_runs"] = rows[p][mk]
        for dk in _POLICY_DIAG_KEYS:
            arrays[f"{prefix}_{dk}_runs"] = policy_diag[p][dk]

    extra_meta = {
        "experiment": "composition_load_migration",
        "sweep_policies": sweep_policies,
        "composition_scales": [float(v) for v in scales.tolist()],
        "lambdas": [float(v) for v in loads.tolist()],
        "label_threshold_percentail": float(cfg["label_threshold_percentail"]),
        "normalization": "exact per-request total-token preservation (fig11 transform)",
        "answer_preserved": False,
        "labels_recomputed": "oracle quantile on transformed answers",
        "identity_anchor": "a=1, lambda=32 (matches fig11 a=1, lambda=32)",
        "invariant_note": (
            "per-request total tokens are preserved; only the prefill/decode split "
            "changes with a, while lambda governs the arrival load"
        ),
    }

    npz_path, meta_path = save_results(
        plot_dir, str(args.out_prefix), arrays, cfg,
        vars(args), config_path=str(cfg_path), extra_meta=extra_meta,
    )
    print(f"saved data: {npz_path}")
    print(f"saved meta: {meta_path}")
    print(f"next: python -m scripts.draw.draw_fig13 --data \"{npz_path}\"")


if __name__ == "__main__":
    main()
