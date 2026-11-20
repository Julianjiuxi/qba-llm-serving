"""fig8: prompt 长度缩放（prompt_scale）扫参实验 — 观察工作负载尺度对调度的影响。

在固定引擎、到达率、answer 长度与预测标签的条件下，对基础 workload 的 prompt
长度做确定性缩放:

    P_i^(a) = min(prompt_len_cap, max(5, round(a * P_i)))

设计要点:
- 缩放发生在"基础请求生成 + 标签分配"之后，且不使用随机数，因此同一 seed 在所有
  倍率下共享同一条基础随机流；answer 长度、到达时间、真值/预测标签均保持不变。
- a=1 时直接保留原始整数长度（不引入舍入），可与未适配的原始仿真逐位对齐。
- 参与扫参的策略：FCFS / PSF / AA / QBA（prompt 缩放会改变整体 workload，故
  FCFS 的整体指标也随倍率变化，需一并扫参）。
- RR 为内部策略，本轮不运行、不绘图。

实现方式: 不修改 workload.py / 引擎，仅在实验脚本内部临时替换
``simulator.generate_heapq``，对原始生成器返回的请求堆做后处理，并用 ``finally``
恢复原函数。注意这是模块级替换，同一进程内不要并行跑不同倍率。

用法:
    python -m scripts.experiment.sweep_prompt_scale
    python -m scripts.experiment.sweep_prompt_scale --prompt-scales "0.25,0.5,1,2,4"
    python -m scripts.experiment.sweep_prompt_scale --seed-count 2 --sim-admission-number 300

输出:
    plot/fig8_data.npz + plot/fig8_meta.json
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
    run_sim_stats,
    save_results,
)

# prompt 缩放的硬下限（与 workload.py 生成阶段保持一致）
_PROMPT_LOWER_BOUND = 5

_METRICS_KEYS = [
    "mean_runs", "p95_runs", "p99_runs",
    "ts_mean_runs", "ts_p99_runs",
    "tl_mean_runs", "tl_p99_runs",
    "actual_epsilon_SL_runs", "actual_epsilon_LS_runs",
]

# 与策略无关的 workload 诊断（同一 (seed, scale) 只记录一次）
_DIAG_KEYS = [
    "prompt_mean", "answer_mean", "token_ratio",
    "prompt_lower_clip_fraction", "prompt_upper_clip_fraction",
]


def _parse_prompt_scales(raw: str) -> np.ndarray:
    """解析倍率列表：有限、严格大于 0、无重复，按升序返回。"""
    parts = [s.strip() for s in str(raw).split(",") if s.strip()]
    if not parts:
        raise ValueError("--prompt-scales 不能为空")
    scales = [float(s) for s in parts]
    for a in scales:
        if not np.isfinite(a) or a <= 0.0:
            raise ValueError(f"非法 prompt_scale: {a}（必须为有限且 > 0）")
    if len(set(scales)) != len(scales):
        raise ValueError(f"--prompt-scales 存在重复值: {parts}")
    return np.asarray(sorted(scales), dtype=np.float64)


def _apply_prompt_scale(
    heap: list,
    scale: float,
    prompt_cap: int,
    diag: dict | None = None,
) -> None:
    """就地缩放请求堆中每条请求的 prompt 长度，并写入诊断统计。

    只更新 ``prompt_len`` 与 ``prefill_res_token``，不触碰到达时间、answer 长度、
    真值/预测标签。诊断中 token_ratio 定义为 mean(prompt) / mean(answer)。
    """
    cap = int(prompt_cap)
    n = 0
    sum_prompt = 0.0
    sum_answer = 0.0
    lower_clip = 0
    upper_clip = 0
    for _, _, req in heap:
        p = int(req.prompt_len)
        raw = p if scale == 1.0 else int(round(float(scale) * float(p)))
        if raw < _PROMPT_LOWER_BOUND:
            lower_clip += 1
        if raw > cap:
            upper_clip += 1
        new_p = min(cap, max(_PROMPT_LOWER_BOUND, raw))
        req.prompt_len = new_p
        req.prefill_res_token = new_p
        sum_prompt += new_p
        sum_answer += int(req.answer_len)
        n += 1
    if n > 0 and diag is not None:
        mean_prompt = sum_prompt / n
        mean_answer = sum_answer / n
        diag["prompt_mean"] = mean_prompt
        diag["answer_mean"] = mean_answer
        diag["token_ratio"] = (mean_prompt / mean_answer) if mean_answer > 0 else float("nan")
        diag["prompt_lower_clip_fraction"] = lower_clip / n
        diag["prompt_upper_clip_fraction"] = upper_clip / n


def _make_scaled_generator(original_generate, scale: float, diag: dict):
    """返回一个包装后的生成器：先产出基础请求堆，再做确定性的 prompt 缩放。"""
    def _scaled_generate(cfg: dict):
        heap = original_generate(cfg)
        cap = int(cfg.get("prompt_len_cap", 10 ** 9))
        _apply_prompt_scale(heap, scale, cap, diag)
        return heap

    return _scaled_generate


def _run_prompt_scale_sweep(
    cfg: dict,
    lam: float,
    scales: np.ndarray,
    policies: list[str],
    seeds: list[int],
) -> tuple[dict, dict]:
    """按 prompt_scale 扫参多个策略。

    Returns:
        (out, diag)
        out: {policy: {"mean_runs": np.ndarray[n_seeds, n_points], ...}}
        diag: {key: np.ndarray[n_seeds, n_points]}，与策略无关的 workload 诊断
    """
    n_seeds = len(seeds)
    n_points = len(scales)

    out = {
        p: {k: np.full((n_seeds, n_points), np.nan, dtype=np.float64)
            for k in _METRICS_KEYS}
        for p in policies
    }
    diag = {k: np.full((n_seeds, n_points), np.nan, dtype=np.float64)
            for k in _DIAG_KEYS}

    original_generate = simulator.generate_heapq
    total = n_points * len(policies) * n_seeds
    done = 0
    print(f"stage: SIM prompt-scale sweep runs={total} seeds={n_seeds} "
          f"points={n_points} policies={len(policies)}")
    for si, seed in enumerate(seeds):
        for li, a in enumerate(scales.tolist()):
            scale = float(a)
            cell_diag: dict = {}
            simulator.generate_heapq = _make_scaled_generator(original_generate, scale, cell_diag)
            try:
                for p in policies:
                    try:
                        s = run_sim_stats(cfg, {
                            "lambda_arrival": float(lam),
                            "schedule_method": p,
                            "seed": int(seed),
                        })
                    except Exception as exc:  # 保留原异常，补充实验上下文
                        raise RuntimeError(
                            f"Experiment failed: seed={seed} "
                            f"prompt_scale={scale} policy={p}"
                        ) from exc
                    out[p]["mean_runs"][si, li] = float(s["mean"])
                    out[p]["p95_runs"][si, li] = float(s["p95"])
                    out[p]["p99_runs"][si, li] = float(s["p99"])
                    out[p]["ts_mean_runs"][si, li] = float(s["ts_mean"])
                    out[p]["ts_p99_runs"][si, li] = float(s["ts_p99"])
                    out[p]["tl_mean_runs"][si, li] = float(s["tl_mean"])
                    out[p]["tl_p99_runs"][si, li] = float(s["tl_p99"])
                    out[p]["actual_epsilon_SL_runs"][si, li] = float(s["actual_epsilon_SL"])
                    out[p]["actual_epsilon_LS_runs"][si, li] = float(s["actual_epsilon_LS"])
                    done += 1
                    progress_update(done, total, "SIM prompt-scale sweep")
                    if p == policies[0]:
                        # 诊断与策略无关，取首次调用（FCFS）收集到的值
                        for dk in _DIAG_KEYS:
                            diag[dk][si, li] = float(cell_diag.get(dk, float("nan")))
            finally:
                simulator.generate_heapq = original_generate
    progress_done("SIM prompt-scale sweep", total)
    return out, diag


def main():
    parser = argparse.ArgumentParser(description="fig8: prompt_scale 扫参")
    parser.add_argument("--prompt-scales", type=str, default="0.25,0.5,1,2,4")
    parser.add_argument("--reference-load", type=float, default=32.0)
    parser.add_argument("--gamma", type=float, default=0.80)
    parser.add_argument("--quota-k", type=int, default=5)
    parser.add_argument("--plot-dir", type=str, default=os.path.join(
        os.path.dirname(__file__), "..", "plot"))
    parser.add_argument("--out-prefix", type=str, default="fig8")
    parser.add_argument("--sim-admission-number", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--seed-count", type=int, default=100)
    args = parser.parse_args()

    # 加载基础配置
    cfg_path = Path(__file__).resolve().parent / "configs" / "sweep_prompt_scale_base.yaml"
    cfg = load_config(cfg_path)
    if args.sim_admission_number is not None:
        cfg["admission_number"] = int(args.sim_admission_number)
    # 固定老化相关参数，仅让 prompt_scale 变化（控制变量）
    cfg["gamma"] = float(args.gamma)
    cfg["quota_k"] = int(args.quota_k)

    plot_dir = os.path.abspath(args.plot_dir)

    reference_lambda = float(args.reference_load)
    scales = _parse_prompt_scales(args.prompt_scales)
    seeds = [int(args.seed_start) + i for i in range(int(args.seed_count))]

    # 参与扫参的策略：prompt 缩放改变整体 workload，FCFS 整体指标亦变化，一并扫参
    sweep_policies = ["FCFS", "PSF", "AA", "QBA"]

    print(f"stage: SIM prompt-scale sweep scales={scales.tolist()} lambda={reference_lambda} "
          f"gamma={cfg['gamma']} quota_k={cfg['quota_k']} repeats={len(seeds)}")
    rows, diag = _run_prompt_scale_sweep(cfg, reference_lambda, scales, sweep_policies, seeds)

    # 保存输出
    arrays = {
        "reference_lambda": np.asarray([float(reference_lambda)], dtype=np.float64),
        "prompt_scales": np.asarray(scales, dtype=np.float64),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "prompt_mean_runs": diag["prompt_mean"],
        "answer_mean_runs": diag["answer_mean"],
        "token_ratio_runs": diag["token_ratio"],
        "prompt_lower_clip_fraction_runs": diag["prompt_lower_clip_fraction"],
        "prompt_upper_clip_fraction_runs": diag["prompt_upper_clip_fraction"],
    }

    for p, prefix in [("FCFS", "fcfs"), ("PSF", "psf"), ("AA", "psfa"), ("QBA", "psfaq")]:
        for mk in ["mean", "p95", "p99", "ts_mean", "ts_p99", "tl_mean", "tl_p99"]:
            arrays[f"{prefix}_{mk}_runs"] = rows[p][f"{mk}_runs"]
        arrays[f"{prefix}_actual_epsilon_SL_runs"] = rows[p]["actual_epsilon_SL_runs"]
        arrays[f"{prefix}_actual_epsilon_LS_runs"] = rows[p]["actual_epsilon_LS_runs"]

    extra_meta = {
        "experiment": "prompt_scale",
        "sweep_policies": sweep_policies,
        "ref_policies": [],
        "prompt_scales": np.asarray(scales, dtype=np.float64).tolist(),
        "scaling_stage": "after_base_workload_generation_and_label_assignment",
        "scaling_rule": "round_then_clip",
        "scaling_formula": "P^(a) = min(prompt_len_cap, max(5, round(a * P)))",
        "answer_preserved": True,
        "labels_preserved": True,
    }

    npz_path, meta_path = save_results(
        plot_dir, str(args.out_prefix), arrays, cfg,
        vars(args), config_path=str(cfg_path), extra_meta=extra_meta,
    )
    print(f"saved data: {npz_path}")
    print(f"saved meta: {meta_path}")
    print(f"next: python draw_fig8.py --data \"{npz_path}\"")


if __name__ == "__main__":
    main()
