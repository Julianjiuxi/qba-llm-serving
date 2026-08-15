"""实验脚本共享工具模块。

提供进度条、单次仿真运行、多 seed 扫参框架、参考策略运行、
npz+meta.json 保存等三个实验脚本共用的基础功能。
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Callable

import numpy as np

from llm_in_queuing.config import load_config
from llm_in_queuing.simulator import run


# ── 进度条 ──────────────────────────────────────────────────

def progress_update(done: int, total: int, stage: str) -> None:
    """输出进度行，覆写当前行。"""
    total = int(total)
    done = int(done)
    if total <= 0:
        return
    done = max(0, min(done, total))
    pct = 100.0 * float(done) / float(total)
    w = 28
    filled = int(round(float(w) * float(done) / float(total)))
    filled = max(0, min(filled, w))
    bar = "#" * filled + "-" * (w - filled)
    sys.stdout.write(f"\r[{stage}] {done}/{total} {pct:6.2f}% {bar}")
    sys.stdout.flush()


def progress_done(stage: str, total: int) -> None:
    """结束进度行输出。"""
    if int(total) > 0:
        progress_update(int(total), int(total), stage)
    sys.stdout.write("\n")
    sys.stdout.flush()


# ── 仿真运行 ────────────────────────────────────────────────

def run_one_sim(cfg: dict) -> dict:
    """用新引擎跑一次仿真，返回 metrics 字典。

    与旧 refine.py 的 run_simulation + 手动提取 waiting_time 等价，
    但返回完整的 metrics（waiting / E2E / TTFT / OOM / throughput 等）。
    """
    return run(dict(cfg))


def run_sim_stats(cfg: dict, overrides: dict | None = None) -> dict:
    """运行一次仿真并提取 waiting time 统计量。

    Args:
        cfg: 基础配置（不会被修改）
        overrides: 需要覆盖的字段（lambda_arrival, schedule_method, seed 等）

    Returns:
        {"mean": ..., "p95": ..., "p99": ..., plus per-class and error-rate keys}
    """
    local_cfg = dict(cfg)
    if overrides:
        local_cfg.update(overrides)
    metrics = run(local_cfg)
    return {
        "mean": float(metrics.get("waiting_time平均", float("nan"))),
        "p95": float(metrics.get("waiting_time p95", float("nan"))),
        "p99": float(metrics.get("waiting_time p99", float("nan"))),
        # per-class waiting
        "ts_mean": float(metrics.get("waiting_time_true_short_mean", float("nan"))),
        "ts_p99": float(metrics.get("waiting_time_true_short_p99", float("nan"))),
        "tl_mean": float(metrics.get("waiting_time_true_long_mean", float("nan"))),
        "tl_p99": float(metrics.get("waiting_time_true_long_p99", float("nan"))),
        # actual error rates
        "n_true_short": int(metrics.get("n_true_short", 0)),
        "n_true_long": int(metrics.get("n_true_long", 0)),
        "n_short_to_long": int(metrics.get("n_short_to_long", 0)),
        "n_long_to_short": int(metrics.get("n_long_to_short", 0)),
        "actual_epsilon_SL": float(metrics.get("actual_epsilon_SL", 0.0)),
        "actual_epsilon_LS": float(metrics.get("actual_epsilon_LS", 0.0)),
    }


# ── 扫参框架 ────────────────────────────────────────────────

def run_references(
    cfg: dict,
    lam: float,
    policies: list[str],
    seeds: list[int],
    *,
    extended: bool = False,
) -> dict:
    """运行参考策略（固定 lambda，多 seed）。

    Args:
        extended: 若 True，额外返回 per-class waiting 和 error rate 统计

    Returns:
        {policy: {"mean_runs": np.ndarray, "p95_runs": ..., "p99_runs": ..., ...}}
    """
    refs: dict = {}
    for policy in policies:
        mean_runs = []
        p95_runs = []
        p99_runs = []
        ts_mean_runs = []
        ts_p99_runs = []
        tl_mean_runs = []
        tl_p99_runs = []
        err_sl_runs = []
        err_ls_runs = []
        for seed in seeds:
            s = run_sim_stats(cfg, {"lambda_arrival": float(lam),
                                    "schedule_method": policy,
                                    "seed": int(seed)})
            mean_runs.append(s["mean"])
            p95_runs.append(s["p95"])
            p99_runs.append(s["p99"])
            ts_mean_runs.append(s["ts_mean"])
            ts_p99_runs.append(s["ts_p99"])
            tl_mean_runs.append(s["tl_mean"])
            tl_p99_runs.append(s["tl_p99"])
            err_sl_runs.append(s["actual_epsilon_SL"])
            err_ls_runs.append(s["actual_epsilon_LS"])
        entry = {
            "mean_runs": np.asarray(mean_runs, dtype=np.float64),
            "p95_runs": np.asarray(p95_runs, dtype=np.float64),
            "p99_runs": np.asarray(p99_runs, dtype=np.float64),
        }
        if extended:
            entry["ts_mean_runs"] = np.asarray(ts_mean_runs, dtype=np.float64)
            entry["ts_p99_runs"] = np.asarray(ts_p99_runs, dtype=np.float64)
            entry["tl_mean_runs"] = np.asarray(tl_mean_runs, dtype=np.float64)
            entry["tl_p99_runs"] = np.asarray(tl_p99_runs, dtype=np.float64)
            entry["actual_epsilon_SL_runs"] = np.asarray(err_sl_runs, dtype=np.float64)
            entry["actual_epsilon_LS_runs"] = np.asarray(err_ls_runs, dtype=np.float64)
        refs[policy] = entry
    return refs


def run_1d_sweep(
    cfg: dict,
    lam: float,
    policies: list[str],
    sweep_values: np.ndarray,
    seeds: list[int],
    *,
    extra_overrides: Callable[[float], dict] | None = None,
    stage_label: str = "SIM sweep",
) -> dict:
    """一维参数扫参。

    Args:
        cfg: 基础配置
        lam: 固定到达率
        policies: 扫参的策略列表
        sweep_values: 扫参值数组（如 gammas, quota_ks）
        seeds: seed 列表
        extra_overrides: 每个 sweep 值的额外配置覆写函数，签名为 f(sweep_val) -> dict
        stage_label: 进度条标签

    Returns:
        {policy: {"mean_runs": np.ndarray[n_seeds, n_sweep_vals],
                   "p95_runs": ..., "p99_runs": ...}}
    """
    n_seeds = len(seeds)
    n_points = len(sweep_values)
    out = {
        p: {
            "mean_runs": np.full((n_seeds, n_points), np.nan, dtype=np.float64),
            "p95_runs": np.full((n_seeds, n_points), np.nan, dtype=np.float64),
            "p99_runs": np.full((n_seeds, n_points), np.nan, dtype=np.float64),
        }
        for p in policies
    }
    total = n_points * len(policies) * n_seeds
    done = 0
    print(f"stage: {stage_label} runs={total} seeds={n_seeds} "
          f"points={n_points} policies={len(policies)}")
    for si, seed in enumerate(seeds):
        for li, val in enumerate(sweep_values.tolist()):
            overrides_base: dict = {"lambda_arrival": float(lam),
                                    "seed": int(seed)}
            if extra_overrides is not None:
                overrides_base.update(extra_overrides(float(val)))
            for p in policies:
                overrides_base["schedule_method"] = p
                s = run_sim_stats(cfg, overrides_base)
                out[p]["mean_runs"][si, li] = float(s["mean"])
                out[p]["p95_runs"][si, li] = float(s["p95"])
                out[p]["p99_runs"][si, li] = float(s["p99"])
                done += 1
                progress_update(done, total, stage_label)
    progress_done(stage_label, total)
    return out


# ── 结果保存 ────────────────────────────────────────────────

_CFG_SNAPSHOT_KEYS = [
    # 仿真规模
    "admission_number",
    "seed",
    # KV / 预算
    "budget_tokens",
    "request_block_size",
    "KV_total_capacity",
    "KV_safety_threshold",
    # 调度器参数
    "gamma",
    "quota_k",
    "ageing_measure",
    # 预测误差
    "label_threshold_percentail",
    "TN", "TP", "FN", "FP",
    # workload 分布参数
    "norm_ratio",
    "norm_loc",
    "norm_scale",
    "pareto_shape",
    "pareto_scale",
    "beta0",
    "beta1",
    "sigma",
    "prompt_len_cap",
    "answer_len_cap",
    # batch-time 参数（当前为 engine 硬编码，参见 batch_time_source 字段）
    "BT_BASE_C",
    "BT_SLOPE_A",
    "BT_KNEE_B0",
    "BT_CONC_GAMMA",
    "BT_JITTER_STD",
    "BT_MIN",
    # 稳态
    "steady_state",
]


def _cfg_snapshot(cfg: dict) -> dict:
    return {k: cfg.get(k) for k in _CFG_SNAPSHOT_KEYS}


def _get_git_info(repo_dir: str | None = None) -> dict:
    """获取当前 git 仓库信息（commit hash + dirty status）。"""
    cwd = repo_dir or os.path.dirname(__file__)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd, stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        commit = None
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=cwd, stderr=subprocess.DEVNULL, text=True,
        ).strip()
        dirty = bool(status)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        dirty = None
    return {
        "git_commit": commit,
        "git_dirty": dirty,
    }


def _compute_file_sha256(file_path: str) -> str:
    """计算文件 SHA256 校验和。"""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def save_results(
    plot_dir: str,
    prefix: str,
    arrays: dict,
    cfg: dict,
    args_dict: dict,
    *,
    config_path: str | None = None,
    extra_meta: dict | None = None,
) -> tuple[str, str]:
    """保存扫参结果为 npz + meta.json。

    Args:
        plot_dir: 输出目录
        prefix: 文件名前缀（如 fig3, fig4）
        arrays: 要保存的 numpy 数组字典
        cfg: 配置快照
        args_dict: argparse 参数字典
        config_path: YAML 配置文件路径（用于记录代码溯源）
        extra_meta: 额外要写入 meta.json 的内容

    Returns:
        (npz_path, meta_path)
    """
    os.makedirs(plot_dir, exist_ok=True)
    npz_path = os.path.join(plot_dir, f"{prefix}_data.npz")
    meta_path = os.path.join(plot_dir, f"{prefix}_meta.json")

    np.savez_compressed(npz_path, **arrays)
    npz_sha256 = _compute_file_sha256(npz_path)

    git_info = _get_git_info()

    meta = {
        "cfg_sim_snapshot": _cfg_snapshot(cfg),
        "batch_time_source": "engine_constants",
        "args": dict(args_dict),
        "git": git_info,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "npz_sha256": npz_sha256,
    }
    if config_path is not None:
        meta["config_path"] = os.path.abspath(config_path)
    if extra_meta:
        meta.update(extra_meta)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return npz_path, meta_path
