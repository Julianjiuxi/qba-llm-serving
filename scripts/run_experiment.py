"""一键运行仿真实验的入口脚本。

默认使用 configs/conference_default.yaml 配置文件，
也可以通过命令行参数指定其他配置文件路径。

用法:
    python scripts/run_experiment.py                           # 默认配置
    python scripts/run_experiment.py configs/my_config.yaml    # 指定配置
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from llm_in_queuing.config import load_config
from llm_in_queuing.simulator import run
from llm_in_queuing.types import EngineState


def _method_filename_prefix(cfg: dict) -> str:
    """根据调度方法生成文件前缀，与绘图命名保持一致。"""
    method = str(cfg.get("schedule_method", "Unknown"))
    ageing = str(cfg.get("ageing_measure", "waiting_queue"))
    if method == "AA":
        return f"{method}_{ageing}_PSFA"
    elif method == "QBA":
        return f"{method}_{ageing}_PSFAQ"
    else:
        return method


def _save_metrics_json(metrics: dict, cfg: dict, elapsed: float) -> Path:
    """将仿真指标记录为 JSON 文件，保存到 scripts/output_exp/ 目录。

    Args:
        metrics: run() 返回的指标字典
        cfg: 配置字典
        elapsed: 仿真耗时（秒）

    Returns:
        输出文件路径
    """
    out_dir = Path(__file__).resolve().parent / "output_exp"
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = _method_filename_prefix(cfg)
    filename = f"{prefix}_results.json"
    out_path = out_dir / filename

    payload: dict = {
        "config_snapshot": {
            k: cfg.get(k) for k in [
                "schedule_method", "ageing_measure", "gamma", "quota_k",
                "admission_number", "lambda_arrival", "seed",
                "budget_tokens", "request_block_size",
                "KV_total_capacity", "KV_safety_threshold",
                "label_threshold_percentail",
                "steady_state",
            ]
        },
        "elapsed_seconds": round(elapsed, 3),
        "metrics": metrics,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[指标保存] {out_path}")
    return out_path


def _plot_results(state: EngineState, cfg: dict) -> None:
    """绘制仿真结果的 3x3 诊断图，保存到 scripts/output_exp/ 目录。

    图表内容：waiting_time / answer_len / prompt_len 分布直方图 +
    KV容量 / 吞吐率 / 老化阈值 时间序列
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finished = state.finished_queue
    batch_list = state.batch_list
    budget_tokens = state.budget_tokens

    if len(finished) == 0:
        print("[绘图跳过] 无已完成请求")
        return

    waiting_times = np.array([float(r.waiting_time) for r in finished], dtype=np.float64)
    service_times = np.array(
        [float(r.end_service_time) - float(r.start_service_time) for r in finished],
        dtype=np.float64,
    )
    answer_lens_done = np.array([int(r.answer_len) for r in finished], dtype=np.int64)
    prompt_lens_done = np.array([int(r.prompt_len) for r in finished], dtype=np.int64)

    batch_times = (
        np.array([float(b["batch_time"]) for b in batch_list], dtype=np.float64)
        if len(batch_list) > 0 else np.array([], dtype=np.float64)
    )
    kv_caps = (
        np.array([float(b["current_KV_capacity"]) for b in batch_list], dtype=np.float64)
        if len(batch_list) > 0 else np.array([], dtype=np.float64)
    )
    throughput_ratio = (
        np.array([float(b["batch_usage"]) / float(max(1, budget_tokens)) for b in batch_list], dtype=np.float64)
        if len(batch_list) > 0 else np.array([], dtype=np.float64)
    )

    method = str(cfg.get("schedule_method", "Unknown"))

    fig, axs = plt.subplots(3, 3, figsize=(18, 13))
    axs = axs.ravel()

    axs[0].hist(waiting_times, bins=60)
    axs[0].set_title(f"{method} waiting_time")
    axs[0].set_xlabel("seconds")
    axs[0].set_ylabel("count")

    axs[1].hist(answer_lens_done, bins=60)
    axs[1].set_title(f"{method} answer_len distribution")
    axs[1].set_xlabel("tokens")
    axs[1].set_ylabel("count")

    axs[2].hist(prompt_lens_done, bins=60)
    axs[2].set_title(f"{method} prompt_len distribution")
    axs[2].set_xlabel("tokens")
    axs[2].set_ylabel("count")

    if len(batch_times) > 0:
        axs[3].plot(batch_times, kv_caps, linewidth=1.0)
    axs[3].set_title(f"{method} KV capacity over time")
    axs[3].set_xlabel("time (s)")
    axs[3].set_ylabel("KV tokens")

    axs[4].hist(service_times, bins=60)
    axs[4].set_title(f"{method} service_time distribution")
    axs[4].set_xlabel("seconds")
    axs[4].set_ylabel("count")

    if len(batch_times) > 0:
        axs[5].plot(batch_times, throughput_ratio, linewidth=1.0)
    axs[5].set_title(f"{method} throughput ratio over time")
    axs[5].set_xlabel("time (s)")
    axs[5].set_ylabel("batch_usage / budget_tokens")
    axs[5].set_ylim(0.0, max(1.05, float(np.max(throughput_ratio)) * 1.05) if len(throughput_ratio) > 0 else 1.05)

    ts = np.asarray(state.aging_threshold_times, dtype=np.float64)
    th = np.asarray(state.aging_threshold_values, dtype=np.float64)
    if len(ts) > 0 and len(th) == len(ts):
        axs[6].plot(ts, th, linewidth=1.0)
    ageing_measure = cfg.get("ageing_measure", "waiting_queue")
    gamma = float(cfg.get("gamma", 0.0))
    axs[6].set_title(f"{method} aging threshold ({ageing_measure}, gamma={gamma})")
    axs[6].set_xlabel("time (s)")
    axs[6].set_ylabel("threshold")

    axs[7].axis("off")
    axs[8].axis("off")

    fig.tight_layout()

    # 输出目录
    out_dir = Path(__file__).resolve().parent / "output_exp"
    out_dir.mkdir(parents=True, exist_ok=True)

    ageing = str(cfg.get("ageing_measure", "waiting_queue"))
    if method == "AA":
        filename = f"{method}_{ageing}_distributions_PSFA.png"
    elif method == "QBA":
        filename = f"{method}_{ageing}_distributions_PSFAQ.png"
    else:
        filename = f"{method}_distributions.png"

    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[绘图保存] {out_path}")


def run_experiment(config_path: str | Path | None = None) -> dict:
    """加载配置并运行一次仿真实验。

    Args:
        config_path: YAML 配置文件路径，默认使用 configs/conference_default.yaml

    Returns:
        仿真结果指标字典
    """
    if config_path is None:
        repo_root = Path(__file__).resolve().parent.parent
        config_path = repo_root / "configs" / "conference_default.yaml"

    print(f"[加载配置] {config_path}")
    cfg = load_config(config_path)

    method = cfg.get("schedule_method", "FCFS")
    n_requests = cfg.get("admission_number", "?")
    print(f"[调度策略] {method}")
    print(f"[请求数量] {n_requests}")
    ss_cfg = cfg.get("steady_state", None) or {}
    if ss_cfg.get("enabled", False):
        trim = ss_cfg.get("trim_ratio", 0.1)
        print(f"[稳态窗口] trim_ratio={trim} (前后各 {trim*100:.0f}% 丢弃)")
    print(f"[运行中...]")

    t0 = time.perf_counter()
    metrics, state = run(cfg, return_state=True)
    elapsed = time.perf_counter() - t0

    print(f"\n{'='*50}")
    print(f"仿真完成 (耗时 {elapsed:.1f}s)")
    print(f"{'='*50}")
    print(f"调度策略:        {metrics.get('schedule_method', '-')}")
    print(f"老化度量方式:    {metrics.get('ageing_measure', '-')}")
    print(f"OOM概率:         {metrics.get('OOM概率', 0):.6f}")
    print(f"{'─'*35}")
    print(f"Waiting Time 平均: {metrics.get('waiting_time平均', 0):.4f}")
    print(f"Waiting Time p95:  {metrics.get('waiting_time p95', 0):.4f}")
    print(f"Waiting Time p99:  {metrics.get('waiting_time p99', 0):.4f}")
    print(f"max Waiting Time:  {metrics.get('max waiting_time', 0):.4f}")
    print(f"{'─'*35}")
    print(f"E2E Latency 平均:  {metrics.get('E2E Latency平均', 0):.4f}")
    print(f"E2E Latency p95:   {metrics.get('E2E Latency p95', 0):.4f}")
    print(f"E2E Latency p99:   {metrics.get('E2E Latency p99', 0):.4f}")
    print(f"max E2E Latency:   {metrics.get('max E2E Latency', 0):.4f}")
    print(f"{'─'*35}")
    print(f"TTFT 平均:         {metrics.get('TTFT平均', 0):.4f}")
    print(f"TTFT p95:          {metrics.get('TTFT p95', 0):.4f}")
    print(f"TTFT p99:          {metrics.get('TTFT p99', 0):.4f}")
    print(f"max TTFT:          {metrics.get('max TTFT', 0):.4f}")
    print(f"{'─'*35}")
    print(f"平均吞吐量:        {metrics.get('平均吞吐量', 0):.4f}")
    print(f"平均batch使用率:   {metrics.get('平均batch使用率', 0):.4f}")
    if metrics.get('PSFAQ_promoted_total', 0) > 0:
        print(f"{'─'*35}")
        print(f"PSFAQ promoted:    {metrics.get('PSFAQ_promoted_total', 0)}")
        print(f"PSFAQ short_admit: {metrics.get('PSFAQ_short_admit_since_last_promoted', 0)}")
    print(f"{'='*50}\n")

    _plot_results(state, cfg)
    _save_metrics_json(metrics, cfg, elapsed)
    return metrics


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_experiment(config_path)
