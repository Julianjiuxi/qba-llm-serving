from __future__ import annotations

import numpy as np

from llm_in_queuing.types import EngineState


def metrics_calculate(state: EngineState) -> dict:
    """计算模拟结果的指标，返回与 refine.py metrics_calculate 完全一致的 dict"""
    finished_all = state.finished_queue
    batch_list_all = state.batch_list

    # 稳态窗口过滤
    if state.t_measure_end > 0.0:
        finished = [r for r in finished_all
                    if state.t_warm_end <= r.arrival_time <= state.t_measure_end]
        window_ids = {r.id for r in finished}
        batch_list = [b for b in batch_list_all
                      if any(rid in window_ids
                             for rid in b.get("prefill_ids", []) + b.get("decode_ids", []))]
    else:
        finished = finished_all
        batch_list = batch_list_all

    budget_tokens = state.budget_tokens

    oom_count = sum(1 for b in batch_list if b.get('current_OOM', False))
    oom_prob = (float(oom_count) / float(len(batch_list))) if len(batch_list) > 0 else 0.0

    # waiting time
    if finished:
        waiting_times = np.array([float(r.waiting_time) for r in finished], dtype=np.float64)
        avg_wait = float(np.mean(waiting_times))
        p95_wait = float(np.percentile(waiting_times, 95))
        p99_wait = float(np.percentile(waiting_times, 99))
        max_wait = float(np.max(waiting_times))
    else:
        avg_wait = p95_wait = p99_wait = max_wait = 0.0

    # E2E latency
    if finished:
        e2e = np.array([float(r.end_service_time) - float(r.arrival_time) for r in finished], dtype=np.float64)
        e2e_avg = float(np.mean(e2e))
        e2e_p95 = float(np.percentile(e2e, 95))
        e2e_p99 = float(np.percentile(e2e, 99))
        max_e2e = float(np.max(e2e))
    else:
        e2e_avg = e2e_p95 = e2e_p99 = max_e2e = 0.0

    # TTFT
    if finished:
        ttft = np.array([float(r.TTFT) for r in finished], dtype=np.float64)
        ttft_avg = float(np.mean(ttft))
        ttft_p95 = float(np.percentile(ttft, 95))
        ttft_p99 = float(np.percentile(ttft, 99))
        max_ttft = float(np.max(ttft))
    else:
        ttft_avg = ttft_p95 = ttft_p99 = max_ttft = 0.0

    # throughput
    if finished:
        total_tokens = sum(r.prompt_len + r.answer_len for r in finished)
        total_time = max(r.end_service_time for r in finished) - min(r.arrival_time for r in finished)
        avg_throughput = total_tokens / total_time if total_time > 0 else 0.0
    else:
        avg_throughput = 0.0

    throughput_ratio = (
        np.array([float(b["batch_usage"]) / float(max(1, budget_tokens)) for b in batch_list], dtype=np.float64)
        if len(batch_list) > 0
        else np.array([], dtype=np.float64)
    )
    average_batch_occupy_rate = float(np.mean(throughput_ratio))

    # ── 按 true_label 分类的 waiting time ──
    short_finished = [r for r in finished if getattr(r, 'true_label', '') == "short"]
    long_finished = [r for r in finished if getattr(r, 'true_label', '') == "long"]

    def _per_class_stats(requests):
        if not requests:
            return 0.0, 0.0
        wt = np.array([float(r.waiting_time) for r in requests], dtype=np.float64)
        return float(np.mean(wt)), float(np.percentile(wt, 99))

    ts_mean, ts_p99 = _per_class_stats(short_finished)
    tl_mean, tl_p99 = _per_class_stats(long_finished)

    # ── 实际 prediction error rate（基于全体 finished，不受稳态窗口影响）──
    all_finished = finished_all
    n_true_short = sum(1 for r in all_finished if getattr(r, 'true_label', '') == "short")
    n_true_long = sum(1 for r in all_finished if getattr(r, 'true_label', '') == "long")
    n_sl = sum(1 for r in all_finished
               if getattr(r, 'true_label', '') == "short" and getattr(r, 'predict_label', '') == "long")
    n_ls = sum(1 for r in all_finished
               if getattr(r, 'true_label', '') == "long" and getattr(r, 'predict_label', '') == "short")
    actual_epsilon_SL = float(n_sl) / float(n_true_short) if n_true_short > 0 else 0.0
    actual_epsilon_LS = float(n_ls) / float(n_true_long) if n_true_long > 0 else 0.0

    return {
        'schedule_method': state.schedule_method,
        'ageing_measure': state.ageing_measure,
        'OOM概率': oom_prob,
        'waiting_time平均': avg_wait,
        'waiting_time p95': p95_wait,
        'waiting_time p99': p99_wait,
        'max waiting_time': max_wait,
        'E2E Latency平均': e2e_avg,
        'E2E Latency p95': e2e_p95,
        'E2E Latency p99': e2e_p99,
        'max E2E Latency': max_e2e,
        'TTFT平均': ttft_avg,
        'TTFT p95': ttft_p95,
        'TTFT p99': ttft_p99,
        'max TTFT': max_ttft,
        '平均吞吐量': avg_throughput,
        '平均batch使用率': average_batch_occupy_rate,
        'PSFAQ_promoted_total': float(state.psfaq_promoted_total),
        'PSFAQ_short_admit_since_last_promoted': float(state.psfaq_short_admit_since_last_promoted),
        'PSFAQ_promotion_debt': float(1.0 if state.psfaq_promotion_debt else 0.0),
        # per-class waiting
        'waiting_time_true_short_mean': ts_mean,
        'waiting_time_true_short_p99': ts_p99,
        'waiting_time_true_long_mean': tl_mean,
        'waiting_time_true_long_p99': tl_p99,
        # actual error rates
        'n_true_short': n_true_short,
        'n_true_long': n_true_long,
        'n_short_to_long': n_sl,
        'n_long_to_short': n_ls,
        'actual_epsilon_SL': actual_epsilon_SL,
        'actual_epsilon_LS': actual_epsilon_LS,
    }
