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


def mechanism_metrics(state: EngineState) -> dict:
    """聚合 Tier2 旁路机制统计（``state.mech``）。

    仅当 ``cfg["mech_diag_enabled"]`` 为真（``state.mech.enabled``）时返回统计；
    否则返回空 dict。本函数只读聚合，不修改 state，不参与任何调度决策。

    指标口径（分母明确区分）:
        η_eligibility = 有候选的债务批 / **KV 门开启期间调度的**债务批
            = (candidate_blocked + success) / (no_eligible + candidate_blocked + success)
        η_feasibility = 成功提升的债务批 / 有候选的债务批
            = success / (candidate_blocked + success)
        上述三个分桶仅在 KV 安全门**开启**且本批真正调度出 batch 时计数，互斥；
        ``debt_batch_kv_gate_closed`` 属于门关闭分支，**不计入**上述分母。因此 η 是
        「门开启条件下」的候选率与成功率；门关闭占比需与三分桶合并后才能得到完整债务批分解。
    """
    mech = getattr(state, "mech", None)
    if mech is None or not mech.enabled:
        return {}

    durs = (np.asarray(list(mech.debt_episode_durations), dtype=np.float64)
            if mech.debt_episode_durations else np.array([], dtype=np.float64))
    dur_mean = float(durs.mean()) if durs.size else 0.0
    dur_p99 = float(np.percentile(durs, 99)) if durs.size else 0.0
    dur_max = float(durs.max()) if durs.size else 0.0

    unresolved = int(mech.debt_active)
    unresolved_dur = (float(state.current_time) - float(mech.debt_start_time)) if mech.debt_active else 0.0

    total_debt_batches = (int(mech.debt_batch_no_eligible)
                          + int(mech.debt_batch_candidate_blocked)
                          + int(mech.debt_batch_success))
    eligible_batches = int(mech.debt_batch_candidate_blocked) + int(mech.debt_batch_success)
    eta_eligibility = (float(eligible_batches) / float(total_debt_batches)
                       if total_debt_batches > 0 else 0.0)
    eta_feasibility = (float(mech.debt_batch_success) / float(eligible_batches)
                       if eligible_batches > 0 else 0.0)

    return {
        "mech_debt_episode_count": float(mech.debt_episode_count),
        "mech_debt_episode_mean_duration": dur_mean,
        "mech_debt_episode_p99_duration": dur_p99,
        "mech_debt_episode_max_duration": dur_max,
        "mech_debt_unresolved_at_end": float(unresolved),
        "mech_debt_unresolved_duration": float(unresolved_dur),
        "mech_promotion_success_count": float(mech.promotion_success_count),
        "mech_debt_batch_no_eligible": float(mech.debt_batch_no_eligible),
        "mech_debt_batch_candidate_blocked": float(mech.debt_batch_candidate_blocked),
        "mech_debt_batch_success": float(mech.debt_batch_success),
        "mech_debt_batch_kv_gate_closed": float(mech.debt_batch_kv_gate_closed),
        "mech_debt_batch_time_no_eligible": float(mech.debt_batch_time_no_eligible),
        "mech_debt_batch_time_candidate_blocked": float(mech.debt_batch_time_candidate_blocked),
        "mech_debt_batch_time_success": float(mech.debt_batch_time_success),
        "mech_eta_eligibility": eta_eligibility,
        "mech_eta_feasibility": eta_feasibility,
        "mech_long_admit_normal": float(mech.long_admit_normal),
        "mech_long_admit_promoted": float(mech.long_admit_promoted),
        "mech_long_admit_eligible_at_entry": float(mech.long_admit_eligible_at_entry),
        "mech_budget_break_count": float(mech.budget_break_count),
        "mech_kv_break_count": float(mech.kv_break_count),
        "mech_concurrency_break_count": float(mech.concurrency_break_count),
        "mech_continuation_break_count": float(mech.continuation_break_count),
        "mech_prefill_token_sum": float(mech.prefill_token_sum),
        "mech_decode_token_sum": float(mech.decode_token_sum),
        "mech_continuation_count": float(mech.continuation_count),
        "mech_batch_count": float(mech.batch_count),
    }
