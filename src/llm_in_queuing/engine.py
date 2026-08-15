from __future__ import annotations

import numpy as np
from llm_in_queuing.types import Request, BatchEvent, EngineState
from llm_in_queuing.state import add_to_wait_queue

# batch time model constants (mirrors Config defaults)
BT_BASE_C: float = 0.003
BT_SLOPE_A: float = 0.00002
BT_KNEE_B0: int = 64
BT_CONC_GAMMA: float = 0.0
BT_JITTER_STD: float = 0.0
BT_MIN: float = 1e-4


def batch_time_calculate(batch_usage: int, current_concurrent: int) -> float:
    """
    计算本轮的batch时间
    batch_usage: 本轮 token load（核心自变量）
    current_concurrent: 本轮并发请求数（可选二阶修正）
    """
    b = max(0.0, float(batch_usage))
    conc = max(0.0, float(current_concurrent))

    t = BT_BASE_C + BT_SLOPE_A * max(0.0, b - float(BT_KNEE_B0))

    t *= 1.0 + BT_CONC_GAMMA * conc

    jitter_std = BT_JITTER_STD
    if jitter_std > 0.0:
        t *= max(0.7, 1.0 + np.random.normal(0.0, jitter_std))
    return max(BT_MIN, t)


def handle_batch_event(ev: BatchEvent, state: EngineState) -> None:
    prefill_ids = [r.id for r in ev.prefill_requests]
    kv_peak = float(state.current_KV_capacity)

    for e in ev.decode_requests:
        delta = min(int(state.request_block_size), int(e.answer_len) - int(e.processed_len))
        if delta > 0:
            e.processed_len += delta
            state.current_KV_capacity += delta
            kv_peak = max(kv_peak, float(state.current_KV_capacity))

    state.current_concurrent = len(ev.decode_requests)

    for e in ev.prefill_requests:
        prompt_delta = int(ev.prefill_token_map.get(e.id, 0))
        if prompt_delta > 0:
            state.current_KV_capacity += prompt_delta
            kv_peak = max(kv_peak, float(state.current_KV_capacity))

        if e.prefill_res_token == 0 and e.processed_len == 0:
            e.TTFT = state.current_time - e.arrival_time
            delta = min(int(state.request_block_size), int(e.answer_len) - int(e.processed_len))
            if delta > 0:
                e.processed_len += delta
                state.current_KV_capacity += delta
                kv_peak = max(kv_peak, float(state.current_KV_capacity))
            state.current_batch_decode.append(e)
            state.current_concurrent += 1
        elif e.prefill_res_token > 0:
            add_to_wait_queue(e, state, front=True)

    ev.prefill_requests = []
    state.current_batch_prefill = []

    decode_finish_status = []
    for e in ev.decode_requests:
        if e.processed_len >= e.answer_len:
            state.finished_queue.append(e)
            decode_finish_status.append(e)
            e.end_service_time = state.current_time
            state.current_concurrent -= 1
            state.current_KV_capacity -= e.answer_len
            state.current_KV_capacity -= e.prompt_len
            e.in_processing = False
            e.end_service_time = state.current_time

    for e in decode_finish_status:
        ev.decode_requests.remove(e)

    batch_oom = kv_peak >= float(state.KV_total_capacity)

    snap = {
        "id": ev.id,
        "sim_time": state.current_time,
        "batch_time": ev.batch_time,
        "batch_usage": ev.batch_usage,
        "current_concurrent": ev.current_concurrent,
        "current_KV_capacity": ev.current_KV_capacity,
        "current_KV_capacity_end": state.current_KV_capacity,
        "kv_peak": kv_peak,
        "current_OOM": batch_oom,
        "prefill_ids": prefill_ids,
        "decode_ids": [r.id for r in ev.decode_requests],
        "psfaq_promoted_this_batch": state.psfaq_promoted_this_batch,
        "psfaq_promotion_debt_end": state.psfaq_promotion_debt,
        "psfaq_short_counter_end": state.psfaq_short_admit_since_last_promoted,
    }
    state.batch_list.append(snap)

    state.engine_isbusy = False
    state.current_batch = 0


def handle_arrival_event(req: Request, state: EngineState) -> None:
    add_to_wait_queue(req, state)
    a = req.arrival_time
    state.arrived_not_started_n += 1
    state.arrived_not_started_sum_arrival += a
    state.arrived_not_started_sum_arrival_sq += a * a
