from __future__ import annotations

import numpy as np

from llm_in_queuing.types import Request, EngineState


def init_simulator_state(cfg: dict) -> EngineState:
    """初始化模拟器状态 (refine.py L144-L194 迁移)"""
    KV_total_capacity = int(cfg["KV_total_capacity"])
    KV_safety_threshold = float(KV_total_capacity) * float(cfg["KV_safety_threshold"])
    budget_tokens = int(cfg["budget_tokens"])
    request_block_size = int(cfg["request_block_size"])
    gamma = float(cfg.get("gamma", 0.95))
    ageing_measure = str(cfg.get("ageing_measure", "waiting_queue"))
    quota_k = int(cfg.get("quota_k", 5))
    mech_diag_enabled = bool(cfg.get("mech_diag_enabled", False))

    state = EngineState(
        KV_total_capacity=KV_total_capacity,
        KV_safety_threshold=KV_safety_threshold,
        budget_tokens=budget_tokens,
        request_block_size=request_block_size,
        gamma=gamma,
        ageing_measure=ageing_measure,
        quota_k=max(1, quota_k),
        schedule_method=str(cfg.get("schedule_method", "FCFS")),
    )
    state.mech.enabled = mech_diag_enabled
    return state


def is_prefill_continuation(req: Request) -> bool:
    """
    判断是否为prefill请求的继续
    """
    return bool(req.prefill_chunk) and int(req.prefill_res_token) > 0


def add_to_wait_queue(ev: Request, state: EngineState, front: bool = False) -> None:
    """
    将事件ev添加到等待队列state中
    front: 是否将ev添加到队列的最前面（默认False）
    """
    if front:
        state.wait_queue.insert(0, ev)
    else:
        state.wait_queue.append(ev)
    a = float(ev.arrival_time)
    state.wait_queue_n += 1
    state.wait_queue_sum_arrival += a
    state.wait_queue_sum_arrival_sq += a * a


def remove_from_wait_queue(ev: Request, state: EngineState) -> None:
    """
    从等待队列state中移除事件ev
    """
    state.wait_queue.remove(ev)
    a = float(ev.arrival_time)
    state.wait_queue_n -= 1
    state.wait_queue_sum_arrival -= a
    state.wait_queue_sum_arrival_sq -= a * a


def mark_request_started(ev: Request, state: EngineState) -> None:
    """
    标记请求开始
    """
    a = float(ev.arrival_time)
    w = float(ev.waiting_time)

    ev.started = True
    state.arrived_not_started_n -= 1
    state.arrived_not_started_sum_arrival -= a
    state.arrived_not_started_sum_arrival_sq -= a * a

    state.arrived_started_n += 1
    state.arrived_started_sum_wait += w
    state.arrived_started_sum_wait_sq += w * w
    state.arrived_started_waits.append(w)


def compute_and_record_aging_threshold(state: EngineState) -> float:
    """
    计算老化阈值并记录
    """
    ageing_measure = str(state.ageing_measure)
    q = float(state.gamma)
    if not np.isfinite(q):
        q = 0.95
    q = float(np.clip(q, 0.0, 1.0))

    if ageing_measure == "waiting_queue":
        n = int(state.wait_queue_n)
        if n > 0:
            t_now = float(state.current_time)
            waits = np.asarray([t_now - float(e.arrival_time) for e in state.wait_queue], dtype=np.float64)
            threshold = float(np.quantile(waits, q))
            mean_wt = float(waits.mean())
            std_wt = float(waits.std(ddof=0))
        else:
            threshold = 0.0
            mean_wt = 0.0
            std_wt = 0.0
    elif ageing_measure == "history":
        waits = np.asarray(state.arrived_started_waits, dtype=np.float64)
        if waits.size > 0:
            threshold = float(np.quantile(waits, q))
            mean_wt = float(waits.mean())
            std_wt = float(waits.std(ddof=0))
        else:
            threshold = 0.0
            mean_wt = 0.0
            std_wt = 0.0
    else:
        raise ValueError(f"Unknown ageing_measure: {ageing_measure}")

    state.wait_queue_wait_mean = float(mean_wt)
    state.wait_queue_wait_var = float(std_wt) * float(std_wt)
    state.wait_queue_wait_std = float(std_wt)
    state.aging_threshold_times.append(float(state.current_time))
    state.aging_threshold_values.append(float(threshold))
    return float(threshold)
