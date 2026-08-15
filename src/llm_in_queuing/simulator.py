from __future__ import annotations

import heapq
from typing import Callable

from llm_in_queuing.types import Request, BatchEvent, EngineState
from llm_in_queuing.workload import generate_heapq
from llm_in_queuing.state import init_simulator_state
from llm_in_queuing.engine import handle_batch_event, handle_arrival_event
from llm_in_queuing.metrics import metrics_calculate


def _get_scheduler(method: str) -> Callable[[list, EngineState], bool]:
    if method == "FCFS":
        from llm_in_queuing.schedulers.fcfs import schedule_fcfs
        return schedule_fcfs
    elif method == "PSF":
        from llm_in_queuing.schedulers.psf import schedule_psf
        return schedule_psf
    elif method == "AA":
        from llm_in_queuing.schedulers.aging import schedule_aging
        return schedule_aging
    elif method == "QBA":
        from llm_in_queuing.schedulers.qba import schedule_qba
        return schedule_qba
    elif method == "RR":
        from llm_in_queuing.schedulers.rr import schedule_rr
        return schedule_rr
    else:
        raise ValueError(f"Unknown schedule method: {method}")


def run(cfg: dict, return_state: bool = False):
    """运行仿真。

    Args:
        cfg: 配置字典
        return_state: 若 True，返回 (metrics, state) 元组，方便画图等后处理

    Returns:
        metrics dict，或 (metrics, EngineState) 元组
    """
    heap = generate_heapq(cfg)
    state = init_simulator_state(cfg)

    # 稳态窗口配置
    ss_cfg = cfg.get("steady_state", None) or {}
    steady_enabled = bool(ss_cfg.get("enabled", False))
    t_warm_end: float = 0.0
    t_measure_end: float = 0.0

    if steady_enabled:
        trim_ratio = float(ss_cfg.get("trim_ratio", 0.1))
        all_arrivals = sorted(ev.arrival_time for _, _, ev in heap if isinstance(ev, Request))
        n = len(all_arrivals)
        t_warm_end = all_arrivals[int(n * trim_ratio)]
        t_measure_end = all_arrivals[int(n * (1.0 - trim_ratio))]
        state.t_warm_end = t_warm_end
        state.t_measure_end = t_measure_end

    schedule = _get_scheduler(str(cfg.get("schedule_method", "FCFS")))

    while heap or (not state.engine_isbusy and (state.wait_queue or state.current_batch_decode)):
        ev = None
        if heap:
            key, _, ev = heapq.heappop(heap)
            state.current_time = key

        if isinstance(ev, Request):
            if steady_enabled and ev.arrival_time > t_measure_end:
                pass  # 跳过准入：超过测量窗口的请求不进入系统
            else:
                handle_arrival_event(ev, state)
        elif isinstance(ev, BatchEvent):
            handle_batch_event(ev, state)

        if schedule(heap, state):
            if not heap:
                if not state.wait_queue and not state.current_batch_decode:
                    break
                raise RuntimeError(
                    f"deadlock: no schedulable work (time={state.current_time}, "
                    f"kv={state.current_KV_capacity}, wait_queue={len(state.wait_queue)}, "
                    f"active_decode={len(state.current_batch_decode)})"
                )
            continue

        if not steady_enabled:
            if len(state.finished_queue) == int(cfg["admission_number"]):
                break

    if return_state:
        return metrics_calculate(state), state
    return metrics_calculate(state)

