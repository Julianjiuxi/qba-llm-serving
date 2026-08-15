from __future__ import annotations

import heapq

from llm_in_queuing.types import BatchEvent, EngineState
from llm_in_queuing.state import (
    add_to_wait_queue,
    compute_and_record_aging_threshold,
    mark_request_started,
    remove_from_wait_queue,
)
from llm_in_queuing.engine import batch_time_calculate


def schedule_fcfs(heap: list, state: EngineState) -> bool:
    """
    FCFS调度方法
    """
    if not state.engine_isbusy:
        state.itr += 1
        processing_in_current_itr = []
        state.current_batch_prefill = []
        state.current_batch = 0

        max_active_decode = int(state.budget_tokens) // int(state.request_block_size)
        state.current_concurrent = len(state.current_batch_decode)
        decode_tokens_this_batch = 0
        for r in state.current_batch_decode:
            remaining = int(r.answer_len) - int(r.processed_len)
            if remaining > 0:
                decode_tokens_this_batch += min(int(state.request_block_size), remaining)
        decode_tokens_this_batch = min(int(state.budget_tokens), int(decode_tokens_this_batch))
        prefill_budget = int(state.budget_tokens) - int(decode_tokens_this_batch)

        compute_and_record_aging_threshold(state)

        prefill_token_map: dict[int, int] = {}

        if state.current_KV_capacity < state.KV_safety_threshold:
            planned_kv_increase = 0
            for e in state.wait_queue:
                if state.current_concurrent >= max_active_decode:
                    break

                remaining_prefill_budget = int(prefill_budget) - int(state.current_batch)
                if remaining_prefill_budget <= 0:
                    break

                rem = int(e.prefill_res_token)
                take = min(rem, int(remaining_prefill_budget))
                if take <= 0:
                    break

                extra_decode0 = 0
                if take == rem:
                    extra_decode0 = min(int(state.request_block_size), int(e.answer_len) - int(e.processed_len))

                projected_kv_peak = (
                    int(state.current_KV_capacity)
                    + int(decode_tokens_this_batch)
                    + int(planned_kv_increase)
                    + int(take)
                    + int(extra_decode0)
                )
                if projected_kv_peak > int(state.KV_total_capacity) or projected_kv_peak > float(state.KV_safety_threshold):
                    break

                if rem <= remaining_prefill_budget:
                    state.current_batch_prefill.append(e)
                    state.current_batch += int(rem)
                    e.prefill_res_token = 0
                    prefill_token_map[int(e.id)] = int(rem)
                    planned_kv_increase += int(rem) + int(extra_decode0)
                    processing_in_current_itr.append(e)
                    if e.start_service_time == 0.0:
                        e.start_service_time = state.current_time
                        e.waiting_time = e.start_service_time - e.arrival_time
                        mark_request_started(e, state)
                else:
                    e.prefill_chunk = True
                    state.current_batch_prefill.append(e)
                    e.prefill_res_token = int(e.prefill_res_token) - int(take)
                    state.current_batch += int(take)
                    prefill_token_map[int(e.id)] = int(take)
                    planned_kv_increase += int(take)
                    processing_in_current_itr.append(e)
                    if e.start_service_time == 0.0:
                        e.start_service_time = state.current_time
                        e.waiting_time = e.start_service_time - e.arrival_time
                        mark_request_started(e, state)
                    break

            for e in processing_in_current_itr:
                remove_from_wait_queue(e, state)

        batch_usage = int(state.current_batch) + int(decode_tokens_this_batch)
        state.current_batch_time = state.current_time + batch_time_calculate(batch_usage, state.current_concurrent)
        current_OOM = False
        if state.current_KV_capacity > state.KV_total_capacity:
            current_OOM = True
        batch_event = BatchEvent(
            id=state.itr,
            batch_time=state.current_batch_time,
            batch_usage=batch_usage,
            current_concurrent=state.current_concurrent,
            current_KV_capacity=state.current_KV_capacity,
            current_OOM=current_OOM,
            prefill_requests=list(state.current_batch_prefill),
            prefill_token_map=dict(prefill_token_map),
            decode_requests=state.current_batch_decode,
        )
        if batch_event.batch_usage > 0:
            heapq.heappush(heap, (batch_event.batch_time, batch_event.id, batch_event))
            state.engine_isbusy = True
            return False
        else:
            return True

    return False
