from __future__ import annotations

import heapq

from llm_in_queuing.types import Request, BatchEvent, EngineState
from llm_in_queuing.state import (
    compute_and_record_aging_threshold,
    is_prefill_continuation,
    mark_request_started,
    remove_from_wait_queue,
)
from llm_in_queuing.engine import batch_time_calculate


def _partition_wait_queue(state: EngineState, threshold: float) -> tuple[list[Request], list[Request], list[Request], list[Request]]:
    continuation = [e for e in state.wait_queue if is_prefill_continuation(e)]
    rest = [e for e in state.wait_queue if not is_prefill_continuation(e)]
    aged_long: list[Request] = []
    short: list[Request] = []
    long: list[Request] = []
    t_now = float(state.current_time)
    threshold = float(threshold)
    for e in rest:
        wt_e = t_now - float(e.arrival_time)
        if e.predict_label == "long":
            if wt_e > threshold:
                aged_long.append(e)
            else:
                long.append(e)
        else:
            short.append(e)
    aged_long.sort(key=lambda x: x.arrival_time)
    short.sort(key=lambda x: x.arrival_time)
    long.sort(key=lambda x: x.arrival_time)
    return continuation, short, aged_long, long


def _build_order(continuation: list[Request], short: list[Request], aged_long: list[Request], long: list[Request], state: EngineState) -> tuple[list[Request], int | None]:
    quota_k = int(state.quota_k)
    if int(state.psfaq_short_admit_since_last_promoted) >= quota_k:
        state.psfaq_promotion_debt = True
    promotion_eligible = bool(state.psfaq_promotion_debt)

    if promotion_eligible and len(aged_long) > 0:
        promoted = aged_long[0]
        promoted_id = int(promoted.id)
        remaining_longs = aged_long[1:] + long
        remaining_longs.sort(key=lambda x: x.arrival_time)
        ordered = continuation + [promoted] + short + remaining_longs
        return ordered, promoted_id

    merged_longs = aged_long + long
    merged_longs.sort(key=lambda x: x.arrival_time)
    ordered = continuation + short + merged_longs
    return ordered, None


def schedule_qba(heap: list, state: EngineState) -> bool:
    if not state.engine_isbusy:
        state.itr += 1
        processing_in_current_itr: list[Request] = []
        state.current_batch_prefill = []
        state.current_batch = 0
        state.psfaq_promoted_this_batch = 0

        max_active_decode = int(state.budget_tokens) // int(state.request_block_size)
        state.current_concurrent = len(state.current_batch_decode)
        decode_tokens_this_batch = 0
        for r in state.current_batch_decode:
            remaining = int(r.answer_len) - int(r.processed_len)
            if remaining > 0:
                decode_tokens_this_batch += min(int(state.request_block_size), remaining)
        decode_tokens_this_batch = min(int(state.budget_tokens), int(decode_tokens_this_batch))
        prefill_budget = int(state.budget_tokens) - int(decode_tokens_this_batch)

        threshold = compute_and_record_aging_threshold(state)
        prefill_token_map: dict[int, int] = {}
        promoted_candidate_id: int | None = None

        if state.current_KV_capacity < state.KV_safety_threshold:
            continuation, short, aged_long, long = _partition_wait_queue(state, threshold)
            ordered, promoted_candidate_id = _build_order(continuation, short, aged_long, long, state)
            state.wait_queue = ordered

            planned_kv_increase = 0
            for e in state.wait_queue:
                if state.current_concurrent >= max_active_decode:
                    break

                remaining_prefill_budget = int(prefill_budget) - int(state.current_batch)
                if remaining_prefill_budget <= 0:
                    break

                # 为可复现性，沿用 refine.py 的旧语义：短期计数判断在处理之后内联求值。
                # 旧语义下，发生 prefill 切分的新到 short 请求（prefill_chunk=True 且 res_token>0）
                # 会被视为 continuation，从而不计入 quota 计数器。
                # 新语义（切分的 short 首次准入也计入 quota）暂缓，以后再讨论：
                #   was_continuation = is_prefill_continuation(e)

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

                admitted_is_promoted = (promoted_candidate_id is not None) and (int(e.id) == int(promoted_candidate_id))

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

                if admitted_is_promoted:
                    state.psfaq_short_admit_since_last_promoted = 0
                    state.psfaq_promotion_debt = False
                    state.psfaq_promoted_total = int(state.psfaq_promoted_total) + 1
                    state.psfaq_promoted_this_batch = 1
                else:
                    if (e.predict_label == "short") and (not is_prefill_continuation(e)):
                        if not bool(state.psfaq_promotion_debt):
                            state.psfaq_short_admit_since_last_promoted = int(state.psfaq_short_admit_since_last_promoted) + 1
                            state.psfaq_short_admit_since_last_promoted = min(
                                int(state.psfaq_short_admit_since_last_promoted),
                                int(state.quota_k),
                            )

                if int(e.prefill_res_token) > 0:
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
