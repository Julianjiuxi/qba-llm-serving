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
        mech = state.mech
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

        # ── Tier2 旁路计量：局部累加，末尾一次性并入 state.mech（不参与调度决策）──
        debt_owed_this_batch = False
        promoted_admitted = False
        aged_long_count = 0
        budget_break = False
        kv_break = False
        concurrency_break = False
        continuation_break = False
        continuation_steps = 0
        long_normal = 0
        long_promoted = 0
        long_eligible = 0

        if state.current_KV_capacity < state.KV_safety_threshold:
            continuation, short, aged_long, long = _partition_wait_queue(state, threshold)
            aged_long_count = len(aged_long)
            ordered, promoted_candidate_id = _build_order(continuation, short, aged_long, long, state)
            state.wait_queue = ordered

            if mech.enabled:
                # debt 是否在本批调度时存在（含 _build_order 内新置 True 的情形）
                debt_owed_this_batch = bool(state.psfaq_promotion_debt)
                if debt_owed_this_batch and not mech.debt_active:
                    mech.debt_active = True
                    mech.debt_start_time = float(state.current_time)
                    mech.debt_episode_count += 1

            planned_kv_increase = 0
            for e in state.wait_queue:
                if state.current_concurrent >= max_active_decode:
                    concurrency_break = True
                    break

                remaining_prefill_budget = int(prefill_budget) - int(state.current_batch)
                if remaining_prefill_budget <= 0:
                    budget_break = True
                    break

                # Quota accounting: the short-admission counter advances only
                # when a predicted-short request completes prefill and becomes
                # fully admitted in the current batch. The is_prefill_continuation
                # test below runs after the prefill fields are updated, so a
                # request whose remaining prefill fits entirely here (res_token=0)
                # is not a continuation and counts, whereas a chunked request with
                # residual prefill tokens is a continuation and defers its quota
                # increment to the batch that finishes its prefill.

                rem = int(e.prefill_res_token)
                take = min(rem, int(remaining_prefill_budget))
                if take <= 0:
                    budget_break = True
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
                    kv_break = True
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
                    continuation_steps += 1
                    if e.start_service_time == 0.0:
                        e.start_service_time = state.current_time
                        e.waiting_time = e.start_service_time - e.arrival_time
                        mark_request_started(e, state)

                # long 去向（旁路）：predicted-long 经 promotion 还是普通通道准入
                if mech.enabled and e.predict_label == "long":
                    if admitted_is_promoted:
                        long_promoted += 1
                    else:
                        long_normal += 1
                        if (float(state.current_time) - float(e.arrival_time)) > float(threshold):
                            long_eligible += 1

                if admitted_is_promoted:
                    e.promoted = True
                    state.psfaq_short_admit_since_last_promoted = 0
                    state.psfaq_promotion_debt = False
                    state.psfaq_promoted_total = int(state.psfaq_promoted_total) + 1
                    state.psfaq_promoted_this_batch = 1
                    promoted_admitted = True
                else:
                    if (e.predict_label == "short") and (not is_prefill_continuation(e)):
                        if not bool(state.psfaq_promotion_debt):
                            state.psfaq_short_admit_since_last_promoted = int(state.psfaq_short_admit_since_last_promoted) + 1
                            state.psfaq_short_admit_since_last_promoted = min(
                                int(state.psfaq_short_admit_since_last_promoted),
                                int(state.quota_k),
                            )

                if int(e.prefill_res_token) > 0:
                    continuation_break = True
                    break

            for e in processing_in_current_itr:
                remove_from_wait_queue(e, state)
        else:
            # KV 安全门关闭：本轮未调度等待队列（旁路记录债务是否悬空）
            if mech.enabled and bool(state.psfaq_promotion_debt):
                mech.debt_batch_kv_gate_closed += 1
                if not mech.debt_active:
                    mech.debt_active = True
                    mech.debt_start_time = float(state.current_time)
                    mech.debt_episode_count += 1

        batch_usage = int(state.current_batch) + int(decode_tokens_this_batch)
        state.current_batch_time = state.current_time + batch_time_calculate(batch_usage, state.current_concurrent)
        current_OOM = False
        if state.current_KV_capacity > state.KV_total_capacity:
            current_OOM = True

        # ── Tier2：把本批旁路计量并入 state.mech（仅本批真正调度出 batch 时，
        #    以与 batch_list 快照一一对应）──
        if mech.enabled and batch_usage > 0:
            batch_dur = float(state.current_batch_time) - float(state.current_time)
            if promoted_admitted and mech.debt_active:
                mech.debt_episode_durations.append(float(state.current_time) - float(mech.debt_start_time))
                mech.debt_active = False
                mech.promotion_success_count += 1
            if debt_owed_this_batch:
                if promoted_admitted:
                    mech.debt_batch_success += 1
                    mech.debt_batch_time_success += batch_dur
                elif promoted_candidate_id is None:
                    mech.debt_batch_no_eligible += 1
                    mech.debt_batch_time_no_eligible += batch_dur
                else:
                    mech.debt_batch_candidate_blocked += 1
                    mech.debt_batch_time_candidate_blocked += batch_dur
            mech.batch_count += 1
            mech.prefill_token_sum += float(state.current_batch)
            mech.decode_token_sum += float(decode_tokens_this_batch)
            mech.continuation_count += continuation_steps
            mech.long_admit_normal += long_normal
            mech.long_admit_promoted += long_promoted
            mech.long_admit_eligible_at_entry += long_eligible
            if budget_break:
                mech.budget_break_count += 1
            if kv_break:
                mech.kv_break_count += 1
            if concurrency_break:
                mech.concurrency_break_count += 1
            if continuation_break:
                mech.continuation_break_count += 1

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
            mech_enabled=bool(mech.enabled),
            mech_debt_owed=bool(debt_owed_this_batch),
            mech_promoted_candidate=bool(promoted_candidate_id is not None),
            mech_promoted_admitted=bool(promoted_admitted),
            mech_aged_long_count=int(aged_long_count),
            mech_budget_break=bool(budget_break),
            mech_kv_break=bool(kv_break),
            mech_concurrency_break=bool(concurrency_break),
            mech_continuation_break=bool(continuation_break),
            mech_prefill_tokens=int(state.current_batch),
            mech_decode_tokens=int(decode_tokens_this_batch),
            mech_active_decode=int(len(state.current_batch_decode)),
            mech_residual_prefill_budget=int(prefill_budget) - int(state.current_batch),
            mech_continuation_steps=int(continuation_steps),
        )
        if batch_event.batch_usage > 0:
            heapq.heappush(heap, (batch_event.batch_time, batch_event.id, batch_event))
            state.engine_isbusy = True
            return False
        else:
            return True

    return False
