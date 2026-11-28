from __future__ import annotations

from dataclasses import dataclass, field

@dataclass
class Request:
    id: int
    arrival_time: float
    prompt_len: int
    answer_len: int
    start_service_time: float = 0.0
    end_service_time: float = 0.0
    TTFT: float = 0.0
    waiting_time: float = 0.0
    processed_len: int = 0
    prefill_chunk: bool = False
    prefill_res_token: int = 0
    true_label: str = ""
    predict_label: str = ""
    in_processing: bool = False
    # 独立的「是否已开始服务」状态：不依赖 start_service_time == 0.0 哨兵，
    # 避免 t=0 开始服务的请求被误判为未开始（chunked prefill 会重复标记）。
    started: bool = False
    # QBA：是否经由 promotion（保护）通道准入。仅作旁路标记，不参与调度决策。
    promoted: bool = False


@dataclass
class MechanismDiag:
    """Tier2 旁路机制统计（仅当 ``cfg["mech_diag_enabled"]`` 为真时累积）。

    全部为只读旁路计数/记录，不参与任何调度决策；默认关闭时所有字段保持初值，
    对请求顺序、等待时间与 promotion 数零影响（由 ``test_mechanism_diag.py`` 保证）。
    """
    enabled: bool = False
    # ── Quota / debt episodes ──
    debt_episode_count: int = 0
    debt_episode_durations: list = field(default_factory=list)
    debt_active: bool = False
    debt_start_time: float = 0.0
    debt_unresolved_at_end: int = 0
    debt_unresolved_duration: float = 0.0
    promotion_success_count: int = 0
    # ── debt 存续期间的按批分类（一个 batch 只计入一个桶，条件不叠加）──
    debt_batch_no_eligible: int = 0        # 有债务但等待队列中无 aged-long 候选
    debt_batch_candidate_blocked: int = 0  # 有候选但未被准入（预算/KV/并发阻塞）
    debt_batch_success: int = 0            # 有候选且成功提升
    debt_batch_kv_gate_closed: int = 0     # 有债务但 KV 安全门关闭，未调度
    debt_batch_time_no_eligible: float = 0.0
    debt_batch_time_candidate_blocked: float = 0.0
    debt_batch_time_success: float = 0.0
    # ── long 去向 ──
    long_admit_normal: int = 0             # predicted-long 经普通通道准入
    long_admit_promoted: int = 0           # predicted-long 经 promotion 准入
    long_admit_eligible_at_entry: int = 0  # 普通准入时已满足 aging eligibility
    # ── 资源 / 执行 ──
    budget_break_count: int = 0            # 因无剩余 prefill budget 而终止本轮
    kv_break_count: int = 0                # 因 KV 投影越界而终止本轮
    concurrency_break_count: int = 0       # 因达到最大并发 decode 而终止本轮
    continuation_break_count: int = 0      # 因 chunk continuation 残留下轮而终止本轮
    prefill_token_sum: float = 0.0
    decode_token_sum: float = 0.0
    continuation_count: int = 0            # 被准入的 chunk continuation 步数
    batch_count: int = 0


@dataclass
class BatchEvent:
    id: int = 0
    batch_time: float = 0.0
    batch_usage: int = 0
    current_concurrent: int = 0
    current_KV_capacity: int = 0
    current_OOM: bool = False
    prefill_requests: list[Request] = field(default_factory=list)
    prefill_token_map: dict[int, int] = field(default_factory=dict)
    decode_requests: list[Request] = field(default_factory=list)
    # ── Tier2 旁路机制快照（仅 mech_diag_enabled 时填充并写入 batch_list）──
    mech_enabled: bool = False
    mech_debt_owed: bool = False
    mech_promoted_candidate: bool = False
    mech_promoted_admitted: bool = False
    mech_aged_long_count: int = 0
    mech_budget_break: bool = False
    mech_kv_break: bool = False
    mech_concurrency_break: bool = False
    mech_continuation_break: bool = False
    mech_prefill_tokens: int = 0
    mech_decode_tokens: int = 0
    mech_active_decode: int = 0
    mech_residual_prefill_budget: int = 0
    mech_continuation_steps: int = 0


@dataclass
class EngineState:
    current_time: float = 0.0
    current_batch: int = 0
    current_batch_time: float = 0.0
    current_KV_capacity: int = 0
    current_concurrent: int = 0
    KV_total_capacity: int = 8192
    KV_safety_threshold: float = 0.0
    budget_tokens: int = 128
    request_block_size: int = 8
    wait_queue: list[Request] = field(default_factory=list)
    current_batch_prefill: list[Request] = field(default_factory=list)
    current_batch_decode: list[Request] = field(default_factory=list)
    finished_queue: list[Request] = field(default_factory=list)
    batch_list: list = field(default_factory=list)
    engine_isbusy: bool = False
    schedule_method: str = ""
    # QBA 相关
    quota_k: int = 5
    psfaq_short_admit_since_last_promoted: int = 0
    psfaq_promotion_debt: bool = False
    psfaq_promoted_total: int = 0
    psfaq_promoted_this_batch: int = 0
    # RR 相关
    rr_next_class: str = "short"
    # 迭代/老化相关
    itr: int = 0
    gamma: float = 0.95
    ageing_measure: str = "waiting_queue"
    # 等待队列统计
    wait_queue_n: int = 0
    wait_queue_sum_arrival: float = 0.0
    wait_queue_sum_arrival_sq: float = 0.0
    wait_queue_wait_mean: float = 0.0
    wait_queue_wait_var: float = 0.0
    wait_queue_wait_std: float = 0.0
    # 到达但未开始请求统计
    arrived_not_started_n: int = 0
    arrived_not_started_sum_arrival: float = 0.0
    arrived_not_started_sum_arrival_sq: float = 0.0
    # 已开始请求统计
    arrived_started_n: int = 0
    arrived_started_sum_wait: float = 0.0
    arrived_started_sum_wait_sq: float = 0.0
    arrived_started_waits: list[float] = field(default_factory=list)
    # 老化阈值记录
    aging_threshold_times: list[float] = field(default_factory=list)
    aging_threshold_values: list[float] = field(default_factory=list)
    # 稳态窗口
    t_warm_end: float = 0.0
    t_measure_end: float = 0.0
    # Tier2 旁路机制统计
    mech: "MechanismDiag" = field(default_factory=MechanismDiag)
