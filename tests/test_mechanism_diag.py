"""Tier2 旁路机制统计（``mech_diag_enabled``）测试。

核心不变量：
1. 开关默认关闭，``mechanism_metrics`` 返回空 dict。
2. **开启与关闭统计时，请求顺序、等待时间与 promotion 数完全一致**（零行为改变）。
3. 开启时计数自洽：promotion 计数与 ``psfaq_promoted_total`` 一致；债务批互斥分桶；
   η 落在 [0, 1]；break 计数与 batch_list 快照对应（允许末尾未处理批 1 的容差）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llm_in_queuing.config import load_config  # noqa: E402
from llm_in_queuing.metrics import mechanism_metrics  # noqa: E402
from llm_in_queuing.simulator import run  # noqa: E402

_BASE_YAML = (_REPO_ROOT / "scripts" / "experiment" / "configs"
              / "sweep_composition_load_base.yaml")


def _cfg(mech: bool, *, policy: str = "QBA", seed: int = 42, n: int = 800) -> dict:
    cfg = load_config(_BASE_YAML)
    cfg["admission_number"] = int(n)
    cfg["schedule_method"] = policy
    cfg["seed"] = int(seed)
    cfg["mech_diag_enabled"] = bool(mech)
    return cfg


def _run_seq(state):
    return [(int(r.id), float(r.waiting_time),
             float(r.start_service_time), float(r.end_service_time))
            for r in state.finished_queue]


def _batch_seq(state):
    return [(int(b["id"]), float(b["sim_time"]),
             int(b["batch_usage"]), bool(b["current_OOM"]))
            for b in state.batch_list]


# ── 默认关闭 → 空指标 ────────────────────────────────────────

def test_mech_metrics_empty_when_disabled():
    _, state = run(_cfg(False), return_state=True)
    assert mechanism_metrics(state) == {}


# ── 开启 / 关闭零行为改变 ─────────────────────────────────────

@pytest.mark.parametrize("policy", ["FCFS", "PSF", "AA", "QBA"])
def test_mech_off_on_request_order_identical(policy):
    m_off, s_off = run(_cfg(False, policy=policy), return_state=True)
    m_on, s_on = run(_cfg(True, policy=policy), return_state=True)

    for k in ("waiting_time平均", "waiting_time p95", "waiting_time p99",
              "waiting_time_true_long_mean", "waiting_time_true_long_p99",
              "PSFAQ_promoted_total", "OOM概率"):
        assert m_off[k] == m_on[k], f"{policy}: metrics key {k} changed"

    assert _run_seq(s_off) == _run_seq(s_on), f"{policy}: finished order changed"
    assert _batch_seq(s_off) == _batch_seq(s_on), f"{policy}: batch order changed"


# ── 开启时计数自洽 ────────────────────────────────────────────

def test_mech_counters_self_consistent():
    _, state = run(_cfg(True), return_state=True)
    mm = mechanism_metrics(state)
    assert mm, "mech metrics should be non-empty when enabled"

    promoted = float(state.psfaq_promoted_total)
    assert mm["mech_promotion_success_count"] == promoted
    assert mm["mech_long_admit_promoted"] == promoted
    # 债务批互斥分桶：成功数 == promotion 数
    assert mm["mech_debt_batch_success"] == promoted
    total_buckets = (mm["mech_debt_batch_no_eligible"]
                     + mm["mech_debt_batch_candidate_blocked"]
                     + mm["mech_debt_batch_success"])
    assert total_buckets <= mm["mech_batch_count"]
    # η 定义域
    for k in ("mech_eta_eligibility", "mech_eta_feasibility"):
        assert 0.0 <= mm[k] <= 1.0
    # 债务 episode 数 >= 已解决 episode 数
    assert mm["mech_debt_episode_count"] >= mm["mech_promotion_success_count"]


def test_mech_break_counts_match_snapshots():
    _, state = run(_cfg(True), return_state=True)
    mm = mechanism_metrics(state)
    snaps = state.batch_list

    # batch_count 与快照一一对应（允许末尾 1 个已调度但未处理完成的批）
    assert abs(mm["mech_batch_count"] - len(snaps)) <= 1

    def _snap_sum(key):
        return sum(1 for b in snaps if b.get(key, False))

    assert abs(mm["mech_budget_break_count"] - _snap_sum("mech_budget_break")) <= 1
    assert abs(mm["mech_kv_break_count"] - _snap_sum("mech_kv_break")) <= 1
    assert abs(mm["mech_concurrency_break_count"] - _snap_sum("mech_concurrency_break")) <= 1
    assert abs(mm["mech_continuation_break_count"] - _snap_sum("mech_continuation_break")) <= 1


def test_mech_promoted_flag_matches_promoted_total():
    """被提升请求的 ``promoted`` 标记数应与 ``psfaq_promoted_total`` 一致。"""
    _, state = run(_cfg(True), return_state=True)
    flagged = sum(1 for r in state.finished_queue if getattr(r, "promoted", False))
    assert flagged == int(state.psfaq_promoted_total)
