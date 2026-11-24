"""逐请求总 token 守恒的构成扫描（fig11）适配层的一致性测试。

只验证核心不变量（逐请求总 token 守恒、长度约束、a=1 还原、标签重算、适配层可恢复、
边界请求），不复现百种子回归、不重跑全量 sweep。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llm_in_queuing import simulator  # noqa: E402
from llm_in_queuing.config import load_config  # noqa: E402
from llm_in_queuing.types import Request  # noqa: E402
from llm_in_queuing.workload import generate_heapq  # noqa: E402
from scripts.experiment import sweep_request_composition as src  # noqa: E402
from scripts.experiment._common import run_sim_stats  # noqa: E402

_BASE_YAML = _REPO_ROOT / "scripts" / "experiment" / "configs" / "sweep_request_composition_base.yaml"


def _small_cfg() -> dict:
    """小规模配置：只用于验证逻辑，不用于科学结论。"""
    cfg = load_config(_BASE_YAML)
    cfg["admission_number"] = 300
    return cfg


def _overrides(policy: str, seed: int) -> dict:
    return {"lambda_arrival": 32.0, "schedule_method": policy, "seed": int(seed)}


def _base_labels(cfg: dict) -> dict:
    heap = generate_heapq(dict(cfg))
    return {r.id: r.true_label for _, _, r in heap}


def _snapshot(cfg: dict, scale: float, base_labels: dict | None = None) -> tuple[dict, dict]:
    """按给定倍率生成并变换请求，返回 ({id: Request}, diag)。"""
    diag: dict = {}
    heap = generate_heapq(dict(cfg))
    src._apply_request_composition(heap, scale, cfg, diag, base_labels)
    return {r.id: r for _, _, r in heap}, diag


def _synth_heap(pairs: list[tuple[int, int]]) -> list:
    """构造合成请求堆（用于边界用例）：返回 [(arrival_time, id, Request)]。"""
    heap = []
    for i, (p, d) in enumerate(pairs):
        req = Request(
            id=i, arrival_time=float(i), prompt_len=int(p),
            answer_len=int(d), prefill_res_token=int(p),
        )
        heap.append((float(i), i, req))
    return heap


# ── 倍率解析 ────────────────────────────────────────────────

def test_parse_composition_scales_sorts_ascending():
    scales = src._parse_composition_scales("1, 4,0.25,2,0.5")
    assert scales.tolist() == [0.25, 0.5, 1.0, 2.0, 4.0]


@pytest.mark.parametrize("bad", ["", "0", "-1", "1,1", "nan", "inf", "1,abc"])
def test_parse_composition_scales_rejects_invalid(bad):
    with pytest.raises(ValueError):
        src._parse_composition_scales(bad)


# ── 验证 A：请求级不变量（id/到达时间/长度约束/prefill 一致性） ──

@pytest.mark.parametrize("a", [0.25, 0.5, 2.0, 4.0])
def test_request_level_invariants(a):
    cfg = _small_cfg()
    base = {r.id: r for _, _, r in generate_heapq(dict(cfg))}
    got, diag = _snapshot(cfg, a)

    p_cap = int(cfg["prompt_len_cap"])
    a_cap = int(cfg["answer_len_cap"])
    assert set(base) == set(got)
    for rid, r in got.items():
        b = base[rid]
        # id 与到达时间不变
        assert r.id == b.id
        assert r.arrival_time == b.arrival_time
        # 逐请求总 token 严格守恒
        assert r.prompt_len + r.answer_len == b.prompt_len + b.answer_len
        # 长度约束
        assert 5 <= r.prompt_len <= p_cap
        assert 1 <= r.answer_len <= a_cap
        # prefill 剩余 token 与 prompt 一致
        assert r.prefill_res_token == r.prompt_len
    assert diag["max_request_total_error"] == 0.0


def test_input_heap_time_keys_unchanged():
    """变换不得改变输入堆的 (arrival_time, id) 时间键。"""
    cfg = _small_cfg()
    heap = generate_heapq(dict(cfg))
    keys_before = [(k, i) for k, i, _ in heap]
    src._apply_request_composition(heap, 4.0, cfg, {}, None)
    keys_after = [(k, i) for k, i, _ in heap]
    assert keys_before == keys_after


def test_apply_does_not_consume_randomness():
    """确定性标签重算不得改变全局随机状态（不消耗额外随机数）。"""
    cfg = _small_cfg()
    heap = generate_heapq(dict(cfg))
    labels = {r.id: r.true_label for _, _, r in heap}

    np.random.seed(20260101)
    before = np.random.get_state()
    src._apply_request_composition(heap, 4.0, cfg, {}, labels)
    after = np.random.get_state()

    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


# ── 验证 B：逐请求总 token 守恒（均值与方差跨倍率一致） ──────

@pytest.mark.parametrize("a", [0.25, 0.5, 2.0, 4.0])
def test_per_request_total_preserved(a):
    cfg = _small_cfg()
    base, base_diag = _snapshot(cfg, 1.0)
    got, diag = _snapshot(cfg, a)

    for rid, r in got.items():
        b = base[rid]
        assert r.prompt_len + r.answer_len == b.prompt_len + b.answer_len

    assert diag["max_request_total_error"] == 0.0
    assert diag["total_mean"] == pytest.approx(diag["base_total_mean"], rel=0, abs=0)
    assert diag["total_var"] == pytest.approx(diag["base_total_var"], rel=0, abs=0)
    assert diag["total_mean"] == pytest.approx(base_diag["base_total_mean"], rel=0, abs=0)


# ── 验证 C：构成比随 a 上升 ─────────────────────────────────

def test_token_ratio_increases_with_a():
    cfg = _small_cfg()
    _, d_low = _snapshot(cfg, 0.25)
    _, d_one = _snapshot(cfg, 1.0)
    _, d_high = _snapshot(cfg, 4.0)
    assert d_low["token_ratio"] < d_one["token_ratio"] < d_high["token_ratio"]


# ── 验证 D：a=1 完全还原原始 workload（含标签） ──────────────

def test_scale_one_is_identity():
    cfg = _small_cfg()
    raw = {r.id: r for _, _, r in generate_heapq(dict(cfg))}
    got, diag = _snapshot(cfg, 1.0)

    assert set(raw) == set(got)
    for rid, r in got.items():
        b = raw[rid]
        assert r.prompt_len == b.prompt_len
        assert r.prefill_res_token == b.prompt_len
        assert r.answer_len == b.answer_len
        assert r.true_label == b.true_label
        assert r.predict_label == b.predict_label
    assert diag["max_request_total_error"] == 0.0
    assert diag["lower_bound_fraction"] == 0.0
    assert diag["upper_bound_fraction"] == 0.0
    assert diag["token_ratio"] == pytest.approx(
        np.mean([r.prompt_len for r in raw.values()])
        / np.mean([r.answer_len for r in raw.values()]),
    )


# ── 验证 E：标签按新 answer 分布重算且为 oracle ─────────────

def test_labels_recomputed_on_new_distribution():
    cfg = _small_cfg()
    base = {r.id: r for _, _, r in generate_heapq(dict(cfg))}
    labels = {r.id: r.true_label for r in base.values()}

    got, diag = _snapshot(cfg, 4.0, base_labels=labels)

    ans = np.array([r.answer_len for r in got.values()], dtype=np.float64)
    thr = float(np.quantile(ans, float(cfg["label_threshold_percentail"])))
    for r in got.values():
        expected = "short" if r.answer_len <= thr else "long"
        assert r.true_label == expected
        assert r.predict_label == r.true_label  # oracle（无误判）
    # 逐请求缩放可能改变 answer 排序，flip 比例只记录、不设阈值；但仍应在 [0,1]
    assert 0.0 <= diag["label_flip_fraction"] <= 1.0


def test_non_oracle_config_rejected():
    """非 oracle 误判率配置应被显式拒绝。"""
    cfg = _small_cfg()
    cfg["FN"] = 0.1
    heap = generate_heapq(dict(cfg))
    with pytest.raises(ValueError, match=r"oracle"):
        src._apply_request_composition(heap, 2.0, cfg, {}, None)


def test_work_diag_records_pi_s_base_total_and_variance():
    cfg = _small_cfg()
    got, diag = _snapshot(cfg, 4.0)

    # pi_s 应与真实 short 占比一致
    n_short = sum(1 for r in got.values() if r.true_label == "short")
    assert diag["pi_s"] == pytest.approx(n_short / len(got))
    assert 0.0 <= diag["pi_s"] <= 1.0

    # base_total_mean/var 为 a=1 基准，且变换后总 token 的均值与方差完全守恒
    _, diag_one = _snapshot(cfg, 1.0)
    assert diag["base_total_mean"] == pytest.approx(diag_one["base_total_mean"])
    assert diag["base_total_var"] == pytest.approx(diag_one["base_total_var"])
    assert diag["total_mean"] == pytest.approx(diag["base_total_mean"])
    assert diag["total_var"] == pytest.approx(diag["base_total_var"])


# ── 验证 F：边界请求（联合可行区间唯一/受限） ───────────────

def test_boundary_requests_total_preserved():
    """构造极端请求，验证任何倍率下都逐请求守恒且满足长度约束。

    其中 (4096, 4096) 的 T=8192 使可行区间退化为单点，所有倍率必须保持 4096+4096。
    """
    cfg = _small_cfg()
    pairs = [
        (5, 1),        # prompt/answer 同时位于下限
        (5, 4096),     # answer 到达上限
        (4096, 1),     # prompt 到达上限
        (4096, 4096),  # 两者均为上限：T=8192，可行区间单点
        (5, 6),        # 总长度很小
        (100, 100),    # 常规
    ]
    for a in [0.25, 0.5, 1.0, 2.0, 4.0]:
        heap = _synth_heap(pairs)
        diag: dict = {}
        src._apply_request_composition(heap, a, cfg, diag, None)
        for (p0, d0), (_, _, r) in zip(pairs, heap):
            assert r.prompt_len + r.answer_len == p0 + d0
            assert 5 <= r.prompt_len <= int(cfg["prompt_len_cap"])
            assert 1 <= r.answer_len <= int(cfg["answer_len_cap"])
        assert diag["max_request_total_error"] == 0.0

    # T=8192 的极端请求：可行区间单点，所有倍率都必须是 4096+4096
    for a in [0.25, 0.5, 1.0, 2.0, 4.0]:
        heap = _synth_heap([(4096, 4096)])
        src._apply_request_composition(heap, a, cfg, {}, None)
        _, _, r = heap[0]
        assert r.prompt_len == 4096
        assert r.answer_len == 4096


def test_boundary_adjustment_fractions_reported():
    """受限请求应体现在 lower/upper_bound_fraction 中，且比例在 [0,1]。"""
    cfg = _small_cfg()
    heap = _synth_heap([(4096, 4096), (5, 1), (100, 100)])
    diag: dict = {}
    src._apply_request_composition(heap, 4.0, cfg, diag, None)
    assert 0.0 <= diag["lower_bound_fraction"] <= 1.0
    assert 0.0 <= diag["upper_bound_fraction"] <= 1.0


# ── 验证 G：a=1 数值锚点 + 恢复路径 ─────────────────────────

def test_scale_one_matches_raw_simulation():
    """经 a=1 适配层的仿真，与直接运行的指标逐位一致，且生成器已被恢复。"""
    cfg = _small_cfg()
    original = simulator.generate_heapq
    policies = ["FCFS", "QBA"]
    seed = int(cfg["seed"])

    rows, _, work_diag = src._run_request_composition_sweep(
        cfg, 32.0, np.asarray([1.0]), policies, [seed],
    )
    assert simulator.generate_heapq is original  # 已恢复
    for pol in policies:
        ref = run_sim_stats(cfg, _overrides(pol, seed))
        assert rows[pol]["mean"][0, 0] == ref["mean"]
        assert rows[pol]["p99"][0, 0] == ref["p99"]
    assert work_diag["max_request_total_error"][0, 0] == 0.0


def test_generator_restored_on_failure(monkeypatch):
    """仿真抛异常时，finally 仍恢复原生成器，并带上实验上下文。"""
    cfg = _small_cfg()
    original = simulator.generate_heapq

    def _boom(cfg):
        raise RuntimeError("boom")

    monkeypatch.setattr(src, "run_one_sim", _boom)
    with pytest.raises(RuntimeError, match=r"Experiment failed: seed=.*composition_scale=.*policy="):
        src._run_request_composition_sweep(cfg, 32.0, np.asarray([2.0]), ["FCFS"], [42])
    assert simulator.generate_heapq is original
