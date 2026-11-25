"""fig14: 长请求流负载—构成实验适配层的一致性测试。

只验证核心不变量（长度解析、前缀一致性、逐请求总 token 守恒、序号/时间双口径
窗口守恒、生成器可恢复、机制数组存在），不复现百种子回归、不重跑全量 sweep。
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
from scripts.experiment import sweep_long_stream as src  # noqa: E402

_BASE_YAML = (_REPO_ROOT / "scripts" / "experiment" / "configs"
              / "sweep_long_stream_base.yaml")


def _cfg() -> dict:
    """基础配置（admission_number 由各测试按需覆盖）。"""
    return load_config(_BASE_YAML)


def _base_stream(cfg: dict, lam: float, n_max: int) -> list:
    """按固定 seed / λ 生成最长基础流（与主脚本一致）。"""
    base_cfg = dict(cfg)
    base_cfg.update({"lambda_arrival": float(lam), "admission_number": int(n_max)})
    heap = simulator.generate_heapq(base_cfg)
    return sorted((req for _, _, req in heap), key=lambda r: int(r.id))


# ── 长度解析 ────────────────────────────────────────────────

def test_parse_lengths_sorts_ascending():
    assert src._parse_lengths("480, 120,240").tolist() == [120, 240, 480]


@pytest.mark.parametrize("bad", ["", "0", "-1", "120,120", "nan", "1.5", "abc"])
def test_parse_lengths_rejects_invalid(bad):
    with pytest.raises(ValueError):
        src._parse_lengths(bad)


# ── quota_k 网格解析 ────────────────────────────────────────

def test_parse_ks_sorts():
    assert src._parse_ks("9, 3,5").tolist() == [3, 5, 9]


@pytest.mark.parametrize("bad", ["", "0", "-2", "nan", "3.5", "abc", "3,3"])
def test_parse_ks_rejects_invalid(bad):
    with pytest.raises(ValueError):
        src._parse_ks(bad)


# ── 主扫参：4D 布局 [n_seeds, n_a, n_lambda, n_N] ───────────

def test_sweep_shape_and_conservation():
    cfg = _cfg()
    seed = int(cfg["seed"])
    scales = np.asarray([0.5, 2.0])
    loads = np.asarray([30.0, 32.0])
    lengths = np.asarray([60, 120])
    out = src._run_long_stream_sweep(
        cfg, scales, loads, lengths, ["QBA"], [seed], n_windows=3)
    (rows, policy_diag, stream, mech, transform_diag, prefix_diag, window,
     mechk) = out

    shape = (1, 2, 2, 2)
    assert rows["QBA"]["mean"].shape == shape
    assert policy_diag["QBA"]["oom_prob"].shape == shape
    assert stream["QBA"]["drain_time"].shape == shape
    # 积压/提升双口径键均存在
    for sk in src._STREAM_KEYS:
        assert stream["QBA"][sk].shape == shape
    assert mech["QBA"]["mech_debt_episode_count"].shape == shape
    # 变换诊断只依赖 (seed, a, λ)（基于 N_max）
    assert transform_diag["max_request_total_error"].shape == (1, 2, 2)
    assert np.all(transform_diag["max_request_total_error"] == 0.0)
    # 实际分类阈值随 (seed, a, λ) 记录，供分阶段补跑合并校验
    assert transform_diag["label_threshold"].shape == (1, 2, 2)
    # 前缀诊断按 N 维度
    assert prefix_diag["pi_s"].shape == shape
    # 双口径窗口 [n_seeds, n_a, n_l, n_N, n_windows]，边界多一维
    for mode in ("index", "time"):
        assert window["QBA"][mode]["wait_mean"].shape == (1, 2, 2, 2, 3)
        assert window["QBA"][mode]["edges"].shape == (1, 2, 2, 2, 4)
    # 未启用机制 k 网格时，mechk 末维为 0（空数组）
    assert mechk["mech_eta_eligibility"].shape == (1, 2, 2, 0)


def test_label_reference_n_stabilizes_prefix_labels():
    """固定 --label-reference-n 时，较短 N 的标签阈值不变（分阶段补跑稳定）。"""
    cfg = _cfg()
    seed = int(cfg["seed"])
    scales = np.asarray([1.0])
    loads = np.asarray([32.0])
    # 同一 seed/λ/a，第一轮只跑 120、第二轮补跑 120+240；两轮都锚定 N_ref=480
    (_, _, _, _, tr_short, _, _, _) = src._run_long_stream_sweep(
        cfg, scales, loads, np.asarray([120]), ["QBA"], [seed], n_windows=2,
        label_reference_n=480)
    (_, _, _, _, tr_long, _, _, _) = src._run_long_stream_sweep(
        cfg, scales, loads, np.asarray([120, 240]), ["QBA"], [seed], n_windows=2,
        label_reference_n=480)
    # 共享同一 N_ref 参考流 → 阈值逐位一致
    assert (tr_short["label_threshold"][0, 0, 0]
            == tr_long["label_threshold"][0, 0, 0])

    # 锚定 N_ref 时，两轮前缀（120 vs 120+240）各自得到的标签完全一致
    cfg2 = _cfg()
    seed2 = int(cfg2["seed"])
    (_, _, _, _, _, pre_a, _, _) = src._run_long_stream_sweep(
        cfg2, scales, loads, np.asarray([120]), ["QBA"], [seed2], n_windows=2,
        label_reference_n=480)
    (_, _, _, _, _, pre_b, _, _) = src._run_long_stream_sweep(
        cfg2, scales, loads, np.asarray([120, 240]), ["QBA"], [seed2], n_windows=2,
        label_reference_n=480)
    assert pre_a["pi_s"][0, 0, 0, 0] == pre_b["pi_s"][0, 0, 0, 0]


def test_mechk_grid_shape_and_finite():
    """启用 quota_k 网格时，mechk 形状 [n_seeds, n_a, n_l, n_k]，QBA 指标有限。"""
    cfg = _cfg()
    seed = int(cfg["seed"])
    scales = np.asarray([0.5, 2.0])
    loads = np.asarray([30.0, 32.0])
    lengths = np.asarray([60, 120])
    ks = np.asarray([3, 5])
    n_k = 2
    (_, _, _, _, _, _, _,
     mechk) = src._run_long_stream_sweep(
        cfg, scales, loads, lengths, ["QBA"], [seed], n_windows=3,
        quota_ks=ks, mech_n=120)

    shape_mk = (1, 2, 2, n_k)
    for mk in src._MECHK_KEYS:
        assert mechk[mk].shape == shape_mk
    # QBA 提升相关量应为有限非负值
    promo = mechk["promotion_ratio"]
    assert np.all(np.isfinite(promo))
    assert np.all(promo >= 0.0)


def test_qba_identity_anchor_matches_raw():
    """a=1、N=admission_number 时与直接仿真逐位一致（验收锚点）。"""
    cfg = _cfg()
    seed = int(cfg["seed"])
    n = 150
    cfg["admission_number"] = n
    ref_cfg = dict(cfg)
    ref_cfg.update({"lambda_arrival": 32.0, "schedule_method": "QBA"})
    metrics, _ = simulator.run(ref_cfg, return_state=True)
    ref_mean = float(metrics.get("waiting_time平均", float("nan")))

    rows, *_ = src._run_long_stream_sweep(
        cfg, np.asarray([1.0]), np.asarray([32.0]), np.asarray([n]),
        ["QBA"], [seed], n_windows=3)
    assert rows["QBA"]["mean"][0, 0, 0, 0] == ref_mean


# ── 前缀一致性 ──────────────────────────────────────────────

def test_prefix_streams_are_nested_and_conserving():
    cfg = _cfg()
    n_max = 200
    base_reqs = _base_stream(cfg, 32.0, n_max)
    p_new, d_new, thr = src._transform_lengths(base_reqs, 2.0, cfg)
    base_arrival = np.array([float(r.arrival_time) for r in base_reqs])
    ids = np.array([int(r.id) for r in base_reqs])

    small = src._make_prefix_generator(base_arrival, p_new, d_new, ids, 80, thr)
    large = src._make_prefix_generator(base_arrival, p_new, d_new, ids, 200, thr)
    hs = sorted(small(cfg), key=lambda t: int(t[2].id))
    hl = sorted(large(cfg), key=lambda t: int(t[2].id))
    assert len(hs) == 80 and len(hl) == 200
    assert [int(t[2].id) for t in hs] == ids[:80].tolist()

    # 前缀逐字段一致
    for a, b in zip(hs, hl[:80]):
        ra, rb = a[2], b[2]
        assert ra.id == rb.id
        assert ra.arrival_time == rb.arrival_time
        assert ra.prompt_len == rb.prompt_len
        assert ra.answer_len == rb.answer_len
        assert ra.true_label == rb.true_label

    # 逐请求总 token 守恒（Request 层面）
    tot = np.array([r.prompt_len + r.answer_len for _, _, r in hl], dtype=np.int64)
    base_tot = np.array([r.prompt_len + r.answer_len for r in base_reqs], dtype=np.int64)
    assert np.array_equal(tot, base_tot)


def test_prefix_short_ratio_matches_threshold():
    d = np.arange(1, 11, dtype=np.int64)
    n_short, ratio = src._prefix_short_ratio(d, 10, 5.0)
    assert n_short == 5
    assert ratio == pytest.approx(0.5)
    n_short2, ratio2 = src._prefix_short_ratio(d, 6, 5.0)
    assert n_short2 == 5
    assert ratio2 == pytest.approx(5.0 / 6.0)


def test_prefix_short_ratio_conserves_full_stream():
    """共同阈值取自 N_max 时，完整流 short 比例接近目标 p。"""
    cfg = _cfg()
    n_max = 2000
    base_reqs = _base_stream(cfg, 32.0, n_max)
    _, d_new, thr = src._transform_lengths(base_reqs, 1.0, cfg)
    n_short, ratio = src._prefix_short_ratio(d_new, n_max, thr)
    assert ratio == pytest.approx(float(cfg["label_threshold_percentail"]), abs=0.02)
    assert n_short > 0


# ── 双口径窗口守恒 ──────────────────────────────────────────

def test_window_conservation():
    cfg = _cfg()
    cfg["admission_number"] = 200
    cfg["schedule_method"] = "QBA"
    cfg["lambda_arrival"] = 32.0
    _, state = simulator.run(dict(cfg), return_state=True)
    n = len(state.finished_queue)
    assert n > 0
    w = 5

    idx, idx_edges = src._window_arrays(state.finished_queue, state.batch_list, w, "index")
    # 序号口径：backlog 恒为 0，count == arrivals == completions
    assert np.all(idx["backlog"] == 0.0)
    assert np.all(idx["arrivals"] == idx["completions"])
    assert np.nansum(idx["count"]) == n
    # 序号边界为请求序号（w+1 个边界，首 0 尾 n）
    assert idx_edges.shape == (w + 1,)
    assert idx_edges[0] == 0.0 and idx_edges[-1] == float(n)

    tm, tm_edges = src._window_arrays(state.finished_queue, state.batch_list, w, "time")
    diag = src._scalar_stream_diag(state.finished_queue)
    # 时间口径：到达总数守恒为 N（所有请求都在 [0, t_last_arrival] 内到达）
    assert np.nansum(tm["arrivals"]) == n
    # 完成总数 ≤ N：时间窗口不含排空阶段，期间未完成者由 unfinished_at_arrival_stop 承担
    assert np.nansum(tm["completions"]) == n - diag["unfinished_at_arrival_stop"]
    # 时间窗口覆盖 [0, t_last_arrival]（不含排空），边界长度 w+1
    assert tm_edges.shape == (w + 1,)
    assert tm_edges[0] == 0.0
    assert tm_edges[-1] == pytest.approx(diag["t_last_arrival"])


def test_scalar_stream_diag_consistency():
    cfg = _cfg()
    cfg["admission_number"] = 200
    cfg["schedule_method"] = "QBA"
    cfg["lambda_arrival"] = 32.0
    _, state = simulator.run(dict(cfg), return_state=True)
    diag = src._scalar_stream_diag(state.finished_queue)
    assert diag["drain_time"] == pytest.approx(
        diag["t_last_completion"] - diag["t_last_arrival"])
    assert 0.0 <= diag["promotion_ratio"] <= 1.0
    # 积压双口径：unfinished ≥ waiting（未完成一定尚未开始或正在服务）
    assert 0.0 <= diag["unfinished_fraction"] <= 1.0
    assert diag["unfinished_at_arrival_stop"] >= diag["waiting_at_arrival_stop"] >= 0.0
    # 提升双口径：phi_Q 为 NaN（无 predicted-long）或 ≥ promotion_ratio（分母更小）
    phi = diag["phi_Q"]
    if np.isfinite(phi):
        assert 0.0 <= phi <= 1.0
        assert phi >= diag["promotion_ratio"] - 1e-12


# ── 生成器恢复 ──────────────────────────────────────────────

def test_generator_restored_on_failure(monkeypatch):
    cfg = _cfg()
    original = simulator.generate_heapq

    def _boom(cfg):
        raise RuntimeError("boom")

    monkeypatch.setattr(src.simulator, "run", _boom)
    with pytest.raises(RuntimeError,
                       match=r"Experiment failed: seed=.* a=.* lambda=.* N=.* policy="):
        src._run_long_stream_sweep(
            cfg, np.asarray([2.0]), np.asarray([38.0]), np.asarray([100]),
            ["FCFS"], [42], n_windows=3)
    assert simulator.generate_heapq is original
