"""fig15: KV 容量敏感性实验适配层的一致性测试。

只验证核心不变量（容量解析、5D 布局、跨容量共享请求流/标签、KV 口径边界、
越界阈值等价、执行并发记录、生成器可恢复），不复现大规模种子回归、不重跑全量 sweep。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llm_in_queuing import simulator  # noqa: E402
from llm_in_queuing.config import load_config  # noqa: E402
from scripts.experiment import sweep_kv_capacity as src  # noqa: E402

_BASE_YAML = (_REPO_ROOT / "scripts" / "experiment" / "configs"
              / "sweep_kv_capacity_base.yaml")


def _cfg() -> dict:
    """基础配置（admission_number 由各测试按需覆盖）。"""
    return load_config(_BASE_YAML)


# ── 容量解析 ────────────────────────────────────────────────

def test_parse_capacities_sorts():
    assert src._parse_capacities("16384, 8192,12288").tolist() == [8192, 12288, 16384]


@pytest.mark.parametrize("bad", ["", "0", "-1", "8192,8192", "nan", "1.5", "abc"])
def test_parse_capacities_rejects_invalid(bad):
    with pytest.raises(ValueError):
        src._parse_capacities(bad)


# ── 主扫参：5D 布局 [n_seeds, n_a, n_lambda, n_K, n_N] ──────

def test_kv_sweep_shape_and_conservation():
    cfg = _cfg()
    seed = int(cfg["seed"])
    scales = np.asarray([0.5, 2.0])
    loads = np.asarray([30.0, 32.0])
    caps = np.asarray([8192, 16384])
    lengths = np.asarray([60, 120])
    res = src._run_kv_capacity_sweep(
        cfg, scales, loads, caps, lengths, ["QBA"], [seed], n_windows=3)

    shape = (1, 2, 2, 2, 2)
    assert res["rows"]["QBA"]["mean"].shape == shape
    assert res["policy_diag"]["QBA"]["oom_prob"].shape == shape
    assert res["stream"]["QBA"]["drain_time"].shape == shape
    for sk in src._STREAM_KEYS:
        assert res["stream"]["QBA"][sk].shape == shape
    for kk in src._KV_DIAG_KEYS:
        assert res["kv"]["QBA"][kk].shape == shape
    for ek in src._EXEC_DIAG_KEYS:
        assert res["exec"]["QBA"][ek].shape == shape
    for lk in src._LONG_ADMIT_KEYS:
        assert res["long_admit"]["QBA"][lk].shape == shape
    assert res["mech"]["QBA"]["mech_debt_episode_count"].shape == shape
    # 变换诊断只依赖 (seed, a, λ)（基于 N_ref），逐请求总 token 严格守恒
    assert res["transform"]["max_request_total_error"].shape == (1, 2, 2)
    assert np.all(res["transform"]["max_request_total_error"] == 0.0)
    assert res["transform"]["label_threshold"].shape == (1, 2, 2)
    # 前缀诊断只依赖 (seed, a, λ, N)，不含 K 维
    assert res["prefix"]["pi_s"].shape == (1, 2, 2, 2)
    assert res["prefix"]["n_short"].shape == (1, 2, 2, 2)
    assert np.all(np.isfinite(res["prefix"]["pi_s"]))
    assert np.all(np.isfinite(res["prefix"]["n_short"]))
    # 双口径窗口 [n_seeds, n_a, n_l, n_K, n_N, n_windows]，边界多一维
    for mode in ("index", "time"):
        assert res["window"]["QBA"][mode]["wait_mean"].shape == (1, 2, 2, 2, 2, 3)
        assert res["window"]["QBA"][mode]["edges"].shape == (1, 2, 2, 2, 2, 4)


def test_kv_capacity_shares_stream_across_K(monkeypatch):
    """同一 (seed, a, λ, N) 下，不同 K 的请求流与标签逐字段一致。"""
    cfg = _cfg()
    seen: list = []
    real_run = simulator.run

    def _capture(local, return_state=False):
        heap = simulator.generate_heapq(dict(local))
        reqs = sorted((r for _, _, r in heap), key=lambda r: int(r.id))
        seen.append((int(local["KV_total_capacity"]),
                     [(int(r.id), float(r.arrival_time), int(r.prompt_len),
                       int(r.answer_len), r.true_label, r.predict_label)
                      for r in reqs]))
        return real_run(local, return_state=return_state)

    monkeypatch.setattr(src.simulator, "run", _capture)
    src._run_kv_capacity_sweep(
        cfg, np.asarray([2.0]), np.asarray([32.0]),
        np.asarray([8192, 32768]), np.asarray([80]), ["QBA"], [42], n_windows=2)

    by_k = dict(seen)
    assert set(by_k.keys()) == {8192, 32768}
    assert by_k[8192] == by_k[32768]


def test_kv_diag_bounds():
    cfg = _cfg()
    res = src._run_kv_capacity_sweep(
        cfg, np.asarray([1.0]), np.asarray([32.0]),
        np.asarray([8192]), np.asarray([120]), ["QBA"], [42], n_windows=2)
    kv = res["kv"]["QBA"]
    assert 0.0 <= kv["kv_hard_touch_rate"][0, 0, 0, 0, 0] <= 1.0
    assert 0.0 <= kv["kv_gate_closed_rate"][0, 0, 0, 0, 0] <= 1.0
    peak_mean = kv["kv_peak_mean"][0, 0, 0, 0, 0]
    peak_max = kv["kv_peak_max"][0, 0, 0, 0, 0]
    over = kv["kv_peak_over_capacity_mean"][0, 0, 0, 0, 0]
    assert np.isfinite(peak_mean) and np.isfinite(peak_max) and np.isfinite(over)
    assert peak_max >= peak_mean >= 0.0
    # 峰值/容量 = 峰值均值 / K（K=8192）
    assert over == pytest.approx(peak_mean / 8192.0)


def test_execution_concurrency_recorded():
    cfg = _cfg()
    res = src._run_kv_capacity_sweep(
        cfg, np.asarray([1.0]), np.asarray([32.0]),
        np.asarray([8192]), np.asarray([120]), ["QBA"], [42], n_windows=2)
    ex = res["exec"]["QBA"]
    c_max = ex["exec_concurrent_max"][0, 0, 0, 0, 0]
    assert np.isfinite(c_max) and c_max >= 0.0
    ad_max = ex["exec_active_decode_max"][0, 0, 0, 0, 0]
    assert np.isfinite(ad_max) and ad_max >= 0.0


# ── 越界阈值等价检验 ────────────────────────────────────────

def test_out_of_bounds_threshold_equivalence():
    """(K=8192, s=0.7) 与 (K=32768, s=0.175) 绝对安全门相同 → 调度性能与 KV 轨迹一致。"""
    cfg = _cfg()
    n = 200
    base = dict(cfg)
    base.update({"lambda_arrival": 32.0, "admission_number": n, "seed": 42,
                 "schedule_method": "QBA"})

    cfg_a = dict(base)
    cfg_a.update({"KV_total_capacity": 8192, "KV_safety_threshold": 0.7})
    cfg_b = dict(base)
    cfg_b.update({"KV_total_capacity": 32768, "KV_safety_threshold": 0.175})

    ma, sa = simulator.run(cfg_a, return_state=True)
    mb, sb = simulator.run(cfg_b, return_state=True)

    # 绝对安全门一致（0.7*8192 = 0.175*32768 = 5734.4）
    assert sa.KV_safety_threshold == pytest.approx(5734.4)
    assert sb.KV_safety_threshold == pytest.approx(5734.4)
    assert sa.KV_safety_threshold == pytest.approx(sb.KV_safety_threshold)

    # 调度性能一致：等待与提升数逐位相同
    assert ma["waiting_time平均"] == mb["waiting_time平均"]
    assert ma["waiting_time p99"] == mb["waiting_time p99"]
    assert ma["PSFAQ_promoted_total"] == mb["PSFAQ_promoted_total"]

    # 绝对 KV 轨迹逐位相同（仅越界判定阈值不同）
    pa = [float(b["kv_peak"]) for b in sa.batch_list]
    pb = [float(b["kv_peak"]) for b in sb.batch_list]
    assert pa == pb


# ── 生成器恢复 ──────────────────────────────────────────────

def test_generator_restored_on_failure(monkeypatch):
    cfg = _cfg()
    original = simulator.generate_heapq

    def _boom(cfg):
        raise RuntimeError("boom")

    monkeypatch.setattr(src.simulator, "run", _boom)
    with pytest.raises(RuntimeError,
                       match=r"Experiment failed: seed=.* a=.* lambda=.* K=.* N=.* policy="):
        src._run_kv_capacity_sweep(
            cfg, np.asarray([2.0]), np.asarray([38.0]),
            np.asarray([8192]), np.asarray([100]), ["FCFS"], [42], n_windows=3)
    assert simulator.generate_heapq is original


# ── 前缀诊断：四维（不含 K）、有限、计数自洽 ────────────────

def test_prefix_diag_four_dims_finite_and_correct():
    """P1: 前缀诊断四维 [n_seeds,n_a,n_l,n_N]（不含 K），全部有限且计数自洽。

    覆盖 3 档容量与 3 档长度，确保写入不再把 N 索引落到 K 轴上（旧版会得到 NaN）。
    """
    cfg = _cfg()
    scales = np.asarray([0.5, 2.0])
    loads = np.asarray([30.0, 32.0])
    caps = np.asarray([8192, 12288, 32768])   # 3 档容量
    lengths = np.asarray([60, 120, 200])      # 3 档 N
    res = src._run_kv_capacity_sweep(
        cfg, scales, loads, caps, lengths, ["QBA"], [42, 43], n_windows=2)

    pi_s = res["prefix"]["pi_s"]
    n_short = res["prefix"]["n_short"]
    # 形状由 (seed, a, λ, N) 决定，与容量数量无关
    assert pi_s.shape == (2, 2, 2, 3)
    assert n_short.shape == (2, 2, 2, 3)
    assert np.all(np.isfinite(pi_s))
    assert np.all(np.isfinite(n_short))
    # 计数范围与比例关系
    assert np.all(n_short >= 0.0)
    assert np.all((pi_s >= 0.0) & (pi_s <= 1.0))
    for ni, n in enumerate(lengths.tolist()):
        assert np.all(n_short[..., ni] <= float(n))
        assert np.allclose(pi_s[..., ni], n_short[..., ni] / float(n))


# ── 执行并发：非 QBA 的 active_decode 记为 NaN ──────────────

def test_exec_active_decode_nan_for_non_qba():
    """P2: 非 QBA 策略未采集 active_decode → NaN（不能写成 0）。"""
    cfg = _cfg()
    res = src._run_kv_capacity_sweep(
        cfg, np.asarray([1.0]), np.asarray([32.0]),
        np.asarray([8192]), np.asarray([120]),
        ["FCFS", "PSF", "AA", "QBA"], [42], n_windows=2)
    for pol in ("FCFS", "PSF", "AA"):
        ex = res["exec"][pol]
        assert np.isnan(ex["exec_active_decode_mean"][0, 0, 0, 0, 0])
        assert np.isnan(ex["exec_active_decode_max"][0, 0, 0, 0, 0])
        # 批次起始并发对所有策略均采集且有限
        assert np.isfinite(ex["exec_concurrent_mean"][0, 0, 0, 0, 0])
        assert np.isfinite(ex["exec_concurrent_max"][0, 0, 0, 0, 0])
    # QBA 有采集
    q = res["exec"]["QBA"]
    assert np.isfinite(q["exec_active_decode_mean"][0, 0, 0, 0, 0])
    assert np.isfinite(q["exec_active_decode_max"][0, 0, 0, 0, 0])


# ── 分片落盘 / 恢复 / 合并 ──────────────────────────────────

def _micro_run(cfg, ckpt_dir, seed, caps=(8192,), lengths=(120,)):
    return src._run_kv_capacity_sweep(
        cfg, np.asarray([1.0]), np.asarray([32.0]),
        np.asarray(list(caps)), np.asarray(list(lengths)), ["QBA"], [seed],
        n_windows=2, stage="short", ckpt_dir=ckpt_dir)


def test_checkpoint_atomic_write_and_resume(tmp_path, monkeypatch):
    """分片原子落盘（无 .tmp 残留）；resume 命中坐标一致则整 seed 跳过仿真。"""
    cfg = _cfg()
    ckpt_dir = str(tmp_path / "ckpt")
    res1 = _micro_run(cfg, ckpt_dir, 42)
    ckpt = src._seed_ckpt_path(ckpt_dir, "short", 42)
    assert os.path.exists(ckpt)
    assert not os.path.exists(ckpt + ".tmp.npz")  # 无残留临时文件

    def _boom(*a, **k):
        raise AssertionError("resume 未跳过仿真：simulator.run 仍被调用")

    monkeypatch.setattr(src.simulator, "run", _boom)
    res2 = src._run_kv_capacity_sweep(
        cfg, np.asarray([1.0]), np.asarray([32.0]),
        np.asarray([8192]), np.asarray([120]), ["QBA"], [42],
        n_windows=2, stage="short", ckpt_dir=ckpt_dir, resume=True)
    assert np.allclose(res1["rows"]["QBA"]["mean"], res2["rows"]["QBA"]["mean"],
                       equal_nan=True)
    assert np.allclose(res1["prefix"]["pi_s"], res2["prefix"]["pi_s"],
                       equal_nan=True)
    for mode in ("index", "time"):
        assert np.allclose(res1["window"]["QBA"][mode]["edges"],
                           res2["window"]["QBA"][mode]["edges"], equal_nan=True)


def test_resume_coord_mismatch_recomputes(tmp_path, monkeypatch):
    """分片坐标与本轮网格不一致时，resume 必须忽略分片并重算。"""
    cfg = _cfg()
    ckpt_dir = str(tmp_path / "ckpt")
    _micro_run(cfg, ckpt_dir, 42, caps=(8192,))   # 旧分片：K=8192
    calls = {"n": 0}
    real_run = simulator.run

    def _count(local, return_state=False):
        calls["n"] += 1
        return real_run(local, return_state=return_state)

    monkeypatch.setattr(src.simulator, "run", _count)
    src._run_kv_capacity_sweep(
        cfg, np.asarray([1.0]), np.asarray([32.0]),
        np.asarray([32768]), np.asarray([120]), ["QBA"], [42],
        n_windows=2, stage="short", ckpt_dir=ckpt_dir, resume=True)
    assert calls["n"] == 1  # 坐标不一致 → 重算


def test_merge_ckpt_disjoint_seeds(tmp_path):
    """合并不同 seed 的分片：seed 去重排序，数据沿首维堆叠。"""
    cfg = _cfg()
    d1, d2 = str(tmp_path / "a"), str(tmp_path / "b")
    _micro_run(cfg, d1, 42)
    _micro_run(cfg, d2, 43)
    p1 = src._load_npz(src._seed_ckpt_path(d1, "short", 42))
    p2 = src._load_npz(src._seed_ckpt_path(d2, "short", 43))
    merged = src._merge_ckpt_payloads([p1, p2])
    assert merged["seeds"].tolist() == [42, 43]
    assert merged["psfaq_mean_runs"].shape[0] == 2
    assert merged["capacities"].tolist() == [8192]


def test_merge_rejects_duplicate_seed(tmp_path):
    """相同 seed 的分片不能合并。"""
    cfg = _cfg()
    d1, d2 = str(tmp_path / "a"), str(tmp_path / "b")
    _micro_run(cfg, d1, 42)
    _micro_run(cfg, d2, 42)
    p1 = src._load_npz(src._seed_ckpt_path(d1, "short", 42))
    p2 = src._load_npz(src._seed_ckpt_path(d2, "short", 42))
    with pytest.raises(ValueError, match="重复 seed"):
        src._merge_ckpt_payloads([p1, p2])


def test_merge_rejects_coord_mismatch(tmp_path):
    """容量坐标不一致的分片不能合并。"""
    cfg = _cfg()
    d1, d2 = str(tmp_path / "a"), str(tmp_path / "b")
    _micro_run(cfg, d1, 42, caps=(8192,))
    _micro_run(cfg, d2, 43, caps=(32768,))
    p1 = src._load_npz(src._seed_ckpt_path(d1, "short", 42))
    p2 = src._load_npz(src._seed_ckpt_path(d2, "short", 43))
    with pytest.raises(ValueError, match="坐标不一致"):
        src._merge_ckpt_payloads([p1, p2])
