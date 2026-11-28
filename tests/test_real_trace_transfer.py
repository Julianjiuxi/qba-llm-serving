"""scaled real-trace replay：核心不变量测试（模块 A–E 适配层）。

只验证可快速检查的不变量，不重跑全量 sweep：
- 逐请求总 token 守恒与 a=1 恒等锚点；
- 整数化（向零截断）与上下限夹取；
- 预统计保留零间隔（不改用删零的旧口径）；
- 确定性方向性误判（掩码跨调用共享、方向正确）；
- 固定评价群体跨 a/K/policy 共享请求 ID；
- 模块 A–E 的形状、逐请求守恒、a=1 恒等；
- 分片原子写 / 恢复跳过 / 合并拒绝重复窗口与坐标不符；
- 模块 D 标签阈值在**完整区间**确定（避开旧 loader 按 n_take 重算的陷阱）。
"""

from __future__ import annotations

import json
import types
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llm_in_queuing.config import load_config  # noqa: E402
from scripts import real_trace_transfer as rt  # noqa: E402

_BASE_YAML = (_REPO_ROOT / "scripts" / "experiment" / "configs"
              / "real_trace_transfer_base.yaml")


def _cfg() -> dict:
    cfg = load_config(_BASE_YAML)
    cfg["mech_diag_enabled"] = True
    return cfg


def _ns(**kw) -> types.SimpleNamespace:
    base = dict(err_seed=7, ckpt_dir=None, resume=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _mk_window(wid: int = 0, n: int = 160, seed: int = 0, source_key: str = ""):
    """构造一个小型 ReplayWindow（真实到来形状的模拟：递增时刻 + 随机长度）。"""
    rng = np.random.default_rng(seed)
    ts = np.cumsum(rng.random(n) * 0.1)
    ct = rng.integers(100, 3000, n).astype(float)
    gt = rng.integers(1, 500, n).astype(float)
    return rt._make_window(wid, ts, ct, gt, _cfg(), source_key=source_key)


def _win_with_rate(wid: int, n: int, span: float, seed: int, source_key: str = ""):
    """构造指定到达率（≈ n / span，等请求数、不同跨度）的窗口，用于模块 E 强度差异测试。"""
    ts = np.linspace(0.0, float(span), int(n))
    rng = np.random.default_rng(seed)
    ct = rng.integers(100, 3000, int(n)).astype(float)
    gt = rng.integers(1, 500, int(n)).astype(float)
    return rt._make_window(wid, ts, ct, gt, _cfg(), source_key=source_key)


# ── 构成变换：守恒 + a=1 恒等 ──────────────────────────────

def test_compose_lengths_conservation_and_identity():
    cfg = _cfg()
    win = _mk_window()
    p1, d1 = rt._compose_lengths(win.p_base, win.d_base, 1.0, cfg)
    assert np.array_equal(p1, win.p_base) and np.array_equal(d1, win.d_base)
    base_total = win.p_base + win.d_base
    for a in (0.0625, 0.25, 4.0):
        p, d = rt._compose_lengths(win.p_base, win.d_base, a, cfg)
        assert np.all(p + d == base_total)          # 逐请求总 token 守恒
        assert np.all(p >= 1) and np.all(d >= 1)    # 合法下界


# ── 整数化：向零截断 + 上下限夹取 ──────────────────────────

def test_scale_tokens_truncates_and_clamps():
    cfg = {"token_scale": 0.5, "prompt_len_cap": 1000}
    v = rt._scale_tokens(np.array([1.0, 3.9, 4000.0]), cfg, 5, "prompt_len_cap")
    # 0.5→0→夹到下限 5；1.95→1→夹到下限 5；2000→夹到上限 1000（向零截断，不四舍五入）
    assert v.tolist() == [5, 5, 1000]
    assert v.dtype == np.int64


# ── 预统计：保留零间隔 ────────────────────────────────────

def test_pre_stats_keeps_zero_gaps():
    cfg = _cfg()
    ts = np.array([0.0, 1.0, 1.0, 3.0, 4.0])       # 含一个同时到达（零间隔）
    win = rt._make_window(0, ts, np.full(5, 100.0), np.full(5, 10.0), cfg)
    stats = rt._compute_pre_stats(win, cfg)
    inter = np.diff(win.arrivals)                   # [1,0,2,1]
    assert np.any(inter == 0.0)                     # 零间隔确实存在
    mu = inter.mean()
    var = inter.var(ddof=1)
    assert stats["inter_arrival_Ca2"] == pytest.approx(var / mu ** 2)
    # 若误删零间隔，均值会变成 4/3；这里应为 4/4 = 1.0
    assert mu == pytest.approx(1.0)


# ── 方向性误判：确定 + 方向正确 + 掩码共享 ─────────────────

def test_label_stream_oracle_and_directions():
    d = np.arange(1.0, 21.0)
    thr = 10.0
    lab, pred = rt._label_stream(d, thr, err=None, window_id=0, err_seed=1)
    assert np.array_equal(lab, pred)
    assert set(lab.tolist()) == {"short", "long"}

    lab2, pred2 = rt._label_stream(d, thr, err=("SL", 1.0), window_id=0, err_seed=1)
    assert np.all(pred2[lab2 == "short"] == "long")   # 单方向：short→long
    assert np.all(pred2[lab2 == "long"] == "long")

    lab3, pred3 = rt._label_stream(d, thr, err=("LS", 1.0), window_id=0, err_seed=1)
    assert np.all(pred3[lab3 == "long"] == "short")   # 单方向：long→short
    assert np.all(pred3[lab3 == "short"] == "short")


def test_label_stream_mask_shared_and_deterministic():
    d = np.arange(1.0, 41.0)
    thr = 20.0
    a = rt._label_stream(d, thr, err=("SL", 0.3), window_id=3, err_seed=9)[1]
    b = rt._label_stream(d, thr, err=("SL", 0.3), window_id=3, err_seed=9)[1]
    assert np.array_equal(a, b)                       # 同 (seed, 窗口, 方向) 掩码一致
    c = rt._label_stream(d, thr, err=("SL", 0.3), window_id=4, err_seed=9)[1]
    assert not np.array_equal(a, c)                   # 不同窗口掩码不同


# ── 固定评价群体：基线掩码存在且比例合理 ──────────────────

def test_fixed_group_masks_fixed_by_baseline():
    win = _mk_window(n=400, seed=3)
    assert 0.05 <= float(win.fixed_ans_mask.mean()) <= 0.25
    assert 0.01 <= float(win.fixed_tot_mask.mean()) <= 0.12


# ── 模块 A：形状 / 守恒 / a=1 恒等 / 固定群体按策略 ─────────

def test_module_a_shapes_and_anchors():
    cfg = _cfg()
    wins = [_mk_window(0, 120, 0), _mk_window(1, 120, 1)]
    scales = np.asarray([0.25, 1.0])
    caps = np.asarray([8192, 16384])
    loads = np.asarray([32.0])
    arrays = rt._run_module_a(cfg, wins, scales, caps, loads,
                              rt._ALL_POLICIES, 3, _ns())
    # 第 0 维是窗口维
    assert arrays["psfaq_mean"].shape == (2, 2, 2, 1)
    assert arrays["transform_max_request_total_error"].shape == (2, 2)
    assert np.all(arrays["transform_max_request_total_error"] == 0.0)
    # a=1（index 1）恒等锚点
    assert np.allclose(arrays["transform_prompt_mean"][:, 1],
                       arrays["pre_scaled_E_P"])
    assert np.allclose(arrays["transform_answer_mean"][:, 1],
                       arrays["pre_scaled_E_D"])
    # 固定评价群体按策略分别记录
    for pfx in ("fcfs", "psf", "psfa", "psfaq"):
        assert f"{pfx}_fixed_fx_tot_long_wait_p99" in arrays
        assert f"{pfx}_fixed_fx_ans_long_n" in arrays


# ── 模块 B：p×k 形状 + χ_pred 与固定群体 ───────────────────

def test_module_b_shapes():
    cfg = _cfg()
    wins = [_mk_window(0, 120, 0)]
    caps = np.asarray([8192, 16384])
    ps = np.asarray([0.80, 0.85])
    ks = np.asarray([3, 5])
    arrays = rt._run_module_b(cfg, wins, caps, ps, ks, 32.0,
                              rt._ALL_POLICIES, 2, _ns())
    assert arrays["psfaq_mean"].shape == (1, 2, 2, 2)   # [n_w, n_K, n_p, n_k]
    assert arrays["psf_mean"].shape == (1, 2, 2)
    assert arrays["fcfs_mean"].shape == (1, 2)
    assert arrays["b_chi_pred"].shape == (1, 2, 2)
    assert arrays["psfaq_fixed_fx_tot_long_wait_p99"].shape == (1, 2, 2, 2)
    # FCFS true-long 分类指标随 p 变化（[n_w, n_K, n_p]）
    assert arrays["fcfs_tl_group_wait_p99"].shape == (1, 2, 2)
    assert arrays["fcfs_tl_group_n"].shape == (1, 2, 2)


# ── 模块 C：四群体形状 + 方向语义 ──────────────────────────

def test_module_c_quadrants_direction():
    cfg = _cfg()
    wins = [_mk_window(0, 200, 0)]
    caps = np.asarray([8192])
    arrays = rt._run_module_c(cfg, wins, caps, rt._ERR_CONFIGS, 32.0, 0.85, 5,
                              rt._ALL_POLICIES, 2, _ns())
    quad = arrays["quad_psfaq"]                     # [n_w, n_K, n_err, 4, 7]
    assert quad.shape == (1, 1, len(rt._ERR_CONFIGS), 4, len(rt._QUAD_STATS))
    assert arrays["psfaq_fixed_fx_tot_long_wait_p99"].shape == (1, 1, len(rt._ERR_CONFIGS))
    # err 顺序：oracle, SL0.1, SL0.3, LS0.1, LS0.3
    # oracle：sl / ls 两群体应为 0（无跨类别）
    assert quad[0, 0, 0, 1, 0] == 0.0               # short→long
    assert quad[0, 0, 0, 2, 0] == 0.0               # long→short
    # SL0.3：short→long 人数 > 0，long→short 仍为 0
    assert quad[0, 0, 2, 1, 0] > 0.0
    assert quad[0, 0, 2, 2, 0] == 0.0
    # LS0.3：long→short 人数 > 0，short→long 仍为 0
    assert quad[0, 0, 4, 2, 0] > 0.0
    assert quad[0, 0, 4, 1, 0] == 0.0


# ── 模块 D：形状 + 标签阈值在完整区间确定（避开 loader 陷阱）─

def test_module_d_prefix_uses_full_interval_threshold(monkeypatch):
    cfg = _cfg()
    wins = [_mk_window(0, 150, 0)]
    seen: list = []
    real = rt._label_stream

    def _cap(d_new, thr, *, err, window_id, err_seed):
        seen.append((int(d_new.shape[0]), float(thr)))
        return real(d_new, thr, err=err, window_id=window_id, err_seed=err_seed)

    monkeypatch.setattr(rt, "_label_stream", _cap)
    arrays = rt._run_module_d(cfg, wins, np.asarray([60, 150]), np.asarray([1.0]),
                              np.asarray([8192]), np.asarray([32.0]), ["QBA"], 2, _ns())
    # 每个 (window, a) 只调用一次标签，且用**完整** 150 条请求确定阈值
    assert seen and all(n == 150 for n, _ in seen)
    assert arrays["psfaq_mean"].shape == (1, 2, 1, 1, 1)   # [n_w, n_N, n_a, n_K, n_l]
    # D 亦保存固定评价群体（完整参考窗口确定掩码，按前缀 ID 统计）
    assert arrays["psfaq_fixed_fx_tot_long_wait_p99"].shape == (1, 2, 1, 1, 1)
    # 上下界比例诊断须被填充（不得恒为 NaN）
    assert np.any(np.isfinite(arrays["transform_lower_bound_fraction"]))
    assert np.any(np.isfinite(arrays["transform_upper_bound_fraction"]))


# ── 模块 E：统一时间缩放（逐窗口 λ_eff）────────────────────

def test_module_e_shared_time_scale():
    cfg = _cfg()
    wins = [_mk_window(0, 120, 0), _mk_window(1, 120, 1)]
    caps = np.asarray([8192, 32768])
    ct = 1.411
    arrays = rt._run_module_e(cfg, wins, caps, ct, rt._ALL_POLICIES, 2, _ns())
    assert arrays["psfaq_mean"].shape == (2, 2)          # [n_w, n_K]
    # E 亦保存固定评价群体
    assert arrays["psfaq_fixed_fx_tot_long_wait_p99"].shape == (2, 2)
    e = arrays["e_lambda_eff"]
    assert e.shape == (2,)
    # λ_eff = raw_lambda / c_t
    assert np.allclose(e, np.asarray([w.raw_lambda / ct for w in wins]))


# ── 分片：原子写 / 恢复跳过 / 合并拒绝 ──────────────────────

def _micro_a(cfg, ckpt_dir, caps=(8192,), scales=(1.0,), resume=False, err_seed=7):
    wins = [_mk_window(0, 100, 0), _mk_window(1, 100, 1)]
    return rt._run_module_a(
        cfg, wins, np.asarray(list(scales)), np.asarray(list(caps)),
        np.asarray([32.0]), ["QBA"], 2,
        _ns(ckpt_dir=ckpt_dir, resume=resume, err_seed=err_seed))


def test_ckpt_atomic_write_and_resume(tmp_path, monkeypatch):
    cfg = _cfg()
    d = str(tmp_path / "ckpt")
    a1 = _micro_a(cfg, d)
    for wi in (0, 1):
        p = rt._win_ckpt_path(d, "A", wi)
        assert Path(p).exists()
        assert not Path(p + ".tmp.npz").exists()        # 无临时残留

    def _boom(*a, **k):
        raise AssertionError("resume 未跳过：simulator.run 仍被调用")

    monkeypatch.setattr(rt.simulator, "run", _boom)
    a2 = _micro_a(cfg, d, resume=True)
    assert np.allclose(a1["psfaq_mean"], a2["psfaq_mean"], equal_nan=True)


def test_merge_rejects_duplicate_window(tmp_path):
    cfg = _cfg()
    d1, d2 = str(tmp_path / "a"), str(tmp_path / "b")
    _micro_a(cfg, d1)
    _micro_a(cfg, d2)
    with pytest.raises(ValueError, match="重复窗口编号"):
        rt._merge_module("A", [d1, d2], {"composition_scales": np.asarray([1.0])})


def test_merge_rejects_coord_mismatch(tmp_path):
    cfg = _cfg()
    d1, d2 = str(tmp_path / "a"), str(tmp_path / "b")
    _micro_a(cfg, d1, caps=(8192,))
    _micro_a(cfg, d2, caps=(16384,))
    # 只取部分窗口以构造坐标不一致：合并两个单窗口分片（窗口编号相同）会先触发重复，
    # 因此这里直接比较坐标逻辑入口。
    p1 = rt._load_npz(rt._win_ckpt_path(d1, "A", 0))
    p2 = rt._load_npz(rt._win_ckpt_path(d2, "A", 0))
    coord = {k[len("coord_"):]: v for k, v in p1.items()
             if k.startswith("coord_") and k != "coord_window_index"}
    assert not rt._coords_match(p2, coord)


def test_merge_module_ok_two_windows(tmp_path):
    """两个不同窗口编号的分片可合并，窗口按编号排序；返回 (arrays, sig, meta)。"""
    cfg = _cfg()
    wins = [_mk_window(0, 100, 0), _mk_window(1, 100, 1)]
    arrays = rt._run_module_a(cfg, wins, np.asarray([1.0]), np.asarray([8192]),
                              np.asarray([32.0]), ["QBA"], 2, _ns(ckpt_dir=str(tmp_path / "c")))
    merged, sig, meta = rt._merge_module("A", [str(tmp_path / "c")], {
        "composition_scales": np.asarray([1.0]),
        "capacities": np.asarray([8192]),
        "lambdas": np.asarray([32.0]),
    })
    assert isinstance(sig, dict) and "err_seed" in sig
    assert isinstance(meta, dict) and meta.get("experiment") == "scaled_real_trace_replay"
    assert merged["windows"].tolist() == [0, 1]
    assert np.allclose(merged["psfaq_mean"], arrays["psfaq_mean"], equal_nan=True)


def test_merge_rejects_cli_axis_mismatch(tmp_path):
    """CLI 给出的轴与 checkpoint 已验证坐标不符时必须拒绝合并（不得沿用 CLI 标注）。"""
    cfg = _cfg()
    d = str(tmp_path / "c")
    _micro_a(cfg, d, caps=(8192,))
    with pytest.raises(ValueError, match="合并轴与分片坐标不符"):
        rt._merge_module("A", [d], {"capacities": np.asarray([32768])})


def test_merge_round_trip_err_names_readable(tmp_path):
    """模块 C 合并后 err_names 必须是可 allow_pickle=False 读取的字符串数组。"""
    cfg = _cfg()
    d = str(tmp_path / "c")
    wins = [_mk_window(0, 120, 0), _mk_window(1, 120, 1)]
    rt._run_module_c(cfg, wins, np.asarray([8192]), rt._ERR_CONFIGS, 32.0, 0.85, 5,
                     ["QBA"], 2, _ns(ckpt_dir=d))
    merged, _sig, _meta = rt._merge_module("C", [d], None)
    assert merged["err_names"].dtype.kind in "US"       # 字符串数组，非 object
    out = tmp_path / "rt_C_data.npz"
    np.savez(out, **merged)
    with np.load(out, allow_pickle=False) as z:          # 必须能无 pickle 读取
        names = [str(x) for x in z["err_names"].reshape(-1)]
    assert names == [e[0] for e in rt._ERR_CONFIGS]
    # _axis_stub 亦须返回字符串数组（不得退化为 object）
    stub = rt._axis_stub(cfg, _ns(long_caps="8192,32768"), "C")
    assert stub["err_names"].dtype.kind in "US"


def test_resume_rejects_on_param_change(tmp_path, monkeypatch):
    """运行签名绑定 err_seed 等参数：参数改变后 --resume 必须拒绝复用旧分片并重算。"""
    cfg = _cfg()
    d = str(tmp_path / "ckpt")
    _micro_a(cfg, d, err_seed=1)
    calls = {"n": 0}
    real_run = rt.simulator.run

    def _count(*a, **k):
        calls["n"] += 1
        return real_run(*a, **k)

    monkeypatch.setattr(rt.simulator, "run", _count)
    _micro_a(cfg, d, err_seed=2, resume=True)
    assert calls["n"] > 0                                # 未复用，重新仿真


def test_resume_rejects_on_input_change(tmp_path, monkeypatch):
    """单窗口**输入身份**绑定输入：换输入（source_key/内容）后 --resume 必须拒绝复用。"""
    cfg = _cfg()
    d = str(tmp_path / "ckpt")

    def _mk(src):
        return [_mk_window(0, 100, 0, source_key=src),
                _mk_window(1, 100, 1, source_key=src)]

    def _run(wins, resume=False):
        return rt._run_module_a(cfg, wins, np.asarray([1.0]), np.asarray([8192]),
                                np.asarray([32.0]), ["QBA"], 2,
                                _ns(ckpt_dir=d, resume=resume, err_seed=7))

    # 公共签名只绑定 partition（内容无关的全局网格），单窗口身份另行随分片保存
    sig = rt._run_signature(cfg, _ns(err_seed=7), "A",
                            {"source": "subset", "glob": "*.csv",
                             "sources": ["subset:a.csv", "subset:b.csv"]})
    assert sig["partition"]["sources"] == ["subset:a.csv", "subset:b.csv"]

    _run(_mk("subset:a.csv"))
    calls = {"n": 0}
    real_run = rt.simulator.run

    def _count(*a, **k):
        calls["n"] += 1
        return real_run(*a, **k)

    monkeypatch.setattr(rt.simulator, "run", _count)
    _run(_mk("subset:b.csv"), resume=True)               # 换输入身份
    assert calls["n"] > 0                                # 未复用，重新仿真


def test_cfg_fingerprint_covers_full_effective_config():
    """指纹须覆盖旧签名遗漏的字段（预算 / 并发 / aging / batch-time / 稳态）。"""
    fp = rt._cfg_fingerprint(_cfg())
    for key in ("budget_tokens", "request_block_size", "ageing_measure",
                "BT_BASE_C", "BT_SLOPE_A", "BT_KNEE_B0", "BT_CONC_GAMMA",
                "steady_state", "KV_total_capacity", "KV_safety_threshold"):
        assert key in fp, f"配置指纹遗漏字段: {key}"


def test_resume_rejects_on_budget_change(tmp_path, monkeypatch):
    """改 budget_tokens 后 --resume 必须拒绝复用旧分片（旧签名漏列该参数）。"""
    cfg = _cfg()
    d = str(tmp_path / "ckpt")
    c128, c64 = dict(cfg), dict(cfg)
    c128["budget_tokens"], c64["budget_tokens"] = 128, 64
    assert rt._cfg_fingerprint(c128) != rt._cfg_fingerprint(c64)
    _micro_a(c128, d)
    calls = {"n": 0}
    real_run = rt.simulator.run

    def _count(*a, **k):
        calls["n"] += 1
        return real_run(*a, **k)

    monkeypatch.setattr(rt.simulator, "run", _count)
    _micro_a(c64, d, resume=True)
    assert calls["n"] > 0                                # 未复用，重新仿真


def test_resume_rejects_on_stat_windows_change(tmp_path, monkeypatch):
    """统计分窗数 w（来自 CLI，不在 cfg 中）变化后 --resume 必须拒绝复用旧分片。"""
    cfg = _cfg()
    d = str(tmp_path / "ckpt")
    wins = [_mk_window(0, 100, 0), _mk_window(1, 100, 1)]

    def _run(w, resume=False):
        return rt._run_module_a(cfg, wins, np.asarray([1.0]), np.asarray([8192]),
                                np.asarray([32.0]), ["QBA"], w,
                                _ns(ckpt_dir=d, resume=resume, err_seed=7))

    sig = rt._run_signature(cfg, _ns(err_seed=7), "A", None, w=2)
    assert sig["stat_windows"] == 2
    _run(2)
    calls = {"n": 0}
    real_run = rt.simulator.run

    def _count(*a, **k):
        calls["n"] += 1
        return real_run(*a, **k)

    monkeypatch.setattr(rt.simulator, "run", _count)
    _run(3, resume=True)                                 # w: 2 → 3
    assert calls["n"] > 0                                # 未复用，重新仿真


def test_merge_config_snapshot_from_original_signature(tmp_path, monkeypatch):
    """合并输出 meta 的配置快照须来自**原始运行**（_sig['config']），而非合并时 CLI。

    复现：实验 gamma=0.6、合并用默认 gamma=0.8 时，同一 JSON 内 args 与
    cfg_sim_snapshot 必须一致（均为 0.6），并保留完整原始签名。
    """
    cfg = _cfg()                                         # 基线 gamma=0.80
    cfg06 = dict(cfg)
    cfg06["gamma"] = 0.6
    d = str(tmp_path / "ckpt")
    partition = {"source": "subset", "glob": "*.csv",
                 "sources": ["subset:w0.csv", "subset:w1.csv"]}
    wins = [_mk_window(0, 100, 0, source_key="subset:w0.csv"),
            _mk_window(1, 100, 1, source_key="subset:w1.csv")]
    rt._run_module_a(cfg06, wins, np.asarray([1.0]), np.asarray([8192]),
                     np.asarray([32.0]), ["QBA"], 2,
                     _ns(err_seed=7, gamma=0.6, ckpt_dir=d), partition)

    out_dir = str(tmp_path / "merged")
    monkeypatch.setattr(sys, "argv", [
        "prog", "--module", "A", "--merge-dirs", d, "--out-dir", out_dir,
        "--scales", "1", "--caps", "8192", "--lambdas", "32"])
    rt.main()
    meta = json.loads(Path(out_dir, "rt_A_meta.json").read_text(encoding="utf-8"))
    assert meta["args"]["gamma"] == 0.6
    assert meta["cfg_sim_snapshot"]["gamma"] == 0.6      # 非合并默认 0.8
    assert meta["run_signature"]["config"]["gamma"] == 0.6
    assert meta["run_signature"]["err_seed"] == 7


# ── 两进程分片 → 合并 ↔ 串行（用户实测缺陷的回归）────────────

def test_two_process_shards_merge_equals_serial(tmp_path):
    """两个独立进程分别处理全局窗口 0 / 1 → 合并结果须与串行全量一致。

    覆盖：分片文件名/窗口编号必须用全局 ``win.wid``（否则两进程都写 win0000 互相覆盖）；
    公共签名不得包含"本进程窗口身份"（否则处理不同窗口的合法 worker 被误拒）。
    """
    cfg = _cfg()
    d = str(tmp_path / "ckpt")
    partition = {"source": "subset",
                 "glob": "Azure_work_load_subset[0-9][0-9][0-9].csv",
                 "sources": ["subset:w000.csv", "subset:w001.csv"]}
    win0 = _mk_window(0, 100, 0, source_key="subset:w000.csv")
    win1 = _mk_window(1, 100, 1, source_key="subset:w001.csv")

    def _run_one(win):
        return rt._run_module_a(cfg, [win], np.asarray([1.0]), np.asarray([8192]),
                                np.asarray([32.0]), ["QBA"], 2, _ns(ckpt_dir=d), partition)

    # 进程 1 只加载窗口 0，进程 2 只加载窗口 1，共用同一 checkpoint 目录
    _run_one(win0)
    _run_one(win1)
    assert Path(rt._win_ckpt_path(d, "A", 0)).exists()
    assert Path(rt._win_ckpt_path(d, "A", 1)).exists()   # 不得互相覆盖成 win0000

    # 串行参考：一次调用内处理两个窗口
    serial = rt._run_module_a(cfg, [win0, win1], np.asarray([1.0]), np.asarray([8192]),
                              np.asarray([32.0]), ["QBA"], 2, _ns(), partition)

    merged, sig, _meta = rt._merge_module("A", [d], None)
    assert isinstance(sig, dict) and sig["partition"]["sources"] == partition["sources"]
    assert merged["windows"].tolist() == [0, 1]
    assert np.allclose(merged["psfaq_mean"], serial["psfaq_mean"], equal_nan=True)


def test_merge_rejects_input_identity_mismatch(tmp_path):
    """分片窗口的来源与公共网格不符时必须拒绝合并（防止拼错窗口）。"""
    cfg = _cfg()
    d = str(tmp_path / "ckpt")
    partition = {"source": "subset", "glob": "*.csv",
                 "sources": ["subset:w000.csv", "subset:w001.csv"]}
    # 窗口 0 的来源与网格 sources[0] 不符
    bad = _mk_window(0, 100, 0, source_key="subset:wrong.csv")
    rt._run_module_a(cfg, [bad], np.asarray([1.0]), np.asarray([8192]),
                     np.asarray([32.0]), ["QBA"], 2, _ns(ckpt_dir=d), partition)
    with pytest.raises(ValueError, match="输入身份与全局网格不符"):
        rt._merge_module("A", [d], None)


# ── 模块 E：时间缩放不随分片方式改变 ───────────────────────

def test_grid_pooled_rate_over_full_grid():
    """全局 pooled rate = Σn_i / Σspan_i（全体 grid 窗口，与分片无关）。"""
    ts = np.arange(1000.0)
    # 两个窗口各 100 条，跨度均为 ts[s+99]-ts[s] = 99
    r = rt._grid_pooled_rate(ts, [0, 100], 100)
    assert r == pytest.approx(200.0 / (99.0 + 99.0))


def test_module_e_time_scale_independent_of_sharding():
    """E 的统一时间缩放基于**全体窗口**预先确定，保留窗口间强度差异（不逐窗口归一化）。

    两个等请求数窗口原始到达率 20 / 40；全局 pooled rate = 200/7.5 = 26.667，
    shared_c_t = 26.667/32 → 缩放后 λ_eff = 24 / 48（差异保留）。
    修复前每个 worker 用自身窗口子集重算 → 两窗口都被归一化到 λ=32。
    """
    cfg = _cfg()
    win0 = _win_with_rate(0, 100, 5.0, 0, source_key="w0")
    win1 = _win_with_rate(1, 100, 2.5, 1, source_key="w1")
    assert win0.raw_lambda == pytest.approx(20.0)
    assert win1.raw_lambda == pytest.approx(40.0)

    grid_rate = 200.0 / (5.0 + 2.5)                      # 全体窗口 pooled rate
    shared_ct = grid_rate / 32.0

    pair = rt._run_module_e(cfg, [win0, win1], np.asarray([8192]), shared_ct,
                            ["QBA"], 2, _ns())
    e = pair["e_lambda_eff"]
    assert np.allclose(e, np.asarray([20.0, 40.0]) / shared_ct)
    assert not np.isclose(e[0], e[1])                    # 强度差异被保留

    # 只加载窗口 0 的 worker（用同一 shared_ct）→ 窗口 0 的时间轴/结果不变
    single = rt._run_module_e(cfg, [win0], np.asarray([8192]), shared_ct,
                              ["QBA"], 2, _ns())
    assert np.allclose(single["e_lambda_eff"], e[:1])
    assert np.allclose(single["psfaq_mean"][0], pair["psfaq_mean"][0], equal_nan=True)


# ── 真实整周 CSV 表头：TIMESTAMP 解析 + 同时到达保留 ────────

def test_week_header_timestamp_parse_and_window():
    import pandas as pd
    df = pd.DataFrame({
        "TIMESTAMP": ["2024-05-12 00:00:00.001163+00:00",
                      "2024-05-12 00:00:00.001163+00:00",   # 同时到达
                      "2024-05-12 00:00:01.500000+00:00"],
        "ContextTokens": [1000, 2000, 3000],
        "GeneratedTokens": [10, 20, 30],
    })
    t = rt._time_seconds(df)
    assert t.shape == (3,)
    assert t[0] == 0.0
    assert t[1] == 0.0                                    # 零间隔保留
    # 相对首行归零：(1.500000 − 0.001163) 秒（float64 epoch 精度 ~1e-6）
    assert t[2] == pytest.approx(1.5 - 0.001163, abs=1e-6)
    # 绝对秒解析亦正确（首行小数为 0.001163）
    t_abs = rt._parse_timestamp_seconds(df["TIMESTAMP"])
    assert t_abs[2] - t_abs[0] == pytest.approx(1.5 - 0.001163, abs=1e-6)
    win = rt._make_window(0, t, df["ContextTokens"].to_numpy(np.float64),
                          df["GeneratedTokens"].to_numpy(np.float64), _cfg())
    assert win.arrivals.shape == (3,)
    # 时间非递减校验
    with pytest.raises(ValueError):
        rt._make_window(0, np.array([2.0, 1.0, 3.0]), np.array([1.0, 1.0, 1.0]),
                        np.array([1.0, 1.0, 1.0]), _cfg())


# ── 解析器 / 分层抽样 ──────────────────────────────────────

def test_parse_ps_validates():
    assert rt._parse_ps("0.85, 0.70").tolist() == [0.70, 0.85]
    for bad in ("", "0", "1", "1.2", "0.8,0.8", "abc"):
        with pytest.raises(ValueError):
            rt._parse_ps(bad)


def test_week_long_starts_are_disjoint():
    ts = np.arange(1000.0)
    starts = rt._week_long_starts(ts, 5, 100)
    assert len(starts) == 5
    for a, b in zip(starts, starts[1:]):
        assert a + 100 <= b                               # 区间互不重叠
    assert starts[-1] + 100 <= int(ts.size)
    # 边界：n_full*n_windows == n_total 仍不得重叠
    s2 = rt._week_long_starts(np.arange(500.0), 5, 100)
    for a, b in zip(s2, s2[1:]):
        assert a + 100 <= b
    # 容量不足应报错
    with pytest.raises(ValueError):
        rt._week_long_starts(np.arange(100.0), 5, 100)
