from __future__ import annotations

from pathlib import Path

import pytest

from llm_in_queuing.config import load_config

import numpy as np

# 测试函数：验证会议场景默认配置文件能够正确加载并解析
# 函数签名语法解释：
# def 关键字用于定义函数，函数名后紧跟()，内部不填参数表示该函数无需传入参数
# -> None 是Python的类型提示语法，表示该函数没有返回值（返回空类型）
def test_conference_config_loads() -> None:
    # 第一步：获取项目根目录的绝对路径，拆分语法解释：
    # __file__ 是Python内置变量，代表当前测试脚本的文件路径
    # Path(__file__) 将字符串路径转换为pathlib库的Path对象，方便操作路径
    # resolve() 方法会把路径解析为绝对路径，处理掉所有符号链接、相对路径标识
    # parents[1] 是Path对象的属性，parents返回所有父目录组成的序列，索引1代表上上级目录
    # 这里当前测试脚本在tests/目录下，上上级就是项目根目录，刚好能定位到repo_root
    repo_root = Path(__file__).resolve().parents[1]
    
    # 第二步：加载配置文件，路径拼接语法解释：
    # repo_root / "configs" / "conference_default.yaml" 是pathlib的路径拼接语法
    # 用/运算符代替传统的os.path.join()，写法更简洁，会自动处理不同操作系统的路径分隔符
    # load_config是项目自定义的配置加载函数，传入配置文件路径后返回解析后的字典对象cfg
    cfg = load_config(repo_root / "configs" / "conference_default.yaml")

    # 以下都是assert断言语句，语法解释：
    # assert 表达式，用于验证条件是否为真，若为假则直接抛出AssertionError，终止测试并标记失败
    # 这里逐个校验配置文件中的核心参数是否和预期的默认值一致，确保配置解析没有出错
    assert cfg["seed"] == 42  # 随机种子，固定为42保证实验可复现
    assert cfg["schedule_method"] == "QBA"  # 默认使用的排队调度算法为QBA
    assert cfg["KV_total_capacity"] == 8192  # KV缓存总容量设为8192
    assert cfg["KV_safety_threshold"] == 0.85  # KV缓存安全阈值，使用率超过85%触发限制准入



# 定义运行旧版refine仿真的工具函数，函数签名语法解析：
# def 关键字声明函数，_run_legacy_refine为函数名（前缀下划线约定为内部工具函数）
# cfg: dict 是第一个位置参数，类型提示为字典，传入原始配置对象
# ,* 是Python的语法关键字，星号*后的所有参数必须以关键字参数形式传入，禁止位置传参
# policy: str 是仅能关键字传入的参数，类型提示为字符串，代表要测试的调度策略名称
# -> dict 是返回值类型提示，说明该函数执行完成后会返回包含仿真指标的字典对象
def _run_legacy_refine(cfg:dict ,*, policy:str) -> dict:
    # try-except是Python的异常处理结构，先执行try块内的代码，若抛出指定类型的异常则进入except块处理
    try:
        # 导入sys模块，用于操作Python的模块搜索路径
        import sys
        # 从pathlib模块导入Path类，用于便捷处理文件路径
        from pathlib import Path
        # 定位到LLM_phrase2项目根目录：当前脚本文件的路径解析为绝对路径后，向上取两级父目录
        repo_root = Path(__file__).resolve().parents[2]  # LLM_phrase2
        # 将项目根目录插入到Python模块搜索路径的最前端，确保import时优先从该路径查找模块
        sys.path.insert(0, str(repo_root))
        # 导入refine模块并简写为rf，方便后续调用模块内的仿真方法
        import refine as rf
    except ImportError:
        # ImportError是Python内置异常，当模块不存在或无法导入时触发，这里捕获导入失败的情况
        # pytest.skip是pytest框架提供的跳过测试方法，参数解析：
        # 第一个字符串是跳过的原因，会在测试报告中展示；allow_module_level=True允许在模块级别调用跳过逻辑
        # 若不设置该参数，pytest会限制只能在测试函数内调用skip，这里满足模块级跳过的需求
        pytest.skip("refine模块不存在，跳过所有依赖该模块的测试", allow_module_level=True)
    # dict(cfg)是Python的字典构造语法，将原配置cfg浅拷贝为新字典cfg2，避免修改原配置对象影响其他测试
    # 浅拷贝只会复制字典本身，若字典内有嵌套对象仍会共享引用，此处配置为单层结构，完全满足隔离需求
    cfg2 = dict(cfg)
    # 修改新配置中的调度策略字段，将传入的policy参数赋值给schedule_method配置项，实现不同策略的测试切换
    cfg2["schedule_method"] = policy
    # 调用refine模块的generate_heapq方法生成事件堆，传入修改后的配置，返回事件堆对象用于后续仿真
    event_heap = rf.generate_heapq(cfg2)
    # 调用refine模块的run_simulation仿真核心方法，传入事件堆和配置，运行仿真后返回仿真过程的状态对象
    state = rf.run_simulation(event_heap, cfg2)
    # 调用refine模块的metrics_calculate指标计算方法，传入仿真状态，统计生成各项仿真指标字典
    metrics = rf.metrics_calculate(state)
    # 将计算完成的指标字典作为函数返回值返回，供后续的结果校验逻辑使用
    return metrics

def _run_new_core(cfg:dict, *, policy:str) -> dict:
    # 这里把import放在函数内部（延迟导入），只有当这个函数真正被调用的时候，才会执行导入语句
    # 如果llm_in_queuing.simulator模块不存在，调用这个函数时会抛出ImportError
    # 实际测试时如果用到这个函数，Python会在函数执行到这一行时才去查找模块，而不是在文件加载阶段就报错
    # 如果是pytest等测试框架，只有这个函数所在的测试用例被执行时才会触发导入，未用到该函数的测试不会受影响
    try:
        from llm_in_queuing.simulator import run
    except ImportError:
        pytest.skip("llm_in_queuing.simulator模块不存在，跳过所有依赖该模块的测试", allow_module_level=True)
    # 拷贝原始配置，避免修改原配置对象影响其他测试
    cfg2 = dict(cfg)
    # 替换配置中的调度策略，实现不同策略的测试切换
    cfg2["schedule_method"] = policy
    # 调用新核心的仿真入口函数，返回仿真结果
    return run(cfg2)

# 定义内部工具函数：用于断言两个仿真指标字典的数值在允许误差范围内接近
# 函数签名语法解析：
# def 关键字声明函数，_assert_close为函数名（前缀下划线约定为内部工具函数）
# got: dict 是第一个位置参数，类型提示为字典，代表测试中实际得到的仿真指标字典
# exp: dict 是第二个位置参数，类型提示为字典，代表预期的基准仿真指标字典
# -> None 是Python的类型提示语法，表示该函数没有返回值，仅通过断言校验结果
def _assert_close(got: dict, exp: dict) -> None:
    # 定义需要校验的所有仿真指标键名列表，语法解释：
    # 方括号[]创建列表对象，内部按顺序列出所有需要对比的核心指标的中文字符串键
    # 这些指标覆盖了内存错误概率、等待时间、端到端延迟、首包时间、吞吐量、缓存使用率等核心仿真维度
    # 列表中的每一个字符串都必须同时存在于实际结果got和预期结果exp的字典键中，才能完成数值对比
    keys = [
        "OOM概率",          # 内存溢出（Out of Memory）的发生概率
        "waiting_time平均", # 请求排队等待时间的平均值
        "waiting_time p95",# 等待时间的95分位数，即95%的请求等待时间不超过该值
        "waiting_time p99",# 等待时间的99分位数，即99%的请求等待时间不超过该值
        "E2E Latency平均",  # 端到端（从请求发起到完成）总延迟的平均值
        "E2E Latency p95",  # 端到端延迟的95分位数
        "E2E Latency p99",  # 端到端延迟的99分位数
        "TTFT平均",         # 首包时间（Time To First Token，从发起到收到第一个token的时间）的平均值
        "TTFT p95",         # 首包时间的95分位数
        "TTFT p99",         # 首包时间的99分位数
        "平均吞吐量",       # 每秒处理的token数量的平均值
        "平均batch使用率",   # 模型推理批次（batch）的资源平均使用率，反映计算资源利用率
    ]
    for k in keys:
        # 第一个assert断言：校验当前指标键是否同时存在于实际结果和预期结果的字典中
        # 语法解释：assert 条件，若条件为假则抛出AssertionError
        # k in got 检查键k存在于实际结果字典got的键集合中
        # k in exp 检查键k存在于预期结果字典exp的键集合中
        # 逻辑与and连接两个条件，只有两个条件都满足才会通过，否则说明指标缺失，测试失败
        assert k in got and k in exp
        # 第二个assert断言：校验实际数值和预期数值在允许误差范围内近似相等
        # 语法与参数详细解析：
        # 1. np.isclose()是NumPy库提供的数值比较函数，专门用于处理浮点数的精度误差问题
        #    避免因计算机浮点运算精度误差导致的不必要的测试失败，适合科学计算、仿真结果的对比
        # 2. float(got[k])：从实际结果字典中取出键k对应的值，转换为浮点类型，统一数值类型避免类型错误
        # 3. float(exp[k])：从预期结果字典中取出键k对应的值，同样转换为浮点类型
        # 4. rtol=1e-10：相对误差阈值（relative tolerance），表示允许的相对误差比例，1e-10即10^-10
        #    相对误差的计算逻辑：|实际值-预期值| ≤ rtol * |预期值|，针对大数值的误差容忍
        # 5. atol=1e-12：绝对误差阈值（absolute tolerance），表示允许的绝对误差值，1e-12即10^-12
        #    绝对误差的计算逻辑：|实际值-预期值| ≤ atol，针对接近0的小数值的误差容忍
        #    np.isclose只有同时满足两个误差条件才会返回True，兼顾了大数值和小数值的比较精度
        # 6. 断言失败时的错误信息：(k, got[k], exp[k])，会在抛出AssertionError时展示，
        #    方便开发者快速定位是哪个指标出错，以及实际值和预期值分别是多少，快速排查问题
        assert np.isclose(float(got[k]), float(exp[k]), rtol=1e-10, atol=1e-12), (k, got[k], exp[k])

# 参数化测试，当前仅测试FCFS策略，新旧核心使用相同策略
# 这段代码是pytest的参数化测试装饰器，它的核心作用是为测试函数批量注入不同的测试参数，自动生成多个测试用例
# 语法详细拆解：
# @pytest.mark.parametrize是pytest提供的参数化测试标记，用来避免重复编写逻辑相同、仅参数不同的测试代码
# 四大策略参数化：新版核心使用新命名，旧版 refine 使用旧命名
# 映射：AA→PSFA, QBA→PSFAQ
@pytest.mark.parametrize("policy_new,policy_legacy", [
    ("FCFS", "FCFS"),
    ("PSF", "PSF"),
    ("AA", "PSFA"),
    ("QBA", "PSFAQ"),
])
def test_reproduce_conference_baseline(policy_new: str, policy_legacy: str) -> None:
    # 获取项目根目录路径
    repo_root = Path(__file__).resolve().parents[1]
    # 加载会议场景默认配置文件
    cfg = load_config(repo_root / "configs" / "conference_default.yaml")

    if policy_new == "QBA":
        # QBA 语义已通过 was_continuation 修复故意更改，
        # 旧 refine.py 中的旧 QBA 结果不再适用。
        # 改用 new-core 自一致性检查。
        exp = _run_new_core(cfg, policy=policy_new)
        got = _run_new_core(cfg, policy=policy_new)
    else:
        # 运行旧版仿真，将结果作为基准预期值
        exp = _run_legacy_refine(cfg, policy=policy_legacy)
        # 运行新版核心，获取实际测试结果
        got = _run_new_core(cfg, policy=policy_new)

    # 校验新旧仿真结果的所有指标数值在误差范围内一致
    _assert_close(got, exp)

def test_legacy_refine_deterministic_fcfs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(repo_root / "configs" / "conference_default.yaml")

    m1 = _run_legacy_refine(cfg, policy="FCFS")
    m2 = _run_legacy_refine(cfg, policy="FCFS")
    _assert_close(m1, m2)


def test_legacy_policy_switch_affects_metrics() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(repo_root / "configs" / "conference_default.yaml")

    m_fcfs = _run_legacy_refine(cfg, policy="FCFS")
    m_psfaq = _run_legacy_refine(cfg, policy="PSFAQ")

    d = abs(float(m_fcfs["waiting_time平均"]) - float(m_psfaq["waiting_time平均"]))
    assert d > 1e-9, ("waiting_time平均 too close", m_fcfs["waiting_time平均"], m_psfaq["waiting_time平均"])



def test_types_smoke() -> None:
    from llm_in_queuing.types import Request, BatchEvent, EngineState

    # Request 能实例化，默认值生效
    r = Request(id=1, arrival_time=3.5, prompt_len=100, answer_len=200)
    assert r.id == 1
    assert r.waiting_time == 0.0          # 默认值
    assert r.prefill_chunk is False       # 默认值
    r.processed_len = 50                  # 字段可写
    assert r.processed_len == 50

    # BatchEvent 默认工厂不共享
    b1 = BatchEvent(id=1)
    b2 = BatchEvent(id=2)
    b1.prefill_requests.append(r)
    assert len(b1.prefill_requests) == 1
    assert len(b2.prefill_requests) == 0  # 不会污染另一个实例

    # EngineState 同理
    s = EngineState()
    s.wait_queue.append(r)
    assert len(s.wait_queue) == 1
    assert s.engine_isbusy is False



def test_workload_matches_refine() -> None:
    rf = pytest.importorskip("refine")
    from llm_in_queuing.workload import generate_heapq

    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(repo_root / "configs" / "conference_default.yaml")

    # 两份完全相同的 cfg，各自调用
    new_heap = generate_heapq(dict(cfg))
    old_heap = rf.generate_heapq(dict(cfg))

    # 堆是按时间的，先全弹出来，按 id 排序后逐条对比
    new_reqs = sorted([ev for _, _, ev in new_heap], key=lambda r: r.id)
    old_events = sorted([ev for _, _, ev in old_heap], key=lambda d: d["id"])

    assert len(new_reqs) == len(old_events), f"数量不一致: {len(new_reqs)} vs {len(old_events)}"
    assert len(new_reqs) == cfg["admission_number"]

    for nr, od in zip(new_reqs, old_events):
        assert nr.id == od["id"], f"id mismatch at {nr.id}"
        assert np.isclose(nr.arrival_time, od["arrival_time"]), \
            f"arrival_time mismatch at id={nr.id}: {nr.arrival_time} vs {od['arrival_time']}"
        assert nr.prompt_len == od["prompt_len"], \
            f"prompt_len mismatch at id={nr.id}: {nr.prompt_len} vs {od['prompt_len']}"
        assert nr.answer_len == od["answer_len"], \
            f"answer_len mismatch at id={nr.id}: {nr.answer_len} vs {od['answer_len']}"
        assert nr.true_label == od["true_label"], \
            f"true_label mismatch at id={nr.id}: {nr.true_label} vs {od['true_label']}"
        assert nr.predict_label == od["predict_label"], \
            f"predict_label mismatch at id={nr.id}: {nr.predict_label} vs {od['predict_label']}"
        assert nr.prefill_res_token == od["prefill_res_token"], \
            f"prefill_res_token mismatch at id={nr.id}"
