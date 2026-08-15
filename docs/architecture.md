
# 项目架构（初步设计）

本文件描述 `LLM_in_Queuing` 的初步架构设计：模块边界、数据流、核心不变量与扩展点。目标是在不牺牲科研迭代速度的前提下，保证语义口径可追溯、实验可复现、策略可插拔。

## 1. 设计目标

- 语义冻结：batch 语义、KV accounting、统计口径（尤其是稳态窗口与 OOM 口径）必须可追溯
- 责任边界：engine / scheduler / admission / predictor / metrics 解耦；策略不侵入引擎语义
- 可插拔：通过配置切换 predictor / admission / scheduler / metrics，而不是改主循环
- 可复现：每个实验点保存 config、seed、workload summary、metrics schema、产物索引
- 可扩展：支持 uncertainty-aware、multi-tenant QoS、risk-aware admission、分析层 surrogate 与 trace calibration

## 2. 系统抽象（连续 batching 的最小语义）

仿真器抽象一个 LLM serving 系统：

- 以离散轮次推进，每轮有 token 预算 `budget_tokens`
- decode 优先：优先服务正在 decode 的活跃请求，然后用剩余预算做 prefill
- 单请求单轮 decode 推进受 `request_block_size` 限制（block）
- prefill 允许分段（chunk / continuation），未完成的 prefill 会回到等待队列前部
- KV 记账：
  - prefill 每写入 1 token，KV 增 1
  - decode 每生成 1 token，KV 增 1
  - 请求完成时释放 `prompt_len + answer_len`
  - OOM 以“释放前批内峰值 kv_peak”判定

上述语义是论文/实验可比性的基础，任何偏离必须写入 docs（spec/ADR）并配套回归点。

## 3. 模块边界（12 模块骨架）

目录以 `src/llm_in_queuing/` 为核心包，模块职责如下。

### 3.1 config（配置与口径入口）

- `config.py`：加载/合并 YAML 配置，构造运行期 Config 对象（包含版本号/默认值/覆盖规则）
- `configs/`：保存基线配置与实验 sweep 配置（fig-level）

关键约束：配置是唯一口径来源，代码中不再硬编码论文参数。

### 3.2 types（数据模型与 schema）

- `types.py`：定义 Request / BatchPlan / BatchEvent / MetricsRecord 等数据模型（dataclass 或 TypedDict）

关键约束：字段语义必须稳定；新增字段（tenant、SLA、uncertainty）以向后兼容方式扩展。

### 3.3 workload（负载生成）

- `workload.py`：arrival 过程、prompt/answer 分布、截断、workload shift、multi-tenant arrival

关键约束：workload 必须可复现（seed 驱动），并输出 workload summary（便于复现实验与对照）。

### 3.4 predictor（预测接口）

- `predictor.py`：预测器统一接口与实现族
  - binary label、噪声标签
  - length quantile / posterior（面向 KV 风险）
  - overflow/OOM probability（面向风险准入）

关键约束：scheduler/admission 只能使用 predictor 输出，不能读取真值字段参与决策（评价仍用真值）。

### 3.5 state（系统状态与队列）

- `state.py`：EngineState（系统时钟、队列、KV、轨迹）与队列维护工具

关键约束：state 的写权限要收敛：engine 负责执行语义更新；simulator 负责事件推进；metrics 只读；scheduler/admission 只读或返回决策，不直接改引擎关键状态。

### 3.6 engine（执行语义核心）

- `engine.py`：执行一个 batch 的语义：prefill/decode 推进、continuation、首 token/TTFT、完成释放、kv_peak 计算、batch_time 模型

关键约束：engine 不包含任何策略分支；所有策略差异体现在 scheduler/admission。

### 3.7 admission（准入控制）

- `admission.py`：准入策略（KV-safe、projected KV、risk-aware、class cap）

建议的接口形态：

- 输入：当前 state（只读）、候选 request、当前 batch plan（已选集合与预算消耗）
- 输出：admit/defer + 原因码（用于 debug/metrics）

### 3.8 schedulers（排序与提升规则）

- `schedulers/base.py`：统一调度器接口
- `schedulers/fcfs.py`：FCFS baseline
- `schedulers/psf.py`：short-first baseline
- `schedulers/aging.py`：AA（aging / anti-starvation）
- `schedulers/qba.py`：QBA（quota + promotion debt）

关键约束：scheduler 只决定“候选顺序/提升规则”，不负责：

- 预算分配（由 batch builder/engine 负责）
- KV 风险门控（由 admission 负责）

### 3.9 simulator（事件循环）

- `simulator.py`：事件驱动主循环（event heap / time advancement）

职责是把组件拼起来：

workload → predictor → scheduler → admission → engine → trace → metrics

### 3.10 metrics（指标与统计口径）

- `metrics.py`：指标计算与口径冻结（overall + class-wise）

重点支持：

- mean/p95/p99 waiting、E2E、TTFT、TPOT
- throughput、batch usage ratio、KV utilization、OOM probability
- fairness/QoS：starvation、SLA violation、class-wise tail
- 稳态统计：warm-up + measurement window + drain（避免右删失）

### 3.11 analytical（分析层 surrogate）

- `analytical/mg_n.py`：M/G/n 近似（均值层趋势）
- `analytical/phi.py`：φ(ρ) 形式与参数化
- `analytical/calibration.py`：从 simulator trace 做 calibration

关键约束：analysis 层不依赖 engine 内部实现细节，但可以消费 trace/metrics schema。

### 3.12 experiments / plotting（实验与画图）

- `experiments/*`：扫参入口（lambda/quota/gamma…），以 YAML 驱动
- `plotting/*`：画图入口（fig2/fig3/fig4…），消费实验产物
- `io.py`：统一结果落盘与读取（npz/json/schema），避免重复实现

## 4. 运行时数据流（端到端）

一次“配置驱动实验”的端到端路径建议如下：

1) `scripts/run_experiment.py` 读取 `configs/experiments/*.yaml`
2) `config.py` 合并基线配置与 sweep 覆盖，生成运行期 Config
3) `workload.py` 生成 arrival 序列与请求对象（带真值字段）
4) `predictor.py` 生成每个请求的预测字段（不确定性/分位数/风险概率）
5) `simulator.py` 启动事件循环：
   - scheduler 决定候选顺序
   - admission 决定能否准入（KV-safe / risk-aware）
   - engine 执行 batch 并返回 trace snapshot（kv_peak、batch_usage、完成/释放）
6) `metrics.py` 在指定统计窗口内聚合指标
7) `io.py` 将 config/seed/workload summary/metrics/schema 与产物落盘到 `artifacts/`
8) `scripts/draw_figure.py` 调用 `plotting/*` 读取产物并画图

## 5. 核心不变量（必须写进测试）

以下不变量建议作为最早的 regression 测试集合：

- KV 不为负；请求完成后释放 `prompt_len + answer_len`，KV 回到期望值
- `processed_len` 单调且不超过 `answer_len`
- `prefill_remaining` 单调不增且不为负；continuation 不丢 token
- `kv_peak` 必须是“释放前峰值”，OOM 判定只使用真值轨迹
- 同 seed 同配置结果 deterministic（至少在单进程下）
- batch usage 口径与 KV 增量口径一致（若存在“同轮首个 decode block”之类特殊增量，必须明确计入哪一方，并写入 spec）

## 6. 扩展点（面向 journal）

### 6.1 uncertainty-aware predictor

在 `predictor.py` 中扩展输出：

- `predict_answer_len_mean` / `predict_answer_len_q{p}`
- `overflow_prob` 或 `kv_risk_score`

并确保 admission/scheduler 的输入接口不需要改动主循环。

### 6.2 risk-aware admission

在 `admission.py` 中实现：

- 保守上界（kv_reserved_pred_sum）
- 有限视界 lookahead
- 轻量 lookahead（基于“下一次 decode 完成点”的短期峰值门控）

要求：评价 OOM 永远以真值轨迹计数，保证策略间可比。

### 6.3 multi-tenant QoS

在 `types.py` 为 Request 增加：

- `tenant_class`、`sla_target`、`priority_tier`

在 `metrics.py` 支持 class-wise 指标与 violation；在 `schedulers/*` 与 `admission.py` 增加 class-aware 版本（优先用“总序/score”而不是插队矩阵）。

## 7. 建议的实施顺序（第一阶段）

建议按“先复现语义，再扩展功能”的顺序推进：

1) 选定一个固定配置点作为回归基线（mean/p99/OOM/promotion count 容差）
2) 先落地 8 核心模块（types/state/workload/engine/admission/schedulers/simulator/metrics）
3) 再落地 experiments/plotting/io，把脚本变成配置驱动
4) 最后引入 uncertainty/multi-tenant/risk-aware，并为每个语义改变补 ADR 与回归点
