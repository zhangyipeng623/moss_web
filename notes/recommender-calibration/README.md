# 推荐参数校准 — 决策演进记录

> Status: in-progress
> 更新: 2026-09-06

## 当前定案

当前代码以内容观测和画像为输入，通过 ABM/EM 推断推荐权重，再输出在线配置。这里记录实现事实，未认定权重在真实数据上有效，也未补造正式方案批准。

- [校准引擎](../../analysis/recommender_parameter_inference.py) 包含语义嵌入、Kernel PCA 立场轴、种群扩增、向量化 ABM 与 Optuna E/M 步，搜索兴趣、热度、时效、随机四维权重。
- 引擎包含留出切分、消融与种子鲁棒性入口；这些是可调用能力，不等于已有有效实验结论。当前本地增量使用 `asdict` 读取 dataclass 观测，并在画像目录加载时过滤非画像 JSON。
- [离线入口](../../analysis/run_analysis.py) 将推断结果映射至 `calibration_profile.yaml`；[配置模型](../../core/calibration_profile.py) 承载权重、嵌入模型及模拟参数，[在线推荐服务](../../backend/services/social_recsys.py) 通过 `configure()` 注入配置。
- 当前默认嵌入模型为 `Alibaba-NLP/gte-multilingual-base`；离线和在线需保持同一模型。双方复用 [打分函数](../../core/scoring.py)，但 ABM 仍有传播者层级均值等近似，不能将其描述为整个机制完全相同。

详见 [原校准文档](../../docs/projects/02_参数校准.md)。输入由 [数据准备](../data-preparation/README.md) 和 [用户画像](../user-portraits/README.md) 提供；产物交给 [Agent 社交模拟](../agent-simulation/README.md)。横截面数据下时间参数的证据边界与文档表述冲突见 [研究验证](../research-validation/README.md)。

## 演进时间线

| 时间 | 方案 | 结论 |
|---|---|---|
| 2026-09-06 | 收录 ABM/EM、YAML 闭环和当前嵌入/输入兼容修改 | 采用功能级现状记录；保留数据级验证缺口；日期为收录日期，不是实现或审批日期 |

## 实现状态

- [x] 代码包含四维校准、诊断入口和 YAML 配置输出。
- [x] 本地工作树包含嵌入默认值调整、dataclass 观测转换和非画像过滤。
- [ ] 验证真实输入可完成校准并被在线加载，保存配置与输出证据。
- [ ] 明确留出评估中逐内容概率再拟合的解释，避免夸大泛化结论。
- [ ] 根据实际数据明确时间参数可辨识范围与实验指标。

待办未形成获批实现计划。本轮未运行校准、加载嵌入模型或确认统计指标有效。
