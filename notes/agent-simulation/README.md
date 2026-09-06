# Agent 社交模拟 — 决策演进记录

> Status: in-progress
> 更新: 2026-09-06

## 当前定案

收录当前在线平台及本地修改，未补造审批或端到端验收。运行入口是 YAML 校准配置，前端需构建后由后端提供静态页面。

- [主入口](../../main.py) 启动 Backend 与 Agent 两个进程，进行健康检查并注入校准配置；[运行时](../../core/runtime.py) 管理每次运行的数据库、日志及配置快照。
- [AgentGraph](../../moss_agent_client/agent_graph.py) 按 `profile_mode` 选择模型：配置 `llm_small` 时 simple 使用小模型，其余使用大模型；未配置时全员回退大模型。当前还按 `p_online` 唤醒并限制并发（`simulation.agent_concurrency` 默认 30）。
- 当前 `step_all` 在部分活跃 Agent 失败时记录并继续；全部活跃 Agent 失败才抛错中止。此行为不同于 [原社交模拟文档](../../docs/projects/03_社交模拟.md) 的“任一失败中止”。
- [配置模型](../../core/calibration_profile.py) 与 [画像解析器](../../core/agent_profile_resolver.py) 已包含默认关闭的 `simulation.l1_l3_pool` 候选池抽样；启用后主入口按现有 Agent 数量为锚补入 simple 用户，数据失败时记录并跳过。
- [Agent](../../moss_agent_client/agent.py) 将画像 `simulation_init` 传入 [记忆管理器](../../moss_agent_client/memory_manager.py)；已有短期、事件记忆和动态状态更新。
- [推荐服务](../../backend/services/social_recsys.py) 接收校准权重与嵌入配置，当前候选查询排除本人帖子；[帖子 DAO](../../backend/dao/posts.py) 包含白名单排序，[前端](../../frontend/src/App.tsx) 提供对应浏览入口。

画像来源见 [用户画像](../user-portraits/README.md)，推荐参数见 [推荐参数校准](../recommender-calibration/README.md)，效果与成本对照候选见 [研究验证](../research-validation/README.md)。

## 演进时间线

| 时间 | 方案 | 结论 |
|---|---|---|
| 2026-09-06 | 收录在线闭环及本地唤醒、并发、候选池、初始状态和浏览修改 | 采用功能级现状记录；运行验收未完成；日期为收录日期，不是实现或审批日期 |

## 实现状态

- [x] 代码包含双进程、工具动作、推荐、记忆、模型分流及运行归档。
- [x] 本地工作树包含分层唤醒、并发上限、部分失败继续、候选池及初始状态注入。
- [ ] 验收启动、唤醒、模型分流、部分/全部失败和关闭进程的关键路径。
- [ ] 验证候选池数量与分层、帖子排序和排除本人帖子行为。
- [ ] 评估动作重试幂等、记忆落盘及 `simulation.feed` 注入等既有缺口。

待办仅是后续候选，未获准实施。本轮未启动后端、调用模型或执行前端构建，不将历史测试数量作为验收依据。
