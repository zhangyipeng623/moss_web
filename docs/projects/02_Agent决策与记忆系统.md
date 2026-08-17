# Agent 决策与记忆系统

> 毕业设计项目：MOSS Web 社交媒体舆情模拟推演平台
> 完成人：张艺鹏（个人完成）
> 文档范围：Agent 的决策循环、画像注入、动作工具与记忆系统。

## 1. 功能目标

让每个 Agent 在“感知—决策—动作—记忆更新”的闭环中持续行动：每轮获取个性化信息流与模拟时间，调用 LLM 及发帖、评论、点赞、转发、引用等工具，并把动作轨迹与结构化决策结果写入进程内记忆，形成带画像、短期上下文和事件记忆的可持续决策。

## 2. 决策循环

`AgentGraph` 负责多 Agent 的调度（[agent_graph.py](/Users/zhangyipeng/Documents/moss_web/moss_agent_client/agent_graph.py:14)）：

- `load_from_config` 批量创建 Agent；`start_all` 并发注册/登录；`step_all` 并发执行每轮决策；
- 任一 Agent 在本轮执行失败会中止整轮（`step_all` 抛错），避免重复副作用动作污染实验数据；
- 每轮结束后按 `system_time` 步进模拟时钟。

`MossAgent` 是单个 Agent 的执行体（[agent.py](/Users/zhangyipeng/Documents/moss_web/moss_agent_client/agent.py:85)）：用 LangChain `create_agent` 把 `ChatOpenAI` 与动作工具绑定，`temperature=0` 固定采样以保证可复现（[main.py](/Users/zhangyipeng/Documents/moss_web/main.py:148)）。

## 3. 画像注入

Agent 通过 `profile_mode` 选择画像模板（[agent.py](/Users/zhangyipeng/Documents/moss_web/moss_agent_client/agent.py:165)）：

- `default`：注入 5 阶段生成的完整结构化画像（`FIXED_USER_INFO_TEMPLATE`，九项固定字段）；
- `custom`：注入自定义模板；
- `simple`：注入 `SIMPLE_USER_TEMPLATE`（bio + 按 tier 的行为预期），供低层级预设用户低成本驱动。

`belief_text`（default/custom 用 `identity_summary + interest_summary`，simple 用 bio）随注册写入数据库，供在线推荐做语义匹配（[agent.py](/Users/zhangyipeng/Documents/moss_web/moss_agent_client/agent.py:155)）。

## 4. 动作工具

动作规格集中在 `moss_agent_client/actions.py`，通过 `StructuredTool` 暴露给 LLM：发帖、评论、点赞帖子、转发、引用、点赞评论、不动作（do_nothing）。每个动作经 `RemotePlatform` 调 Backend API，结果回写 `ActionTrace` 留痕。

## 5. 记忆系统

记忆模型定义在 `memory.py`，管理逻辑在 `memory_manager.py`：

| 记忆 | 结构 | 说明 |
|---|---|---|
| 静态上下文 | `StaticContext` | 画像文本 + 全局事件，全程不变 |
| 短期记忆 | `ShortTermMemory` | 最近 N 轮（默认 3）的环境快照 + 动作 + 输出 |
| 事件记忆 | `EventMemory` | 最多 50 条，按 `importance` 降序，去重合并 |
| 动态状态 | `AgentState` | mood/emotion/intensity/current_goal/focus_topics/stance/attention_target |

每轮决策后，LLM 输出结构化 `DecisionResultPayload`（状态更新 + 事件记忆写入），`MemoryManager.update_after_step` 据此更新状态与事件记忆；结构化解析失败时保留上一轮状态，不污染记忆（[memory_manager.py](/Users/zhangyipeng/Documents/moss_web/moss_agent_client/memory_manager.py:137)）。

## 6. 事件记忆检索

`select_relevant_events` 用乘法评分 + 时间衰减从事件记忆取相关记录（[memory_manager.py](/Users/zhangyipeng/Documents/moss_web/moss_agent_client/memory_manager.py:49)）：

```
score = importance × exp(-λ × rounds_ago) × (1 + min(context_boost, cap))
```

- `importance`：Agent 当初自评的重要度（乘法保证低重要性事件无法反超高重要性事件）；
- 时间衰减：Ebbinghaus 遗忘曲线，`λ` 可配置（默认 0.07，半衰期约 10 轮）；
- 上下文联想：引用提醒、相关人物出现、话题命中当前关注等，加成上限 0.3，锦上添花不喧宾夺主。

记忆参数（窗口/容量/衰减/加成）由 `calibration_profile.yaml` 的 `simulation.memory` 段注入（[agent.py](/Users/zhangyipeng/Documents/moss_web/moss_agent_client/agent.py:110)）。

## 7. 当前状态

**已实现**：多 Agent 并发调度、画像分层注入、工具化动作留痕、短期/事件/状态三层记忆、带时间衰减的乘法检索评分、记忆参数 YAML 注入、`temperature=0` 可复现。

**待实现**：带副作用动作的幂等键/事务边界（失败重试可能重放外部动作）、记忆持久化到运行归档（当前可序列化但未落盘）、覆盖记忆排序与动作幂等的自动化测试。
