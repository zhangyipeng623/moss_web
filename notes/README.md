# MOSS Web 功能决策索引

> 更新：2026-09-06
> 本目录按功能保存当前定案、演进摘要和实现状态，供后续开发工作流恢复上下文。

## 从哪里开始

| 功能 | 状态 | 内容与下一步 |
|---|---|---|
| [数据准备](data-preparation/README.md) | in-progress | 公开数据转换与观测构建已有本地脚本，待验证产物口径 |
| [用户画像](user-portraits/README.md) | in-progress | 画像生成、模式与分层已有实现，待验收当前修改 |
| [推荐参数校准](recommender-calibration/README.md) | approved | 方案评审通过，五项串行实现计划待批准；既有 ABM 实现保留 |
| [Agent 社交模拟](agent-simulation/README.md) | in-progress | 在线推荐、记忆和大小模型分流已有实现，待收口运行缺口 |
| [研究验证](research-validation/README.md) | proposed | 汇总研究目标和证据边界，尚未形成获批实验方案 |

`in-progress` 表示已有实现、尚有明确缺口或当前验收未完成，不表示整个功能尚未编写。`proposed` 表示待设计或评审；`approved` 仅用于有批准依据的方案；`implemented` 仅用于已完成交付验收的定案；`superseded` 指向替代功能。

## 目录约定

```text
notes/
  README.md
  <feature-slug>/
    README.md       # 当前定案、演进时间线、实现状态
    docs/           # 当前方案与必要的评审、验收证据
    plans/          # 实现计划与独立补救计划
    archived/       # 完全被取代的方案和计划
```

功能名使用 kebab-case。空目录以 `.gitkeep` 保留；出现实际文档时可移除占位文件。不为已有实现补造历史方案、计划、审批或测试记录。

现有 [项目文档](../docs/projects/README.md) 继续解释系统和实现；[研究待办](../docs/plan/论文写作与后续工作待办.md) 与 [研究总览](../docs/plan/研究总览_创新点与验证.md) 保留研究细节。Notes 链接这些来源，不复制全文；源文档没有被完整取代时不搬入 `archived/`。后续涉及具体功能的决策演进统一追加到对应 Note。

## 接入 dev-workflow

1. **Design**：读取功能 Note、适用 `AGENTS.md`、代码和 Git 状态；在 `docs/YYYY-MM-DD-design.md` 写方案，Note 设为 `proposed`。
2. **Design Review**：记录需求、职责、失败、兼容和验证审查；按工作流取得批准依据后设为 `approved`。
3. **Plan**：在 `plans/YYYY-MM-DD-implementation.md` 写规格映射、精确修改范围、依赖和验收方式；按工作流完成计划批准。
4. **Develop & Validate**：开始实施设为 `in-progress`；验证与适用审查、checkpoint 完成后才设为 `implemented`。审查补救独立写入 `plans/YYYY-MM-DD-review-remediation-n.md`，不回填已开始实施的原计划。

恢复任务时在对应方案或计划中维护 `ResolvedContext`：`feature_slug`、`note_root`、`design_path`、`plan_path`、`remediation_plan_paths`、`workspace_roots`、`current_phase`、`continuation_mode`、`risk_level`、`artifact_depth`、`approval_mode`、`execution_mode`、`workspace_mode`、`final_stage_mode`、`commit_mode`、`available_project_skills`、`matched_project_skills`。路径以仓库相对路径记录，工作区使用本轮实际路径；尚不存在的方案与计划明确记为未建立。

默认在当前 checkout 工作、按阶段 checkpoint，只提交任务自身变更。阶段与批准从证据恢复，不能仅凭 `in-progress` 推断存在获批计划。没有获批方案的新工作从 Design 开始；已有实现的单独验收可在明确验收范围后进入 Closeout。阶段连续执行与审批规则以实际加载的 `dev-workflow` 为准。

## 本次收录边界

- 基线：Git `880b6a8` 加 2026-09-06 本地工作树；存在用户原有未提交修改及未跟踪的 `scripts/`。本次只新增 `notes/`，引用本地脚本不代表脚本已提交。
- 时间线中的 2026-09-06 是现状收录日期，不是功能实现日期。只追加有证据的演进，不根据文件时间推断审批。
- 项目文档提到的 `test.py` 49 项验证是历史陈述；本次未运行，也不将其视为当前修改的验收证据。
- 本次验收限于目录、状态字段、相对链接、文档事实及变更范围；没有调用模型、运行仿真或评价研究结果。

## 维护检查

每个功能入口须有 `Status`、更新日期、当前定案、演进时间线、实现状态。时间线写明日期、方案和结论；结论有来源。方案完全被取代时移入 `archived/`，保留否决原因；当前权威链接不得指向已废弃方案。新增功能同步更新本索引，并检查相对链接和 `git diff --check -- notes`。
