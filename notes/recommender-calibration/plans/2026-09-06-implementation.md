# 推荐校准数据拆分与对比实验实现计划

**Goal:** 独立准备推文，以全部训练记录拟合公共概率和推荐参数，在固定测试集比较两组系数及概率敏感性。
**Architecture:** 准备脚本产出固定数据包，训练入口只读训练分区，比较入口只运行冻结参数；共享现有画像、ABM 和基础打分函数。
**Tech Stack:** Python 3.12.11、标准库 unittest/json/csv/hashlib/tempfile/pathlib/logging、现有 pandas/openpyxl/NumPy/Optuna/joblib/SciPy/Pydantic。

> 状态：待批准，未开始实施。方案：[技术方案](../docs/2026-09-06-design.md)；依据：[Design Review](../docs/2026-09-06-design-review.md)。设计评审 checkpoint：`dc6e493`。
> 执行：当前本地 checkout、当前 agent 串行执行；用户明确禁止 subagent。以下命令为实施时的验证入口，本轮仅验证文档，不声称这些尚未建立的测试已经通过。

## Global Constraints

### 授权与工作区

- 工作区：`/Users/zhangyipeng/ZYPRoom/cuc/project/moss_web`。不创建 worktree、不复制仓库、不切换分支、不调用 subagent。
- 当前源码存在用户原有未提交修改。开始每个 Task 前记录该 Task 文件的差异基线；只暂存任务拥有的片段。无法安全隔离则停止 checkpoint，不 stash、reset、restore 或顺带提交用户变更。
- 方案已按用户条件授权完成评审；本计划尚未批准。用户批准实现后，按 Task 执行 Red、Green、Refactor、本地只读审查和验收，再创建 checkpoint。纯文档不引入伪测试。
- 原计划一旦开始实施不回填后续审查发现；可操作发现另写 `notes/recommender-calibration/plans/2026-09-06-review-remediation-<n>.md`，功能 Note 追加摘要。Notes 的维护属于工作流，不扩张下列生产代码 write set。

### 实验不变量

- 测试结果不参与锚点、训练选择、概率、权重或超参数拟合；比较不调用 Optuna。所有训练 trial 包含完整训练 ID。
- 公共概率、权重和衰减成对保存。两组模型只替换批准的系数，人口、画像、训练种子、在线概率、信念参数、轮数、时间尺度等一致。
- 只统计缩放后的终点传播量；不能把达标率称为推荐准确率，不能把参数替换称为真实系统的因果效果。
- JSON 写入禁止 NaN/Infinity；零分母按规格输出 null/CSV 空值。失败必须非零退出，不丢样本后成功，不覆盖已有实验。

### 工具与错误处理

- 系统 `python3` 是 3.9.6，不用于导入运行项目；`.venv/bin/python` 已确认是 3.12.11，`uv` 已定位于 `/opt/homebrew/bin/uv`。
- 用现有依赖，不增加测试框架或数据验证框架。测试使用 `unittest discover`，不需要 `tests/__init__.py`。测试的生成数据仅写普通临时目录，不创建临时 Git 仓库。
- 通用 code-hygiene profile：内部异常保留 cause，最外 CLI 记录一次可定位错误并退出；不吞错、不重复打印、不记录正文/密钥。预期非画像文件可排除并记录计数；清理错误不覆盖主异常。
- 实施前运行 `.venv/bin/python -c "import numpy,pandas,optuna,scipy,pydantic,openpyxl"`。缺依赖时按仓库依赖恢复，不改变版本约束；不得把未运行的集成验证记为通过。

## 文件与接口地图

所有路径相对工作区；路径名称用于明确写入范围，不表示本轮已经创建实现文件。

| 文件 | 所有权与复用 |
|---|---|
| `analysis/recommender_data.py` | 记录解析、数据包校验、散列与原子输出；用标准库字典、Path，复用现有表格转换逻辑并修正静默过滤 |
| `scripts/prepare_recommender_data.py` | 薄 CLI；脚本直接运行时显式将仓库根加入导入路径 |
| `analysis/recommender_parameter_inference.py` | 复用 ABM/画像/兴趣处理，公共概率交替优化与全量评估 |
| `analysis/run_analysis.py` | recommender 参数解析、训练产物输出；其他命令保持原接口 |
| `core/calibration_profile.py` | 可选 `p_base_global`；不改已有在线字段语义 |
| `analysis/compare_recommenders.py` | 纯指标函数、固定参数实验、对照输入、CSV/JSON CLI |
| `tests/test_recommender_data.py` | 数据、原子输出和无泄漏测试 |
| `tests/test_recommender_training.py` | 全量遍历、随机性、成对最优及状态重置测试 |
| `tests/test_recommender_cli.py` | 训练命令、YAML 迁移、失败与指纹测试 |
| `tests/test_recommender_comparison.py` | 指标、固定参数、网格、CSV 与 bootstrap 测试 |
| `tests/test_recommender_integration.py` | 不联网的三入口集成及真实 ABM 小样本 |
| `analysis/README.md` | 完整命令与迁移、指标和研究限制 |

共享接口采用普通 dict/list，数据边界集中校验，内部不反复解析。内部权重名称沿用 `w_i/w_pop/w_time/w_rand`；文件使用 `w_interest/w_popularity/w_time/w_random`，在输入输出边界各转换一次。

## Task 1：可复用的数据准备包

**Files / Write set:**
- Create: `analysis/recommender_data.py`
- Create: `scripts/prepare_recommender_data.py`
- Test/Create: `tests/test_recommender_data.py`

**Interfaces:**
- Consumes: 原始 CSV/XLSX 或 `{records: [...]}` JSON，方案 §4 的 CLI 参数。
- Produces: `prepare_dataset(source: Path, output_dir: Path, *, num_agents: int, text_column: str | None, retweet_columns: str, view_column: str, id_column: str, test_ratio: float=0.3, random_seed: int=42, anchor_percentile: float=0.8, min_scaled_target: int=5, selection: str='all') -> Path`，返回 manifest 路径。
- Produces: `load_split(path: Path, *, expected_split: str) -> tuple[dict, dict]` 返回分区与 manifest；只读取所请求分区和相邻 manifest，不打开另一分区；校验版本、字段、散列、ID 与规模。
- Produces: `file_sha256(path: Path) -> str`；`publish_output(output_dir: Path)` 上下文管理器，向调用者提供临时目录，成功才发布。

**Minimal implementation:** 从现有 StoryManager 搬入过滤、缩放、十档抽样算法，不导入重量级推断模块或 `run_analysis`；pandas 读表，标准库写 JSON/散列。函数内数据结构即可，无 Dataset 类层级。发布使用同父目录临时目录和独占创建的同名锁文件；已存在输出或锁则失败，rename 前重新检查；只清理本进程创建的临时目录和锁。硬杀遗留锁报错并提示核实进程后人工移除，不自动抢锁。

**Acceptance:** 70/30 固定划分先于结果过滤；锚点仅取训练候选；默认 all 不抽样；测试保留零转发；文本保真、重复 ID/缺列/非有限数失败；文件数值与 manifest 一致；同路径双进程仅一个发布成功。

- [ ] Red：写 `test_split_before_filter`、`test_test_target_does_not_change_train`、`test_selection_all_preserves_eligible_rows`、`test_stratified_reproducible`；构造足够浏览量和转发量的小数据，记录固定 seed 的测试 ID，只改变这些 ID 的结果后重跑，训练文件字节及锚点应不变。
- [ ] Red：写 CSV/XLSX/JSON 正文保真、零浏览排除计数、缺列/重复/NaN/空正文失败，及训练 0 条、总数 1 条失败测试；测试文件可在临时目录生成，不提交数据集。
- [ ] Red：写已有输出、上下文中抛异常、两进程竞争、损坏散列及 split 不符测试。`load_split(train)` 中拦截对 test 文件的 open，确认未发生访问。
- [ ] 执行 `.venv/bin/python -m unittest discover -s tests -p 'test_recommender_data.py' -v`，记录目标失败；按上述接口实现后重复同一命令，应全部通过。
- [ ] 运行 `.venv/bin/python scripts/prepare_recommender_data.py --help` 与 `.venv/bin/python -m py_compile analysis/recommender_data.py scripts/prepare_recommender_data.py`。检查成功产物无临时目录、输入文件未变。

**Dependencies:** 无前置；blocks Task 3。**Parallelism:** 用户要求本地串行，不并行。

## Task 2：公共概率与全量训练引擎

**Files / Write set:**
- Modify: `analysis/recommender_parameter_inference.py`
- Test/Create: `tests/test_recommender_training.py`

**Interfaces:**
- Consumes: 已校验且带 `I_pop` 的完整故事列表，每条保留 story_id 与 scaled_target；现有 VectorizedABMEngine。
- Produces: `EMCalibrationEngine.run_global_calibration(iterations=3, duration=24, n_repeats=5, p_trials=20, weight_trials=50) -> dict`；结果为 `weights`（包含 decay_lambda）、`p_base_global`、`loss`、`diagnostics`。
- Produces: `RecommendationParameterInferer.load_prepared_stories(records: list[dict]) -> None`，建立全部故事映射，不筛选或切分；现有 `precompute_interests()` 继续可用。
- Produces: 可由训练与比较共用的 `run_fixed_simulations(engine, stories, weights, p_base, *, duration, n_repeats, seed, n_cpu=1) -> np.ndarray`，行按 stories 顺序、列按重复序号，返回终值；明确每任务独立引擎/状态。

**Minimal implementation:** 保留现有类和 ABM，复用 `_weights_from_trial_params`；公共概率和权重目标闭包共享同一全量计数损失。旧逐条方法不作为新入口调用，不增加策略注册器。现有 M 步和消融的 20 条截断移除，使显式旧诊断也全量，避免留下貌似全量的限流路径。未知外部 Python 调用不作兼容承诺，必要参数移除报错而非忽略。

**Acceptance:** 25 条记录每个 trial 全覆盖；概率与权重使用方案 §5.2 的平均绝对计数误差/人口规模；同 seed 重放；选全轮最佳成对参数；空记录、非法预算和非有限预测失败；不使用轨迹损失；实际步数传播到所有调用。

- [ ] Red：用轻量确定性引擎记录每次输入 story ID（由不同 I_pop 映射）、概率、轮数和 seed；25 条、至少两个 trial，断言每个目标评估覆盖所有 25 条，不依赖总调用数间接推断。
- [ ] Red：构造不同轮最佳损失与最终轮相反的优化结果，断言返回概率和权重来自同一最佳轮；用手算终值验证两步统一损失，零目标不导致无穷。
- [ ] Red：实际小 ABM 验证同 seed 同输出、连续运行状态重置和 duration=2/3 被应用；稳定种子不得包含 trial 序号，sampler 显式固定 seed。并行结果按索引排序回收。
- [ ] 运行 `.venv/bin/python -m unittest discover -s tests -p 'test_recommender_training.py' -v`，观察上述失败后实现最小改动，再跑至通过。
- [ ] 运行 `.venv/bin/python -m py_compile analysis/recommender_parameter_inference.py`；搜索 `max_stories_per_trial|max_stories|sample_size`，逐处确认活动训练/诊断不剩 20 条限制，不以简单字符串消失替代运行验证。

**Dependencies:** 按本计划串行在 Task 1 checkpoint 后执行；算法本身不依赖 Task 1 代码。blocks Task 3/4。**Parallelism:** 无。

## Task 3：训练 CLI、可重建模型与 YAML 迁移

**Files / Write set:**
- Modify: `analysis/run_analysis.py`
- Modify: `analysis/recommender_parameter_inference.py`
- Modify: `core/calibration_profile.py`
- Test/Create: `tests/test_recommender_cli.py`

**Interfaces:**
- Consumes: Task 1 的 load_split/publish_output/file_sha256，Task 2 的 prepared stories、公共校准结果。
- Produces: `recommender --train-file --portraits-dir --output-dir`；训练预算/模型/时间参数按方案 §5.1，人口从训练文件读取。
- Produces: `model.json`，键 `schema_version=1, data, weights, decay_lambda, p_base_mode, p_base_global, training, environment, portraits, embedding`；data 保存 train_hash、manifest_hash、train_story_ids 和 scale_ratio；environment 保存训练种子、完整 ABM 参数；training 保存预算、loss_name、逐轮及重放诊断。
- Produces: `load_portrait_bundle(path: Path) -> tuple[list[dict], list[dict]]`，返回按相对文件名排序的 personas 和 `{path, sha256}` 清单；复用 `_portrait_to_persona` 与非画像识别。
- Produces: `RecommenderConfig.p_base_global: float | None`，非 None 必须在 `[0.001,0.999]`；YAML 与 model 权重、概率、时长一致。

**Minimal implementation:** 利用现有 argparse 子命令及 YAML 生成函数，不重构 portrait/retier；保留旧参数的解析以产生定向迁移错误，拒绝新旧混用，移出表格静默过滤及自动评估调用。画像 bundle 复用转换逻辑；合法非画像 JSON 不进入清单，损坏 JSON 报错保留 cause，不能继续训练出不同人口。为稳定 KernelPCA 路径显式传入训练 seed（同一环境保证重放，不承诺跨依赖版本逐位一致）。

**Acceptance:** train 输入全部使用，训练过程中 test 文件可以不可读而仍成功；无画像/坏画像失败；model 是比较权威产物；旧 YAML 可解析且在线字段不变；旧输入选项报迁移提示，非推荐命令 help 不变；输出目录不覆盖。

- [ ] Red：用 unittest.mock 替换嵌入编码为固定向量，运行真实训练编排；拦截测试文件访问及 `evaluate_holdout/split_holdout/select_representative_stories`，任一调用抛错，训练仍应通过。
- [ ] Red：测试模型保存完整种群种子与 ABM 设置、画像清单稳定、双产物参数一致、空 calibrated_p_base、旧 YAML 无新字段可解析；非法概率拒绝。
- [ ] Red：旧 `--data-file`、`--input`、筛选参数及新旧混用明确非零退出；测试画像损坏、缺文件、已有输出不产生部分成功目录。
- [ ] 运行 `.venv/bin/python -m unittest discover -s tests -p 'test_recommender_cli.py' -v`，观察失败，实现后通过；再运行已有 Task 1/2 测试。
- [ ] 运行 `.venv/bin/python -m analysis.run_analysis recommender --help` 和 `.venv/bin/python -m py_compile analysis/run_analysis.py core/calibration_profile.py analysis/recommender_parameter_inference.py`。

**Dependencies:** blocked-by Task 1/2；blocks Task 4/5。**Parallelism:** 与 Task 2 同文件，严格串行。

## Task 4：固定参数比较、误差与敏感性

**Files / Write set:**
- Create: `analysis/compare_recommenders.py`
- Test/Create: `tests/test_recommender_comparison.py`

**Interfaces:**
- Consumes: model v1、test v1、相邻 manifest、baseline JSON 和画像 bundle；Task 2 的 run_fixed_simulations。
- Produces: `compute_metrics(targets: np.ndarray, finals: np.ndarray, *, relative_threshold: float, absolute_threshold: float) -> dict`，finals 形状为内容×重复。
- Produces: `compare_models(model_path: Path, test_path: Path, baseline_path: Path, portraits_dir: Path, output_dir: Path, *, n_repeats: int=30, seed: int=2026, p_base_grid: list[float] | None=None, relative_threshold: float=0.2, absolute_threshold: float=1.0) -> Path` 返回 summary 路径。
- Produces: CLI 与方案 §6.1/6.3/6.4 一致，增加 `--seed` 显式控制比较随机性。
- Produces: summary 顶层 `schema_version, inputs, configuration, main, sensitivity`；main 包含 candidate/baseline 指标、improvement 和 mae_difference_ci95；per_story 列严格按方案 §6.3。

**Minimal implementation:** 标准库解析/CSV；用 NumPy 直接算指标与 2000 次配对 bootstrap，不引入统计框架。两组共享一次重建的种群与兴趣，重复模拟时状态归零。baseline 边界统一转内部权重名；唯一纯指标函数便于手算验证，不构造通用模型适配器。

**Acceptance:** 两组共用模型公共概率和环境；输入目标改变仅改变指标；train/test ID 交集、散列、画像、人口、尺度、模型版本不符失败。baseline 缺项/未知字段/负数/非有限/总和零失败；未提供 decay 共用，提供时标注比较范围。全部固定网格均输出，主参数不被择优覆盖。

- [ ] Red：手算 targets `[0,2,4]`、每条恒定预测 `[1,2,6]`，MAE=1、RMSE=sqrt(5/3)、MRE_nonzero=0.25、relative_pass_rate_nonzero=0.5、absolute_pass_rate=2/3；测试全零目标的相对指标 null，单条 bootstrap 区间 null。
- [ ] Red：相同系数+相同 seed 两组逐条输出相同、MAE 差及区间为零；测试全部网格、重复网格去重、主网格复用、参数取值边界及 bootstrap 配对索引。
- [ ] Red：替换优化函数为抛错函数，固定参数比较仍完成；两次使用相同文本/ID、不同真实目标的合法测试包，模拟输出一致。单独验证损坏散列拒绝，而非绕过加载验证完成目标扰动测试。
- [ ] Red：测试基线衰减省略/提供标签、数据交集、画像缺失、种群种子不被比较 seed 替换、非法输出值、n_repeats<30、阈值/grid 非法；错误退出无成功目录。
- [ ] Red：CSV 外部字符串含换行/逗号/公式前缀正确转义，零分母空值，JSON allow_nan=False。报告留存原权重、归一化权重、截断/零目标数量和全部输入指纹。
- [ ] 运行 `.venv/bin/python -m unittest discover -s tests -p 'test_recommender_comparison.py' -v`，观察目标失败后实现并通过；运行 `.venv/bin/python -m analysis.compare_recommenders --help` 和对应 py_compile。

**Dependencies:** blocked-by Task 1/2/3；blocks Task 5。**Parallelism:** 无。

## Task 5：命令级集成、迁移说明与交付验收

**Files / Write set:**
- Test/Create: `tests/test_recommender_integration.py`
- Modify: `analysis/README.md`

**Interfaces:**
- Consumes: 三个 CLI、数据包、模型、比较输出的已完成契约。
- Produces: 一个 discover 可运行的离线集成测试文件；README 的真实命令和指标说明。

**Minimal implementation:** unittest 临时生成表格与画像；固定小向量只替换嵌入模型，保留真实数据处理、校准器和 ABM。无网络测试在同进程通过 CLI 参数解析进入三个真实编排入口；另用 subprocess 验证模块 help 和非法参数退出。真实嵌入冒烟使用本地已缓存模型，不为测试新增下载器或 fixture 框架。

**Acceptance:** 完整 prepare→train→compare 产物可校验、同组零差、人口与轮数一致、旧 YAML 可读；README 明示旧命令迁移、测试数据冻结、误差单位、零目标、截断、固定概率偏差及时间弱辨识。保存真实验证范围，不把固定嵌入冒烟称为真实模型集成。

- [ ] Red：写跨模块集成测试，原始数据≥40 条并含零转发，保证训练筛选后≥25 条；小人口≥50，使缩放后仍有训练记录。测试预算可在底层调用时降低，但不改变生产 CLI 默认；比较重复数仍为 30。
- [ ] 运行 `.venv/bin/python -m unittest discover -s tests -p 'test_recommender_integration.py' -v`，若失败先定位规格或接线问题。需要改前序任务文件时，按工作流新建补救计划并明确新 write set，不在本 Task 越界修改。
- [ ] 更新 README 三阶段可复制命令，baseline JSON 完整例子、输出字段、迁移错误示例和研究解释；文档变更以链接与命令解析检查代替 Red 测试。
- [ ] 全量运行 `.venv/bin/python -m unittest discover -s tests -p 'test_recommender_*.py' -v`；再运行 `.venv/bin/python -m py_compile analysis/recommender_data.py analysis/recommender_parameter_inference.py analysis/run_analysis.py analysis/compare_recommenders.py scripts/prepare_recommender_data.py core/calibration_profile.py` 和 `git diff --check`。
- [ ] 检查本地嵌入资源；可用时以少量真实画像、25 条训练推文和最小预算跑真实 ABM 冒烟，归档输入/参数/输出指纹、实际运行命令及耗时。资源不可用则记录缺口及恢复条件，不能发布“完整集成已验收”。

**Dependencies:** blocked-by Task 1–4；blocks 整体审查与 closeout。**Parallelism:** 无。

## 需求、风险与验收覆盖

| 规格 | Task / 关键证据 |
|---|---|
| §4 独立筛选/正文/先划分/训练锚点 | Task 1；测试目标扰动、保真与散列 |
| §5 全量/公共概率/一致损失/成对最优 | Task 2；25 条每 trial、手算损失、跨轮最优 |
| §5 模型/YAML/训练不读测试 | Task 3；拦截访问、兼容配置、画像指纹 |
| §6 固定环境/误差/对照/网格/区间 | Task 4；同参零差、目标扰动、指标手算、网格全输出 |
| §7 原子发布/错误/隐私/迁移 | Task 1/3/4；中断/并发/非法输入；Task 5 文档 |
| §8 成本与科学边界 | Task 2/4 进度和预算记录；Task 5 文档与冒烟耗时 |
| §10 真实入口与回退 | Task 5；全量回归、原始文件与历史产物未变 |

关键路径：Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → 本地整体审查与验收。用户禁止 subagent，全部串行，无可并行任务。

## 审查与 checkpoint

每个 Task 目标测试通过后，当前 agent 分别进行规格核对、只读缺陷审查、复杂度和错误/日志专项，再检查实际 diff 的所有权，按任务片段 checkpoint。因用户禁止 subagent，独立 reviewer 不可用，明确记录本地自审的局限；不能把自审称为独立审查。

Task 5 后对本功能累计差异做整体审查与相关回归。关闭阶段前使用 verification-closeout 核对最终规格、测试、迁移和真实集成证据；未解决的可修复问题继续修复。只有检查和全部所需 checkpoint 成功，Note 才可标为 implemented。代码任务以中文提交说明，不 push。

## 恢复上下文

```yaml
feature_slug: recommender-calibration
note_root: notes/recommender-calibration
design_path: notes/recommender-calibration/docs/2026-09-06-design.md
plan_path: notes/recommender-calibration/plans/2026-09-06-implementation.md
remediation_plan_paths: []
workspace_roots:
  - /Users/zhangyipeng/ZYPRoom/cuc/project/moss_web
current_phase: Plan
continuation_mode: prompted # 用户仅授权评审通过后自动写计划，未授权实施
risk_level: high
artifact_depth: high
approval_mode: stage-gated
execution_mode: local # 用户明确禁止 subagent
workspace_mode: shared-checkout
final_stage_mode: develop-and-close
commit_mode: checkpoint
available_project_skills: [academic-paper, academic-paper-reviewer, academic-pipeline, academic-researcher, deep-research, drawio, paper-search, ppt-master]
matched_project_skills: [academic-researcher]
implementation_status: not-started
plan_approval: pending
```

下一步：用户批准本计划后，在当前 checkout 本地串行实施。
