# 张艺鹏开题报告汇报PPT - Design Spec

> 学位论文选题报告答辩汇报。源文件为《融合推荐机制校准的社交仿真平台设计与实现》选题报告申请表。
> 设计目标：学术答辩场景的稳重期刊风；统一术语「参数校准」（导师批注要求弃用「反演/反推」）；嵌入 4 张系统架构图；保留中国传媒大学校徽。

## I. Project Information

| Item | Value |
| ---- | ----- |
| **Project Name** | 张艺鹏-开题报告汇报PPT |
| **Canvas Format** | PPT 16:9 (1280×720) |
| **Page Count** | 16 页 |
| **Design Style** | pyramid + editorial（结论先行 + 学术期刊风） |
| **Target Audience** | 学位论文开题答辩评审老师 |
| **Use Case** | 正式学术答辩场合 |
| **Content Strategy** | balanced — 忠于申请表事实与论证链，按答辩讲法重组层级，统一「校准」术语，补足创新点与架构图，不引入源外新论据 |
| **Created Date** | 2026-06-27 |

---

## II. Canvas Specification

| Property | Value |
| -------- | ----- |
| **Format** | PPT 16:9 |
| **Dimensions** | 1280×720 |
| **viewBox** | `0 0 1280 720` |
| **Margins** | 左右 64px，上下 48px |
| **Content Area** | 1152×624 |

---

## III. Visual Theme

### Theme Style

- **Mode**: pyramid —— 结论先行，每页一个核心论断，MECE 展开；适合答辩论证与评审快速抓重点。
- **Visual style**: editorial —— 杂志式栏目层级，规则线 + 栏目分隔，serif 标题与 sans 正文互文，大量留白。
- **Theme**: Light theme
- **Tone**: 学术、稳重、严谨、克制

### Color Scheme

| Role | HEX | Purpose |
| ---- | --- | ------- |
| **Background** | `#FAF8F4` | 暖白页面底 |
| **Secondary bg** | `#F1ECE3` | 卡片/栏目底、引文块 |
| **Primary** | `#1B3A5B` | 深海军蓝——标题、页眉规则线、章节号、主结构 |
| **Accent** | `#C8A24B` | 暖金——关键词强调、规则线点缀、数据高亮 |
| **Secondary accent** | `#4A6B8A` | 钢蓝——次级强调、流程节点、图示辅助 |
| **Body text** | `#2B2B2B` | 正文主色 |
| **Secondary text** | `#5A5A52` | 注释、副标题 |
| **Tertiary text** | `#8C8C82` | 页脚、页码、出处 |
| **Border/divider** | `#D8D0C2` | 卡片边框、分隔细线 |
| **Success** | `#3C7A5A` | 正向指标（拟合提升） |
| **Warning** | `#B24A38` | 问题/空白标记 |

### Gradient Scheme

无渐变。editorial 风以实色块、规则线与留白构图为主，不使用渐变（避免投影下灰带）。

---

## IV. Typography System

### Font Plan

**Typography direction**: academic serif title + modern CJK sans body（学术 serif 标题领衔个性，正文中文黑体保证可读）

| Role | Chinese | English | Fallback tail |
| ---- | ------- | ------- | ------------- |
| **Title** | `SimSun` | `Cambria` | `serif` |
| **Body** | `"Microsoft YaHei"` | `Arial` | `sans-serif` |
| **Emphasis** | `SimSun` | `Cambria` | `serif` |
| **Code** | — | `Consolas, "Courier New"` | `monospace` |

**Per-role font stacks**:

- Title: `Cambria, SimSun, serif`（Latin 领衔，学术期刊气质；中文回落宋体）
- Body: `"Microsoft YaHei", Arial, sans-serif`
- Emphasis: `Cambria, SimSun, serif`
- Code: `Consolas, "Courier New", monospace`

### Font Size Hierarchy

**Baseline**: Body font size = 20px（中密度，答辩页信息量适中）

| Purpose | Ratio to body | @ body=20 | Weight |
| ------- | ------------- | --------- | ------ |
| Cover title | 2.5-5x | 56-64px | Bold |
| Chapter opener | 2-2.5x | 44-50px | Bold |
| Page title | 1.5-2x | 32-38px | Bold |
| Hero number | 1.5-2x | 32-40px | Bold |
| Subtitle | 1.2-1.5x | 24-30px | SemiBold |
| **Body content** | **1x** | **20px** | Regular |
| Annotation / caption | 0.7-0.85x | 14-17px | Regular |
| Page number / footnote | 0.5-0.65x | 11-13px | Regular |

公式策略：`mixed`——`exp(-λ·dt)` 指数衰减、四维加权评分式等渲染为 PNG；单变量 `w_i`、`p_base` 等保持可编辑文本。

---

## V. Layout Principles

### Page Structure

- **Header area**: 高 ~72px。左上章节号（深蓝）+ 章节名；右上校徽（小尺寸，~36px 高）。下方一条主色规则线贯穿。
- **Content area**: 高 ~540px。栏目化排布，规则线/留白分区。
- **Footer area**: 高 ~36px。左侧论文题目缩写，右侧页码；细分隔线。

### Layout Pattern Library

| Pattern | Suitable Scenarios |
| ------- | ----------------- |
| **Single column centered** | 封面、章节页、总结 |
| **Asymmetric split (4:6 / 3:7)** | 论点 + 架构图、文字 + 图示 |
| **Top-bottom split** | 超宽架构图（图4）、流程闭环 |
| **Three column** | 三条研究脉络、三个创新点、三类验证 |
| **Center-radiating** | 闭环总览、研究动因 |
| **Negative-space-driven** | 章节过渡、核心论断 |

### Spacing Specification

**Universal**: 安全边距 64px；内容块间距 28px；图标-文字间距 12px。
**Card-based**: 卡片间距 24px；内边距 24px；圆角 8px；三栏卡宽 ~360px。
**Non-card**: 行高 1.5×（正文）/ 1.7×（引文）；规则线分区，留白承载节奏。

---

## VI. Icon Usage Specification

### Source

- **Built-in icon library**: `tabler-outline`（线性图标，stroke_width 2），轻量学术感。
- **Usage method**: `<use data-icon="tabler-outline/icon-name" .../>`。

### Recommended Icon List

| Purpose | Icon Path | Page |
| ------- | --------- | ---- |
| 数据/数据库 | `tabler-outline/database` | P03/P08/P09 |
| 网络/传播 | `tabler-outline/share` | P03 |
| 用户/画像 | `tabler-outline/users` | P03/P09 |
| 推荐/曝光 | `tabler-outline/eye` | P03/P10 |
| 立场/天平 | `tabler-outline/scale` | P04 |
| 机制/齿轮 | `tabler-outline/settings` | P04 |
| 闭环/循环 | `tabler-outline/refresh` | P04/P08 |
| 文献/论文 | `tabler-outline/file-text` | P05 |
| 规则模型 | `tabler-outline/math-function` | P05 |
| 大模型/AI | `tabler-outline/robot` | P05/P09 |
| 平台/服务器 | `tabler-outline/server` | P05/P11 |
| 空白/缺口 | `tabler-outline/puzzle` | P06 |
| 目标/靶心 | `tabler-outline/target` | P07 |
| 校准/调节 | `tabler-outline/adjustments` | P07/P10 |
| 时间衰减 | `tabler-outline/clock` | P08/P10 |
| 热度/火 | `tabler-outline/flame` | P10 |
| 随机/骰子 | `tabler-outline/dice` | P10 |
| 决策循环 | `tabler-outline/cpu` | P11 |
| 验证/检查 | `tabler-outline/checklist` | P11/P12 |
| 进度/日历 | `tabler-outline/calendar` | P13 |
| 创新/灯泡 | `tabler-outline/bulb` | 创新点页 |
| 分层/堆叠 | `tabler-outline/stack` | 创新点页/P09 |

> 实际文件名以 `icon_sync.py` 校验为准。

---

## VII. Visualization Reference List

Catalog read: 71 templates

| Page | Template | Path | Summary-quote (verbatim from `charts_index.json`) | Usage |
| ---- | -------- | ---- | ------------------------------------------------- | ----- |
| P05 | vertical_pillars | `templates/charts/vertical_pillars.svg` | "Pick for 1×3 / 1×4 / 1×5 vertical column layout where each pillar = one independent category with title + bullets — PEST" | 三条研究脉络并列（规则 ABM / LLM Agent 混合 / 平台化推荐） |
| P07 | circular_stages | `templates/charts/circular_stages.svg` | "Pick for 4-6 stage closed loop where stages compose a cycle — PDCA, flywheel compounding loops (Attract → Engage → Delig" | 四目标闭环（画像生成→参数校准→在线仿真→闭环验证） |
| P10 | icon_grid | `templates/charts/icon_grid.svg` | "Pick for 4-9 parallel features/capabilities/services as icon cards — feature grid, service lineup, benefits matrix, bran" | 四维推荐评分（兴趣匹配/内容热度/时间衰减/随机探索） |
| 创新点 | vertical_list | `templates/charts/vertical_list.svg` | "Pick for 3-6 numbered key points each with a short description — design principles, core tenets, action items, key takea" | 三个递进创新点（参数校准方法/离线-在线闭环/分层架构） |
| P13 | timeline | `templates/charts/timeline.svg` | "Pick for 3-8 milestone events on a horizontal time axis (no duration). Skip for tasks with start/end ranges (use gantt_c" | 2026.6-12 六阶段进度 |

**Runners-up considered**:

- `roadmap_vertical` | rejected for P13：进度为水平时间推进、六个等距阶段，水平 timeline 更贴合答辩横版表达，垂直 roadmap 占高过多。
- `numbered_steps` | rejected for 创新点：创新点是并列递进的论点而非操作步骤，vertical_list 的「编号 + 短描述」更契合，numbered_steps 偏「how-it-works」流程语义。
- `segmented_wheel` | rejected for P10：四维评分是加权求和的并列因子而非环绕一个中心主题的均权切分，icon_grid 的卡片网格更利于逐维配图标说明。
- `hub_spoke` | rejected for P07：四目标是有先后的闭环循环（输出→校准→运行→验证→回流），circular_stages 表达循环优于中心-辐射的 hub_spoke。

---

## VIII. Image Resource List

| Filename | Dimensions | Ratio | Purpose | Type | Layout pattern | Acquire Via | Status | Reference | text_policy | page_role |
| -------- | --------- | ----- | ------- | ---- | -------------- | ----------- | ------ | --------- | ----------- | --------- |
| cuc_logo.png | 123×157 | 0.78 | 中国传媒大学校徽——封面主信息区 + 每页页眉右上 | Logo | #19 inline accent logo | user | Existing | 校徽，原 PPT 中已有 | | |
| fig1_closed_loop.png | 2582×1544 | 1.67 | MOSS 数据—校准—运行闭环总览（P08 技术路线主图） | Diagram | #4 single dominant figure + caption | user | Existing | 闭环总览架构图 | | |
| fig2_portrait.png | 2266×1564 | 1.45 | 用户画像生成模块设计（P09 主图） | Diagram | #2 left-third figure + right text column | user | Existing | 画像生成模块图 | | |
| fig3_abm.png | 2296×1174 | 1.96 | ABM 推荐参数校准模块设计（P10 主图） | Diagram | #5 top figure band + bottom text | user | Existing | ABM 参数校准模块图 | | |
| fig4_online.png | 2346×1114 | 2.11 | 在线社交网络模拟系统架构（P11 主图） | Diagram | #5 top full-width figure band + bottom text | user | Existing | 在线模拟系统架构图 | | |

> 全部为用户提供素材（B 路径），不做 AI 生成。四张架构图均为信息密集示意图，`spec_lock.md images` 中全部标 `no-crop`，按原始比例 `meet` 完整呈现，不裁切。

---

## IX. Content Outline

### Part 0: 封面与导航

#### Slide 01 - Cover

- **Cover impact**: 钩子 = 核心问题「如何从真实传播数据校准推荐机制参数，并注入大小模型协同的社交仿真平台」；构图 = 学术期刊封面式，主色规则线分区 + 暖金细线，校徽置于信息区，标题 serif 大字居左偏上，留白克制。
- **Layout**: 单列偏左，顶部细规则线，标题区 + 信息区两段，右上/信息区含校徽。
- **Title**: 融合推荐机制校准的社交仿真平台设计与实现
- **Subtitle**: 学位论文选题报告
- **Info**: 汇报人：张艺鹏　学号：202520085411016　责任导师：黄浩程　培养单位：媒体融合与传播国家重点实验室　研究方向：数据智能技术与应用　2026年6月　+ 中国传媒大学校徽

#### Slide 02 - 目录

- **Layout**: 左侧大字「目录 / CONTENTS」+ 右侧六章编号列表，中部一行核心问题引文块。
- **Title**: 汇报结构
- **Core message**: 本汇报围绕「从真实传播数据校准推荐参数并注入社交仿真平台」这一核心问题，分六章展开。
- **Content**:
  - 01 选题背景与研究意义 · 02 国内外研究现状 · 03 研究目标与内容
  - 04 技术路线与方案 · 05 可行性分析 · 06 进度安排
  - 核心问题（引文块）：如何从真实传播数据校准推荐机制参数，并注入 LLM Agent 与规则 Agent 协同运行的社交仿真平台？

### Part 1: 选题背景与研究意义

#### Slide 03 - 章节页+研究动因

- **Layout**: 上部章节标识（01 选题背景与研究意义），下部左文右图——三条研究动因 + 右侧「信息接触→互动行为→立场变化」的推荐曝光示意（center-radiating 小图）。
- **Title**: 为什么要做可校准的社交仿真
- **Core message**: 社交媒体是公共议题扩散与舆论形成的关键空间，而真实平台的算法黑箱使仿真成为从「结果描述」走向「机制解释」的必要路径。
- **Content**:
  - 社交媒体已成为公共议题扩散、舆论形成与平台治理的重要空间。
  - 真实平台存在算法黑箱、曝光日志难获取、对照实验受伦理与监管限制等问题。
  - 仿真的价值不只是复现宏观现象，而是解释用户行为、网络结构与平台机制如何共同生成传播结果。
  - 关键转向（强调）：从「结果描述」走向「机制解释」——推荐曝光串联 信息接触 / 互动行为 / 立场变化。

#### Slide 04 - 研究意义

- **Layout**: 三栏卡（理论/方法/工程）+ 底部一行关键转向论断。
- **Title**: 研究意义：推荐机制成为内生变量
- **Core message**: 本课题把推荐曝光从平台背景提升为可校准的内生机制变量，贯通「数据治理—参数校准—配置注入—在线仿真—结果归档」闭环。
- **Content**:
  - 理论意义：把推荐曝光从平台背景提升为传播过程中的机制变量，解释信息扩散与态度演化的形成链条。
  - 方法意义：以真实传播结果校准兴趣匹配、内容热度、时间衰减与随机探索等参数。
  - 工程意义：形成「数据治理—参数校准—配置注入—在线仿真—结果归档」的平台闭环。
  - 关键转向（强调）：推荐参数不再由研究者主观设定，而由真实传播数据校准后进入在线仿真。

### Part 2: 国内外研究现状

#### Slide 05 - 三条相关脉络

- **Layout**: 三栏并列（vertical_pillars），各栏含图标 + 标题 + 描述。
- **Title**: 三条相关脉络
- **Core message**: 现有研究在规则 ABM、LLM Agent 混合仿真、平台化推荐三条脉络上基础充分，但「真实数据→推荐参数校准→在线仿真注入」的链条仍不完整。
- **Visualization**: vertical_pillars（见 VII）
- **Content**:
  - 规则 ABM 与传播动力学：Bounded Confidence、SIR、Hawkes、元胞自动机等模型成本低、可解释；但弱在文本生成与个体行为解释。
  - LLM Agent 与混合仿真：Generative Agents、HiSim、FDE-LLM、GASim、TopoSim、APS 等推动画像、记忆、信念与混合架构。
  - 平台化系统与推荐机制：OASIS、GGBond、POSIM、PolicySim 等开始纳入平台环境和推荐排序，但参数多依赖预设或离线调参。
  - 底部论断（强调）：研究基础充分，但「真实传播数据 → 推荐参数校准 → 在线仿真注入」的链条仍不完整。

#### Slide 06 - 研究空白

- **Layout**: 左右对照（左「已有做法」/ 右「本课题拟解决」），底部一行转向论断。
- **Title**: 研究空白：推荐机制缺少实证校准
- **Core message**: 已有系统的推荐参数多来自人工预设或离线调参，缺少面向真实传播数据的实证校准，正是本课题的切入点。
- **Content**:
  - 已有系统的常见做法：无显式推荐模块（邻居传播、时间线或随机曝光）；有推荐模块者用兴趣公式、hot-score、人工权重或离线调参；政策优化类工作更关注干预效果。
  - 本课题拟解决：将推荐评分拆解为可解释维度；用真实传播结果拟合校准参数；把校准参数写入 YAML 并驱动在线 Feed 排序。
  - 底部论断（强调）：从主观预设到数据校准。

### Part 3: 研究目标与创新点

#### Slide 07 - 研究目标

- **Layout**: 顶部总目标论断 + 四目标闭环图（circular_stages）。
- **Title**: 目标：形成可复现实验闭环
- **Core message**: 设计并实现融合推荐机制校准的 MOSS 社交仿真平台，形成从真实数据到在线推演的可复现闭环。
- **Visualization**: circular_stages（见 VII）
- **Content**:
  - 总目标（引文）：设计并实现一个融合推荐机制校准的 MOSS 社交仿真平台，形成从真实数据到在线推演的可复现闭环。
  - 画像生成：从 user/post 数据生成稳定画像、行为画像和 Agent 运行画像。
  - 参数校准：拟合真实传播结果，校准推荐权重、时间衰减与层级影响参数。
  - 在线仿真：通过 Backend 与 Agent 双进程运行发帖、评论、点赞、转发等行为。
  - 闭环验证：用传播曲线、态度演化、互动分布和消融实验评估有效性。

#### Slide 08 - 创新点（新增）

- **Layout**: 三条编号递进（vertical_list），每条含创新点标题 + 一句话支撑。
- **Title**: 创新点：方法、闭环、架构三重递进
- **Core message**: 本课题在推荐参数校准方法、离线-在线全流程闭环、大小模型分层协同架构三方面形成递进创新。
- **Visualization**: vertical_list（见 VII）
- **Content**:
  - 创新一 · 推荐系统参数校准方法：将推荐抽象为兴趣匹配、内容热度、时间衰减、随机探索等可解释维度，用规则 ABM 拟合真实传播结果，通过 EM / Optuna / 贝叶斯优化校准权重——参数由真实传播过程校准而非主观假设。
  - 创新二 · 离线校准—在线仿真全流程闭环：将推荐权重、时间衰减、行为阈值与实验元数据写入统一 YAML，由在线平台读取驱动推荐服务、Agent 决策与实验记录，实现参数自动流转，提升可复现性与跨领域复用。
  - 创新三 · 大模型与小模型分层协同架构：高影响力核心用户由大模型驱动承担原创与复杂行为，普通用户由小模型完成低成本互动，在行为保真度、可解释性与运行效率间取得平衡。

### Part 4: 技术路线与方案

#### Slide 09 - 技术路线总览

- **Layout**: 顶部小标题 + 全幅闭环总览架构图（fig1，top-bottom，图占主体），底部一行配置中枢论断。
- **Title**: 数据—校准—运行—反馈闭环
- **Core message**: 系统以单一配置中枢 calibration_profile.yaml 串联离线校准与在线推荐，形成数据—校准—运行—反馈的端到端闭环。
- **Content**:
  - 架构图（fig1_closed_loop.png，完整呈现）：真实数据 → 画像生成 → 参数校准（EM+Optuna）→ 在线仿真（Feed 排序 / Agent 决策）→ 验证反馈（DTW / Pearson / MAE / 消融）。
  - 底部论断（强调）：单一配置中枢——calibration_profile.yaml 连接离线校准与在线推荐服务。

#### Slide 10 - 核心模块一：用户画像生成

- **Layout**: 左文右图（asymmetric 4:6 或左文 + 右 fig2）——左侧输入与五阶段流水线，右侧 fig2 画像生成模块图。
- **Title**: 核心模块一：用户画像生成
- **Core message**: 画像生成模块以五阶段 LLM 流水线把原始社交数据转为可被 ABM 与 Agent 消费的结构化画像，并按影响力分层控制成本。
- **Content**:
  - 输入与特征：user/post 原始数据；发帖、互动、活跃时段、粉丝数；帖子影响力与账号影响力指标；高质量证据帖筛选。
  - 五阶段 LLM 流水线：证据抽取 → 稳定画像 → 行为画像 → Agent 画像压缩 → 模拟初始状态。
  - 右侧配 fig2_portrait.png（完整呈现）。
  - 成本控制（强调）：L4-L5 高影响力用户使用完整流水线；L1-L3 用户采用 simple profile mode 与行为库。

#### Slide 11 - 核心模块二：ABM 推荐参数校准

- **Layout**: 顶部四维评分 icon_grid + 中部 fig3 模块图 + 底部输出说明（top-bottom 三段）。
- **Title**: 核心模块二：ABM 推荐参数校准
- **Core message**: 参数校准模块用向量化 ABM 拟合真实传播结果，通过 EM 联合校准四维推荐权重与时间衰减，输出可直接注入在线服务的配置。
- **Visualization**: icon_grid（四维评分，见 VII）
- **Content**:
  - 推荐可见概率由四类分数加权得到：w_i 兴趣匹配 · w_pop 内容热度 · w_time 时间衰减 · w_rand 随机探索。
  - 配 fig3_abm.png（完整呈现）。
  - EMCalibrationEngine：E 步搜索内容级 p_base；M 步在 Dirichlet 约束下联合搜索推荐权重与 decay_lambda。
  - 输出（强调）：calibration_profile.yaml，供在线 SocialRecSys.configure 直接注入。

#### Slide 12 - 核心模块三：在线模拟与验证

- **Layout**: 顶部 fig4 系统架构图（超宽，top band）+ 底部左右两栏（在线模拟 / 验证指标）。
- **Title**: 核心模块三：在线模拟与验证方案
- **Core message**: 在线模块以双进程架构运行校准后的推荐与 Agent 决策，并用多指标与消融实验验证校准参数是否提升拟合程度。
- **Content**:
  - 配 fig4_online.png（完整呈现，系统架构）。
  - 在线社交网络模拟：main.py 启动 Backend 与 Agent 双进程；Backend 提供 Feed 推荐、内容管理、时间控制与持久化；Agent 决策循环执行发帖、评论、点赞和转发。
  - 验证指标：功能完整性 / 参数一致性（DTW / Pearson）/ MAE / MRE / 随机种子方差 / 参数消融。
  - 验证重点（强调）：校准参数是否提升传播曲线、态度演化与互动行为分布的拟合程度。

### Part 5: 可行性分析

#### Slide 13 - 可行性

- **Layout**: 四象限/四栏卡（理论/方法/数据画像/工程），每栏图标 + 支撑研究。
- **Title**: 理论、方法、工程基础均具备
- **Core message**: 本课题在理论、方法、数据画像与工程工具四方面均有充分前人基础，可按计划开展。
- **Content**:
  - 理论基础：HiSim、FDE-LLM、OASIS 等研究证明混合仿真与平台化系统可行。
  - 方法基础：EM、Optuna、贝叶斯优化与奖励反推研究可迁移到推荐参数估计。
  - 数据与画像基础：用户画像生成与分层建模已有 Two-stage Profiling、TWICE 等研究支撑。
  - 工程基础：FastAPI、SQLite、向量检索与 LangChain 等工具链可支撑系统实现。

### Part 6: 进度安排

#### Slide 14 - 进度安排

- **Layout**: 水平时间轴（timeline），六个节点。
- **Title**: 进度安排：2026年6月至12月
- **Core message**: 按「先闭环、再优化、再实验」的节奏，六个月内完成从需求分析到论文答辩的全流程。
- **Visualization**: timeline（见 VII）
- **Content**:
  - 6-7月 需求分析与总体方案 · 7-8月 数据准备与画像模块 · 8-9月 ABM 参数校准初步实验
  - 9-10月 在线模拟系统开发联调 · 10-11月 系统实验与结果分析 · 11-12月 系统优化、论文与答辩
  - 底部论断（强调）：节奏遵循「先闭环、再优化、再实验」，确保论文写作有稳定系统与实验结果支撑。

### Part 7: 总结

#### Slide 15 - 总结

- **Closing impact**: 留给评审一句话——本课题以推荐机制校准为核心，把数据治理、参数校准、分层仿真与多指标验证连成可复用闭环；构图为单列居中论断 + 暖金规则线 + 预期贡献三标签。
- **Layout**: 单列居中，核心论断大字 + 三个预期贡献标签。
- **Title**: 总结
- **Core message**: 本课题将真实传播数据治理、ABM 参数校准、大小模型分层仿真与多指标验证连接为一个可复用的社交仿真平台闭环。
- **Content**:
  - 核心论断：本课题以推荐机制校准为核心，将真实传播数据治理、ABM 参数校准、大小模型分层仿真与多指标验证连接为一个可复用的社交仿真平台闭环。
  - 预期贡献（三标签）：现实参照 · 机制解释力 · 跨议题复用能力。

#### Slide 16 - 致谢

- **Layout**: 极简单列居中，留白主导；校徽 + 致谢语。
- **Title**: 谢谢
- **Core message**: 致谢评审并请予指正。
- **Content**:
  - 谢谢各位老师，请批评指正。
  - 汇报人：张艺鹏 · 2026年6月 · 中国传媒大学校徽

---

## X. Speaker Notes Requirements

每页一份讲稿，存入 `notes/`，文件名匹配 SVG 名。风格：正式学术口吻；含要点、时长提示、过渡语。总时长约 12-15 分钟。

---

## XI. Technical Constraints Reminder

### SVG Generation Must Follow:

1. viewBox: `0 0 1280 720`
2. 背景用 `<rect>`；透明度用 `fill-opacity` / `stroke-opacity`，`rgba()` 禁止
3. 文本换行用 `<tspan>`；禁止 `foreignObject` / `mask` / `<style>` / `class` / `textPath` / `animate*` / `script`
4. 字符写原生 Unicode（`—` `→` `·` 等）；XML 保留字 `& < >` 转义
5. 架构图按 `no-crop` 用 `preserveAspectRatio="xMidYMid meet"` 完整呈现
6. `<g opacity>` 禁止——逐元素设置 opacity

### PPT Compatibility Rules:

- 字体栈尾部均为预装字体（Cambria / SimSun / Microsoft YaHei / Arial / Consolas）
- 图片透明度用覆盖遮罩矩形，不用 group opacity
