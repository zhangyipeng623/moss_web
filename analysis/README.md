# 离线分析脚本

该目录现在提供一个统一的离线脚本入口：

- `run_analysis.py`

这个脚本包含两个子命令：

- `portrait`
  根据用户资料和历史帖子生成结构化用户画像。
- `recommender`
  根据外部传播观测数据反推推荐系统参数。

这两个能力都是离线执行的，不会在 `main.py` 运行实验时自动触发。推荐的工作流是：

1. 先离线运行脚本，生成画像文件或推荐参数文件。
2. 再把这些产物路径写入实验配置或手动同步到运行代码。

## 运行方式

统一入口：

```bash
uv run python -m analysis.run_analysis <子命令> ...
```

### 1. 生成用户画像

命令示例：

```bash
uv run python -m analysis.run_analysis portrait \
  --data-path /path/to/group_data \
  --user-name agent_alpha \
  --reference-time "2026-04-01 12:00:00" \
  --output analysis_outputs/portraits/agent_alpha.json \
  --model gpt-4o
```

如果不写 `--output`，默认输出到：

```text
analysis_outputs/portraits/<user_name>.json
```

#### 用户需要提供的数据结构

画像脚本默认只按 Excel/CSV 数据目录取数，不再走画像 JSON 输入；用户关注话题会由模型根据历史帖子自动归纳。

你只需要提供一个数据目录，目录中至少包含：

- `user.xlsx`
- `post.xlsx`

命令里再提供一个目标用户名：

- `--data-path`
- `--user-name`
- `--reference-time`
  建议显式提供。若不提供，脚本会在运行时交互询问参考时间；不带时区时默认按 `Asia/Shanghai` 解析。

这套列名约定对齐旧版
`/Users/zhangyipeng/ZYPRoom/cuc/project/node_llm/src/utils/data.py`，
脚本会按同样的标签名读取。

`user.xlsx` 或 `user.csv` 至少应包含这些列：

- `用户名`
- `昵称`
- `简介`
- `性别`
- `地域`
- `关注`
- `粉丝`
- `收藏`
- `源用户名`
- `用户地址`
- `创建时间戳`
- `头像链接`

`post.xlsx` 或 `post.csv` 至少应包含这些列：

- `用户名`
- `发文内容`
- `发布时间`
- `发布时间戳`
- `发文类型`

`post.xlsx` 推荐额外包含这些列：

- `点赞数`
- `评论数`
- `转发数`
- `帖子ID` 或 `推文ID`

脚本会自动做这些事：

- 从 `user.xlsx` 中找到指定 `用户名` 的用户资料
- 从 `post.xlsx` 中筛出该用户全部帖子
- 如果同名用户有多行，优先选择字段最完整的一行
- 帖子会按时间倒序整理，并按 `帖子ID` 或内容时间组合去重
- 直接根据用户历史帖子正文和帖子互动统计自动归纳潜在关注话题，不依赖 `topics.json`

#### 生成结果会存在哪里

- 如果显式传了 `--output`，就写到你指定的位置。
- 如果没有传 `--output`，默认写到 `analysis_outputs/portraits/<user_name>.json`。
- 如果画像生成连续 3 次失败，不会输出该用户画像，而会把失败用户名单写到同目录下的 `failed_users.json`。

#### 生成结果如何具体使用

画像输出文件中最重要的字段包括：

- `stable_profile`
- `behavior_profile`
- `agent_profile`
- `simulation_init`
- `identity_summary`
- `interest_summary`
- `value_summary`
- `style_summary`
- `behavior_summary`
- `interaction_summary`
- `speaking_rules_text`
- `action_rules_text`
- `avoidance_rules_text`

在真正运行系统时，推荐这样使用：

1. 把生成出来的画像 JSON 路径写入 `agents.csv`。
2. 把该 Agent 的 `profile_mode` 设为 `default`。
3. 把 `profile_path` 指向刚生成的画像文件。

示例 `agents.csv`：

```csv
username,name,bio,profile_mode,profile_path
agent_alpha,Alpha,我是军事议题分析用户,default,../../analysis_outputs/portraits/agent_alpha.json
```

在这种模式下，系统会直接使用 Agent 内置的固定模板来渲染这些结构化字段。

如果你希望完全使用自己的画像模板，可以在 `agents.csv` 中额外配置：

- `profile_mode=custom`
- `user_info_template`
- 或 `user_info_template_path`

### 2. 反推推荐系统参数

命令示例：

```bash
uv run python -m analysis.run_analysis recommender \
  --data-file /path/to/post_original_4.xlsx \
  --portraits-dir analysis_outputs/portraits/ \
  --output analysis_outputs/recommender/post_original_4.json
```

如果不写 `--output`，默认输出到：

```text
analysis_outputs/recommender/<输入文件名>.json
```

#### 用户需要提供的输入结构

推荐参数脚本需要两类输入：

1. **内容观测表**（`--data-file`）：支持 `.xlsx` 或 `.csv`，默认读取以下列：
   - `文章ID`、`观看量`、`转发`、`分享`、`Quotes`
   - 如列名不同，可通过 `--retweet-columns`、`--view-column`、`--id-column` 覆盖

2. **用户画像目录**（`--portraits-dir`）：由 `portrait --batch` 生成的画像 JSON 目录，用于语义处理（立场轴计算、兴趣匹配）和种群扩增。种子用户代表总人口的 10%（对应 90:9:1 法则），ABM 规模自动按 `种子数 × 10` 计算。

如果你的列名和默认值不同，可以显式覆盖：

```bash
uv run python -m analysis.run_analysis recommender \
  --data-file /path/to/your_posts.xlsx \
  --portraits-dir analysis_outputs/portraits/ \
  --retweet-columns 转发数,分享数,引用数 \
  --view-column 曝光量 \
  --id-column 帖子ID
```

常用可调参数：

- `--portraits-dir`
  用户画像 JSON 目录路径，用于立场轴和兴趣匹配（推荐提供）
- `--anchor-percentile`
  鲁棒缩放锚点分位数，默认 `0.8`
- `--max-iterations`
  EM 校准最大迭代次数，默认 `3`
- `--num-agents`
  ABM 代理数，未指定时按 `种子数 × 10` 自动推算
- `--min-scaled-target`
  最小缩放目标，默认 `5`
- `--embedding-model`
  文本嵌入模型，默认 `BAAI/bge-m3`
- `--n-cpu`
  并行校准使用的 CPU 核心数，默认 `4`

#### 新版流程说明

新版采用**向量化 ABM + Optuna EM**引擎（对应 `social_recommender_system/test.py` v8），与旧版的关键差异：

1. **语义处理**：使用 Kernel PCA (RBF) 将用户画像嵌入投影到一维立场轴，结合话题嵌入计算兴趣匹配度
2. **种群合成**：种子用户（L3-L5）通过精英保留 + Zipf 影响力分布扩展为全量仿真种群，替代旧版的优先连接图
3. **Soft Backfire**：信念更新引入回火效应 — 立场冲突时，极端用户有概率反而强化原有立场
4. **Optuna 搜索**：E 步和 M 步均使用 Optuna 进行超参优化，替代旧版随机采样

#### 生成结果如何具体使用

输出文件中最重要的字段包括：

- `representative_stories`
- `calibrated_probs`（每条内容的 `p_base` 概率）
- `best_weights`（`w_i`、`w_pop`、`w_time`、`w_rand`）
- `weight_fit_diagnostics`

当前系统运行时还不会自动加载这个结果文件，所以它的使用方式是手动同步：

1. 打开生成结果中的 `best_weights`。
2. 将权重映射到 [social_recsys.py](/Users/zhangyipeng/ZYPRoom/cuc/project/moss_web/backend/services/social_recsys.py) 顶部的权重常量：

| best_weights 字段 | 对应常量 | 含义 |
|-------------------|----------|------|
| `w_i` | `W_BELIEF` | 兴趣/立场匹配度 |
| `w_pop` | `W_POP` | 源用户影响力 |
| `w_time` | `W_CHRONO` | 时间衰减 |
| `w_rand` | `W_RAND` | 随机探索 |

## 与运行系统的关系

当前的推荐用法是：

- 画像脚本生成 `portrait JSON`
- 运行系统通过 `agents.csv` 的 `profile_mode=default` 和 `profile_path` 消费画像
- 推荐参数脚本生成 `recommender JSON`
- 运行系统目前需要手动把 `best_weights` 同步到推荐服务代码

也就是说：

- 脚本本身是离线工具
- 生成的数据才是运行系统真正消费的输入
