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
  --output analysis_outputs/recommender/post_original_4.json
```

如果不写 `--output`，默认输出到：

```text
analysis_outputs/recommender/<输入文件名>.json
```

#### 用户需要提供的输入结构

推荐参数脚本默认按旧版推荐系统的 Excel 方式取数，不需要你先手工整理成 JSON。

你只需要提供一个内容观测表文件：

- `--data-file /path/to/post_original_4.xlsx`

这个表可以是 `.xlsx`，也可以是 `.csv`。默认会按旧版
`/Users/zhangyipeng/ZYPRoom/cuc/project/social_recommender_system/social_recommender_system8.py`
里的口径读取以下列：

- `文章ID`
- `观看量`
- `转发`
- `分享`
- `Quotes`

脚本会自动做这些事：

- 读取整张表
- 把 `转发 + 分享 + Quotes` 汇总成总转发量
- 过滤 `总转发量 > 0` 且 `观看量 > 100` 的内容
- 生成 `story_id / repost_count / view_count` 观测记录
- 再进入代表内容筛选和推荐参数反推流程
- 反推阶段使用终态多目标拟合，同时约束缩放后的 `曝光量`、`转发量` 和 `转发率`
- 模拟器固定 `p_online=0.1`、固定 `duration=24` 小时，`decay_lambda` 继续参与搜索

如果你的列名和默认值不同，可以显式覆盖：

```bash
uv run python -m analysis.run_analysis recommender \
  --data-file /path/to/your_posts.xlsx \
  --retweet-columns 转发数,分享数,引用数 \
  --view-column 曝光量 \
  --id-column 帖子ID
```

常用可调参数：

- `--anchor-percentile`
  鲁棒缩放锚点分位数，默认 `0.8`
- `--max-iterations`
  EM 校准最大迭代次数，默认 `3`
- `--num-agents`
  ABM 代理数，默认 `1500`
- `--avg-degree`
  平均连接度，默认 `20`
- `--verified-ratio`
  认证账号比例，默认 `0.01`
- `--min-scaled-target`
  最小缩放目标，默认 `3`
- `--n-trials-per-story`
  单条内容概率校准试验次数，默认 `40`
- `--n-trials-per-weight`
  权重优化试验次数，默认 `100`
- `--n-simulations-per-trial`
  每次试验的模拟次数，默认 `5`

#### 生成结果会存在哪里

- 如果显式传了 `--output`，就写到你指定的位置。
- 如果没有传 `--output`，默认写到 `analysis_outputs/recommender/<输入文件名>.json`。

#### 生成结果如何具体使用

输出文件中最重要的字段包括：

- `representative_stories`
- `calibrated_probs`
- `best_weights`
- `weight_fit_diagnostics`

当前系统运行时还不会自动加载这个结果文件，所以它的使用方式是手动同步：

1. 打开生成结果中的 `best_weights`。
2. 读取其中的 `chrono`、`belief`、`pop`、`rand`。
3. 把这些值写入 [social_recsys.py](/Users/zhangyipeng/ZYPRoom/cuc/project/moss_web/backend/services/social_recsys.py) 顶部的权重常量。

当前对应位置是：

- `W_CHRONO`
- `W_BELIEF`
- `W_POP`
- `W_RAND`

注意：

- `best_weights` 里的 `decay_lambda` 目前是离线推断结果，当前在线推荐实现还没有直接消费这个字段。
- 如果后续你想让系统自动读取推荐参数文件，再单独把这个 JSON 文件接进配置层即可。

## 与运行系统的关系

当前的推荐用法是：

- 画像脚本生成 `portrait JSON`
- 运行系统通过 `agents.csv` 的 `profile_mode=default` 和 `profile_path` 消费画像
- 推荐参数脚本生成 `recommender JSON`
- 运行系统目前需要手动把 `best_weights` 同步到推荐服务代码

也就是说：

- 脚本本身是离线工具
- 生成的数据才是运行系统真正消费的输入
