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

### 2. 反推推荐系统参数（公共概率全量校准）

新版把筛选、训练、比较拆成三个独立入口，固定数据包与模型产物，训练不读取测试结果。

**阶段 1：准备数据包**

```bash
uv run python scripts/prepare_recommender_data.py --data-file data/posts.xlsx --text-column 正文 --num-agents 1500 --output-dir analysis_outputs/datasets/example
```

产出 **train.json**、**test.json**、**manifest.json**。划分先于筛选：仅训练分区做转发量/浏览量/缩放目标过滤，测试分区保留零转发与低传播内容。默认 **--test-ratio 0.3**、**--random-seed 42**、**--selection all**（**stratified** 开启十档抽样）。

**阶段 2：训练（产出可重建模型）**

```bash
uv run python -m analysis.run_analysis recommender --train-file analysis_outputs/datasets/example/train.json --portraits-dir analysis_outputs/portraits/ --output-dir analysis_outputs/calibration/example
```

产出 **model.json** 与 **calibration_profile.yaml**。训练只用训练分区拟合公共基础概率 **p_base_global** 与四维权重/衰减，全部训练推文全量参与，不读取测试文件。

**阶段 3：同测试集比较**

```bash
uv run python -m analysis.compare_recommenders --model analysis_outputs/calibration/example/model.json --test-file analysis_outputs/datasets/example/test.json --baseline data/baseline.json --portraits-dir analysis_outputs/portraits/ --output-dir analysis_outputs/comparison/example
```

产出 **summary.json** 与 **per_story.csv**。比较模块不调用优化，两组共用同一模拟环境与 **p_base_global**，只替换系数。

**旧命令迁移**：**--data-file**、**--input** 及筛选参数已停止接受训练，会返回迁移错误，例如：

```text
旧 recommender 参数（--data-file）已迁移：请先用 scripts/prepare_recommender_data.py 准备数据包，再用 --train-file 训练。
```

**baseline.json 示例**：

```json
{
  "name": "另一系统",
  "weights": {"w_interest": 0.4, "w_popularity": 0.3, "w_time": 0.2, "w_random": 0.1},
  "decay_lambda": 0.5
}
```

四项必须有限、非负、总和大于零，导入后归一化；省略 **decay_lambda** 时共用模型衰减（比较标签为“仅权重”）。

**指标与口径**：终点是缩放后的传播总量；**mae**/**rmse** 含零目标，**mre_nonzero**/**relative_pass_rate_nonzero** 只在非零目标上计算（全零为 null）。误差达标率不是推荐准确率；**mae_difference_ci95** 用 2000 次配对 bootstrap，只描述固定训练产物在当前采样下的不确定性。逐条 CSV 零目标相对误差留空，报告 **n_clipped**（人口上限截断数）。

**研究限制**：公共概率与权重会相互补偿，比较限于同一 ABM；训练筛选偏向有传播内容，测试保留低传播内容；只有横截面总量时时间参数继续作为拟合参数，不声称辨识真实时间机制；参数替换不等于真实系统的因果效果。

## 与运行系统的关系

- 画像脚本生成画像 JSON，运行系统通过 **agents.csv** 的 **profile_mode=default** + **profile_path** 消费。
- 训练脚本生成 **model.json**（比较的权威产物）与 **calibration_profile.yaml**（供 **main.py --config** 加载）。
- 在线推荐服务仍只消费现有配置字段；公共概率不改变在线动作语义。
