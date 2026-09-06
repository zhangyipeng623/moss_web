# 数据准备 — 决策演进记录

> Status: in-progress
> 更新: 2026-09-06

## 当前定案

本地工作树已包含公开数据集转换和内容观测构建脚本，目标是产出画像及校准入口所需的数据。当前仅收录实现事实；脚本仍未跟踪，本记录不表示已完成数据质量验收或确定最终实验数据集。

| 入口 | 可见职责 |
|---|---|
| [convert_weibo63641.py](../../scripts/convert_weibo63641.py) | 读取微博 SQL，抽样用户，写画像表和内容观测 |
| [convert_pheme.py](../../scripts/convert_pheme.py) | 按事件转换 PHEME，构建时间分箱观测 |
| [convert_fakenewsnet.py](../../scripts/convert_fakenewsnet.py) | 转换 FakeNewsNet，构建画像、校准和时序输入 |
| [build_content_observations.py](../../scripts/build_content_observations.py) | 从帖子表构建校准观测 JSON |
| [generate_handle_queries.py](../../scripts/generate_handle_queries.py) | 按事件生成账号查询字符串 |

下游分别见 [用户画像](../user-portraits/README.md) 与 [推荐参数校准](../recommender-calibration/README.md)。实验是否能使用传播轨迹，需要确认观测事件的含义、完整性与采样口径，不能由脚本包含 `repost_curve` 推断。

## 演进时间线

| 时间 | 方案 | 结论 |
|---|---|---|
| 2026-09-06 | 收录当前数据转换脚本 | 采用功能级记录；实现可见，产物正确性和最终数据选择待验证；此日期仅为收录日期 |

## 实现状态

- [x] 本地存在三个数据集转换入口及观测构建、查询辅助脚本。
- [ ] 验证抽样、缺失值、去重、时间单位及转发/回复等事件口径。
- [ ] 固定实验数据版本、来源、规模和可重复转换命令。
- [ ] 保存实际数据验证结果，确认下游画像与校准能够消费产物。

本次未执行脚本或读取原始用户数据；上述待办未形成获批实现计划。下一步是在选定数据集后编写数据口径与验证方案。
