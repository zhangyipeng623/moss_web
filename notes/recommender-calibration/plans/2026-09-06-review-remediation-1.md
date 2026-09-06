# 推荐实验审查补救计划

状态：用户已授权修复；当前 agent 本地串行，无 subagent。
原计划：[实现计划](2026-09-06-implementation.md)。发现来源：本任务对当前实现的代码审查及缩放/CSV 复现。

## 发现与处置

1. P1：比较未核对模型的 manifest、训练散列和尺度，可能混用数据包；在加载嵌入前拒绝不一致输入。
2. P2：绝对达标率改善方向相反；改为候选减对照。
3. P2：敏感性条件没有逐条导出；每个唯一条件保存两组明细，与主实验相同的概率复用计算。
4. P2：公共训练损失忽略 n_cpu；复用固定参数并行模拟入口，保持全量、串并行相同随机流和独立状态。

## 修改范围

- analysis/compare_recommenders.py
- analysis/recommender_parameter_inference.py
- tests/test_recommender_comparison.py（修正虚假模型指纹测试数据）
- tests/test_recommender_regressions.py（新增行为回归）

## 执行与验证

先新增不一致输入、改善方向、网格逐条覆盖、并行入口及串并行一致性测试，观察失败；再做最小修复。运行 `.venv/bin/python -m unittest discover -s tests -p 'test_recommender_*.py' -q` 与相关 py_compile。检查最终差异只涉及声明范围和 Notes，重新本地审查四项根因，无数据修改，不运行完整真实数据实验。

## 验收结论

四项修复完成。新增回归在修复前出现 5 个失败断言，修复后全套 57 项测试通过（含随机引擎串行/双进程同结果），相关 py_compile 与限定范围 diff --check 通过。本地最终缺陷审查无新增发现，复杂度审查 lean：复用固定参数模拟器，删除重复损失循环，无新增依赖。按用户要求未使用独立 reviewer/subagent；此次结论限于上述四项修复，不代表真实数据实验完成。

比较数据包必须与训练时 manifest 完全一致，不能手工编辑测试后继续使用旧模型。CSV 的 main 条件包含主实验，其他唯一概率条件使用 p_base=<值>；与主实验相同的网格概率复用 main 行。训练随机流改为稳定推文 ID 派生，同版本串并行一致；与修复前按位置派生的随机流不同，不承诺旧训练数值完全复现。
