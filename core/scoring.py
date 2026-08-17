"""推荐打分共享模块：离线 ABM 与在线 SocialRecSys 共用同一份四维打分逻辑。

目的（见 docs/plan/在线仿真与参数一致性方案.md Part B）：
从根上杜绝离线校准与在线打分的公式漂移——两侧 import 同一份
interest/popularity/time/random 计算与归一化逻辑。

四维原始分数契约（两侧一致）：
- interest    : 余弦相似度（可叠加立场亲和分量），裁剪到 [0,1]
- popularity  : log1p(传播/互动人数) / log1p(总人口 N) ∈ [0,1]，再乘 TIER_WEIGHT[tier]
                （tier 放大后允许 >1，与在线侧批内 Min-Max 归一化配合）
- time        : exp(-decay_lambda * dt_hours) ∈ [0,1]，dt 单位统一为小时
- random      : [0,1] 均匀随机

归一化契约：
- 所有维度先构造到同尺度（[0,1]），再由调用方按候选批做 Min-Max 归一化；
- ABM 中 popularity/time 在单个仿真步内是标量（全体候选共享），对其按批
  Min-Max 会退化为常数 0.5 并抹掉跨步动态，因此这两维以构造归一化（∈[0,1]）
  为准，interest/random 为逐候选维度按批归一化；在线侧四维均为逐候选维度，
  全部按批 Min-Max 归一化。两侧原始分数公式完全一致，见
  analysis/recommender_parameter_inference.py 与 backend/services/social_recsys.py。
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np

# Rogers 5 级影响力预设权重（不参与校准）。
# 待实现工作方案 A-2：除以 max 归一化到 (0,1]，使 pop 维与 interest/time/random
# 同尺度 ∈[0,1]，四维权重的解释可比（原 5 级 2.0 会让 pop 维最高 ~2.0）。
TIER_WEIGHT_DEFAULT: Dict[int, float] = {1: 0.2, 2: 0.35, 3: 0.5, 4: 0.75, 5: 1.0}


def interest_score(
    cosine_similarity: float,
    stance_affinity: float = 0.0,
) -> float:
    """兴趣分：余弦相似度叠加可选立场亲和分量，裁剪到 [0,1]。

    ABM 侧 stance_affinity = |stance| * 0.1（立场投影基线）；在线侧暂未建模
    立场轴，传 0.0。
    """
    return float(np.clip(float(cosine_similarity) + float(stance_affinity), 0.0, 1.0))


def popularity_base(reach_count: float, population_size: int) -> float:
    """热度基数：log1p(传播/互动人数) / log1p(总人口)，裁剪到 [0,1]，不含 tier 放大。

    - ABM 侧：reach_count = 累计感染人数（≤ N，自然 ∈[0,1]）；
    - 在线侧：reach_count = 加权互动计数，population_size = 注册用户数；
      加权互动计数可能 > 注册用户数（同一用户多次互动），此处 clip 保证 ≤1。
    """
    population = max(int(population_size), 1)
    raw = math.log1p(max(float(reach_count), 0.0)) / math.log1p(population)
    return float(min(1.0, raw))


def popularity_score(
    reach_count: float,
    population_size: int,
    tier: int = 3,
    tier_weight: Optional[Dict[int, float]] = None,
) -> float:
    """热度分：popularity_base × TIER_WEIGHT[tier]。

    - ABM 侧：tier = 传播源 tier 均值系数（avg_tier_coeff）；
    - 在线侧：tier = 作者 tier。
    """
    weights = tier_weight or TIER_WEIGHT_DEFAULT
    return popularity_base(reach_count, population_size) * weights.get(
        int(tier), 1.0
    )


def time_decay_score(dt_hours: float, decay_lambda: float) -> float:
    """时效分：exp(-decay_lambda * dt_hours)。

    两侧 dt 单位必须一致：小时。ABM 侧通过 hours_per_step 把仿真步换算为小时。
    """
    return float(math.exp(-float(decay_lambda) * max(float(dt_hours), 0.0)))


def cosine_from_vec_distance(distance: float) -> float:
    """把 sqlite-vec 的 cosine 距离（1 - cosθ）还原为余弦相似度，裁剪到 [0,1]。"""
    return float(np.clip(1.0 - float(distance), 0.0, 1.0))


def min_max_normalize(
    values: Sequence[float] | np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Per-Batch Min-Max 归一化到 [0,1]；max==min 时返回 0.5。"""
    arr = np.asarray(values, dtype=float)
    v_min = float(arr.min())
    v_max = float(arr.max())
    if v_max - v_min < eps:
        return np.full_like(arr, 0.5)
    return (arr - v_min) / (v_max - v_min + eps)


def weighted_score(
    norm_interest: float,
    norm_popularity: float,
    norm_time: float,
    norm_random: float,
    w_interest: float,
    w_popularity: float,
    w_time: float,
    w_random: float,
) -> float:
    """四维加权总分（归一化后的维度线性加权）。

    维度输入可为标量或等长数组（ABM 向量化按候选批计算时传入数组，
    numpy 自动广播；在线侧传标量）。
    """
    return (
        float(w_interest) * norm_interest
        + float(w_popularity) * norm_popularity
        + float(w_time) * norm_time
        + float(w_random) * norm_random
    )


def normalized_weight_vector(
    w_interest: float,
    w_popularity: float,
    w_time: float,
    w_random: float,
) -> Dict[str, float]:
    """把四个原始权重归一化为和为 1 的权重向量（Optuna 搜索结果回填用）。"""
    total = (
        float(w_interest) + float(w_popularity) + float(w_time) + float(w_random)
    )
    if total < 1e-9:
        raise ValueError("四个权重之和为 0，无法归一化。")
    return {
        "w_i": float(w_interest) / total,
        "w_pop": float(w_popularity) / total,
        "w_time": float(w_time) / total,
        "w_rand": float(w_random) / total,
    }
