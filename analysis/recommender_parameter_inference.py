from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import KernelPCA

from core.scoring import (
    TIER_WEIGHT_DEFAULT,
    min_max_normalize,
    normalized_weight_vector,
    popularity_base,
    time_decay_score,
    weighted_score,
)
from analysis.recommender_data import file_sha256

logging.getLogger("optuna").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# Tier 影响力预设权重（Rogers 5 级，不参与校准；与在线 SocialRecSys 共用同一张表）
TIER_WEIGHT: Dict[int, float] = TIER_WEIGHT_DEFAULT

# 轨迹级损失（P1-A）：DTW 距离与 Pearson 相关的混合权重
DTW_LOSS_WEIGHT = 0.5
PEARSON_LOSS_WEIGHT = 0.5


def min_max_norm(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-Batch Min-Max 归一化（core.scoring.min_max_normalize 别名，向后兼容）。"""
    return min_max_normalize(values, eps=eps)


# ============================================================
# 0. 嵌入服务
# ============================================================
class EmbeddingService:
    """基于 SentenceTransformer 的文本嵌入服务。"""

    def __init__(self, model_name: str = "Alibaba-NLP/gte-multilingual-base"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("加载嵌入模型: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
            self._model.eval()
            dim = self._model.get_sentence_embedding_dimension()
            logger.info("向量维度: %d", dim)
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            embeddings = self.model.encode(
                list(texts),
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        else:
            embeddings = self.model.encode(
                list(texts),
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        return np.asarray(embeddings)


# ============================================================
# 1. 语义处理模块
# ============================================================
@dataclass(slots=True)
class SemanticOutput:
    vectors: np.ndarray
    raw_topics: List[Dict[str, float]]
    influences: np.ndarray
    tiers: np.ndarray


class SemanticLoader:
    """从用户画像 JSON 中提取文本，计算向量嵌入。"""

    def __init__(self, embedding_service: EmbeddingService):
        self.embed = embedding_service

    def process_personas(self, user_json_list: List[Dict[str, Any]]) -> SemanticOutput:
        """将画像 JSON 列表转为向量、话题字典和影响力数组。"""
        texts: List[str] = []
        user_topics: List[Dict[str, float]] = []
        influences: List[float] = []
        tiers: List[int] = []

        logger.info("[Loader] 处理 %d 个用户画像...", len(user_json_list))
        for u in user_json_list:
            labels = u.get("label", [])
            if isinstance(labels, list):
                label_text = ",".join(str(lb) for lb in labels)
            else:
                label_text = str(labels)
            user_info = u.get("user_info", "")
            text = f"政治立场:{label_text} 个人描述:{user_info}"
            texts.append(text)

            topic = u.get("topic", {})
            if isinstance(topic, dict):
                user_topics.append({str(k): float(v) for k, v in topic.items()})
            else:
                user_topics.append({})

            inf = u.get("account_influence", 1.0)
            try:
                influences.append(float(inf))
            except (TypeError, ValueError):
                influences.append(1.0)

            # 画像层级（Rogers 1-5），缺失时按 L4 兜底
            try:
                tier = int(u.get("influence_tier", 4))
            except (TypeError, ValueError):
                tier = 4
            tiers.append(max(1, min(5, tier)))

        vectors = self.embed.embed_documents(texts)
        return SemanticOutput(
            vectors=np.asarray(vectors),
            raw_topics=user_topics,
            influences=np.asarray(influences, dtype=float),
            tiers=np.asarray(tiers, dtype=int),
        )


class AutoCompass:
    """立场轴计算与兴趣匹配。"""

    def __init__(
        self,
        user_vectors: np.ndarray,
        user_topics: List[Dict[str, float]],
        embedding_service: EmbeddingService,
    ):
        self.user_vectors = np.asarray(user_vectors)
        self.user_topics = user_topics
        self.embed = embedding_service
        self.stance_scores: np.ndarray = np.array([])
        self._topic_cache: Dict[str, np.ndarray] = {}

    def compute_stance(self) -> np.ndarray:
        """使用 Kernel PCA (RBF) 将用户向量投影到一维立场轴。"""
        logger.info("[Compass] 计算立场轴 (Kernel PCA RBF)...")
        n_samples = len(self.user_vectors)
        if n_samples < 5:
            self.stance_scores = np.zeros(n_samples)
            return self.stance_scores

        kpca = KernelPCA(
            n_components=1,
            kernel="rbf",
            gamma=None,
            fit_inverse_transform=False,
        )
        projection = kpca.fit_transform(self.user_vectors).flatten()

        median = float(np.median(projection))
        q75, q25 = np.percentile(projection, [75, 25])
        iqr = q75 - q25
        if iqr == 0:
            iqr = 1.0
        robust_z = (projection - median) / (iqr / 1.349)

        self.stance_scores = np.clip(robust_z / 3.0, -1.0, 1.0).astype(float)
        logger.info(
            "  -> 立场分布: Mean=%.2f, Std=%.2f",
            float(self.stance_scores.mean()),
            float(self.stance_scores.std()),
        )
        return self.stance_scores

    def compute_interest(self, tweet_text: str) -> np.ndarray:
        """计算单条推文与所有种子用户的兴趣匹配度。"""
        if isinstance(tweet_text, dict):
            tweet_text = str(tweet_text.get("text", tweet_text))
        tweet_vec = self.embed.embed_documents([str(tweet_text)])[0]

        # 缓存话题嵌入
        uncached: List[str] = []
        for t_dict in self.user_topics:
            for k in t_dict:
                if k not in self._topic_cache:
                    uncached.append(k)
        if uncached:
            unique_uncached = list(set(uncached))
            vecs = self.embed.embed_documents(unique_uncached)
            for k, v in zip(unique_uncached, vecs):
                self._topic_cache[k] = np.asarray(v)

        interests = np.empty(len(self.user_topics), dtype=float)
        for i, topics in enumerate(self.user_topics):
            base = abs(float(self.stance_scores[i])) * 0.1 if len(self.stance_scores) > i else 0.0
            match = 0.0
            for k, w in topics.items():
                if k in self._topic_cache:
                    sim = float(np.dot(tweet_vec, self._topic_cache[k]))
                    weighted = sim * float(w)
                    if weighted > match:
                        match = weighted
            interests[i] = float(np.clip(base + match, 0.0, 1.0))
        return interests


# ============================================================
# 2. 种群合成模块
# ============================================================
@dataclass(slots=True)
class Population:
    S: np.ndarray          # 立场值数组 (N,)
    Inf: np.ndarray        # 影响力数组 (N,)
    log_saturation_threshold: float
    source_indices: np.ndarray  # 映射到种子用户的索引
    tiers: np.ndarray      # Rogers 层级数组 (N,)，随种群扩增保存


class PopulationSynthesizer:
    """将种子用户扩展为全量仿真种群。"""

    def __init__(self, target_size: Optional[int] = None):
        self.target_size = target_size

    def synthesize(
        self,
        seed_S: np.ndarray,
        seed_Inf: np.ndarray,
        seed_tiers: Optional[np.ndarray] = None,
        seed: int = 42,
    ) -> Population:
        # 使用局部 RNG，避免全局 np.random.seed 污染后续调用（P0 附带 / P2-E）
        rng = np.random.default_rng(seed)
        current_n = len(seed_S)
        target_size = self.target_size
        if target_size is None:
            target_size = max(int(current_n / 0.16), current_n)
        if seed_tiers is None:
            seed_tiers = np.full(current_n, 4, dtype=int)
        seed_tiers = np.asarray(seed_tiers, dtype=int)
        if len(seed_tiers) != current_n:
            raise ValueError("seed_tiers 长度必须与 seed_S 一致")
        if current_n == 0:
            return Population(
                S=np.zeros(target_size),
                Inf=np.zeros(target_size),
                log_saturation_threshold=1.0,
                source_indices=np.zeros(target_size, dtype=int),
                tiers=np.zeros(target_size, dtype=int),
            )

        # 精英保留 (top 5%)
        n_elite = max(1, int(current_n * 0.05))
        elite_idx = np.argsort(seed_Inf)[::-1][:n_elite]
        real_elite_S = seed_S[elite_idx].copy()
        real_elite_Inf = seed_Inf[elite_idx].copy()

        final_S: List[float] = list(real_elite_S)
        source_indices: List[int] = list(elite_idx)
        final_tiers: List[int] = list(seed_tiers[elite_idx])
        crowd_idx = [i for i in range(current_n) if i not in elite_idx]

        while len(final_S) < target_size:
            src = int(rng.choice(crowd_idx))
            s_noise = float(rng.normal(0, 0.1))
            final_S.append(float(np.clip(float(seed_S[src]) + s_noise, -1.0, 1.0)))
            source_indices.append(src)
            final_tiers.append(int(seed_tiers[src]))

        # 合成影响力 (Zipf 分布)
        n_crowd = target_size - len(real_elite_Inf)
        final_Inf: np.ndarray
        if n_crowd > 0:
            zipf_crowd = rng.zipf(1.5, n_crowd).astype(float)
            cap_val = float(np.percentile(zipf_crowd, 99.5))
            if cap_val == 0:
                cap_val = float(zipf_crowd.max())
            zipf_crowd = np.clip(zipf_crowd, 1, cap_val)

            transition_val = float(real_elite_Inf.min())
            crowd_inf_scaled = (zipf_crowd / float(zipf_crowd.max())) * transition_val
            crowd_inf_scaled = np.maximum(crowd_inf_scaled, 1.0)

            final_Inf = np.concatenate([real_elite_Inf, crowd_inf_scaled])
        else:
            final_Inf = real_elite_Inf.copy()

        # 按影响力降序排列
        sort_idx = np.argsort(final_Inf)[::-1]
        final_Inf = final_Inf[sort_idx]

        # 同时重排 S、source_indices 和 tiers
        final_S_arr = np.array(final_S)[sort_idx]
        source_arr = np.array(source_indices, dtype=int)[sort_idx]
        tiers_arr = np.array(final_tiers, dtype=int)[sort_idx]

        raw_threshold = float(real_elite_Inf.min())
        log_saturation_threshold = float(np.log1p(raw_threshold))

        logger.info(
            "[Synthesizer] 大V影响力阈值 (Raw): %.1f, (Log): %.2f",
            raw_threshold,
            log_saturation_threshold,
        )
        return Population(
            S=final_S_arr.astype(float),
            Inf=final_Inf.astype(float),
            log_saturation_threshold=log_saturation_threshold,
            source_indices=source_arr,
            tiers=tiers_arr,
        )

    def expand_interest_for_population(
        self,
        seed_I: np.ndarray,
        source_indices: np.ndarray,
        seed: int = 42,
    ) -> np.ndarray:
        """根据 source_indices 将种子兴趣扩展到全量种群。"""
        pop_I = seed_I[source_indices].copy()
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 0.05, size=len(pop_I))
        return np.clip(pop_I + noise, 0.0, 1.0)


# ============================================================
# 3. 故事筛选模块
# ============================================================
class StoryManager:
    """从观测数据中筛选代表性内容，并缩放到 ABM 规模。"""

    def __init__(self, num_agents: int = 1500, min_scaled_target: int = 5):
        self.num_agents = num_agents
        self.min_scaled_target = min_scaled_target
        self.representative_stories: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _get_data_num(population_size: int) -> int:
        """设计效应校正后的样本量。"""
        z_score = 1.96
        p = 0.5
        deff = 1.5
        e = 0.05
        n_infinite = ((z_score**2) * p * (1 - p)) / (e**2)
        n_finite = n_infinite / (1 + ((n_infinite - 1) / max(population_size, 1)))
        return int(round(n_finite * deff))

    def select_representative_stories(
        self,
        records: Sequence[Dict[str, Any]],
        anchor_percentile: float = 0.80,
    ) -> Dict[str, Dict[str, Any]]:
        """根据记录列表筛选代表性内容。

        records 每项需包含 story_id, repost_count, view_count, text(可选)。
        """
        logger.info("Step 1: 筛选代表性推文 ...")
        if not records:
            return {}

        df = pd.DataFrame(records)
        df["总转发量"] = df["repost_count"].astype(float)
        df["view_count"] = df["view_count"].astype(float)

        # 过滤无效数据
        df = df[(df["总转发量"] > 0) & (df["view_count"] > 100)].copy()
        if df.empty:
            logger.warning("  警告：有效数据为空。")
            return {}

        # 鲁棒缩放
        target_view_anchor = float(df["view_count"].quantile(anchor_percentile))
        if target_view_anchor <= 100:
            target_view_anchor = float(df["view_count"].max())

        global_scale_ratio = self.num_agents / max(target_view_anchor, 1e-9)
        logger.info(
            "  [系统] 鲁棒缩放因子: %.6f (Anchor: %.1f)",
            global_scale_ratio,
            target_view_anchor,
        )

        df["scaled_target"] = df["总转发量"].apply(
            lambda v: min(int(round(float(v) * global_scale_ratio)), self.num_agents)
        )
        valid_df = df[df["scaled_target"] >= self.min_scaled_target].copy()
        logger.info("  原始: %d -> 有效: %d", len(df), len(valid_df))

        if valid_df.empty:
            return {}

        n_original = self._get_data_num(len(valid_df))

        # 分层抽样
        valid_df["pct_rank"] = valid_df["scaled_target"].rank(pct=True, method="first")
        quantiles = np.linspace(0, 1, 11)
        valid_df["percentile_bin"] = pd.cut(
            valid_df["pct_rank"],
            bins=quantiles,
            labels=False,
            include_lowest=True,
        )

        representative_rows: List[pd.DataFrame] = []
        bin_counts = valid_df["percentile_bin"].value_counts().sort_index()
        unique_bins = bin_counts.index.tolist()
        n_bins = max(len(unique_bins), 1)
        base_per_bin = n_original // n_bins
        remainder = n_original % n_bins

        targets: Dict[int, int] = {int(b): base_per_bin for b in unique_bins}
        sorted_bins = bin_counts.sort_values(ascending=False).index
        for i in range(remainder):
            targets[int(sorted_bins[i])] += 1

        for bin_label, group in valid_df.groupby("percentile_bin"):
            target_n = targets.get(int(bin_label), base_per_bin)
            n_group = len(group)
            take_n = min(n_group, target_n)
            sampled = group.sample(n=take_n) if n_group > take_n else group
            representative_rows.append(sampled)

        rep_df = pd.concat(representative_rows, ignore_index=True)

        self.representative_stories = {}
        for idx_val, row in rep_df.iterrows():
            story_id = str(row.get("story_id", idx_val))
            text_content = row.get("text", row.get("content", ""))
            if isinstance(text_content, float) and np.isnan(text_content):
                text_content = ""
            self.representative_stories[story_id] = {
                "real_repost": float(row["总转发量"]),
                "view_count": float(row["view_count"]),
                "scaled_target": float(row["scaled_target"]),
                "text": str(text_content),
            }

        return self.representative_stories


# ============================================================
# 4. 向量化 ABM 引擎
# ============================================================
class VectorizedABMEngine:
    """向量化仿真引擎，支持 Soft Backfire 信念更新。

    四维打分（interest/popularity/time/random）与在线 SocialRecSys 共用
    core.scoring 的公式（见 docs/plan/在线仿真与参数一致性方案.md Part B）。
    """

    def __init__(
        self,
        S: np.ndarray,
        Inf: np.ndarray,
        log_saturation_threshold: float,
        p_online: float = 0.1,
        backfire_mu: float = 0.4,
        backfire_k: float = 10.0,
        learning_rate: float = 0.1,
        tier_labels: Optional[np.ndarray] = None,
        p_online_map: Optional[Dict[int, float]] = None,
        hours_per_step: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.N = len(S)
        self.init_S = S.copy()
        self.current_S = S.copy()
        self.Inf = Inf
        self.log_saturation_threshold = max(log_saturation_threshold, 1e-9)
        self.p_online = p_online
        self.p_online_map = p_online_map
        self.backfire_mu = backfire_mu
        self.backfire_k = backfire_k
        self.learning_rate = learning_rate
        # 种子 tier 随 PopulationSynthesizer 扩增保存，传入引擎参与打分（P0-3）
        if tier_labels is not None:
            self.tier_labels = np.asarray(tier_labels, dtype=int).copy()
        else:
            self.tier_labels = np.full(self.N, 4, dtype=int)
        # 一个仿真步对应的小时数：dt_hours = 步数 × hours_per_step（B-2 时间单位统一）
        self.hours_per_step = hours_per_step
        # 引擎局部 RNG：可复现且不污染全局 np.random（P0 附带 / P2-E）
        self._rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self.state = np.zeros(self.N, dtype=bool)
        self.time = np.full(self.N, -1.0)
        self.current_S = self.init_S.copy()
        # 选取立场绝对值最大的 3 个作为初始种子
        seeds = np.argsort(np.abs(self.current_S))[-3:]
        self.state[seeds] = True
        self.time[seeds] = 0.0

    def _update_beliefs(
        self,
        target_indices: np.ndarray,
        source_indices: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Soft Backfire 信念更新。"""
        if len(target_indices) == 0:
            return
        local_rng = rng if rng is not None else self._rng
        S_tgt = self.current_S[target_indices]
        S_src = self.current_S[source_indices]

        diff_prod = S_tgt * S_src
        abs_S_tgt = np.abs(S_tgt)
        direction = S_src - S_tgt

        # 检测立场冲突
        conflict_mask = diff_prod < 0
        if conflict_mask.any():
            exponent = -self.backfire_k * (abs_S_tgt[conflict_mask] - self.backfire_mu)
            p_backfire = 1.0 / (1.0 + np.exp(exponent))
            rand_vals = local_rng.random(len(p_backfire))
            is_backfire = rand_vals < p_backfire

            conflict_local = np.where(conflict_mask)[0]
            backfire_local = conflict_local[is_backfire]
            direction[backfire_local] *= -1.0

        new_S = S_tgt + self.learning_rate * direction
        self.current_S[target_indices] = np.clip(new_S, -1.0, 1.0)

    def run_simulation(
        self,
        weights: Dict[str, float],
        p_base: float,
        I_pop: np.ndarray,
        duration: int = 24,
        alpha: float = 1.0,
        beta: float = 1.0,
        decay_lambda: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, int]:
        """向量化仿真。

        duration 单位为小时；时间衰减 dt 同样以小时计
        （dt_hours = 仿真步 × hours_per_step），与在线侧单位一致（B-2）。

        rng：可选独立随机源（P2-E 可复现）。传入时本次仿真完全由该 RNG 驱动，
        不消耗引擎内部 RNG，便于校准器按种子精确重放最优 trial。
        """
        self.reset()
        local_rng = rng if rng is not None else self._rng
        history: List[int] = []
        curr_count = int(self.state.sum())
        total_views = 0

        w_i = float(weights.get("w_i", 0.35))
        w_pop = float(weights.get("w_pop", 0.25))
        w_time = float(weights.get("w_time", 0.25))
        w_rand = float(weights.get("w_rand", 0.15))

        safe_p = np.clip(p_base, 0.001, 0.999)
        base_logit = np.log(safe_p / (1.0 - safe_p))

        for t in range(duration):
            # 逐 tier 在线概率（p_online 可为标量或 {tier: prob} 字典）
            if self.p_online_map is not None:
                probs = np.asarray(
                    [self.p_online_map.get(int(tier), 0.1) for tier in self.tier_labels],
                    dtype=float,
                )
                online_mask = local_rng.random(self.N) < probs
            else:
                online_mask = local_rng.random(self.N) < self.p_online

            active_src = self.state
            active_tgt = (~self.state) & online_mask

            if not active_tgt.any():
                history.append(curr_count)
                continue

            n_active = int(active_tgt.sum())

            # Step 1: raw scores（与 core.scoring / 在线 SocialRecSys 共用公式）
            # 兴趣：候选用户对该内容的兴趣度（已在 [0,1]，立场基线在 I_pop 中体现）
            raw_interest = I_pop[active_tgt]

            # 热度：log1p(累计传播人数)/log1p(N) × 传播源 tier 均值系数
            # （ABM 无单一作者，用传播者 tier 均值近似作者 tier，与在线侧 TIER_WEIGHT 同表）
            src_tiers = self.tier_labels[active_src]
            tier_coeffs = np.array([TIER_WEIGHT.get(int(tier), 1.0) for tier in src_tiers])
            avg_tier_coeff = float(np.mean(tier_coeffs)) if len(tier_coeffs) > 0 else 1.0
            raw_pop_scalar = popularity_base(curr_count, self.N) * avg_tier_coeff

            # 时效：exp(-decay_lambda * dt_hours)，dt 单位统一为小时（B-2）
            dt_steps = t - self.time[active_src].min() if active_src.any() else 0.0
            dt_hours = max(dt_steps, 0.0) * self.hours_per_step
            raw_time_scalar = time_decay_score(dt_hours, decay_lambda)

            raw_rand = local_rng.random(n_active)

            # Step 2: Per-Batch Min-Max 归一化（逐候选维度；pop/time 为步内标量，
            # 以构造归一化 ∈[0,1] 为准，见 core/scoring.py 归一化契约）
            norm_interest = min_max_norm(raw_interest)
            norm_rand = min_max_norm(raw_rand)

            # Step 3: Weighted sum
            scores = weighted_score(
                norm_interest,
                raw_pop_scalar,
                raw_time_scalar,
                norm_rand,
                w_i,
                w_pop,
                w_time,
                w_rand,
            )

            total_views += n_active

            visible = local_rng.random(n_active) < np.clip(scores, 0.0, 1.0)
            visible_idx = np.where(visible)[0]
            if len(visible_idx) > 0:
                tgt_indices_all = np.where(active_tgt)[0]
                actual_indices = tgt_indices_all[visible_idx]
                S_vis = self.current_S[actual_indices]

                internal_term = alpha * (np.abs(S_vis) ** 2)
                action_probs = 1.0 / (1.0 + np.exp(-(base_logit + internal_term)))

                actions = local_rng.random(len(actual_indices)) < action_probs
                new_infected = actual_indices[actions]

                if len(new_infected) > 0:
                    self.state[new_infected] = True
                    self.time[new_infected] = float(t)
                    curr_count += len(new_infected)

                    # 信念更新：按影响力加权抽样传播源，而非全局最强源（P0 附带）
                    src_indices = np.where(active_src)[0]
                    if len(src_indices) > 0:
                        src_inf = self.Inf[src_indices].astype(float)
                        total_inf = float(src_inf.sum())
                        if total_inf > 0:
                            src_probs = src_inf / total_inf
                            chosen_src = local_rng.choice(
                                src_indices, size=len(new_infected), p=src_probs
                            )
                            self._update_beliefs(new_infected, chosen_src, rng=local_rng)

            history.append(curr_count)

        return np.array(history), total_views


# ============================================================
# 5. E-M 校准引擎 (Optuna)
# ============================================================
def dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    """动态时间规整距离（平方欧氏成本，O(n*m)，用于短曲线）。"""
    n, m = len(x), len(y)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diff = float(x[i - 1] - y[j - 1])
            cost[i, j] = diff * diff + min(
                cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]
            )
    return float(cost[n, m])


def trajectory_loss(hist: np.ndarray, real_curve: Sequence[float]) -> float:
    """轨迹级损失（P1-A）：DTW 距离 + Pearson 相关混合。

    两条曲线都归一化到 [0,1]，真实曲线线性插值到仿真网格后比较形状：
    loss = DTW_LOSS_WEIGHT * DTW_norm + PEARSON_LOSS_WEIGHT * (1 - Pearson)。
    DTW 对时间错位鲁棒，Pearson 衡量趋势方向。
    """
    sim = np.asarray(hist, dtype=float)
    if sim.size < 2:
        return 0.0
    sim_norm = sim / max(float(sim.max()), 1e-9)

    real = np.asarray(real_curve, dtype=float)
    real_norm = real / max(float(real.max()), 1e-9)
    if len(real_norm) != len(sim_norm):
        real_grid = np.linspace(0.0, 1.0, len(real_norm))
        sim_grid = np.linspace(0.0, 1.0, len(sim_norm))
        real_norm = np.interp(sim_grid, real_grid, real_norm)

    dtw_norm = dtw_distance(sim_norm, real_norm) / max(
        len(sim_norm) + len(real_norm), 2
    )
    pearson = 0.0
    if sim_norm.std() > 1e-12 and real_norm.std() > 1e-12:
        pearson = float(np.corrcoef(sim_norm, real_norm)[0, 1])
        if np.isnan(pearson):
            pearson = 0.0
    return DTW_LOSS_WEIGHT * dtw_norm + PEARSON_LOSS_WEIGHT * (1.0 - pearson)


def story_scalar_loss(
    hist: np.ndarray,
    views: float,
    story: Dict[str, Any],
    mode: str,
) -> float:
    """单点标量损失（无逐时间点曲线数据时的回退）。

    mode="rate"：E 步校准 p_base，比较仿真转发率 vs 真实转发率；
    mode="count"：M 步校准权重，比较仿真终值 vs scaled_target。
    """
    if mode == "rate":
        real_repost = float(story.get("real_repost", story.get("real_retweets", 1.0)))
        real_view = float(story.get("view_count", story.get("real_views", 1.0)))
        target_rate = real_repost / max(real_view, 1e-9)
        sim_rate = float(hist[-1]) / max(float(views), 1e-9)
        return abs(sim_rate - target_rate) / max(target_rate, 1e-9)
    target_count = float(story.get("scaled_target", 1.0))
    return abs(float(hist[-1]) - target_count) / max(target_count, 1e-9)


def story_loss(
    hist: np.ndarray,
    views: float,
    story: Dict[str, Any],
    mode: str,
) -> float:
    """单条内容的仿真损失：优先轨迹级（有逐时间点曲线时），否则回退单点。"""
    real_curve = story.get("repost_curve")
    if real_curve and len(real_curve) >= 2:
        return trajectory_loss(hist, real_curve)
    return story_scalar_loss(hist, views, story, mode)


def _stable_rng_base(*parts: Any) -> int:
    """从任意组件生成跨进程稳定的整数种子（P2-E 可复现）。

    不用内置 hash()（受 PYTHONHASHSEED 影响），改用 zlib.crc32。
    """
    import zlib

    key = "|".join(str(part) for part in parts).encode("utf-8")
    return int(zlib.crc32(key))


def _run_one_fixed_simulation(
    engine: VectorizedABMEngine,
    story: Dict[str, Any],
    weights: Dict[str, float],
    p_base: float,
    duration: int,
    n_repeats: int,
    seed: int,
) -> np.ndarray:
    """固定参数下模拟单条内容，返回各次重复的终值。"""
    i_pop = np.asarray(story["I_pop"])
    decay_lambda = float(weights.get("decay_lambda", 0.5))
    story_id = story.get("story_id", "")
    rng_base = _stable_rng_base(seed, story_id)
    finals: List[float] = []
    for repeat in range(n_repeats):
        repeat_rng = np.random.default_rng(rng_base + repeat * 100003)
        hist, _ = engine.run_simulation(
            weights, p_base, i_pop, duration=duration,
            decay_lambda=decay_lambda, rng=repeat_rng,
        )
        finals.append(float(hist[-1]))
    return np.asarray(finals, dtype=float)


def run_fixed_simulations(
    engine: VectorizedABMEngine,
    stories: Sequence[Dict[str, Any]],
    weights: Dict[str, float],
    p_base: float,
    *,
    duration: int,
    n_repeats: int,
    seed: int,
    n_cpu: int = 1,
) -> np.ndarray:
    """固定参数模拟：行按 stories 顺序、列按重复序号，返回终值。

    稳定种子仅由实验种子与推文 ID 构成（不含模型名或 p_base）。
    每次 run_simulation 都重置引擎状态，任务间互不污染。
    """
    if n_cpu <= 1:
        results = [
            _run_one_fixed_simulation(engine, story, weights, p_base, duration, n_repeats, seed)
            for story in stories
        ]
    else:
        results = Parallel(n_jobs=n_cpu)(
            delayed(_run_one_fixed_simulation)(engine, story, weights, p_base, duration, n_repeats, seed)
            for story in stories
        )
    if not results:
        return np.empty((0, n_repeats), dtype=float)
    return np.vstack(results)


class EMCalibrationEngine:
    """使用 Optuna 进行 E-M 交替校准。"""

    def __init__(
        self,
        abm_engine: VectorizedABMEngine,
        stories: List[Dict[str, Any]],
        n_cpu: int = 4,
        seed: Optional[int] = None,
    ):
        self.engine = abm_engine
        self.stories = stories
        self.story_params: Dict[int, float] = {}
        self.n_cpu = n_cpu
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def _calibrate_single_story(
        self,
        story: Dict[str, Any],
        story_id: str,
        current_weights: Dict[str, float],
        duration: int = 24,
        decay_lambda: float = 0.5,
    ) -> float:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        logging.getLogger("optuna").setLevel(logging.ERROR)

        I_pop = np.asarray(story["I_pop"])

        def objective(trial: optuna.Trial) -> float:
            p = trial.suggest_float("p_base", 0.01, 0.99)
            losses: List[float] = []
            # P2-E：每次重复用确定性独立 RNG，同种子下 E 步完全可复现
            rng_base = _stable_rng_base("estep", story_id, trial.number)
            for repeat in range(5):
                repeat_rng = np.random.default_rng(rng_base + repeat * 100003)
                hist, views = self.engine.run_simulation(
                    current_weights,
                    p,
                    I_pop,
                    duration=duration,
                    decay_lambda=decay_lambda,
                    rng=repeat_rng,
                )
                losses.append(story_loss(hist, views, story, mode="rate"))
            return float(np.mean(losses))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20, show_progress_bar=False)
        return float(study.best_params["p_base"])

    def e_step(
        self,
        current_weights: Dict[str, float],
        duration: int = 24,
        decay_lambda: float = 0.5,
    ) -> None:
        logger.info("  [E-Step] 并行校准 %d 条推文...", len(self.stories))
        results = Parallel(n_jobs=self.n_cpu)(
            delayed(self._calibrate_single_story)(
                story, str(i), current_weights, duration, decay_lambda
            )
            for i, story in enumerate(self.stories)
        )
        for i, p in enumerate(results):
            self.story_params[i] = p

    @staticmethod
    def _weights_from_trial_params(params: Dict[str, float]) -> Dict[str, float]:
        """把 Optuna trial 的四个原始权重归一化为和为 1 的权重向量（P0-1）。"""
        try:
            weights = normalized_weight_vector(
                params["w_i"], params["w_pop"], params["w_time"], params["w_rand"]
            )
        except ValueError:
            return {}
        weights["decay_lambda"] = float(params["decay_lambda"])
        return weights

    def _evaluate_story(
        self,
        story: Dict[str, Any],
        p_base: float,
        weights: Dict[str, float],
        duration: int,
        n_repeats: int,
        mode: str,
        rng_base: int = 0,
    ) -> Tuple[float, List[float]]:
        """固定权重下重复仿真，返回 (平均损失, 各次终值)。

        rng_base：确定性随机种子基（P2-E）。同一 rng_base 下评估结果
        完全一致，保证 M 步回填一致性验证可以精确重放最优 trial。
        """
        I_pop = np.asarray(story["I_pop"])
        decay_lambda = float(weights.get("decay_lambda", 0.5))
        losses: List[float] = []
        finals: List[float] = []
        for repeat in range(n_repeats):
            repeat_rng = np.random.default_rng(rng_base + repeat * 100003)
            hist, _ = self.engine.run_simulation(
                weights, p_base, I_pop, duration=duration, decay_lambda=decay_lambda,
                rng=repeat_rng,
            )
            losses.append(story_loss(hist, 0.0, story, mode=mode))
            finals.append(float(hist[-1]))
        return float(np.mean(losses)), finals

    def _evaluate_weights(
        self,
        weights: Dict[str, float],
        indices: Sequence[int],
        duration: int,
        n_repeats: int,
        mode: str = "count",
        rng_base: int = 0,
    ) -> Tuple[float, List[float]]:
        """在给定内容子集上评估一组权重，返回 (平均损失, 逐内容损失)。"""
        losses: List[float] = []
        for offset, i in enumerate(indices):
            story = self.stories[i]
            p_base = self.story_params.get(i, 0.1)
            loss, _ = self._evaluate_story(
                story, p_base, weights, duration, n_repeats, mode,
                rng_base=rng_base + offset * 1000003,
            )
            losses.append(loss)
        return float(np.mean(losses)), losses

    def m_step(
        self,
        duration: int = 24,
        n_repeats: int = 5,
        n_trials: int = 50,
    ) -> Tuple[Dict[str, float], float, Dict[str, Any]]:
        """M 步：Optuna 直接搜索四个权重（P0-1），回填与最优 trial 严格一致。

        修复前：alpha_param 只控制 Dirichlet 集中度，权重靠 trial.number 随机抽，
        且回填用固定种子重建，与最优 loss 的权重不是同一组。
        修复后：四个权重直接作为搜索参数并归一化；回填用 study.best_params，
        并附一致性验证（用 best_weights 重跑，loss 应接近 study.best_value）。
        """
        logger.info("  [M-Step] 优化全局推荐权重 W + decay_lambda...")
        import optuna

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        logging.getLogger("optuna").setLevel(logging.ERROR)

        story_count = len(self.stories)
        if story_count == 0:
            raise RuntimeError("没有可校准的内容（stories 为空）。")
        # 全量遍历：每个 trial 使用全部训练内容，不再抽样限流
        fixed_indices = list(range(story_count))

        def objective(trial: optuna.Trial) -> float:
            params = {
                "w_i": trial.suggest_float("w_i", 0.0, 1.0),
                "w_pop": trial.suggest_float("w_pop", 0.0, 1.0),
                "w_time": trial.suggest_float("w_time", 0.0, 1.0),
                "w_rand": trial.suggest_float("w_rand", 0.0, 1.0),
                "decay_lambda": trial.suggest_float("decay_lambda", 0.01, 3.0, log=True),
            }
            weights = self._weights_from_trial_params(params)
            if not weights:
                return float("inf")
            indices = fixed_indices
            trial.set_user_attr("sample_indices", indices)
            # 确定性 RNG 基：同一 trial 号 + 同一采样集 → 损失完全可重放（P2-E）
            rng_base = _stable_rng_base("mstep", trial.number, tuple(indices))
            loss, _ = self._evaluate_weights(
                weights, indices, duration, n_repeats, rng_base=rng_base
            )
            return loss

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_weights = self._weights_from_trial_params(study.best_params)
        if not best_weights:
            raise RuntimeError("Optuna 搜索失败：所有 trial 的权重和均为 0。")

        # 一致性验证：用 best_params 回填的权重 + 最优 trial 的采样集与 RNG 基
        # 精确重放——返回的 best_weights 与 study.best_value 必须严格对应（P0-1）
        best_indices = study.best_trial.user_attrs.get("sample_indices") or list(
            range(story_count)
        )
        verify_rng_base = _stable_rng_base(
            "mstep", study.best_trial.number, tuple(best_indices)
        )
        verify_loss, verify_per_story = self._evaluate_weights(
            best_weights, best_indices, duration, n_repeats, rng_base=verify_rng_base
        )
        diagnostics: Dict[str, Any] = {
            "verify_loss": float(verify_loss),
            "study_best_value": float(study.best_value),
            "loss_diff_abs": float(abs(verify_loss - study.best_value)),
            "loss_diff_rel": float(
                abs(verify_loss - study.best_value) / max(abs(study.best_value), 1e-9)
            ),
            "n_trials": n_trials,
            "best_trial_number": int(study.best_trial.number),
            "sampler": type(study.sampler).__name__,
            "n_repeats": n_repeats,
            "sample_size": story_count,
            "story_count": story_count,
            "fixed_sample_indices": [int(i) for i in fixed_indices],
            "trajectory_loss_used": any(
                bool(story.get("repost_curve")) for story in self.stories
            ),
            "verify_per_story_loss": [float(v) for v in verify_per_story],
        }
        if diagnostics["loss_diff_rel"] > 0.3:
            logger.warning(
                "  [M-Step] 一致性验证偏差较大（相对 %.2f%%），请检查 best_weights 与 best_value 的对应关系。",
                diagnostics["loss_diff_rel"] * 100,
            )
        return best_weights, float(study.best_value), diagnostics

    def run_ablation(
        self,
        weights: Dict[str, float],
        duration: int = 24,
        n_repeats: int = 5,
    ) -> Dict[str, Any]:
        """消融实验（P1-B）：依次去掉某一维权重，观察损失退化。

        去掉某维后其余维度重新归一化；loss 上升越多说明该维度贡献越大，
        直接支撑开题报告“推荐参数消融实验”。
        """
        logger.info("  [Ablation] 逐维消融验证各权重贡献...")
        story_count = len(self.stories)
        # 全量遍历：消融评估同样使用全部内容，不抽样限流
        indices = list(range(story_count))

        base_rng = _stable_rng_base("ablation-base", tuple(indices))
        base_loss, _ = self._evaluate_weights(
            weights, indices, duration, n_repeats, rng_base=base_rng
        )
        result: Dict[str, Any] = {"base_loss": float(base_loss), "ablations": {}}
        dims = [
            ("w_i", "interest"),
            ("w_pop", "popularity"),
            ("w_time", "time"),
            ("w_rand", "random"),
        ]
        for key, label in dims:
            ablated = dict(weights)
            ablated[key] = 0.0
            total = (
                float(ablated.get("w_i", 0.0))
                + float(ablated.get("w_pop", 0.0))
                + float(ablated.get("w_time", 0.0))
                + float(ablated.get("w_rand", 0.0))
            )
            if total < 1e-9:
                continue
            for dim_key in ("w_i", "w_pop", "w_time", "w_rand"):
                ablated[dim_key] = float(ablated[dim_key]) / total
            # 消融评估与基线使用相同采样集与同一随机流（隔离权重贡献，P2-E 可比）
            loss, _ = self._evaluate_weights(
                ablated, indices, duration, n_repeats, rng_base=base_rng
            )
            delta = float(loss - base_loss)
            result["ablations"][key] = {
                "label": label,
                "loss": float(loss),
                "delta": delta,
                "delta_rel": float(delta / max(abs(base_loss), 1e-9)),
            }
            logger.info(
                "  [Ablation] 去掉 %s 后 loss=%.4f（Δ%.4f）",
                label,
                loss,
                delta,
            )
        return result

    def run_em_loop(
        self,
        iterations: int = 3,
        duration: int = 24,
        n_repeats: int = 5,
        n_trials: int = 50,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        weights: Dict[str, float] = {
            "w_i": 0.35, "w_pop": 0.25, "w_time": 0.25, "w_rand": 0.15, "decay_lambda": 0.5,
        }
        iteration_diagnostics: List[Dict[str, Any]] = []
        logger.info("\n=== 启动 E-M (Optuna) 校准循环 (Max %d 轮) ===", iterations)
        for k in range(iterations):
            logger.info("\n--- Iteration %d ---", k + 1)
            e_step_weights = {
                key: value for key, value in weights.items() if key != "decay_lambda"
            }
            self.e_step(
                e_step_weights,
                duration=duration,
                decay_lambda=float(weights.get("decay_lambda", 0.5)),
            )
            weights, loss, diagnostics = self.m_step(
                duration=duration,
                n_repeats=n_repeats,
                n_trials=n_trials,
            )
            diagnostics["iteration"] = k + 1
            iteration_diagnostics.append(diagnostics)
            logger.info("  >>> Iteration %d 最佳参数: %s", k + 1, weights)
            logger.info("  >>> Global MRE: %.4f", loss)
        return weights, {"iterations": iteration_diagnostics}

    def _global_count_loss(
        self,
        p_base: float,
        weights: Dict[str, float],
        *,
        duration: int,
        n_repeats: int,
        rng_base: int,
    ) -> float:
        """统一缩放计数损失：mean_i(mean_r(|final - scaled_target| / N))。

        所有训练内容等权，每次评估完整遍历全部 stories，不使用轨迹损失。
        每条内容与重复序号使用稳定种子（不含 trial 号），同一参数下损失确定。
        """
        if not self.stories:
            raise RuntimeError("没有可校准的内容（stories 为空）。")
        decay_lambda = float(weights.get("decay_lambda", 0.5))
        losses: List[float] = []
        for offset, story in enumerate(self.stories):
            i_pop = np.asarray(story["I_pop"])
            target = float(story.get("scaled_target", 0.0))
            story_rng_base = rng_base + offset * 1000003
            finals: List[float] = []
            for repeat in range(n_repeats):
                repeat_rng = np.random.default_rng(story_rng_base + repeat * 100003)
                hist, _ = self.engine.run_simulation(
                    weights, p_base, i_pop, duration=duration,
                    decay_lambda=decay_lambda, rng=repeat_rng,
                )
                final = float(hist[-1])
                if not np.isfinite(final):
                    raise RuntimeError(
                        f"仿真终值非有限：story={story.get('story_id', offset)}"
                    )
                finals.append(final)
            per_story = float(np.mean([abs(f - target) for f in finals])) / max(
                self.engine.N, 1
            )
            losses.append(per_story)
        return float(np.mean(losses))

    def _optimize_p_base(
        self,
        weights: Dict[str, float],
        duration: int,
        n_repeats: int,
        p_trials: int,
        rng_base: int,
    ) -> Tuple[float, float]:
        """固定权重，优化公共基础概率 p_base。"""
        import optuna

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        logging.getLogger("optuna").setLevel(logging.ERROR)

        def objective(trial: optuna.Trial) -> float:
            p = trial.suggest_float("p_base", 0.001, 0.999)
            return self._global_count_loss(
                p, weights, duration=duration, n_repeats=n_repeats, rng_base=rng_base
            )

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )
        study.optimize(objective, n_trials=p_trials, show_progress_bar=False)
        return float(study.best_params["p_base"]), float(study.best_value)

    def _optimize_weights(
        self,
        p_base: float,
        duration: int,
        n_repeats: int,
        weight_trials: int,
        rng_base: int,
    ) -> Tuple[Dict[str, float], float]:
        """固定公共概率，优化四维权重与衰减参数。"""
        import optuna

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        logging.getLogger("optuna").setLevel(logging.ERROR)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "w_i": trial.suggest_float("w_i", 0.0, 1.0),
                "w_pop": trial.suggest_float("w_pop", 0.0, 1.0),
                "w_time": trial.suggest_float("w_time", 0.0, 1.0),
                "w_rand": trial.suggest_float("w_rand", 0.0, 1.0),
                "decay_lambda": trial.suggest_float("decay_lambda", 0.01, 3.0, log=True),
            }
            weights = self._weights_from_trial_params(params)
            if not weights:
                return float("inf")
            return self._global_count_loss(
                p_base, weights, duration=duration, n_repeats=n_repeats, rng_base=rng_base
            )

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )
        study.optimize(objective, n_trials=weight_trials, show_progress_bar=False)
        best_weights = self._weights_from_trial_params(study.best_params)
        if not best_weights:
            raise RuntimeError("Optuna 搜索失败：所有 trial 的权重和均为 0。")
        return best_weights, float(study.best_value)

    @staticmethod
    def _select_best_round(round_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """按损失选择最佳轮次（成对返回该轮的 p_base 与 weights）。"""
        if not round_records:
            raise RuntimeError("没有任何校准轮次记录。")
        return min(round_records, key=lambda r: float(r["loss"]))

    def run_global_calibration(
        self,
        iterations: int = 3,
        duration: int = 24,
        n_repeats: int = 5,
        p_trials: int = 20,
        weight_trials: int = 50,
    ) -> Dict[str, Any]:
        """公共概率与推荐参数的交替校准（全量训练）。

        每轮先固定权重优化公共概率 p_base，再固定 p_base 优化四维权重与衰减；
        两步共用同一缩放计数损失。返回全轮最佳成对参数，并按相同种子重放核对。
        """
        if iterations < 1:
            raise ValueError("iterations 必须大于 0。")
        if p_trials < 1 or weight_trials < 1:
            raise ValueError("p_trials/weight_trials 必须大于 0。")
        if n_repeats < 1:
            raise ValueError("n_repeats 必须大于 0。")
        if not self.stories:
            raise RuntimeError("没有可校准的内容（stories 为空）。")

        base_seed = self.seed if self.seed is not None else 0
        rng_base = _stable_rng_base("global", base_seed)

        weights: Dict[str, float] = {
            "w_i": 0.35,
            "w_pop": 0.25,
            "w_time": 0.25,
            "w_rand": 0.15,
            "decay_lambda": 0.5,
        }
        p_base = 0.1
        round_records: List[Dict[str, Any]] = []
        for k in range(iterations):
            t0 = time.time()
            p_base, _ = self._optimize_p_base(
                weights, duration, n_repeats, p_trials, rng_base
            )
            weights, _ = self._optimize_weights(
                p_base, duration, n_repeats, weight_trials, rng_base
            )
            round_loss = self._global_count_loss(
                p_base, weights, duration=duration, n_repeats=n_repeats, rng_base=rng_base
            )
            round_records.append(
                {
                    "iteration": k + 1,
                    "p_base": float(p_base),
                    "weights": dict(weights),
                    "loss": float(round_loss),
                    "n_stories": len(self.stories),
                    "elapsed_seconds": time.time() - t0,
                }
            )
            logger.info(
                "  [Global] 第 %d 轮 loss=%.6f p_base=%.4f", k + 1, round_loss, p_base
            )

        best = self._select_best_round(round_records)
        best_weights = dict(best["weights"])
        best_p_base = float(best["p_base"])
        replay_loss = self._global_count_loss(
            best_p_base, best_weights, duration=duration, n_repeats=n_repeats, rng_base=rng_base
        )
        return {
            "weights": best_weights,
            "p_base_global": best_p_base,
            "loss": float(best["loss"]),
            "diagnostics": {
                "rounds": round_records,
                "best_iteration": int(best["iteration"]),
                "best_loss": float(best["loss"]),
                "replay_loss": float(replay_loss),
                "loss_name": "mean_abs_scaled_count_error",
                "n_stories": len(self.stories),
                "population_size": int(self.engine.N),
                "duration": int(duration),
                "n_repeats": int(n_repeats),
                "seed": base_seed,
            },
        }


# ============================================================
# 6. 推荐参数反推器（统一入口）
# ============================================================
class RecommendationParameterInferer:
    """推荐系统参数反推的顶层编排器。

    工作流程：
    1. 加载用户画像 → 语义处理 → 种群扩增
    2. 筛选代表性内容 → 预计算兴趣向量
    3. 启动 E-M 校准循环 → 输出最优权重
    """

    def __init__(
        self,
        num_agents: int = 1500,
        min_scaled_target: int = 5,
        p_online: float | Dict[int, float] = 0.1,
        embedding_model: str = "Alibaba-NLP/gte-multilingual-base",
        n_cpu: int = 4,
        target_size_for_sampling: Optional[int] = None,
        random_seed: Optional[int] = None,
        time_scale: float = 3600.0,
    ):
        self.num_agents = num_agents
        if target_size_for_sampling is None:
            target_size_for_sampling = num_agents
        self.min_scaled_target = min_scaled_target
        # p_online 可为标量或 {tier: prob} 分层字典（P0-3 逐 tier 在线概率）
        self.p_online = p_online
        self.n_cpu = n_cpu
        # 全链路可复现种子（P2-E）：种群合成 / 引擎 / M 步内容抽样共用
        self.random_seed = random_seed
        # A-1：实验 system_time.time_scale（每步秒数），单一真值源；
        # ABM hours_per_step = time_scale / 3600，dt 两侧统一到小时
        self.time_scale = float(time_scale)

        # 延迟初始化
        self._embed_service: Optional[EmbeddingService] = None
        self._compass: Optional[AutoCompass] = None
        self._population: Optional[Population] = None
        self._engine: Optional[VectorizedABMEngine] = None

        self.story_manager = StoryManager(
            num_agents=target_size_for_sampling,
            min_scaled_target=min_scaled_target,
        )
        self.representative_stories: Dict[str, Dict[str, Any]] = {}
        self.calibrated_probs: Dict[str, Dict[str, float]] = {}
        self.best_weights: Dict[str, float] = {}
        self.weight_fit_diagnostics: Dict[str, Dict[str, float]] = {}

        # C-1 留出验证：EM 只用 train，最终指标在 test（模型未见内容）上算
        self.train_story_ids: List[str] = []
        self.test_story_ids: List[str] = []
        self._calibration_story_ids: List[str] = []

        self._embedding_model_name = embedding_model

    @property
    def embed_service(self) -> EmbeddingService:
        if self._embed_service is None:
            self._embed_service = EmbeddingService(
                model_name=self._embedding_model_name
            )
        return self._embed_service

    def load_portraits(self, portrait_json_list: List[Dict[str, Any]]) -> None:
        """加载用户画像，执行语义处理和种群扩增。"""
        loader = SemanticLoader(self.embed_service)
        semantic = loader.process_personas(portrait_json_list)

        self._compass = AutoCompass(
            semantic.vectors, semantic.raw_topics, self.embed_service
        )
        seed_S = self._compass.compute_stance()

        synth = PopulationSynthesizer(target_size=self.num_agents)
        self._population = synth.synthesize(
            seed_S,
            semantic.influences,
            seed_tiers=semantic.tiers,
            seed=self.random_seed,
        )

        # A-1：hours_per_step 从 time_scale 单一推导（每步 1 小时基准时 = 1.0）
        hours_per_step = self.time_scale / 3600.0
        self._engine = VectorizedABMEngine(
            S=self._population.S,
            Inf=self._population.Inf,
            log_saturation_threshold=self._population.log_saturation_threshold,
            p_online=self.p_online if isinstance(self.p_online, (int, float)) else 0.1,
            p_online_map=self.p_online if isinstance(self.p_online, dict) else None,
            tier_labels=self._population.tiers,
            hours_per_step=hours_per_step,
            seed=self.random_seed,
        )

    def split_holdout(self, test_ratio: float = 0.3) -> None:
        """C-1：70/30 留出切分（固定种子，可复现）。

        校准（E/M 步）只在训练集上进行；验证指标在测试集（模型未见过的
        内容）上计算，防过拟合。
        """
        story_ids = list(self.representative_stories.keys())
        if len(story_ids) < 5:
            logger.warning(
                "内容数 <5，留出切分退化为全量校准（无独立验证集，指标不可过度解读）。"
            )
            self.train_story_ids = list(story_ids)
            self.test_story_ids = []
            return
        rng = np.random.default_rng(self.random_seed)
        shuffled = rng.permutation(story_ids).tolist()
        n_test = max(1, int(round(len(story_ids) * test_ratio)))
        self.test_story_ids = sorted(shuffled[:n_test])
        test_set = set(self.test_story_ids)
        self.train_story_ids = sorted(
            [sid for sid in story_ids if sid not in test_set]
        )
        logger.info(
            "[Holdout] 训练集 %d 条 / 验证集 %d 条（test_ratio=%.2f）",
            len(self.train_story_ids),
            len(self.test_story_ids),
            test_ratio,
        )

    def _stories_by_ids(self, story_ids: Sequence[str]) -> List[Dict[str, Any]]:
        """按 id 顺序取代表性内容（保证与 story_params 位置索引一一对应）。"""
        return [self.representative_stories[sid] for sid in story_ids]

    def select_representative_stories(
        self,
        observations: Sequence[Any],
        anchor_percentile: float = 0.80,
    ) -> Dict[str, Dict[str, Any]]:
        """根据观测数据筛选代表性内容，并缩放到 ABM 规模。"""
        records: List[Dict[str, Any]] = []
        for item in observations:
            if isinstance(item, dict):
                d = dict(item)
            elif is_dataclass(item):
                # slots=True 的 dataclass 没有 __dict__，需用 asdict 展开
                d = asdict(item)
            else:
                continue
            records.append(d)

        self.representative_stories = self.story_manager.select_representative_stories(
            records, anchor_percentile=anchor_percentile
        )
        return self.representative_stories

    def load_prepared_stories(self, records: Sequence[Dict[str, Any]]) -> None:
        """加载已准备的内容记录，建立全部故事映射，不筛选或切分。

        输入来自数据包分区（train.json 的 records），每条含 story_id、text、
        repost_count、view_count、scaled_target 等；配合 precompute_interests()
        建立 I_pop 后即可用于公共概率全量校准。
        """
        self.representative_stories = {}
        for record in records:
            story_id = str(record["story_id"])
            self.representative_stories[story_id] = {
                "story_id": story_id,
                "text": str(record.get("text", "")),
                "repost_count": float(record.get("repost_count", 0.0)),
                "view_count": float(record.get("view_count", 0.0)),
                "scaled_target": float(record.get("scaled_target", 0.0)),
                "unclipped_target": float(record.get("unclipped_target", 0.0)),
                "target_clipped": bool(record.get("target_clipped", False)),
            }

    def run_global_calibration(
        self,
        iterations: int = 3,
        duration: int = 24,
        n_repeats: int = 5,
        p_trials: int = 20,
        weight_trials: int = 50,
    ) -> Dict[str, Any]:
        """对已加载内容运行公共概率与推荐参数全量校准。"""
        if self._engine is None:
            raise RuntimeError("请先调用 load_portraits() 加载用户画像。")
        stories = list(self.representative_stories.values())
        if not stories:
            raise RuntimeError("没有已加载的内容（请先 load_prepared_stories）。")
        calibrator = EMCalibrationEngine(
            self._engine, stories, n_cpu=self.n_cpu, seed=self.random_seed
        )
        result = calibrator.run_global_calibration(
            iterations=iterations,
            duration=duration,
            n_repeats=n_repeats,
            p_trials=p_trials,
            weight_trials=weight_trials,
        )
        self.best_weights = dict(result["weights"])
        return result

    def environment_snapshot(self) -> Dict[str, Any]:
        """返回重建模拟环境所需的完整 ABM 参数快照。"""
        if self._engine is None:
            raise RuntimeError("请先调用 load_portraits() 加载用户画像。")
        engine = self._engine
        return {
            "population_size": int(self.num_agents),
            "p_online": self.p_online,
            "belief_update": {
                "backfire_mu": float(engine.backfire_mu),
                "backfire_k": float(engine.backfire_k),
                "learning_rate": float(engine.learning_rate),
            },
            "tier_weight": dict(TIER_WEIGHT_DEFAULT),
            "hours_per_step": float(engine.hours_per_step),
            "time_scale": float(self.time_scale),
        }

    def precompute_interests(self) -> None:
        """为每条代表性内容预计算全量种群兴趣向量。"""
        if self._compass is None or self._population is None:
            raise RuntimeError("请先调用 load_portraits() 加载用户画像。")

        synth = PopulationSynthesizer(target_size=self.num_agents)
        stories = list(self.representative_stories.values())

        logger.info("预计算 %d 条内容的兴趣向量...", len(stories))
        for story in stories:
            tweet_text = str(story.get("text", ""))
            seed_I = self._compass.compute_interest(tweet_text)
            I_pop_story = synth.expand_interest_for_population(
                seed_I, self._population.source_indices, seed=self.random_seed
            )
            story["I_pop"] = I_pop_story

    def calibrate_probabilities(
        self,
        current_weights: Optional[Dict[str, float]] = None,
        story_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """E 步：固定权重，校准每条内容的 p_base 概率。

        story_ids：参与校准的内容 id（C-1 下默认只含训练集）。
        """
        if self._engine is None:
            raise RuntimeError("请先调用 load_portraits() 加载用户画像。")

        ids = list(story_ids or self.train_story_ids or self.representative_stories.keys())
        stories = self._stories_by_ids(ids)
        if not stories:
            return {}

        weights = current_weights or {
            "w_i": 0.4,
            "w_pop": 0.3,
            "w_time": 0.2,
            "w_rand": 0.1,
        }

        calibrator = EMCalibrationEngine(
            self._engine, stories, n_cpu=self.n_cpu, seed=self.random_seed
        )
        calibrator.e_step(weights)

        calibrated: Dict[str, Dict[str, float]] = {}
        for i, story_id in enumerate(ids):
            story_info = self.representative_stories[story_id]
            p_base = calibrator.story_params.get(i, 0.1)
            calibrated[story_id] = {
                "p_base": p_base,
                "real_repost": float(story_info.get("real_repost", 0)),
                "view_count": float(story_info.get("view_count", 0)),
                "scaled_target": float(story_info.get("scaled_target", 0)),
            }

        self.calibrated_probs = calibrated
        self._calibrator = calibrator
        self._calibration_story_ids = ids
        return calibrated

    def optimize_recommendation_weights(
        self, current_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """M 步：固定概率，搜索最优推荐权重（C-1 下只在训练集上进行）。"""
        if not self.calibrated_probs:
            raise RuntimeError("请先完成概率校准 (calibrate_probabilities)。")

        ids = list(self._calibration_story_ids or self.calibrated_probs.keys())
        stories = self._stories_by_ids(ids)
        calibrator = EMCalibrationEngine(
            self._engine, stories, n_cpu=self.n_cpu, seed=self.random_seed
        )

        # 复用 E 步的结果（位置索引与 ids 一一对应）
        for i in range(len(stories)):
            calibrator.story_params[i] = self._calibrator.story_params.get(i, 0.1)

        best_weights, best_loss, step_diagnostics = calibrator.m_step()

        self.best_weights = best_weights

        # 构建诊断信息（P1-B：多次重复并报告均值/标准差，不单次下结论）
        diagnostics: Dict[str, Dict[str, float]] = {}
        decay_lambda = float(best_weights.get("decay_lambda", 0.5))
        for i, story_id in enumerate(ids):
            story = stories[i]
            p_base = calibrator.story_params.get(i, 0.1)
            I_pop = np.asarray(story["I_pop"])
            finals: List[float] = []
            last_views = 0.0
            for _ in range(5):
                hist, views = self._engine.run_simulation(
                    best_weights, p_base, I_pop, duration=24,
                    decay_lambda=decay_lambda,
                )
                finals.append(float(hist[-1]))
                last_views = float(views)
            finals_arr = np.asarray(finals)
            diagnostics[story_id] = {
                "mean_scaled_repost": float(finals_arr.mean()),
                "std_scaled_repost": float(finals_arr.std()),
                "mean_scaled_view": last_views,
                "scaled_target": float(story.get("scaled_target", 0)),
            }

        diagnostics["_meta"] = {
            "best_loss": float(best_loss),
            **step_diagnostics,
        }
        self.weight_fit_diagnostics = diagnostics

        return {
            **best_weights,
            "best_loss": best_loss,
            "duration": 24,
            "p_online": self.p_online,
        }

    def evaluate_holdout(
        self,
        weights: Dict[str, float],
        n_repeats: int = 30,
        duration: int = 24,
    ) -> Dict[str, Any]:
        """C-1/C-2/C-3：在留出验证集上做多指标验证。

        - 权重固定，测试内容各自先做一次 E 步校准 p_base（p_base 是内容级
          音量参数，不在“权重”的验证范围；权重的泛化性才是被验证对象）；
        - N≥30 次独立仿真，报告均值 / 95% CI / Cohen's d；
        - 分布级 KS 检验 + Spearman 秩相关 + MAE/MRE。
        """
        if not self.test_story_ids:
            return {"available": False}
        from scipy import stats

        test_ids = self.test_story_ids
        test_stories = self._stories_by_ids(test_ids)
        decay_lambda = float(weights.get("decay_lambda", 0.5))
        e_weights = {k: v for k, v in weights.items() if k != "decay_lambda"}

        # 测试内容各自的 p_base（内容级音量参数，固定权重下校准）
        test_calibrator = EMCalibrationEngine(
            self._engine, test_stories, n_cpu=self.n_cpu, seed=self.random_seed
        )
        test_calibrator.e_step(e_weights, duration=duration, decay_lambda=decay_lambda)

        per_story_means: List[float] = []
        targets: List[float] = []
        sim_finals_all: List[float] = []
        per_story_mre: List[float] = []
        per_story_stats: Dict[str, Dict[str, float]] = {}
        for i, story_id in enumerate(test_ids):
            story = test_stories[i]
            p_base = test_calibrator.story_params.get(i, 0.1)
            target = float(story.get("scaled_target", 0.0))
            targets.append(target)
            rng_base = _stable_rng_base("holdout", story_id, self.random_seed)
            finals: List[float] = []
            for repeat in range(n_repeats):
                repeat_rng = np.random.default_rng(rng_base + repeat * 100003)
                hist, _ = self._engine.run_simulation(
                    weights, p_base, np.asarray(story["I_pop"]),
                    duration=duration, decay_lambda=decay_lambda, rng=repeat_rng,
                )
                finals.append(float(hist[-1]))
            finals_arr = np.asarray(finals, dtype=float)
            mean_final = float(finals_arr.mean())
            per_story_means.append(mean_final)
            sim_finals_all.extend(finals)
            mre = abs(mean_final - target) / max(target, 1e-9)
            per_story_mre.append(mre)
            per_story_stats[story_id] = {
                "mean_sim": mean_final,
                "std_sim": float(finals_arr.std()),
                "target": target,
                "mre": mre,
            }

        targets_arr = np.asarray(targets, dtype=float)
        means_arr = np.asarray(per_story_means, dtype=float)
        mre_arr = np.asarray(per_story_mre, dtype=float)
        sim_all_arr = np.asarray(sim_finals_all, dtype=float)

        mae = float(np.mean(np.abs(means_arr - targets_arr)))
        mre_mean = float(np.mean(mre_arr))
        # 95% CI（均值相对误差，跨内容）
        mre_ci_half = 1.96 * float(mre_arr.std(ddof=1) / np.sqrt(len(mre_arr))) if len(mre_arr) > 1 else 0.0

        # KS 检验：仿真传播量分布 vs 真实分布（原始 + 均值归一化两种口径）
        ks_stat_raw, ks_p_raw = stats.ks_2samp(sim_all_arr, targets_arr)
        sim_norm = sim_all_arr / max(float(sim_all_arr.mean()), 1e-9)
        tgt_norm = targets_arr / max(float(targets_arr.mean()), 1e-9)
        ks_stat_norm, ks_p_norm = stats.ks_2samp(sim_norm, tgt_norm)

        # Spearman 秩相关：能否分辨“哪些内容会火”
        spearman_rho: Optional[float] = None
        spearman_p: Optional[float] = None
        if len(means_arr) >= 3:
            spearman = stats.spearmanr(means_arr, targets_arr)
            spearman_rho = float(spearman.statistic)
            spearman_p = float(spearman.pvalue)

        # Cohen's d：仿真分布 vs 真实分布（效应量）
        pooled_std = np.sqrt(
            (sim_all_arr.std(ddof=1) ** 2 + targets_arr.std(ddof=1) ** 2) / 2
        )
        cohens_d = float(
            (sim_all_arr.mean() - targets_arr.mean()) / max(pooled_std, 1e-9)
        )

        return {
            "available": True,
            "test_story_ids": test_ids,
            "n_test": len(test_ids),
            "n_repeats": n_repeats,
            "mae": mae,
            "mre_mean": mre_mean,
            "mre_ci95": [float(max(mre_mean - mre_ci_half, 0.0)), float(mre_mean + mre_ci_half)],
            "ks_stat_raw": float(ks_stat_raw),
            "ks_p_raw": float(ks_p_raw),
            "ks_stat_norm": float(ks_stat_norm),
            "ks_p_norm": float(ks_p_norm),
            "spearman_rho": spearman_rho,
            "spearman_p": spearman_p,
            "cohens_d": cohens_d,
            "per_story": per_story_stats,
            "p_base": [test_calibrator.story_params.get(i, 0.1) for i in range(len(test_ids))],
        }

    def run_seed_robustness(
        self,
        seeds: Sequence[int] = (0, 1, 2),
        duration: int = 24,
        n_repeats: int = 5,
        n_trials: int = 20,
    ) -> Dict[str, Any]:
        """C-5：扰动随机种子重跑 M 步，报告校准权重的稳定性（mean/std/range）。

        对应 TRAILS 的鲁棒性审计思想；论文写作时再补充非参数检验
        （Mann-Whitney U + Holm 校正）。
        """
        ids = list(self.train_story_ids or self.representative_stories.keys())
        stories = self._stories_by_ids(ids)
        if not stories:
            return {"available": False}
        base_params = dict(
            getattr(self, "_calibrator", None).story_params
        ) if getattr(self, "_calibrator", None) is not None else {}
        weights_by_seed: Dict[str, Dict[str, float]] = {}
        for seed in seeds:
            cal = EMCalibrationEngine(
                self._engine, stories, n_cpu=self.n_cpu, seed=int(seed)
            )
            cal.story_params = dict(base_params)
            w, _, _ = cal.m_step(
                duration=duration, n_repeats=n_repeats, n_trials=n_trials
            )
            weights_by_seed[str(seed)] = w
        dims = ["w_i", "w_pop", "w_time", "w_rand", "decay_lambda"]
        per_dim: Dict[str, Dict[str, Any]] = {}
        for dim in dims:
            arr = np.asarray([weights_by_seed[s][dim] for s in weights_by_seed], dtype=float)
            per_dim[dim] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "range": [float(arr.min()), float(arr.max())],
            }
        return {
            "available": True,
            "seeds": [int(s) for s in seeds],
            "weights_by_seed": weights_by_seed,
            "per_dim": per_dim,
        }

    def run_em_calibration_loop(
        self,
        max_iterations: int = 3,
        holdout_n_repeats: int = 30,
        robustness_seeds: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """执行完整 E-M 交替校准（C 系列验证全部接入）。

        流程：训练集 EM → 消融（训练集损失）→ 留出验证（N≥30 重复 +
        KS/Spearman/CI/Cohen's d）→ 消融接验证集 → 可选种子鲁棒性审计。
        """
        ids = list(self.train_story_ids or self.representative_stories.keys())
        stories = self._stories_by_ids(ids)
        if not stories:
            logger.warning("没有代表性内容，无法校准。")
            return {}

        if self._engine is None:
            raise RuntimeError("请先调用 load_portraits() 加载用户画像。")

        calibrator = EMCalibrationEngine(
            self._engine, stories, n_cpu=self.n_cpu, seed=self.random_seed
        )
        start_time = time.time()
        best_weights, loop_diagnostics = calibrator.run_em_loop(
            iterations=max_iterations
        )
        self._calibration_story_ids = ids
        self._calibrator = calibrator

        # calibrated_probs：按训练集 id 对齐写入（供 YAML calibrated_p_base）
        self.calibrated_probs = {
            sid: {
                "p_base": calibrator.story_params.get(i, 0.1),
                "real_repost": float(self.representative_stories[sid].get("real_repost", 0)),
                "view_count": float(self.representative_stories[sid].get("view_count", 0)),
                "scaled_target": float(self.representative_stories[sid].get("scaled_target", 0)),
            }
            for i, sid in enumerate(ids)
        }

        # 消融（P1-B / C-4）：训练集损失退化
        ablation = calibrator.run_ablation(best_weights)

        # C-1/C-2/C-3：留出验证（测试集 + N≥30 重复 + 多指标）
        holdout = self.evaluate_holdout(best_weights, n_repeats=holdout_n_repeats)

        # C-4：消融接验证集指标（复用留出 E 步的 p_base）
        ablation_holdout: Dict[str, Any] = {"available": False}
        if holdout.get("available") and self.test_story_ids:
            test_stories = self._stories_by_ids(self.test_story_ids)
            test_calibrator = EMCalibrationEngine(
                self._engine, test_stories, n_cpu=self.n_cpu, seed=self.random_seed
            )
            for i, p in enumerate(holdout["p_base"]):
                test_calibrator.story_params[i] = p
            ablation_holdout = test_calibrator.run_ablation(best_weights)

        # C-5：种子扰动鲁棒性（可选，较耗时）
        robustness: Dict[str, Any] = {"available": False, "skipped": True}
        if robustness_seeds:
            robustness = self.run_seed_robustness(seeds=robustness_seeds)

        self.best_weights = best_weights
        self.weight_fit_diagnostics["_meta"] = {
            "em_iterations": loop_diagnostics["iterations"],
            "ablation": ablation,
            "ablation_holdout": ablation_holdout,
            "holdout": holdout,
            "robustness": robustness,
            "train_story_ids": ids,
            "test_story_ids": self.test_story_ids,
        }
        logger.info("\n================ FINAL RESULT ================")
        logger.info("总耗时: %.2fs", time.time() - start_time)
        logger.info("计算出的推荐系统权重:")
        for k, v in best_weights.items():
            logger.info("  %s: %.4f", k, v)
        logger.info("消融结果（loss 上升越多说明该维度越关键）:")
        for key, info in ablation["ablations"].items():
            logger.info(
                "  去掉 %s: Δ%.4f (%.1f%%)",
                info["label"],
                info["delta"],
                info["delta_rel"] * 100,
            )
        if holdout.get("available"):
            logger.info(
                "留出验证（%d 条，N=%d）：MAE=%.2f, MRE=%.4f, "
                "KS(p)=%.3f, Spearman=%.3f (p=%.3f)",
                holdout["n_test"],
                holdout["n_repeats"],
                holdout["mae"],
                holdout["mre_mean"],
                holdout["ks_p_norm"],
                holdout["spearman_rho"] or float("nan"),
                holdout["spearman_p"] or float("nan"),
            )

        return best_weights


# ============================================================
# 7. 辅助函数
# ============================================================
@dataclass(slots=True)
class StoryObservation:
    """单条内容的观测数据。"""

    story_id: str
    repost_count: float
    view_count: float
    text: str = ""
    content: str = ""


def build_story_observations(
    records: Iterable[Dict[str, Any]],
) -> List[StoryObservation]:
    """将外部表格或数据库记录转换成统一观测结构。"""
    observations: List[StoryObservation] = []
    for index, record in enumerate(records):
        story_id = str(record.get("story_id", index))
        repost_count = float(record.get("repost_count", 0.0))
        view_count = float(record.get("view_count", 0.0))
        text = str(record.get("text", record.get("content", "")))
        observations.append(
            StoryObservation(
                story_id=story_id,
                repost_count=repost_count,
                view_count=view_count,
                text=text,
                content=text,
            )
        )
    return observations


def load_portraits_from_dir(portraits_dir: Path) -> List[Dict[str, Any]]:
    """从目录中加载所有画像 JSON 文件，转换为推理器所需格式。"""
    personas: List[Dict[str, Any]] = []
    for json_path in sorted(portraits_dir.glob("*.json")):
        if json_path.name == "failed_users.json":
            continue
        try:
            with json_path.open("r", encoding="utf-8") as f:
                profile = json.load(f)
        except Exception:
            logger.warning("跳过无法解析的文件: %s", json_path.name)
            continue

        # 跳过非画像 JSON（如 recommender 结果 content_observations.json）
        if not isinstance(profile.get("stable_profile"), dict):
            continue

        persona = _portrait_to_persona(profile)
        if persona:
            personas.append(persona)
    return personas


def load_portrait_bundle(
    portraits_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """加载画像目录，返回 (personas, manifest)。

    personas 按相对文件名排序；manifest 记录实际加载画像的相对文件名与 SHA-256。
    合法非画像 JSON（无 stable_profile）不进入清单；损坏 JSON 报错并保留 cause，
    避免静默跳过导致训练出不同人口。
    """
    portraits_dir = Path(portraits_dir)
    personas: List[Dict[str, Any]] = []
    manifest: List[Dict[str, Any]] = []
    for json_path in sorted(portraits_dir.glob("*.json")):
        if json_path.name == "failed_users.json":
            continue
        try:
            with json_path.open("r", encoding="utf-8") as f:
                profile = json.load(f)
        except Exception as exc:
            raise ValueError(f"画像文件无法解析: {json_path}") from exc
        if not isinstance(profile, dict) or not isinstance(profile.get("stable_profile"), dict):
            continue
        persona = _portrait_to_persona(profile)
        if persona is None:
            continue
        personas.append(persona)
        manifest.append(
            {
                "path": json_path.name,
                "sha256": file_sha256(json_path),
            }
        )
    return personas, manifest


def _portrait_to_persona(profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """将画像 JSON 转换为 SemanticLoader 所需格式。"""
    stable = profile.get("stable_profile") or {}
    agent = profile.get("agent_profile") or {}
    stats = profile.get("stats") or {}

    if not isinstance(stable, dict) or not isinstance(stats, dict):
        return None

    # label: 从 value_anchors 提取立场文本
    value_anchors = stable.get("value_anchors") or []
    labels: List[str] = []
    if isinstance(value_anchors, list):
        for va in value_anchors:
            if isinstance(va, dict):
                stance = va.get("stance", "")
                if stance:
                    labels.append(str(stance))

    # user_info: 从 agent_profile 或 stable_profile 提取
    # B-3：与在线 belief_text 口径一致——identity_summary + interest_summary
    user_info = ""
    if isinstance(agent, dict):
        identity = str(agent.get("identity_summary", "") or "").strip()
        interest = str(agent.get("interest_summary", "") or "").strip()
        user_info = f"{identity} {interest}".strip()
    if not user_info:
        user_info = str(stable.get("profile_summary", ""))

    # topic: 从 content_topics 提取
    content_topics = stable.get("content_topics") or []
    topics: Dict[str, float] = {}
    if isinstance(content_topics, list):
        for t in content_topics:
            topics[str(t)] = 1.0

    influence = 1.0
    if isinstance(stats, dict):
        try:
            influence = float(stats.get("account_influence", 1.0))
        except (TypeError, ValueError):
            influence = 1.0

    if not labels and not user_info:
        return None

    return {
        "label": labels,
        "user_info": user_info,
        "topic": topics,
        "account_influence": influence,
        "influence_tier": profile.get("influence_tier", 4),
    }
