from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import KernelPCA

logging.getLogger("optuna").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# Tier 影响力预设权重（Rogers 5 级，不参与校准）
TIER_WEIGHT: Dict[int, float] = {1: 0.4, 2: 0.7, 3: 1.0, 4: 1.5, 5: 2.0}


def min_max_norm(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-Batch Min-Max 归一化到 [0, 1]。当 max==min 时返回 0.5。"""
    v_min = values.min()
    v_max = values.max()
    if v_max - v_min < eps:
        return np.full_like(values, 0.5)
    return (values - v_min) / (v_max - v_min + eps)


# ============================================================
# 0. 嵌入服务
# ============================================================
class EmbeddingService:
    """基于 SentenceTransformer 的文本嵌入服务。"""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("加载嵌入模型: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
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


class SemanticLoader:
    """从用户画像 JSON 中提取文本，计算向量嵌入。"""

    def __init__(self, embedding_service: EmbeddingService):
        self.embed = embedding_service

    def process_personas(self, user_json_list: List[Dict[str, Any]]) -> SemanticOutput:
        """将画像 JSON 列表转为向量、话题字典和影响力数组。"""
        texts: List[str] = []
        user_topics: List[Dict[str, float]] = []
        influences: List[float] = []

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

        vectors = self.embed.embed_documents(texts)
        return SemanticOutput(
            vectors=np.asarray(vectors),
            raw_topics=user_topics,
            influences=np.asarray(influences, dtype=float),
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


class PopulationSynthesizer:
    """将种子用户扩展为全量仿真种群。"""

    def __init__(self, target_size: Optional[int] = None):
        self.target_size = target_size

    def synthesize(
        self,
        seed_S: np.ndarray,
        seed_Inf: np.ndarray,
        seed: int = 42,
    ) -> Population:
        current_n = len(seed_S)
        target_size = self.target_size
        if target_size is None:
            target_size = max(int(current_n / 0.16), current_n)
        if current_n == 0:
            return Population(
                S=np.zeros(target_size),
                Inf=np.zeros(target_size),
                log_saturation_threshold=1.0,
                source_indices=np.zeros(target_size, dtype=int),
            )

        if seed is not None:
            np.random.seed(seed)

        # 精英保留 (top 5%)
        n_elite = max(1, int(current_n * 0.05))
        elite_idx = np.argsort(seed_Inf)[::-1][:n_elite]
        real_elite_S = seed_S[elite_idx].copy()
        real_elite_Inf = seed_Inf[elite_idx].copy()

        final_S: List[float] = list(real_elite_S)
        source_indices: List[int] = list(elite_idx)
        crowd_idx = [i for i in range(current_n) if i not in elite_idx]

        while len(final_S) < target_size:
            src = int(np.random.choice(crowd_idx))
            s_noise = float(np.random.normal(0, 0.1))
            final_S.append(float(np.clip(float(seed_S[src]) + s_noise, -1.0, 1.0)))
            source_indices.append(src)

        # 合成影响力 (Zipf 分布)
        n_crowd = target_size - len(real_elite_Inf)
        final_Inf: np.ndarray
        if n_crowd > 0:
            zipf_crowd = np.random.zipf(1.5, n_crowd).astype(float)
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

        # 同时重排 S 和 source_indices
        final_S_arr = np.array(final_S)[sort_idx]
        source_arr = np.array(source_indices, dtype=int)[sort_idx]

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
        )

    def expand_interest_for_population(
        self,
        seed_I: np.ndarray,
        source_indices: np.ndarray,
    ) -> np.ndarray:
        """根据 source_indices 将种子兴趣扩展到全量种群。"""
        pop_I = seed_I[source_indices].copy()
        noise = np.random.normal(0, 0.05, size=len(pop_I))
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
    """向量化仿真引擎，支持 Soft Backfire 信念更新。"""

    def __init__(
        self,
        S: np.ndarray,
        Inf: np.ndarray,
        log_saturation_threshold: float,
        p_online: float = 0.1,
        backfire_mu: float = 0.4,
        backfire_k: float = 10.0,
        learning_rate: float = 0.1,
    ):
        self.N = len(S)
        self.init_S = S.copy()
        self.current_S = S.copy()
        self.Inf = Inf
        self.log_saturation_threshold = max(log_saturation_threshold, 1e-9)
        self.p_online = p_online
        self.backfire_mu = backfire_mu
        self.backfire_k = backfire_k
        self.learning_rate = learning_rate
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
        self, target_indices: np.ndarray, source_indices: np.ndarray
    ) -> None:
        """Soft Backfire 信念更新。"""
        if len(target_indices) == 0:
            return
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
            rand_vals = np.random.random(len(p_backfire))
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
        tier_labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, int]:
        """向量化仿真：指数时间衰减 + tier 加权热度 + Per-Batch Min-Max 归一化。"""
        self.reset()
        history: List[int] = []
        curr_count = int(self.state.sum())
        total_views = 0

        w_i = float(weights.get("w_i", 0.35))
        w_pop = float(weights.get("w_pop", 0.25))
        w_time = float(weights.get("w_time", 0.25))
        w_rand = float(weights.get("w_rand", 0.15))

        safe_p = np.clip(p_base, 0.001, 0.999)
        base_logit = np.log(safe_p / (1.0 - safe_p))

        if tier_labels is None:
            tier_labels = np.full(self.N, 4, dtype=int)

        for t in range(duration):
            online_mask = np.random.random(self.N) < self.p_online
            active_src = self.state
            active_tgt = (~self.state) & online_mask

            if not active_tgt.any():
                history.append(curr_count)
                continue

            n_active = int(active_tgt.sum())

            # Step 1: raw scores
            raw_interest = I_pop[active_tgt]

            src_tiers = tier_labels[active_src]
            tier_coeffs = np.array([TIER_WEIGHT.get(int(tier), 1.0) for tier in src_tiers])
            avg_tier_coeff = float(np.mean(tier_coeffs)) if len(tier_coeffs) > 0 else 1.0
            raw_pop_scalar = np.log1p(curr_count) / np.log1p(self.N) * avg_tier_coeff

            dt = t - self.time[active_src].min() if active_src.any() else 0.0
            raw_time_scalar = np.exp(-decay_lambda * max(dt, 0.0))

            raw_rand = np.random.random(n_active)

            # Step 2: Per-Batch Min-Max normalization
            norm_interest = min_max_norm(raw_interest)
            norm_rand = min_max_norm(raw_rand)

            # Step 3: Weighted sum
            scores = (
                w_i * norm_interest
                + w_pop * raw_pop_scalar
                + w_time * raw_time_scalar
                + w_rand * norm_rand
            )

            total_views += n_active

            visible = np.random.random(n_active) < np.clip(scores, 0.0, 1.0)
            visible_idx = np.where(visible)[0]
            if len(visible_idx) > 0:
                tgt_indices_all = np.where(active_tgt)[0]
                actual_indices = tgt_indices_all[visible_idx]
                S_vis = self.current_S[actual_indices]

                internal_term = alpha * (np.abs(S_vis) ** 2)
                action_probs = 1.0 / (1.0 + np.exp(-(base_logit + internal_term)))

                actions = np.random.random(len(actual_indices)) < action_probs
                new_infected = actual_indices[actions]

                if len(new_infected) > 0:
                    self.state[new_infected] = True
                    self.time[new_infected] = float(t)
                    curr_count += len(new_infected)

                    # Belief update for newly infected
                    if hasattr(self, '_update_beliefs'):
                        src_indices = np.where(active_src)[0]
                        if len(src_indices) > 0:
                            # pick the most influential source
                            src_inf = self.Inf[src_indices]
                            best_src = src_indices[int(np.argmax(src_inf))]
                            src_arr = np.full(len(new_infected), best_src)
                            self._update_beliefs(new_infected, src_arr)

            history.append(curr_count)

        return np.array(history), total_views


# ============================================================
# 5. E-M 校准引擎 (Optuna)
# ============================================================
class EMCalibrationEngine:
    """使用 Optuna 进行 E-M 交替校准。"""

    def __init__(
        self,
        abm_engine: VectorizedABMEngine,
        stories: List[Dict[str, Any]],
        n_cpu: int = 4,
    ):
        self.engine = abm_engine
        self.stories = stories
        self.story_params: Dict[int, float] = {}
        self.n_cpu = n_cpu

    def _calibrate_single_story(
        self, story: Dict[str, Any], current_weights: Dict[str, float]
    ) -> float:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        logging.getLogger("optuna").setLevel(logging.ERROR)

        real_repost = float(story.get("real_repost", story.get("real_retweets", 1.0)))
        real_view = float(story.get("view_count", story.get("real_views", 1.0)))
        target_rate = real_repost / max(real_view, 1e-9)
        I_pop = np.asarray(story["I_pop"])

        def objective(trial: optuna.Trial) -> float:
            p = trial.suggest_float("p_base", 0.01, 0.99)
            sim_rates: List[float] = []
            for _ in range(5):
                hist, views = self.engine.run_simulation(
                    current_weights, p, I_pop, duration=24
                )
                rate = float(hist[-1]) / max(float(views), 1e-9)
                sim_rates.append(rate)
            return float(abs(np.mean(sim_rates) - target_rate) / max(target_rate, 1e-9))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20, show_progress_bar=False)
        return float(study.best_params["p_base"])

    def e_step(self, current_weights: Dict[str, float]) -> None:
        logger.info("  [E-Step] 并行校准 %d 条推文...", len(self.stories))
        results = Parallel(n_jobs=self.n_cpu)(
            delayed(self._calibrate_single_story)(story, current_weights)
            for story in self.stories
        )
        for i, p in enumerate(results):
            self.story_params[i] = p

    def m_step(self) -> Tuple[Dict[str, float], float]:
        logger.info("  [M-Step] 优化全局推荐权重 W + decay_lambda...")
        import optuna

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        logging.getLogger("optuna").setLevel(logging.ERROR)

        def objective(trial: optuna.Trial) -> float:
            # Dirichlet 采样 4 个权重 (自动满足 sum=1)
            alpha_param = trial.suggest_float("alpha_param", 0.5, 5.0)
            raw = np.random.default_rng(trial.number).dirichlet(np.full(4, alpha_param))
            w_i, w_pop, w_time, w_rand = float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])

            decay_lambda = trial.suggest_float("decay_lambda", 0.01, 3.0, log=True)

            weights: Dict[str, float] = {
                "w_i": w_i, "w_pop": w_pop, "w_time": w_time, "w_rand": w_rand,
            }

            total_rel_error = 0.0
            sample_indices = list(range(min(len(self.stories), 20)))
            if not sample_indices:
                return 0.0

            for i in sample_indices:
                story = self.stories[i]
                p_base = self.story_params[i]
                I_pop = np.asarray(story["I_pop"])
                target_count = float(story["scaled_target"])

                sim_finals: List[float] = []
                for _ in range(3):
                    hist, _ = self.engine.run_simulation(
                        weights, p_base, I_pop, duration=24, decay_lambda=decay_lambda
                    )
                    sim_finals.append(float(hist[-1]))
                rel_error = abs(np.mean(sim_finals) - target_count) / max(target_count, 1e-9)
                total_rel_error += rel_error

            return total_rel_error / len(sample_indices)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50, show_progress_bar=False)

        best = study.best_params
        raw = np.random.default_rng(0).dirichlet(np.full(4, best["alpha_param"]))
        best_weights: Dict[str, float] = {
            "w_i": float(raw[0]),
            "w_pop": float(raw[1]),
            "w_time": float(raw[2]),
            "w_rand": float(raw[3]),
            "decay_lambda": float(best["decay_lambda"]),
        }
        return best_weights, float(study.best_value)

    def run_em_loop(self, iterations: int = 3) -> Dict[str, float]:
        weights: Dict[str, float] = {
            "w_i": 0.35, "w_pop": 0.25, "w_time": 0.25, "w_rand": 0.15, "decay_lambda": 0.5,
        }
        logger.info("\n=== 启动 E-M (Optuna) 校准循环 (Max %d 轮) ===", iterations)
        for k in range(iterations):
            logger.info("\n--- Iteration %d ---", k + 1)
            e_step_weights = {k: v for k, v in weights.items() if k != "decay_lambda"}
            self.e_step(e_step_weights)
            weights, loss = self.m_step()
            logger.info("  >>> Iteration %d 最佳参数: %s", k + 1, weights)
            logger.info("  >>> Global MRE: %.4f", loss)
        return weights


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
        p_online: float = 0.1,
        embedding_model: str = "BAAI/bge-m3",
        n_cpu: int = 4,
        target_size_for_sampling: Optional[int] = None,
    ):
        self.num_agents = num_agents
        if target_size_for_sampling is None:
            target_size_for_sampling = num_agents
        self.min_scaled_target = min_scaled_target
        self.p_online = p_online
        self.n_cpu = n_cpu

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
        self._population = synth.synthesize(seed_S, semantic.influences)

        self._engine = VectorizedABMEngine(
            S=self._population.S,
            Inf=self._population.Inf,
            log_saturation_threshold=self._population.log_saturation_threshold,
            p_online=self.p_online,
        )

    def select_representative_stories(
        self,
        observations: Sequence[Any],
        anchor_percentile: float = 0.80,
    ) -> Dict[str, Dict[str, Any]]:
        """根据观测数据筛选代表性内容，并缩放到 ABM 规模。"""
        records: List[Dict[str, Any]] = []
        for item in observations:
            if hasattr(item, "__dict__"):
                d = {k: v for k, v in item.__dict__.items() if not k.startswith("_")}
            elif isinstance(item, dict):
                d = dict(item)
            else:
                continue
            records.append(d)

        self.representative_stories = self.story_manager.select_representative_stories(
            records, anchor_percentile=anchor_percentile
        )
        return self.representative_stories

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
                seed_I, self._population.source_indices
            )
            story["I_pop"] = I_pop_story

    def calibrate_probabilities(
        self, current_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Dict[str, float]]:
        """E 步：固定权重，校准每条内容的 p_base 概率。"""
        if self._engine is None:
            raise RuntimeError("请先调用 load_portraits() 加载用户画像。")

        stories = list(self.representative_stories.values())
        if not stories:
            return {}

        weights = current_weights or {
            "w_i": 0.4,
            "w_pop": 0.3,
            "w_time": 0.2,
            "w_rand": 0.1,
        }

        calibrator = EMCalibrationEngine(self._engine, stories, n_cpu=self.n_cpu)
        calibrator.e_step(weights)

        calibrated: Dict[str, Dict[str, float]] = {}
        story_ids = list(self.representative_stories.keys())
        for i, story_id in enumerate(story_ids):
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
        return calibrated

    def optimize_recommendation_weights(
        self, current_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """M 步：固定概率，搜索最优推荐权重。"""
        if not self.calibrated_probs:
            raise RuntimeError("请先完成概率校准 (calibrate_probabilities)。")

        stories = list(self.representative_stories.values())
        calibrator = EMCalibrationEngine(self._engine, stories, n_cpu=self.n_cpu)

        # 复用 E 步的结果
        for i in range(len(stories)):
            calibrator.story_params[i] = self._calibrator.story_params.get(i, 0.1)

        best_weights, best_loss = calibrator.m_step()

        self.best_weights = best_weights

        # 构建诊断信息
        diagnostics: Dict[str, Dict[str, float]] = {}
        story_ids = list(self.representative_stories.keys())
        for i, story_id in enumerate(story_ids):
            story = stories[i]
            p_base = calibrator.story_params.get(i, 0.1)
            I_pop = np.asarray(story["I_pop"])
            hist, views = self._engine.run_simulation(
                best_weights, p_base, I_pop, duration=24
            )
            diagnostics[story_id] = {
                "mean_scaled_repost": float(hist[-1]),
                "mean_scaled_view": float(views),
                "scaled_target": float(story.get("scaled_target", 0)),
            }

        self.weight_fit_diagnostics = diagnostics

        return {
            **best_weights,
            "best_loss": best_loss,
            "duration": 24,
            "p_online": self.p_online,
        }

    def run_em_calibration_loop(self, max_iterations: int = 3) -> Dict[str, Any]:
        """执行完整 E-M 交替校准。"""
        stories = list(self.representative_stories.values())
        if not stories:
            logger.warning("没有代表性内容，无法校准。")
            return {}

        if self._engine is None:
            raise RuntimeError("请先调用 load_portraits() 加载用户画像。")

        calibrator = EMCalibrationEngine(self._engine, stories, n_cpu=self.n_cpu)
        start_time = time.time()
        best_weights = calibrator.run_em_loop(iterations=max_iterations)

        self.best_weights = best_weights
        logger.info("\n================ FINAL RESULT ================")
        logger.info("总耗时: %.2fs", time.time() - start_time)
        logger.info("计算出的推荐系统权重:")
        for k, v in best_weights.items():
            logger.info("  %s: %.4f", k, v)

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

        persona = _portrait_to_persona(profile)
        if persona:
            personas.append(persona)
    return personas


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
    user_info = ""
    if isinstance(agent, dict):
        user_info = str(agent.get("identity_summary", ""))
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
