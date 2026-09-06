import asyncio
import json
import random
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
from backend.dao.database import get_db_connection, format_stats
from core.scoring import (
    TIER_WEIGHT_DEFAULT,
    cosine_from_vec_distance,
    interest_score,
    min_max_normalize,
    popularity_score,
    time_decay_score,
    weighted_score,
)
from sentence_transformers import SentenceTransformer
from backend.services.logger_service import logger
from backend.services.time_service import time_service


class SocialRecSys:
    # 类级可配置参数（启动时通过 configure() 注入）
    W_BELIEF: float = 0.3
    W_POP: float = 0.2
    W_CHRONO: float = 0.3
    W_RAND: float = 0.2
    DECAY_LAMBDA: float = 0.5
    TIER_WEIGHT: Dict[int, float] = dict(TIER_WEIGHT_DEFAULT)
    _embedding_model_name: str = "Alibaba-NLP/gte-multilingual-base"
    _normalize_embeddings: bool = True

    @classmethod
    def configure(cls, recommender_config, embedding_config) -> None:
        """从 CalibrationProfile 注入推荐参数。

        必须在 Backend 子进程内、create_app() 之前调用（A-1）：
        spawn 模式下主进程的类属性不会跨进程传递。
        """
        cls.W_BELIEF = recommender_config.weights.w_interest
        cls.W_POP = recommender_config.weights.w_popularity
        cls.W_CHRONO = recommender_config.weights.w_time
        cls.W_RAND = recommender_config.weights.w_random
        cls.DECAY_LAMBDA = recommender_config.decay_lambda
        cls.TIER_WEIGHT = dict(recommender_config.tier_weight)
        cls._embedding_model_name = embedding_config.model_name
        cls._normalize_embeddings = bool(embedding_config.normalize_embeddings)

    def __init__(self):
        # Lazy load model
        self._model = None

    @property
    def model(self):
        if self._model is None:
            logger.info("Loading SentenceTransformer model...")
            # A-2：使用 configure() 注入的模型名，与离线 ABM 保持同一嵌入空间
            self._model = SentenceTransformer(
                SocialRecSys._embedding_model_name, trust_remote_code=True
            )
            logger.info("Model loaded.")
        return self._model

    def _encode(self, text: str):
        """编码文本（与 ABM 侧一致地归一化嵌入，保证余弦度量可比）。"""
        return self.model.encode(
            text,
            show_progress_bar=False,
            normalize_embeddings=SocialRecSys._normalize_embeddings,
        )

    async def add_post_vector(self, post_id: int, content: str):
        if not content:
            return

        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(None, lambda: self._encode(content))
        # Convert to list for JSON serialization
        embedding_list = embedding.tolist()
        embedding_json = json.dumps(embedding_list)

        async with get_db_connection() as db:
            try:
                await db.execute(
                    "INSERT INTO posts_vec(rowid, content_embedding) VALUES (?, ?)",
                    (post_id, embedding_json),
                )
                await db.commit()
                logger.debug(f"Inserted vector for post {post_id}")
            except Exception as e:
                logger.error(f"Failed to insert vector for post {post_id}: {e}")

    async def get_recommended_feed(
        self, user_id: int, limit: int = 5
    ) -> List[Dict[str, Any]]:
        async with get_db_connection() as db:
            # 1. 取用户立场/兴趣文本：优先 belief_text（B-5），回退 bio
            async with db.execute(
                "SELECT COALESCE(NULLIF(belief_text,''), NULLIF(bio,'')) AS belief_text"
                " FROM users WHERE id = ?",
                (user_id,),
            ) as cursor:
                user_row = await cursor.fetchone()
                user_belief = user_row["belief_text"] if user_row else ""

            # 在线总人口 N：注册用户数（B-1 热度公式分母，用注册用户数近似）
            async with db.execute(
                "SELECT COUNT(*) FROM users WHERE type != 'GOD'"
            ) as cursor:
                count_row = await cursor.fetchone()
                population_n = int(count_row[0]) if count_row and count_row[0] else 1

            posts = []
            distances = {}  # Map post_id -> distance

            # 2. Candidate Generation
            if user_belief:
                # Check if there are any vectors in the vec table to avoid crash
                async with db.execute("SELECT count(*) FROM posts_vec") as cursor:
                    row = await cursor.fetchone()
                    vec_count = row[0]

                if vec_count > 0:
                    # Vector Search
                    loop = asyncio.get_running_loop()
                    # Use a lambda to pass show_progress_bar
                    user_embedding = await loop.run_in_executor(
                        None, lambda: self._encode(user_belief)
                    )
                    user_embedding_json = json.dumps(user_embedding.tolist())

                    # Search top 100 similar posts using sqlite-vec
                    try:
                        async with db.execute(
                            """
                            SELECT rowid, distance
                            FROM posts_vec
                            WHERE content_embedding MATCH ?
                            ORDER BY distance
                            LIMIT 100
                        """,
                            (user_embedding_json,),
                        ) as cursor:
                            vec_results = await cursor.fetchall()

                        if vec_results:
                            post_ids = [row[0] for row in vec_results]
                            distances = {row[0]: row[1] for row in vec_results}

                            # Fetch full post data for these IDs
                            placeholders = ",".join("?" * len(post_ids))
                            async with db.execute(
                                f"""
                                SELECT p.*, u.nickname as author_nickname, u.tier as author_tier
                                FROM posts p
                                JOIN users u ON p.user_id = u.id
                                WHERE p.id IN ({placeholders}) AND p.user_id != ?
                            """,
                                (*post_ids, user_id),
                            ) as cursor:
                                posts = [dict(row) for row in await cursor.fetchall()]
                    except Exception as e:
                        logger.error(
                            f"Vector search failed: {e}. Falling back to chronological."
                        )
                        # Fallback will happen if posts is empty

            # Fallback or if no bio: Chronological
            if not posts:
                async with db.execute("""
                    SELECT p.*, u.nickname as author_nickname, u.tier as author_tier
                    FROM posts p
                    JOIN users u ON p.user_id = u.id
                    WHERE p.user_id != ?
                    ORDER BY p.created_at DESC, p.id DESC LIMIT 100
                """, (user_id,)) as cursor:
                    posts = [dict(row) for row in await cursor.fetchall()]

            if not posts:
                return []

            # 3. Calculate Scores
            current_time = time_service.get_current_time()

            # First pass: collect raw scores for each dimension
            raw_interest = []
            raw_pop = []
            raw_time_vals = []
            raw_rand = []
            scored_posts = []

            for post in posts:
                # S_time: Decay based on time difference
                try:
                    post_time = datetime.strptime(
                        str(post["created_at"]), "%Y-%m-%d %H:%M:%S"
                    )
                except ValueError:
                    try:
                        post_time = datetime.fromisoformat(str(post["created_at"]))
                    except:
                        post_time = current_time

                delta = current_time - post_time
                delta_hours = max(0, delta.total_seconds() / 3600.0)
                # B-2：时间衰减单位统一为小时，与 ABM 同一公式
                s_time = time_decay_score(delta_hours, SocialRecSys.DECAY_LAMBDA)

                # S_similarity（B-3：sqlite-vec cosine 距离还原为余弦相似度）
                s_similarity = 0.0
                if user_belief and post["id"] in distances:
                    dist = distances[post["id"]]
                    s_similarity = cosine_from_vec_distance(dist)
                    s_similarity = interest_score(s_similarity, stance_affinity=0.0)

                # S_popularity (with tier weight)
                stats = format_stats(post["stats"])
                likes = stats.get("like_count", 0)
                replies = stats.get("reply_count", 0)
                retweets = stats.get("retweet_count", 0)
                quotes = stats.get("quote_count", 0)
                shares = stats.get("share_count", 0)

                if retweets == 0 and quotes == 0 and shares > 0:
                    retweets = shares

                weighted_interactions = (
                    (likes * 1) + (replies * 2) + (retweets * 3) + (quotes * 3)
                )
                author_tier = post.get("author_tier", 3)
                # B-1：热度统一为 ABM 公式 log1p(传播/互动人数)/log1p(N) × TIER_WEIGHT
                s_popularity_weighted = popularity_score(
                    weighted_interactions,
                    population_n,
                    tier=author_tier,
                    tier_weight=SocialRecSys.TIER_WEIGHT,
                )

                # S_random
                s_random = random.random()

                raw_interest.append(s_similarity)
                raw_pop.append(s_popularity_weighted)
                raw_time_vals.append(s_time)
                raw_rand.append(s_random)

                post["stats"] = stats
                scored_posts.append(post)

            # Min-Max normalize each dimension across the batch（core.scoring 共享）
            ni = min_max_normalize(raw_interest).tolist()
            np_vals = min_max_normalize(raw_pop).tolist()
            nt = min_max_normalize(raw_time_vals).tolist()
            nr = min_max_normalize(raw_rand).tolist()

            # Total Score using class-level weights（core.scoring 共享加权）
            for i, post in enumerate(scored_posts):
                post["score"] = weighted_score(
                    ni[i],
                    np_vals[i],
                    nt[i],
                    nr[i],
                    SocialRecSys.W_BELIEF,
                    SocialRecSys.W_POP,
                    SocialRecSys.W_CHRONO,
                    SocialRecSys.W_RAND,
                )

            # 4. Sort and return
            scored_posts.sort(key=lambda x: x["score"], reverse=True)
            # Return results without scores, only post info
            results = scored_posts[:limit]
            for p in results:
                p.pop("score", None)
            return results


# Singleton instance
recsys = SocialRecSys()
