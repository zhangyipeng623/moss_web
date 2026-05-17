import asyncio
import json
import random
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
from backend.dao.database import get_db_connection, format_stats
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
    TIER_WEIGHT: Dict[int, float] = {1: 0.4, 2: 0.7, 3: 1.0, 4: 1.5, 5: 2.0}
    _embedding_model_name: str = "all-MiniLM-L6-v2"

    @classmethod
    def configure(cls, recommender_config, embedding_config) -> None:
        """从 CalibrationProfile 注入推荐参数。"""
        cls.W_BELIEF = recommender_config.weights.w_interest
        cls.W_POP = recommender_config.weights.w_popularity
        cls.W_CHRONO = recommender_config.weights.w_time
        cls.W_RAND = recommender_config.weights.w_random
        cls.DECAY_LAMBDA = recommender_config.decay_lambda
        cls.TIER_WEIGHT = dict(recommender_config.tier_weight)
        cls._embedding_model_name = embedding_config.model_name

    def __init__(self):
        # Lazy load model
        self._model = None

    @property
    def model(self):
        if self._model is None:
            logger.info("Loading SentenceTransformer model...")
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Model loaded.")
        return self._model

    async def add_post_vector(self, post_id: int, content: str):
        if not content:
            return

        loop = asyncio.get_running_loop()
        # Use a lambda or partial to pass show_progress_bar
        embedding = await loop.run_in_executor(
            None, lambda: self.model.encode(content, show_progress_bar=False)
        )
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
            # 1. Get user bio for belief similarity
            async with db.execute(
                "SELECT bio FROM users WHERE id = ?", (user_id,)
            ) as cursor:
                user_row = await cursor.fetchone()
                user_bio = user_row["bio"] if user_row else ""

            posts = []
            distances = {}  # Map post_id -> distance

            # 2. Candidate Generation
            if user_bio:
                # Check if there are any vectors in the vec table to avoid crash
                async with db.execute("SELECT count(*) FROM posts_vec") as cursor:
                    row = await cursor.fetchone()
                    vec_count = row[0]

                if vec_count > 0:
                    # Vector Search
                    loop = asyncio.get_running_loop()
                    # Use a lambda to pass show_progress_bar
                    user_embedding = await loop.run_in_executor(
                        None,
                        lambda: self.model.encode(user_bio, show_progress_bar=False),
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
                                WHERE p.id IN ({placeholders})
                            """,
                                post_ids,
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
                    ORDER BY p.created_at DESC, p.id DESC LIMIT 100
                """) as cursor:
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
                s_time = np.exp(-SocialRecSys.DECAY_LAMBDA * delta_hours)

                # S_similarity
                s_similarity = 0.0
                if user_bio and post["id"] in distances:
                    dist = distances[post["id"]]
                    s_similarity = 1.0 / (1.0 + dist)

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
                s_popularity = np.log1p(weighted_interactions) / 10.0
                s_popularity = min(s_popularity, 1.0)

                author_tier = post.get("author_tier", 3)
                s_popularity_weighted = s_popularity * SocialRecSys.TIER_WEIGHT.get(
                    author_tier, 1.0
                )

                # S_random
                s_random = random.random()

                raw_interest.append(s_similarity)
                raw_pop.append(s_popularity_weighted)
                raw_time_vals.append(s_time)
                raw_rand.append(s_random)

                post["stats"] = stats
                scored_posts.append(post)

            # Min-Max normalize each dimension across the batch
            def _norm(vals):
                v_min, v_max = min(vals), max(vals)
                if v_max - v_min < 1e-8:
                    return [0.5] * len(vals)
                return [(v - v_min) / (v_max - v_min) for v in vals]

            ni = _norm(raw_interest)
            np_vals = _norm(raw_pop)
            nt = _norm(raw_time_vals)
            nr = _norm(raw_rand)

            # Total Score using class-level weights
            for i, post in enumerate(scored_posts):
                score = (
                    SocialRecSys.W_BELIEF * ni[i]
                    + SocialRecSys.W_POP * np_vals[i]
                    + SocialRecSys.W_CHRONO * nt[i]
                    + SocialRecSys.W_RAND * nr[i]
                )
                post["score"] = score

            # 4. Sort and return
            scored_posts.sort(key=lambda x: x["score"], reverse=True)
            # Return results without scores, only post info
            results = scored_posts[:limit]
            for p in results:
                p.pop("score", None)
            return results


# Singleton instance
recsys = SocialRecSys()
