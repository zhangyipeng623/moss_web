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

# Weights
W_CHRONO = 0.3
W_BELIEF = 0.3
W_POP = 0.2
W_RAND = 0.2


class SocialRecSys:
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
                    "INSERT INTO posts_vss(rowid, content_embedding) VALUES (?, ?)",
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
                # Check if there are any vectors in the VSS table to avoid crash
                async with db.execute("SELECT count(*) FROM posts_vss") as cursor:
                    row = await cursor.fetchone()
                    vss_count = row[0]

                if vss_count > 0:
                    # Vector Search
                    loop = asyncio.get_running_loop()
                    # Use a lambda to pass show_progress_bar
                    user_embedding = await loop.run_in_executor(
                        None,
                        lambda: self.model.encode(user_bio, show_progress_bar=False),
                    )
                    user_embedding_json = json.dumps(user_embedding.tolist())

                    # Search top 100 similar posts using sqlite-vss
                    # Note: vss_search takes the vector JSON
                    try:
                        async with db.execute(
                            """
                            SELECT rowid, distance 
                            FROM posts_vss 
                            WHERE vss_search(content_embedding, ?)
                            LIMIT 100
                        """,
                            (user_embedding_json,),
                        ) as cursor:
                            vss_results = await cursor.fetchall()

                        if vss_results:
                            post_ids = [row[0] for row in vss_results]
                            distances = {row[0]: row[1] for row in vss_results}

                            # Fetch full post data for these IDs
                            placeholders = ",".join("?" * len(post_ids))
                            async with db.execute(
                                f"""
                                SELECT p.*, u.nickname as author_nickname
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
                    SELECT p.*,  u.nickname as author_nickname
                    FROM posts p 
                    JOIN users u ON p.user_id = u.id 
                    ORDER BY p.created_at DESC, p.id DESC LIMIT 100
                """) as cursor:
                    posts = [dict(row) for row in await cursor.fetchall()]

            if not posts:
                return []

            # 3. Calculate Scores
            scored_posts = []
            current_time = time_service.get_current_time()

            for post in posts:
                # S_time: Decay based on time difference
                # Parse created_at (assuming 'YYYY-MM-DD HH:MM:SS')
                try:
                    # SQLite default format might vary slightly, usually it is this:
                    post_time = datetime.strptime(
                        str(post["created_at"]), "%Y-%m-%d %H:%M:%S"
                    )
                except ValueError:
                    # Try with ISO format if needed or handle milliseconds
                    try:
                        post_time = datetime.fromisoformat(str(post["created_at"]))
                    except:
                        post_time = current_time  # Fallback

                # Calculate time difference in hours/days/seconds
                # Use a decay function: exp(-lambda * delta_hours)
                # Half-life = 1 hour => lambda = ln(2)/1 approx 0.693
                delta = current_time - post_time
                delta_hours = max(0, delta.total_seconds() / 3600.0)
                s_time = np.exp(-0.693 * delta_hours)

                # S_similarity
                s_similarity = 0.0
                if user_bio and post["id"] in distances:
                    dist = distances[post["id"]]
                    # Assuming L2 distance. Sim = 1 / (1 + dist) to map [0, inf) to [1, 0)
                    s_similarity = 1.0 / (1.0 + dist)
                elif user_bio:
                    pass

                # S_popularity
                # Weights: Like=1, Reply=2, Retweet=3, Quote=3
                stats = format_stats(post["stats"])
                likes = stats.get("like_count", 0)
                replies = stats.get("reply_count", 0)
                retweets = stats.get("retweet_count", 0)
                quotes = stats.get("quote_count", 0)
                shares = stats.get("share_count", 0)

                # Backward compatibility for old data (where retweets/quotes are not tracked separately)
                if retweets == 0 and quotes == 0 and shares > 0:
                    retweets = shares

                weighted_interactions = (
                    (likes * 1) + (replies * 2) + (retweets * 3) + (quotes * 3)
                )

                # Normalize to 0-1 using log1p.
                # log1p(weighted_interactions) / 10.0 handles up to ~22k weighted interactions
                s_popularity = np.log1p(weighted_interactions) / 10.0
                s_popularity = min(s_popularity, 1.0)

                # S_random
                s_random = random.random()

                # Total Score
                score = (
                    (W_CHRONO * s_time)
                    + (W_BELIEF * s_similarity)
                    + (W_POP * s_popularity)
                    + (W_RAND * s_random)
                )

                post["score"] = score
                post["stats"] = stats
                scored_posts.append(post)

            # 4. Sort and return
            scored_posts.sort(key=lambda x: x["score"], reverse=True)
            # Return results without scores, only post info
            results = scored_posts[:limit]
            for p in results:
                p.pop("score", None)
            return results


# Singleton instance
recsys = SocialRecSys()
