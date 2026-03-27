import json
from typing import Optional, Dict, List, Any

import aiosqlite
from backend.dao.database import get_db_connection, format_stats
from backend.services.logger_service import logger
from backend.services.time_service import time_service


async def _update_post_stats(db, post_id, field, delta):
    async with db.execute("SELECT stats FROM posts WHERE id = ?", (post_id,)) as cursor:
        row = await cursor.fetchone()
        if not row:
            return
        stats_json = row["stats"]
        stats = format_stats(stats_json)

    stats[field] = stats.get(field, 0) + delta
    new_stats_json = json.dumps(stats)

    await db.execute(
        "UPDATE posts SET stats = ? WHERE id = ?", (new_stats_json, post_id)
    )


async def _record_trace(db, user_id, action_type, details=None):
    if details is None:
        details = {}
    details_json = json.dumps(details)
    current_time = time_service.get_current_time()
    await db.execute(
        "INSERT INTO trace (user_id, action_type, action_details, created_at) VALUES (?, ?, ?, ?)",
        (user_id, action_type, details_json, current_time),
    )


async def get_post_detail(
    post_id: int, user_id: Optional[int] = None
) -> Dict[str, Any]:
    async with get_db_connection() as db:
        async with db.execute(
            """
            SELECT p.*, u.nickname as author_nickname, u.type as author_type
            FROM posts p
            JOIN users u ON p.user_id = u.id 
            WHERE p.id = ?
        """,
            (post_id,),
        ) as cursor:
            post = await cursor.fetchone()

        if not post:
            return None

        # Fetch comments
        async with db.execute(
            """
            SELECT c.*, u.nickname as author_nickname, u.type as author_type
            FROM comments c
            JOIN users u ON c.user_id = u.id
            WHERE c.post_id = ?
            ORDER BY c.created_at ASC, c.id ASC
            """,
            (post_id,),
        ) as cursor:
            comments = await cursor.fetchall()

        post_dict = dict(post)
        post_dict["stats"] = format_stats(post_dict["stats"])

        # Check interaction for post
        post_dict["is_liked"] = False
        post_dict["is_reposted"] = False

        if user_id:
            # Check like
            async with db.execute(
                "SELECT 1 FROM interactions WHERE user_id = ? AND target_id = ? AND target_type = 'POST'",
                (user_id, post_id),
            ) as cursor:
                if await cursor.fetchone():
                    post_dict["is_liked"] = True

            # Check repost (including QUOTE as it's a type of repost/share?)
            # User said "forwarded" (repost). Usually QUOTE is also a share.
            # But let's check for REPOST type specifically or both?
            # The schema has is_reposted. I'll check for type='REPOST'
            async with db.execute(
                "SELECT 1 FROM posts WHERE user_id = ? AND ref_id = ? AND type = 'REPOST'",
                (user_id, post_id),
            ) as cursor:
                if await cursor.fetchone():
                    post_dict["is_reposted"] = True

        # Process comments
        comments_list = []
        for c in comments:
            c_dict = dict(c)
            c_dict["is_liked"] = False
            if user_id:
                async with db.execute(
                    "SELECT 1 FROM interactions WHERE user_id = ? AND target_id = ? AND target_type = 'COMMENT'",
                    (user_id, c_dict["id"]),
                ) as cursor:
                    if await cursor.fetchone():
                        c_dict["is_liked"] = True
            comments_list.append(c_dict)

        post_dict["comments"] = comments_list
        return post_dict


async def create_post(user_id: int, content: str) -> int:
    async with get_db_connection() as db:
        try:
            current_time = time_service.get_current_time()
            cursor = await db.execute(
                "INSERT INTO posts (user_id, content, type, created_at) VALUES (?, ?, ?, ?)",
                (user_id, content, "ORIGINAL", current_time),
            )
            post_id = cursor.lastrowid

            # Trace
            await _record_trace(
                db,
                user_id,
                "create_post",
                {"post_id": post_id, "content": content},
            )

            await db.commit()
            return post_id
        except Exception as e:
            logger.error(f"Error creating post: {e}")
            raise e


async def create_comment(user_id: int, post_id: int, content: str):
    async with get_db_connection() as db:
        try:
            current_time = time_service.get_current_time()
            await db.execute(
                "INSERT INTO comments (user_id, post_id, content, created_at) VALUES (?, ?, ?, ?)",
                (user_id, post_id, content, current_time),
            )

            # Update stats
            await _update_post_stats(db, post_id, "reply_count", 1)

            # Trace
            await _record_trace(
                db,
                user_id,
                "create_comment",
                {"post_id": post_id, "content": content},
            )

            await db.commit()
        except Exception as e:
            logger.error(f"Error creating comment: {e}")
            raise e


async def like_post(user_id: int, post_id: int) -> Dict[str, Any]:
    async with get_db_connection() as db:
        try:
            # Check if already liked
            async with db.execute(
                "SELECT 1 FROM interactions WHERE user_id = ? AND target_id = ? AND target_type = 'POST'",
                (user_id, post_id),
            ) as cursor:
                if await cursor.fetchone():
                    return {
                        "status": "failed",
                        "message": "You have already liked this post.",
                    }

            try:
                await db.execute(
                    "INSERT INTO interactions (user_id, target_id, target_type) VALUES (?, ?, ?)",
                    (user_id, post_id, "POST"),
                )
                await _update_post_stats(db, post_id, "like_count", 1)

                # Trace
                await _record_trace(db, user_id, "like_post", {"post_id": post_id})

                await db.commit()
                return {"status": "success", "message": "Post liked"}
            except aiosqlite.IntegrityError:
                return {
                    "status": "failed",
                    "message": "You have already liked this post.",
                }
        except Exception as e:
            logger.error(f"Error liking post: {e}")
            raise e


async def repost(user_id: int, post_id: int) -> Dict[str, Any]:
    async with get_db_connection() as db:
        try:
            # Check if already reposted
            async with db.execute(
                "SELECT 1 FROM posts WHERE user_id = ? AND ref_id = ? AND type = 'REPOST'",
                (user_id, post_id),
            ) as cursor:
                if await cursor.fetchone():
                    return {
                        "status": "failed",
                        "message": "You have already reposted this post.",
                    }

            # Fetch original post content
            async with db.execute(
                "SELECT content FROM posts WHERE id = ?",
                (post_id,),
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return {
                        "status": "failed",
                        "message": "Original post not found.",
                    }
                original_content = row["content"]

            repost_content = f"转发 post #{post_id}:{original_content}"

            current_time = time_service.get_current_time()
            cursor = await db.execute(
                "INSERT INTO posts (user_id, content, type, ref_id, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    user_id,
                    repost_content,
                    "REPOST",
                    post_id,
                    current_time,
                ),
            )
            new_post_id = cursor.lastrowid
            await _update_post_stats(db, post_id, "share_count", 1)
            await _update_post_stats(db, post_id, "retweet_count", 1)

            # Trace
            await _record_trace(
                db,
                user_id,
                "repost",
                {
                    "post_id": new_post_id,
                    "original_post_id": post_id,
                    "content": repost_content,
                },
            )

            await db.commit()
            return {
                "status": "success",
                "message": "Reposted",
                "data": {"post_id": new_post_id},
            }
        except Exception as e:
            logger.error(f"Error reposting: {e}")
            raise e


async def quote(user_id: int, post_id: int, content: str) -> Dict[str, Any]:
    async with get_db_connection() as db:
        try:
            # Allow multiple quotes, so no duplicate check here.

            current_time = time_service.get_current_time()
            cursor = await db.execute(
                "INSERT INTO posts (user_id, content, type, ref_id, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    user_id,
                    content,
                    "QUOTE",
                    post_id,
                    current_time,
                ),
            )
            new_post_id = cursor.lastrowid
            await _update_post_stats(db, post_id, "share_count", 1)
            await _update_post_stats(db, post_id, "quote_count", 1)

            # Trace
            await _record_trace(
                db,
                user_id,
                "quote",
                {
                    "post_id": new_post_id,
                    "original_post_id": post_id,
                    "content": content,
                },
            )

            await db.commit()
            return {
                "status": "success",
                "message": "Quoted",
                "data": {"post_id": new_post_id},
            }
        except Exception as e:
            logger.error(f"Error quoting: {e}")
            raise e


async def like_comment(user_id: int, comment_id: int) -> Dict[str, Any]:
    async with get_db_connection() as db:
        try:
            # Check if already liked
            async with db.execute(
                "SELECT 1 FROM interactions WHERE user_id = ? AND target_id = ? AND target_type = 'COMMENT'",
                (user_id, comment_id),
            ) as cursor:
                if await cursor.fetchone():
                    return {
                        "status": "failed",
                        "message": "You have already liked this comment.",
                    }

            try:
                await db.execute(
                    "INSERT INTO interactions (user_id, target_id, target_type) VALUES (?, ?, ?)",
                    (user_id, comment_id, "COMMENT"),
                )
                # Update comment like count
                await db.execute(
                    "UPDATE comments SET like_count = like_count + 1 WHERE id = ?",
                    (comment_id,),
                )

                # Trace
                await _record_trace(
                    db,
                    user_id,
                    "like_comment",
                    {"comment_id": comment_id},
                )

                await db.commit()
                return {"status": "success", "message": "Comment liked"}
            except aiosqlite.IntegrityError:
                return {
                    "status": "failed",
                    "message": "You have already liked this comment.",
                }
        except Exception as e:
            logger.error(f"Error liking comment: {e}")
            raise e


async def do_nothing(user_id: int) -> Dict[str, Any]:
    async with get_db_connection() as db:
        await _record_trace(db, user_id, "do_nothing")
        await db.commit()
    return {"status": "success", "message": "Did nothing"}


async def get_all_posts(limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
    async with get_db_connection() as db:
        async with db.execute(
            """
            SELECT p.*, u.nickname as author_nickname, u.type as author_type
            FROM posts p
            JOIN users u ON p.user_id = u.id
            ORDER BY p.created_at DESC, p.id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ) as cursor:
            posts = await cursor.fetchall()

        result = []
        for post in posts:
            post_dict = dict(post)
            post_dict["stats"] = format_stats(post_dict["stats"])
            post_dict["comments"] = []  # Don't fetch comments for all posts list
            post_dict["is_liked"] = False
            post_dict["is_reposted"] = False
            result.append(post_dict)
        return result


async def get_recent_traces() -> List[Dict[str, Any]]:
    async with get_db_connection() as db:
        async with db.execute(
            """
            SELECT t.*, u.nickname as user_nickname
            FROM trace t
            JOIN users u ON t.user_id = u.id
            ORDER BY t.created_at DESC, t.id DESC
            LIMIT 50
            """,
        ) as cursor:
            traces = await cursor.fetchall()

        result = []
        for t in traces:
            t_dict = dict(t)
            try:
                t_dict["action_details"] = json.loads(t_dict["action_details"])
            except (json.JSONDecodeError, TypeError):
                t_dict["action_details"] = {}
            result.append(t_dict)
        return result
