import json
from typing import Dict, Any, Optional

from .database import get_db_connection


async def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    async with get_db_connection() as db:
        async with db.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ) as cursor:
            user = await cursor.fetchone()
            if user:
                return dict(user)
            return None


async def update_user_tier_belief(
    username: str,
    tier: int,
    belief_text: Optional[str] = None,
) -> Dict[str, Any]:
    """B-4：登录命中已存在用户时刷新 tier/belief_text（同库复用不残留旧值）。"""
    async with get_db_connection() as db:
        await db.execute(
            "UPDATE users SET tier = ?, belief_text = ? WHERE username = ?",
            (tier, belief_text or "", username),
        )
        await db.commit()
    user = await get_user_by_username(username)
    if user is None:
        raise RuntimeError(f"更新后仍找不到用户：{username}")
    return user


async def create_user(
    username: str,
    nickname: str,
    bio: Optional[str] = None,
    user_info: Optional[Dict[str, Any]] = None,
    tier: int = 3,
    belief_text: Optional[str] = None,
) -> Dict[str, Any]:
    async with get_db_connection() as db:
        cursor = await db.execute(
            (
                "INSERT INTO users (username, nickname, bio, user_info, tier, belief_text) "
                "VALUES (?, ?, ?, ?, ?, ?) RETURNING *"
            ),
            (
                username,
                nickname,
                bio or "",
                json.dumps(user_info or {}, ensure_ascii=False),
                tier,
                belief_text or "",
            ),
        )
        new_user = await cursor.fetchone()
        await db.commit()
        return dict(new_user)
