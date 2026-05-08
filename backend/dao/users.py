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


async def create_user(
    username: str,
    nickname: str,
    bio: Optional[str] = None,
    user_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    async with get_db_connection() as db:
        cursor = await db.execute(
            (
                "INSERT INTO users (username, nickname, bio, user_info) "
                "VALUES (?, ?, ?, ?) RETURNING *"
            ),
            (username, nickname, bio, json.dumps(user_info or {}, ensure_ascii=False)),
        )
        new_user = await cursor.fetchone()
        await db.commit()
        return dict(new_user)
