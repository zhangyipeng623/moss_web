import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

import aiosqlite
import sqlite_vec
from backend.services.logger_service import logger


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "moss.db"


def get_database_path() -> Path:
    configured_path = os.environ.get("MOSS_DB_PATH")
    if configured_path:
        return Path(configured_path).resolve()
    return DEFAULT_DATABASE_PATH


async def load_vec(db):
    await db.enable_load_extension(True)
    await db.load_extension(sqlite_vec.loadable_path())
    await db.enable_load_extension(False)


@asynccontextmanager
async def get_db_connection():
    database_path = get_database_path()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(database_path) as db:
        await load_vec(db)
        db.row_factory = aiosqlite.Row
        yield db


async def init_db(embedding_dim: int = 384):
    database_path = get_database_path()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Initializing database...")
    if database_path.exists():
        logger.info(f"Database file {database_path} already exists. remove it.")
        database_path.unlink()
    else:
        logger.info(f"Database file {database_path} not found. Creating...")
    async with aiosqlite.connect(database_path) as db:
        await load_vec(db)
        # Enable foreign keys
        await db.execute("PRAGMA foreign_keys = ON")

        # Create users table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                nickname TEXT NOT NULL,
                type TEXT DEFAULT 'AGENT',
                bio TEXT,
                user_info TEXT,
                tier INTEGER DEFAULT 3,
                belief_text TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create posts table
        # stats is a JSON field storing like_count, reply_count, share_count
        await db.execute("""
            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                content TEXT,
                type TEXT DEFAULT 'ORIGINAL', -- ORIGINAL, REPOST, QUOTE
                ref_id INTEGER, -- Reference to another post for repost/quote
                stats TEXT DEFAULT '{}', -- JSON string
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)

        # Create posts_vec virtual table（cosine 度量：distance = 1 - cosθ，
        # 与 ABM/在线统一使用余弦相似度，见 docs/plan/在线仿真与参数一致性方案.md B-3）
        try:
            await db.execute(
                f"""CREATE VIRTUAL TABLE IF NOT EXISTS posts_vec USING vec0(
                    content_embedding float[{embedding_dim}] distance_metric=cosine
                )"""
            )
        except Exception as e:
            logger.warning(
                f"Failed to create vec table (might already exist or extension issue): {e}"
            )

        # Create comments table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                post_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                like_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (post_id) REFERENCES posts (id)
            )
        """)

        # Create interactions table (likes, etc.)
        # target_type: POST, COMMENT
        await db.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                target_type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                UNIQUE(user_id, target_id, target_type)
            )
        """)

        # Create trace table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS trace (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                action_type TEXT NOT NULL,
                action_details TEXT, -- JSON string
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)

        # Initialize God user
        await db.execute("""
            INSERT OR IGNORE INTO users (username, nickname, type, bio, user_info)
            VALUES ('god', 'God', 'GOD', 'The Omniscient Observer', '{}')
        """)

        await db.commit()
    logger.info("Database initialized successfully.")


# Helper to format stats
def format_stats(stats_json: Optional[str]) -> Dict[str, int]:
    default_stats = {"like_count": 0, "reply_count": 0, "share_count": 0}
    if not stats_json:
        return default_stats
    try:
        return {**default_stats, **json.loads(stats_json)}
    except json.JSONDecodeError:
        return default_stats
