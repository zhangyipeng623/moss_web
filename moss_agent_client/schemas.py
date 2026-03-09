from datetime import datetime
from typing import List, Optional, Dict, Any

from pydantic import BaseModel


# --- User ---
class UserResponse(BaseModel):
    id: int
    username: str
    nickname: str
    bio: Optional[str] = None
    created_at: datetime


# --- Post ---
class CommentResponse(BaseModel):
    id: int
    user_id: int
    post_id: int
    content: str
    like_count: int
    created_at: datetime
    author_nickname: Optional[str] = None
    is_liked: bool = False


class PostResponse(BaseModel):
    id: int
    user_id: int
    content: Optional[str]
    type: str
    ref_id: Optional[int]
    stats: Dict[str, int]
    created_at: datetime
    author_nickname: Optional[str] = None
    comments: List[CommentResponse] = []
    is_liked: bool = False
    is_reposted: bool = False


# --- Action ---
class ActionResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None


# --- System Time ---
class SystemTimeConfig(BaseModel):
    mode: str
    start_time: datetime
    time_scale: float = 1.0
