from datetime import datetime
from typing import List, Optional, Dict, Any

from pydantic import BaseModel


# --- User ---
class UserCreate(BaseModel):
    username: str
    nickname: str
    bio: Optional[str] = None
    user_info: Optional[Dict[str, Any]] = None
    # B-5：tier 与 belief_text 全链路写入，供在线打分使用
    tier: int = 3
    belief_text: Optional[str] = None


class UserResponse(BaseModel):
    id: int
    username: str
    nickname: str
    bio: Optional[str] = None
    created_at: datetime


# --- Post ---
class PostCreate(BaseModel):
    user_id: int
    content: Optional[str] = None
    type: str = "ORIGINAL"  # ORIGINAL, REPOST, QUOTE
    ref_id: Optional[int] = None


class CommentResponse(BaseModel):
    id: int
    user_id: int
    post_id: int
    content: str
    like_count: int
    created_at: datetime
    author_nickname: Optional[str] = None
    author_type: Optional[str] = None
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
    author_type: Optional[str] = None
    comments: List[CommentResponse] = []
    is_liked: bool = False
    is_reposted: bool = False


# --- Trace ---
class TraceResponse(BaseModel):
    id: int
    user_id: int
    user_nickname: str
    action_type: str
    action_details: Optional[Dict[str, Any]] = None
    created_at: datetime


# --- Action ---
class ActionResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None


# --- Specific Action Requests ---
class CreatePostRequest(BaseModel):
    user_id: int
    content: str


class CreateCommentRequest(BaseModel):
    user_id: int
    post_id: int
    content: str


class LikePostRequest(BaseModel):
    user_id: int
    post_id: int


class RepostRequest(BaseModel):
    user_id: int
    post_id: int


class QuoteRequest(BaseModel):
    user_id: int
    post_id: int
    content: str


class LikeCommentRequest(BaseModel):
    user_id: int
    comment_id: int


class DoNothingRequest(BaseModel):
    user_id: int


# --- Feed ---
class FeedResponse(BaseModel):
    posts: List[PostResponse]


# --- Time ---
class TimeConfig(BaseModel):
    mode: str  # "step" or "time"
    start_time: datetime
    time_scale: float = 1.0
