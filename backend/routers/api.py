from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from backend.dao import posts, users
from backend.schemas.schemas import (
    ActionResponse,
    PostResponse,
    TraceResponse,
    UserCreate,
    UserResponse,
    TimeConfig,
    CreatePostRequest,
    CreateCommentRequest,
    LikePostRequest,
    RepostRequest,
    QuoteRequest,
    LikeCommentRequest,
    DoNothingRequest,
)
from backend.services.logger_service import logger
from backend.services.social_recsys import recsys
from backend.services.time_service import time_service, TimeMode

router = APIRouter()


@router.get("/api/v1/time")
async def get_current_time():
    return {
        "current_time": time_service.get_current_time(),
        "mode": time_service.mode,
        "step": time_service.current_step,
    }


@router.post("/api/v1/time/config")
async def configure_time(config: TimeConfig):
    try:
        mode = TimeMode(config.mode)
        time_service.initialize(
            start_time=config.start_time, mode=mode, time_scale=config.time_scale
        )
        return {
            "status": "success",
            "message": f"Time service configured to {mode} mode",
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid time mode")


@router.post("/api/v1/time/step")
async def update_step():
    time_service.update_step()
    return {"status": "success", "current_time": time_service.get_current_time()}


@router.post("/api/v1/login", response_model=UserResponse)
async def login(user_in: UserCreate):
    tier = max(1, min(5, int(user_in.tier)))
    user = await users.get_user_by_username(user_in.username)
    if user:
        # B-4：同库复用时刷新 tier/belief_text，不残留旧值
        return await users.update_user_tier_belief(
            user_in.username, tier, user_in.belief_text
        )

    # Register（B-5：tier / belief_text 透传给 DAO）
    new_user = await users.create_user(
        user_in.username,
        user_in.nickname,
        user_in.bio,
        user_in.user_info,
        tier=tier,
        belief_text=user_in.belief_text,
    )
    return new_user


@router.get("/api/v1/feed", response_model=List[PostResponse])
async def get_feed(user_id: int, limit: int = 20):
    posts = await recsys.get_recommended_feed(user_id, limit)
    return posts


@router.get("/api/v1/posts", response_model=List[PostResponse])
async def get_all_posts(limit: int = 20, offset: int = 0):
    all_posts = await posts.get_all_posts(limit, offset)
    return all_posts


@router.get("/api/v1/traces", response_model=List[TraceResponse])
async def get_recent_traces():
    traces = await posts.get_recent_traces()
    return traces


@router.get("/api/v1/posts/{post_id}", response_model=PostResponse)
async def get_post(post_id: int, user_id: Optional[int] = Query(None)):
    post = await posts.get_post_detail(post_id, user_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return post


@router.post("/api/v1/posts", response_model=ActionResponse)
async def create_post(request: CreatePostRequest):
    try:
        post_id = await posts.create_post(request.user_id, request.content)

        # Add vector
        await recsys.add_post_vector(post_id, request.content)

        return ActionResponse(
            status="success", message="Post created", data={"post_id": post_id}
        )
    except Exception as e:
        logger.error(f"Error creating post: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/comments", response_model=ActionResponse)
async def create_comment(request: CreateCommentRequest):
    try:
        await posts.create_comment(
            request.user_id, request.post_id, request.content
        )
        return ActionResponse(status="success", message="Comment created")
    except Exception as e:
        logger.error(f"Error creating comment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/posts/like", response_model=ActionResponse)
async def like_post(request: LikePostRequest):
    try:
        result = await posts.like_post(request.user_id, request.post_id)
        return ActionResponse(**result)
    except Exception as e:
        logger.error(f"Error liking post: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/posts/repost", response_model=ActionResponse)
async def repost(request: RepostRequest):
    try:
        result = await posts.repost(request.user_id, request.post_id)

        # Repost has no content, so no vector to add.

        return ActionResponse(**result)
    except Exception as e:
        logger.error(f"Error reposting: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/posts/quote", response_model=ActionResponse)
async def quote(request: QuoteRequest):
    try:
        result = await posts.quote(
            request.user_id, request.post_id, request.content
        )

        if result["status"] == "success":
            new_post_id = result["data"]["post_id"]
            await recsys.add_post_vector(new_post_id, request.content)

        return ActionResponse(**result)
    except Exception as e:
        logger.error(f"Error quoting: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/comments/like", response_model=ActionResponse)
async def like_comment(request: LikeCommentRequest):
    try:
        result = await posts.like_comment(request.user_id, request.comment_id)
        return ActionResponse(**result)
    except Exception as e:
        logger.error(f"Error liking comment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/actions/do_nothing", response_model=ActionResponse)
async def do_nothing(request: DoNothingRequest):
    try:
        result = await posts.do_nothing(request.user_id)
        return ActionResponse(**result)
    except Exception as e:
        logger.error(f"Error doing nothing: {e}")
        raise HTTPException(status_code=500, detail=str(e))
