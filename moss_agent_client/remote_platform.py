import logging
from typing import Dict, Any, List, Optional

import httpx

from moss_agent_client.schemas import UserResponse, PostResponse, ActionResponse

logger = logging.getLogger(__name__)


class RemotePlatform:
    def __init__(self, base_url: str, client: Optional[httpx.AsyncClient] = None):
        self.base_url = base_url
        if client:
            self.client = client
        else:
            self.client = httpx.AsyncClient(base_url=base_url, timeout=30.0)
        self.user_data: Optional[UserResponse] = None

    async def close(self):
        await self.client.aclose()

    async def register_or_login(
        self,
        username: str,
        nickname: str,
        bio: str,
        user_info: Optional[Dict[str, Any]] = None,
    ) -> UserResponse:
        resp = await self.client.post(
            "/api/v1/login",
            json={
                "username": username,
                "nickname": nickname,
                "bio": bio,
                "user_info": user_info or {},
            },
        )
        resp.raise_for_status()
        self.user_data = UserResponse(**resp.json())
        logger.info(f"Logged in as {self.user_data.username} (ID: {self.user_data.id})")
        return self.user_data

    async def get_feed(self) -> List[PostResponse]:
        if not self.user_data:
            raise RuntimeError("Not logged in")
        resp = await self.client.get(f"/api/v1/feed?user_id={self.user_data.id}")
        resp.raise_for_status()
        return [PostResponse(**item) for item in resp.json()]

    async def get_time(self):
        resp = await self.client.get("/api/v1/time")
        resp.raise_for_status()
        return resp.json()

    async def set_system_config(
        self,
        mode: str,  # "step" or "time"
        start_time,
        time_scale: float = 1.0,
    ) -> Dict[str, Any]:
        # Convert datetime to ISO format string for JSON serialization
        if hasattr(start_time, "isoformat"):
            start_time = start_time.isoformat()

        config = {
            "mode": mode,
            "start_time": start_time,
            "time_scale": time_scale,
        }
        # Corrected endpoint from /api/v1/system/config to /api/v1/time/config
        resp = await self.client.post("/api/v1/time/config", json=config)
        resp.raise_for_status()
        return resp.json()

    async def increment_step(self) -> Dict[str, Any]:
        resp = await self.client.post("/api/v1/time/step")
        resp.raise_for_status()
        return resp.json()

    async def get_post(self, post_id: int) -> PostResponse:
        params = None
        if self.user_data is not None:
            params = {"user_id": self.user_data.id}
        resp = await self.client.get(f"/api/v1/posts/{post_id}", params=params)
        resp.raise_for_status()
        return PostResponse(**resp.json())

    async def create_post(self, content: str) -> ActionResponse:
        if not self.user_data:
            raise RuntimeError("Not logged in")
        user_id = self.user_data.id
        payload = {"user_id": user_id, "content": content}
        resp = await self.client.post("/api/v1/posts", json=payload)
        resp.raise_for_status()
        return ActionResponse(**resp.json())

    async def create_comment(self, post_id: int, content: str) -> ActionResponse:
        if not self.user_data:
            raise RuntimeError("Not logged in")
        user_id = self.user_data.id
        payload = {
            "user_id": user_id,
            "post_id": post_id,
            "content": content,
        }
        resp = await self.client.post("/api/v1/comments", json=payload)
        resp.raise_for_status()
        return ActionResponse(**resp.json())

    async def like_post(self, post_id: int) -> ActionResponse:
        if not self.user_data:
            raise RuntimeError("Not logged in")
        user_id = self.user_data.id
        payload = {"user_id": user_id, "post_id": post_id}
        resp = await self.client.post("/api/v1/posts/like", json=payload)
        resp.raise_for_status()
        return ActionResponse(**resp.json())

    async def repost(self, post_id: int) -> ActionResponse:
        if not self.user_data:
            raise RuntimeError("Not logged in")
        user_id = self.user_data.id
        payload = {
            "user_id": user_id,
            "post_id": post_id,
        }
        resp = await self.client.post("/api/v1/posts/repost", json=payload)
        resp.raise_for_status()
        return ActionResponse(**resp.json())

    async def quote(self, post_id: int, content: str) -> ActionResponse:
        if not self.user_data:
            raise RuntimeError("Not logged in")
        user_id = self.user_data.id
        payload = {
            "user_id": user_id,
            "post_id": post_id,
            "content": content,
        }
        resp = await self.client.post("/api/v1/posts/quote", json=payload)
        resp.raise_for_status()
        return ActionResponse(**resp.json())

    async def like_comment(self, comment_id: int) -> ActionResponse:
        if not self.user_data:
            raise RuntimeError("Not logged in")
        user_id = self.user_data.id
        payload = {"user_id": user_id, "comment_id": comment_id}
        resp = await self.client.post("/api/v1/comments/like", json=payload)
        resp.raise_for_status()
        return ActionResponse(**resp.json())

    async def do_nothing(self) -> ActionResponse:
        if not self.user_data:
            raise RuntimeError("Not logged in")
        user_id = self.user_data.id
        payload = {"user_id": user_id}
        resp = await self.client.post("/api/v1/actions/do_nothing", json=payload)
        resp.raise_for_status()
        return ActionResponse(**resp.json())
