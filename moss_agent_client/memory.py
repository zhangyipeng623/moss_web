from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


@dataclass
class StaticContext:
    profile_text: str
    global_event_text: str


@dataclass
class FeedItemSnapshot:
    post_id: int
    author_name: str
    created_at: str
    content: str
    post_type: str
    ref_id: Optional[int] = None
    like_count: int = 0
    reply_count: int = 0
    share_count: int = 0
    retweet_count: int = 0
    quote_count: int = 0
    is_liked: bool = False
    is_reposted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "post_id": self.post_id,
            "author_name": self.author_name,
            "created_at": self.created_at,
            "content": self.content,
            "post_type": self.post_type,
            "ref_id": self.ref_id,
            "like_count": self.like_count,
            "reply_count": self.reply_count,
            "share_count": self.share_count,
            "retweet_count": self.retweet_count,
            "quote_count": self.quote_count,
            "is_liked": self.is_liked,
            "is_reposted": self.is_reposted,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeedItemSnapshot":
        return cls(
            post_id=int(payload.get("post_id", 0)),
            author_name=str(payload.get("author_name") or ""),
            created_at=str(payload.get("created_at") or ""),
            content=str(payload.get("content") or ""),
            post_type=str(payload.get("post_type") or ""),
            ref_id=payload.get("ref_id"),
            like_count=int(payload.get("like_count", 0)),
            reply_count=int(payload.get("reply_count", 0)),
            share_count=int(payload.get("share_count", 0)),
            retweet_count=int(payload.get("retweet_count", 0)),
            quote_count=int(payload.get("quote_count", 0)),
            is_liked=bool(payload.get("is_liked", False)),
            is_reposted=bool(payload.get("is_reposted", False)),
        )


@dataclass
class EnvironmentSnapshot:
    current_time: str
    feed_items: list[FeedItemSnapshot] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_time": self.current_time,
            "feed_items": [item.to_dict() for item in self.feed_items],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EnvironmentSnapshot":
        return cls(
            current_time=str(payload.get("current_time") or ""),
            feed_items=[
                FeedItemSnapshot.from_dict(item)
                for item in payload.get("feed_items", [])
            ],
        )

    def combined_text(self) -> str:
        return "\n".join(
            " ".join(
                [
                    item.author_name,
                    item.content,
                    item.post_type,
                    str(item.post_id),
                ]
            ).strip()
            for item in self.feed_items
        )


@dataclass
class ActionTrace:
    action_type: str
    post_id: Optional[int] = None
    comment_id: Optional[int] = None
    content: str = ""
    result_post_id: Optional[int] = None
    status: str = ""
    message: str = ""
    result_data: dict[str, Any] = field(default_factory=dict)
    raw_args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "post_id": self.post_id,
            "comment_id": self.comment_id,
            "content": self.content,
            "result_post_id": self.result_post_id,
            "status": self.status,
            "message": self.message,
            "result_data": self.result_data,
            "raw_args": self.raw_args,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ActionTrace":
        post_id = payload.get("post_id")
        comment_id = payload.get("comment_id")
        result_post_id = payload.get("result_post_id")

        try:
            parsed_post_id = int(post_id) if post_id is not None else None
        except (TypeError, ValueError):
            parsed_post_id = None

        try:
            parsed_comment_id = (
                int(comment_id) if comment_id is not None else None
            )
        except (TypeError, ValueError):
            parsed_comment_id = None

        try:
            parsed_result_post_id = (
                int(result_post_id) if result_post_id is not None else None
            )
        except (TypeError, ValueError):
            parsed_result_post_id = None

        return cls(
            action_type=str(payload.get("action_type") or ""),
            post_id=parsed_post_id,
            comment_id=parsed_comment_id,
            content=str(payload.get("content") or ""),
            result_post_id=parsed_result_post_id,
            status=str(payload.get("status") or ""),
            message=str(payload.get("message") or ""),
            result_data=dict(payload.get("result_data") or {}),
            raw_args=dict(payload.get("raw_args") or {}),
        )


@dataclass
class ShortTermRecord:
    round_id: int
    time: str
    feed_items: list[FeedItemSnapshot] = field(default_factory=list)
    actions: list[ActionTrace] = field(default_factory=list)
    output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "time": self.time,
            "feed_items": [item.to_dict() for item in self.feed_items],
            "actions": [item.to_dict() for item in self.actions],
            "output": self.output,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ShortTermRecord":
        return cls(
            round_id=int(payload.get("round_id", 0)),
            time=str(payload.get("time") or ""),
            feed_items=[
                FeedItemSnapshot.from_dict(item)
                for item in payload.get("feed_items", [])
            ],
            actions=[
                ActionTrace.from_dict(item)
                for item in payload.get("actions", [])
            ],
            output=str(payload.get("output") or ""),
        )


@dataclass
class ShortTermMemory:
    max_rounds: int = 3
    records: list[ShortTermRecord] = field(default_factory=list)

    def add(self, record: ShortTermRecord) -> None:
        self.records.append(record)
        if len(self.records) > self.max_rounds:
            self.records = self.records[-self.max_rounds :]


@dataclass
class AgentState:
    mood: str = "平静"
    emotion_reason: str = ""
    intensity: float = 0.0
    current_goal: str = ""
    focus_topics: list[str] = field(default_factory=list)
    stance_summary: str = ""
    attention_target: str = ""

    def update_from_payload(self, payload: "StateUpdatePayload") -> None:
        self.mood = payload.mood
        self.emotion_reason = payload.emotion_reason
        self.intensity = payload.intensity
        self.current_goal = payload.current_goal
        self.focus_topics = payload.focus_topics
        self.stance_summary = payload.stance_summary
        self.attention_target = payload.attention_target

    def to_dict(self) -> dict[str, Any]:
        return {
            "mood": self.mood,
            "emotion_reason": self.emotion_reason,
            "intensity": self.intensity,
            "current_goal": self.current_goal,
            "focus_topics": self.focus_topics,
            "stance_summary": self.stance_summary,
            "attention_target": self.attention_target,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentState":
        state = cls()
        state.mood = str(payload.get("mood") or state.mood)
        state.emotion_reason = str(payload.get("emotion_reason") or "")
        try:
            intensity = float(payload.get("intensity", state.intensity))
        except (TypeError, ValueError):
            intensity = state.intensity
        state.intensity = max(0.0, min(1.0, intensity))
        state.current_goal = str(payload.get("current_goal") or "")
        state.focus_topics = [
            str(item).strip()
            for item in payload.get("focus_topics", [])
            if str(item).strip()
        ]
        state.stance_summary = str(payload.get("stance_summary") or "")
        state.attention_target = str(payload.get("attention_target") or "")
        return state


@dataclass
class EventMemoryRecord:
    time: str
    summary: str
    importance: float
    topic: str = ""
    related_users: list[str] = field(default_factory=list)
    source_post_id: Optional[int] = None
    impact: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "summary": self.summary,
            "importance": self.importance,
            "topic": self.topic,
            "related_users": self.related_users,
            "source_post_id": self.source_post_id,
            "impact": self.impact,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EventMemoryRecord":
        source_post_id = payload.get("source_post_id")
        try:
            parsed_post_id = int(source_post_id) if source_post_id is not None else None
        except (TypeError, ValueError):
            parsed_post_id = None

        try:
            importance = float(payload.get("importance", 0.0))
        except (TypeError, ValueError):
            importance = 0.0

        return cls(
            time=str(payload.get("time") or ""),
            summary=str(payload.get("summary") or ""),
            importance=max(0.0, min(1.0, importance)),
            topic=str(payload.get("topic") or ""),
            related_users=[
                str(item).strip()
                for item in payload.get("related_users", [])
                if str(item).strip()
            ],
            source_post_id=parsed_post_id,
            impact=str(payload.get("impact") or ""),
        )


@dataclass
class EventMemory:
    records: list[EventMemoryRecord] = field(default_factory=list)
    max_size: int = 50

    def add(self, record: EventMemoryRecord) -> None:
        if not record.summary.strip():
            return

        duplicated = next(
            (
                item
                for item in self.records
                if item.summary == record.summary and item.time == record.time
            ),
            None,
        )
        if duplicated:
            duplicated.importance = max(duplicated.importance, record.importance)
            duplicated.impact = record.impact or duplicated.impact
            duplicated.topic = record.topic or duplicated.topic
            if record.related_users:
                duplicated.related_users = record.related_users
            if record.source_post_id is not None:
                duplicated.source_post_id = record.source_post_id
        else:
            self.records.append(record)

        self.records.sort(key=lambda item: item.importance, reverse=True)
        if len(self.records) > self.max_size:
            self.records = self.records[: self.max_size]


@dataclass
class AgentMemoryContext:
    static: StaticContext
    short_term: ShortTermMemory
    state: AgentState
    event_memory: EventMemory

    def to_dict(self) -> dict[str, Any]:
        return {
            "static": {
                "profile_text": self.static.profile_text,
                "global_event_text": self.static.global_event_text,
            },
            "short_term": {
                "max_rounds": self.short_term.max_rounds,
                "records": [record.to_dict() for record in self.short_term.records],
            },
            "state": self.state.to_dict(),
            "event_memory": {
                "max_size": self.event_memory.max_size,
                "records": [record.to_dict() for record in self.event_memory.records],
            },
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        static_context: StaticContext,
        short_term_max_rounds: int,
        event_max_size: int,
    ) -> "AgentMemoryContext":
        short_term_payload = payload.get("short_term", {})
        event_memory_payload = payload.get("event_memory", {})
        return cls(
            static=static_context,
            short_term=ShortTermMemory(
                max_rounds=short_term_max_rounds,
                records=[
                    ShortTermRecord.from_dict(item)
                    for item in short_term_payload.get("records", [])
                ],
            ),
            state=AgentState.from_dict(payload.get("state", {})),
            event_memory=EventMemory(
                max_size=event_max_size,
                records=[
                    EventMemoryRecord.from_dict(item)
                    for item in event_memory_payload.get("records", [])
                ],
            ),
        )


class StateUpdatePayload(BaseModel):
    mood: str = "平静"
    emotion_reason: str = ""
    intensity: float = 0.0
    current_goal: str = ""
    focus_topics: list[str] = Field(default_factory=list)
    stance_summary: str = ""
    attention_target: str = ""

    @field_validator("intensity")
    @classmethod
    def validate_intensity(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))


class EventMemoryPayload(BaseModel):
    should_store: bool = False
    summary: str = ""
    importance: float = 0.0
    topic: str = ""
    related_users: list[str] = Field(default_factory=list)
    source_post_id: Optional[int] = None
    impact: str = ""

    @field_validator("importance")
    @classmethod
    def validate_importance(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))


class MemoryUpdatePayload(BaseModel):
    state: StateUpdatePayload = Field(default_factory=StateUpdatePayload)
    event_memory: EventMemoryPayload = Field(default_factory=EventMemoryPayload)


class DecisionResultPayload(BaseModel):
    is_structured: bool = Field(default=True, exclude=True)
    final_output: str = ""
    state: StateUpdatePayload = Field(default_factory=StateUpdatePayload)
    event_memory: EventMemoryPayload = Field(default_factory=EventMemoryPayload)
