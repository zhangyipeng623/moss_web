from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from statistics import mean
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)
LlmCallable = Callable[[str, str], Awaitable[Dict[str, Any]]]


@dataclass(slots=True)
class UserPost:
    """用户帖子数据。"""

    content: str
    timestamp: int
    content_type: str = "Social Post"
    publish_time: str = ""
    like_count: int = 0
    comment_count: int = 0
    share_count: int = 0
    post_id: Optional[int] = None


@dataclass(slots=True)
class UserProfileSource:
    """生成画像所需的原始用户数据。"""

    user_name: str
    user_info: Dict[str, Any]
    posts: List[UserPost] = field(default_factory=list)


@dataclass(slots=True)
class UserStats:
    """用户统计画像。"""

    username: str = ""
    nickname: str = ""
    description: str = ""
    region: str = ""
    sex: str = ""
    post_count: int = 0
    collect_count: int = 0
    fans_count: int = 0
    follow_count: int = 0
    register_date: int = 0
    forward_count: int = 0
    quote_count: int = 0
    original_count: int = 0
    comment_count: int = 0
    active_hour_statistic: Dict[int, int] = field(default_factory=dict)
    top_active_hours: List[Dict[str, int]] = field(default_factory=list)
    home_page_url: str = ""
    follow_user: str = ""
    profile_img_url: str = ""
    trend: str = "W"
    post_influence: float = 0.0
    recent_post_count: int = 0
    account_influence: float = 0.0
    recent_7d_post_count: int = 0
    recent_30d_post_count: int = 0
    active_day_count: int = 0
    avg_post_length: float = 0.0
    avg_like_count: float = 0.0
    avg_comment_count: float = 0.0
    avg_share_count: float = 0.0
    avg_engagement: float = 0.0
    original_ratio: float = 0.0
    comment_ratio: float = 0.0
    forward_ratio: float = 0.0
    quote_ratio: float = 0.0
    hashtag_post_count: int = 0
    mention_post_count: int = 0
    url_post_count: int = 0
    latest_post_timestamp: int = 0
    earliest_post_timestamp: int = 0


@dataclass(slots=True)
class EvidencePost:
    """用于画像推理的证据帖子。"""

    evidence_id: str
    source_post_id: Optional[int]
    content_type: str
    content: str
    publish_time: str
    timestamp: int
    like_count: int
    comment_count: int
    share_count: int
    influence_score: float
    selection_reasons: List[str] = field(default_factory=list)

    def to_prompt_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source_post_id": self.source_post_id,
            "content_type": self.content_type,
            "publish_time": self.publish_time,
            "timestamp": self.timestamp,
            "like_count": self.like_count,
            "comment_count": self.comment_count,
            "share_count": self.share_count,
            "influence_score": round(self.influence_score, 4),
            "selection_reasons": self.selection_reasons,
            "content": self.content,
        }


@dataclass(slots=True)
class PortraitGenerationContext:
    """画像生成上下文。"""

    source: UserProfileSource
    stats: UserStats
    evidence_posts: List[EvidencePost]
    evidence_chunks: List[List[EvidencePost]]
    keyword_hints: List[str] = field(default_factory=list)
    observed_data_scope: List[str] = field(default_factory=list)


class TopicSignal(BaseModel):
    topic: str = ""
    confidence: float = 0.0
    evidence_ids: List[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    evidence_id: str = ""
    source_type: str = "post"
    source_ref: str = ""
    summary: str = ""


class ChunkObservation(BaseModel):
    chunk_id: int = 0
    topic_candidates: List[TopicSignal] = Field(default_factory=list)
    value_signals: List[str] = Field(default_factory=list)
    style_observations: List[str] = Field(default_factory=list)
    behavior_signals: List[str] = Field(default_factory=list)
    demographic_hints: List[str] = Field(default_factory=list)
    unknown_fields: List[str] = Field(default_factory=list)
    representative_evidence_ids: List[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    evidence_items: List[EvidenceItem] = Field(default_factory=list)
    topic_candidates: List[TopicSignal] = Field(default_factory=list)
    style_examples: List[str] = Field(default_factory=list)
    behavior_signals: List[str] = Field(default_factory=list)
    demographic_hints: List[str] = Field(default_factory=list)
    value_signals: List[str] = Field(default_factory=list)
    unknown_fields: List[str] = Field(default_factory=list)
    coverage_score: float = 0.0
    chunk_count: int = 0


class ExpressionStyle(BaseModel):
    tone: str = ""
    formality: str = ""
    verbosity: str = ""
    argument_style: str = ""


class ValueAnchor(BaseModel):
    topic: str = ""
    stance: str = ""
    confidence: float = 0.0
    evidence_ids: List[str] = Field(default_factory=list)


class SocialRole(BaseModel):
    role: str = ""
    description: str = ""
    confidence: float = 0.0
    evidence_ids: List[str] = Field(default_factory=list)


class Demographics(BaseModel):
    age_range: str = "unknown"
    gender: str = "unknown"
    job: str = "unknown"
    education: str = "unknown"
    income: str = "unknown"


class StableProfile(BaseModel):
    language: str = "zh-CN"
    long_term_interests: List[str] = Field(default_factory=list)
    content_topics: List[str] = Field(default_factory=list)
    expression_style: ExpressionStyle = Field(default_factory=ExpressionStyle)
    value_anchors: List[ValueAnchor] = Field(default_factory=list)
    social_role: SocialRole = Field(default_factory=SocialRole)
    demographics: Demographics = Field(default_factory=Demographics)
    profile_summary: str = ""
    uncertainties: List[str] = Field(default_factory=list)


class ActivityPattern(BaseModel):
    top_active_hours: List[int] = Field(default_factory=list)
    recent_activity_level: str = ""
    posting_frequency_level: str = ""


class ActionPreferences(BaseModel):
    post: float = 0.0
    comment: float = 0.0
    like: float = 0.0
    repost: float = 0.0
    quote: float = 0.0


class InteractionPreferences(BaseModel):
    prefers_hot_topics: bool = False
    prefers_followed_authors: bool = False
    prefers_argumentative_threads: bool = False


class ContentPreferences(BaseModel):
    preferred_length: str = ""
    preferred_content_type: List[str] = Field(default_factory=list)
    emotion_intensity: str = ""
    stance_explicitness: str = ""


class TriggerRule(BaseModel):
    condition: str = ""
    likely_actions: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    evidence_ids: List[str] = Field(default_factory=list)


class BehaviorProfile(BaseModel):
    activity_pattern: ActivityPattern = Field(default_factory=ActivityPattern)
    action_preferences: ActionPreferences = Field(default_factory=ActionPreferences)
    interaction_preferences: InteractionPreferences = Field(default_factory=InteractionPreferences)
    content_preferences: ContentPreferences = Field(default_factory=ContentPreferences)
    trigger_rules: List[TriggerRule] = Field(default_factory=list)
    profile_summary: str = ""
    uncertainties: List[str] = Field(default_factory=list)


class AgentProfile(BaseModel):
    identity_summary: str = ""
    interest_summary: str = ""
    value_summary: str = ""
    style_summary: str = ""
    behavior_summary: str = ""
    interaction_summary: str = ""
    speaking_rules: List[str] = Field(default_factory=list)
    action_rules: List[str] = Field(default_factory=list)
    avoidance_rules: List[str] = Field(default_factory=list)
    initial_focus_topics: List[str] = Field(default_factory=list)
    current_goal_hint: str = ""


class SimulationInit(BaseModel):
    mood: str = ""
    emotion_reason: str = ""
    intensity: float = 0.0
    current_goal: str = ""
    focus_topics: List[str] = Field(default_factory=list)
    stance_summary: str = ""
    attention_target: str = ""


class SimulationUserProfile(BaseModel):
    version: str = "v3-single-persona"
    user_id: str = ""
    generated_at: str = ""
    stats: Dict[str, Any] = Field(default_factory=dict)
    stable_profile: StableProfile = Field(default_factory=StableProfile)
    behavior_profile: BehaviorProfile = Field(default_factory=BehaviorProfile)
    agent_profile: AgentProfile = Field(default_factory=AgentProfile)
    simulation_init: SimulationInit = Field(default_factory=SimulationInit)
    evidence_meta: EvidencePack = Field(default_factory=EvidencePack)
    generation_meta: Dict[str, Any] = Field(default_factory=dict)
    identity_summary: str = ""
    interest_summary: str = ""
    value_summary: str = ""
    style_summary: str = ""
    behavior_summary: str = ""
    interaction_summary: str = ""
    speaking_rules_text: str = ""
    action_rules_text: str = ""
    avoidance_rules_text: str = ""


class PortraitGenerationError(RuntimeError):
    """画像生成失败。"""

    def __init__(self, stage_name: str, message: str):
        super().__init__(f"{stage_name}: {message}")
        self.stage_name = stage_name
        self.message = message


class UserPortraitGenerator:
    """面向社交媒体模拟的单主人格用户画像生成器。"""

    def __init__(
        self,
        post_influence_weights: Optional[Dict[str, float]] = None,
        user_influence_weights: Optional[Dict[str, float]] = None,
        max_posts_per_chunk: int = 20,
        max_chunks: int = 6,
        max_recent_posts: int = 80,
        max_high_impact_posts: int = 30,
        max_style_posts: int = 20,
        reference_timestamp: Optional[int] = None,
    ):
        self.post_influence_weights = {
            "post_like": 1.0,
            "post_comment": 2.0,
            "post_share": 3.0,
            "time_decay": 1 / (30 * 24 * 60 * 60),
            **(post_influence_weights or {}),
        }
        self.user_influence_weights = {
            "user_fans": 0.5,
            "user_post_influence": 0.3,
            "user_activity": 0.2,
            **(user_influence_weights or {}),
        }
        self.max_posts_per_chunk = max(5, max_posts_per_chunk)
        self.max_chunks = max(1, max_chunks)
        self.max_recent_posts = max(10, max_recent_posts)
        self.max_high_impact_posts = max(5, max_high_impact_posts)
        self.max_style_posts = max(5, max_style_posts)
        self.reference_timestamp = int(reference_timestamp) if reference_timestamp else int(time.time())

    def analyze_stats(self, source: UserProfileSource) -> UserStats:
        """基于用户信息和帖子生成统计画像。"""
        user_info = source.user_info
        stats = UserStats(
            username=str(user_info.get("username") or source.user_name),
            nickname=str(user_info.get("nickname") or ""),
            description=self._clean_text(user_info.get("description", "")),
            region=self._clean_text(user_info.get("region", "")),
            sex=self._clean_text(user_info.get("sex", "")),
            collect_count=self._safe_int(user_info.get("collect_count", 0)),
            fans_count=self._safe_int(user_info.get("fans_count", 0)),
            follow_count=self._safe_int(user_info.get("follow_count", 0)),
            follow_user=self._clean_text(user_info.get("follow_user", "")),
            home_page_url=str(user_info.get("home_page_url") or ""),
            profile_img_url=str(user_info.get("profile_img_url") or ""),
        )

        register_ts = self._safe_int(user_info.get("register_timestamp", 0))
        if register_ts > 0:
            dt = datetime.fromtimestamp(register_ts, tz=timezone.utc)
            stats.register_date = int(f"{dt.year}{dt.month:02d}")

        stats.post_count = len(source.posts)
        if not source.posts:
            return stats

        now_ts = self.reference_timestamp
        last_7days_ts = now_ts - 7 * 24 * 60 * 60
        last_30days_ts = now_ts - 30 * 24 * 60 * 60
        last_100days_ts = now_ts - 100 * 24 * 60 * 60

        scored_posts = sorted(
            ((post, self._compute_post_influence(post)) for post in source.posts),
            key=lambda item: item[1],
            reverse=True,
        )
        interaction_max = max(1, int(len(scored_posts) * 0.3)) if scored_posts else 0
        interaction_total = 0.0
        interaction_count = 0
        post_lengths: List[int] = []
        like_values: List[int] = []
        comment_values: List[int] = []
        share_values: List[int] = []
        active_days = set()

        timestamps = [post.timestamp for post in source.posts if post.timestamp > 0]
        if timestamps:
            stats.latest_post_timestamp = max(timestamps)
            stats.earliest_post_timestamp = min(timestamps)

        for index, (post, post_influence) in enumerate(scored_posts):
            if index < interaction_max:
                interaction_total += post_influence
                interaction_count += 1

            if post.timestamp > last_100days_ts:
                stats.recent_post_count += 1
            if post.timestamp > last_30days_ts:
                stats.recent_30d_post_count += 1
            if post.timestamp > last_7days_ts:
                stats.recent_7d_post_count += 1

            normalized_type = self._normalize_content_type(post.content_type)
            if normalized_type in {"comment", "reply"}:
                stats.comment_count += 1
            elif normalized_type == "repost":
                stats.forward_count += 1
            elif normalized_type == "quote":
                stats.quote_count += 1
            else:
                stats.original_count += 1

            if post.timestamp > 0:
                dt = datetime.fromtimestamp(post.timestamp, tz=timezone.utc)
                stats.active_hour_statistic[dt.hour] = (
                    stats.active_hour_statistic.get(dt.hour, 0) + 1
                )
                active_days.add(dt.date().isoformat())

            text = self._clean_text(post.content)
            post_lengths.append(len(text))
            like_values.append(self._safe_int(post.like_count))
            comment_values.append(self._safe_int(post.comment_count))
            share_values.append(self._safe_int(post.share_count))

            if self._HASHTAG_PATTERN.search(text):
                stats.hashtag_post_count += 1
            if self._MENTION_PATTERN.search(text):
                stats.mention_post_count += 1
            if self._URL_PATTERN.search(text):
                stats.url_post_count += 1

        if interaction_count > 0:
            stats.post_influence = round(interaction_total / interaction_count, 2)
        stats.account_influence = round(self._compute_user_influence(stats), 4)
        stats.active_day_count = len(active_days)
        stats.avg_post_length = round(mean(post_lengths), 2) if post_lengths else 0.0
        stats.avg_like_count = round(mean(like_values), 2) if like_values else 0.0
        stats.avg_comment_count = round(mean(comment_values), 2) if comment_values else 0.0
        stats.avg_share_count = round(mean(share_values), 2) if share_values else 0.0
        stats.avg_engagement = round(
            mean(
                like + comment + share
                for like, comment, share in zip(
                    like_values,
                    comment_values,
                    share_values,
                )
            ),
            2,
        )

        total_posts = max(stats.post_count, 1)
        stats.original_ratio = round(stats.original_count / total_posts, 4)
        stats.comment_ratio = round(stats.comment_count / total_posts, 4)
        stats.forward_ratio = round(stats.forward_count / total_posts, 4)
        stats.quote_ratio = round(stats.quote_count / total_posts, 4)

        sorted_hours = sorted(
            stats.active_hour_statistic.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        stats.top_active_hours = [
            {"hour": hour, "post_count": count}
            for hour, count in sorted_hours[:3]
        ]
        return stats

    def prepare_generation_context(
        self,
        source: UserProfileSource,
    ) -> PortraitGenerationContext:
        """准备多阶段画像生成所需的上下文。"""
        stats = self.analyze_stats(source)
        evidence_posts = self._select_evidence_posts(source)
        evidence_chunks = self._chunk_evidence_posts(evidence_posts)
        keyword_hints = self._extract_keyword_hints(evidence_posts)
        observed_data_scope = [
            "user_info",
            "authored_posts",
            "post_level_engagement_stats",
        ]
        return PortraitGenerationContext(
            source=source,
            stats=stats,
            evidence_posts=evidence_posts,
            evidence_chunks=evidence_chunks,
            keyword_hints=keyword_hints,
            observed_data_scope=observed_data_scope,
        )

    async def generate_evidence_pack(
        self,
        context: PortraitGenerationContext,
        llm_callable: LlmCallable,
    ) -> EvidencePack:
        """阶段 A：分块抽取证据包。"""
        chunk_observations: List[ChunkObservation] = []
        for chunk_index, chunk_posts in enumerate(context.evidence_chunks, start=1):
            raw_output = await llm_callable(
                self.build_chunk_system_prompt(),
                self.build_chunk_user_prompt(context, chunk_index, chunk_posts),
            )
            observation = self._validate_stage_output(
                raw_output,
                ChunkObservation,
                stage_name=f"chunk_observation_{chunk_index}",
            )
            if observation.chunk_id <= 0:
                observation.chunk_id = chunk_index
            chunk_observations.append(observation)

        topic_map: Dict[str, TopicSignal] = {}
        style_examples: List[str] = []
        behavior_signals: List[str] = []
        demographic_hints: List[str] = []
        value_signals: List[str] = []
        unknown_fields: List[str] = []

        for observation in chunk_observations:
            for signal in observation.topic_candidates:
                topic = signal.topic.strip()
                if not topic:
                    continue
                merged = topic_map.get(topic)
                if merged is None:
                    topic_map[topic] = TopicSignal(
                        topic=topic,
                        confidence=self._clamp_score(signal.confidence),
                        evidence_ids=self._dedupe_strings(signal.evidence_ids),
                    )
                    continue
                merged.confidence = max(merged.confidence, self._clamp_score(signal.confidence))
                merged.evidence_ids = self._dedupe_strings(
                    [*merged.evidence_ids, *signal.evidence_ids]
                )
            style_examples.extend(observation.style_observations)
            behavior_signals.extend(observation.behavior_signals)
            demographic_hints.extend(observation.demographic_hints)
            value_signals.extend(observation.value_signals)
            unknown_fields.extend(observation.unknown_fields)

        for keyword in context.keyword_hints[:6]:
            if keyword not in topic_map:
                topic_map[keyword] = TopicSignal(
                    topic=keyword,
                    confidence=0.35,
                    evidence_ids=[],
                )

        evidence_items = [
            EvidenceItem(
                evidence_id=item.evidence_id,
                source_type="post",
                source_ref=str(item.source_post_id or item.evidence_id),
                summary=self._truncate_text(item.content, 60),
            )
            for item in context.evidence_posts
        ]
        evidence_items.extend(self._build_profile_evidence_items(context))
        evidence_items.extend(self._build_stats_evidence_items(context))

        observed_post_ratio = (
            len(context.evidence_posts) / max(len(context.source.posts), 1)
            if context.source.posts
            else 0.0
        )
        coverage_score = min(1.0, 0.55 + observed_post_ratio * 0.45)
        return EvidencePack(
            evidence_items=evidence_items,
            topic_candidates=sorted(
                topic_map.values(),
                key=lambda item: item.confidence,
                reverse=True,
            )[:12],
            style_examples=self._dedupe_strings(style_examples)[:10],
            behavior_signals=self._dedupe_strings(behavior_signals)[:12],
            demographic_hints=self._dedupe_strings(demographic_hints)[:8],
            value_signals=self._dedupe_strings(value_signals)[:10],
            unknown_fields=self._dedupe_strings(unknown_fields)[:12],
            coverage_score=round(self._clamp_score(coverage_score), 4),
            chunk_count=len(chunk_observations),
        )

    async def generate_stable_profile(
        self,
        context: PortraitGenerationContext,
        evidence_pack: EvidencePack,
        llm_callable: LlmCallable,
    ) -> StableProfile:
        """阶段 B：生成稳定画像。"""
        raw_output = await llm_callable(
            self.build_stable_profile_system_prompt(),
            self.build_stable_profile_user_prompt(context, evidence_pack),
        )
        stable_profile = self._validate_stage_output(
            raw_output,
            StableProfile,
            stage_name="stable_profile",
        )
        stable_profile.long_term_interests = self._dedupe_strings(
            stable_profile.long_term_interests
        )[:8]
        stable_profile.content_topics = self._dedupe_strings(
            stable_profile.content_topics
        )[:10]
        stable_profile.uncertainties = self._dedupe_strings(
            [*stable_profile.uncertainties, *evidence_pack.unknown_fields]
        )[:12]
        stable_profile.value_anchors = [
            ValueAnchor(
                topic=item.topic.strip(),
                stance=item.stance.strip(),
                confidence=self._clamp_score(item.confidence),
                evidence_ids=self._dedupe_strings(item.evidence_ids),
            )
            for item in stable_profile.value_anchors
            if item.topic.strip() and item.stance.strip()
        ][:8]
        stable_profile.social_role.confidence = self._clamp_score(
            stable_profile.social_role.confidence
        )
        stable_profile.social_role.evidence_ids = self._dedupe_strings(
            stable_profile.social_role.evidence_ids
        )
        return stable_profile

    async def generate_behavior_profile(
        self,
        context: PortraitGenerationContext,
        evidence_pack: EvidencePack,
        stable_profile: StableProfile,
        llm_callable: LlmCallable,
    ) -> BehaviorProfile:
        """阶段 C：生成行为模式画像。"""
        raw_output = await llm_callable(
            self.build_behavior_profile_system_prompt(),
            self.build_behavior_profile_user_prompt(
                context,
                evidence_pack,
                stable_profile,
            ),
        )
        behavior_profile = self._validate_stage_output(
            raw_output,
            BehaviorProfile,
            stage_name="behavior_profile",
        )
        behavior_profile.activity_pattern.top_active_hours = [
            self._safe_int(item)
            for item in behavior_profile.activity_pattern.top_active_hours
            if 0 <= self._safe_int(item) <= 23
        ][:6]
        behavior_profile.content_preferences.preferred_content_type = self._dedupe_strings(
            behavior_profile.content_preferences.preferred_content_type
        )[:6]
        behavior_profile.trigger_rules = [
            TriggerRule(
                condition=item.condition.strip(),
                likely_actions=self._dedupe_strings(item.likely_actions),
                confidence=self._clamp_score(item.confidence),
                evidence_ids=self._dedupe_strings(item.evidence_ids),
            )
            for item in behavior_profile.trigger_rules
            if item.condition.strip()
        ][:8]
        behavior_profile.uncertainties = self._dedupe_strings(
            behavior_profile.uncertainties
        )[:12]
        behavior_profile.action_preferences = ActionPreferences.model_validate(
            self._normalize_probability_map(
                behavior_profile.action_preferences.model_dump(),
                expected_keys=("post", "comment", "like", "repost", "quote"),
            )
        )
        return behavior_profile

    async def generate_agent_profile(
        self,
        context: PortraitGenerationContext,
        evidence_pack: EvidencePack,
        stable_profile: StableProfile,
        behavior_profile: BehaviorProfile,
        llm_callable: LlmCallable,
    ) -> AgentProfile:
        """阶段 D：生成可直接驱动 Agent 的单主人格摘要。"""
        raw_output = await llm_callable(
            self.build_agent_profile_system_prompt(),
            self.build_agent_profile_user_prompt(
                context,
                evidence_pack,
                stable_profile,
                behavior_profile,
            ),
        )
        agent_profile = self._validate_stage_output(
            raw_output,
            AgentProfile,
            stage_name="agent_profile",
        )
        agent_profile.speaking_rules = self._dedupe_strings(agent_profile.speaking_rules)[:8]
        agent_profile.action_rules = self._dedupe_strings(agent_profile.action_rules)[:8]
        agent_profile.avoidance_rules = self._dedupe_strings(agent_profile.avoidance_rules)[:8]
        agent_profile.initial_focus_topics = self._dedupe_strings(
            agent_profile.initial_focus_topics
        )[:6]
        return self._fill_agent_profile_defaults(
            context=context,
            stable_profile=stable_profile,
            behavior_profile=behavior_profile,
            agent_profile=agent_profile,
        )

    async def generate_simulation_init(
        self,
        context: PortraitGenerationContext,
        stable_profile: StableProfile,
        behavior_profile: BehaviorProfile,
        agent_profile: AgentProfile,
        llm_callable: LlmCallable,
        global_event: str = "",
    ) -> SimulationInit:
        """阶段 E：生成模拟初始状态。"""
        raw_output = await llm_callable(
            self.build_simulation_init_system_prompt(),
            self.build_simulation_init_user_prompt(
                context,
                stable_profile,
                behavior_profile,
                agent_profile,
                global_event=global_event,
            ),
        )
        simulation_init = self._validate_stage_output(
            raw_output,
            SimulationInit,
            stage_name="simulation_init",
        )
        simulation_init.intensity = self._clamp_score(simulation_init.intensity)
        simulation_init.focus_topics = self._dedupe_strings(simulation_init.focus_topics)[:6]
        return simulation_init

    async def run_pipeline(
        self,
        source: UserProfileSource,
        llm_callable: LlmCallable,
        global_event: str = "",
    ) -> SimulationUserProfile:
        """执行完整画像流水线。"""
        context = self.prepare_generation_context(source)
        evidence_pack = await self.generate_evidence_pack(context, llm_callable)
        stable_profile = await self.generate_stable_profile(
            context,
            evidence_pack,
            llm_callable,
        )
        behavior_profile = await self.generate_behavior_profile(
            context,
            evidence_pack,
            stable_profile,
            llm_callable,
        )
        agent_profile = await self.generate_agent_profile(
            context,
            evidence_pack,
            stable_profile,
            behavior_profile,
            llm_callable,
        )
        simulation_init = await self.generate_simulation_init(
            context,
            stable_profile,
            behavior_profile,
            agent_profile,
            llm_callable,
            global_event=global_event,
        )
        flattened = self.build_template_fields(agent_profile)
        return SimulationUserProfile(
            version="v3-single-persona",
            user_id=context.stats.username,
            generated_at=datetime.now(timezone.utc).isoformat(),
            stats=asdict(context.stats),
            stable_profile=stable_profile,
            behavior_profile=behavior_profile,
            agent_profile=agent_profile,
            simulation_init=simulation_init,
            evidence_meta=evidence_pack,
            generation_meta={
                "mode": "single_persona_multi_stage",
                "chunk_count": len(context.evidence_chunks),
                "selected_post_count": len(context.evidence_posts),
                "total_post_count": len(context.source.posts),
                "observed_data_scope": context.observed_data_scope,
                "keyword_hints": context.keyword_hints,
                "reference_timestamp": self.reference_timestamp,
                "reference_time_utc": datetime.fromtimestamp(
                    self.reference_timestamp,
                    tz=timezone.utc,
                ).isoformat(),
            },
            **flattened,
        )

    async def generate_portrait(
        self,
        source: UserProfileSource,
        llm_callable: LlmCallable,
        global_event: str = "",
    ) -> Dict[str, Any]:
        """对外暴露的一键画像生成入口。"""
        profile = await self.run_pipeline(
            source=source,
            llm_callable=llm_callable,
            global_event=global_event,
        )
        return profile.model_dump()

    def build_template_fields(self, agent_profile: AgentProfile) -> Dict[str, Any]:
        """构建默认模板所需的扁平字段。"""
        return {
            "identity_summary": agent_profile.identity_summary,
            "interest_summary": agent_profile.interest_summary,
            "value_summary": agent_profile.value_summary,
            "style_summary": agent_profile.style_summary,
            "behavior_summary": agent_profile.behavior_summary,
            "interaction_summary": agent_profile.interaction_summary,
            "speaking_rules_text": self._join_rule_lines(agent_profile.speaking_rules),
            "action_rules_text": self._join_rule_lines(agent_profile.action_rules),
            "avoidance_rules_text": self._join_rule_lines(agent_profile.avoidance_rules),
        }

    def build_chunk_system_prompt(self) -> str:
        """构建阶段 A 的系统提示词。"""
        return "\n".join(
            [
                "你是一位社交媒体用户画像研究员，当前任务仅做证据抽取，不做最终画像。",
                "请基于输入的帖子证据和基础资料提取主题、风格、行为信号与不确定项。",
                "每条判断都必须保守，不能凭空补全人格。",
                "若无证据，请输出 unknown 或空列表。",
                "输出必须是合法 JSON，所有说明使用中文简体。",
                "用户关注话题需要根据历史发帖内容自行归纳，不要从预设主题列表中挑选。",
            ]
        )

    def build_chunk_user_prompt(
        self,
        context: PortraitGenerationContext,
        chunk_index: int,
        chunk_posts: Sequence[EvidencePost],
    ) -> str:
        """构建阶段 A 的用户提示词。"""
        payload = {
            "chunk_id": chunk_index,
            "user_info": self._build_user_info_summary(context.source.user_info),
            "stats_summary": self._build_stats_summary(context.stats),
            "observed_data_scope": context.observed_data_scope,
            "keyword_hints": context.keyword_hints,
            "posts": [item.to_prompt_dict() for item in chunk_posts],
            "output_schema": {
                "chunk_id": chunk_index,
                "topic_candidates": [
                    {"topic": "", "confidence": 0.0, "evidence_ids": []}
                ],
                "value_signals": [],
                "style_observations": [],
                "behavior_signals": [],
                "demographic_hints": [],
                "unknown_fields": [],
                "representative_evidence_ids": [],
            },
        }
        return "\n".join(
            [
                "请只对这一批证据做局部观察，不要提前写总画像。",
                "注意：输入只覆盖用户资料、用户发帖文本和帖子层级互动统计，不包含浏览、点击、停留时长等隐式行为日志。",
                "keyword_hints 是根据历史帖子正文自动抽取的潜在线索，只能作为辅助参考，最终话题判断仍需以帖子证据为准。",
                json.dumps(payload, ensure_ascii=False, indent=2),
            ]
        )

    def build_stable_profile_system_prompt(self) -> str:
        """构建阶段 B 的系统提示词。"""
        return "\n".join(
            [
                "你是一位社交媒体用户画像专家，需要根据证据包生成稳定画像。",
                "这里只能输出跨时间较稳定的特征，如长期兴趣、价值立场、表达风格和社交角色。",
                "不能把某条帖子的短期情绪当成长期人格。",
                "如果某结论缺少充分证据，请写 unknown 或放入 uncertainties。",
                "输出必须是合法 JSON，键名使用英文，值说明使用中文简体。",
            ]
        )

    def build_stable_profile_user_prompt(
        self,
        context: PortraitGenerationContext,
        evidence_pack: EvidencePack,
    ) -> str:
        """构建阶段 B 的用户提示词。"""
        payload = {
            "user_info": self._build_user_info_summary(context.source.user_info),
            "stats_summary": self._build_stats_summary(context.stats),
            "evidence_pack": evidence_pack.model_dump(),
            "output_schema": {
                "language": "zh-CN",
                "long_term_interests": [],
                "content_topics": [],
                "expression_style": {
                    "tone": "",
                    "formality": "",
                    "verbosity": "",
                    "argument_style": "",
                },
                "value_anchors": [
                    {"topic": "", "stance": "", "confidence": 0.0, "evidence_ids": []}
                ],
                "social_role": {
                    "role": "",
                    "description": "",
                    "confidence": 0.0,
                    "evidence_ids": [],
                },
                "demographics": {
                    "age_range": "unknown",
                    "gender": "unknown",
                    "job": "unknown",
                    "education": "unknown",
                    "income": "unknown",
                },
                "profile_summary": "",
                "uncertainties": [],
            },
        }
        return "\n".join(
            [
                "请只输出稳定画像，不要输出当前事件下的即时反应。",
                json.dumps(payload, ensure_ascii=False, indent=2),
            ]
        )

    def build_behavior_profile_system_prompt(self) -> str:
        """构建阶段 C 的系统提示词。"""
        return "\n".join(
            [
                "你是一位社交媒体行为建模专家，需要生成行为模式画像。",
                "重点说明该用户在平台上更可能如何发帖、评论、点赞、转发，以及什么情况下更活跃。",
                "动作概率只能表示相对倾向，不能伪装成精确统计真值。",
                "如果输入没有显式日志支持，请在 uncertainties 中标明限制。",
                "输出必须是合法 JSON，键名使用英文，值说明使用中文简体。",
            ]
        )

    def build_behavior_profile_user_prompt(
        self,
        context: PortraitGenerationContext,
        evidence_pack: EvidencePack,
        stable_profile: StableProfile,
    ) -> str:
        """构建阶段 C 的用户提示词。"""
        payload = {
            "stats_summary": self._build_stats_summary(context.stats),
            "observed_data_scope": context.observed_data_scope,
            "stable_profile": stable_profile.model_dump(),
            "evidence_pack": evidence_pack.model_dump(),
            "output_schema": {
                "activity_pattern": {
                    "top_active_hours": [],
                    "recent_activity_level": "",
                    "posting_frequency_level": "",
                },
                "action_preferences": {
                    "post": 0.0,
                    "comment": 0.0,
                    "like": 0.0,
                    "repost": 0.0,
                    "quote": 0.0,
                },
                "interaction_preferences": {
                    "prefers_hot_topics": False,
                    "prefers_followed_authors": False,
                    "prefers_argumentative_threads": False,
                },
                "content_preferences": {
                    "preferred_length": "",
                    "preferred_content_type": [],
                    "emotion_intensity": "",
                    "stance_explicitness": "",
                },
                "trigger_rules": [
                    {"condition": "", "likely_actions": [], "confidence": 0.0, "evidence_ids": []}
                ],
                "profile_summary": "",
                "uncertainties": [],
            },
        }
        return "\n".join(
            [
                "请输出可驱动单一主人格模拟的行为模式。",
                json.dumps(payload, ensure_ascii=False, indent=2),
            ]
        )

    def build_agent_profile_system_prompt(self) -> str:
        """构建阶段 D 的系统提示词。"""
        return "\n".join(
            [
                "你是一位社交媒体模拟设计专家，需要把稳定画像和行为画像压缩为单一、可直接驱动 Agent 的主人格摘要。",
                "不能输出多个子人格，也不能设计切换机制。",
                "请给出高信号的身份概述、风格摘要、行为摘要，以及明确的表达规则和行动规则。",
                "输出必须是合法 JSON，键名使用英文，值说明使用中文简体。",
            ]
        )

    def build_agent_profile_user_prompt(
        self,
        context: PortraitGenerationContext,
        evidence_pack: EvidencePack,
        stable_profile: StableProfile,
        behavior_profile: BehaviorProfile,
    ) -> str:
        """构建阶段 D 的用户提示词。"""
        payload = {
            "user_name": context.stats.username,
            "stable_profile": stable_profile.model_dump(),
            "behavior_profile": behavior_profile.model_dump(),
            "evidence_pack": evidence_pack.model_dump(),
            "output_schema": {
                "identity_summary": "",
                "interest_summary": "",
                "value_summary": "",
                "style_summary": "",
                "behavior_summary": "",
                "interaction_summary": "",
                "speaking_rules": [],
                "action_rules": [],
                "avoidance_rules": [],
                "initial_focus_topics": [],
                "current_goal_hint": "",
            },
        }
        return "\n".join(
            [
                "请将该用户收敛为一套稳定的主人格画像，用于后续模板注入。",
                json.dumps(payload, ensure_ascii=False, indent=2),
            ]
        )

    def build_simulation_init_system_prompt(self) -> str:
        """构建阶段 E 的系统提示词。"""
        return "\n".join(
            [
                "你是一位社交媒体模拟初始化专家，需要为单一主人格生成初始状态。",
                "全局事件只影响当前状态，不允许改写长期画像。",
                "输出必须是合法 JSON，键名使用英文，值说明使用中文简体。",
            ]
        )

    def build_simulation_init_user_prompt(
        self,
        context: PortraitGenerationContext,
        stable_profile: StableProfile,
        behavior_profile: BehaviorProfile,
        agent_profile: AgentProfile,
        global_event: str = "",
    ) -> str:
        """构建阶段 E 的用户提示词。"""
        payload = {
            "user_name": context.stats.username,
            "stable_profile": stable_profile.model_dump(),
            "behavior_profile": behavior_profile.model_dump(),
            "agent_profile": agent_profile.model_dump(),
            "global_event": global_event,
            "output_schema": {
                "mood": "",
                "emotion_reason": "",
                "intensity": 0.0,
                "current_goal": "",
                "focus_topics": [],
                "stance_summary": "",
                "attention_target": "",
            },
        }
        return "\n".join(
            [
                "请输出当前模拟启动时的初始状态。",
                json.dumps(payload, ensure_ascii=False, indent=2),
            ]
        )

    def _select_evidence_posts(self, source: UserProfileSource) -> List[EvidencePost]:
        """挑选用于多阶段推理的证据帖子。"""
        if not source.posts:
            return []

        indexed_posts = list(enumerate(source.posts))
        recent_rank = sorted(indexed_posts, key=lambda item: item[1].timestamp, reverse=True)
        impact_rank = sorted(
            indexed_posts,
            key=lambda item: self._compute_post_influence(item[1]),
            reverse=True,
        )
        style_rank = sorted(
            indexed_posts,
            key=lambda item: len(self._clean_text(item[1].content)),
            reverse=True,
        )

        selected_reasons: Dict[int, set[str]] = {}

        def add_ranked_posts(posts: Sequence[tuple[int, UserPost]], limit: int, reason: str) -> None:
            for index, _ in posts[:limit]:
                selected_reasons.setdefault(index, set()).add(reason)

        add_ranked_posts(recent_rank, self.max_recent_posts, "recent")
        add_ranked_posts(impact_rank, self.max_high_impact_posts, "high_impact")
        add_ranked_posts(style_rank, self.max_style_posts, "style_representative")

        type_buckets: Dict[str, List[tuple[int, UserPost]]] = {}
        for index, post in indexed_posts:
            normalized_type = self._normalize_content_type(post.content_type)
            type_buckets.setdefault(normalized_type, []).append((index, post))
        for bucket_posts in type_buckets.values():
            ranked_bucket = sorted(
                bucket_posts,
                key=lambda item: (
                    self._compute_post_influence(item[1]),
                    item[1].timestamp,
                ),
                reverse=True,
            )
            add_ranked_posts(ranked_bucket, 4, "type_representative")

        max_total_posts = self.max_posts_per_chunk * self.max_chunks
        selected_indices = sorted(
            selected_reasons.keys(),
            key=lambda index: (
                source.posts[index].timestamp,
                self._compute_post_influence(source.posts[index]),
            ),
            reverse=True,
        )[:max_total_posts]

        evidence_posts: List[EvidencePost] = []
        for index in selected_indices:
            post = source.posts[index]
            evidence_posts.append(
                EvidencePost(
                    evidence_id=self._build_evidence_id(post, index),
                    source_post_id=post.post_id,
                    content_type=post.content_type,
                    content=self._clean_text(post.content),
                    publish_time=post.publish_time,
                    timestamp=self._safe_int(post.timestamp),
                    like_count=self._safe_int(post.like_count),
                    comment_count=self._safe_int(post.comment_count),
                    share_count=self._safe_int(post.share_count),
                    influence_score=self._compute_post_influence(post),
                    selection_reasons=sorted(selected_reasons.get(index, {"recent"})),
                )
            )
        return evidence_posts

    def _chunk_evidence_posts(
        self,
        evidence_posts: Sequence[EvidencePost],
    ) -> List[List[EvidencePost]]:
        """对证据帖子进行分块。"""
        if not evidence_posts:
            return [[]]
        chunks: List[List[EvidencePost]] = []
        for start in range(0, len(evidence_posts), self.max_posts_per_chunk):
            chunks.append(list(evidence_posts[start : start + self.max_posts_per_chunk]))
        return chunks[: self.max_chunks]

    def _extract_keyword_hints(
        self,
        evidence_posts: Sequence[EvidencePost],
    ) -> List[str]:
        """根据历史帖子正文自动抽取潜在话题线索。"""
        phrase_scores: Dict[str, float] = {}
        phrase_post_coverage: Dict[str, set[str]] = {}
        phrase_peak_weight: Dict[str, float] = {}

        now_ts = self.reference_timestamp
        for post in evidence_posts:
            text = self._clean_text(post.content)
            if not text:
                continue
            age_days = (
                max(0.0, now_ts - post.timestamp) / (24 * 60 * 60)
                if post.timestamp > 0
                else 999.0
            )
            post_weight = 1.0 + min(1.5, math.log1p(max(0.0, post.influence_score)) / 4)
            post_weight *= 1.0 + 0.35 * math.exp(-age_days / 45)

            seen_in_post = set()
            for phrase, phrase_weight in self._extract_topic_phrases_from_text(text):
                normalized_phrase = self._normalize_topic_phrase(phrase)
                if not self._is_valid_topic_phrase(normalized_phrase):
                    continue
                if normalized_phrase in seen_in_post:
                    continue
                seen_in_post.add(normalized_phrase)
                phrase_scores[normalized_phrase] = (
                    phrase_scores.get(normalized_phrase, 0.0)
                    + post_weight * phrase_weight
                )
                phrase_post_coverage.setdefault(normalized_phrase, set()).add(post.evidence_id)
                phrase_peak_weight[normalized_phrase] = max(
                    phrase_peak_weight.get(normalized_phrase, 0.0),
                    phrase_weight,
                )

        ranked_phrases: List[tuple[str, float, int]] = []
        for phrase, score in phrase_scores.items():
            coverage = len(phrase_post_coverage.get(phrase, set()))
            peak_weight = phrase_peak_weight.get(phrase, 0.0)
            if len(phrase) <= 2 and coverage < 2 and peak_weight < 2.5:
                continue
            if coverage == 1 and score < 2.8 and peak_weight < 2.5:
                continue
            adjusted_score = score + coverage * 0.9 + min(len(phrase), 6) * 0.22
            ranked_phrases.append((phrase, adjusted_score, coverage))

        ranked_phrases.sort(
            key=lambda item: (item[1], item[2], len(item[0])),
            reverse=True,
        )

        selected: List[str] = []
        selected_meta: Dict[str, tuple[float, int]] = {}
        for phrase, score, coverage in ranked_phrases:
            should_skip = False
            for kept in selected:
                kept_score, kept_coverage = selected_meta[kept]
                if (
                    phrase in kept
                    and len(kept) >= len(phrase)
                    and kept_coverage >= coverage
                    and kept_score >= score * 0.95
                ):
                    should_skip = True
                    break
            if should_skip:
                continue

            removable = [
                kept
                for kept in selected
                if (
                    kept in phrase
                    and len(phrase) > len(kept)
                    and coverage >= selected_meta[kept][1]
                    and score >= selected_meta[kept][0] * 0.95
                )
            ]
            for kept in removable:
                selected.remove(kept)
                selected_meta.pop(kept, None)

            selected.append(phrase)
            selected_meta[phrase] = (score, coverage)
            if len(selected) >= 12:
                break
        return selected

    def _extract_topic_phrases_from_text(self, text: str) -> List[tuple[str, float]]:
        """从单条帖子正文中抽取候选议题短语。"""
        cleaned = self._URL_PATTERN.sub(" ", self._clean_text(text))
        candidates: List[tuple[str, float]] = []

        for match in self._HASHTAG_PATTERN.findall(cleaned):
            phrase = self._normalize_topic_phrase(match)
            if phrase:
                candidates.append((phrase, 2.8))

        cleaned = self._HASHTAG_PATTERN.sub(" ", cleaned)
        cleaned = self._MENTION_PATTERN.sub(" ", cleaned)

        for token in re.findall(r"[A-Za-z][A-Za-z0-9_\-/+]{1,24}", cleaned):
            normalized = token.upper() if token.isupper() else token.lower()
            if normalized.lower() in self._ENGLISH_TOPIC_STOPWORDS:
                continue
            candidates.append((normalized, 1.2 if normalized.islower() else 1.5))

        for raw_segment in self._CJK_SEGMENT_PATTERN.findall(cleaned):
            for segment in self._split_topic_segment(raw_segment):
                if 2 <= len(segment) <= 8 and self._is_valid_topic_phrase(segment):
                    candidates.append((segment, 1.8 if len(segment) >= 4 else 1.3))
                max_window = min(6, len(segment))
                for size in range(3, max_window + 1):
                    for start in range(0, len(segment) - size + 1):
                        phrase = segment[start : start + size]
                        if not self._is_valid_topic_phrase(phrase):
                            continue
                        candidates.append((phrase, 0.7 + size * 0.22))
        return candidates

    def _split_topic_segment(self, value: str) -> List[str]:
        """将长中文片段按叙述连接词切开，保留更像议题的子短语。"""
        parts = [part.strip() for part in self._TOPIC_SEGMENT_SPLIT_PATTERN.split(value) if part.strip()]
        return [part for part in parts if len(part) >= 2] or [value]

    def _normalize_topic_phrase(self, value: str) -> str:
        """清洗候选议题短语。"""
        text = self._clean_text(value)
        text = self._URL_PATTERN.sub(" ", text)
        text = self._HASHTAG_PATTERN.sub(lambda match: match.group(1).strip(), text)
        text = self._MENTION_PATTERN.sub(" ", text)
        text = re.sub(r"[^\w\u4e00-\u9fff\s\-+/]", " ", text)
        text = re.sub(r"\s+", " ", text).strip(" -_/+")
        return text

    def _is_valid_topic_phrase(self, value: str) -> bool:
        """判断候选短语是否适合作为议题线索。"""
        text = self._clean_text(value)
        if not text or len(text) < 2 or len(text) > 16:
            return False
        if text.lower() in self._ENGLISH_TOPIC_STOPWORDS or text in self._TOPIC_STOPWORDS:
            return False
        if any(marker in text for marker in self._TOPIC_INTERNAL_STOPWORDS):
            return False
        if text[0] in self._TOPIC_EDGE_STOP_CHARS or text[-1] in self._TOPIC_EDGE_STOP_CHARS:
            return False
        if text.isdigit():
            return False
        if re.fullmatch(r"[_\-/+]+", text):
            return False
        if re.fullmatch(r"[A-Za-z0-9_\-/+]+", text):
            return len(text) >= 3 and text.lower() not in self._ENGLISH_TOPIC_STOPWORDS
        if re.fullmatch(r"[\u4e00-\u9fff]+", text):
            if len(text) == 2 and text in self._TOPIC_STOPWORDS:
                return False
            return True
        return False

    def _build_profile_evidence_items(
        self,
        context: PortraitGenerationContext,
    ) -> List[EvidenceItem]:
        """构建来自个人资料的证据条目。"""
        items: List[EvidenceItem] = []
        for key, value in context.source.user_info.items():
            if isinstance(value, (dict, list, tuple, set)):
                continue
            text = self._clean_text(value)
            if not text:
                continue
            items.append(
                EvidenceItem(
                    evidence_id=f"profile:{key}",
                    source_type="profile",
                    source_ref=key,
                    summary=f"{key}={self._truncate_text(text, 40)}",
                )
            )
        return items[:20]

    def _build_stats_evidence_items(
        self,
        context: PortraitGenerationContext,
    ) -> List[EvidenceItem]:
        """构建来自统计特征的证据条目。"""
        stats = context.stats
        return [
            EvidenceItem(
                evidence_id="stat:post_count",
                source_type="stat",
                source_ref="post_count",
                summary=f"总发文数={stats.post_count}",
            ),
            EvidenceItem(
                evidence_id="stat:top_active_hours",
                source_type="stat",
                source_ref="top_active_hours",
                summary=f"高活跃时段={stats.top_active_hours}",
            ),
            EvidenceItem(
                evidence_id="stat:content_ratio",
                source_type="stat",
                source_ref="content_ratio",
                summary=(
                    "内容类型占比="
                    f"原创{stats.original_ratio}、评论{stats.comment_ratio}、"
                    f"转发{stats.forward_ratio}、引用{stats.quote_ratio}"
                ),
            ),
            EvidenceItem(
                evidence_id="stat:engagement",
                source_type="stat",
                source_ref="engagement",
                summary=(
                    "平均互动="
                    f"点赞{stats.avg_like_count}、评论{stats.avg_comment_count}、分享{stats.avg_share_count}"
                ),
            ),
        ]

    def _build_user_info_summary(self, user_info: Dict[str, Any]) -> Dict[str, Any]:
        """过滤并压缩用户资料。"""
        summary: Dict[str, Any] = {}
        for key, value in user_info.items():
            if isinstance(value, (dict, list, tuple, set)):
                continue
            text = self._clean_text(value)
            if text:
                summary[key] = text
        return summary

    def _build_stats_summary(self, stats: UserStats) -> Dict[str, Any]:
        """构建用于提示词的统计摘要。"""
        return {
            "username": stats.username,
            "nickname": stats.nickname,
            "post_count": stats.post_count,
            "recent_post_count": stats.recent_post_count,
            "recent_30d_post_count": stats.recent_30d_post_count,
            "recent_7d_post_count": stats.recent_7d_post_count,
            "avg_post_length": stats.avg_post_length,
            "avg_engagement": stats.avg_engagement,
            "content_ratios": {
                "original": stats.original_ratio,
                "comment": stats.comment_ratio,
                "repost": stats.forward_ratio,
                "quote": stats.quote_ratio,
            },
            "top_active_hours": stats.top_active_hours,
            "account_influence": stats.account_influence,
            "fans_count": stats.fans_count,
            "follow_count": stats.follow_count,
            "hashtag_post_count": stats.hashtag_post_count,
            "mention_post_count": stats.mention_post_count,
            "url_post_count": stats.url_post_count,
        }

    def _validate_stage_output(
        self,
        payload: Dict[str, Any],
        model: type[BaseModel],
        stage_name: str,
    ) -> BaseModel:
        """校验阶段输出。"""
        if not payload:
            raise PortraitGenerationError(stage_name, "模型未返回合法 JSON。")
        try:
            return model.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            raise PortraitGenerationError(stage_name, f"结构化输出校验失败：{exc}") from exc

    def _compute_post_influence(self, post: UserPost) -> float:
        """计算单条帖子的影响力。"""
        weights = self.post_influence_weights
        influence = (
            post.like_count * weights["post_like"]
            + post.comment_count * weights["post_comment"]
            + post.share_count * weights["post_share"]
        )
        time_weight = math.exp(
            -weights["time_decay"] * max(0, self.reference_timestamp - post.timestamp)
        )
        return influence * time_weight

    def _compute_user_influence(self, stats: UserStats) -> float:
        """计算账号综合影响力。"""
        weights = self.user_influence_weights
        fans_influence = math.log10(stats.fans_count) if stats.fans_count > 10 else 1.0
        post_influence = math.log10(stats.post_influence) if stats.post_influence > 10 else 1.0
        activity = math.log10(stats.recent_post_count) if stats.recent_post_count > 10 else 1.0
        return (
            fans_influence * weights["user_fans"]
            + post_influence * weights["user_post_influence"]
            + activity * weights["user_activity"]
        )

    def _normalize_probability_map(
        self,
        values: Dict[str, Any],
        expected_keys: Sequence[str],
    ) -> Dict[str, float]:
        """将概率字典规范化。"""
        normalized = {
            key: max(0.0, self._safe_float(values.get(key, 0.0)))
            for key in expected_keys
        }
        total = sum(normalized.values())
        if total <= 0:
            if not expected_keys:
                return {}
            default_value = round(1.0 / len(expected_keys), 6)
            return {key: default_value for key in expected_keys}
        return {key: round(value / total, 6) for key, value in normalized.items()}

    def _fill_agent_profile_defaults(
        self,
        context: PortraitGenerationContext,
        stable_profile: StableProfile,
        behavior_profile: BehaviorProfile,
        agent_profile: AgentProfile,
    ) -> AgentProfile:
        """用稳定画像和行为画像兜底主人格摘要。"""
        if not agent_profile.identity_summary:
            role_text = stable_profile.social_role.role or "社交媒体用户"
            summary = stable_profile.profile_summary or "具备相对稳定议题偏好的用户"
            agent_profile.identity_summary = (
                f"{context.stats.nickname or context.stats.username}是一名{role_text}，{summary}"
            )

        if not agent_profile.interest_summary:
            interests = self._dedupe_strings(
                [*stable_profile.long_term_interests, *stable_profile.content_topics]
            )
            agent_profile.interest_summary = (
                "长期关注：" + "、".join(interests[:6])
                if interests
                else "长期兴趣暂未明确，但会围绕已有证据中的稳定议题发声。"
            )

        if not agent_profile.value_summary:
            values = [item.stance for item in stable_profile.value_anchors if item.stance]
            agent_profile.value_summary = (
                "核心立场：" + "；".join(values[:4])
                if values
                else "整体立场应与已有证据保持一致，并避免过度延伸。"
            )

        if not agent_profile.style_summary:
            style = stable_profile.expression_style
            style_parts = [
                item
                for item in [style.tone, style.formality, style.verbosity, style.argument_style]
                if item
            ]
            agent_profile.style_summary = (
                "表达风格：" + "、".join(style_parts)
                if style_parts
                else "表达应保持与历史文本接近的语气和论证方式。"
            )

        if not agent_profile.behavior_summary:
            action_preferences = behavior_profile.action_preferences.model_dump()
            top_actions = sorted(
                action_preferences.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:2]
            action_text = "、".join(name for name, _ in top_actions if _ > 0)
            agent_profile.behavior_summary = (
                f"平台行为上更偏向：{action_text}。"
                if action_text
                else "平台行为偏好暂不稳定，应根据已有行为统计谨慎行动。"
            )

        if not agent_profile.interaction_summary:
            interaction_flags = []
            if behavior_profile.interaction_preferences.prefers_hot_topics:
                interaction_flags.append("更关注热点议题")
            if behavior_profile.interaction_preferences.prefers_followed_authors:
                interaction_flags.append("更关注熟悉对象")
            if behavior_profile.interaction_preferences.prefers_argumentative_threads:
                interaction_flags.append("更容易进入有争议度的话题串")
            agent_profile.interaction_summary = (
                "互动偏好：" + "；".join(interaction_flags)
                if interaction_flags
                else "互动偏好应以已有行为证据为准，避免凭空增加社交习惯。"
            )

        if not agent_profile.speaking_rules:
            agent_profile.speaking_rules = [
                "保持与长期画像一致的语气和立场",
                "证据不足时避免编造细节",
            ]
        if not agent_profile.action_rules:
            agent_profile.action_rules = [
                "优先执行与历史行为倾向一致的动作",
                "遇到高相关议题时可以更主动表达",
            ]
        if not agent_profile.avoidance_rules:
            agent_profile.avoidance_rules = [
                "不要凭空虚构身份经历",
                "不要输出与稳定画像明显矛盾的观点",
            ]
        if not agent_profile.initial_focus_topics:
            agent_profile.initial_focus_topics = stable_profile.content_topics[:4]
        if not agent_profile.current_goal_hint:
            agent_profile.current_goal_hint = "围绕长期关注议题进行稳定、克制的表达。"
        return agent_profile

    @staticmethod
    def _join_rule_lines(values: Sequence[str]) -> str:
        """将规则列表拼接成多行文本。"""
        lines = [f"- {item}" for item in values if str(item).strip()]
        return "\n".join(lines) if lines else "- 暂无额外规则"

    @staticmethod
    def _build_evidence_id(post: UserPost, index: int) -> str:
        """构建证据 ID。"""
        if post.post_id is not None:
            return f"post:{post.post_id}"
        return f"post_idx:{index}"

    @staticmethod
    def _dedupe_strings(values: Sequence[str]) -> List[str]:
        """去重并清洗字符串列表。"""
        seen = set()
        result: List[str] = []
        for value in values:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result

    @staticmethod
    def _normalize_content_type(value: Any) -> str:
        """归一化内容类型。"""
        text = str(value or "").strip().lower()
        if text in {"comment", "reply"}:
            return text
        if text in {"repost", "retweet"}:
            return "repost"
        if text == "quote":
            return "quote"
        return "original"

    @staticmethod
    def _truncate_text(value: str, limit: int = 80) -> str:
        """截断文本。"""
        text = str(value or "").strip()
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    @staticmethod
    def _clamp_score(value: Any) -> float:
        """约束分数范围。"""
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return round(max(0.0, min(1.0, score)), 6)

    @staticmethod
    def _clean_text(value: Any) -> str:
        """清洗文本。"""
        text = str(value or "").strip()
        return "" if text.lower() == "nan" else text

    @staticmethod
    def _safe_int(value: Any) -> int:
        """安全转整数。"""
        try:
            if value is None:
                return 0
            return int(value)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _safe_float(value: Any) -> float:
        """安全转浮点数。"""
        try:
            if value is None:
                return 0.0
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    _HASHTAG_PATTERN = re.compile(r"#([^#\s]{1,30})#")
    _MENTION_PATTERN = re.compile(r"@([A-Za-z0-9_\-\u4e00-\u9fff]{1,30})")
    _URL_PATTERN = re.compile(r"https?://\S+")
    _CJK_SEGMENT_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,24}")
    _TOPIC_SEGMENT_SPLIT_PATTERN = re.compile(
        r"如果|因为|但是|然后|正在|继续|已经|不会|没有|不是|以及|或者|还有|目前|值得|需要|都会|将会|可能|发动|升级|上升|下降|改变|推动|部署|承压|回应|反击|访问|会谈|表态|风险|格局|之后|以后|对|和|与|及"
    )
    _TOPIC_EDGE_STOP_CHARS = set("的了和是在就都而及与对把被将给从向并让为因但还也又呢吗啊吧着过得")
    _TOPIC_INTERNAL_STOPWORDS = {
        "如果",
        "因为",
        "但是",
        "然后",
        "正在",
        "继续",
        "已经",
        "不会",
        "没有",
        "不是",
        "以及",
        "或者",
        "还有",
        "目前",
        "值得",
        "需要",
        "都会",
        "将会",
        "可能",
    }
    _TOPIC_STOPWORDS = {
        "我们",
        "你们",
        "他们",
        "自己",
        "大家",
        "有人",
        "这个",
        "那个",
        "这些",
        "那些",
        "这里",
        "那里",
        "今天",
        "昨天",
        "刚刚",
        "目前",
        "已经",
        "还是",
        "真的",
        "觉得",
        "表示",
        "可以",
        "应该",
        "不是",
        "没有",
        "如果",
        "因为",
        "但是",
        "然后",
        "什么",
        "怎么",
        "为什么",
        "一下",
        "一种",
        "一些",
        "很多",
        "任何",
        "所有",
        "其中",
        "时候",
        "事情",
        "问题",
        "情况",
        "内容",
        "平台",
        "网友",
        "视频",
        "图片",
        "评论",
        "转发",
        "点赞",
        "帖子",
        "发文",
        "用户",
        "账号",
        "消息",
    }
    _ENGLISH_TOPIC_STOPWORDS = {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "into",
        "onto",
        "over",
        "under",
        "about",
        "after",
        "before",
        "because",
        "while",
        "where",
        "when",
        "what",
        "which",
        "who",
        "whose",
        "will",
        "would",
        "could",
        "should",
        "have",
        "has",
        "had",
        "been",
        "being",
        "were",
        "was",
        "are",
        "is",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "an",
        "a",
        "or",
        "not",
        "no",
        "yes",
        "you",
        "your",
        "they",
        "them",
        "their",
        "we",
        "our",
        "he",
        "she",
        "it",
        "its",
        "rt",
    }
