import time
import traceback
from typing import Any, Optional

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import create_model

from core.experiment_config import MemoryExperimentConfig
from moss_agent_client.actions import ACTION_SPECS, ACTION_SPEC_BY_NAME, ActionSpec
from moss_agent_client.agent_logger import logger
from moss_agent_client.memory import (
    ActionTrace,
    DecisionResultPayload,
    EnvironmentSnapshot,
    FeedItemSnapshot,
    StaticContext,
)
from moss_agent_client.memory_manager import MemoryManager
from moss_agent_client.prompt_builder import PromptBuilder
from moss_agent_client.remote_platform import RemotePlatform
from moss_agent_client.utils import extract_json_dict, normalize_text_content

FIXED_USER_INFO_TEMPLATE = """你现在需要扮演 {name}，一个社交网络上的用户。

[身份概述]
{identity_summary}

[长期兴趣]
{interest_summary}

[价值与立场]
{value_summary}

[表达风格]
{style_summary}

[行为偏好]
{behavior_summary}

[互动规则]
{interaction_summary}

[表达要求]
{speaking_rules_text}

[行动要求]
{action_rules_text}

[避免事项]
{avoidance_rules_text}"""

FIXED_USER_INFO_FIELDS = (
    "identity_summary",
    "interest_summary",
    "value_summary",
    "style_summary",
    "behavior_summary",
    "interaction_summary",
    "speaking_rules_text",
    "action_rules_text",
    "avoidance_rules_text",
)

SIMPLE_USER_TEMPLATE = """你现在需要扮演 {name}，一个社交网络上的用户。
个人简介：{bio}
{behavior_expectation}"""

TIER_BEHAVIOR = {
    5: "你是一个社交媒体上的重度活跃用户和创新者，乐于表达观点、发起讨论，是社区的意见领袖。你会积极参与信息流互动，频繁点赞、评论、转发感兴趣的内容。",
    4: "你是一个比较活跃的社交媒体用户，愿意尝试新话题，经常参与讨论。你会有选择地浏览信息流，对自己感兴趣的内容进行点赞、评论或转发。",
    3: "你是一个比较活跃的用户，会定期浏览信息流，对自己感兴趣的内容进行点赞、评论或转发。如果没有特别感兴趣的内容，你可以选择不互动。",
    2: "你偶尔会浏览社交网络，很少主动发帖，主要在遇到特别感兴趣的内容时才会点赞或评论。",
    1: "你几乎不发帖，以浏览为主。只有在极少数情况下才会互动。",
}

CURRENT_EVENT_TEMPLATE = """[全局事件]
当前世界关注的核心事件如下：
"{global_event_description}"
"""


class MossAgent:
    def __init__(
        self,
        platform: RemotePlatform,
        username: str,
        nickname: str,
        bio: str,
        global_event: str,
        llm: ChatOpenAI,
        user_info: Optional[dict[str, Any]] = None,
        user_info_template: Optional[str] = None,
        profile_mode: str = "default",
        memory_config: Optional[MemoryExperimentConfig] = None,
    ):
        self.platform = platform
        self.username = username
        self.nickname = nickname
        self.bio = bio
        self.user_info = user_info or {}
        self.global_event = global_event
        self.user_data = None
        self.user_info_template = user_info_template
        self.profile_mode = profile_mode
        self.round_id = 0
        self._step_actions: list[ActionTrace] = []
        memory_config = memory_config or MemoryExperimentConfig()
        self.step_retry_limit = memory_config.step_retry_limit

        self.llm = llm
        self.static_context = self._build_static_context()
        self.memory_manager = MemoryManager(
            username=self.username,
            static_context=self.static_context,
            short_term_max_rounds=memory_config.short_term_max_rounds,
            short_term_max_posts=memory_config.short_term_max_posts,
            event_max_size=memory_config.event_max_size,
            event_decay_lambda=memory_config.event_decay_lambda,
            context_boost_cap=memory_config.context_boost_cap,
        )
        self.tools = self._create_tools()
        self.system_prompt = PromptBuilder.build_system_prompt(self.static_context)
        self.agent = create_agent(
            self.llm, tools=self.tools, system_prompt=self.system_prompt
        )

    async def start(self):
        # B-5：tier 与 belief_text 全链路写入 DB，供在线打分使用
        tier = self._resolve_tier()
        belief_text = self._resolve_belief_text()
        self.user_data = await self.platform.register_or_login(
            username=self.username,
            nickname=self.nickname,
            bio=self.bio,
            user_info=self.user_info,
            tier=tier,
            belief_text=belief_text,
        )
        logger.info(f"Agent {self.nickname} started. User ID: {self.user_data.id}")

    def _resolve_tier(self) -> int:
        """解析层级：simple 用 user_info.tier；default/custom 优先画像 influence_tier。"""
        raw = self.user_info.get("tier")
        if raw is None:
            raw = self.user_info.get("influence_tier")
        try:
            tier = int(raw) if raw is not None else 3
        except (TypeError, ValueError):
            tier = 3
        return max(1, min(5, tier))

    def _resolve_belief_text(self) -> str:
        """解析立场/兴趣文本：default/custom 用 identity_summary+interest_summary，
        simple 用 bio；缺失时回退 bio。与 ABM 的画像 embedding 口径一致（B-5.5）。"""
        if self.profile_mode != "simple":
            identity = str(self.user_info.get("identity_summary") or "").strip()
            interest = str(self.user_info.get("interest_summary") or "").strip()
            if identity or interest:
                return f"{identity} {interest}".strip()
        return (self.bio or "").strip()

    def _build_static_context(self) -> StaticContext:
        if self.profile_mode == "simple":
            tier = self.user_info.get("tier", 3) if self.user_info else 3
            behavior = TIER_BEHAVIOR.get(tier, TIER_BEHAVIOR[3])
            return StaticContext(
                profile_text=SIMPLE_USER_TEMPLATE.format(
                    name=self.nickname or self.username,
                    bio=self.bio,
                    behavior_expectation=behavior,
                ),
                global_event_text=CURRENT_EVENT_TEMPLATE.format(
                    global_event_description=self.global_event
                ),
            )

        if self.user_info_template:
            try:
                profile_text = self.user_info_template.format(**self.user_info)
            except KeyError as exc:
                missing_key = str(exc.args[0])
                raise ValueError(
                    f"Agent {self.username} 的 user_info_template 缺少字段：{missing_key}"
                ) from exc
        else:
            missing_fields = [
                field_name
                for field_name in FIXED_USER_INFO_FIELDS
                if field_name not in self.user_info
            ]
            if missing_fields:
                raise ValueError(
                    f"Agent {self.username} 的固定画像模板缺少字段：{', '.join(missing_fields)}"
                )
            profile_text = FIXED_USER_INFO_TEMPLATE.format(
                **self.user_info,
                name=self.user_info.get("name") or self.nickname,
            )
        global_event_text = CURRENT_EVENT_TEMPLATE.format(
            global_event_description=self.global_event
        )
        return StaticContext(
            profile_text=profile_text,
            global_event_text=global_event_text,
        )

    async def execute_action(
        self,
        action_type: str,
        params: dict[str, Any],
    ):
        spec = ACTION_SPEC_BY_NAME.get(action_type)
        if spec is None:
            raise ValueError(f"Unknown action type: {action_type}")

        method = getattr(self.platform, spec.platform_method)
        args = [params.get(arg_name) for arg_name in spec.arg_names]
        return await method(*args)

    def _create_tools(self):
        """创建平台动作工具。"""
        tools = []

        for spec in ACTION_SPECS:
            tools.append(self._build_tool_from_spec(spec))

        return tools

    def _build_tool_from_spec(self, spec: ActionSpec):
        async def _tool_impl(**kwargs):
            trace = self._build_action_trace(spec.name, kwargs)
            logger.info(
                f"Agent {self.nickname} is executing {spec.name} with params: {kwargs}"
            )
            try:
                response = await self.execute_action(spec.name, kwargs)
                if spec.result_mode == "post_detail":
                    item = self._build_feed_item_snapshot(response)
                    comments = []
                    for comment in response.comments:
                        comment_author = (
                            comment.author_nickname or f"User {comment.user_id}"
                        )
                        comments.append(
                            f"- {comment_author}: {comment.content} "
                            f"(点赞 {comment.like_count}，我已点赞 {comment.is_liked})"
                        )
                    trace.message = "已获取帖子详情"
                    self._append_step_action(trace)
                    return PromptBuilder.render_post_detail(item, comments)

                payload = response.model_dump()
                self._apply_action_result(trace, payload)
                self._append_step_action(trace)
                return payload
            except Exception:
                trace.status = "error"
                trace.message = "动作执行失败"
                self._append_step_action(trace)
                raise

        schema_fields = {}
        for arg_name in spec.arg_names:
            if arg_name in {"post_id", "comment_id"}:
                schema_fields[arg_name] = (int, ...)
            else:
                schema_fields[arg_name] = (str, ...)
        args_schema = create_model(f"{spec.name.title()}Args", **schema_fields)

        return StructuredTool.from_function(
            func=None,
            coroutine=_tool_impl,
            name=spec.name,
            description=spec.description,
            args_schema=args_schema,
        )

    def _build_feed_item_snapshot(self, post) -> FeedItemSnapshot:
        return FeedItemSnapshot(
            post_id=post.id,
            author_name=post.author_nickname or f"User {post.user_id}",
            created_at=str(post.created_at),
            content=post.content or "",
            post_type=post.type,
            ref_id=post.ref_id,
            like_count=post.stats.get("like_count", 0),
            reply_count=post.stats.get("reply_count", 0),
            share_count=post.stats.get("share_count", 0),
            retweet_count=post.stats.get("retweet_count", 0),
            quote_count=post.stats.get("quote_count", 0),
            is_liked=post.is_liked,
            is_reposted=post.is_reposted,
        )

    def _reset_step_actions(self) -> None:
        self._step_actions = []

    def _append_step_action(self, trace: ActionTrace) -> None:
        self._step_actions.append(trace)

    def _get_step_actions(self) -> list[ActionTrace]:
        return list(self._step_actions)

    async def get_environment_snapshot(self) -> Optional[EnvironmentSnapshot]:
        try:
            feed = await self.platform.get_feed()
            post_ids = [p.id for p in feed]
            logger.info(f"Agent {self.nickname} saw posts: {post_ids}")

            time_data = await self.platform.get_time()
            current_time = str(time_data.get("current_time", ""))
            feed_items = [self._build_feed_item_snapshot(post) for post in feed]
            return EnvironmentSnapshot(current_time=current_time, feed_items=feed_items)
        except Exception as e:
            logger.error(f"Failed to get perception: {e}")
            traceback.print_exc()
            return None

    def _build_action_trace(
        self,
        action_type: str,
        raw_args: Any,
    ) -> ActionTrace:
        args = raw_args if isinstance(raw_args, dict) else {}

        post_id = args.get("post_id")
        if post_id is None:
            post_id = args.get("original_post_id")

        comment_id = args.get("comment_id")
        content = args.get("content", "")

        try:
            parsed_post_id = int(post_id) if post_id is not None else None
        except (TypeError, ValueError):
            parsed_post_id = None

        try:
            parsed_comment_id = int(comment_id) if comment_id is not None else None
        except (TypeError, ValueError):
            parsed_comment_id = None

        normalized_args = args if isinstance(args, dict) else {"raw": raw_args}
        return ActionTrace(
            action_type=action_type,
            post_id=parsed_post_id,
            comment_id=parsed_comment_id,
            content=str(content or ""),
            raw_args=normalized_args,
        )

    @staticmethod
    def _parse_optional_int(value: Any) -> Optional[int]:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _extract_action_result_payload(self, raw_result: Any) -> dict[str, Any]:
        if isinstance(raw_result, dict):
            return raw_result

        payload = extract_json_dict(
            raw_result,
            log_prefix=f"Agent {self.nickname} tool result",
        )
        return payload if isinstance(payload, dict) else {}

    def _apply_action_result(
        self,
        trace: ActionTrace,
        raw_result: Any,
        tool_status: str = "",
    ) -> None:
        payload = self._extract_action_result_payload(raw_result)
        if not payload:
            if trace.action_type == "get_post":
                trace.message = "已获取帖子详情"
            elif tool_status:
                trace.status = tool_status
            return

        status = payload.get("status") or tool_status
        if status:
            trace.status = str(status)

        message = payload.get("message")
        if message:
            trace.message = str(message)

        result_data = payload.get("data")
        if isinstance(result_data, dict):
            trace.result_data = result_data
        elif result_data is not None:
            trace.result_data = {"raw": result_data}

        result_post_id = None
        if isinstance(result_data, dict):
            result_post_id = result_data.get("post_id")
        if result_post_id is None:
            result_post_id = payload.get("post_id")
        trace.result_post_id = self._parse_optional_int(result_post_id)

    def _collect_actions_from_messages(self, messages: list[Any]) -> list[ActionTrace]:
        actions: list[ActionTrace] = []
        trace_by_tool_call_id: dict[str, ActionTrace] = {}

        for msg in messages:
            tool_calls = getattr(msg, "tool_calls", None) or []
            for tool_call in tool_calls:
                trace = self._build_action_trace(
                    action_type=tool_call["name"],
                    raw_args=tool_call["args"],
                )
                actions.append(trace)

                tool_call_id = tool_call.get("id")
                if tool_call_id:
                    trace_by_tool_call_id[str(tool_call_id)] = trace

            tool_call_id = getattr(msg, "tool_call_id", None)
            if not tool_call_id:
                continue

            trace = trace_by_tool_call_id.get(str(tool_call_id))
            if trace is None:
                continue

            raw_result = getattr(msg, "artifact", None)
            if raw_result in (None, ""):
                raw_result = getattr(msg, "content", None)
            self._apply_action_result(
                trace=trace,
                raw_result=raw_result,
                tool_status=str(getattr(msg, "status", "") or ""),
            )

        return actions

    @staticmethod
    def _is_valid_decision_payload(payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False

        required_keys = {"final_output", "state", "event_memory"}
        if not required_keys.issubset(payload):
            return False

        return isinstance(payload.get("state"), dict) and isinstance(
            payload.get("event_memory"), dict
        )

    def _parse_decision_result(self, raw_content: Any) -> DecisionResultPayload:
        payload = extract_json_dict(
            raw_content,
            log_prefix=f"Agent {self.nickname}",
        )
        if not payload:
            return DecisionResultPayload(
                final_output=normalize_text_content(raw_content),
                is_structured=False,
            )

        if not self._is_valid_decision_payload(payload):
            logger.warning(
                f"Agent {self.nickname} 主决策结果缺少必需字段，已按非结构化输出处理。"
            )
            return DecisionResultPayload(
                final_output=normalize_text_content(raw_content),
                is_structured=False,
            )

        try:
            return DecisionResultPayload.model_validate(payload)
        except Exception as e:
            logger.warning(f"Agent {self.nickname} 主决策结果校验失败：{e}")
            return DecisionResultPayload(
                final_output=normalize_text_content(raw_content),
                is_structured=False,
            )

    async def step(self):
        self.round_id += 1
        last_error: Optional[Exception] = None
        snapshot: Optional[EnvironmentSnapshot] = None

        for attempt in range(1, self.step_retry_limit + 1):
            if snapshot is None:
                snapshot = await self.get_environment_snapshot()
                if not snapshot:
                    last_error = RuntimeError(
                        f"Agent {self.nickname} 获取环境信息失败（第 {attempt} 次尝试）"
                    )
                    logger.error(str(last_error))
                    if attempt >= self.step_retry_limit:
                        raise last_error
                    continue
                logger.info(
                    f"Agent {self.nickname} 已冻结本轮环境快照，后续重试将复用这份快照。"
                )
            elif attempt > 1:
                logger.info(
                    f"Agent {self.nickname} 第 {attempt} 次尝试复用首次获取的环境快照。"
                )

            self._reset_step_actions()
            relevant_events = self.memory_manager.select_relevant_events(
                snapshot=snapshot,
                current_round=self.round_id,
            )
            full_input = PromptBuilder.build_runtime_context(
                context=self.memory_manager.context,
                snapshot=snapshot,
                relevant_events=relevant_events,
            )
            logger.info(
                f"Agent {self.nickname} input (attempt {attempt}/{self.step_retry_limit}): {full_input}"
            )

            short_term_recorded = False
            try:
                messages = [HumanMessage(content=full_input)]
                result = await self.agent.ainvoke({"messages": messages})

                decision_result = DecisionResultPayload()

                if isinstance(result, dict) and "messages" in result:
                    final_msgs = result["messages"]
                    if final_msgs:
                        decision_result = self._parse_decision_result(
                            final_msgs[-1].content
                        )
                else:
                    decision_result = self._parse_decision_result(
                        result.get("output", "")
                    )

                actions = self._get_step_actions()
                self.memory_manager.record_step(
                    round_id=self.round_id,
                    snapshot=snapshot,
                    actions=actions,
                    output=decision_result.final_output,
                )
                short_term_recorded = True
                self.memory_manager.apply_decision_result(
                    snapshot=snapshot,
                    decision_result=decision_result,
                    round_id=self.round_id,
                )

                if attempt > 1:
                    logger.info(
                        f"Agent {self.nickname} 在第 {attempt} 次尝试时完成本轮执行。"
                    )
                return
            except Exception as e:
                last_error = e
                logger.error(
                    f"Agent {self.nickname} 第 {attempt} 次 step 尝试失败：{e}"
                )
                traceback.print_exc()
                actions = self._get_step_actions()
                if actions and not short_term_recorded:
                    try:
                        self.memory_manager.record_step(
                            round_id=self.round_id,
                            snapshot=snapshot,
                            actions=actions,
                            output=f"本轮第 {attempt} 次尝试在后处理中断",
                        )
                    except Exception as memory_error:
                        logger.error(
                            f"Agent {self.nickname} 记录部分成功动作失败：{memory_error}"
                        )
                        traceback.print_exc()

                if attempt >= self.step_retry_limit:
                    raise

                logger.warning(
                    f"Agent {self.nickname} 将基于当前记忆进行第 {attempt + 1} 次重试。"
                )
            finally:
                self._reset_step_actions()

        if last_error is not None:
            raise last_error


if __name__ == "__main__":
    import asyncio
    import logging
    import os
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.ERROR)

    username = "test_agent_1"
    nickname = "Test Agent"
    event = "A new AI model has been released."

    platform = RemotePlatform("http://localhost:8000")
    llm = ChatOpenAI(
        model="deepseek/deepseek-v3.2",
        api_key=os.environ.get("API_KEY", ""),
        base_url="https://openrouter.ai/api/v1",
        timeout=180,
    )
    agent = MossAgent(
        platform=platform,
        username=username,
        nickname=nickname,
        bio=f"I am {nickname}.",
        user_info={},
        llm=llm,
        global_event=event,
    )
    try:
        asyncio.run(agent.start())
        for _ in range(2):
            asyncio.run(agent.step())
            time.sleep(1)
    except Exception as e:
        print(f"Agent run failed: {e}")
    except KeyboardInterrupt:
        print("Agent stopped.")
