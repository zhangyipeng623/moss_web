from moss_agent_client.memory import (
    ActionTrace,
    AgentMemoryContext,
    AgentState,
    EnvironmentSnapshot,
    EventMemoryRecord,
    FeedItemSnapshot,
    ShortTermRecord,
    StaticContext,
)


class PromptBuilder:
    @staticmethod
    def build_system_prompt(static_context: StaticContext) -> str:
        return (
            f"{static_context.profile_text}\n\n"
            f"{static_context.global_event_text}\n\n"
            "用户画像是最高优先级。请始终以该用户的身份进行观察、思考和表达，优先使用简体中文，并严格保持与用户画像一致。"
            "当前信息流只是环境输入之一，而不是唯一行动来源。"
            "如果该用户的人设偏主动，即使当前信息流没有关注内容，你也可以主动发布符合用户画像长期兴趣的话题。"
            "所有内容均以第一人称输出。"
        )

    @staticmethod
    def render_feed_item(item: FeedItemSnapshot) -> str:
        lines = [
            f"帖子 ID：{item.post_id}",
            f"作者：{item.author_name}",
            f"时间：{item.created_at}",
        ]
        if item.ref_id is not None:
            lines.append(f"{item.post_type} 原帖 ID：{item.ref_id}")
        lines.extend(
            [
                f"内容：{item.content}",
                (
                    "统计："
                    f"点赞 {item.like_count}，"
                    f"评论 {item.reply_count}，"
                    f"分享 {item.share_count}，"
                    f"直接转发 {item.retweet_count}，"
                    f"带评论引用 {item.quote_count}"
                ),
                (
                    "我的互动："
                    f"已点赞 {item.is_liked}，"
                    f"已转发 {item.is_reposted}"
                ),
            ]
        )
        return "\n".join(lines)

    @classmethod
    def render_environment(cls, snapshot: EnvironmentSnapshot) -> str:
        if not snapshot.feed_items:
            feed_text = "你的信息流中暂时没有新帖子。"
        else:
            feed_text = "\n---\n".join(
                cls.render_feed_item(item) for item in snapshot.feed_items
            )
        return f"当前时间：{snapshot.current_time}\n以下是你的信息流：\n{feed_text}"

    @classmethod
    def render_post_detail(cls, item: FeedItemSnapshot, comments: list[str]) -> str:
        parts = [cls.render_feed_item(item)]
        if comments:
            parts.append("评论：\n" + "\n".join(comments))
        else:
            parts.append("评论：无")
        return "\n".join(parts)

    @staticmethod
    def render_state(state: AgentState) -> str:
        focus_topics = "、".join(state.focus_topics) if state.focus_topics else "无"
        return (
            f"情绪：{state.mood}\n"
            f"情绪原因：{state.emotion_reason or '无'}\n"
            f"情绪强度：{state.intensity:.2f}\n"
            f"当前目标：{state.current_goal or '无'}\n"
            f"关注主题：{focus_topics}\n"
            f"立场摘要：{state.stance_summary or '无'}\n"
            f"重点关注对象：{state.attention_target or '无'}"
        )

    @classmethod
    def render_action_trace(cls, trace: ActionTrace) -> str:
        pieces = [f"动作：{trace.action_type}"]
        if trace.post_id is not None:
            pieces.append(f"帖子：{trace.post_id}")
        if trace.comment_id is not None:
            pieces.append(f"评论：{trace.comment_id}")
        if trace.content:
            pieces.append(f"内容：{trace.content}")
        if trace.status:
            pieces.append(f"结果：{trace.status}")
        if trace.result_post_id is not None:
            pieces.append(f"新帖子：{trace.result_post_id}")
        if trace.message:
            pieces.append(f"返回信息：{trace.message}")
        return "；".join(pieces)

    @classmethod
    def render_short_term_record(cls, record: ShortTermRecord) -> str:
        feed_text = (
            "\n---\n".join(cls.render_feed_item(item) for item in record.feed_items)
            if record.feed_items
            else "无"
        )
        actions_text = (
            "\n".join(cls.render_action_trace(item) for item in record.actions)
            if record.actions
            else "无"
        )
        return (
            f"轮次：{record.round_id}\n"
            f"时间：{record.time}\n"
            f"信息流快照：\n{feed_text}\n"
            f"执行动作：{actions_text}\n"
            f"输出：{record.output or '无'}"
        )

    @classmethod
    def render_short_term_memory(cls, context: AgentMemoryContext) -> str:
        if not context.short_term.records:
            return ""
        return "[短期记忆]\n" + "\n\n".join(
            cls.render_short_term_record(record)
            for record in context.short_term.records
        )

    @staticmethod
    def render_event_record(record: EventMemoryRecord) -> str:
        related_users = "、".join(record.related_users) if record.related_users else "无"
        source_post = record.source_post_id if record.source_post_id is not None else "无"
        return (
            f"时间：{record.time}\n"
            f"事件：{record.summary}\n"
            f"主题：{record.topic or '无'}\n"
            f"关联用户：{related_users}\n"
            f"关联帖子：{source_post}\n"
            f"影响：{record.impact or '无'}\n"
            f"重要度：{record.importance:.2f}"
        )

    @classmethod
    def render_event_memory(cls, records: list[EventMemoryRecord]) -> str:
        if not records:
            return ""
        return "[事件记忆]\n" + "\n\n".join(
            cls.render_event_record(record) for record in records
        )

    @classmethod
    def render_action_traces(cls, actions: list[ActionTrace]) -> str:
        if not actions:
            return "无"
        return "\n".join(cls.render_action_trace(item) for item in actions)

    @classmethod
    def build_runtime_context(
        cls,
        context: AgentMemoryContext,
        snapshot: EnvironmentSnapshot,
        relevant_events: list[EventMemoryRecord],
    ) -> str:
        sections = []
        short_term_context = cls.render_short_term_memory(context)
        if short_term_context:
            sections.append(short_term_context)

        sections.append("[当前状态]\n" + cls.render_state(context.state))

        event_context = cls.render_event_memory(relevant_events)
        if event_context:
            sections.append(event_context)

        sections.append("[当前环境]\n" + cls.render_environment(snapshot))
        sections.append(
            "请基于以上信息决定下一步动作。若你需要深入查看某条帖子后再行动，可以先调用 get_post。"
        )
        sections.append(
            "用户画像是第一准则。你的所有动作选择、最终表达、状态更新和事件记忆判断，都必须首先服从用户画像。"
            "当前信息流只是环境输入之一，而不是唯一行动来源。"
            "如果用户画像本身偏主动，即使当前信息流没有你关注的内容，你也可以主动发布符合用户画像长期兴趣的话题。"
        )
        sections.append(
            "在完成所有必要的工具调用后，最后一条回复必须只输出一个 JSON 对象，不要输出解释、代码块或额外文本。"
            "JSON 结构必须为："
            '{'
            '"final_output":"",'
            '"state":{"mood":"","emotion_reason":"","intensity":0.0,"current_goal":"","focus_topics":[],"stance_summary":"","attention_target":""},'
            '"event_memory":{"should_store":false,"summary":"","importance":0.0,"topic":"","related_users":[],"source_post_id":null,"impact":""}'
            '}'
            "其中 final_output 用第一人称简短概括你本轮的最终表达或立场；"
            "state 必须严格符合用户画像；"
            "event_memory 只有在对该用户画像真正重要时才允许 should_store=true。"
        )
        return "\n\n".join(sections)
