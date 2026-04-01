from moss_agent_client.agent_logger import logger
from moss_agent_client.memory import (
    ActionTrace,
    AgentMemoryContext,
    AgentState,
    DecisionResultPayload,
    EnvironmentSnapshot,
    EventMemory,
    EventMemoryRecord,
    ShortTermMemory,
    ShortTermRecord,
    StaticContext,
)
from moss_agent_client.prompt_builder import PromptBuilder


class MemoryManager:
    def __init__(
        self,
        username: str,
        static_context: StaticContext,
        short_term_max_rounds: int = 3,
        short_term_max_posts: int = 3,
        event_max_size: int = 50,
    ):
        self.username = username
        self.static_context = static_context
        self.short_term_max_rounds = short_term_max_rounds
        self.short_term_max_posts = short_term_max_posts
        self.event_max_size = event_max_size
        self.context = AgentMemoryContext(
            static=static_context,
            short_term=ShortTermMemory(max_rounds=short_term_max_rounds),
            state=AgentState(),
            event_memory=EventMemory(max_size=event_max_size),
        )

    def select_relevant_events(
        self,
        snapshot: EnvironmentSnapshot,
        top_k: int = 5,
    ) -> list[EventMemoryRecord]:
        if not self.context.event_memory.records:
            return []

        current_post_ids = {item.post_id for item in snapshot.feed_items}
        current_authors = {
            item.author_name for item in snapshot.feed_items if item.author_name
        }
        current_text = snapshot.combined_text()
        focus_topics = set(self.context.state.focus_topics)
        attention_target = self.context.state.attention_target

        scored_records = []
        for record in self.context.event_memory.records:
            score = record.importance
            if (
                record.source_post_id is not None
                and record.source_post_id in current_post_ids
            ):
                score += 1.0
            if any(user in current_authors for user in record.related_users):
                score += 0.7
            if record.topic and record.topic in current_text:
                score += 0.5
            if record.topic and record.topic in focus_topics:
                score += 0.3
            if attention_target and attention_target in " ".join(record.related_users):
                score += 0.2
            scored_records.append((score, record))

        scored_records.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored_records[:top_k]]

    def update_after_step(
        self,
        round_id: int,
        snapshot: EnvironmentSnapshot,
        actions: list[ActionTrace],
        decision_result: DecisionResultPayload,
    ) -> None:
        self.record_step(
            round_id=round_id,
            snapshot=snapshot,
            actions=actions,
            output=decision_result.final_output,
        )

        self.apply_decision_result(
            snapshot=snapshot,
            decision_result=decision_result,
        )

    def record_step(
        self,
        round_id: int,
        snapshot: EnvironmentSnapshot,
        actions: list[ActionTrace],
        output: str,
    ) -> None:
        self._update_short_term_memory(
            round_id=round_id,
            snapshot=snapshot,
            actions=actions,
            output=output,
        )

    def apply_decision_result(
        self,
        snapshot: EnvironmentSnapshot,
        decision_result: DecisionResultPayload,
    ) -> None:
        if not decision_result.is_structured:
            logger.warning(
                f"Agent {self.username} 本轮结构化结果解析失败，保留上一轮状态与事件记忆。"
            )
            return

        self.context.state.update_from_payload(decision_result.state)
        logger.info(
            f"Agent {self.username} 状态已更新：{PromptBuilder.render_state(self.context.state)}"
        )

        if not decision_result.event_memory.should_store:
            return

        record = EventMemoryRecord(
            time=snapshot.current_time,
            summary=decision_result.event_memory.summary.strip(),
            importance=decision_result.event_memory.importance,
            topic=decision_result.event_memory.topic.strip(),
            related_users=[
                item.strip()
                for item in decision_result.event_memory.related_users
                if item.strip()
            ],
            source_post_id=decision_result.event_memory.source_post_id,
            impact=decision_result.event_memory.impact.strip(),
        )
        self.context.event_memory.add(record)
        logger.info(
            f"Agent {self.username} 事件记忆已写入：{PromptBuilder.render_event_record(record)}"
        )

    def _update_short_term_memory(
        self,
        round_id: int,
        snapshot: EnvironmentSnapshot,
        actions: list[ActionTrace],
        output: str,
    ) -> None:
        key_feed_items = self._select_key_feed_items(snapshot, actions)
        record = ShortTermRecord(
            round_id=round_id,
            time=snapshot.current_time,
            feed_items=key_feed_items,
            actions=actions,
            output=output or "无",
        )
        self.context.short_term.add(record)
        logger.info(
            f"Agent {self.username} 短期记忆已更新，当前窗口大小：{len(self.context.short_term.records)}"
        )

    def _select_key_feed_items(
        self,
        snapshot: EnvironmentSnapshot,
        actions: list[ActionTrace],
    ) -> list:
        if not snapshot.feed_items:
            return []

        post_ids_from_actions = {
            action.post_id
            for action in actions
            if action.post_id is not None
        }
        selected_items = []
        selected_post_ids = set()

        for item in snapshot.feed_items:
            if (
                item.post_id in post_ids_from_actions
                and item.post_id not in selected_post_ids
            ):
                selected_items.append(item)
                selected_post_ids.add(item.post_id)
                if len(selected_items) >= self.short_term_max_posts:
                    return selected_items

        for item in snapshot.feed_items:
            if item.post_id in selected_post_ids:
                continue
            selected_items.append(item)
            selected_post_ids.add(item.post_id)
            if len(selected_items) >= self.short_term_max_posts:
                break

        return selected_items
