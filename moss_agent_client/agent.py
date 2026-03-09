import time
import traceback
from typing import Dict, Any, Optional, List

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from moss_agent_client.agent_logger import logger
from moss_agent_client.remote_platform import RemotePlatform

# Prompt Template
USER_PROFILE_TEMPLATE = """You are {name}, a user on a social network.
{user_profile}"""

CURRENT_EVENT_TEMPLATE = """[CURRENT EVENT]
The world is currently focusing on the following event:
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
        user_info: Optional[Dict[str, Any]] = None,
        user_info_template: Optional[str] = None,
    ):
        self.platform = platform
        self.username = username
        self.nickname = nickname
        self.bio = bio
        self.user_info = user_info
        self.global_event = global_event
        self.user_data = None
        self.user_info_template = user_info_template

        # Memory to store summaries of actions
        self.memory: List[str] = []

        # Initialize LangChain components
        self.llm = llm
        self.tools = self._create_tools()

        # System Prompt
        self.system_prompt = self._get_system_prompt()

        self.agent = create_agent(
            self.llm, tools=self.tools, system_prompt=self.system_prompt
        )

    async def start(self):
        self.user_data = await self.platform.register_or_login(
            self.username, self.nickname, self.bio
        )
        logger.info(f"Agent {self.nickname} started. User ID: {self.user_data.id}")

    def _get_system_prompt(self):
        user_profile = ""
        if self.user_info_template:
            user_profile += self.user_info_template.format(**self.user_info)
        else:
            user_profile += USER_PROFILE_TEMPLATE.format(
                name=self.nickname, user_profile=self.bio
            )
        user_profile += CURRENT_EVENT_TEMPLATE.format(
            global_event_description=self.global_event
        )
        return user_profile

    def _create_tools(self):
        """Create tools that wrap platform actions."""

        @tool
        async def create_post(content: str):
            """Create a new post with the given content."""
            logger.info(f"Agent {self.nickname} is creating a post: {content}")
            response = await self.platform.create_post(content)
            return response.model_dump()

        @tool
        async def create_comment(post_id: int, content: str):
            """Create a comment on a post."""
            logger.info(
                f"Agent {self.nickname} is commenting on post {post_id}: {content}"
            )
            response = await self.platform.create_comment(post_id, content)
            return response.model_dump()

        @tool
        async def like_post(post_id: int):
            """Like a post."""
            logger.info(f"Agent {self.nickname} is liking post {post_id}")
            response = await self.platform.like_post(post_id)
            return response.model_dump()

        @tool
        async def like_comment(comment_id: int):
            """Like a comment."""
            logger.info(f"Agent {self.nickname} is liking comment {comment_id}")
            response = await self.platform.like_comment(comment_id)
            return response.model_dump()

        @tool
        async def repost(post_id: int):
            """Repost a post. This is a pure share action without adding new content."""
            logger.info(f"Agent {self.nickname} is reposting post {post_id}")
            response = await self.platform.repost(post_id)
            return response.model_dump()

        @tool
        async def quote(post_id: int, content: str):
            """Quote a post with content."""
            logger.info(f"Agent {self.nickname} is quoting post {post_id}")
            response = await self.platform.quote(post_id, content)
            return response.model_dump()

        @tool
        async def do_nothing():
            """Do nothing for this turn. Use this when you don't want to take any specific action."""
            logger.info(f"Agent {self.nickname} decided to do nothing")
            response = await self.platform.do_nothing()
            return response.model_dump()

        @tool
        async def get_post(post_id: int):
            """Get details of a post, including comments."""
            logger.info(f"Agent {self.nickname} is looking at post {post_id} in detail")
            post = await self.platform.get_post(post_id)

            author_name = post.author_nickname or f"User {post.user_id}"
            post_str = f"Post ID: {post.id}\n"
            post_str += f"Author: {author_name}\n"
            post_str += f"Time: {post.created_at}\n"
            post_str += f"Content: {post.content or ''}\n"
            post_str += f"Stats: Likes: {post.stats.get('like_count', 0)}, Comments: {post.stats.get('reply_count', 0)}, Shares: {post.stats.get('share_count', 0)}\n"
            post_str += f"My Interaction: Liked: {post.is_liked}, Reposted: {post.is_reposted}\n"

            if post.comments:
                post_str += "Comments:\n"
                for comment in post.comments:
                    comment_author = (
                        comment.author_nickname or f"User {comment.user_id}"
                    )
                    post_str += f"  - {comment_author}: {comment.content} (Likes: {comment.like_count}, Liked by me: {comment.is_liked})\n"
            else:
                post_str += "Comments: None\n"

            return post_str

        return [
            create_post,
            create_comment,
            like_post,
            like_comment,
            repost,
            quote,
            do_nothing,
            get_post,
        ]

    async def get_env_info(self) -> Optional[str]:
        try:
            feed = await self.platform.get_feed()
            # Log seen posts
            post_ids = [p.id for p in feed]
            logger.info(f"Agent {self.nickname} saw posts: {post_ids}")

            time_data = await self.platform.get_time()
            current_time = time_data.get("current_time")

            # Format the feed
            formatted_feed = []
            if not feed:
                formatted_feed.append("No new posts in your feed.")
            else:
                for post in feed:
                    author_name = post.author_nickname or f"User {post.user_id}"
                    post_str = f"Post ID: {post.id}\n"
                    post_str += f"Author: {author_name}\n"
                    post_str += f"Time: {post.created_at}\n"
                    if post.ref_id:
                        post_str += f"{post.type} Post ID: {post.ref_id}\n"
                    post_str += f"Content: {post.content or ''}\n"
                    post_str += f"Stats: Likes: {post.stats.get('like_count', 0)}, Comments: {post.stats.get('reply_count', 0)}, Shares: {post.stats.get('share_count', 0)}\n"
                    post_str += f"My Interaction: Liked: {post.is_liked}, Reposted: {post.is_reposted}\n"
                    formatted_feed.append(post_str)

            feed_str = "\n---\n".join(formatted_feed)

            return f"Current Time: {current_time}\nHere is your feed:\n{feed_str}"

        except Exception as e:
            logger.error(f"Failed to get perception: {e}")
            traceback.print_exc()
            return None

    async def step(self):
        # 1. Perception
        env_info = await self.get_env_info()
        if not env_info:
            return

        # Construct User Message
        user_msg = f"{env_info}\n\nWhat do you want to do next?"

        # Add Long-Term Memory Context
        memory_context = ""
        if self.memory:
            memory_context = (
                "Here is a summary of your recent actions:\n"
                + "\n".join(self.memory)
                + "\n\n"
            )

        full_input = memory_context + user_msg

        logger.info(f"Agent {self.nickname} input: {user_msg}")

        # 2. Decision & Execution
        try:
            # Construct messages
            messages = [
                HumanMessage(content=full_input),
            ]

            result = await self.agent.ainvoke({"messages": messages})

            output = ""
            actions = []

            # Handle result
            if isinstance(result, dict) and "messages" in result:
                final_msgs = result["messages"]
                if final_msgs:
                    output = final_msgs[-1].content

                # Extract actions
                for msg in final_msgs:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            actions.append(
                                f"Called tool {tc['name']} with input {tc['args']}"
                            )
            else:
                # Fallback
                output = str(result.get("output", ""))
                # Try to get intermediate_steps if present
                steps = result.get("intermediate_steps", [])
                for step in steps:
                    if isinstance(step, tuple):
                        actions.append(
                            f"Called tool {step[0].tool} with input {step[0].tool_input}"
                        )

            # 3. Summarization
            await self._summarize_and_store_memory(user_msg, actions, output)

        except Exception as e:
            logger.error(f"Agent step failed: {e}")
            traceback.print_exc()

    async def _summarize_and_store_memory(self, user_input, actions, output):
        """Summarize the interaction and update memory."""

        actions_str = "; ".join(actions) if actions else "No tools called"

        # Create a summary using the LLM
        summary_prompt = f"""
        As the agent itself, summarize your own interaction memory into a single concise first-person sentence that clearly includes: the time of the event, the reason why you took the corresponding actions, and the specific things/actions you did.

        Your Input Context: {user_input}.
        Actions You Took: {actions_str}
        Your Output/Thought: {output}

        Summary:
        """

        try:
            summary_response = await self.llm.ainvoke(summary_prompt)
            summary = summary_response.content.strip()

            logger.info(f"anget {self.nickname} Interaction Summary: {summary}")
            self.memory.append(summary)

            # Prune memory if too long
            if len(self.memory) > 10:
                await self._prune_memory()

        except Exception as e:
            logger.error(f"Failed to summarize: {e}")

    async def _prune_memory(self):
        """Summarize the memory list when it gets too long."""
        old_memory = "\n".join(self.memory)
        prune_prompt = f"""
        The following is a list of your past action memories.
        Please summarize them from your own perspective into a short paragraph (max 3 sentences) to serve as your long-term context.

        Your memories:
        {old_memory}

        Summary:
        """
        try:
            response = await self.llm.ainvoke(prune_prompt)
            condensed_memory = response.content.strip()
            self.memory = [condensed_memory]
            logger.info(
                f"""agent {self.nickname} Memory pruned and summarized.\n{condensed_memory}"""
            )
        except Exception as e:
            logger.error(f"Failed to prune memory: {e}")


if __name__ == "__main__":
    import logging
    import sys
    import asyncio
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    httpx_logger = logging.getLogger("httpx")
    # 设置日志级别为WARNING（只输出WARNING及以上级别的日志）
    # 也可以设置为ERROR，只输出错误日志
    httpx_logger.setLevel(logging.ERROR)

    # Test run
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

            time.sleep(1)  # Loop
    except Exception as e:
        print(f"Agent run failed: {e}")
    except KeyboardInterrupt:
        print("Agent stopped.")
