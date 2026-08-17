import asyncio
import time
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI

from core.experiment_config import MemoryExperimentConfig
from moss_agent_client.agent import MossAgent
from moss_agent_client.agent_logger import logger
from moss_agent_client.remote_platform import RemotePlatform
from moss_agent_client.schemas import ActionResponse, SystemTimeConfig


class AgentGraph:
    def __init__(
        self,
        platform: RemotePlatform,
        global_event: str,
        system_time_config: SystemTimeConfig,
        llm: Optional[ChatOpenAI] = None,
        memory_config: Optional[MemoryExperimentConfig] = None,
    ):
        """
        Initialize the AgentGraph.

        :param platform: The RemotePlatform instance used for communication.
        :param global_event: The global event context for the agents.
        :param llm: The Language Model instance to be used by agents.
        :param memory_config: 记忆系统参数（来自 YAML simulation.memory 段）。
        """
        self.platform = platform
        self.global_event = global_event
        self.llm = llm
        self._agents: List[MossAgent] = []
        self.system_time_config = system_time_config
        self.memory_config = memory_config

    def add_agent(
        self,
        username: str,
        name: str,
        bio: str,
        user_info: Optional[Dict[str, Any]] = None,
        user_info_template: Optional[str] = None,
        profile_mode: str = "default",
    ) -> MossAgent:
        """
        Create and add a new agent to the graph.

        :param username: The username of the agent.
        :param name: The display name of the agent.
        :param bio: The biography of the agent.
        :param user_info: Optional user information dictionary.
        :param user_info_template: Optional template for user profile generation.
        :param profile_mode: default/custom/simple，决定画像模板分支（A-3）。
        :return: The created MossAgent instance.
        """
        if self.llm is None:
            # For fallback/testing if not provided, though ideally should be passed
            # Assuming environment variables are set for API keys if this path is taken
            from langchain_openai import ChatOpenAI

            self.llm = ChatOpenAI(model="gpt-4o")

        # Create a new RemotePlatform instance for each agent to ensure thread safety and state isolation
        # Note: With httpx.AsyncClient, we can share the client, but here we keep the pattern
        # of creating a wrapper, but reuse the underlying client.
        agent_platform = RemotePlatform(
            self.platform.base_url, client=self.platform.client
        )
        agent = MossAgent(
            platform=agent_platform,
            username=username,
            nickname=name,
            bio=bio,
            global_event=self.global_event,
            llm=self.llm,
            user_info=user_info,
            user_info_template=user_info_template,
            profile_mode=profile_mode,
            memory_config=self.memory_config,
        )
        self._agents.append(agent)
        logger.info(f"Added agent: {name} ({username})")
        return agent

    async def execute_agent_action(
        self, agent: MossAgent, action_type: str, params: Dict[str, Any]
    ) -> Optional[ActionResponse]:
        """
        Execute a specific action for a given agent asynchronously.

        :param agent: The MossAgent instance to execute the action.
        :param action_type: The type of action to execute (e.g., 'create_post', 'like_post').
        :param params: The parameters for the action.
        :return: The result of the action execution or None if failed.
        """
        logger.info(f"Executing action '{action_type}' for agent {agent.nickname}")
        try:
            result = await agent.execute_action(action_type, params)
            if hasattr(result, "model_dump"):
                return result
            return None
        except ValueError:
            logger.warning(f"Unknown action type: {action_type}")
            return None
        except Exception as e:
            logger.error(
                f"Failed to execute action '{action_type}' for agent {agent.nickname}: {e}"
            )
            return None

    async def execute_agent_actions(self, actions: List[Dict[str, Any]]):
        """
        Execute a list of actions concurrently for different agents using asyncio.

        Each item in the actions list should be a dictionary with:
        - 'agent': The MossAgent instance
        - 'action_type': The action string
        - 'params': The parameters dictionary

        :param actions: List of action dictionaries.
        """
        logger.info(f"Executing {len(actions)} concurrent actions...")

        tasks = []
        for item in actions:
            agent = item.get("agent")
            action_type = item.get("action_type")
            params = item.get("params", {})

            if agent and action_type:
                tasks.append(self.execute_agent_action(agent, action_type, params))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error during concurrent action execution: {result}")

        logger.info("Concurrent actions execution completed.")

    def load_from_config(self, configs: List[Dict[str, Any]]):
        """
        Load multiple agents from a list of configuration dictionaries.

        :param configs: A list of dictionaries containing agent configuration.
        """
        for config in configs:
            self.add_agent(**config)

    async def start_all(self):
        """
        Start (register/login) all agents in the graph concurrently.
        """
        logger.info("Starting all agents...")
        if not self._agents:
            logger.info("No agents to start.")
            return

        tasks = [agent.start() for agent in self._agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        started_agents: List[MossAgent] = []
        failed_agent_names: List[str] = []

        for i, result in enumerate(results):
            agent = self._agents[i]
            if isinstance(result, Exception):
                logger.error(
                    f"Error starting agent {agent.nickname}: {result}"
                )
                failed_agent_names.append(agent.nickname)
                continue

            if agent.user_data is None:
                logger.error(f"Agent {agent.nickname} 启动后仍未完成登录，已跳过。")
                failed_agent_names.append(agent.nickname)
                continue

            started_agents.append(agent)

        self._agents = started_agents

        if failed_agent_names:
            failed_names = "、".join(failed_agent_names)
            raise RuntimeError(
                f"以下 agent 启动失败或未完成登录：{failed_names}。"
                "为避免实验结果失真，已终止本次运行。"
            )

        logger.info("All agents started.")

    async def step_all(self):
        """
        Execute one step for all agents concurrently.
        """
        logger.info("Executing step for all agents...")
        if not self._agents:
            return

        tasks = [agent.step() for agent in self._agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        failures = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failures.append((self._agents[i].nickname, result))
                logger.error(
                    f"Error during agent {self._agents[i].nickname} step: {result}"
                )

        if failures:
            failure_messages = [
                f"{nickname}: {error}" for nickname, error in failures
            ]
            raise RuntimeError(
                "以下 agent 在本轮执行失败，实验已中止以避免重复执行副作用动作："
                + "；".join(failure_messages)
            ) from failures[0][1]

        logger.info("Step execution completed.")

    async def run_loop(
        self,
        round: int = 3,
        interval: float = 10.0,
    ):
        """
        Run the agent loop continuously.

        :param interval: The time in seconds to wait between steps.
        """
        logger.info(f"Starting agent loop with interval {interval}s")
        try:
            # Request system config
            try:
                await self.platform.set_system_config(
                    **self.system_time_config.model_dump()
                )
                logger.info(f"System config loaded: {self.system_time_config}")
            except Exception as e:
                logger.warning(f"Failed to load system config: {e}")

            await self.start_all()
            if not self._agents:
                logger.warning("没有可运行的 agent，终止执行循环。")
                return

            for _ in range(round):
                start_time = time.time()
                await self.step_all()

                # If step mode, increment step
                if self.system_time_config and self.system_time_config.mode == "step":
                    try:
                        await self.platform.increment_step()
                        logger.info("Step incremented successfully")
                    except Exception as e:
                        logger.error(f"Failed to increment step: {e}")

                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
        except asyncio.CancelledError:
            logger.info("Agent loop stopped by user.")
        except Exception as e:
            logger.error(f"Agent loop failed: {e}")
            raise
        finally:
            await self.platform.close()


async def main():
    # Setup logging configuration if running directly
    import logging
    import sys
    import datetime

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    httpx_logger = logging.getLogger("httpx")
    # 设置日志级别为WARNING（只输出WARNING及以上级别的日志）
    # 也可以设置为ERROR，只输出错误日志
    httpx_logger.setLevel(logging.ERROR)

    base_url = "http://localhost:8000"
    platform = RemotePlatform(base_url)

    # Need to initialize LLM
    # Attempt to use environment variable or default

    llm = ChatOpenAI(
        model="deepseek/deepseek-v3.2",
        api_key="",
        base_url="https://openrouter.ai/api/v1",
        timeout=180,
    )

    system_time_config: SystemTimeConfig = SystemTimeConfig(
        mode="step", start_time=datetime.datetime.now(), time_scale=600
    )
    graph = AgentGraph(
        platform,
        # 全球事件：2026年3月3日美伊冲突最新升级（真实事件）
        global_event="北京时间2026年3月3日，美伊冲突持续升级：伊朗伊斯兰革命卫队凌晨对美国驻沙特大使馆发动无人机袭击（造成建筑起火受损），并向巴林谢赫伊萨空军基地发射20架无人机与3枚弹道导弹（击中主指挥部大楼）；美军则对伊朗展开反击，摧毁其革命卫队指挥控制设施、防空系统及导弹/无人机发射阵地，双方军事对抗进入白热化阶段。",
        llm=llm,
        system_time_config=system_time_config,
    )

    # Agent 1：伊朗军事行动分析专家（Alpha）
    graph.add_agent(
        username="agent_iran_military_analyst",
        name="Alpha",
        bio="我是资深军事分析专家，专攻伊朗伊斯兰革命卫队（IRGC）作战行动，拥有8年追踪伊朗无人机与导弹作战能力的经验，擅长分析伊朗针对中东地区美军目标发起的“真实承诺-4”报复性军事行动。使用简体中文语言进行分析和沟通。",
    )

    # Agent 2：美国中东战略评估顾问（Beta）
    graph.add_agent(
        username="agent_us_mideast_strategy_advisor",
        name="Beta",
        bio="我是美国中东安全战略顾问，聚焦美国中央司令部（CENTCOM）对伊朗的反击作战行动，精通评估美伊冲突升级对地区稳定及全球能源安全产生的政治、军事影响。使用简体中文语言进行分析和沟通。",
    )

    await graph.run_loop(interval=3)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
