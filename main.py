import os
import time
import argparse
import asyncio
import multiprocessing
import dotenv
import random
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

import numpy as np
import uvicorn

from core.experiment_config import ExperimentConfig, load_experiment_config
from core.runtime import (
    RunContext,
    configure_logging,
    create_run_context,
    export_run_context,
    save_config_snapshot,
)

dotenv.load_dotenv()


DEFAULT_EXPERIMENT_CONFIG = (
    Path(__file__).resolve().parent / "configs" / "experiments" / "default.json"
)
logger = configure_logging(
    Path(os.environ.get("MOSS_BOOTSTRAP_LOG", "/tmp/moss_bootstrap.log")),
    "main",
    console_enabled=True,
)


def get_backend_bind_host() -> str:
    """获取后端监听地址（用于 uvicorn bind）。"""
    host = os.environ.get("BACKEND_URL", "127.0.0.1")
    return host


def get_backend_port() -> int:
    """获取后端端口。"""
    return int(os.environ.get("BACKEND_PORT", "8000"))


def get_backend_base_url() -> str:
    """获取客户端访问后端的基础 URL。"""
    explicit_base_url = os.environ.get("BACKEND_BASE_URL")
    if explicit_base_url:
        return explicit_base_url.rstrip("/")

    host = get_backend_bind_host()
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    return f"http://{host}:{get_backend_port()}"


def wait_for_backend_ready(timeout: int = 30, interval: float = 1.0) -> bool:
    """等待后端健康检查成功。"""
    health_url = f"{get_backend_base_url()}/api/v1/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    logger.info(f"后端健康检查通过: {health_url}")
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            pass

        time.sleep(interval)

    logger.error(f"后端在 {timeout}s 内未就绪: {health_url}")
    return False



def apply_random_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def start_backend(run_context: RunContext, experiment_data: dict):
    """启动后端服务"""
    experiment = ExperimentConfig.model_validate(experiment_data)
    export_run_context(run_context)
    configure_logging(run_context.backend_log_path, "main", console_enabled=False)
    apply_random_seed(experiment.runtime.random_seed)

    from backend import create_app

    app = create_app()

    uvicorn.run(
        app,
        host=get_backend_bind_host(),
        port=get_backend_port(),
        reload=False,
        log_config=None,
    )


def start_agent(run_context: RunContext, experiment_data: dict):
    """启动 Agent 服务"""
    experiment = ExperimentConfig.model_validate(experiment_data)
    from moss_agent_client.remote_platform import RemotePlatform
    from langchain_openai import ChatOpenAI
    from moss_agent_client.schemas import SystemTimeConfig
    from moss_agent_client.agent_graph import AgentGraph

    export_run_context(run_context)
    configure_logging(run_context.agent_log_path, "moss.agent", console_enabled=True)
    apply_random_seed(experiment.runtime.random_seed)

    platform = RemotePlatform(get_backend_base_url())

    llm = ChatOpenAI(
        model=experiment.llm.model,
        api_key=os.environ.get(experiment.llm.api_key_env),
        base_url=os.environ.get(experiment.llm.base_url_env),
        timeout=experiment.llm.timeout,
    )

    system_time_config: SystemTimeConfig = SystemTimeConfig(
        mode=experiment.system_time.mode,
        start_time=datetime.fromisoformat(experiment.system_time.start_time),
        time_scale=experiment.system_time.time_scale,
    )
    graph = AgentGraph(
        platform,
        global_event=experiment.global_event,
        llm=llm,
        system_time_config=system_time_config,
    )

    graph.load_from_config(
        [agent.model_dump() for agent in experiment.agents]
    )
    asyncio.run(
        graph.run_loop(
            round=experiment.runtime.rounds,
            interval=experiment.runtime.interval_seconds,
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description="启动 MOSS 推演实验")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_EXPERIMENT_CONFIG),
        help="实验配置文件路径",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    experiment = load_experiment_config(args.config)
    run_context = create_run_context()
    export_run_context(run_context)
    configure_logging(run_context.backend_log_path, "main", console_enabled=True)
    save_config_snapshot(Path(args.config).resolve(), run_context.config_snapshot_path)

    logger.info(f"本次运行目录: {run_context.run_dir}")
    logger.info(f"实验配置快照: {run_context.config_snapshot_path}")
    logger.info(f"实验名称: {experiment.name}")

    backend_process = multiprocessing.Process(
        target=start_backend,
        args=(run_context, experiment.model_dump()),
        name="Backend",
    )
    agent_process = multiprocessing.Process(
        target=start_agent,
        args=(run_context, experiment.model_dump()),
        name="Agent",
    )

    backend_process.start()

    if not wait_for_backend_ready(timeout=30, interval=1):
        backend_process.terminate()
        backend_process.join()
        raise RuntimeError("后端启动失败，已停止 Agent 启动流程")

    agent_process.start()

    try:
        backend_process.join()
        agent_process.join()
    except KeyboardInterrupt:
        logger.info("Stopping services...")
        backend_process.terminate()
        agent_process.terminate()
        backend_process.join()
        agent_process.join()
        logger.info("Services stopped.")


if __name__ == "__main__":
    main()
