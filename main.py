import os
import time
import argparse
import asyncio
import multiprocessing
import dotenv
import random
import urllib.request
import urllib.error
import yaml
from datetime import datetime
from pathlib import Path

import numpy as np
import uvicorn

from core.experiment_config import ExperimentConfig, LLMConfig, load_experiment_config
from core.calibration_profile import CalibrationProfile
from core.runtime import (
    RunContext,
    configure_logging,
    create_run_context,
    export_run_context,
    save_config_snapshot,
)
from backend.services.social_recsys import SocialRecSys

dotenv.load_dotenv()


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


def start_backend(
    run_context: RunContext,
    experiment_data: dict,
    recommender_data: dict,
    embedding_data: dict,
):
    """启动后端服务。

    A-1：校准权重与嵌入配置随子进程参数传入。multiprocessing spawn 会重新
    import 模块，主进程里 configure() 设置的类属性不会跨进程传递，
    因此必须在子进程内、create_app() 之前重新注入。
    """
    experiment = ExperimentConfig.model_validate(experiment_data)
    export_run_context(run_context)
    configure_logging(run_context.backend_log_path, "main", console_enabled=False)
    apply_random_seed(experiment.runtime.random_seed)

    # 在 create_app() 之前注入校准参数（关键：必须在子进程内 configure）
    from core.calibration_profile import EmbeddingConfig, RecommenderConfig
    from backend.services.social_recsys import SocialRecSys

    SocialRecSys.configure(
        RecommenderConfig.model_validate(recommender_data),
        EmbeddingConfig.model_validate(embedding_data),
    )

    from backend import create_app

    app = create_app()

    uvicorn.run(
        app,
        host=get_backend_bind_host(),
        port=get_backend_port(),
        reload=False,
        log_config=None,
    )


def _build_llm(
    llm_config: LLMConfig,
    random_seed: int | None,
    label: str,
) -> "ChatOpenAI":
    """根据 LLMConfig 构建 ChatOpenAI 实例（core 大模型 / mass 小模型共用）。

    Part C 可复现性：temperature=0 固定采样；端点支持时附加固定 seed。
    """
    from langchain_openai import ChatOpenAI

    llm_kwargs: dict = {
        "model": llm_config.model,
        "api_key": os.environ.get(llm_config.api_key_env),
        "base_url": os.environ.get(llm_config.base_url_env),
        "timeout": llm_config.timeout,
        "temperature": 0.0,
    }
    if random_seed is not None:
        llm_kwargs["seed"] = random_seed
    try:
        llm = ChatOpenAI(**llm_kwargs)
    except Exception as exc:  # 端点不支持 seed 时降级为只固定 temperature
        logger.warning(
            f"ChatOpenAI({label}) 不支持 seed 参数，已降级（temperature=0 仍生效）：{exc}"
        )
        llm_kwargs.pop("seed", None)
        llm = ChatOpenAI(**llm_kwargs)
    logger.info(f"已构建 {label} 模型：{llm_config.model}")
    return llm


def start_agent(run_context: RunContext, experiment_data: dict, config_path: str):
    """启动 Agent 服务"""
    experiment = ExperimentConfig.model_validate(experiment_data)
    from moss_agent_client.remote_platform import RemotePlatform
    from moss_agent_client.schemas import SystemTimeConfig
    from moss_agent_client.agent_graph import AgentGraph
    from core.agent_profile_resolver import resolve_agent_payloads

    export_run_context(run_context)
    configure_logging(run_context.agent_log_path, "moss.agent", console_enabled=True)
    apply_random_seed(experiment.runtime.random_seed)

    platform = RemotePlatform(get_backend_base_url())

    # 大小模型分层：core 大模型始终构建；mass 小模型仅在配置 llm_small 时构建。
    llm = _build_llm(experiment.llm, experiment.runtime.random_seed, "core")
    llm_small = None
    if experiment.llm_small is not None:
        llm_small = _build_llm(experiment.llm_small, experiment.runtime.random_seed, "mass")

    system_time_config: SystemTimeConfig = SystemTimeConfig(
        mode=experiment.system_time.mode,
        start_time=datetime.fromisoformat(experiment.system_time.start_time),
        time_scale=experiment.system_time.time_scale,
    )
    async def _run() -> None:
        graph = AgentGraph(
            platform,
            global_event=experiment.global_event,
            llm=llm,
            llm_small=llm_small,
            system_time_config=system_time_config,
            memory_config=experiment.memory,
        )
        agent_payloads = await resolve_agent_payloads(
            experiment=experiment,
            config_path=config_path,
        )
        graph.load_from_config(agent_payloads)
        await graph.run_loop(
            round=experiment.runtime.rounds,
            interval=experiment.runtime.interval_seconds,
        )

    asyncio.run(_run())


def load_calibration_profile(path: str | Path) -> CalibrationProfile:
    """加载统一校准+模拟配置文件（YAML）。"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return CalibrationProfile.model_validate(data)


def parse_args():
    parser = argparse.ArgumentParser(description="启动 MOSS 推演实验")
    parser.add_argument(
        "--config",
        required=True,
        help="校准配置文件路径（calibration_profile.yaml）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载 calibration_profile.yaml（替代原 experiment.json）
    if not Path(args.config).exists():
        raise SystemExit(f"配置文件不存在：{args.config}")

    calibration = load_calibration_profile(args.config)

    # 从 YAML experiment 段提取实验配置
    experiment = calibration.to_experiment_config()

    # 注入推荐服务参数（替代硬编码常量）
    SocialRecSys.configure(calibration.recommender, calibration.embedding)

    run_context = create_run_context()
    export_run_context(run_context)
    configure_logging(run_context.backend_log_path, "main", console_enabled=True)
    save_config_snapshot(Path(args.config).resolve(), run_context.config_snapshot_path)

    logger.info(f"本次运行目录: {run_context.run_dir}")
    logger.info(f"实验配置快照: {run_context.config_snapshot_path}")
    logger.info(f"实验名称: {experiment.name}")

    backend_process = multiprocessing.Process(
        target=start_backend,
        args=(
            run_context,
            experiment.model_dump(),
            calibration.recommender.model_dump(),
            calibration.embedding.model_dump(),
        ),
        name="Backend",
    )
    agent_process = multiprocessing.Process(
        target=start_agent,
        args=(run_context, experiment.model_dump(), str(Path(args.config).resolve())),
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
