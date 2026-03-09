import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    model: str
    timeout: int = 180
    api_key_env: str = "API_KEY"
    base_url_env: str = "BASE_URL"


class AgentConfig(BaseModel):
    username: str
    name: str
    bio: str
    user_info: Optional[dict[str, Any]] = None
    user_info_template: Optional[str] = None


class SystemTimeExperimentConfig(BaseModel):
    mode: str = "step"
    start_time: str
    time_scale: float = 600


class RuntimeExperimentConfig(BaseModel):
    rounds: int = Field(default=3, ge=1)
    interval_seconds: float = Field(default=3.0, gt=0)
    random_seed: Optional[int] = None


class ExperimentConfig(BaseModel):
    name: str
    description: str = ""
    global_event: str
    llm: LLMConfig
    system_time: SystemTimeExperimentConfig
    runtime: RuntimeExperimentConfig = RuntimeExperimentConfig()
    agents: list[AgentConfig]


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return ExperimentConfig.model_validate(data)
