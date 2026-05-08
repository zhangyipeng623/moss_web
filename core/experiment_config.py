import json
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class LLMConfig(BaseModel):
    model: str
    timeout: int = 180
    api_key_env: str = "API_KEY"
    base_url_env: str = "BASE_URL"


class AgentConfig(BaseModel):
    username: str
    name: str
    bio: str
    profile_mode: Literal["default", "custom"] = "default"
    user_info: Optional[dict[str, Any]] = None
    user_info_json: Optional[str] = None
    user_info_template: Optional[str] = None
    user_info_template_path: Optional[str] = None
    profile_path: Optional[str] = None
    profile_text: Optional[str] = None


class PortraitConfig(BaseModel):
    topic_candidates: list[str] = Field(default_factory=list)
    csv_encoding: str = "utf-8-sig"


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
    runtime: RuntimeExperimentConfig = Field(default_factory=RuntimeExperimentConfig)
    portrait: PortraitConfig = Field(default_factory=PortraitConfig)
    agents: list[AgentConfig] = Field(default_factory=list)
    agents_csv: Optional[str] = None

    @model_validator(mode="after")
    def validate_agent_sources(self) -> "ExperimentConfig":
        if not self.agents and not self.agents_csv:
            raise ValueError("实验配置必须至少提供 agents 或 agents_csv 之一。")
        return self


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return ExperimentConfig.model_validate(data)
