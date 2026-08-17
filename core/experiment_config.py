import json
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator, field_validator


class LLMConfig(BaseModel):
    model: str
    timeout: int = 180
    api_key_env: str = "API_KEY"
    base_url_env: str = "BASE_URL"


class AgentConfig(BaseModel):
    username: str
    name: str
    bio: str
    profile_mode: Literal["default", "custom", "simple"] = "default"
    tier: Optional[int] = None
    user_info: Optional[dict[str, Any]] = None
    user_info_json: Optional[str] = None
    user_info_template: Optional[str] = None
    user_info_template_path: Optional[str] = None
    profile_path: Optional[str] = None
    profile_text: Optional[str] = None

    @field_validator("tier")
    @classmethod
    def validate_tier(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and not (1 <= v <= 5):
            raise ValueError(f"tier must be 1-5, got {v}")
        return v


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


class MemoryExperimentConfig(BaseModel):
    """Agent 记忆系统参数（来自 YAML simulation.memory 段）。

    event_decay_lambda / context_boost_cap 为检索评分参数，
    见 docs/plan/记忆系统优化方案.md。
    """
    short_term_max_rounds: int = Field(default=3, ge=1)
    short_term_max_posts: int = Field(default=3, ge=1)
    event_max_size: int = Field(default=50, ge=1)
    step_retry_limit: int = Field(default=3, ge=1)
    event_decay_lambda: float = Field(default=0.07, ge=0.0)
    context_boost_cap: float = Field(default=0.3, ge=0.0)


class ExperimentConfig(BaseModel):
    name: str
    description: str = ""
    global_event: str
    llm: LLMConfig
    system_time: SystemTimeExperimentConfig
    runtime: RuntimeExperimentConfig = Field(default_factory=RuntimeExperimentConfig)
    portrait: PortraitConfig = Field(default_factory=PortraitConfig)
    memory: MemoryExperimentConfig = Field(default_factory=MemoryExperimentConfig)
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
