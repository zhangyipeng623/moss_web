"""CalibrationProfile: 统一校准+模拟配置的 Pydantic 模型。

YAML 文件由 run_analysis recommender 命令自动生成，main.py 启动时读取。
包含 5 个段：meta / experiment / recommender / embedding / simulation
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from core.experiment_config import (
    ExperimentConfig,
    LLMConfig,
    MemoryExperimentConfig,
    PortraitConfig,
    RuntimeExperimentConfig,
    SystemTimeExperimentConfig,
)


class MetaInfo(BaseModel):
    """YAML meta 段 —— 自动生成，记录校准运行元信息。"""
    generated_at: str = ""
    generator_version: str = "v8-vectorized-abm-optuna"
    portraits_dir: str = ""
    num_seed_users: int = 0
    abm_population_size: int = 500
    input_data_file: str = ""


class ExperimentYamlConfig(BaseModel):
    """YAML experiment 段 —— 映射原 experiment.json 全部字段。"""
    name: str = "默认实验"
    description: str = ""
    global_event: str = ""
    llm: LLMConfig = Field(default_factory=lambda: LLMConfig(model="gpt-4o"))
    llm_small: Optional[LLMConfig] = None
    agents_csv: str = "configs/experiments/default_agents.csv"
    portrait: PortraitConfig = Field(default_factory=PortraitConfig)
    system_time: SystemTimeExperimentConfig = Field(
        # D-1 时间基准：每步 1 小时（time_scale=3600 秒/步）
        default_factory=lambda: SystemTimeExperimentConfig(
            start_time="2026-05-17T12:00:00", time_scale=3600
        )
    )
    runtime: RuntimeExperimentConfig = Field(
        # D-1 时间基准：24 步 × 1 小时 = 总跨度 24 小时
        default_factory=lambda: RuntimeExperimentConfig(rounds=24)
    )


class RecommenderWeights(BaseModel):
    """推荐系统权重 —— ABM+EM 校准结果，自动填写。"""
    w_interest: float = Field(default=0.35, ge=0.0, le=1.0)
    w_popularity: float = Field(default=0.25, ge=0.0, le=1.0)
    w_time: float = Field(default=0.25, ge=0.0, le=1.0)
    w_random: float = Field(default=0.15, ge=0.0, le=1.0)


class RecommenderConfig(BaseModel):
    """YAML recommender 段 —— ABM+EM 校准结果，自动填写。"""
    weights: RecommenderWeights = Field(default_factory=RecommenderWeights)
    decay_lambda: float = Field(default=0.5, gt=0.0)
    tier_weight: Dict[int, float] = Field(
        # A-2：与 core.scoring.TIER_WEIGHT_DEFAULT 一致（除以 max 归一化到 (0,1]）
        default_factory=lambda: {1: 0.2, 2: 0.35, 3: 0.5, 4: 0.75, 5: 1.0}
    )
    calibrated_p_base: Dict[str, Any] = Field(default_factory=dict)
    fit_diagnostics: Dict[str, Any] = Field(default_factory=dict)


class EmbeddingConfig(BaseModel):
    """YAML embedding 段 —— ABM 和 Backend 共用，自动填写。"""
    model_name: str = "BAAI/bge-m3"
    normalize_embeddings: bool = True


class SimulationDefaults(BaseModel):
    """YAML simulation 段 —— 模拟引擎默认参数。"""
    p_online: Dict[int, float] = Field(
        default_factory=lambda: {1: 0.003, 2: 0.01, 3: 0.04, 4: 0.10, 5: 0.20}
    )
    belief_update: Dict[str, float] = Field(
        default_factory=lambda: {"backfire_mu": 0.4, "backfire_k": 10.0, "learning_rate": 0.1}
    )
    content_filter: Dict[str, float] = Field(
        default_factory=lambda: {"min_scaled_target": 5, "anchor_percentile": 0.8}
    )
    feed: Dict[str, int] = Field(
        default_factory=lambda: {"candidate_limit": 100, "feed_limit": 5}
    )
    memory: Dict[str, Any] = Field(
        default_factory=lambda: {
            "short_term_max_rounds": 3,
            "short_term_max_posts": 3,
            "event_max_size": 50,
            "step_retry_limit": 3,
            # 记忆检索评分（docs/plan/记忆系统优化方案.md）：
            "event_decay_lambda": 0.07,  # 半衰期 ~10 轮
            "context_boost_cap": 0.3,   # 上下文联想加成上限
        }
    )


class CalibrationProfile(BaseModel):
    """统一校准+模拟配置文件（YAML）的根模型。

    由 run_analysis recommender 自动生成，main.py 启动时通过 --config 加载。
    """
    meta: MetaInfo = Field(default_factory=MetaInfo)
    experiment: ExperimentYamlConfig = Field(default_factory=ExperimentYamlConfig)
    recommender: RecommenderConfig = Field(default_factory=RecommenderConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    simulation: SimulationDefaults = Field(default_factory=SimulationDefaults)

    def to_experiment_config(self) -> ExperimentConfig:
        """将 YAML 的 experiment 段转换为 ExperimentConfig（替代原 experiment.json 加载）。"""
        exp = self.experiment
        return ExperimentConfig(
            name=exp.name,
            description=exp.description,
            global_event=exp.global_event,
            llm=exp.llm,
            llm_small=exp.llm_small,
            system_time=exp.system_time,
            runtime=exp.runtime,
            portrait=exp.portrait,
            agents_csv=exp.agents_csv,
            memory=MemoryExperimentConfig(**self.simulation.memory),
        )
