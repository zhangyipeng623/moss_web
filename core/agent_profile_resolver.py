import csv
import json
import logging
from pathlib import Path
from typing import Any

from core.experiment_config import AgentConfig, ExperimentConfig


logger = logging.getLogger(__name__)


async def resolve_agent_payloads(
    experiment: ExperimentConfig,
    config_path: str | Path,
) -> list[dict[str, Any]]:
    """解析实验中的 Agent 配置，统一生成可注入 AgentGraph 的负载。"""
    config_file = Path(config_path).resolve()
    base_dir = config_file.parent
    agent_configs = list(experiment.agents)
    agent_configs.extend(
        _load_agents_from_csv(
            base_dir=base_dir,
            csv_path=experiment.agents_csv,
            encoding=experiment.portrait.csv_encoding,
        )
    )

    resolved_payloads: list[dict[str, Any]] = []
    for agent in agent_configs:
        user_info = _resolve_user_info(
            agent=agent,
            base_dir=base_dir,
        )
        user_info = _augment_user_info(
            user_info=user_info,
            agent=agent,
        )
        user_info_template = _resolve_user_info_template(
            agent=agent,
            experiment=experiment,
            user_info=user_info,
            base_dir=base_dir,
        )
        resolved_payloads.append(
            {
                "username": agent.username,
                "name": agent.name,
                "bio": agent.bio,
                "user_info": user_info or {},
                "user_info_template": user_info_template,
            }
        )
    return resolved_payloads


def _load_agents_from_csv(
    base_dir: Path,
    csv_path: str | None,
    encoding: str,
) -> list[AgentConfig]:
    """从 CSV 读取 Agent 配置。"""
    if not csv_path:
        return []

    resolved_path = _resolve_optional_path(base_dir, csv_path)
    if resolved_path is None or not resolved_path.exists():
        raise FileNotFoundError(f"找不到 agents_csv 文件：{csv_path}")
    csv_base_dir = resolved_path.parent

    configs: list[AgentConfig] = []
    with resolved_path.open("r", encoding=encoding, newline="") as file:
        reader = csv.DictReader(file)
        for row_index, row in enumerate(reader, start=2):
            normalized_row = {
                key: value.strip() if isinstance(value, str) else value
                for key, value in row.items()
                if key
            }
            normalized_row = _normalize_csv_paths(normalized_row, csv_base_dir)
            if not any(str(value or "").strip() for value in normalized_row.values()):
                continue
            try:
                configs.append(AgentConfig.model_validate(normalized_row))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(
                    f"CSV 第 {row_index} 行 Agent 配置校验失败：{exc}"
                ) from exc
    return configs


def _resolve_user_info(
    agent: AgentConfig,
    base_dir: Path,
) -> dict[str, Any] | None:
    """根据模式解析 user_info。"""
    if agent.profile_mode == "default":
        user_info = _resolve_inline_or_file_profile(agent, base_dir)
        if user_info is None:
            raise ValueError(
                f"Agent {agent.username} 配置为 default，但未提供结构化画像。"
            )
        return user_info

    if agent.profile_mode != "custom":
        raise ValueError(f"不支持的 profile_mode：{agent.profile_mode}")
    return _resolve_inline_or_file_profile(agent, base_dir) or {}


def _resolve_inline_or_file_profile(
    agent: AgentConfig,
    base_dir: Path,
) -> dict[str, Any] | None:
    """解析内联或文件画像。"""
    if agent.user_info:
        return dict(agent.user_info)

    if agent.user_info_json:
        try:
            payload = json.loads(agent.user_info_json)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Agent {agent.username} 的 user_info_json 不是合法 JSON。"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(
                f"Agent {agent.username} 的 user_info_json 必须是 JSON 对象。"
            )
        return payload

    if agent.profile_text:
        return {"profile_text": agent.profile_text}

    profile_path = _resolve_optional_path(base_dir, agent.profile_path)
    if profile_path:
        if not profile_path.exists():
            raise FileNotFoundError(f"找不到画像文件：{profile_path}")
        return _load_json_object(profile_path)
    return None


def _resolve_user_info_template(
    agent: AgentConfig,
    experiment: ExperimentConfig,
    user_info: dict[str, Any] | None,
    base_dir: Path,
) -> str | None:
    """解析最终注入 Agent 的模板。"""
    if agent.profile_mode == "default":
        return None

    if agent.user_info_template:
        return agent.user_info_template

    template_path = _resolve_optional_path(base_dir, agent.user_info_template_path)
    if template_path:
        if not template_path.exists():
            raise FileNotFoundError(f"找不到用户模板文件：{template_path}")
        return template_path.read_text(encoding="utf-8")

    raise ValueError(
        f"Agent {agent.username} 配置为 custom，但未提供 user_info_template 或 user_info_template_path。"
    )


def _augment_user_info(
    user_info: dict[str, Any] | None,
    agent: AgentConfig,
) -> dict[str, Any]:
    """补齐模板注入所需的基础字段。"""
    payload = dict(user_info or {})
    payload.setdefault("username", agent.username)
    payload.setdefault("name", agent.name)
    payload.setdefault("nickname", agent.name)
    payload.setdefault("bio", agent.bio)
    return payload


def _load_json_object(path: Path) -> dict[str, Any]:
    """读取 JSON 对象。"""
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"文件 {path} 必须是 JSON 对象。")
    return payload


def _resolve_optional_path(base_dir: Path, path_value: str | None) -> Path | None:
    """解析相对路径。"""
    if not path_value:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _normalize_csv_paths(
    payload: dict[str, Any],
    csv_base_dir: Path,
) -> dict[str, Any]:
    """把 CSV 行内的相对路径转成相对 CSV 文件自身的绝对路径。"""
    normalized = dict(payload)
    for field_name in ("profile_path", "user_info_template_path"):
        value = normalized.get(field_name)
        if not isinstance(value, str) or not value.strip():
            continue
        path = Path(value.strip())
        if path.is_absolute():
            normalized[field_name] = str(path)
            continue
        normalized[field_name] = str((csv_base_dir / path).resolve())
    return normalized
