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

    if not agent_configs:
        raise ValueError(
            "实验未配置任何 Agent：请提供画像目录（--portraits-dir）、"
            "或配置 agents_csv，或启用 simulation.l1_l3_pool。"
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
                # A-3：透传 profile_mode，simple 用户走 SIMPLE_USER_TEMPLATE 分支
                "profile_mode": agent.profile_mode,
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
    if agent.profile_mode == "simple":
        return {"bio": agent.bio, "tier": agent.tier or 3}

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
    if agent.profile_mode == "simple":
        return None  # Uses built-in SIMPLE_USER_TEMPLATE in agent.py

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


def build_agents_from_portraits_dir(portraits_dir: str | Path) -> list[AgentConfig]:
    """从画像目录批量构建 AgentConfig（供 main.py --portraits-dir 使用）。

    目录下每个画像 JSON 映射为一个 default 模式的 Agent：
    username=user_id，name=昵称，bio=简介，profile_path=画像文件绝对路径，
    tier=influence_tier。非画像 JSON（如 failed_users.json、summary.json）自动跳过。
    """
    directory = Path(portraits_dir).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"画像目录不存在：{directory}")

    agents: list[AgentConfig] = []
    for json_path in sorted(directory.glob("*.json")):
        data = _load_json_object(json_path)
        if not isinstance(data.get("stable_profile"), dict):
            # 跳过非画像 JSON（failed_users.json、summary.json 等）
            continue
        stats = data.get("stats") or {}
        user_id = data.get("user_id") or stats.get("username") or json_path.stem
        nickname = stats.get("nickname") or user_id
        bio = stats.get("description") or ""
        tier = data.get("influence_tier")
        if not isinstance(tier, int):
            tier = 4
        agents.append(
            AgentConfig(
                username=str(user_id),
                name=str(nickname),
                bio=str(bio),
                profile_mode="default",
                profile_path=str(json_path),
                tier=tier,
            )
        )

    if not agents:
        raise ValueError(f"画像目录中没有可用的画像 JSON：{directory}")
    return agents


def build_l1_l3_agents_from_pool(pool_config: Any, n_l45: int) -> list[AgentConfig]:
    """从候选池按 Rogers 比例随机抽取 L1-L3 simple 用户。

    pool_config：L1L3PoolConfig（simulation.l1_l3_pool）；
    n_l45：当前 L4+L5 画像数量，作为比例锚点。

    候选池 CSV 需含 username/name/bio/followers（可选 verified）列。
    """
    if not getattr(pool_config, "enabled", False) or not getattr(pool_config, "csv_path", ""):
        return []

    import random

    import pandas as pd

    csv_path = Path(pool_config.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"L1-L3 候选池文件不存在：{csv_path}")

    # Rogers 比例：以 L4+L5（ratio_l45）为锚反推各层数量
    scale = n_l45 / max(float(pool_config.ratio_l45), 1e-9)
    counts = {
        1: max(0, round(scale * float(pool_config.ratio_l1))),
        2: max(0, round(scale * float(pool_config.ratio_l2))),
        3: max(0, round(scale * float(pool_config.ratio_l3))),
    }

    usecols = ["username", "name", "bio", "followers"]
    if pool_config.exclude_verified:
        usecols.append("verified")
    df = pd.read_csv(csv_path, usecols=usecols)
    df["followers"] = pd.to_numeric(df["followers"], errors="coerce").fillna(0)
    df = df[df["bio"].notna() & (df["bio"].astype(str).str.strip() != "")]
    if pool_config.exclude_verified and "verified" in df.columns:
        df = df[~df["verified"].astype(bool)]

    bins = {
        1: tuple(pool_config.l1_followers),
        2: tuple(pool_config.l2_followers),
        3: tuple(pool_config.l3_followers),
    }

    rng = random.Random(pool_config.seed)
    agents: list[AgentConfig] = []
    for tier in (1, 2, 3):
        lo, hi = bins[tier]
        sub = df[(df["followers"] >= lo) & (df["followers"] < hi)]
        n = counts[tier]
        if sub.empty or n <= 0:
            continue
        n_take = min(n, len(sub))
        chosen = sub.sample(n=n_take, random_state=rng.randint(0, 2**31 - 1))
        for _, row in chosen.iterrows():
            agents.append(
                AgentConfig(
                    username=str(row["username"]),
                    name=str(row["name"]),
                    bio=str(row["bio"]),
                    profile_mode="simple",
                    tier=tier,
                )
            )
    return agents
