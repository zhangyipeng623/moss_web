import json
import re
from typing import Any

from json_repair import repair_json

from moss_agent_client.agent_logger import logger


def normalize_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    text_parts.append(str(text))
            else:
                text_parts.append(str(item))
        return "\n".join(part.strip() for part in text_parts if str(part).strip())
    if content is None:
        return ""
    return str(content).strip()


def _load_json_candidate(candidate: str) -> dict[str, Any]:
    try:
        payload = json.loads(candidate)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        pass

    try:
        repaired = repair_json(candidate)
        payload = json.loads(repaired)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def extract_json_dict(raw_content: Any, log_prefix: str = "") -> dict[str, Any]:
    text = normalize_text_content(raw_content)
    if not text:
        return {}

    candidates = [text]
    fenced_blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    candidates.extend(fenced_blocks)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        candidates.append(text[start : end + 1])

    seen_candidates = set()
    for candidate in candidates:
        normalized_candidate = candidate.strip()
        if not normalized_candidate or normalized_candidate in seen_candidates:
            continue
        seen_candidates.add(normalized_candidate)

        payload = _load_json_candidate(normalized_candidate)
        if payload:
            return payload

    prefix = f"{log_prefix} " if log_prefix else ""
    logger.warning(f"{prefix}无法解析结构化响应，原始内容：{text[:500]}")
    return {}
