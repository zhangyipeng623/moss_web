"""固定参数比较、误差与公共概率敏感性实验。

读取 model.json（候选系数）、test.json（固定测试集）、baseline.json（对照系数）
与同一画像目录，重建同一模拟环境后比较两组系数，输出 summary.json 与 per_story.csv。
本模块不调用 Optuna / E 步，不读取测试结果拟合任何值。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from analysis.recommender_data import file_sha256, load_split, publish_output, _read_json
from analysis.recommender_parameter_inference import (
    RecommendationParameterInferer,
    load_portrait_bundle,
    run_fixed_simulations,
)
from core.scoring import TIER_WEIGHT_DEFAULT

SCHEMA_VERSION = 1
N_BOOTSTRAP = 2000
DEFAULT_P_BASE_GRID = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.9, 0.999]

_WEIGHT_KEYS = ("w_i", "w_pop", "w_time", "w_rand")
_BASELINE_WEIGHT_FIELDS = ("w_interest", "w_popularity", "w_time", "w_random")


def compute_metrics(
    targets: np.ndarray,
    finals: np.ndarray,
    *,
    relative_threshold: float,
    absolute_threshold: float,
) -> Dict[str, Any]:
    """计算缩放终点传播量的误差指标。

    finals 形状为 (内容数, 重复数)；目标为零时相对指标返回 null。
    """
    targets = np.asarray(targets, dtype=float)
    finals = np.asarray(finals, dtype=float)
    if finals.ndim == 1:
        finals = finals[:, None]
    means = finals.mean(axis=1)
    abs_err = np.abs(means - targets)

    mae = float(abs_err.mean())
    rmse = float(np.sqrt(np.mean((means - targets) ** 2)))

    nonzero = targets > 0
    n_nonzero = int(nonzero.sum())
    if n_nonzero > 0:
        rel_err = abs_err[nonzero] / targets[nonzero]
        mre_nonzero = float(rel_err.mean())
        relative_pass_rate_nonzero = float(np.mean(rel_err <= relative_threshold))
    else:
        mre_nonzero = None
        relative_pass_rate_nonzero = None

    absolute_pass_rate = float(np.mean(abs_err <= absolute_threshold))
    zero_mask = targets == 0
    zero_target_mae = float(abs_err[zero_mask].mean()) if zero_mask.any() else None

    return {
        "mae": mae,
        "rmse": rmse,
        "mre_nonzero": mre_nonzero,
        "relative_pass_rate_nonzero": relative_pass_rate_nonzero,
        "absolute_pass_rate": absolute_pass_rate,
        "n_total": int(len(targets)),
        "n_nonzero": n_nonzero,
        "zero_target_mae": zero_target_mae,
    }


def _bootstrap_mae_diff(
    cand_means: np.ndarray,
    base_means: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
    n_bootstrap: int = N_BOOTSTRAP,
) -> Optional[List[float]]:
    """配对 bootstrap 计算 baseline_mae - candidate_mae 的 95% 百分位区间。

    少于 2 条测试内容时返回 None。
    """
    targets = np.asarray(targets, dtype=float)
    cand_means = np.asarray(cand_means, dtype=float)
    base_means = np.asarray(base_means, dtype=float)
    if len(targets) < 2:
        return None
    rng = np.random.default_rng(seed)
    cand_err = np.abs(cand_means - targets)
    base_err = np.abs(base_means - targets)
    n = len(targets)
    diffs = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diffs[i] = float(np.mean(base_err[idx] - cand_err[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return [float(lo), float(hi)]


def _parse_baseline(
    path: Path, model_decay: float
) -> Tuple[str, Dict[str, float], bool, Dict[str, float]]:
    """解析 baseline JSON，返回 (name, weights, decay_provided, raw_weights)。

    weights 已归一化并含 decay_lambda；raw_weights 为原始未归一化值。
    四项必须有限、非负、总和大于零；未知字段或缺项报错。
    """
    payload = _read_json(path)
    name = str(payload.get("name", path.stem))
    raw_weights = payload.get("weights")
    if not isinstance(raw_weights, dict):
        raise ValueError(f"baseline {path} 缺少 weights。")
    for field in _BASELINE_WEIGHT_FIELDS:
        if field not in raw_weights:
            raise ValueError(f"baseline weights 缺少字段：{field}")
    for field in raw_weights:
        if field not in _BASELINE_WEIGHT_FIELDS:
            raise ValueError(f"baseline weights 未知字段：{field}")

    raw = {
        "w_i": float(raw_weights["w_interest"]),
        "w_pop": float(raw_weights["w_popularity"]),
        "w_time": float(raw_weights["w_time"]),
        "w_rand": float(raw_weights["w_random"]),
    }
    if not all(np.isfinite(v) and v >= 0 for v in raw.values()):
        raise ValueError("baseline 权重必须有限且非负。")
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("baseline 权重总和必须大于零。")

    weights = {k: v / total for k, v in raw.items()}

    decay = payload.get("decay_lambda")
    decay_provided = decay is not None
    if decay is None:
        weights["decay_lambda"] = float(model_decay)
    else:
        decay = float(decay)
        if not np.isfinite(decay) or decay <= 0:
            raise ValueError("baseline decay_lambda 必须有限且大于零。")
        weights["decay_lambda"] = decay
    return name, weights, decay_provided, raw


def _load_model(model_path: Path) -> Dict[str, Any]:
    model = _read_json(model_path)
    if model.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"model schema_version 不支持：{model.get('schema_version')}")
    for key in ("data", "weights", "decay_lambda", "p_base_mode", "p_base_global",
                "environment", "portraits", "embedding"):
        if key not in model:
            raise ValueError(f"model 缺少字段：{key}")
    if model.get("p_base_mode") != "global":
        raise ValueError("仅支持 p_base_mode=global。")
    return model


def _csv_safe(value: Any) -> str:
    """把外部字符串转成 CSV 安全形式，防止 = + - @ 开头被当作公式。"""
    text = "" if value is None else str(value)
    if text[:1] in ("=", "+", "-", "@"):
        return "'" + text
    return text


_PER_STORY_HEADER = [
    "story_id", "condition", "p_base", "group_name", "target",
    "prediction_mean", "prediction_std", "absolute_error", "relative_error",
    "relative_pass", "absolute_pass", "n_repeats", "target_clipped",
]


def _build_per_story_rows(
    story_ids: List[str],
    targets: np.ndarray,
    clipped: np.ndarray,
    finals_a: np.ndarray,
    condition_a: str,
    group_a: str,
    finals_b: np.ndarray,
    condition_b: str,
    group_b: str,
    *,
    p_base: float,
    n_repeats: int,
    relative_threshold: float,
    absolute_threshold: float,
) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for i, sid in enumerate(story_ids):
        target = float(targets[i])
        for condition, group, finals in (
            (condition_a, group_a, finals_a),
            (condition_b, group_b, finals_b),
        ):
            mean = float(np.mean(finals[i]))
            std = float(np.std(finals[i]))
            abs_err = abs(mean - target)
            rel_err = "" if target == 0 else (abs_err / target)
            rel_pass = "" if target == 0 else (1 if abs_err / target <= relative_threshold else 0)
            abs_pass = 1 if abs_err <= absolute_threshold else 0
            rows.append(
                [
                    _csv_safe(sid), condition, p_base, _csv_safe(group),
                    target, mean, std, abs_err, rel_err, rel_pass, abs_pass,
                    n_repeats, int(bool(clipped[i])),
                ]
            )
    return rows


def compare_models(
    model_path: Path,
    test_path: Path,
    baseline_path: Path,
    portraits_dir: Path,
    output_dir: Path,
    *,
    n_repeats: int = 30,
    seed: int = 2026,
    p_base_grid: Optional[List[float]] = None,
    relative_threshold: float = 0.2,
    absolute_threshold: float = 1.0,
) -> Path:
    """在同一测试集上比较候选与对照系数，输出 summary.json 与 per_story.csv。"""
    model_path = Path(model_path)
    test_path = Path(test_path)
    baseline_path = Path(baseline_path)
    portraits_dir = Path(portraits_dir)
    output_dir = Path(output_dir)

    if n_repeats < 30:
        raise ValueError("n_repeats 不得小于 30。")
    if not np.isfinite(relative_threshold) or relative_threshold < 0:
        raise ValueError("relative_threshold 必须有限且非负。")
    if not np.isfinite(absolute_threshold) or absolute_threshold < 0:
        raise ValueError("absolute_threshold 必须有限且非负。")
    grid = p_base_grid if p_base_grid is not None else list(DEFAULT_P_BASE_GRID)
    if not grid:
        raise ValueError("p_base_grid 不能为空。")
    if any(not np.isfinite(p) or not 0.001 <= p <= 0.999 for p in grid):
        raise ValueError("p_base_grid 值必须在 [0.001, 0.999] 内。")

    model = _load_model(model_path)
    test_partition, _manifest = load_split(test_path, expected_split="test")
    test_records = test_partition["records"]
    if not test_records:
        raise ValueError("测试分区 records 为空。")

    train_ids = set(str(i) for i in model["data"].get("train_story_ids", []))
    test_ids = set(str(r["story_id"]) for r in test_records)
    if train_ids & test_ids:
        raise ValueError("模型训练 ID 与测试 ID 存在交集。")
    if int(test_partition["num_agents"]) != int(model["environment"]["population_size"]):
        raise ValueError("测试分区人口与模型环境不一致。")

    portraits_dir = Path(portraits_dir)
    if not portraits_dir.is_dir():
        raise ValueError(f"画像目录不存在：{portraits_dir}")
    personas, loaded_manifest = load_portrait_bundle(portraits_dir)
    if not personas:
        raise ValueError(f"画像目录没有可用画像：{portraits_dir}")
    portrait_manifest = model["portraits"].get("files", [])
    if portrait_manifest != loaded_manifest:
        raise ValueError("画像清单与模型记录不一致（文件或散列变化）。")

    env = model["environment"]
    embedding = model["embedding"]
    inferer = RecommendationParameterInferer(
        num_agents=int(env["population_size"]),
        embedding_model=embedding["model_name"],
        n_cpu=1,
        target_size_for_sampling=int(env["population_size"]),
        random_seed=int(env["training_seed"]),
        time_scale=float(env["time_scale"]),
        p_online=env["p_online"],
    )
    inferer.load_portraits(personas)
    inferer.load_prepared_stories(test_records)
    inferer.precompute_interests()

    engine = inferer._engine
    belief = env["belief_update"]
    if (
        abs(float(engine.backfire_mu) - float(belief["backfire_mu"])) > 1e-9
        or abs(float(engine.backfire_k) - float(belief["backfire_k"])) > 1e-9
        or abs(float(engine.learning_rate) - float(belief["learning_rate"])) > 1e-9
    ):
        raise ValueError("信念参数与模型环境不一致。")
    tier_env = {int(k): float(v) for k, v in dict(env.get("tier_weight", {})).items()}
    if tier_env != dict(TIER_WEIGHT_DEFAULT):
        raise ValueError("层级权重与模型环境不一致。")

    stories = list(inferer.representative_stories.values())

    name, baseline_weights, decay_provided, baseline_raw = _parse_baseline(
        baseline_path, model["decay_lambda"]
    )
    candidate_weights = {
        "w_i": float(model["weights"]["w_i"]),
        "w_pop": float(model["weights"]["w_pop"]),
        "w_time": float(model["weights"]["w_time"]),
        "w_rand": float(model["weights"]["w_rand"]),
        "decay_lambda": float(model["decay_lambda"]),
    }

    p_base = float(model["p_base_global"])
    targets = np.asarray([r["scaled_target"] for r in test_records], dtype=float)
    clipped = np.asarray([bool(r.get("target_clipped", False)) for r in test_records], dtype=bool)
    story_ids = [str(r["story_id"]) for r in test_records]
    duration = int(model["environment"]["rounds"])

    def _run(weights: Dict[str, float], p: float) -> np.ndarray:
        return run_fixed_simulations(
            engine, stories, weights, p,
            duration=duration, n_repeats=n_repeats, seed=seed, n_cpu=1,
        )

    cand_finals = _run(candidate_weights, p_base)
    base_finals = _run(baseline_weights, p_base)

    cand_metrics = compute_metrics(
        targets, cand_finals, relative_threshold=relative_threshold,
        absolute_threshold=absolute_threshold,
    )
    base_metrics = compute_metrics(
        targets, base_finals, relative_threshold=relative_threshold,
        absolute_threshold=absolute_threshold,
    )
    n_clipped = int(clipped.sum())
    cand_metrics["n_clipped"] = n_clipped
    base_metrics["n_clipped"] = n_clipped

    def _diff(c: float, b: float) -> Optional[float]:
        if c is None or b is None:
            return None
        return float(b - c)

    improvement = {
        "mae": _diff(cand_metrics["mae"], base_metrics["mae"]),
        "rmse": _diff(cand_metrics["rmse"], base_metrics["rmse"]),
        "mre_nonzero": _diff(cand_metrics["mre_nonzero"], base_metrics["mre_nonzero"]),
        "absolute_pass_rate": _diff(
            cand_metrics["absolute_pass_rate"], base_metrics["absolute_pass_rate"]
        ),
    }
    improvement_rel = {
        "mae": _rel(improvement["mae"], base_metrics["mae"]),
        "rmse": _rel(improvement["rmse"], base_metrics["rmse"]),
    }

    mae_diff_ci = _bootstrap_mae_diff(
        cand_finals.mean(axis=1), base_finals.mean(axis=1), targets, seed=seed
    )

    grid_seen: set = set()
    sensitivity: List[Dict[str, Any]] = []
    for p in grid:
        key = float(p)
        if key in grid_seen:
            continue
        grid_seen.add(key)
        if abs(key - p_base) < 1e-12:
            c_metrics, b_metrics = cand_metrics, base_metrics
        else:
            c_metrics = compute_metrics(
                targets, _run(candidate_weights, key),
                relative_threshold=relative_threshold, absolute_threshold=absolute_threshold,
            )
            b_metrics = compute_metrics(
                targets, _run(baseline_weights, key),
                relative_threshold=relative_threshold, absolute_threshold=absolute_threshold,
            )
        sensitivity.append(
            {
                "p_base": key,
                "candidate": c_metrics,
                "baseline": b_metrics,
                "candidate_mae_lower": bool(c_metrics["mae"] < b_metrics["mae"]),
            }
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "model": str(model_path),
            "model_hash": file_sha256(model_path),
            "test": str(test_path),
            "test_hash": file_sha256(test_path),
            "manifest_hash": file_sha256(test_path.parent / "manifest.json"),
            "baseline": str(baseline_path),
            "baseline_hash": file_sha256(baseline_path),
            "portraits_dir": str(portraits_dir),
            "portrait_files": portrait_manifest,
        },
        "configuration": {
            "n_repeats": n_repeats,
            "seed": seed,
            "relative_threshold": relative_threshold,
            "absolute_threshold": absolute_threshold,
            "p_base_grid": [float(p) for p in grid],
            "baseline_name": name,
            "baseline_decay_provided": decay_provided,
            "baseline_weights_raw": baseline_raw,
            "baseline_weights_normalized": {k: baseline_weights[k] for k in _WEIGHT_KEYS},
        },
        "main": {
            "candidate": cand_metrics,
            "baseline": base_metrics,
            "improvement": improvement,
            "improvement_rel": improvement_rel,
            "mae_difference_ci95": mae_diff_ci,
        },
        "sensitivity": sensitivity,
    }

    per_story_rows = _build_per_story_rows(
        story_ids, targets, clipped,
        cand_finals, "candidate", "candidate",
        base_finals, "baseline", name,
        p_base=p_base, n_repeats=n_repeats,
        relative_threshold=relative_threshold, absolute_threshold=absolute_threshold,
    )

    with publish_output(output_dir) as temp_dir:
        with (temp_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)
        with (temp_dir / "per_story.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(_PER_STORY_HEADER)
            for row in per_story_rows:
                writer.writerow(row)

    return output_dir / "summary.json"


def _rel(imp: Optional[float], base: Optional[float]) -> Optional[float]:
    if imp is None or base is None or abs(base) < 1e-12:
        return None
    return float(imp / base)


def _parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="比较候选与对照推荐系数")
    parser.add_argument("--model", required=True, help="候选 model.json 路径")
    parser.add_argument("--test-file", required=True, help="测试分区 test.json 路径")
    parser.add_argument("--baseline", required=True, help="对照系数 baseline.json 路径")
    parser.add_argument("--portraits-dir", required=True, help="用户画像目录")
    parser.add_argument("--output-dir", required=True, help="输出目录（summary.json + per_story.csv）")
    parser.add_argument("--n-repeats", type=int, default=30, help="每组每条重复次数，不得小于 30")
    parser.add_argument("--seed", type=int, default=2026, help="比较随机种子")
    parser.add_argument("--p-base-grid", default=None, help="逗号分隔的公共概率网格（覆盖默认网格）")
    parser.add_argument("--relative-threshold", type=float, default=0.2, help="相对误差达标阈值")
    parser.add_argument("--absolute-threshold", type=float, default=1.0, help="绝对误差达标阈值")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    grid = None
    if args.p_base_grid:
        grid = [float(x.strip()) for x in args.p_base_grid.split(",") if x.strip()]
    summary = compare_models(
        Path(args.model), Path(args.test_file), Path(args.baseline),
        Path(args.portraits_dir), Path(args.output_dir),
        n_repeats=args.n_repeats, seed=args.seed, p_base_grid=grid,
        relative_threshold=args.relative_threshold,
        absolute_threshold=args.absolute_threshold,
    )
    print(f"比较结果已写入：{summary}")


if __name__ == "__main__":
    main()
