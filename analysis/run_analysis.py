from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo


PORTRAIT_USER_SUPPORTED_COLUMNS = (
    "用户名",
    "昵称",
    "简介",
    "性别",
    "地域",
    "关注",
    "粉丝",
    "收藏",
    "源用户名",
    "用户地址",
    "创建时间戳",
    "头像链接",
)
PORTRAIT_USER_REQUIRED_COLUMNS = (
    "用户名",
    "昵称",
    "简介",
    "性别",
    "地域",
    "关注",
    "粉丝",
    "收藏",
    "创建时间戳",
)
PORTRAIT_POST_SUPPORTED_COLUMNS = (
    "用户名",
    "发文内容",
    "发布时间",
    "发布时间戳",
    "发文类型",
    "点赞数",
    "评论数",
    "转发数",
    "帖子ID",
    "推文ID",
    "post_id",
    "id",
)
PORTRAIT_POST_REQUIRED_COLUMNS = (
    "用户名",
    "发文内容",
    "发布时间",
    "发布时间戳",
    "发文类型",
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="离线分析脚本")
    subparsers = parser.add_subparsers(dest="command", required=True)

    portrait_parser = subparsers.add_parser("portrait", help="生成用户画像")
    portrait_parser.add_argument(
        "--data-path",
        help="数据目录路径，目录下默认包含 user.xlsx 和 post.xlsx",
    )
    portrait_parser.add_argument(
        "--user-name",
        help="目标用户名，单用户模式必填；使用 --batch 批量模式时无需提供",
    )
    portrait_parser.add_argument(
        "--user-file",
        default="user.xlsx",
        help="用户信息文件名，默认 user.xlsx",
    )
    portrait_parser.add_argument(
        "--post-file",
        default="post.xlsx",
        help="帖子文件名，默认 post.xlsx",
    )
    portrait_parser.add_argument(
        "--output",
        help=(
            "单用户模式：画像输出 JSON 路径，默认 analysis_outputs/portraits/<user_name>.json；"
            "批量模式：画像输出目录，默认 analysis_outputs/portraits/"
        ),
    )
    portrait_parser.add_argument(
        "--reference-time",
        help=(
            "画像生成参考时间，未提供时会交互输入。"
            "若不带时区，默认按 Asia/Shanghai 解析，例如 2026-04-01 12:00:00"
        ),
    )
    portrait_parser.add_argument(
        "--model",
        default=os.environ.get("MODEL", "gpt-4o"),
        help="模型名称，默认读取环境变量 MODEL",
    )
    portrait_parser.add_argument("--timeout", type=int, default=180, help="模型调用超时时间")
    portrait_parser.add_argument(
        "--api-key-env",
        default="API_KEY",
        help="读取模型密钥的环境变量名",
    )
    portrait_parser.add_argument(
        "--base-url-env",
        default="BASE_URL",
        help="读取模型服务地址的环境变量名",
    )
    portrait_parser.add_argument(
        "--batch",
        action="store_true",
        help="批量模式：自动读取 user.xlsx 中全部用户并依次生成画像，此时 --user-name 无需提供",
    )

    retier_parser = subparsers.add_parser("retier", help="对已有画像 JSON 重新按影响力分级")
    retier_parser.add_argument(
        "--portraits-dir",
        required=True,
        help="存放画像 JSON 文件的目录路径",
    )

    recommender_parser = subparsers.add_parser("recommender", help="反推推荐参数")
    recommender_parser.add_argument(
        "--data-file",
        help="推荐系统观测表路径，支持 xlsx/csv",
    )
    recommender_parser.add_argument(
        "--retweet-columns",
        default="转发,分享,Quotes",
        help="用于汇总总转发量的列名，逗号分隔",
    )
    recommender_parser.add_argument(
        "--view-column",
        default="观看量",
        help="观看量列名，默认 观看量",
    )
    recommender_parser.add_argument(
        "--id-column",
        default="文章ID",
        help="内容 ID 列名，默认 文章ID",
    )
    recommender_parser.add_argument(
        "--anchor-percentile",
        type=float,
        default=0.8,
        help="鲁棒缩放锚点分位数，默认 0.8",
    )
    recommender_parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="EM 校准最大迭代次数，默认 3",
    )
    recommender_parser.add_argument(
        "--num-agents",
        type=int,
        default=1500,
        help="ABM 模拟代理数量，默认 1500",
    )
    recommender_parser.add_argument(
        "--avg-degree",
        type=int,
        default=20,
        help="平均连接度，默认 20",
    )
    recommender_parser.add_argument(
        "--verified-ratio",
        type=float,
        default=0.01,
        help="认证账号占比，默认 0.01",
    )
    recommender_parser.add_argument(
        "--min-scaled-target",
        type=int,
        default=3,
        help="最小缩放目标阈值，默认 3",
    )
    recommender_parser.add_argument(
        "--n-trials-per-story",
        type=int,
        default=40,
        help="单条内容概率校准试验次数，默认 40",
    )
    recommender_parser.add_argument(
        "--n-trials-per-weight",
        type=int,
        default=100,
        help="权重优化试验次数，默认 100",
    )
    recommender_parser.add_argument(
        "--n-simulations-per-trial",
        type=int,
        default=5,
        help="每次试验的模拟次数，默认 5",
    )
    recommender_parser.add_argument(
        "--input",
        help=argparse.SUPPRESS,
    )
    recommender_parser.add_argument(
        "--output",
        help="推荐参数输出 JSON 路径，默认写入 analysis_outputs/recommender/<输入文件名>.json",
    )

    return parser.parse_args()


async def run_portrait_command(args: argparse.Namespace) -> None:
    """执行用户画像生成命令。"""
    from langchain_openai import ChatOpenAI

    from analysis.user_portrait_generator import PortraitGenerationError, UserPortraitGenerator

    source, global_event = _resolve_portrait_source(args)
    reference_timestamp, reference_time_text = _resolve_portrait_reference_time(args.reference_time)
    output_path = _resolve_output_path(
        args.output,
        default_relative_path=f"analysis_outputs/portraits/{source.user_name}.json",
    )

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"环境变量 {args.api_key_env} 未设置，无法调用画像生成模型。")

    llm = ChatOpenAI(
        model=args.model,
        api_key=api_key,
        base_url=os.environ.get(args.base_url_env),
        timeout=args.timeout,
    )
    generator = UserPortraitGenerator(reference_timestamp=reference_timestamp)
    llm_callable = _build_llm_callable(llm)

    last_error = ""
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            profile = await generator.generate_portrait(
                source=source,
                llm_callable=llm_callable,
                global_event=global_event,
            )
            profile["influence_tier"] = 3
            profile["influence_tier_label"] = "活跃参与者（默认）"
            _write_json(output_path, profile)
            print(
                f"用户画像已写入：{output_path}；参考时间：{reference_time_text}；"
                f"尝试次数：{attempt}"
            )
            return
        except PortraitGenerationError as exc:
            last_error = str(exc)
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)

        if attempt < max_attempts:
            print(
                f"用户 {source.user_name} 第 {attempt} 次画像生成失败，准备重试：{last_error}"
            )

    failure_report_path = _record_failed_portrait_user(
        output_path=output_path,
        user_name=source.user_name,
        reference_time_text=reference_time_text,
        error_message=last_error or "未知错误",
    )
    raise RuntimeError(
        f"用户 {source.user_name} 连续 {max_attempts} 次画像生成失败，"
        f"未输出画像文件。失败名单已写入：{failure_report_path}"
    )


async def run_portrait_batch_command(args: argparse.Namespace) -> None:
    """批量生成全部用户画像，并按影响力自动分 3/4/5 级。"""
    from langchain_openai import ChatOpenAI

    from analysis.user_portrait_generator import PortraitGenerationError, UserPortraitGenerator

    data_path = Path(args.data_path).resolve()
    all_usernames = _extract_all_usernames(data_path, args.user_file)

    reference_timestamp, reference_time_text = _resolve_portrait_reference_time(
        args.reference_time
    )

    output_dir = (
        Path(args.output).resolve()
        if args.output
        else Path("analysis_outputs/portraits").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"环境变量 {args.api_key_env} 未设置，无法调用画像生成模型。")

    generator = UserPortraitGenerator(reference_timestamp=reference_timestamp)

    # ----------------------------------------------------------------
    # Phase 1: 预计算所有用户的 account_influence（不调 LLM）
    # ----------------------------------------------------------------
    user_records: list[dict[str, Any]] = []
    phase1_failed: list[dict[str, str]] = []

    print(f"阶段 1/2：预计算影响力分数，共 {len(all_usernames)} 个用户 ...\n")
    for username in all_usernames:
        try:
            source, _ = _build_portrait_source_from_excel(
                data_path=data_path,
                user_name=username,
                user_file=args.user_file,
                post_file=args.post_file,
            )
        except Exception as exc:
            phase1_failed.append(
                {
                    "user_name": username,
                    "reference_time": reference_time_text,
                    "error_message": str(exc),
                }
            )
            continue
        stats = generator.analyze_stats(source)
        user_records.append(
            {
                "username": username,
                "canonical_name": source.user_name,
                "source": source,
                "account_influence": stats.account_influence,
                "fans_count": stats.fans_count,
                "post_count": stats.post_count,
            }
        )

    if not user_records:
        raise RuntimeError("没有可用用户数据，无法进行批量画像生成。")

    # ----------------------------------------------------------------
    # Phase 2: 按分位数切分 3/4/5 级（P60 / P90）
    # ----------------------------------------------------------------
    sorted_records = sorted(user_records, key=lambda r: r["account_influence"])
    n = len(sorted_records)
    p60_threshold = sorted_records[int(n * 0.60)]["account_influence"] if n >= 5 else 0
    p90_threshold = sorted_records[int(n * 0.90)]["account_influence"] if n >= 5 else float("inf")

    tier_counts: dict[int, int] = {3: 0, 4: 0, 5: 0}
    for record in sorted_records:
        score = record["account_influence"]
        if score > p90_threshold and p90_threshold != float("inf"):
            record["tier"] = 5
        elif score > p60_threshold:
            record["tier"] = 4
        else:
            record["tier"] = 3
        tier_counts[record["tier"]] += 1
        record["tier_label"] = _tier_label(record["tier"])

    print(f"影响力分级完成（阈值 P60={p60_threshold:.2f}, P90={p90_threshold:.2f}）："
          f"5级 {tier_counts[5]} 人 / 4级 {tier_counts[4]} 人 / 3级 {tier_counts[3]} 人\n")

    # ----------------------------------------------------------------
    # Phase 3: 调 LLM 生成画像，注入 influence_tier（跳过已存在文件）
    # ----------------------------------------------------------------
    llm = ChatOpenAI(
        model=args.model,
        api_key=api_key,
        base_url=os.environ.get(args.base_url_env),
        timeout=args.timeout,
    )
    llm_callable = _build_llm_callable(llm)

    existing_files = {
        p.stem for p in output_dir.glob("*.json") if p.name != "failed_users.json"
    }
    pending_records = [
        r for r in user_records if r["canonical_name"] not in existing_files
    ]
    skipped_count = len(user_records) - len(pending_records)

    if skipped_count > 0:
        print(f"跳过 {skipped_count} 个已有画像的用户")
        # 对已跳过的用户原地更新 tier 字段
        for record in user_records:
            if record["canonical_name"] in existing_files:
                existing_path = output_dir / f"{record['canonical_name']}.json"
                try:
                    existing_profile = _load_json_object(existing_path)
                except Exception:
                    continue
                existing_profile["influence_tier"] = record["tier"]
                existing_profile["influence_tier_label"] = record["tier_label"]
                _write_json(existing_path, existing_profile)
        print()

    if not pending_records:
        print("所有用户画像均已存在，无需生成。")
        return

    success_count = 0
    phase3_failed: list[dict[str, str]] = []
    total = len(pending_records)

    print(f"阶段 2/2：批量生成用户画像（{total} 个待生成）...\n")

    for idx, record in enumerate(pending_records, start=1):
        source = record["source"]
        username = record["username"]
        tier = record["tier"]
        print(
            f"[{idx}/{total}] {username} (Lv{tier}) 画像生成中 ...",
            end=" ",
            flush=True,
        )

        last_error = ""
        max_attempts = 3
        profile = None
        for attempt in range(1, max_attempts + 1):
            try:
                profile = await generator.generate_portrait(
                    source=source,
                    llm_callable=llm_callable,
                )
                break
            except PortraitGenerationError as exc:
                last_error = str(exc)
            except Exception as exc:
                last_error = str(exc)
            if attempt < max_attempts:
                print(f"\n    第 {attempt} 次失败，重试中...", end=" ", flush=True)

        if profile is None:
            print(f"失败（{max_attempts} 次重试后）：{last_error}")
            phase3_failed.append(
                {
                    "user_name": username,
                    "reference_time": reference_time_text,
                    "error_message": last_error or "未知错误",
                }
            )
            continue

        profile["influence_tier"] = tier
        profile["influence_tier_label"] = record["tier_label"]
        output_path = output_dir / f"{record['canonical_name']}.json"
        _write_json(output_path, profile)
        success_count += 1
        print("完成")

    # ----------------------------------------------------------------
    # 汇总
    # ----------------------------------------------------------------
    full_total = len(user_records)
    all_failed = phase1_failed + phase3_failed
    print(
        f"\n批量画像生成结束：本次生成 {success_count}/{total}；"
        f"总计 {full_total} 人（已跳过 {skipped_count}），失败 {len(all_failed)} 人"
    )
    print(f"分级统计：5级 {tier_counts[5]} 人 / 4级 {tier_counts[4]} 人 / 3级 {tier_counts[3]} 人")

    if all_failed:
        failure_report_path = output_dir / "failed_users.json"
        payload: dict[str, Any] = {
            "failed_users": [item["user_name"] for item in all_failed],
            "failed_details": all_failed,
        }
        _write_json(failure_report_path, payload)
        print(f"失败名单已写入：{failure_report_path}")


def _tier_label(tier: int) -> str:
    """影响力级别对应的中文标签。"""
    labels = {
        3: "活跃参与者",
        4: "影响力用户",
        5: "核心意见领袖",
    }
    return labels.get(tier, f"Lv{tier}")


def run_retier_command(args: argparse.Namespace) -> None:
    """对已有画像 JSON 目录重新计算影响力分级，并回写每个文件。"""
    portraits_dir = Path(args.portraits_dir).resolve()
    if not portraits_dir.is_dir():
        raise ValueError(f"目录不存在：{portraits_dir}")

    json_files = sorted(portraits_dir.glob("*.json"))
    if not json_files:
        raise ValueError(f"目录中没有 JSON 文件：{portraits_dir}")

    # 读取所有画像，提取 account_influence
    records: list[dict[str, Any]] = []
    for path in json_files:
        try:
            profile = _load_json_object(path)
        except Exception:
            print(f"跳过无法解析的文件：{path.name}")
            continue
        stats = profile.get("stats")
        if not isinstance(stats, dict):
            print(f"跳过缺少 stats 字段的文件：{path.name}")
            continue
        score = stats.get("account_influence")
        if not isinstance(score, (int, float)):
            print(f"跳过缺少 account_influence 的文件：{path.name}")
            continue
        records.append(
            {
                "path": path,
                "profile": profile,
                "account_influence": float(score),
                "user_id": profile.get("user_id", path.stem),
            }
        )

    if not records:
        raise RuntimeError("没有可用的画像文件（均缺少 stats.account_influence）。")

    # 按 account_influence 排序，计算 P60 / P90
    sorted_records = sorted(records, key=lambda r: r["account_influence"])
    n = len(sorted_records)
    p60_threshold = sorted_records[int(n * 0.60)]["account_influence"] if n >= 5 else 0
    p90_threshold = sorted_records[int(n * 0.90)]["account_influence"] if n >= 5 else float("inf")

    tier_counts: dict[int, int] = {3: 0, 4: 0, 5: 0}
    for record in sorted_records:
        score = record["account_influence"]
        if score > p90_threshold and p90_threshold != float("inf"):
            tier = 5
        elif score > p60_threshold:
            tier = 4
        else:
            tier = 3
        record["tier"] = tier
        tier_counts[tier] += 1

    # 回写每个 JSON
    for record in sorted_records:
        record["profile"]["influence_tier"] = record["tier"]
        record["profile"]["influence_tier_label"] = _tier_label(record["tier"])
        _write_json(record["path"], record["profile"])

    print(
        f"retier 完成，共处理 {n} 个画像文件\n"
        f"阈值 P60={p60_threshold:.2f}, P90={p90_threshold:.2f}\n"
        f"5级 {tier_counts[5]} 人 / 4级 {tier_counts[4]} 人 / 3级 {tier_counts[3]} 人"
    )


def run_recommender_command(args: argparse.Namespace) -> None:
    """执行推荐参数反推命令。"""
    from analysis.recommender_parameter_inference import (
        RecommendationParameterInferer,
        build_story_observations,
    )

    records, inferer_config, anchor_percentile, max_iterations, input_stem = (
        _resolve_recommender_input(args)
    )
    observations = build_story_observations(records)
    inferer = RecommendationParameterInferer(**inferer_config)
    representative_stories = inferer.select_representative_stories(
        observations,
        anchor_percentile=anchor_percentile,
    )
    best_weights: dict[str, Any] = {}
    if representative_stories:
        best_weights = inferer.run_em_calibration_loop(max_iterations=max_iterations)

    result = {
        "input_meta": {
            "record_count": len(records),
            "selected_story_count": len(representative_stories),
            "anchor_percentile": anchor_percentile,
            "max_iterations": max_iterations,
            "inferer_config": inferer_config,
            "fixed_p_online": inferer.p_online,
            "fixed_duration_hours": inferer.duration,
        },
        "representative_stories": representative_stories,
        "calibrated_probs": inferer.calibrated_probs,
        "best_weights": best_weights,
        "weight_fit_diagnostics": inferer.weight_fit_diagnostics,
    }
    output_path = _resolve_output_path(
        args.output,
        default_relative_path=f"analysis_outputs/recommender/{input_stem}.json",
    )
    _write_json(output_path, result)
    print(f"推荐参数结果已写入：{output_path}")


def _resolve_portrait_source(
    args: argparse.Namespace,
) -> tuple[UserProfileSource, str]:
    """解析画像命令输入来源。"""
    if not args.data_path:
        raise ValueError("请提供 --data-path。")
    if not args.user_name:
        raise ValueError("使用 Excel 数据目录模式时必须提供 --user-name。")

    return _build_portrait_source_from_excel(
        data_path=Path(args.data_path).resolve(),
        user_name=args.user_name,
        user_file=args.user_file,
        post_file=args.post_file,
    )


def _resolve_recommender_input(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any], float, int, str]:
    """解析推荐参数命令输入来源。"""
    if args.input:
        input_path = Path(args.input).resolve()
        payload = _load_json_object(input_path)
        records = payload.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError("推荐参数输入 JSON 需要提供非空 records 数组。")
        _validate_recommender_records(records)

        inferer_config = payload.get("inferer_config") or {}
        if not isinstance(inferer_config, dict):
            raise ValueError("inferer_config 必须是 JSON 对象。")
        _validate_recommender_inferer_config(inferer_config)

        anchor_percentile = float(payload.get("anchor_percentile", 0.8))
        max_iterations = int(payload.get("max_iterations", 3))
        _validate_recommender_meta(anchor_percentile, max_iterations)
        return records, inferer_config, anchor_percentile, max_iterations, input_path.stem

    if not args.data_file:
        raise ValueError("请提供 --data-file，或使用兼容模式的 --input。")

    data_file = Path(args.data_file).resolve()
    records = _build_recommender_records_from_table(
        path=data_file,
        retweet_columns=args.retweet_columns,
        view_column=args.view_column,
        id_column=args.id_column,
    )
    _validate_recommender_records(records)
    inferer_config = {
        "num_agents": args.num_agents,
        "avg_degree": args.avg_degree,
        "verified_ratio": args.verified_ratio,
        "min_scaled_target": args.min_scaled_target,
        "n_trials_per_story": args.n_trials_per_story,
        "n_trials_per_weight": args.n_trials_per_weight,
        "n_simulations_per_trial": args.n_simulations_per_trial,
    }
    _validate_recommender_inferer_config(inferer_config)
    _validate_recommender_meta(args.anchor_percentile, args.max_iterations)
    return (
        records,
        inferer_config,
        args.anchor_percentile,
        args.max_iterations,
        data_file.stem,
    )


def _build_portrait_source_from_excel(
    data_path: Path,
    user_name: str,
    user_file: str,
    post_file: str,
) -> tuple[UserProfileSource, str]:
    """按原有 Excel 数据结构提取单个用户的资料和帖子。"""
    from analysis.user_portrait_generator import UserPost, UserProfileSource

    user_path = data_path / user_file
    post_path = data_path / post_file
    user_rows = _load_tabular_rows(user_path)
    post_rows = _load_tabular_rows(post_path)
    _validate_tabular_columns(
        rows=user_rows,
        path=user_path,
        required_columns=PORTRAIT_USER_REQUIRED_COLUMNS,
        supported_columns=PORTRAIT_USER_SUPPORTED_COLUMNS,
    )
    _validate_tabular_columns(
        rows=post_rows,
        path=post_path,
        required_columns=PORTRAIT_POST_REQUIRED_COLUMNS,
        supported_columns=PORTRAIT_POST_SUPPORTED_COLUMNS,
    )

    target_name = _normalize_username(user_name)
    if not target_name:
        raise ValueError("user_name 不能为空。")

    matched_user_rows = [
        row
        for row in user_rows
        if _normalize_username(row.get("用户名")) == target_name
    ]
    user_row = _select_best_user_row(matched_user_rows)
    if user_row is None:
        raise ValueError(f"在 {user_path} 中找不到用户：{user_name}")

    canonical_user_name = _normalize_scalar(user_row.get("用户名")) or user_name
    matched_posts = [
        row
        for row in post_rows
        if _normalize_username(row.get("用户名")) == target_name
    ]
    if not matched_posts:
        raise ValueError(f"在 {post_path} 中找不到用户 {user_name} 的帖子。")

    ordered_posts = sorted(
        matched_posts,
        key=lambda row: (
            _resolve_post_timestamp(row),
            _normalize_scalar(row.get("发布时间")),
            _safe_float(row.get("转发数", 0))
            + _safe_float(row.get("评论数", 0))
            + _safe_float(row.get("点赞数", 0)),
        ),
        reverse=True,
    )

    user_info = {
        "username": canonical_user_name,
        "nickname": _normalize_scalar(user_row.get("昵称")),
        "description": _normalize_scalar(user_row.get("简介")),
        "sex": _normalize_scalar(user_row.get("性别")),
        "region": _normalize_scalar(user_row.get("地域")),
        "follow_count": _safe_int(user_row.get("关注", 0)),
        "fans_count": _safe_int(user_row.get("粉丝", 0)),
        "collect_count": _safe_int(user_row.get("收藏", 0)),
        "follow_user": _normalize_scalar(user_row.get("源用户名")),
        "home_page_url": _normalize_scalar(user_row.get("用户地址")),
        "register_timestamp": _safe_int(user_row.get("创建时间戳", 0)),
        "profile_img_url": _normalize_scalar(user_row.get("头像链接")),
    }

    posts: list[UserPost] = []
    seen_post_keys: set[tuple[Any, ...]] = set()
    for row in ordered_posts:
        content = _normalize_scalar(row.get("发文内容"))
        if not content:
            continue
        timestamp = _resolve_post_timestamp(row)
        publish_time = _normalize_scalar(row.get("发布时间"))
        content_type = _normalize_scalar(row.get("发文类型")) or "Social Post"
        source_post_id = _safe_optional_int(
            row.get("帖子ID")
            or row.get("post_id")
            or row.get("推文ID")
            or row.get("id")
        )
        if source_post_id is not None:
            dedupe_key = ("post_id", source_post_id)
        else:
            dedupe_key = ("content", content, timestamp, publish_time, content_type)
        if dedupe_key in seen_post_keys:
            continue
        seen_post_keys.add(dedupe_key)
        posts.append(
            UserPost(
                content=content,
                timestamp=timestamp,
                content_type=content_type,
                publish_time=publish_time,
                like_count=_safe_int(row.get("点赞数", 0)),
                comment_count=_safe_int(row.get("评论数", 0)),
                share_count=_safe_int(row.get("转发数", 0)),
                post_id=source_post_id,
            )
        )

    if not posts:
        raise ValueError(f"用户 {user_name} 的帖子数据为空，无法生成画像。")

    source = UserProfileSource(
        user_name=canonical_user_name,
        user_info=user_info,
        posts=posts,
    )
    return source, ""


def _build_recommender_records_from_table(
    path: Path,
    retweet_columns: str,
    view_column: str,
    id_column: str,
) -> list[dict[str, Any]]:
    """从原始推荐系统 Excel 表中抽取观测记录。"""
    rows = _load_tabular_rows(path)
    retweet_fields = [item.strip() for item in retweet_columns.split(",") if item.strip()]
    if not retweet_fields:
        raise ValueError("retweet_columns 不能为空。")

    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        total_repost = sum(_safe_float(row.get(field, 0)) for field in retweet_fields)
        view_count = _safe_float(row.get(view_column, 0))
        story_id = _normalize_scalar(row.get(id_column)) or str(index)
        if total_repost <= 0 or view_count <= 100:
            continue
        records.append(
            {
                "story_id": story_id,
                "repost_count": total_repost,
                "view_count": view_count,
            }
        )
    if not records:
        raise ValueError("按当前 Excel 列配置没有筛出有效的推荐观测数据。")
    return records


def _build_llm_callable(llm: ChatOpenAI) -> LlmCallable:
    """把通用 LLM 客户端包装成画像生成器使用的调用签名。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    LlmCallable = Callable[[str, str], Awaitable[dict[str, Any]]]

    async def _call(system_prompt: str, user_prompt: str) -> dict[str, Any]:
        result = await llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        return _extract_json_dict(result.content)

    return _call


def _extract_json_dict(raw_content: Any) -> dict[str, Any]:
    """使用 json_repair 从模型原始输出中提取 JSON 对象。"""
    from json_repair import repair_json

    text = _normalize_text_content(raw_content)
    if not text:
        return {}
    try:
        payload = json.loads(repair_json(text))
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _normalize_text_content(content: Any) -> str:
    """归一化模型返回文本。"""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
                continue
            parts.append(str(item))
        return "\n".join(part.strip() for part in parts if part.strip())
    if content is None:
        return ""
    return str(content).strip()


def _load_json_object(path: Path) -> dict[str, Any]:
    """读取 JSON 对象。"""
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"文件 {path} 必须是 JSON 对象。")
    return payload


def _load_tabular_rows(path: Path) -> list[dict[str, Any]]:
    """读取 Excel 或 CSV 表格数据。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件：{path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            return [dict(row) for row in reader]
    if suffix != ".xlsx":
        raise ValueError(f"暂不支持的表格格式：{path.suffix}")

    try:
        from openpyxl import load_workbook
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "缺少 openpyxl，请先执行 `uv add openpyxl` 或 `uv sync`。"
        ) from exc

    workbook = load_workbook(path, read_only=True, data_only=True)
    sheet = workbook.active
    rows = list(sheet.iter_rows(values_only=True))
    if not rows:
        return []

    header = [str(cell).strip() if cell is not None else "" for cell in rows[0]]
    result: list[dict[str, Any]] = []
    for row in rows[1:]:
        item = {
            header[index]: row[index] if index < len(row) else None
            for index in range(len(header))
            if header[index]
        }
        if any(value not in (None, "") for value in item.values()):
            result.append(item)
    workbook.close()
    return result


def _validate_tabular_columns(
    rows: list[dict[str, Any]],
    path: Path,
    required_columns: tuple[str, ...],
    supported_columns: tuple[str, ...],
) -> None:
    """校验表格列是否符合旧版数据约定。"""
    if not rows:
        raise ValueError(f"数据文件为空：{path}")
    keys = {str(key).strip() for row in rows for key in row.keys() if str(key).strip()}
    missing_required = [column for column in required_columns if column not in keys]
    if missing_required:
        joined = "、".join(missing_required)
        raise ValueError(f"{path} 缺少必需列：{joined}")
    if not keys.intersection(supported_columns):
        joined = "、".join(supported_columns)
        raise ValueError(f"{path} 不符合预期列结构，应至少包含这些列中的一部分：{joined}")


def _extract_all_usernames(data_path: Path, user_file: str) -> list[str]:
    """从 user.xlsx/csv 中提取所有去重后的用户名列表。"""
    user_path = data_path / user_file
    rows = _load_tabular_rows(user_path)
    seen: set[str] = set()
    usernames: list[str] = []
    for row in rows:
        name = _normalize_username(row.get("用户名"))
        if name and name not in seen:
            seen.add(name)
            usernames.append(name)
    if not usernames:
        raise ValueError(f"在 {user_path} 中未找到任何用户名。")
    return usernames


def _normalize_username(value: Any) -> str:
    """统一用户名匹配规则。"""
    return _normalize_scalar(value).lower()


def _select_best_user_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """同名用户存在多行时，优先选择信息最完整的一行。"""
    if not rows:
        return None
    scored_rows = sorted(
        rows,
        key=lambda row: (
            sum(1 for column in PORTRAIT_USER_SUPPORTED_COLUMNS if _normalize_scalar(row.get(column))),
            _safe_int(row.get("创建时间戳", 0)),
        ),
        reverse=True,
    )
    return scored_rows[0]


def _resolve_post_timestamp(row: dict[str, Any]) -> int:
    """优先使用时间戳，缺失时再解析可读时间。"""
    timestamp = _safe_int(row.get("发布时间戳", 0))
    if timestamp > 0:
        return timestamp
    publish_time = _normalize_scalar(row.get("发布时间"))
    if publish_time:
        return _parse_publish_time_to_timestamp(publish_time)
    return 0


def _validate_recommender_records(records: list[Any]) -> None:
    """校验推荐参数输入记录。"""
    required_fields = ("story_id", "repost_count", "view_count")
    for index, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError(f"records[{index}] 必须是 JSON 对象。")
        for field_name in required_fields:
            if field_name not in item:
                raise ValueError(f"records[{index}] 缺少字段：{field_name}")
        try:
            float(item["repost_count"])
            float(item["view_count"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"records[{index}] 的 repost_count/view_count 必须是数值。"
            ) from exc


def _validate_recommender_meta(anchor_percentile: float, max_iterations: int) -> None:
    """校验推荐参数元数据。"""
    if not 0 < anchor_percentile <= 1:
        raise ValueError("anchor_percentile 必须在 (0, 1] 区间内。")
    if max_iterations <= 0:
        raise ValueError("max_iterations 必须大于 0。")


def _validate_recommender_inferer_config(inferer_config: dict[str, Any]) -> None:
    """校验推荐参数反推器配置。"""
    int_fields = (
        "num_agents",
        "avg_degree",
        "min_scaled_target",
        "n_trials_per_story",
        "n_trials_per_weight",
        "n_simulations_per_trial",
    )
    for field_name in int_fields:
        value = inferer_config.get(field_name)
        if value is None:
            continue
        if int(value) <= 0:
            raise ValueError(f"{field_name} 必须大于 0。")

    verified_ratio = inferer_config.get("verified_ratio")
    if verified_ratio is not None and not 0 <= float(verified_ratio) <= 1:
        raise ValueError("verified_ratio 必须在 [0, 1] 区间内。")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """写入 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _resolve_output_path(path_value: str | None, default_relative_path: str) -> Path:
    """解析输出路径。"""
    if path_value:
        output_path = Path(path_value)
    else:
        output_path = Path(default_relative_path)
    return output_path.resolve()


def _resolve_portrait_reference_time(reference_time: str | None) -> tuple[int, str]:
    """解析画像生成使用的参考时间。"""
    raw_value = reference_time.strip() if isinstance(reference_time, str) else ""
    while not raw_value:
        raw_value = input(
            "请输入画像生成参考时间（例如 2026-04-01 12:00:00，默认按 Asia/Shanghai 解析）："
        ).strip()
    return _parse_reference_time_to_timestamp(raw_value), raw_value


def _record_failed_portrait_user(
    output_path: Path,
    user_name: str,
    reference_time_text: str,
    error_message: str,
) -> Path:
    """记录画像生成失败的用户名单。"""
    report_path = output_path.parent / "failed_users.json"
    payload: dict[str, Any] = {
        "failed_users": [],
        "failed_details": [],
    }
    if report_path.exists():
        try:
            loaded = _load_json_object(report_path)
            if isinstance(loaded.get("failed_users"), list):
                payload["failed_users"] = list(loaded["failed_users"])
            if isinstance(loaded.get("failed_details"), list):
                payload["failed_details"] = list(loaded["failed_details"])
        except Exception:
            payload = {
                "failed_users": [],
                "failed_details": [],
            }

    if user_name not in payload["failed_users"]:
        payload["failed_users"].append(user_name)

    payload["failed_details"] = [
        item
        for item in payload["failed_details"]
        if not isinstance(item, dict) or item.get("user_name") != user_name
    ]
    payload["failed_details"].append(
        {
            "user_name": user_name,
            "reference_time": reference_time_text,
            "error_message": error_message,
        }
    )
    _write_json(report_path, payload)
    return report_path


def _safe_int(value: Any) -> int:
    """安全解析整数。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_optional_int(value: Any) -> int | None:
    """安全解析可选整数。"""
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float:
    """安全解析浮点数。"""
    try:
        if value in (None, ""):
            return 0.0
        text = str(value).strip()
        if text.lower() == "nan":
            return 0.0
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def _normalize_scalar(value: Any) -> str:
    """归一化单元格文本。"""
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _parse_publish_time_to_timestamp(value: str) -> int:
    """尝试把可读时间解析成时间戳。"""
    text = value.strip()
    if not text:
        return 0
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d",
    ):
        try:
            return int(datetime.strptime(text, fmt).timestamp())
        except ValueError:
            continue
    return 0


def _parse_reference_time_to_timestamp(value: str) -> int:
    """解析画像生成参考时间。"""
    text = value.strip()
    if not text:
        raise ValueError("reference_time 不能为空。")
    if text.isdigit():
        timestamp = int(text)
        if timestamp > 0:
            return timestamp

    normalized = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
        return int(dt.timestamp())
    except ValueError:
        pass

    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d",
    ):
        try:
            dt = datetime.strptime(text, fmt).replace(tzinfo=ZoneInfo("Asia/Shanghai"))
            return int(dt.timestamp())
        except ValueError:
            continue
    raise ValueError(
        "reference_time 格式无效，请使用时间戳或类似 2026-04-01 12:00:00 的时间字符串。"
    )


def main() -> None:
    """脚本入口。"""
    try:
        from dotenv import load_dotenv

        _analysis_env = Path(__file__).resolve().parent / ".env"
        if _analysis_env.exists():
            load_dotenv(_analysis_env)
        else:
            load_dotenv()
    except ImportError:
        pass

    args = parse_args()
    if args.command == "portrait":
        if args.batch:
            asyncio.run(run_portrait_batch_command(args))
        elif not args.user_name:
            raise ValueError("请提供 --user-name，或使用 --batch 批量生成全部用户画像。")
        else:
            asyncio.run(run_portrait_command(args))
        return
    if args.command == "recommender":
        run_recommender_command(args)
        return
    if args.command == "retier":
        run_retier_command(args)
        return
    raise ValueError(f"不支持的命令：{args.command}")


if __name__ == "__main__":
    main()
