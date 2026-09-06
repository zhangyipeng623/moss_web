"""PHEME 谣言数据集 → 按事件分组转码（每组 user.xlsx + content_observations.json）。

PHEME 自带原始推文文本 + 转发时间戳 + 用户画像，直接下载即可（figshare），无需 Twitter API。

输出结构（--out 下按事件分组）：
    <out>/
    ├── charliehebdo/
    │   ├── user.xlsx                 # 该事件用户（含用户ID，供你爬取全量推文生成画像）
    │   └── content_observations.json # 该事件源推文 + repost_curve 时间链（校准+时序验证）
    ├── ferguson/ ...
    └── summary.json                  # 各事件统计

用法：
    python analysis/convert_pheme.py \
        --data-dir /path/to/all-rnr-annotated-threads \
        --out analysis_outputs/pheme \
        --max-users-per-event 300

字段映射（Twitter v1.1 → 系统列）：
    tweet.id_str          -> story_id
    tweet.text            -> text
    tweet.retweet_count   -> repost_count
    tweet.favorite_count  -> 点赞数
    reaction.created_at   -> 转发时间链（聚合为 repost_curve）
    user.id_str           -> 用户名
    user.screen_name      -> 昵称
    user.description      -> 简介
    user.location         -> 地域
    user.followers_count  -> 粉丝
    user.friends_count    -> 关注
    user.created_at       -> 创建时间戳
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    from openpyxl import Workbook
except ModuleNotFoundError:
    print("缺少 openpyxl，请先执行 `uv add openpyxl` 或 `uv sync`。")
    raise SystemExit(1)

_TWITTER_DT = "%a %b %d %H:%M:%S %z %Y"

USER_HEADER = [
    "用户名", "昵称", "简介", "性别", "地域", "关注", "粉丝", "收藏",
    "源用户名", "用户地址", "创建时间戳", "头像链接",
]


def _get(obj, *keys, default=None):
    for k in keys:
        if isinstance(obj, dict) and k in obj and obj[k] is not None:
            return obj[k]
    return default


def _to_ts(value) -> int:
    if not value:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(datetime.strptime(str(value), _TWITTER_DT).timestamp())
    except (ValueError, TypeError):
        return 0


def _tweet_id(t) -> str:
    return str(_get(t, "id_str", "id", default=""))


def _user_id(u) -> str:
    return str(_get(u, "id_str", "id", default=""))


def _load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


# ---------------------------------------------------------------
# 1. 解析 PHEME（按事件分组）
# ---------------------------------------------------------------
def parse_pheme(data_dir: Path) -> dict[str, dict]:
    """返回 {event: {"source_tweets": {}, "reactions": [], "users": {}}}。"""
    events: dict[str, dict] = {}

    for event_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        event = event_dir.name.replace("-all-rnr-threads", "")
        events[event] = {"source_tweets": {}, "reactions": [], "users": {}}
        st, reac, users = (events[event]["source_tweets"],
                           events[event]["reactions"],
                           events[event]["users"])

        for p in sorted(event_dir.rglob("*.json")):
            rel = str(p.relative_to(event_dir))
            if "source-tweets" not in rel and "reactions" not in rel:
                continue  # 跳过 annotation.json 等
            d = _load_json(p)
            if "id_str" not in d and "id" not in d:
                continue
            u = d.get("user")
            if isinstance(u, dict):
                uid = _user_id(u)
                if uid:
                    users.setdefault(uid, u)
            if "source-tweets" in rel:
                tid = _tweet_id(d)
                if tid:
                    st[tid] = d
            else:  # reactions
                reac.append(d)

    return events


# ---------------------------------------------------------------
# 2. 转发时间链 → repost_curve
# ---------------------------------------------------------------
def build_repost_curves(
    source_tweets: dict, reactions: list, num_bins: int = 24
) -> dict[str, list[float]]:
    ts_by_parent: dict[str, list[int]] = defaultdict(list)
    for r in reactions:
        parent = (
            _get(r, "in_reply_to_status_id_str", default="")
            or _tweet_id(r.get("retweeted_status") or {})
        )
        if not parent:
            continue
        ts = _to_ts(_get(r, "created_at", default=""))
        if ts > 0:
            ts_by_parent[str(parent)].append(ts)

    curves: dict[str, list[float]] = {}
    for tid, tss in ts_by_parent.items():
        if len(tss) < 2:
            continue
        tss_sorted = sorted(tss)
        t0, t1 = tss_sorted[0], tss_sorted[-1]
        span = max(t1 - t0, 1)
        bins = [0.0] * num_bins
        for ts in tss_sorted:
            idx = min(num_bins - 1, int((ts - t0) / span * num_bins))
            bins[idx] += 1.0
        curve, acc = [], 0.0
        for b in bins:
            acc += b
            curve.append(acc)
        curves[tid] = curve
    return curves


# ---------------------------------------------------------------
# 3. 按事件写 user.xlsx + content_observations.json
# ---------------------------------------------------------------
def write_event(events: dict, out_dir: Path, max_users: int, num_bins: int) -> None:
    summary = {}
    for event, data in events.items():
        ev_dir = out_dir / event
        ev_dir.mkdir(parents=True, exist_ok=True)
        source_tweets = data["source_tweets"]
        users = data["users"]
        reactions = data["reactions"]
        repost_curves = build_repost_curves(source_tweets, reactions, num_bins)

        # ---- user.xlsx：该事件用户（按粉丝数截断）----
        user_list = sorted(
            users.values(),
            key=lambda u: int(_get(u, "followers_count", default=0) or 0),
            reverse=True,
        )[:max_users]
        user_rows = []
        for u in user_list:
            uid = _user_id(u)
            user_rows.append([
                uid,
                str(_get(u, "screen_name", default="") or ""),
                str(_get(u, "description", default="") or ""),
                "",
                str(_get(u, "location", default="") or ""),
                int(_get(u, "friends_count", default=0) or 0),
                int(_get(u, "followers_count", default=0) or 0),
                int(_get(u, "listed_count", default=0) or 0),
                uid,
                str(_get(u, "location", default="") or ""),
                _to_ts(_get(u, "created_at", default="")),
                str(_get(u, "profile_image_url_https", "profile_image_url", default="") or ""),
            ])
        wb = Workbook()
        ws = wb.active
        ws.title = "user"
        ws.append(USER_HEADER)
        for r in user_rows:
            ws.append(r)
        wb.save(ev_dir / "user.xlsx")

        # ---- content_observations.json（含 repost_curve）----
        records = []
        for tid, t in source_tweets.items():
            retweet_count = int(_get(t, "retweet_count", default=0) or 0)
            favorite_count = int(_get(t, "favorite_count", default=0) or 0)
            reply_count = int(_get(t, "reply_count", default=0) or 0)
            rec = {
                "story_id": tid,
                "repost_count": retweet_count,
                "view_count": retweet_count + favorite_count + reply_count,
                "text": str(_get(t, "text", "full_text", default="") or ""),
            }
            if tid in repost_curves:
                rec["repost_curve"] = repost_curves[tid]
            records.append(rec)

        (ev_dir / "content_observations.json").write_text(
            json.dumps(
                {"records": records, "anchor_percentile": 0.8, "max_iterations": 3},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )

        with_curve = sum(1 for r in records if "repost_curve" in r)
        summary[event] = {
            "users": len(user_rows),
            "total_users_in_event": len(users),
            "source_tweets": len(source_tweets),
            "reactions": len(reactions),
            "observations": len(records),
            "with_repost_curve": with_curve,
        }
        print(f"  [{event}] 用户 {len(user_rows)}/{len(users)}，"
              f"源推文 {len(source_tweets)}，观测 {len(records)}，"
              f"带时间链 {with_curve}")

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PHEME → 按事件分组转码")
    parser.add_argument("--data-dir", required=True, help="all-rnr-annotated-threads 根目录")
    parser.add_argument("--out", default="data/pheme/processed", help="输出目录")
    parser.add_argument("--max-users-per-event", type=int, default=300,
                        help="每个事件画像用户数上限（按粉丝数截断）")
    parser.add_argument("--num-bins", type=int, default=24, help="repost_curve 分箱数")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"目录不存在：{data_dir}")

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"解析 {data_dir} …")
    events = parse_pheme(data_dir)
    print(f"事件数：{len(events)}\n")

    write_event(events, out_dir, args.max_users_per_event, args.num_bins)

    print(f"\n=== 完成，输出到 {out_dir} ===")


if __name__ == "__main__":
    main()
