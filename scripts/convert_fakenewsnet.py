"""FakeNewsNet → 画像 + 校准 + 时序验证 输入转码脚本。

把下载好的 FakeNewsNet 数据（按 news_id 分文件夹）转成系统可消费的三份文件：

    portrait_data/user.xlsx          用户画像表（Twitter 用户画像 → 9 必需列）
    portrait_data/post.xlsx          帖子表（推文 → 发文内容/转发数/点赞数/帖子ID）
    content_observations.json        校准观测（story_id/repost_count/view_count/text/repost_curve）

用法：
    python analysis/convert_fakenewsnet.py \
        --data-dir data/fakenewsnet/PolitiFact \
        --out analysis_outputs/fakenewsnet \
        --max-users 500

输入的目录结构（每个 news_id 一个文件夹）：
    <data-dir>/
    └── <news_id>/
        ├── news content.json         新闻正文（可选，取 title/text 作为主题锚点）
        ├── tweets/<tweet_id>.json    推文（Twitter v1.1 原始对象）
        ├── retweets/<retweet_id>.json  转发（含 created_at 时间戳 + retweeted_status）
        └── user_profiles/<user_id>.json 用户画像

字段映射（Twitter → 系统列）：
    user.id_str          -> 用户名
    user.screen_name     -> 昵称
    user.description     -> 简介
    user.location        -> 地域 / 用户地址
    user.friends_count   -> 关注
    user.followers_count -> 粉丝
    user.listed_count    -> 收藏（代理）
    user.created_at      -> 创建时间戳
    tweet.full_text      -> 发文内容 / text
    tweet.retweet_count  -> 转发数 / repost_count
    tweet.favorite_count -> 点赞数
    tweet.created_at     -> 发布时间戳
    retweet.created_at   -> 转发时间链（聚合为 repost_curve）
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

# Twitter created_at 格式："Sun Dec 03 10:00:00 +0000 2017"
_TWITTER_DT = "%a %b %d %H:%M:%S %z %Y"


def _get(obj: dict, *keys, default=None):
    """按优先级取字段，兼容 id / id_str 这类命名差异。"""
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


def _iter_json_files(path: Path):
    """递归列出目录下所有 .json 文件。"""
    if not path.is_dir():
        return
    for p in sorted(path.rglob("*.json")):
        yield p


def _load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _tweet_id(t: dict) -> str:
    return str(_get(t, "id_str", "id", default=""))


def _user_id(u: dict) -> str:
    return str(_get(u, "id_str", "id", default=""))


# ---------------------------------------------------------------
# 1. 解析 FakeNewsNet 目录
# ---------------------------------------------------------------
def parse_fakenewsnet(data_dir: Path) -> dict:
    """遍历 data_dir 下每个 news_id 文件夹，聚合 tweets/retweets/user_profiles。"""
    tweets: dict[str, dict] = {}          # tweet_id -> tweet 对象
    retweets: list[dict] = []             # 所有转发对象（含时间戳）
    users: dict[str, dict] = {}           # user_id -> user 对象
    news_meta: dict[str, dict] = {}       # tweet_id -> {title, news_id}（主题锚点）

    for news_folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        news_id = news_folder.name
        # 新闻正文（可选）
        title = ""
        for p in _iter_json_files(news_folder):
            if "tweets" in str(p.relative_to(news_folder)).split("/")[0]:
                continue
        # 直接找 news content 文件
        for cand in ("news content.json", "news_content.json", "news.json"):
            nc = news_folder / cand
            if nc.is_file():
                d = _load_json(nc)
                title = str(_get(d, "title", default="") or "")
                break

        # tweets
        tweets_dir = news_folder / "tweets"
        for p in _iter_json_files(tweets_dir):
            t = _load_json(p)
            tid = _tweet_id(t)
            if not tid:
                continue
            tweets[tid] = t
            news_meta[tid] = {"news_id": news_id, "title": title}
            # 推文内嵌的 user 对象也并入用户表
            u = t.get("user")
            if isinstance(u, dict):
                uid = _user_id(u)
                if uid:
                    users.setdefault(uid, u)

        # retweets（可能平铺，也可能按 tweet 分文件夹）
        retweets_dir = news_folder / "retweets"
        for p in _iter_json_files(retweets_dir):
            r = _load_json(p)
            if r:
                retweets.append(r)

        # user_profiles
        profiles_dir = news_folder / "user_profiles"
        for p in _iter_json_files(profiles_dir):
            u = _load_json(p)
            uid = _user_id(u)
            if uid:
                users.setdefault(uid, u)

    return {
        "tweets": tweets,
        "retweets": retweets,
        "users": users,
        "news_meta": news_meta,
    }


# ---------------------------------------------------------------
# 2. 转发时间链 → repost_curve
# ---------------------------------------------------------------
def build_repost_curves(
    tweets: dict[str, dict], retweets: list[dict], num_bins: int = 24
) -> dict[str, list[float]]:
    """按父推文聚合转发时间戳，产出累计转发量曲线（Sequence[float]）。

    返回 {tweet_id: [cum_count_0, cum_count_1, ...]}，与 ABM 的 history
    同为“累计计数”，trajectory_loss 会归一化后比较形状。
    """
    # 把每条转发归到其父推文（retweeted_status.id_str）
    ts_by_parent: dict[str, list[int]] = defaultdict(list)
    for r in retweets:
        parent = r.get("retweeted_status") or {}
        pid = _tweet_id(parent)
        if not pid:
            continue
        ts = _to_ts(_get(r, "created_at", default=""))
        if ts > 0:
            ts_by_parent[pid].append(ts)

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
        # 累计
        curve = []
        acc = 0.0
        for b in bins:
            acc += b
            curve.append(acc)
        curves[tid] = curve
    return curves


# ---------------------------------------------------------------
# 3. 输出 user.xlsx / post.xlsx / content_observations.json
# ---------------------------------------------------------------
def write_outputs(
    parsed: dict, out_dir: Path, max_users: int, num_bins: int
) -> None:
    tweets = parsed["tweets"]
    users = parsed["users"]
    news_meta = parsed["news_meta"]
    repost_curves = build_repost_curves(tweets, parsed["retweets"], num_bins=num_bins)

    data_dir = out_dir / "portrait_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # ---- user.xlsx（按粉丝数截断 max_users 个核心用户）----
    user_list = sorted(
        users.values(),
        key=lambda u: int(_get(u, "followers_count", default=0) or 0),
        reverse=True,
    )[:max_users]

    user_header = [
        "用户名", "昵称", "简介", "性别", "地域", "关注", "粉丝", "收藏",
        "源用户名", "用户地址", "创建时间戳", "头像链接",
    ]
    user_rows = []
    for u in user_list:
        uid = _user_id(u)
        user_rows.append([
            uid,
            str(_get(u, "screen_name", default="") or ""),
            str(_get(u, "description", default="") or ""),
            "",  # Twitter 无性别
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
    ws.append(user_header)
    for row in user_rows:
        ws.append(row)
    wb.save(data_dir / "user.xlsx")

    # ---- post.xlsx（每条推文 = 一条帖子）----
    post_header = [
        "用户名", "发文内容", "发布时间", "发布时间戳", "发文类型",
        "点赞数", "评论数", "转发数", "帖子ID",
    ]
    post_rows = []
    for tid, t in tweets.items():
        uid = _user_id(t.get("user") or {})
        post_rows.append([
            uid,
            str(_get(t, "full_text", "text", default="") or ""),
            str(_get(t, "created_at", default="") or ""),
            _to_ts(_get(t, "created_at", default="")),
            "转发" if t.get("retweeted_status") else "原创",
            int(_get(t, "favorite_count", default=0) or 0),
            int(_get(t, "reply_count", default=0) or 0),
            int(_get(t, "retweet_count", default=0) or 0),
            tid,
        ])

    wb = Workbook()
    ws = wb.active
    ws.title = "post"
    ws.append(post_header)
    for row in post_rows:
        ws.append(row)
    wb.save(data_dir / "post.xlsx")

    # ---- content_observations.json（含 repost_curve）----
    records = []
    for tid, t in tweets.items():
        retweet_count = int(_get(t, "retweet_count", default=0) or 0)
        favorite_count = int(_get(t, "favorite_count", default=0) or 0)
        reply_count = int(_get(t, "reply_count", default=0) or 0)
        meta = news_meta.get(tid, {})
        rec = {
            "story_id": tid,
            "repost_count": retweet_count,
            "view_count": retweet_count + favorite_count + reply_count,  # 合成曝光量
            "text": str(_get(t, "full_text", "text", default="") or ""),
            "topic": meta.get("title", ""),
            "news_id": meta.get("news_id", ""),
        }
        if tid in repost_curves:
            rec["repost_curve"] = repost_curves[tid]
        records.append(rec)

    payload = {
        "records": records,
        "anchor_percentile": 0.8,
        "max_iterations": 3,
    }
    (out_dir / "content_observations.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 统计
    with_curve = sum(1 for r in records if "repost_curve" in r)
    print(f"  user.xlsx：{len(user_rows)} 个用户（--max-users={max_users}）")
    print(f"  post.xlsx：{len(post_rows)} 条推文")
    print(f"  content_observations.json：{len(records)} 条观测，"
          f"其中 {with_curve} 条带 repost_curve 时间链")


def main() -> None:
    parser = argparse.ArgumentParser(description="FakeNewsNet → 画像/校准/时序输入转码")
    parser.add_argument("--data-dir", required=True, help="FakeNewsNet 根目录（含各 news_id 文件夹）")
    parser.add_argument("--out", default="data/fakenewsnet", help="输出目录")
    parser.add_argument("--max-users", type=int, default=500, help="画像用户数上限（按粉丝数截断）")
    parser.add_argument("--num-bins", type=int, default=24, help="repost_curve 时间分箱数")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"目录不存在：{data_dir}")

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"解析 {data_dir} …")
    parsed = parse_fakenewsnet(data_dir)
    print(f"  tweets：{len(parsed['tweets'])} 条")
    print(f"  retweets：{len(parsed['retweets'])} 条")
    print(f"  users：{len(parsed['users'])} 个")

    write_outputs(parsed, out_dir, args.max_users, args.num_bins)

    print("\n=== 转码完成 ===")
    print(f"  {out_dir / 'portrait_data' / 'user.xlsx'}")
    print(f"  {out_dir / 'portrait_data' / 'post.xlsx'}")
    print(f"  {out_dir / 'content_observations.json'}")
    print("\n后续命令：")
    print(f"  uv run python -m analysis.run_analysis portrait "
          f"--data-path {out_dir / 'portrait_data'} --batch --reference-time \"2018-06-01 00:00:00\"")
    print(f"  uv run python -m analysis.run_analysis recommender "
          f"--input {out_dir / 'content_observations.json'} "
          f"--portraits-dir analysis_outputs/portraits/")


if __name__ == "__main__":
    main()
