"""weibo.sql（63641 用户新浪微博数据集）→ 画像 + 校准输入 转换脚本。

用法：
    python analysis/convert_weibo63641.py \
        --sql "/path/to/weibo.sql" \
        --out analysis_outputs/weibo_63641 \
        --max-users 100 --min-posts-per-user 5

输出（在 --out 目录下）：
    portrait_data/user.xlsx      画像用户表（9 必需列 + 源用户名/用户地址/头像链接）
    portrait_data/post.xlsx      画像帖子表（5 必需列 + 点赞数/评论数/转发数/帖子ID）
    content_observations.json    推荐参数校准输入（records: story_id/repost_count/view_count/text/topic）
    convert_report.json          统计信息（行数、主题分布、抽样详情）

字段映射：
    user.uid               -> 用户名
    user.screen_name       -> 昵称
    user.name              -> 简介（兜底，name 常与 screen_name 相同）
    user.gender            -> 性别（f/m -> 女/男）
    user.location          -> 地域
    user.friendsnum        -> 关注
    user.followersnum      -> 粉丝
    user.favouritesnum     -> 收藏
    user.created_at        -> 创建时间戳（Asia/Shanghai 转 epoch）
    weibo.mid              -> 帖子ID / story_id
    weibo.text             -> 发文内容 / text
    weibo.date             -> 发布时间 / 发布时间戳
    weibo.repostsnum       -> 转发数 / repost_count
    weibo.commentsnum      -> 评论数
    weibo.attitudesnum     -> 点赞数
    weibo.topic            -> 主题（12 个话题之一）
    weiborelation.smid     -> 该微博为“转发”（smid 转发 tmid）
    view_count             -> repostsnum + commentsnum + attitudesnum（合成曝光量）

注：本脚本只做“列名/格式对齐”，不改动主流程任何逻辑；画像与校准仍需走
`analysis.run_analysis portrait` / `recommender` 两条既有命令。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

try:
    from openpyxl import Workbook
except ModuleNotFoundError:
    print("缺少 openpyxl，请先执行 `uv add openpyxl` 或 `uv sync`。", file=sys.stderr)
    sys.exit(1)


# 亚洲/上海时区偏移（秒），用于把 datetime 字符串转成 epoch 时间戳。
_CN_OFFSET = 8 * 3600

_INSERT_RE = re.compile(r"INSERT INTO `(?P<table>\w+)` VALUES\s*(?P<values>\(.*\));?\s*$")


# ---------------------------------------------------------------
# 1. MySQL dump 逐行流式解析（132MB，不做整文件读入）
# ---------------------------------------------------------------
def _split_mysql_fields(raw: str) -> list[str]:
    """把一条 INSERT 的 VALUES 括号内容按顶层逗号切分，正确处理单引号字符串与反斜杠转义。"""
    # 去掉最外层括号
    inner = raw.strip()
    if inner.startswith("("):
        inner = inner[1:]
    if inner.endswith(")"):
        inner = inner[:-1]

    fields: list[str] = []
    buf: list[str] = []
    in_str = False
    i, n = 0, len(inner)
    while i < n:
        ch = inner[i]
        if in_str:
            if ch == "\\" and i + 1 < n:
                buf.append(inner[i + 1])
                i += 2
                continue
            if ch == "'":
                in_str = False
                i += 1
                continue
            buf.append(ch)
            i += 1
            continue
        # 非字符串状态
        if ch == "'":
            in_str = True
            i += 1
            continue
        if ch == ",":
            fields.append("".join(buf).strip())
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    fields.append("".join(buf).strip())
    return fields


def _to_int(value: str, default: int = 0) -> int:
    value = value.strip()
    if value.startswith("'") and value.endswith("'"):
        value = value[1:-1]
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_timestamp(value: str) -> int:
    value = value.strip()
    if value.startswith("'") and value.endswith("'"):
        value = value[1:-1]
    try:
        dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return 0
    return int((dt - datetime(1970, 1, 1)).total_seconds()) - _CN_OFFSET


def _gender(value: str) -> str:
    value = value.strip().lower()
    if value == "f":
        return "女"
    if value == "m":
        return "男"
    return value


# 各表字段顺序（与 SQL CREATE TABLE 一致）
USER_FIELDS = [
    "uid", "screen_name", "name", "province", "city", "location", "url",
    "gender", "followersnum", "friendsnum", "statusesnum", "favouritesnum", "created_at",
]
WEIBO_FIELDS = [
    "mid", "date", "text", "source", "repostsnum", "commentsnum",
    "attitudesnum", "uid", "topic",
]
USERRELATION_FIELDS = ["suid", "tuid"]
WEIBORELATION_FIELDS = ["smid", "tmid"]


def parse_sql(path: Path) -> tuple[list[dict], list[dict], set, set]:
    """流式解析 weibo.sql，返回 (users, weibos, repost_smids, retweet_pairs)。

    - users: 63641 条用户记录
    - weibos: 84168 条微博记录
    - repost_smids: 作为“转发”方的微博 mid 集合（用于推导发文类型）
    - retweet_pairs: (smid, tmid) 转发关系（保留给后续 repost_curve）
    """
    users: list[dict] = []
    weibos: list[dict] = []
    repost_smids: set[str] = set()
    retweet_pairs: list[tuple[str, str]] = []

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            m = _INSERT_RE.match(line)
            if not m:
                continue
            table = m.group("table")
            fields = _split_mysql_fields(m.group("values"))

            if table == "user":
                if len(fields) < len(USER_FIELDS):
                    continue
                rec = dict(zip(USER_FIELDS, fields))
                users.append(rec)
            elif table == "weibo":
                if len(fields) < len(WEIBO_FIELDS):
                    continue
                rec = dict(zip(WEIBO_FIELDS, fields))
                weibos.append(rec)
            elif table == "userrelation":
                if len(fields) >= 2:
                    pass  # 好友关系本课题暂未消费，保留不落地
            elif table == "weiborelation":
                if len(fields) >= 2:
                    smid, tmid = fields[0].strip(), fields[1].strip()
                    repost_smids.add(smid)
                    retweet_pairs.append((smid, tmid))

            if lineno % 200000 == 0:
                print(f"  [parse] 已读 {lineno} 行 …", flush=True)

    return users, weibos, repost_smids, retweet_pairs


# ---------------------------------------------------------------
# 2. 用户抽样（画像只对少量种子用户生成，63641 全量跑 LLM 不现实）
# ---------------------------------------------------------------
def sample_users(
    users: list[dict],
    weibos: list[dict],
    max_users: int,
    seed: int,
    min_posts: int = 5,
) -> list[dict]:
    """抽取 max_users 个用户：优先保留高影响力，其余按粉丝数均匀铺开。

    min_posts：每个被抽用户至少要发过 min_posts 条微博（保证画像证据充足）。
    本数据集是“话题爬取”而非“用户全量历史”，人均仅 1.3 条帖子，故默认按
    min_posts=5 过滤，宁可少抽几个用户也要保证单个画像的帖子证据量。
    """
    author_posts: dict[str, int] = {}
    for w in weibos:
        uid = w["uid"].strip()
        author_posts[uid] = author_posts.get(uid, 0) + 1

    eligible = [
        u for u in users
        if author_posts.get(u["uid"].strip(), 0) >= min_posts
    ]
    if len(eligible) <= max_users:
        return eligible

    def _followers(u: dict) -> int:
        return _to_int(u.get("followersnum", "0"))

    eligible_sorted = sorted(eligible, key=_followers, reverse=True)
    n_elite = max(1, int(max_users * 0.2))  # 20% 高影响力精英
    elite = eligible_sorted[:n_elite]
    rest = eligible_sorted[n_elite:]

    if len(rest) >= max_users - n_elite:
        # 在剩余用户里均匀取索引，保留“低粉丝→高粉丝”的完整梯度
        step = (len(rest) - 1) / max(1, max_users - n_elite - 1)
        idx = [int(round(i * step)) for i in range(max_users - n_elite)]
        crowd = [rest[i] for i in idx]
    else:
        crowd = rest

    rng = random.Random(seed)
    picked = {u["uid"].strip() for u in elite}
    picked.update(u["uid"].strip() for u in crowd)
    # 不足时用随机用户补齐
    if len(picked) < max_users:
        remaining = [u for u in eligible_sorted if u["uid"].strip() not in picked]
        rng.shuffle(remaining)
        for u in remaining:
            if len(picked) >= max_users:
                break
            picked.add(u["uid"].strip())

    return [u for u in eligible_sorted if u["uid"].strip() in picked]


# ---------------------------------------------------------------
# 3. 输出三份文件
# ---------------------------------------------------------------
def write_portrait_tables(
    sampled_users: list[dict], weibos: list[dict], repost_smids: set[str], out_dir: Path
) -> tuple[int, int]:
    data_dir = out_dir / "portrait_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    sampled_uids = {u["uid"].strip() for u in sampled_users}

    # user.xlsx
    user_header = [
        "用户名", "昵称", "简介", "性别", "地域", "关注", "粉丝", "收藏",
        "源用户名", "用户地址", "创建时间戳", "头像链接",
    ]
    user_rows: list[list] = []
    for u in sampled_users:
        uid = u["uid"].strip()
        user_rows.append([
            uid,
            u["screen_name"].strip(),
            u["name"].strip(),
            _gender(u["gender"]),
            u["location"].strip(),
            _to_int(u["friendsnum"]),
            _to_int(u["followersnum"]),
            _to_int(u["favouritesnum"]),
            uid,  # 源用户名
            u["location"].strip(),  # 用户地址
            _to_timestamp(u["created_at"]),
            "",  # 头像链接（数据集无，置空）
        ])

    wb = Workbook()
    ws = wb.active
    ws.title = "user"
    ws.append(user_header)
    for row in user_rows:
        ws.append(row)
    wb.save(data_dir / "user.xlsx")

    # post.xlsx（只保留抽样用户发布的微博）
    post_header = [
        "用户名", "发文内容", "发布时间", "发布时间戳", "发文类型",
        "点赞数", "评论数", "转发数", "帖子ID",
    ]
    post_rows: list[list] = []
    for w in weibos:
        uid = w["uid"].strip()
        if uid not in sampled_uids:
            continue
        mid = w["mid"].strip()
        post_rows.append([
            uid,
            w["text"].strip(),
            w["date"].strip(),
            _to_timestamp(w["date"]),
            "转发" if mid in repost_smids else "原创",
            _to_int(w["attitudesnum"]),
            _to_int(w["commentsnum"]),
            _to_int(w["repostsnum"]),
            mid,
        ])

    wb = Workbook()
    ws = wb.active
    ws.title = "post"
    ws.append(post_header)
    for row in post_rows:
        ws.append(row)
    wb.save(data_dir / "post.xlsx")

    return len(user_rows), len(post_rows)


def write_content_observations(
    weibos: list[dict], out_dir: Path, min_repost: int = 1
) -> int:
    records: list[dict] = []
    for w in weibos:
        repost = _to_int(w["repostsnum"])
        comment = _to_int(w["commentsnum"])
        attitude = _to_int(w["attitudesnum"])
        if repost < min_repost:
            continue
        records.append({
            "story_id": w["mid"].strip(),
            "repost_count": repost,
            "view_count": repost + comment + attitude,  # 合成曝光量
            "text": w["text"].strip(),
            "topic": w["topic"].strip(),
        })

    payload = {
        "records": records,
        "anchor_percentile": 0.8,
        "max_iterations": 3,
    }
    (out_dir / "content_observations.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="weibo.sql → 画像 + 校准输入转换")
    parser.add_argument("--sql", required=True, help="weibo.sql 文件路径")
    parser.add_argument("--out", default="data/weibo_63641", help="输出目录")
    parser.add_argument("--max-users", type=int, default=100, help="画像抽样用户数")
    parser.add_argument("--min-posts-per-user", type=int, default=5,
                        help="每个被抽用户至少发过的微博数（保证画像证据量）")
    parser.add_argument("--min-repost", type=int, default=1, help="内容观测的最小转发数")
    parser.add_argument("--seed", type=int, default=42, help="抽样随机种子")
    args = parser.parse_args()

    sql_path = Path(args.sql).expanduser().resolve()
    if not sql_path.exists():
        raise SystemExit(f"找不到 SQL 文件：{sql_path}")

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"解析 {sql_path.name} …")
    users, weibos, repost_smids, retweet_pairs = parse_sql(sql_path)
    print(f"  user 表：{len(users)} 条")
    print(f"  weibo 表：{len(weibos)} 条")
    print(f"  weiborelation 转发关系：{len(retweet_pairs)} 条")

    sampled = sample_users(
        users, weibos, args.max_users, args.seed, min_posts=args.min_posts_per_user
    )

    # 抽样用户的帖子数分布（用于确认画像证据是否充足）
    sampled_uids = {u["uid"].strip() for u in sampled}
    sampled_post_counts = sorted(
        (sum(1 for w in weibos if w["uid"].strip() == uid) for uid in sampled_uids),
        reverse=True,
    )
    print(f"抽样用户：{len(sampled)} 个（--max-users={args.max_users}，"
          f"--min-posts-per-user={args.min_posts_per_user}）")
    if sampled_post_counts:
        print(f"  抽样用户帖子数：中位={sampled_post_counts[len(sampled_post_counts)//2]} "
              f"均值={sum(sampled_post_counts)/len(sampled_post_counts):.1f} "
              f"max={sampled_post_counts[0]}")

    n_users, n_posts = write_portrait_tables(sampled, weibos, repost_smids, out_dir)
    n_obs = write_content_observations(weibos, out_dir, min_repost=args.min_repost)

    # 主题分布（内容观测）
    topic_counts: dict[str, int] = {}
    for w in weibos:
        if _to_int(w["repostsnum"]) >= args.min_repost:
            topic_counts[w["topic"].strip()] = topic_counts.get(w["topic"].strip(), 0) + 1

    report = {
        "source_sql": str(sql_path),
        "user_count": len(users),
        "weibo_count": len(weibos),
        "retweet_relation_count": len(retweet_pairs),
        "sampled_user_count": len(sampled),
        "min_posts_per_user": args.min_posts_per_user,
        "sampled_post_median": sampled_post_counts[len(sampled_post_counts)//2] if sampled_post_counts else 0,
        "sampled_post_max": sampled_post_counts[0] if sampled_post_counts else 0,
        "portrait_post_count": n_posts,
        "content_observation_count": n_obs,
        "topic_distribution": topic_counts,
        "seed": args.seed,
        "note": "参考时间建议设为 2014-05-12 00:00:00（数据覆盖 2014-04-30 ~ 2014-05-11）",
    }
    (out_dir / "convert_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== 转换完成 ===")
    print(f"  {out_dir / 'portrait_data' / 'user.xlsx'}  ({n_users} 行)")
    print(f"  {out_dir / 'portrait_data' / 'post.xlsx'}  ({n_posts} 行)")
    print(f"  {out_dir / 'content_observations.json'}  ({n_obs} 条观测)")
    print(f"  {out_dir / 'convert_report.json'}")
    print("\n后续命令：")
    print(f"  uv run python -m analysis.run_analysis portrait "
          f"--data-path {out_dir / 'portrait_data'} --batch --reference-time \"2014-05-12 00:00:00\"")
    print(f"  uv run python -m analysis.run_analysis recommender "
          f"--input {out_dir / 'content_observations.json'} "
          f"--portraits-dir analysis_outputs/portraits/")


if __name__ == "__main__":
    main()
