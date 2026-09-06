"""prepare_recommender_data CLI：把原始观测表/JSON 准备成固定数据包。

脚本直接运行时显式将仓库根加入导入路径，保证不依赖安装即可 import analysis。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.recommender_data import prepare_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="准备推荐校准数据包（train.json / test.json / manifest.json）"
    )
    parser.add_argument("--data-file", required=True, help="原始观测表路径（xlsx/csv）或 records JSON")
    parser.add_argument("--output-dir", required=True, help="输出数据包目录")
    parser.add_argument("--num-agents", type=int, required=True, help="ABM 人口规模")
    parser.add_argument("--text-column", default=None, help="正文列名（JSON 输入省略）")
    parser.add_argument("--retweet-columns", default="转发,分享,Quotes", help="总转发量列名，逗号分隔")
    parser.add_argument("--view-column", default="观看量", help="观看量列名")
    parser.add_argument("--id-column", default="文章ID", help="内容 ID 列名")
    parser.add_argument("--test-ratio", type=float, default=0.3, help="测试分区比例，严格在 0 与 1 之间")
    parser.add_argument("--random-seed", type=int, default=42, help="划分随机种子")
    parser.add_argument("--anchor-percentile", type=float, default=0.8, help="缩放锚点分位数")
    parser.add_argument("--min-scaled-target", type=int, default=5, help="训练最小缩放目标阈值")
    parser.add_argument(
        "--selection",
        default="all",
        choices=["all", "stratified"],
        help="训练选择策略：all 保留全部有效记录，stratified 十档抽样",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = prepare_dataset(
        Path(args.data_file),
        Path(args.output_dir),
        num_agents=args.num_agents,
        text_column=args.text_column,
        retweet_columns=args.retweet_columns,
        view_column=args.view_column,
        id_column=args.id_column,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed,
        anchor_percentile=args.anchor_percentile,
        min_scaled_target=args.min_scaled_target,
        selection=args.selection,
    )
    print(f"数据包已生成: {manifest}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 - 最外层 CLI 统一记录并退出
        print(f"[prepare_recommender_data] 错误: {exc}", file=sys.stderr)
        sys.exit(1)
