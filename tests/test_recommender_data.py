"""Task 1 测试：可复用的推荐校准数据准备包。

覆盖：固定划分先于过滤、测试结果不改变训练、正文保真、校验失败、
原子发布与数据包加载校验。所有生成数据仅写临时目录。
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from analysis.recommender_data import (
    file_sha256,
    load_split,
    prepare_dataset,
    publish_output,
)


def _make_table(tmp: Path, rows: list[dict], name: str = "posts.csv") -> Path:
    """写一个 CSV 表；rows 每项是一行 dict。"""
    path = tmp / name
    cols = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _make_xlsx(tmp: Path, rows: list[dict], name: str = "posts.xlsx") -> Path:
    from openpyxl import Workbook

    path = tmp / name
    wb = Workbook()
    ws = wb.active
    cols = list(rows[0].keys())
    ws.append(cols)
    for row in rows:
        ws.append([row.get(c, "") for c in cols])
    wb.save(path)
    return path


def _make_json(tmp: Path, records: list[dict], name: str = "posts.json") -> Path:
    path = tmp / name
    path.write_text(json.dumps({"records": records}, ensure_ascii=False), encoding="utf-8")
    return path


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_all_records(out: Path) -> dict:
    """读取 train + test 两分区，合并为 story_id -> record 映射。"""
    by_id: dict = {}
    for split in ("train", "test"):
        data = _read_json(out / f"{split}.json")
        for r in data["records"]:
            by_id[r["story_id"]] = r
    return by_id


def _many_rows(n: int = 24) -> list[dict]:
    """n 条正常记录 + 3 条零转发（低传播）+ 2 条零浏览（应整体排除）。"""
    rows = []
    for i in range(n):
        rows.append(
            {
                "文章ID": f"id{i:02d}",
                "正文": f"正文内容 {i}",
                "观看量": 200 + i * 10,
                "转发": i + 1,
                "分享": 0,
                "Quotes": 0,
            }
        )
    for j in range(3):
        rows.append(
            {
                "文章ID": f"low{j}",
                "正文": f"低传播 {j}",
                "观看量": 300 + j,
                "转发": 0,
                "分享": 0,
                "Quotes": 0,
            }
        )
    rows.append({"文章ID": "zero_a", "正文": "零浏览 A", "观看量": 0, "转发": 5, "分享": 0, "Quotes": 0})
    rows.append({"文章ID": "zero_b", "正文": "零浏览 B", "观看量": 0, "转发": 5, "分享": 0, "Quotes": 0})
    return rows


class PrepareDatasetSplitTests(unittest.TestCase):
    """划分先于过滤 / 无泄漏 / 抽样。"""

    def test_split_before_filter(self) -> None:
        """测试分区保留零转发记录，训练分区过滤掉它们；零浏览整体排除。"""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = _make_table(tmp, _many_rows())
            out = tmp / "out"
            m = prepare_dataset(
                src, out, num_agents=1500, text_column="正文",
                retweet_columns="转发,分享,Quotes", view_column="观看量", id_column="文章ID",
                random_seed=42,
            )
            manifest = _read_json(m)
            train = _read_json(out / "train.json")
            test = _read_json(out / "test.json")
            train_ids = [r["story_id"] for r in train["records"]]
            test_ids = [r["story_id"] for r in test["records"]]
            # 测试分区完整保留所有测试记录（含零转发），不应用训练筛选
            self.assertEqual(sorted(test_ids), sorted(manifest["test_ids"]))
            # 训练分区不包含任何零转发记录（被过滤）
            for r in train["records"]:
                self.assertGreater(r["repost_count"], 0)
                self.assertGreater(r["view_count"], 100)
                self.assertGreaterEqual(r["scaled_target"], 5)
            # 零浏览记录整体排除，不出现在任何分区
            for zero_id in ("zero_a", "zero_b"):
                self.assertNotIn(zero_id, train_ids)
                self.assertNotIn(zero_id, test_ids)
            self.assertEqual(manifest["cleaning_counts"]["zero_view_excluded"], 2)

    def test_test_target_does_not_change_train(self) -> None:
        """改动测试分区目标后重跑，训练文件字节与锚点不变。"""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = _make_table(tmp, _many_rows())
            out1 = tmp / "out1"
            m1 = prepare_dataset(
                src, out1, num_agents=1500, text_column="正文",
                retweet_columns="转发,分享,Quotes", view_column="观看量", id_column="文章ID",
                random_seed=42,
            )
            manifest1 = _read_json(m1)
            train1_bytes = (out1 / "train.json").read_bytes()
            test_ids = set(manifest1["test_ids"])
            # 修改源数据：只把测试分区记录的转发量翻倍
            rows = _many_rows()
            for row in rows:
                if row["文章ID"] in test_ids:
                    row["转发"] = row["转发"] * 2 + 1
            src2 = _make_table(tmp, rows, name="posts2.csv")
            out2 = tmp / "out2"
            m2 = prepare_dataset(
                src2, out2, num_agents=1500, text_column="正文",
                retweet_columns="转发,分享,Quotes", view_column="观看量", id_column="文章ID",
                random_seed=42,
            )
            manifest2 = _read_json(m2)
            self.assertEqual(train1_bytes, (out2 / "train.json").read_bytes())
            self.assertEqual(manifest1["anchor"], manifest2["anchor"])
            self.assertEqual(manifest1["scale_ratio"], manifest2["scale_ratio"])
            self.assertEqual(manifest1["train_ids"], manifest2["train_ids"])

    def test_selection_all_preserves_eligible_rows(self) -> None:
        """默认 all 不抽样：训练分区保留所有达过滤条件的记录。"""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = _make_table(tmp, _many_rows(20))
            out = tmp / "out"
            m = prepare_dataset(
                src, out, num_agents=1500, text_column="正文",
                retweet_columns="转发,分享,Quotes", view_column="观看量", id_column="文章ID",
                random_seed=42, selection="all",
            )
            manifest = _read_json(m)
            train = _read_json(out / "train.json")
            # all 模式：训练分区记录数等于训练 ID 数（不过滤已由筛选保证，这里不抽样）
            self.assertEqual(len(train["records"]), manifest["train_count"])
            self.assertEqual(len(train["records"]), len(manifest["train_ids"]))

    def test_stratified_reproducible(self) -> None:
        """stratified 抽样固定 seed 可复现，且样本不超过有效记录数。"""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = _make_table(tmp, _many_rows(40))
            out1 = tmp / "out1"
            out2 = tmp / "out2"
            m1 = prepare_dataset(
                src, out1, num_agents=1500, text_column="正文",
                retweet_columns="转发,分享,Quotes", view_column="观看量", id_column="文章ID",
                random_seed=42, selection="stratified",
            )
            m2 = prepare_dataset(
                src, out2, num_agents=1500, text_column="正文",
                retweet_columns="转发,分享,Quotes", view_column="观看量", id_column="文章ID",
                random_seed=42, selection="stratified",
            )
            self.assertEqual((out1 / "train.json").read_bytes(), (out2 / "train.json").read_bytes())
            manifest = _read_json(m1)
            train = _read_json(out1 / "train.json")
            self.assertLessEqual(len(train["records"]), len(manifest["train_ids"]))
            # 抽样结果全部是有效训练候选（非零转发）
            for r in train["records"]:
                self.assertGreater(r["repost_count"], 0)


class PrepareDatasetFidelityTests(unittest.TestCase):
    """正文保真与输入校验。"""

    def test_csv_text_fidelity(self) -> None:
        text1 = "中文,逗号 和 " + chr(34) + "引号" + chr(34) + "\n换行"
        text2 = "English text with =SUM(1) prefix"
        rows = [
            {"文章ID": "id1", "正文": text1, "观看量": 500, "转发": 10, "分享": 0, "Quotes": 0},
            {"文章ID": "id2", "正文": text2, "观看量": 600, "转发": 12, "分享": 0, "Quotes": 0},
        ]
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = _make_table(tmp, rows)
            out = tmp / "out"
            prepare_dataset(
                src, out, num_agents=1000, text_column="正文",
                retweet_columns="转发,分享,Quotes", view_column="观看量", id_column="文章ID",
                random_seed=42,
            )
            by_id = _load_all_records(out)
            self.assertEqual(by_id["id1"]["text"], text1)
            self.assertEqual(by_id["id2"]["text"], text2)

    def test_xlsx_text_fidelity(self) -> None:
        rows = [
            {"文章ID": "id1", "正文": "中文正文 你好", "观看量": 500, "转发": 10, "分享": 0, "Quotes": 0},
            {"文章ID": "id2", "正文": "English body", "观看量": 600, "转发": 12, "分享": 0, "Quotes": 0},
        ]
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = _make_xlsx(tmp, rows)
            out = tmp / "out"
            prepare_dataset(
                src, out, num_agents=1000, text_column="正文",
                retweet_columns="转发,分享,Quotes", view_column="观看量", id_column="文章ID",
                random_seed=42,
            )
            by_id = _load_all_records(out)
            self.assertEqual(by_id["id1"]["text"], "中文正文 你好")
            self.assertEqual(by_id["id2"]["text"], "English body")

    def test_json_text_fidelity(self) -> None:
        records = [
            {"story_id": "a1", "text": "第一条 正文", "repost_count": 10, "view_count": 500},
            {"story_id": "a2", "text": "second body", "repost_count": 12, "view_count": 600},
        ]
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = _make_json(tmp, records)
            out = tmp / "out"
            prepare_dataset(
                src, out, num_agents=1000, text_column=None,
                retweet_columns="", view_column="", id_column="",
                random_seed=42,
            )
            by_id = _load_all_records(out)
            self.assertEqual(by_id["a1"]["text"], "第一条 正文")
            self.assertEqual(by_id["a2"]["text"], "second body")

    def test_zero_view_excluded_count(self) -> None:
        rows = [
            {"文章ID": "id1", "正文": "正文1", "观看量": 500, "转发": 10, "分享": 0, "Quotes": 0},
            {"文章ID": "id2", "正文": "正文2", "观看量": 600, "转发": 12, "分享": 0, "Quotes": 0},
            {"文章ID": "zero1", "正文": "零浏览", "观看量": 0, "转发": 5, "分享": 0, "Quotes": 0},
        ]
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = _make_table(tmp, rows)
            out = tmp / "out"
            m = prepare_dataset(
                src, out, num_agents=1000, text_column="正文",
                retweet_columns="转发,分享,Quotes", view_column="观看量", id_column="文章ID",
                random_seed=42,
            )
            manifest = _read_json(m)
            self.assertEqual(manifest["cleaning_counts"]["zero_view_excluded"], 1)

    def test_missing_column_fails(self) -> None:
        rows = [{"文章ID": "id1", "正文": "正文1", "观看量": 500, "转发": 10}]
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = _make_table(tmp, rows)
            out = tmp / "out"
            with self.assertRaises(ValueError):
                prepare_dataset(
                    src, out, num_agents=1000, text_column="正文",
                    retweet_columns="转发,分享,Quotes", view_column="观看量", id_column="文章ID",
                )

    def test_duplicate_id_fails(self) -> None:
        rows = [
            {"文章ID": "dup", "正文": "正文1", "观看量": 500, "转发": 10, "分享": 0, "Quotes": 0},
            {"文章ID": "dup", "正文": "正文2", "观看量": 600, "转发": 12, "分享": 0, "Quotes": 0},
        ]
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = _make_table(tmp, rows)
            out = tmp / "out"
            with self.assertRaises(ValueError):
                prepare_dataset(
                    src, out, num_agents=1000, text_column="正文",
                    retweet_columns="转发,分享,Quotes", view_column="观看量", id_column="文章ID",
                )

    def test_nan_value_fails(self) -> None:
        rows = [
            {"文章ID": "id1", "正文": "正文1", "观看量": "nan", "转发": 10, "分享": 0, "Quotes": 0},
        ]
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = _make_table(tmp, rows)
            out = tmp / "out"
            with self.assertRaises(ValueError):
                prepare_dataset(
                    src, out, num_agents=1000, text_column="正文",
                    retweet_columns="转发,分享,Quotes", view_column="观看量", id_column="文章ID",
                )

    def test_empty_text_fails(self) -> None:
        rows = [
            {"文章ID": "id1", "正文": "正文1", "观看量": 500, "转发": 10, "分享": 0, "Quotes": 0},
            {"文章ID": "id2", "正文": "   ", "观看量": 600, "转发": 12, "分享": 0, "Quotes": 0},
        ]
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = _make_table(tmp, rows)
            out = tmp / "out"
            with self.assertRaises(ValueError):
                prepare_dataset(
                    src, out, num_agents=1000, text_column="正文",
                    retweet_columns="转发,分享,Quotes", view_column="观看量", id_column="文章ID",
                )

    def test_empty_train_fails(self) -> None:
        rows = [
            {"文章ID": "id1", "正文": "正文1", "观看量": 500, "转发": 0, "分享": 0, "Quotes": 0},
            {"文章ID": "id2", "正文": "正文2", "观看量": 600, "转发": 0, "分享": 0, "Quotes": 0},
        ]
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = _make_table(tmp, rows)
            out = tmp / "out"
            with self.assertRaises(ValueError):
                prepare_dataset(
                    src, out, num_agents=1000, text_column="正文",
                    retweet_columns="转发,分享,Quotes", view_column="观看量", id_column="文章ID",
                )

    def test_single_record_fails(self) -> None:
        rows = [{"文章ID": "id1", "正文": "正文1", "观看量": 500, "转发": 10, "分享": 0, "Quotes": 0}]
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = _make_table(tmp, rows)
            out = tmp / "out"
            with self.assertRaises(ValueError):
                prepare_dataset(
                    src, out, num_agents=1000, text_column="正文",
                    retweet_columns="转发,分享,Quotes", view_column="观看量", id_column="文章ID",
                )


class PublishAndLoadTests(unittest.TestCase):
    """原子发布与数据包加载校验。"""

    def test_existing_output_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            out = tmp / "out"
            out.mkdir()
            with self.assertRaises(FileExistsError):
                with publish_output(out):
                    pass

    def test_exception_in_context_cleans_up(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            out = tmp / "out"
            with self.assertRaises(RuntimeError):
                with publish_output(out) as temp_dir:
                    (temp_dir / "partial.txt").write_text("x", encoding="utf-8")
                    raise RuntimeError("boom")
            self.assertFalse(out.exists())
            leftovers = [p.name for p in tmp.iterdir()]
            self.assertEqual(leftovers, [])

    def test_lock_prevents_second_publish(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            out = tmp / "out"
            lock = tmp / (out.name + ".lock")
            lock.write_text("999999", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                with publish_output(out):
                    pass

    def test_publish_success_removes_lock(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            out = tmp / "out"
            with publish_output(out) as temp_dir:
                (temp_dir / "file.txt").write_text("ok", encoding="utf-8")
            self.assertTrue((out / "file.txt").exists())
            self.assertFalse((tmp / (out.name + ".lock")).exists())

    def test_load_split_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = _make_table(tmp, _many_rows())
            out = tmp / "out"
            prepare_dataset(
                src, out, num_agents=1500, text_column="正文",
                retweet_columns="转发,分享,Quotes", view_column="观看量", id_column="文章ID",
                random_seed=42,
            )
            partition, manifest = load_split(out / "train.json", expected_split="train")
            self.assertEqual(partition["split"], "train")
            self.assertEqual(partition["schema_version"], 1)
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(
                sorted(r["story_id"] for r in partition["records"]),
                sorted(manifest["train_ids"]),
            )

    def test_corrupted_hash_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = _make_table(tmp, _many_rows())
            out = tmp / "out"
            prepare_dataset(
                src, out, num_agents=1500, text_column="正文",
                retweet_columns="转发,分享,Quotes", view_column="观看量", id_column="文章ID",
                random_seed=42,
            )
            train_path = out / "train.json"
            data = _read_json(train_path)
            data["records"][0]["text"] = "被篡改"
            train_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_split(train_path, expected_split="train")

    def test_split_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = _make_table(tmp, _many_rows())
            out = tmp / "out"
            prepare_dataset(
                src, out, num_agents=1500, text_column="正文",
                retweet_columns="转发,分享,Quotes", view_column="观看量", id_column="文章ID",
                random_seed=42,
            )
            with self.assertRaises(ValueError):
                load_split(out / "train.json", expected_split="test")

    def test_load_split_does_not_open_other_partition(self) -> None:
        """load_split(train) 不读取 test.json。"""
        import analysis.recommender_data as mod

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = _make_table(tmp, _many_rows())
            out = tmp / "out"
            prepare_dataset(
                src, out, num_agents=1500, text_column="正文",
                retweet_columns="转发,分享,Quotes", view_column="观看量", id_column="文章ID",
                random_seed=42,
            )
            opened = []
            real_read = mod._read_json

            def spy(path: Path) -> dict:
                opened.append(Path(path))
                return real_read(path)

            with mock.patch.object(mod, "_read_json", side_effect=spy):
                load_split(out / "train.json", expected_split="train")
            resolved = [p.resolve() for p in opened]
            self.assertNotIn((out / "test.json").resolve(), resolved)
            self.assertIn((out / "train.json").resolve(), resolved)
            self.assertIn((out / "manifest.json").resolve(), resolved)


class FileHashTests(unittest.TestCase):
    def test_file_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            p = tmp / "a.txt"
            p.write_bytes(b"hello")
            self.assertEqual(len(file_sha256(p)), 64)
            self.assertEqual(file_sha256(p), file_sha256(p))
            q = tmp / "b.txt"
            q.write_bytes(b"world")
            self.assertNotEqual(file_sha256(p), file_sha256(q))


if __name__ == "__main__":
    unittest.main()
