"""Task 4 测试：固定参数比较、误差与敏感性。

覆盖：指标手算、零目标 null、bootstrap 区间、baseline 解析、
同系数零差、敏感性全网格、非法输入失败、目标扰动不影响预测。
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from analysis.compare_recommenders import (
    _bootstrap_mae_diff,
    _parse_baseline,
    compare_models,
    compute_metrics,
)
from analysis.recommender_data import prepare_dataset
from analysis.recommender_parameter_inference import EmbeddingService, load_portrait_bundle
from core.scoring import TIER_WEIGHT_DEFAULT


def _fake_embed(texts):
    if isinstance(texts, str):
        texts = [texts]
    import zlib

    out = np.zeros((len(texts), 8), dtype=float)
    for i, t in enumerate(texts):
        rng = np.random.default_rng(zlib.crc32(str(t).encode("utf-8")))
        out[i] = rng.normal(size=8)
    return out


def _make_portraits(tmp: Path, n: int = 6) -> Path:
    d = tmp / "portraits"
    d.mkdir()
    for i in range(n):
        profile = {
            "influence_tier": 3 + (i % 3),
            "stable_profile": {
                "value_anchors": [{"stance": f"立场{i}"}],
                "content_topics": [f"话题{i}"],
                "profile_summary": f"摘要{i}",
            },
            "agent_profile": {"identity_summary": f"身份{i}", "interest_summary": f"兴趣{i}"},
            "stats": {"account_influence": 1.0 + i * 0.5},
        }
        (d / f"u{i}.json").write_text(json.dumps(profile, ensure_ascii=False), encoding="utf-8")
    return d


def _make_dataset(tmp: Path) -> Path:
    rows = []
    for i in range(16):
        rows.append(
            {
                "文章ID": f"id{i:02d}", "正文": f"内容 {i}",
                "观看量": 300 + i * 20, "转发": i + 2, "分享": 0, "Quotes": 0,
            }
        )
    src = tmp / "posts.csv"
    with src.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)
    out = tmp / "dataset"
    prepare_dataset(
        src, out, num_agents=200, text_column="正文",
        retweet_columns="转发,分享,Quotes", view_column="观看量", id_column="文章ID",
        random_seed=42,
    )
    return out


def _make_model(tmp: Path, dataset: Path, portraits: Path, p_base: float = 0.1) -> Path:
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    _, portrait_manifest = load_portrait_bundle(portraits)
    model = {
        "schema_version": 1,
        "data": {
            "train_hash": "d" * 64,
            "manifest_hash": "e" * 64,
            "train_story_ids": manifest["train_ids"],
            "scale_ratio": manifest["scale_ratio"],
        },
        "weights": {"w_i": 0.35, "w_pop": 0.25, "w_time": 0.25, "w_rand": 0.15},
        "decay_lambda": 0.5,
        "p_base_mode": "global",
        "p_base_global": p_base,
        "training": {"budget": {}, "loss_name": "mean_abs_scaled_count_error",
                     "best_iteration": 1, "best_loss": 0.1, "replay_loss": 0.1, "rounds": []},
        "environment": {
            "training_seed": 42,
            "population_size": 200,
            "p_online": 0.1,
            "belief_update": {"backfire_mu": 0.4, "backfire_k": 10.0, "learning_rate": 0.1},
            "tier_weight": dict(TIER_WEIGHT_DEFAULT),
            "hours_per_step": 1.0,
            "time_scale": 3600.0,
            "rounds": 6,
        },
        "portraits": {"num_seed_users": len(portrait_manifest), "files": portrait_manifest},
        "embedding": {"model_name": "test-embedding", "normalize_embeddings": True},
    }
    path = tmp / "model.json"
    path.write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")
    return path


def _make_baseline(tmp: Path, weights, name: str = "baseline", decay=None) -> Path:
    payload = {"name": name, "weights": weights}
    if decay is not None:
        payload["decay_lambda"] = decay
    path = tmp / "baseline.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


class MetricsTests(unittest.TestCase):
    def test_hand_calc(self) -> None:
        targets = np.array([0.0, 2.0, 4.0])
        finals = np.array([[1.0, 1.0], [2.0, 2.0], [6.0, 6.0]])
        m = compute_metrics(targets, finals, relative_threshold=0.2, absolute_threshold=1.0)
        self.assertAlmostEqual(m["mae"], 1.0, places=10)
        self.assertAlmostEqual(m["rmse"], np.sqrt(5 / 3), places=10)
        self.assertAlmostEqual(m["mre_nonzero"], 0.25, places=10)
        self.assertAlmostEqual(m["relative_pass_rate_nonzero"], 0.5, places=10)
        self.assertAlmostEqual(m["absolute_pass_rate"], 2 / 3, places=10)
        self.assertEqual(m["n_nonzero"], 2)

    def test_all_zero_targets_null(self) -> None:
        targets = np.array([0.0, 0.0])
        finals = np.array([[1.0], [2.0]])
        m = compute_metrics(targets, finals, relative_threshold=0.2, absolute_threshold=1.0)
        self.assertIsNone(m["mre_nonzero"])
        self.assertIsNone(m["relative_pass_rate_nonzero"])
        self.assertEqual(m["n_nonzero"], 0)

    def test_bootstrap_null_and_zero(self) -> None:
        # 单条 → null
        self.assertIsNone(_bootstrap_mae_diff(np.array([1.0]), np.array([2.0]), np.array([3.0]), seed=1))
        # 相同均值 → 区间为 0
        ci = _bootstrap_mae_diff(np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([3.0, 4.0]), seed=1)
        self.assertEqual(ci, [0.0, 0.0])


class BaselineParseTests(unittest.TestCase):
    def _w(self, **kw):
        base = {"w_interest": 1.0, "w_popularity": 1.0, "w_time": 1.0, "w_random": 1.0}
        base.update(kw)
        return base

    def test_parse_and_normalize(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = _make_baseline(Path(td), self._w())
            name, weights, decay_provided, raw = _parse_baseline(p, model_decay=0.5)
            self.assertEqual(name, "baseline")
            self.assertAlmostEqual(weights["w_i"], 0.25, places=10)
            self.assertAlmostEqual(weights["decay_lambda"], 0.5, places=10)
            self.assertFalse(decay_provided)
            self.assertEqual(raw["w_i"], 1.0)

    def test_decay_provided(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = _make_baseline(Path(td), self._w(), decay=1.5)
            name, weights, decay_provided, raw = _parse_baseline(p, model_decay=0.5)
            self.assertTrue(decay_provided)
            self.assertAlmostEqual(weights["decay_lambda"], 1.5, places=10)

    def test_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            for bad, msg in (
                ({"w_interest": 1.0, "w_popularity": 1.0, "w_time": 1.0}, "缺少字段"),
                ({"w_interest": 1.0, "w_popularity": 1.0, "w_time": 1.0, "w_random": 1.0, "extra": 0.0}, "未知字段"),
                ({"w_interest": 0.0, "w_popularity": 0.0, "w_time": 0.0, "w_random": 0.0}, "总和为零"),
                ({"w_interest": -1.0, "w_popularity": 1.0, "w_time": 1.0, "w_random": 1.0}, "负数"),
            ):
                p = _make_baseline(tmp, bad)
                with self.assertRaises(ValueError):
                    _parse_baseline(p, model_decay=0.5)


class CompareModelsTests(unittest.TestCase):
    def _setup(self, tmp: Path):
        portraits = _make_portraits(tmp)
        dataset = _make_dataset(tmp)
        model = _make_model(tmp, dataset, portraits, p_base=0.3)
        return portraits, dataset, model

    def test_same_coefficients_zero_diff(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            portraits, dataset, model = self._setup(tmp)
            # 对照系数与模型一致
            baseline = _make_baseline(
                tmp, {"w_interest": 0.35, "w_popularity": 0.25, "w_time": 0.25, "w_random": 0.15}
            )
            out = tmp / "comparison"
            with mock.patch.object(EmbeddingService, "embed_documents", side_effect=_fake_embed):
                summary_path = compare_models(
                    model, dataset / "test.json", baseline, portraits, out,
                    n_repeats=30, seed=2026, p_base_grid=[0.3, 0.5],
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(summary["main"]["improvement"]["mae"], 0.0, places=9)
            ci = summary["main"]["mae_difference_ci95"]
            self.assertIsNotNone(ci)
            self.assertAlmostEqual(ci[0], 0.0, places=9)
            self.assertAlmostEqual(ci[1], 0.0, places=9)
            # 全网格输出（去重后 p_base 网格）
            self.assertEqual(len(summary["sensitivity"]), 2)
            # per_story.csv 存在且行数 = 测试数 × 2
            csv_text = (out / "per_story.csv").read_text(encoding="utf-8")
            self.assertIn("story_id", csv_text)

    def test_different_coefficients_completes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            portraits, dataset, model = self._setup(tmp)
            baseline = _make_baseline(
                tmp, {"w_interest": 0.1, "w_popularity": 0.4, "w_time": 0.4, "w_random": 0.1}
            )
            out = tmp / "comparison"
            with mock.patch.object(EmbeddingService, "embed_documents", side_effect=_fake_embed):
                compare_models(
                    model, dataset / "test.json", baseline, portraits, out,
                    n_repeats=30, seed=2026,
                )
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("improvement", summary["main"])
            self.assertEqual(len(summary["sensitivity"]), len([0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.9, 0.999]))

    def test_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            portraits, dataset, model = self._setup(tmp)
            baseline = _make_baseline(
                tmp, {"w_interest": 0.35, "w_popularity": 0.25, "w_time": 0.25, "w_random": 0.15}
            )
            out = tmp / "comparison"
            with mock.patch.object(EmbeddingService, "embed_documents", side_effect=_fake_embed):
                with self.assertRaises(ValueError):
                    compare_models(model, dataset / "test.json", baseline, portraits, out, n_repeats=29)
                with self.assertRaises(ValueError):
                    compare_models(
                        model, dataset / "test.json", baseline, portraits, out,
                        p_base_grid=[0.0, 0.5],
                    )
                with self.assertRaises(ValueError):
                    compare_models(model, dataset / "test.json", baseline, portraits, out, absolute_threshold=-1)
            # 损坏散列拒绝
            corrupted = dataset / "test_corrupt.json"
            data = json.loads((dataset / "test.json").read_text(encoding="utf-8"))
            data["records"][0]["text"] = "篡改"
            corrupted.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            with mock.patch.object(EmbeddingService, "embed_documents", side_effect=_fake_embed):
                with self.assertRaises(ValueError):
                    compare_models(model, corrupted, baseline, portraits, out)
            # 无成功目录
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
