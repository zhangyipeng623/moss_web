"""Task 3 测试：训练 CLI、model.json 与 YAML 迁移。

覆盖：训练不读测试文件、不调用旧留出/筛选入口、model 完整权威产物、
model 与 YAML 参数一致、旧 YAML 兼容、非法概率拒绝、旧参数迁移错误。
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from analysis.recommender_data import prepare_dataset
from analysis.recommender_parameter_inference import (
    EmbeddingService,
    RecommendationParameterInferer,
    load_portrait_bundle,
)
from core.calibration_profile import CalibrationProfile, RecommenderConfig


def _make_source(tmp: Path, n: int = 12) -> Path:
    rows = []
    for i in range(n):
        rows.append(
            {
                "文章ID": f"id{i:02d}",
                "正文": f"正文内容 {i}",
                "观看量": 300 + i * 20,
                "转发": i + 2,
                "分享": 0,
                "Quotes": 0,
            }
        )
    path = tmp / "posts.csv"
    cols = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return path


def _make_portrait(path: Path, name: str, tier: int = 4) -> None:
    profile = {
        "influence_tier": tier,
        "stable_profile": {
            "value_anchors": [{"stance": f"立场{name}"}],
            "content_topics": [f"话题{name}"],
            "profile_summary": f"摘要{name}",
        },
        "agent_profile": {
            "identity_summary": f"身份{name}",
            "interest_summary": f"兴趣{name}",
        },
        "stats": {"account_influence": 1.0 + tier * 0.5},
    }
    path.write_text(json.dumps(profile, ensure_ascii=False), encoding="utf-8")


def _make_portraits_dir(tmp: Path, n: int = 6) -> Path:
    d = tmp / "portraits"
    d.mkdir()
    for i in range(n):
        _make_portrait(d / f"u{i}.json", str(i), tier=3 + (i % 3))
    return d


def _fake_embed(texts):
    if isinstance(texts, str):
        texts = [texts]
    import zlib

    out = np.zeros((len(texts), 8), dtype=float)
    for i, t in enumerate(texts):
        rng = np.random.default_rng(zlib.crc32(str(t).encode("utf-8")))
        out[i] = rng.normal(size=8)
    return out


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        train_file=None,
        portraits_dir=None,
        output_dir=None,
        embedding_model="test-embedding",
        n_cpu=1,
        random_seed=42,
        time_scale=3600.0,
        rounds=6,
        max_iterations=2,
        n_repeats=2,
        p_trials=3,
        weight_trials=3,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class TrainingOrchestrationTests(unittest.TestCase):
    def _prepare(self, tmp: Path):
        src = _make_source(tmp)
        out = tmp / "dataset"
        prepare_dataset(
            src, out, num_agents=200, text_column="正文",
            retweet_columns="转发,分享,Quotes", view_column="观看量", id_column="文章ID",
            random_seed=42,
        )
        # 删除 test.json 模拟不可读测试文件
        (out / "test.json").unlink()
        portraits = _make_portraits_dir(tmp)
        return out / "train.json", portraits

    def test_training_does_not_read_test_or_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            train_file, portraits = self._prepare(tmp)
            output_dir = tmp / "calibration"
            args = _args(
                train_file=str(train_file),
                portraits_dir=str(portraits),
                output_dir=str(output_dir),
            )
            from analysis.run_analysis import run_recommender_command

            with mock.patch.object(EmbeddingService, "embed_documents", side_effect=_fake_embed), \
                    mock.patch.object(
                        RecommendationParameterInferer, "evaluate_holdout",
                        side_effect=AssertionError("evaluate_holdout 被调用"),
                    ), \
                    mock.patch.object(
                        RecommendationParameterInferer, "split_holdout",
                        side_effect=AssertionError("split_holdout 被调用"),
                    ), \
                    mock.patch.object(
                        RecommendationParameterInferer, "select_representative_stories",
                        side_effect=AssertionError("select_representative_stories 被调用"),
                    ):
                run_recommender_command(args)

            self.assertTrue((output_dir / "model.json").exists())
            self.assertTrue((output_dir / "calibration_profile.yaml").exists())


class ModelArtifactTests(unittest.TestCase):
    def _train(self, tmp: Path):
        src = _make_source(tmp)
        out = tmp / "dataset"
        prepare_dataset(
            src, out, num_agents=200, text_column="正文",
            retweet_columns="转发,分享,Quotes", view_column="观看量", id_column="文章ID",
            random_seed=42,
        )
        portraits = _make_portraits_dir(tmp)
        output_dir = tmp / "calibration"
        args = _args(
            train_file=str(out / "train.json"),
            portraits_dir=str(portraits),
            output_dir=str(output_dir),
        )
        from analysis.run_analysis import run_recommender_command

        with mock.patch.object(EmbeddingService, "embed_documents", side_effect=_fake_embed):
            run_recommender_command(args)
        return output_dir

    def test_model_json_schema(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_dir = self._train(Path(td))
            model = json.loads((output_dir / "model.json").read_text(encoding="utf-8"))
            self.assertEqual(model["schema_version"], 1)
            for key in ("data", "weights", "decay_lambda", "p_base_mode", "p_base_global",
                        "training", "environment", "portraits", "embedding"):
                self.assertIn(key, model)
            self.assertEqual(model["p_base_mode"], "global")
            # data 保存训练/清单散列、训练 ID、缩放比例
            for key in ("train_hash", "manifest_hash", "train_story_ids", "scale_ratio"):
                self.assertIn(key, model["data"])
            # environment 保存完整 ABM 参数
            env = model["environment"]
            for key in ("training_seed", "population_size", "p_online", "belief_update",
                        "tier_weight", "hours_per_step", "time_scale", "rounds"):
                self.assertIn(key, env)
            self.assertEqual(env["population_size"], 200)
            self.assertEqual(env["training_seed"], 42)
            # portraits 保存清单
            self.assertIn("files", model["portraits"])
            self.assertGreaterEqual(len(model["portraits"]["files"]), 1)

    def test_dual_artifact_consistency(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_dir = self._train(Path(td))
            model = json.loads((output_dir / "model.json").read_text(encoding="utf-8"))
            yaml_text = (output_dir / "calibration_profile.yaml").read_text(encoding="utf-8")
            # 权重、概率、时长一致
            profile = CalibrationProfile.model_validate(_parse_yaml(yaml_text))
            rec = profile.recommender
            self.assertAlmostEqual(rec.weights.w_interest, model["weights"]["w_i"], places=6)
            self.assertAlmostEqual(rec.weights.w_popularity, model["weights"]["w_pop"], places=6)
            self.assertAlmostEqual(rec.weights.w_time, model["weights"]["w_time"], places=6)
            self.assertAlmostEqual(rec.weights.w_random, model["weights"]["w_rand"], places=6)
            self.assertAlmostEqual(rec.decay_lambda, model["decay_lambda"], places=6)
            self.assertAlmostEqual(rec.p_base_global, model["p_base_global"], places=6)
            self.assertEqual(profile.experiment.runtime.rounds, model["environment"]["rounds"])
            # 新模型 calibrated_p_base 留空
            self.assertEqual(rec.calibrated_p_base, {})


class ConfigCompatibilityTests(unittest.TestCase):
    def test_old_yaml_without_p_base_global_parses(self) -> None:
        import yaml

        old = {
            "meta": {"generated_at": "", "portraits_dir": "", "num_seed_users": 0, "abm_population_size": 500},
            "experiment": {},
            "recommender": {
                "weights": {"w_interest": 0.35, "w_popularity": 0.25, "w_time": 0.25, "w_random": 0.15},
                "decay_lambda": 0.5,
            },
            "embedding": {"model_name": "Alibaba-NLP/gte-multilingual-base"},
            "simulation": {},
        }
        profile = CalibrationProfile.model_validate(old)
        self.assertIsNone(profile.recommender.p_base_global)

    def test_illegal_p_base_global_rejected(self) -> None:
        with self.assertRaises(Exception):
            RecommenderConfig(p_base_global=1.5)
        with self.assertRaises(Exception):
            RecommenderConfig(p_base_global=0.0)


class PortraitBundleTests(unittest.TestCase):
    def test_bundle_sorted_with_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            d = tmp / "portraits"
            d.mkdir()
            _make_portrait(d / "b.json", "b")
            _make_portrait(d / "a.json", "a")
            # 合法非画像 JSON 不进入清单
            (d / "not_portrait.json").write_text(
                json.dumps({"something": "else"}), encoding="utf-8"
            )
            personas, manifest = load_portrait_bundle(d)
            self.assertEqual(len(personas), 2)
            self.assertEqual(len(manifest), 2)
            self.assertEqual([m["path"] for m in manifest], ["a.json", "b.json"])
            for m in manifest:
                self.assertEqual(len(m["sha256"]), 64)

    def test_corrupted_portrait_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            d = tmp / "portraits"
            d.mkdir()
            _make_portrait(d / "a.json", "a")
            (d / "bad.json").write_text("{ not valid json", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_portrait_bundle(d)

    def test_empty_portraits_fails_training(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = _make_source(tmp)
            out = tmp / "dataset"
            prepare_dataset(
                src, out, num_agents=200, text_column="正文",
                retweet_columns="转发,分享,Quotes", view_column="观看量", id_column="文章ID",
                random_seed=42,
            )
            empty = tmp / "empty_portraits"
            empty.mkdir()
            args = _args(
                train_file=str(out / "train.json"),
                portraits_dir=str(empty),
                output_dir=str(tmp / "calibration"),
            )
            from analysis.run_analysis import run_recommender_command

            with mock.patch.object(EmbeddingService, "embed_documents", side_effect=_fake_embed):
                with self.assertRaises(ValueError):
                    run_recommender_command(args)


class LegacyArgTests(unittest.TestCase):
    def test_legacy_args_rejected(self) -> None:
        from analysis.run_analysis import _raise_if_legacy_recommender_args

        for kwargs in (
            {"data_file": "x.xlsx"},
            {"input": "x.json"},
            {"retweet_columns": "转发"},
            {"view_column": "观看量"},
            {"id_column": "文章ID"},
            {"anchor_percentile": 0.8},
            {"min_scaled_target": 5},
            {"num_agents": 1500},
            {"robustness_seeds": "0,1"},
            {"output": "x.json"},
        ):
            args = _args(**kwargs)
            with self.assertRaises(ValueError):
                _raise_if_legacy_recommender_args(args)


def _parse_yaml(text: str):
    import yaml

    return yaml.safe_load(text)


if __name__ == "__main__":
    unittest.main()
