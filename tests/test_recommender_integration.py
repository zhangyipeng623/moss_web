"""Task 5 测试：三入口离线集成与 README 迁移。

prepare → train → compare 全链路（固定向量替换嵌入，保留真实数据处理/校准器/ABM），
同组零差、人口与轮数一致、旧 YAML 可读；subprocess 验证 CLI help 与非法参数退出。
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from analysis.compare_recommenders import compare_models
from analysis.recommender_data import prepare_dataset
from analysis.recommender_parameter_inference import EmbeddingService, load_portrait_bundle
from core.calibration_profile import CalibrationProfile

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _fake_embed(texts):
    if isinstance(texts, str):
        texts = [texts]
    import zlib

    out = np.zeros((len(texts), 8), dtype=float)
    for i, t in enumerate(texts):
        rng = np.random.default_rng(zlib.crc32(str(t).encode("utf-8")))
        out[i] = rng.normal(size=8)
    return out


def _make_source(tmp: Path, n: int = 45) -> Path:
    rows = []
    for i in range(n):
        rows.append(
            {
                "文章ID": f"id{i:03d}", "正文": f"内容 {i}",
                "观看量": 300 + i * 10, "转发": i + 20, "分享": 0, "Quotes": 0,
            }
        )
    # 零转发记录：测试分区保留
    for j in range(3):
        rows.append(
            {
                "文章ID": f"low{j}", "正文": f"低传播 {j}",
                "观看量": 400 + j, "转发": 0, "分享": 0, "Quotes": 0,
            }
        )
    src = tmp / "posts.csv"
    with src.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return src


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


class FullPipelineTests(unittest.TestCase):
    def test_prepare_train_compare_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = _make_source(tmp)
            portraits = _make_portraits(tmp)

            # 1. prepare
            dataset = tmp / "dataset"
            prepare_dataset(
                src, dataset, num_agents=100, text_column="正文",
                retweet_columns="转发,分享,Quotes", view_column="观看量", id_column="文章ID",
                random_seed=42,
            )
            self.assertTrue((dataset / "train.json").exists())
            self.assertTrue((dataset / "test.json").exists())
            self.assertTrue((dataset / "manifest.json").exists())
            train = json.loads((dataset / "train.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(train["records"]), 25)

            # 2. train（小预算，不改生产默认）
            from analysis.run_analysis import run_recommender_command

            calibration = tmp / "calibration"
            args = argparse.Namespace(
                train_file=str(dataset / "train.json"),
                portraits_dir=str(portraits),
                output_dir=str(calibration),
                embedding_model="test-embedding",
                n_cpu=1, random_seed=42, time_scale=3600.0, rounds=6,
                max_iterations=2, n_repeats=2, p_trials=2, weight_trials=2,
            )
            with mock.patch.object(EmbeddingService, "embed_documents", side_effect=_fake_embed):
                run_recommender_command(args)
            self.assertTrue((calibration / "model.json").exists())
            self.assertTrue((calibration / "calibration_profile.yaml").exists())
            model = json.loads((calibration / "model.json").read_text(encoding="utf-8"))
            self.assertEqual(model["environment"]["population_size"], 100)
            self.assertEqual(model["environment"]["rounds"], 6)
            # 旧 YAML 可读
            profile = CalibrationProfile.model_validate(
                _parse_yaml((calibration / "calibration_profile.yaml").read_text(encoding="utf-8"))
            )
            self.assertIsNotNone(profile.recommender.p_base_global)

            # 3. compare（对照与候选同权重 → 零差）
            baseline = tmp / "baseline.json"
            baseline.write_text(
                json.dumps(
                    {
                        "name": "baseline",
                        "weights": {
                            "w_interest": model["weights"]["w_i"],
                            "w_popularity": model["weights"]["w_pop"],
                            "w_time": model["weights"]["w_time"],
                            "w_random": model["weights"]["w_rand"],
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            comparison = tmp / "comparison"
            with mock.patch.object(EmbeddingService, "embed_documents", side_effect=_fake_embed):
                summary_path = compare_models(
                    Path(calibration / "model.json"), dataset / "test.json",
                    baseline, portraits, comparison,
                    n_repeats=30, seed=2026, p_base_grid=[0.3, 0.5],
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(summary["main"]["improvement"]["mae"], 0.0, places=9)
            self.assertTrue((comparison / "per_story.csv").exists())


class SubprocessTests(unittest.TestCase):
    def _run(self, argv: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, *argv],
            cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=120,
        )

    def test_cli_help_and_legacy_error(self) -> None:
        for argv in (
            ["-m", "analysis.run_analysis", "recommender", "--help"],
            ["scripts/prepare_recommender_data.py", "--help"],
            ["-m", "analysis.compare_recommenders", "--help"],
        ):
            proc = self._run(argv)
            self.assertEqual(proc.returncode, 0, msg=f"{argv} 应成功，stderr={proc.stderr}")

    def test_legacy_recommender_args_exit_nonzero(self) -> None:
        proc = self._run(
            ["-m", "analysis.run_analysis", "recommender", "--data-file", "x.xlsx"]
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("迁移", proc.stderr)


def _parse_yaml(text: str):
    import yaml

    return yaml.safe_load(text)


if __name__ == "__main__":
    unittest.main()
