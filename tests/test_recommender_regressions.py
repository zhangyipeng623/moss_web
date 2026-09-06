"""比较数据绑定、达标率、敏感性明细与训练并行回归。"""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import numpy as np
from test_recommender_comparison import _make_dataset, _make_portraits, _make_model, _make_baseline, _fake_embed
from test_recommender_training import FakeEngine, _stories, _weights
from analysis.compare_recommenders import compare_models
from analysis.recommender_parameter_inference import EmbeddingService, EMCalibrationEngine
import analysis.recommender_parameter_inference as inference

class RandomEngine(FakeEngine):
    """随机终值用于验证不同进程使用同一组推文随机流。"""
    def run_simulation(self, weights, p_base, I_pop, duration=24, decay_lambda=.5, rng=None):
        return np.array([0.0, float(rng.integers(0, self.N))]), 0

class ReviewRegressions(unittest.TestCase):
    def test_comparison_binding_and_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset = _make_dataset(root)
            portraits = _make_portraits(root)
            model = _make_model(root, dataset, portraits)
            original = json.loads(model.read_text())
            baseline = _make_baseline(root, {'w_interest': .35, 'w_popularity': .25, 'w_time': .25, 'w_random': .15})
            for field, value in [('scale_ratio', 99999), ('manifest_hash', 'bad'), ('train_hash', 'bad')]:
                changed = json.loads(json.dumps(original))
                changed['data'][field] = value
                model.write_text(json.dumps(changed))
                with self.subTest(field=field), mock.patch.object(EmbeddingService, 'embed_documents', side_effect=AssertionError('不应加载嵌入')):
                    with self.assertRaises(ValueError):
                        compare_models(model, dataset/'test.json', baseline, portraits, root/'bad')
            model.write_text(json.dumps(original))
            targets = np.array([r['scaled_target'] for r in json.loads((dataset/'test.json').read_text())['records']])
            good = np.repeat(targets[:, None], 30, axis=1)
            bad = good + 10
            with mock.patch.object(EmbeddingService, 'embed_documents', side_effect=_fake_embed), mock.patch('analysis.compare_recommenders.run_fixed_simulations', side_effect=[good,bad,good,bad]):
                path = compare_models(model, dataset/'test.json', baseline, portraits, root/'out', p_base_grid=[.1,.2,.2])
            result = json.loads(path.read_text())
            self.assertEqual(result['main']['improvement']['absolute_pass_rate'], 1.0)
            with (root/'out/per_story.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), len(targets)*4)
            self.assertEqual({float(r['p_base']) for r in rows}, {.1,.2})

    def test_global_loss_uses_parallel_and_matches_serial(self):
        stories = _stories()
        serial = EMCalibrationEngine(RandomEngine(), stories, n_cpu=1, seed=42)
        parallel = EMCalibrationEngine(RandomEngine(), stories, n_cpu=2, seed=42)
        expected = serial._global_count_loss(.2, _weights(), duration=3, n_repeats=2, rng_base=42)
        with mock.patch.object(inference, 'Parallel', wraps=inference.Parallel) as executor:
            actual = parallel._global_count_loss(.2, _weights(), duration=3, n_repeats=2, rng_base=42)
        executor.assert_called_once_with(n_jobs=2)
        self.assertEqual(actual, expected)
