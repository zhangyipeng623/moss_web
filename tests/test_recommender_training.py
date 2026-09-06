"""Task 2 测试：公共概率与全量训练引擎。

覆盖：每个 trial 全量遍历、统一缩放计数损失、成对最佳轮选择、
固定参数重放、状态重置、duration 传递、非法输入失败。
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from analysis.recommender_parameter_inference import (
    EMCalibrationEngine,
    VectorizedABMEngine,
    run_fixed_simulations,
)


class FakeEngine:
    """轻量确定性引擎：记录每次 run_simulation 的输入，返回可控终值。"""

    def __init__(self, N: int = 100):
        self.N = N
        self.calls = []

    def run_simulation(
        self, weights, p_base, I_pop, duration=24, alpha=1.0, beta=1.0,
        decay_lambda=0.5, rng=None,
    ):
        i = np.asarray(I_pop)
        self.calls.append(
            {
                "p_base": float(p_base),
                "I_pop": i.copy(),
                "duration": int(duration),
                "weights": dict(weights),
            }
        )
        final = float(p_base) * float(i[0]) + float(duration)
        return np.array([0.0, final]), 0


def _weights(**overrides):
    w = {"w_i": 0.35, "w_pop": 0.25, "w_time": 0.25, "w_rand": 0.15, "decay_lambda": 0.5}
    w.update(overrides)
    return w


def _stories(n: int = 25, target: float = 100.0):
    return [
        {
            "story_id": f"s{i}",
            "I_pop": np.array([float(i)]),
            "scaled_target": target,
            "text": f"text {i}",
        }
        for i in range(n)
    ]


class GlobalLossTests(unittest.TestCase):
    def test_global_loss_covers_all_stories_and_formula(self) -> None:
        engine = FakeEngine(N=100)
        stories = _stories(25)
        cal = EMCalibrationEngine(engine, stories, n_cpu=1, seed=42)
        loss = cal._global_count_loss(0.5, _weights(), duration=24, n_repeats=2, rng_base=0)
        # 25 条 × 2 重复 = 50 次调用，每条 story 覆盖 2 次
        self.assertEqual(len(engine.calls), 50)
        seen = [c["I_pop"][0] for c in engine.calls]
        self.assertEqual(sorted(set(seen)), list(range(25)))
        # 每条覆盖恰好 2 次
        for i in range(25):
            self.assertEqual(seen.count(float(i)), 2)
        # duration 传递到所有调用
        self.assertTrue(all(c["duration"] == 24 for c in engine.calls))
        # 统一缩放计数损失 = mean_i(mean_r(|final - target| / N))
        finals = 0.5 * np.arange(25) + 24.0
        targets = np.full(25, 100.0)
        expected = float(np.mean(np.abs(finals - targets) / 100.0))
        self.assertAlmostEqual(loss, expected, places=10)

    def test_global_loss_ignores_trajectory_curve(self) -> None:
        engine = FakeEngine(N=100)
        stories = _stories(3)
        # 加一条带 repost_curve 的记录，不应影响计数损失
        stories[1]["repost_curve"] = [0.0, 0.5, 1.0]
        cal = EMCalibrationEngine(engine, stories, n_cpu=1, seed=42)
        loss = cal._global_count_loss(0.5, _weights(), duration=24, n_repeats=1, rng_base=0)
        finals = 0.5 * np.arange(3) + 24.0
        expected = float(np.mean(np.abs(finals - np.full(3, 100.0)) / 100.0))
        self.assertAlmostEqual(loss, expected, places=10)

    def test_global_loss_nonfinite_final_fails(self) -> None:
        engine = FakeEngine(N=100)
        engine.run_simulation = lambda *a, **k: (np.array([0.0, float("nan")]), 0)
        stories = _stories(2)
        cal = EMCalibrationEngine(engine, stories, n_cpu=1, seed=42)
        with self.assertRaises(RuntimeError):
            cal._global_count_loss(0.5, _weights(), duration=24, n_repeats=1, rng_base=0)


class BestRoundTests(unittest.TestCase):
    def test_select_best_round(self) -> None:
        rounds = [
            {"iteration": 1, "p_base": 0.9, "weights": {"w_i": 1.0}, "loss": 0.5},
            {"iteration": 2, "p_base": 0.1, "weights": {"w_i": 2.0}, "loss": 0.2},
            {"iteration": 3, "p_base": 0.7, "weights": {"w_i": 3.0}, "loss": 0.4},
        ]
        best = EMCalibrationEngine._select_best_round(rounds)
        self.assertEqual(best["iteration"], 2)
        self.assertEqual(best["p_base"], 0.1)
        self.assertEqual(best["weights"]["w_i"], 2.0)

    def test_run_global_calibration_selects_best_round_pair(self) -> None:
        engine = FakeEngine(N=100)
        stories = _stories(4)
        cal = EMCalibrationEngine(engine, stories, n_cpu=1, seed=42)
        w1 = _weights(decay_lambda=1.0)
        w2 = _weights(decay_lambda=2.0)
        # 两轮：第 1 轮 loss 高，第 2 轮 loss 低（best），最后 replay loss 与第 2 轮一致
        with mock.patch.object(
            cal, "_optimize_p_base", side_effect=[(0.9, 1.0), (0.1, 0.2)]
        ), mock.patch.object(
            cal, "_optimize_weights", side_effect=[(w1, 0.9), (w2, 0.1)]
        ), mock.patch.object(
            cal, "_global_count_loss", side_effect=[0.9, 0.1, 0.1]
        ):
            result = cal.run_global_calibration(
                iterations=2, duration=24, n_repeats=1, p_trials=2, weight_trials=2
            )
        self.assertEqual(result["p_base_global"], 0.1)
        self.assertEqual(result["weights"]["decay_lambda"], 2.0)
        self.assertEqual(result["loss"], 0.1)
        self.assertEqual(result["diagnostics"]["best_iteration"], 2)


class CalibrationValidationTests(unittest.TestCase):
    def test_empty_stories_fail(self) -> None:
        engine = FakeEngine(N=100)
        cal = EMCalibrationEngine(engine, [], n_cpu=1, seed=42)
        with self.assertRaises(RuntimeError):
            cal.run_global_calibration(iterations=2)

    def test_bad_budget_fails(self) -> None:
        engine = FakeEngine(N=100)
        cal = EMCalibrationEngine(engine, _stories(3), n_cpu=1, seed=42)
        with self.assertRaises(ValueError):
            cal.run_global_calibration(iterations=2, p_trials=0)
        with self.assertRaises(ValueError):
            cal.run_global_calibration(iterations=2, weight_trials=0)
        with self.assertRaises(ValueError):
            cal.run_global_calibration(iterations=0)


def _small_engine(seed: int = 42) -> VectorizedABMEngine:
    S = np.array([-0.8, -0.4, -0.1, 0.1, 0.4, 0.8], dtype=float)
    Inf = np.array([2.0, 1.5, 1.0, 1.0, 1.5, 2.0], dtype=float)
    return VectorizedABMEngine(S, Inf, log_saturation_threshold=1.0, seed=seed)


class FixedSimulationsTests(unittest.TestCase):
    def test_replay_same_seed(self) -> None:
        engine = _small_engine(seed=42)
        stories = [
            {"story_id": "a", "I_pop": np.full(6, 0.2), "scaled_target": 2.0},
            {"story_id": "b", "I_pop": np.full(6, 0.4), "scaled_target": 3.0},
        ]
        w = _weights()
        out1 = run_fixed_simulations(
            engine, stories, w, 0.5, duration=10, n_repeats=3, seed=7, n_cpu=1
        )
        out2 = run_fixed_simulations(
            engine, stories, w, 0.5, duration=10, n_repeats=3, seed=7, n_cpu=1
        )
        self.assertEqual(out1.shape, (2, 3))
        np.testing.assert_array_equal(out1, out2)

    def test_duration_and_state_reset(self) -> None:
        engine = _small_engine(seed=42)
        story = {"story_id": "a", "I_pop": np.full(6, 0.2), "scaled_target": 2.0}
        w = _weights()
        hist_short, _ = engine.run_simulation(
            w, 0.5, story["I_pop"], duration=2, decay_lambda=0.5, rng=np.random.default_rng(1)
        )
        hist_long, _ = engine.run_simulation(
            w, 0.5, story["I_pop"], duration=3, decay_lambda=0.5, rng=np.random.default_rng(1)
        )
        self.assertEqual(len(hist_short), 2)
        self.assertEqual(len(hist_long), 3)
        # 同一 rng 连续运行：状态被 reset，历史长度一致且首步行为可复现
        hist_again, _ = engine.run_simulation(
            w, 0.5, story["I_pop"], duration=2, decay_lambda=0.5, rng=np.random.default_rng(1)
        )
        np.testing.assert_array_equal(hist_short, hist_again)

    def test_parallel_matches_serial(self) -> None:
        engine = _small_engine(seed=42)
        stories = [
            {"story_id": f"s{i}", "I_pop": np.full(6, 0.1 + 0.02 * i), "scaled_target": 2.0}
            for i in range(4)
        ]
        w = _weights()
        serial = run_fixed_simulations(
            engine, stories, w, 0.5, duration=6, n_repeats=2, seed=7, n_cpu=1
        )
        parallel = run_fixed_simulations(
            engine, stories, w, 0.5, duration=6, n_repeats=2, seed=7, n_cpu=2
        )
        np.testing.assert_array_equal(serial, parallel)


class GlobalCalibrationSmokeTests(unittest.TestCase):
    def test_real_abm_smoke(self) -> None:
        """真实小 ABM 跑一次公共概率校准，验证输出成对且有限。"""
        engine = _small_engine(seed=42)
        stories = [
            {"story_id": f"s{i}", "I_pop": np.full(6, 0.1 + 0.02 * i), "scaled_target": 2.0}
            for i in range(5)
        ]
        cal = EMCalibrationEngine(engine, stories, n_cpu=1, seed=42)
        result = cal.run_global_calibration(
            iterations=2, duration=6, n_repeats=2, p_trials=3, weight_trials=3
        )
        self.assertIn("weights", result)
        self.assertIn("decay_lambda", result["weights"])
        self.assertGreaterEqual(result["p_base_global"], 0.001)
        self.assertLessEqual(result["p_base_global"], 0.999)
        self.assertTrue(np.isfinite(result["loss"]))
        # 权重归一化后和为 1
        total = sum(
            result["weights"][k] for k in ("w_i", "w_pop", "w_time", "w_rand")
        )
        self.assertAlmostEqual(total, 1.0, places=6)
        # 重放损失与记录的最佳损失一致（确定性）
        self.assertAlmostEqual(
            result["diagnostics"]["replay_loss"],
            result["diagnostics"]["best_loss"],
            places=9,
        )


if __name__ == "__main__":
    unittest.main()
