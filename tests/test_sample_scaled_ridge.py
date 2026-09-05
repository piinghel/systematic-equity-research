"""Focused contracts for the sample-scaled Ridge estimator."""

from __future__ import annotations

import unittest

import numpy as np
from sklearn.base import clone

from sample_scaled_ridge import SampleScaledRidge


class SampleScaledRidgeTest(unittest.TestCase):
    def test_fit_scales_alpha_by_the_training_row_count(self) -> None:
        features = np.array([[0.0], [1.0], [2.0], [3.0]])
        target = np.array([0.0, 1.0, 2.0, 3.0])

        model = SampleScaledRidge(alpha_per_sample=0.05).fit(features, target)

        self.assertEqual(model.n_train_, 4)
        self.assertAlmostEqual(model.effective_alpha_, 0.2)
        self.assertAlmostEqual(model.alpha, 0.2)

    def test_negative_per_sample_penalty_is_rejected(self) -> None:
        model = SampleScaledRidge(alpha_per_sample=-0.01)

        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            model.fit(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]))

    def test_sklearn_clone_preserves_constructor_parameters(self) -> None:
        model = SampleScaledRidge(
            alpha_per_sample=0.03,
            fit_intercept=False,
            solver="lsqr",
        )

        cloned = clone(model)

        self.assertEqual(cloned.get_params(), model.get_params())
        self.assertIsNot(cloned, model)


if __name__ == "__main__":
    unittest.main()
