"""Show why sample-scaled Ridge keeps the penalty relative to the data fixed."""

from __future__ import annotations

import numpy as np

from sample_scaled_ridge import SampleScaledRidge


def main() -> None:
    # Deliberately overlapping predictors. Repeating the same observations adds
    # no information; it should not weaken a per-observation regularization rule.
    features = np.array([[0.0, 0.1], [1.0, 1.1], [2.0, 1.9], [3.0, 3.2]])
    target = np.array([0.0, 1.0, 1.5, 3.0])
    for repeats in (1, 3):
        model = SampleScaledRidge(alpha_per_sample=0.05).fit(
            np.tile(features, (repeats, 1)), np.tile(target, repeats)
        )
        print(
            f"Rows: {model.n_train_:2d}; alpha: {model.effective_alpha_:.2f}; "
            f"coefficients: {np.round(model.coef_, 4)}"
        )
    print("Synthetic estimator example; not a walk-forward or portfolio comparison.")


if __name__ == "__main__":
    main()
