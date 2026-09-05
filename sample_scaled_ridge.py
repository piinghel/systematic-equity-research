"""Ridge estimator with a penalty scaled by the fitted sample count."""

from __future__ import annotations

from typing import Any, Self

from sklearn.linear_model import Ridge


class SampleScaledRidge(Ridge):
    """Keep the L2 penalty constant relative to an expanding training sample.

    Scikit-learn's Ridge objective uses a summed squared-error term. Setting
    ``alpha = alpha_per_sample * n_train`` therefore keeps ``alpha / n_train``
    fixed across expanding walk-forward folds.
    """

    def __init__(
        self,
        alpha_per_sample: float = 0.01,
        *,
        fit_intercept: bool = True,
        copy_X: bool = True,
        max_iter: int | None = None,
        tol: float = 0.0001,
        solver: str = "auto",
        positive: bool = False,
        random_state: int | None = None,
    ) -> None:
        self.alpha_per_sample = alpha_per_sample
        super().__init__(
            alpha=1.0,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            solver=solver,
            positive=positive,
            random_state=random_state,
        )

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Any | None = None,
    ) -> Self:
        """Set the fold-specific penalty from the exact fitted row count."""
        if self.alpha_per_sample < 0:
            raise ValueError("alpha_per_sample must be non-negative")
        self.n_train_ = int(X.shape[0])
        self.effective_alpha_ = float(self.alpha_per_sample * self.n_train_)
        self.alpha = self.effective_alpha_
        super().fit(X=X, y=y, sample_weight=sample_weight)
        return self
