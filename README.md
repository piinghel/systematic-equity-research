# Systematic equity research

Code for [Ridge and stock rankings](https://piinghel.github.io/quants/2025/02/09/multiple-linear-regression.html)
and an index of the related research repositories.

## Research repositories

| Study | Code |
| --- | --- |
| Low-volatility sizing | [low-vol-to-portfolio](https://github.com/piinghel/low-vol-to-portfolio) |
| Portfolio optimization | [portfolio-optimization-study](https://github.com/piinghel/portfolio-optimization-study) |
| Rebalance tranching | [rebalance-tranching](https://github.com/piinghel/rebalance-tranching) |
| Ridge estimator | This repository |

The optimizer and tranching examples, calculations, figure sources and saved inputs
now live in their dedicated repositories. Their earlier versions remain in Git history.

## Run the Ridge example

With Python 3.12 or later and [uv](https://docs.astral.sh/uv/), from this repository:

```bash
uv sync --locked
uv run python -m ridge_example
```

The [example](ridge_example.py) calls the [sample-scaled estimator](sample_scaled_ridge.py)
on invented, overlapping predictors. Repeating the same observations adds no
information. Scaling the penalty by the number of training rows keeps the fitted
coefficients unchanged instead of weakening regularization.

This demonstrates the penalty convention, not walk-forward validation or the
article's matched portfolio and coefficient comparison. Those empirical results
are not reproduced by this example.

The blog owns the [regression figure sources](https://github.com/piinghel/piinghel.github.io#figure-sources).

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check .
uv run pytest -q
```
