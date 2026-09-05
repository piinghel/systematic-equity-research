# Systematic equity research

Code and portfolio-level evidence for my [research notes](https://piinghel.github.io/).

| Article | Included material |
| --- | --- |
| [Ridge and stock rankings](https://piinghel.github.io/quants/2025/02/09/multiple-linear-regression.html) | Sample-scaled estimator, worked example and tests |
| [Joint sizing and trading controls](https://piinghel.github.io/quants/2026/08/29/portfolio-optimization.html) | One-rebalance control example, saved results and figure generators |
| [Combining rebalance weeks](https://piinghel.github.io/quants/2025/05/10/rebalancing-luck.html) | Sleeve example, daily portfolio returns, mixture calculation and metrics |

The [low-volatility backtest](https://github.com/piinghel/low-vol-to-portfolio)
has its own repository. The blog keeps the [size-choice diagnostic and
factor-correlation renderer](https://github.com/piinghel/piinghel.github.io#figure-sources).

## Start with the methods

Python 3.12 or later and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --locked --extra examples
uv run --extra examples python -m examples.ridge
uv run --extra examples python -m examples.trading_controls
uv run --extra examples python -m examples.tranching
```

The examples use invented inputs, run in seconds, and do not write files:

- **Ridge:** repeating observations leaves the fitted coefficients unchanged
  when the penalty scales with the number of training rows. This calls the
  [included estimator](sample_scaled_ridge.py), not a second implementation.
- **Trading controls:** inspect four stocks, their pre-trade weights, the
  eligibility rule, objective and constraints in
  [one short example](examples/trading_controls.py). A buffer allows an incumbent
  to remain; a penalty makes replacement less attractive. Forced exits remain
  compulsory. The example is long-only and uses illustrative settings—not the
  article's long/short portfolio, historical parameters or execution model.
  Its penalty acts on sizing scores, not calibrated cash costs. The problem is
  expressed directly in [CVXPY](https://www.cvxpy.org/tutorial/intro/index.html).
- **Tranching:** display two schedule rotations, then call the existing mixture
  calculation on invented daily returns. This combines portfolio P&L; it does
  not simulate the holdings that generated each schedule.

## Reproduce the saved evidence

```bash
uv run python -m portfolio_optimization.performance_figure
uv run python -m portfolio_optimization.parameter_sensitivity_figure
uv run python -m portfolio_optimization.rho_ladder_figure
uv run python -m portfolio_optimization.risk_calibration_figure
```

The commands write light/dark SVGs to `outputs/review/figures/`. Their inputs
are included, so these four figures can be rebuilt from a clean checkout.
The performance and parameter-sensitivity commands also write phone-layout
light/dark SVGs; their manifests list all four variants. These display changes
do not recompute portfolios or normalize their different risk levels.
The performance renderer defaults to the saved portfolio path; `--daily-source`
selects another Parquet input with the same schema.

To recompute the timing table from the three standalone daily portfolios:

```bash
uv run python -m portfolio_optimization.rebalance_timing
```

This prints the Development and Later results without writing another copy of
the evidence. Use `--input path/to/timing_daily.parquet` for another input.

The [timing figure renderer](https://github.com/piinghel/piinghel.github.io/blob/main/scripts/render_timing_figure.py)
is maintained in the blog repository and reads the aggregate metrics CSV.

## What the data measure

- Optimization tables average statistics across three standalone schedules.
  The performance path averages their separately compounded indices.
- Timing mixtures average daily P&L per unit of fixed notional first, then
  recompute return, volatility, Sharpe, and drawdown.
- Annualization uses 252 sessions. Geometric return compounds daily returns;
  arithmetic cost drag averages daily gross-minus-net P&L.
- Development ends in December 2021. January 2022–May 2026 subsequently informed
  research choices elsewhere in the series.

The CSV and Parquet files contain portfolio-level series and statistics, not
stock-level inputs. Rebuilding these figures and tables reproduces the saved
evidence, not the underlying stock-selection backtests. The Ridge example
demonstrates the estimator's penalty convention; this checkout does not yet
reproduce the article's matched portfolio and coefficient comparison.

`SOURCE_FILES.json` records hashes of the included research inputs and the
original imported modules at publication. Code can evolve independently of those
source snapshots; the included input hashes remain unchanged. Generated figures
and caches are ignored. Update inputs only after checking their periods and
definitions against the published article.

## Checks

```bash
uv run --extra examples ruff check .
uv run --extra examples ruff format --check .
uv run --extra examples ty check .
uv run --extra examples pytest -q
```
