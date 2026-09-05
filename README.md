# Systematic equity research

Code and portfolio-level evidence for my [research notes](https://piinghel.github.io/).

| Article | Included material |
| --- | --- |
| [Ridge and stock rankings](https://piinghel.github.io/quants/2025/02/09/multiple-linear-regression.html) | Sample-scaled Ridge estimator and tests |
| [Joint sizing and trading controls](https://piinghel.github.io/quants/2026/08/29/portfolio-optimization.html) | Four figure generators and their saved inputs |
| [Combining rebalance weeks](https://piinghel.github.io/quants/2025/05/10/rebalancing-luck.html) | Daily portfolio returns, mixture calculation, and aggregate metrics |

The [low-volatility backtest](https://github.com/piinghel/low-vol-to-portfolio)
has its own repository. The blog keeps the [size-choice diagnostic and
factor-correlation renderer](https://github.com/piinghel/piinghel.github.io#figure-sources).

## Run

Python 3.12 or later and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --locked
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
selects another reviewed Parquet input. Raw backtest extraction stays in the
private research project.

To recompute the timing table from the three standalone daily portfolios:

```bash
uv run python - <<'PY'
import polars as pl
from portfolio_optimization.rebalance_timing import mixtures, summarize

saved = pl.read_parquet('outputs/review/timing/timing_daily.parquet')
for period in ('development', 'later'):
    singles = (
        saved.filter((pl.col('period') == period) & (pl.col('sleeves') == 1))
        .with_columns((pl.col('schedules').cast(pl.Int64) - 1).alias('offset'))
        .select('date', 'offset', 'gross', 'net')
    )
    print(period, summarize(mixtures(singles)))
PY
```

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

The CSV and Parquet files contain aggregate portfolio series and statistics.
The full stock-selection backtests require licensed security-level inputs and
the author's shared research packages. The matched OLS–Ridge return and
coefficient bundle remains missing; the estimator here exposes its penalty
convention. The existing article preserves the reported empirical comparison.

`SOURCE_FILES.json` records hashes of the included research inputs and the
original imported modules at publication. Code can evolve independently of those
source snapshots; the included input hashes remain unchanged. Generated figures
and caches are ignored. The private research project
owns the original backtest runs; update public inputs only after checking their
periods and definitions against the article.

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check .
uv run pytest -q
```
