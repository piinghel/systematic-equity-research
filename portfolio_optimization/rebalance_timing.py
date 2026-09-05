"""Audit and visualize fixed-notional mixtures of three saved rebalance schedules."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import math
from pathlib import Path

import polars as pl


def validate_schedules(frame: pl.DataFrame) -> None:
    """Require all three offsets on every date without duplicate observations."""
    if frame.is_empty() or set(frame["offset"]) != {0, 1, 2}:
        raise ValueError("Expected three non-empty schedules")
    if frame.select("date", "offset").n_unique() != frame.height:
        raise ValueError("Duplicate schedule dates")
    if frame.filter(
        pl.col("date").is_null()
        | pl.any_horizontal(
            pl.col(c).is_null() | ~pl.col(c).is_finite() | (pl.col(c) <= -1)
            for c in ("gross", "net")
        )
    ).height:
        raise ValueError("Invalid return or date")
    if frame.group_by("date").len().filter(pl.col("len") != 3).height:
        raise ValueError("Schedules must have exactly matched dates")


def load_schedules(calendar_root: Path) -> pl.DataFrame:
    """Require identical dates, finite returns, and one record per schedule-date."""
    frames = []
    for offset in range(3):
        frame = (
            pl.scan_csv(
                calendar_root / f"o{offset}" / "returns.csv", try_parse_dates=True
            )
            .filter((pl.col("date") >= dt.date(1998, 9, 22)) | pl.col("date").is_null())
            .select(
                "date",
                pl.lit(offset).alias("offset"),
                pl.col("long_short_gross").alias("gross"),
                pl.col("long_short_net").alias("net"),
            )
            .sort("date")
            .collect()
        )
        if frame.is_empty() or frame["date"].n_unique() != frame.height:
            raise ValueError("Empty or duplicate schedule dates")
        if frame.filter(
            pl.col("date").is_null()
            | pl.any_horizontal(
                pl.col(c).is_null() | ~pl.col(c).is_finite() | (pl.col(c) <= -1)
                for c in ("gross", "net")
            )
        ).height:
            raise ValueError("Invalid return or date")
        if frames and not frame["date"].equals(frames[0]["date"]):
            raise ValueError("Schedules must have exactly matched dates")
        frames.append(frame)
    combined = pl.concat(frames)
    validate_schedules(combined)
    return combined


def mixtures(frame: pl.DataFrame) -> pl.DataFrame:
    """Allocate equal fixed notional to every member of each non-empty subset."""
    validate_schedules(frame)
    pieces = []
    for count in (1, 2, 3):
        for members in itertools.combinations(range(3), count):
            pieces.append(
                frame.filter(pl.col("offset").is_in(members))
                .group_by("date")
                .agg(pl.col("gross").mean(), pl.col("net").mean())
                .sort("date")
                .with_columns(
                    pl.lit("+".join(str(i + 1) for i in members)).alias("schedules"),
                    pl.lit(count).alias("sleeves"),
                )
            )
    return pl.concat(pieces)


def summarize(frame: pl.DataFrame) -> pl.DataFrame:
    """Use 252 sessions, zero cash rate, and an initial unit wealth high-water mark."""
    ordered = frame.sort("schedules", "date").with_columns(
        (1 + pl.col("net")).cum_prod().over("schedules").alias("wealth")
    )
    return (
        ordered.group_by("schedules", "sleeves")
        .agg(
            pl.col("date").min().alias("start"),
            pl.col("date").max().alias("end"),
            pl.len().alias("days"),
            (((1 + pl.col("gross")).product() ** (252 / pl.len()) - 1) * 100).alias(
                "gross_cagr"
            ),
            (((1 + pl.col("net")).product() ** (252 / pl.len()) - 1) * 100).alias(
                "net_cagr"
            ),
            (pl.col("net").mean() * 252 * 100).alias("net_arithmetic"),
            (pl.col("net").std() * math.sqrt(252) * 100).alias("volatility"),
            (pl.col("net").mean() / pl.col("net").std() * math.sqrt(252)).alias(
                "sharpe"
            ),
            (
                (
                    pl.col("wealth") / pl.col("wealth").cum_max().clip(lower_bound=1)
                    - 1
                ).min()
                * 100
            ).alias("drawdown"),
            ((pl.col("gross") - pl.col("net")).mean() * 252 * 100).alias(
                "arithmetic_cost"
            ),
        )
        .sort("sleeves", "schedules")
    )


def run(calendar_root: Path, output: Path) -> None:
    frame = load_schedules(calendar_root)
    all_mixtures = mixtures(frame)
    periods = []
    for period, start, end in (
        ("development", dt.date(1998, 9, 22), dt.date(2021, 12, 31)),
        ("later", dt.date(2022, 1, 3), dt.date(2026, 5, 27)),
    ):
        periods.append(
            all_mixtures.filter(pl.col("date").is_between(start, end)).with_columns(
                pl.lit(period).alias("period")
            )
        )
    daily = pl.concat(periods)
    metrics = pl.concat(
        [
            summarize(p).with_columns(pl.lit(p["period"][0]).alias("period"))
            for p in periods
        ]
    )
    output.mkdir(parents=True, exist_ok=True)
    daily.write_parquet(output / "timing_daily.parquet")
    metrics.write_csv(output / "timing_metrics.csv")
    manifest = {
        "rule": "Frozen B3: Ridge ranking, constrained optimizer with trading controls",
        "aggregation": "Equal fixed-notional mixture of saved daily returns; no new forecasts or trades",
        "annualization": 252,
        "cash_rate": 0,
        "costs": "Existing 5 bp proportional costs, averaged without cross-sleeve netting",
        "sources": {
            f"o{i}/returns.csv": hashlib.sha256(
                (calendar_root / f"o{i}/returns.csv").read_bytes()
            ).hexdigest()
            for i in range(3)
        },
        "later_period": "Reused history, not an untouched holdout",
    }
    (output / "timing_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calendar-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.calendar_root, args.output)


if __name__ == "__main__":
    main()
