"""Calculate fixed-notional mixtures of three saved rebalance schedules."""

from __future__ import annotations

import itertools
import math

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


def mixtures(frame: pl.DataFrame) -> pl.DataFrame:
    """Allocate equal fixed notional to every member of each non-empty subset."""
    validate_schedules(frame)
    pieces = []
    for count in (1, 2, 3):
        for members in itertools.combinations(range(3), count):
            pieces.append(
                frame.lazy()
                .filter(pl.col("offset").is_in(members))
                .group_by("date")
                .agg(pl.col("gross").mean(), pl.col("net").mean())
                .sort("date")
                .with_columns(
                    pl.lit("+".join(str(i + 1) for i in members)).alias("schedules"),
                    pl.lit(count).alias("sleeves"),
                )
            )
    return pl.concat(pieces).collect()


def summarize(frame: pl.DataFrame) -> pl.DataFrame:
    """Use 252 sessions, zero cash rate, and an initial unit wealth high-water mark."""
    ordered = (
        frame.lazy()
        .sort("schedules", "date")
        .with_columns((1 + pl.col("net")).cum_prod().over("schedules").alias("wealth"))
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
        .collect()
    )
