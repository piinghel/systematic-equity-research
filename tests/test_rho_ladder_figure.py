import polars as pl
import pytest

from portfolio_optimization.rho_ladder_figure import LIGHT, PANELS, build_svg


def test_rho_svg_contains_all_cells_and_no_mobile_variant() -> None:
    rows = [
        {
            "allocator": allocator,
            "rho": rho / 10,
            "realised_to_predicted_volatility": 1.2 + rho / 100,
            "beta_mae": 0.18 + rho / 1000,
            "executed_turnover_l1_annualized": 40 - rho,
            "net_sharpe": 1.4 - rho / 100,
        }
        for allocator in ("b2", "b3")
        for rho in range(11)
    ]

    svg = build_svg(pl.DataFrame(rows), palette=LIGHT)

    assert len(PANELS) == 4
    assert svg.count("<circle") == 88
    assert "Optimizer" in svg
    assert "Optimizer + trading controls" in svg
    assert 'x1="90.0" y1="20.0"' in svg
    assert 'x1="250.0" y1="20.0"' in svg
    assert "Current 0.5" not in svg
    assert "Full history" not in svg
    assert "Mean across three schedules" in svg
    assert "Horizontal axis:" not in svg
    assert "Correlation shrinkage, ρ" not in svg
    assert 'height="725"' in svg
    assert "Two-way turnover (× capital)" in svg
    assert "Correlation shrinkage (rho)" not in svg
    assert "mobile" not in svg.lower()


def test_rho_svg_rejects_an_incomplete_allocator_grid() -> None:
    rows = [
        {
            "allocator": allocator,
            "rho": rho / 10,
            "realised_to_predicted_volatility": 1.2 + rho / 100,
            "beta_mae": 0.18 + rho / 1000,
            "executed_turnover_l1_annualized": 40 - rho,
            "net_sharpe": 1.3 - rho / 100,
        }
        for allocator in ("b2", "b3")
        for rho in range(11)
    ]

    with pytest.raises(ValueError, match="complete B2/B3"):
        build_svg(pl.DataFrame(rows[:-1]), palette=LIGHT)
