import numpy as np

from lib.strategy import optimize_scale_with_rolling_cv, VolatilityOverlay


def test_optimize_scale_with_rolling_cv_runs_and_returns_scale():
    preds = np.linspace(-0.02, 0.02, 400)
    returns = np.sin(np.linspace(0, 12, 400)) * 0.01
    result = optimize_scale_with_rolling_cv(preds, returns, candidate_scales=[5.0, 15.0], min_splits=3)
    assert result["scale"] in {5.0, 15.0}
    assert "cv_sharpe" in result
    assert result["n_splits"] >= 2


def test_volatility_overlay_reduces_allocations_when_vol_exceeds_cap():
    allocations = np.ones(200) * 1.8
    returns = np.concatenate([np.full(100, 0.001), np.full(100, 0.05)])
    overlay = VolatilityOverlay(lookback=20, min_periods=10, reference_is_lagged=False)
    result = overlay.transform(allocations, returns)
    assert np.all(result["allocations"] >= 0.0)
    assert result["allocations"][150] < 1.8


def test_volatility_overlay_handles_lagged_reference():
    base_returns = np.concatenate([np.full(10, 0.001), np.full(10, 0.02)])
    lagged = np.concatenate([[np.nan], base_returns[:-1]])
    allocations = np.ones_like(lagged) * 1.2
    overlay = VolatilityOverlay(lookback=5, min_periods=3, reference_is_lagged=True)
    result = overlay.transform(allocations, lagged)
    assert result["allocations"].shape == allocations.shape
    assert np.isfinite(result["allocations"]).all()
