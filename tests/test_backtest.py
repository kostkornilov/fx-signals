import numpy as np
import pandas as pd

from fx_signal.backtest import (
    PRIMARY_METRICS,
    purged_validation_mask,
    run_walk_forward_backtest,
)
from fx_signal.splits import WalkForwardFold, mask_val
from fx_signal.targets import add_targets


def _frame() -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=100)
    step = np.arange(len(dates))
    rates = pd.DataFrame(
        {
            "currency": "KZT",
            "effective_date": dates,
            "rub_per_unit": 0.18 + 0.005 * np.sin(step / 4),
            "feature": np.sin(step / 4) + np.cos(step / 9),
            "has_fact": True,
        }
    )
    return add_targets(rates, horizon=2)


def _fold() -> WalkForwardFold:
    return WalkForwardFold(
        name="wf_test",
        split="wf_oos",
        train_end=pd.Timestamp("2023-03-01"),
        val_start=pd.Timestamp("2023-03-01"),
        val_end=pd.Timestamp("2023-04-03"),
        test_start=pd.Timestamp("2023-04-03"),
        test_end=pd.Timestamp("2023-05-22"),
    )


def test_validation_tail_is_purged_per_currency() -> None:
    frame = _frame()
    fold = _fold()
    raw_validation = mask_val(frame, fold)
    purged = purged_validation_mask(frame, fold, horizon=2)

    assert int(purged.sum()) == int(raw_validation.sum()) - 2
    assert frame.loc[purged, "effective_date"].max() < frame.loc[raw_validation, "effective_date"].max()


def test_backtest_returns_predictions_metrics_and_audit_counts() -> None:
    frame = _frame()
    fold = _fold()
    result = run_walk_forward_backtest(
        frame,
        model_kind="logreg",
        feature_cols=["feature"],
        target_col="target_stay_not_worse_h2",
        horizon=2,
        folds=[fold],
        threshold_grid=[0.5],
        quantile_rates=[],
        target_signals_per_week=(0.0, 10.0),
    )

    test_rows = frame["effective_date"].ge(fold.test_start) & frame["effective_date"].lt(
        fold.test_end
    )
    assert result.predictions.loc[test_rows, "backtest_score"].notna().all()
    assert result.predictions.loc[~test_rows, "backtest_score"].isna().all()
    assert result.predictions.loc[test_rows, "backtest_fold"].eq("wf_test").all()
    assert result.thresholds.loc[0, "validation_rows"] == int(mask_val(frame, fold).sum()) - 2
    assert set(PRIMARY_METRICS).issubset(result.fold_metrics.columns)
    assert set(result.stability["metric"]) == set(PRIMARY_METRICS)


def test_backtest_rejects_overlapping_test_folds() -> None:
    frame = _frame()
    first = _fold()
    overlapping = WalkForwardFold(
        name="overlap",
        split="wf_oos",
        train_end=pd.Timestamp("2023-03-01"),
        val_start=pd.Timestamp("2023-03-01"),
        val_end=pd.Timestamp("2023-04-10"),
        test_start=pd.Timestamp("2023-04-10"),
        test_end=pd.Timestamp("2023-05-22"),
    )

    try:
        run_walk_forward_backtest(
            frame,
            model_kind="logreg",
            feature_cols=["feature"],
            target_col="target_stay_not_worse_h2",
            horizon=2,
            folds=[first, overlapping],
            threshold_grid=[0.5],
            quantile_rates=[],
        )
    except ValueError as error:
        assert "overlaps a previous test window" in str(error)
    else:
        raise AssertionError("overlapping test folds must be rejected")
