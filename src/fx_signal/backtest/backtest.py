from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fx_signal.metrics import (
    add_customer_outcomes,
    evaluate_method,
    summarize_stability,
)
from fx_signal.models import fit_scorer, select_threshold
from fx_signal.splits import (
    WalkForwardFold,
    assert_no_overlap,
    make_walk_forward_folds,
    mask_test,
    mask_train,
    mask_val,
)

PRIMARY_METRICS = (
    "lift",
    "moment_advantage_bps",
    "customer_regret_cvar_95_bps",
    "useful_signals_per_week",
    "pushes_per_week",
)


@dataclass(frozen=True)
class WalkForwardPredictions:
    """Out-of-sample predictions and an audit table for each fitted fold."""

    scores: pd.Series
    signals: pd.Series
    fold_names: pd.Series
    split_names: pd.Series
    thresholds: pd.DataFrame


@dataclass(frozen=True)
class BacktestResult:
    """All artifacts produced by a walk-forward backtest."""

    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    stability: pd.DataFrame
    thresholds: pd.DataFrame


def _validate_frame(
    frame: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    target_col: str,
    horizon: int,
) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if not feature_cols:
        raise ValueError("feature_cols must contain at least one feature")
    required = {
        "currency",
        "effective_date",
        "has_fact",
        "rub_per_unit",
        f"forward_mean_h{horizon}",
        target_col,
        *feature_cols,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if not frame.index.is_unique:
        raise ValueError("frame index must be unique")

    work = frame.copy()
    work["effective_date"] = pd.to_datetime(work["effective_date"], errors="coerce")
    if work["effective_date"].isna().any():
        raise ValueError("effective_date must not contain missing or invalid dates")
    if work.duplicated(["currency", "effective_date"]).any():
        raise ValueError("frame must have at most one row per currency and effective_date")
    return work.sort_values(["currency", "effective_date"], kind="stable")


def purged_validation_mask(
    frame: pd.DataFrame,
    fold: WalkForwardFold,
    *,
    horizon: int,
) -> pd.Series:
    """Select validation rows whose forward labels are known before test starts.

    A target at time ``t`` uses the following ``horizon`` publications. The last
    ``horizon`` rows of each currency's validation window would therefore look
    into the test window and are removed.
    """
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    validation = mask_val(frame, fold)
    keep = pd.Series(False, index=frame.index)
    for _, group in frame.loc[validation].groupby("currency", sort=False):
        group = group.sort_values("effective_date", kind="stable")
        if horizon:
            group = group.iloc[:-horizon] if len(group) > horizon else group.iloc[0:0]
        keep.loc[group.index] = True
    return keep


def walk_forward_predictions(
    frame: pd.DataFrame,
    *,
    model_kind: str,
    feature_cols: Sequence[str],
    target_col: str,
    horizon: int,
    folds: Sequence[WalkForwardFold] | None = None,
    first_test_year: int = 2022,
    oot_start: str | pd.Timestamp = "2025-09-01",
    threshold_grid: Iterable[float] = (0.6, 0.7, 0.8, 0.9),
    quantile_rates: Iterable[float] = (0.10, 0.15, 0.20),
    target_signals_per_week: tuple[float, float] = (0.8, 2.5),
    default_threshold: float = 0.8,
) -> WalkForwardPredictions:
    """Fit on past data and predict each later test window exactly once.

    Training expands over time. The immediately preceding validation window is
    used for probability calibration and threshold choice; its last ``horizon``
    observations per currency are purged. No model is trained on a row from its
    own validation or test window.
    """
    work = _validate_frame(
        frame,
        feature_cols=feature_cols,
        target_col=target_col,
        horizon=horizon,
    )
    selected_folds = list(folds) if folds is not None else make_walk_forward_folds(
        work["effective_date"].min(),
        work["effective_date"].max(),
        first_test_year=first_test_year,
        oot_start=oot_start,
    )
    if not selected_folds:
        raise ValueError("No walk-forward folds can be formed for this date range")

    grid = [float(value) for value in threshold_grid]
    quantiles = [float(value) for value in quantile_rates]
    if not grid and not quantiles:
        raise ValueError("At least one fixed threshold or quantile rate is required")
    min_signals, max_signals = map(float, target_signals_per_week)
    if min_signals < 0 or max_signals < min_signals:
        raise ValueError("target_signals_per_week must be a valid non-negative range")
    if not 0.0 <= default_threshold <= 1.0:
        raise ValueError("default_threshold must be between 0 and 1")

    scores = pd.Series(np.nan, index=work.index, dtype=float, name="backtest_score")
    signals = pd.Series(pd.NA, index=work.index, dtype="boolean", name="backtest_signal")
    fold_names = pd.Series(pd.NA, index=work.index, dtype="string", name="backtest_fold")
    split_names = pd.Series(pd.NA, index=work.index, dtype="string", name="backtest_split")
    audit_rows: list[dict[str, object]] = []

    for fold in selected_folds:
        assert_no_overlap(work, fold, purge_horizon=horizon)
        train_mask = mask_train(work, fold, purge_horizon=horizon)
        validation_mask = purged_validation_mask(work, fold, horizon=horizon)
        test_mask = mask_test(work, fold)
        if (fold_names.notna() & test_mask).any():
            raise ValueError(f"Fold {fold.name} overlaps a previous test window")

        train = work.loc[train_mask].dropna(subset=[target_col])
        validation = work.loc[validation_mask].dropna(
            subset=[target_col, f"forward_mean_h{horizon}"]
        )
        test = work.loc[test_mask]
        if test.empty:
            raise ValueError(f"Fold {fold.name} has no test rows")
        if train.empty:
            raise ValueError(f"Fold {fold.name} has no training rows after purging")
        if train[target_col].nunique() < 2:
            raise ValueError(f"Fold {fold.name} training target has fewer than two classes")

        scorer = fit_scorer(
            train,
            validation,
            kind=model_kind,
            feature_cols=list(feature_cols),
            target_col=target_col,
        )
        if validation.empty:
            threshold = float(default_threshold)
        else:
            validation_scores = scorer.predict_proba(validation)
            threshold = select_threshold(
                validation,
                validation_scores,
                target_col=target_col,
                horizon=horizon,
                grid=grid,
                quantiles=quantiles,
                min_signals_per_week=min_signals,
                max_signals_per_week=max_signals,
            )

        test_scores = scorer.predict_proba(test)
        has_fact = test["has_fact"].fillna(False).astype(bool).to_numpy()
        test_signals = (test_scores >= threshold) & has_fact
        scores.loc[test.index] = test_scores
        signals.loc[test.index] = test_signals
        fold_names.loc[test.index] = fold.name
        split_names.loc[test.index] = fold.split
        audit_rows.append(
            {
                "fold": fold.name,
                "split": fold.split,
                "train_end": fold.train_end,
                "validation_start": fold.val_start,
                "validation_end": fold.val_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test),
                "threshold": threshold,
                "signal_count": int(test_signals.sum()),
            }
        )

    original_index = frame.index
    return WalkForwardPredictions(
        scores=scores.reindex(original_index),
        signals=signals.reindex(original_index),
        fold_names=fold_names.reindex(original_index),
        split_names=split_names.reindex(original_index),
        thresholds=pd.DataFrame(audit_rows),
    )


def run_walk_forward_backtest(
    frame: pd.DataFrame,
    *,
    model_kind: str,
    feature_cols: Sequence[str],
    target_col: str,
    horizon: int,
    folds: Sequence[WalkForwardFold] | None = None,
    first_test_year: int = 2022,
    oot_start: str | pd.Timestamp = "2025-09-01",
    threshold_grid: Iterable[float] = (0.6, 0.7, 0.8, 0.9),
    quantile_rates: Iterable[float] = (0.10, 0.15, 0.20),
    target_signals_per_week: tuple[float, float] = (0.8, 2.5),
    default_threshold: float = 0.8,
    weekly_limit: int = 2,
    cooldown_days: int = 3,
) -> BacktestResult:
    """Run the model and calculate all five product metrics per fold/corridor."""
    predictions = walk_forward_predictions(
        frame,
        model_kind=model_kind,
        feature_cols=feature_cols,
        target_col=target_col,
        horizon=horizon,
        folds=folds,
        first_test_year=first_test_year,
        oot_start=oot_start,
        threshold_grid=threshold_grid,
        quantile_rates=quantile_rates,
        target_signals_per_week=target_signals_per_week,
        default_threshold=default_threshold,
    )
    scored = frame.copy()
    scored["backtest_score"] = predictions.scores
    scored["backtest_signal"] = predictions.signals
    scored["backtest_fold"] = predictions.fold_names
    scored["backtest_split"] = predictions.split_names

    # Calculate price outcomes on the complete history before slicing test folds.
    # This keeps valid future observations just beyond a fold boundary available.
    scored = add_customer_outcomes(scored, horizon=horizon)
    metric_pieces: list[pd.DataFrame] = []
    for audit in predictions.thresholds.to_dict(orient="records"):
        fold_name = str(audit["fold"])
        test = scored.loc[scored["backtest_fold"].eq(fold_name)].copy()
        metrics = evaluate_method(
            test,
            horizon=horizon,
            method=model_kind,
            split=fold_name,
            target_col=target_col,
            signal_col="backtest_signal",
            weekly_limit=weekly_limit,
            cooldown_days=cooldown_days,
        )
        metrics.insert(3, "split_group", audit["split"])
        metrics["threshold"] = audit["threshold"]
        metrics["fold_test_start"] = audit["test_start"]
        metrics["fold_test_end"] = audit["test_end"]
        metric_pieces.append(metrics)

    fold_metrics = (
        pd.concat(metric_pieces, ignore_index=True) if metric_pieces else pd.DataFrame()
    )
    stability = summarize_stability(
        fold_metrics,
        PRIMARY_METRICS,
        lower_is_better=("customer_regret_cvar_95_bps", "pushes_per_week"),
    )
    return BacktestResult(
        predictions=scored,
        fold_metrics=fold_metrics,
        stability=stability,
        thresholds=predictions.thresholds,
    )
