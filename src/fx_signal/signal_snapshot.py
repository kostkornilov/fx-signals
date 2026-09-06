from __future__ import annotations

from collections.abc import Collection
from pathlib import Path

import pandas as pd

from fx_signal.backtest import WalkForwardPredictions, walk_forward_predictions
from fx_signal.data import load_rates, load_yaml
from fx_signal.features import add_features, columns_for_groups
from fx_signal.indicators import add_research_indicators
from fx_signal.meta import add_meta_decisions
from fx_signal.splits import WalkForwardFold
from fx_signal.targets import add_targets, target_column
from fx_signal.texts import load_text_library, render_message

DEFAULT_ALLOWED_SIGNALS = (
    "signal_momentum",
    "signal_level",
    "signal_seasonality",
    "signal_better_than_one_year_ago",
    "signal_better_range_held",
    "signal_larger_than_usual_latest_improvement",
    "signal_better_than_30_day_average",
    "signal_most_recent_changes_favourable",
)

OUTPUT_COLUMNS = (
    "as_of",
    "effective_date",
    "corridor",
    "indicator",
    "direction",
    "strength",
    "speed",
    "recommended_scenario",
    "push_title",
    "push_body",
    "market_score",
    "market_threshold",
    "selected_effect",
    "selection_score",
    "decision_reason",
)


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _parse_as_of(value: str | pd.Timestamp) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError("as_of must be a valid date")
    if parsed.tzinfo is not None:
        parsed = parsed.tz_localize(None)
    return parsed.normalize()


def _point_in_time_rows(frame: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    if "available_at" not in frame:
        raise KeyError("Input data must contain available_at")
    available = pd.to_datetime(frame["available_at"], errors="coerce")
    if available.isna().any():
        raise ValueError("available_at must not contain missing or invalid values")
    return frame.loc[available.le(as_of)].copy()


def prepare_snapshot_frame(
    config: dict,
    repo_root: Path,
    as_of: str | pd.Timestamp,
) -> pd.DataFrame:
    """Build features after removing everything that was unavailable at the cutoff."""
    cutoff = _parse_as_of(as_of)
    rates = _point_in_time_rows(load_rates(_resolve(repo_root, config["data_path"])), cutoff)
    corridors = config.get("corridors")
    if corridors:
        rates = rates.loc[rates["currency"].isin(corridors)].copy()
    if rates.empty:
        raise ValueError(f"No corridor rates are available at or before {cutoff.date()}")

    context = None
    if config.get("context_path"):
        context_path = _resolve(repo_root, config["context_path"])
        if context_path.exists():
            context = _point_in_time_rows(load_rates(context_path), cutoff)

    horizon = int(config["horizon"])
    frame = add_targets(rates, horizon=horizon)
    frame = add_features(
        frame,
        context=context,
        momentum_days=int(config.get("momentum_days", 3)),
        level_window=int(config.get("level_window", 90)),
        level_quantile=float(config.get("level_quantile", 0.10)),
        reversal_window=int(config.get("reversal_window", 20)),
        holiday_lookahead_days=int(config.get("holiday_lookahead_days", 7)),
    )
    return add_research_indicators(frame, **dict(config.get("research_indicators", {})))


def _snapshot_fold(config: dict, as_of: pd.Timestamp) -> WalkForwardFold:
    test_start = as_of - pd.offsets.Day(int(config.get("policy_lookback_days", 10)))
    validation_start = test_start - pd.offsets.Day(int(config.get("validation_days", 365)))
    return WalkForwardFold(
        name=f"snapshot_{as_of.date().isoformat()}",
        split="snapshot",
        train_end=validation_start,
        val_start=validation_start,
        val_end=test_start,
        test_start=test_start,
        test_end=as_of + pd.offsets.Day(1),
    )


def apply_signal_policy(
    frame: pd.DataFrame,
    *,
    candidate_col: str,
    cooldown_days: int,
    weekly_limit: int,
) -> pd.Series:
    """Apply deterministic per-corridor cooldown and calendar-week budget."""
    if cooldown_days < 0:
        raise ValueError("cooldown_days must be non-negative")
    if weekly_limit < 0:
        raise ValueError("weekly_limit must be non-negative")
    accepted = pd.Series(False, index=frame.index, dtype=bool)
    for _, group in frame.groupby("currency", sort=False):
        last_sent: pd.Timestamp | None = None
        weekly_counts: dict[pd.Period, int] = {}
        for index, row in group.sort_values("effective_date", kind="stable").iterrows():
            if not bool(row[candidate_col]):
                continue
            date = pd.Timestamp(row["effective_date"])
            week = date.to_period("W-SUN")
            if last_sent is not None and (date - last_sent).days <= cooldown_days:
                continue
            if weekly_counts.get(week, 0) >= weekly_limit:
                continue
            accepted.loc[index] = True
            last_sent = date
            weekly_counts[week] = weekly_counts.get(week, 0) + 1
    return accepted


def _empty_output() -> pd.DataFrame:
    return pd.DataFrame(columns=list(OUTPUT_COLUMNS))


def apply_meta_policy_to_predictions(
    frame: pd.DataFrame,
    predictions: WalkForwardPredictions,
    *,
    allowed_signals: Collection[str],
    cooldown_days: int,
    weekly_limit: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Turn model predictions into the final explainable, thinned signal stream."""
    predicted = predictions.fold_names.notna()
    work = frame.loc[predicted].copy()
    result = pd.Series(False, index=frame.index, dtype=bool)
    if work.empty:
        return work, result

    threshold_by_fold = predictions.thresholds.set_index("fold")["threshold"].to_dict()
    work["market_score"] = predictions.scores.reindex(work.index)
    work["market_model_signal"] = (
        predictions.signals.reindex(work.index).fillna(False).astype(bool)
    )
    work["market_threshold"] = (
        predictions.fold_names.reindex(work.index).map(threshold_by_fold).astype(float)
    )
    work = add_meta_decisions(
        work,
        score_col="market_score",
        threshold="market_threshold",
        freshness_col="is_fresh",
        allowed_signals=allowed_signals,
    )
    work["candidate_signal"] = work["market_model_signal"] & work["meta_should_send"]
    work["policy_signal"] = apply_signal_policy(
        work,
        candidate_col="candidate_signal",
        cooldown_days=cooldown_days,
        weekly_limit=weekly_limit,
    )
    result.loc[work.index] = work["policy_signal"]
    return work, result


def build_signal_snapshot(
    config: dict,
    repo_root: Path,
    as_of: str | pd.Timestamp,
) -> pd.DataFrame:
    cutoff = _parse_as_of(as_of)
    frame = prepare_snapshot_frame(config, repo_root, cutoff)
    allowed: Collection[str] = tuple(config.get("allowed_signals", DEFAULT_ALLOWED_SIGNALS))
    fold = _snapshot_fold(config, cutoff)
    feature_cols = columns_for_groups(list(config.get("feature_groups", ["A", "B"])), frame)
    if not feature_cols:
        raise ValueError("Configured feature groups do not resolve to any available columns")

    horizon = int(config["horizon"])
    predictions = walk_forward_predictions(
        frame,
        model_kind=str(config.get("model", "catboost")),
        feature_cols=feature_cols,
        target_col=target_column(str(config.get("target", "stay_not_worse")), horizon),
        horizon=horizon,
        folds=[fold],
        threshold_grid=config.get("thresholds", [0.6, 0.7, 0.8, 0.9]),
        quantile_rates=config.get("quantile_rates", [0.10, 0.15, 0.20]),
        target_signals_per_week=tuple(
            map(float, config.get("target_signals_per_week", [0.8, 2.5]))
        ),
        default_threshold=float(config.get("default_threshold", 0.8)),
    )
    work, _ = apply_meta_policy_to_predictions(
        frame,
        predictions,
        allowed_signals=allowed,
        cooldown_days=int(config.get("cooldown_days", 3)),
        weekly_limit=int(config.get("weekly_limit", 2)),
    )

    latest_indexes = work.groupby("currency", sort=False)["effective_date"].idxmax()
    latest = work.loc[latest_indexes].copy()
    age = cutoff - pd.to_datetime(latest["available_at"]).dt.normalize()
    max_staleness = int(config.get("max_staleness_days", 3))
    latest = latest.loc[age.dt.days.le(max_staleness) & latest["policy_signal"]].copy()
    if latest.empty:
        return _empty_output()

    texts_path = _resolve(repo_root, config.get("texts_path", "configs/texts.yaml"))
    library = load_text_library(texts_path, required_signals=set(allowed))
    rows: list[dict[str, object]] = []
    for row in latest.sort_values("currency", kind="stable").to_dict(orient="records"):
        signal = str(row["meta_selected_signal"])
        template = library.scenarios[signal]
        effect = row["meta_selected_effect"]
        parsed_effect = None if pd.isna(effect) else float(effect)
        title, body = render_message(
            template,
            currency=str(row["currency"]),
            effect=parsed_effect,
        )
        rows.append(
            {
                "as_of": cutoff.date().isoformat(),
                "effective_date": pd.Timestamp(row["effective_date"]).date().isoformat(),
                "corridor": f"RUB->{row['currency']}",
                "indicator": signal,
                "direction": template.direction,
                "strength": abs(parsed_effect) if parsed_effect is not None else pd.NA,
                "speed": template.speed,
                "recommended_scenario": template.scenario,
                "push_title": title,
                "push_body": body,
                "market_score": float(row["market_score"]),
                "market_threshold": float(row["market_threshold"]),
                "selected_effect": parsed_effect if parsed_effect is not None else pd.NA,
                "selection_score": row["meta_selection_score"],
                "decision_reason": row["meta_reason"],
            }
        )
    return pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS))


def run_signal_snapshot(
    config_path: Path,
    *,
    as_of: str | pd.Timestamp,
    output_path: Path | None = None,
) -> Path:
    config_path = config_path.resolve()
    repo_root = config_path.parent.parent
    cutoff = _parse_as_of(as_of)
    result = build_signal_snapshot(load_yaml(config_path), repo_root, cutoff)
    destination = output_path or Path("artifacts/signals") / f"signals-{cutoff.date()}.csv"
    destination = _resolve(repo_root, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    return destination
