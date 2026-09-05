from __future__ import annotations

from pathlib import Path

import pandas as pd

from fx_signal.data import load_rates, load_yaml
from fx_signal.indicators import add_baseline_indicators
from fx_signal.metrics.lift import evaluate_lift
from fx_signal.targets import add_local_min_target


def build_baseline_frame(
    config_path: Path, horizon: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config_path = config_path.resolve()
    repo_root = config_path.parent.parent
    config = load_yaml(config_path)
    if horizon is None:
        configured = config.get("target_horizons", [config.get("target_horizon", 3)])
        horizon = int(configured[0])
    data_path = Path(config["data_path"])
    if not data_path.is_absolute():
        data_path = repo_root / data_path
    rates = load_rates(data_path)
    frame = add_local_min_target(rates, horizon=horizon)
    frame = add_baseline_indicators(
        frame,
        momentum_days=int(config["momentum_days"]),
        level_window=int(config["level_window"]),
        level_quantile=float(config["level_quantile"]),
        reversal_window=int(config["reversal_window"]),
        holiday_lookahead_days=int(config["holiday_lookahead_days"]),
    )
    return frame, evaluate_lift(frame, horizon=horizon)


def run_baseline(config_path: Path) -> Path:
    config_path = config_path.resolve()
    repo_root = config_path.parent.parent
    config = load_yaml(config_path)
    horizons = [int(value) for value in config.get("target_horizons", [3])]
    metrics_by_horizon: list[pd.DataFrame] = []
    for horizon in horizons:
        _, horizon_metrics = build_baseline_frame(config_path, horizon=horizon)
        metrics_by_horizon.append(horizon_metrics)

    metrics = pd.concat(metrics_by_horizon, ignore_index=True)
    output = repo_root / Path(config["reports_dir"]) / "tables" / "baseline_all_horizons.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output, index=False)
    for horizon, horizon_metrics in metrics.groupby("horizon"):
        horizon_metrics.to_csv(output.parent / f"baseline_h{horizon}.csv", index=False)
    return output
