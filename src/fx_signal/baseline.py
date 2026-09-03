from __future__ import annotations

from pathlib import Path

import pandas as pd

from fx_signal.data import load_rates, load_yaml
from fx_signal.evaluation import evaluate_lift
from fx_signal.indicators import add_baseline_indicators
from fx_signal.targets import add_local_min_target


def build_baseline_frame(config_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    config_path = config_path.resolve()
    repo_root = config_path.parent.parent
    config = load_yaml(config_path)
    horizon = int(config["target_horizon"])
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
    _, metrics = build_baseline_frame(config_path)
    output = repo_root / Path(config["reports_dir"]) / "tables" / "baseline_h3.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output, index=False)
    return output
