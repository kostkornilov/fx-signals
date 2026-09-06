from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fx_signal.backtest import walk_forward_predictions
from fx_signal.data import load_rates, load_yaml
from fx_signal.explain import add_explanations
from fx_signal.external import load_external_series
from fx_signal.features import add_features, columns_for_groups
from fx_signal.metrics import evaluate_method
from fx_signal.splits import (
    make_walk_forward_folds,
    mask_test,
)
from fx_signal.targets import add_targets, target_column

RULE_METHODS = ("momentum", "level", "reversal", "seasonality")
MODEL_METHODS = ("logreg", "catboost")


def _repo_root(config_path: Path) -> Path:
    return config_path.resolve().parent.parent


def _git_hash(repo_root: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def build_frame(config: dict, repo_root: Path) -> pd.DataFrame:
    data_path = Path(config["data_path"])
    if not data_path.is_absolute():
        data_path = repo_root / data_path
    rates = load_rates(data_path)
    corridors = config.get("corridors")
    if corridors:
        rates = rates[rates["currency"].isin(corridors)].copy()
    context_path = config.get("context_path")
    context = None
    if context_path:
        context_file = Path(context_path)
        if not context_file.is_absolute():
            context_file = repo_root / context_file
        if context_file.exists():
            context = load_rates(context_file)
    external = None
    if config.get("external_path"):
        external_file = Path(config["external_path"])
        if not external_file.is_absolute():
            external_file = repo_root / external_file
        if external_file.exists():
            external = load_external_series(external_file)
    horizon = int(config["horizon"])
    frame = add_targets(rates, horizon=horizon)
    return add_features(
        frame,
        context=context,
        external=external,
        momentum_days=int(config["momentum_days"]),
        level_window=int(config["level_window"]),
        level_quantile=float(config["level_quantile"]),
        reversal_window=int(config["reversal_window"]),
        holiday_lookahead_days=int(config["holiday_lookahead_days"]),
    )


def _random_signal(frame: pd.DataFrame, rng: np.random.Generator, per_week: float = 1.5) -> pd.Series:
    signal = pd.Series(False, index=frame.index)
    for _, group in frame.groupby("currency", sort=False):
        weeks = max((group["effective_date"].max() - group["effective_date"].min()).days / 7, 1)
        n_draw = min(len(group), max(round(per_week * weeks), 1))
        chosen = rng.choice(group.index.to_numpy(), size=n_draw, replace=False)
        signal.loc[chosen] = True
    return signal


def _walk_forward_model(
    frame: pd.DataFrame,
    *,
    kind: str,
    feature_cols: list[str],
    target_col: str,
    horizon: int,
    folds,
    config: dict,
) -> tuple[pd.Series, pd.Series, dict[str, float]]:
    grid = [float(value) for value in config.get("thresholds", [0.6, 0.7, 0.8, 0.9])]
    quantiles = [float(value) for value in config.get("quantile_rates", [0.10, 0.15, 0.20])]
    min_spw, max_spw = config.get("target_signals_per_week", [0.8, 2.5])
    result = walk_forward_predictions(
        frame,
        model_kind=kind,
        feature_cols=feature_cols,
        target_col=target_col,
        horizon=horizon,
        folds=folds,
        threshold_grid=grid,
        quantile_rates=quantiles,
        target_signals_per_week=(float(min_spw), float(max_spw)),
    )
    thresholds = dict(zip(result.thresholds["fold"], result.thresholds["threshold"], strict=True))
    return result.scores, result.signals.fillna(False).astype(bool), thresholds


def _split_frame(frame: pd.DataFrame, folds, split: str) -> pd.DataFrame:
    if split == "wf_oos":
        names = [fold.name for fold in folds if fold.split == "wf_oos"]
        mask = pd.Series(False, index=frame.index)
        for fold in folds:
            if fold.name in names or fold.split == "wf_oos":
                mask |= mask_test(frame, fold)
        return frame.loc[mask]
    for fold in folds:
        if fold.split == split or fold.name == split:
            return frame.loc[mask_test(frame, fold)]
    raise KeyError(split)


def evaluate_all_splits(
    frame: pd.DataFrame,
    *,
    method: str,
    signal_col: str,
    target_col: str,
    horizon: int,
    folds,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for split in ("y2022", "wf_oos", "oot"):
        subset = _split_frame(frame, folds, split)
        if subset.empty:
            continue
        pieces.append(
            evaluate_method(
                subset,
                horizon=horizon,
                method=method,
                split=split,
                target_col=target_col,
                signal_col=signal_col,
            )
        )
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def run_method(frame: pd.DataFrame, config: dict, method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizon = int(config["horizon"])
    target_kind = str(config.get("target", "stay_not_worse"))
    target_col = target_column(target_kind, horizon)
    folds = make_walk_forward_folds(
        frame["effective_date"].min(),
        frame["effective_date"].max(),
        first_test_year=int(config.get("first_test_year", 2022)),
        oot_start=config.get("oot_start", "2025-09-01"),
    )
    work = frame.copy()
    if method in RULE_METHODS:
        work["eval_signal"] = work[f"signal_{method}"].eq(True)
    elif method == "random":
        work["eval_signal"] = _random_signal(work, np.random.default_rng(int(config.get("seed", 0))))
    elif method in MODEL_METHODS:
        feature_cols = columns_for_groups(list(config.get("feature_groups", ["A", "B"])), work)
        if method == "catboost" and "D" in config.get("feature_groups", []) and not feature_cols:
            pass
        _, signal, _ = _walk_forward_model(
            work,
            kind=method,
            feature_cols=feature_cols,
            target_col=target_col,
            horizon=horizon,
            folds=folds,
            config=config,
        )
        work["eval_signal"] = signal.fillna(False).astype(bool)
        work = add_explanations(work, signal_col="eval_signal", target_kind=target_kind)
    else:
        raise ValueError(f"Unknown method: {method}")
    metrics = evaluate_all_splits(
        work,
        method=method,
        signal_col="eval_signal",
        target_col=target_col,
        horizon=horizon,
        folds=folds,
    )
    return work, metrics


def _append_journal(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = pd.DataFrame([row])
    if path.exists():
        previous = pd.read_csv(path)
        pd.concat([previous, line], ignore_index=True).to_csv(path, index=False)
    else:
        line.to_csv(path, index=False)


def run_train(config_path: Path, exp_name: str, method: str | None = None) -> Path:
    config_path = config_path.resolve()
    repo_root = _repo_root(config_path)
    config = load_yaml(config_path)
    frame = build_frame(config, repo_root)
    chosen = method or str(config.get("model", "logreg"))
    scored, metrics = run_method(frame, config, chosen)
    reports = repo_root / Path(config.get("reports_dir", "reports"))
    tables = reports / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    artifacts = repo_root / "artifacts" / "experiments" / exp_name
    artifacts.mkdir(parents=True, exist_ok=True)
    scored.to_csv(artifacts / "predictions.csv", index=False)
    metrics.to_csv(artifacts / "metrics.csv", index=False)
    (artifacts / "config.yaml").write_text(
        yaml.safe_dump({**config, "model": chosen, "exp_name": exp_name}, sort_keys=False),
        encoding="utf-8",
    )
    oot = metrics[metrics["split"] == "oot"]
    journal_row = {
        "exp_id": exp_name,
        "git_hash": _git_hash(repo_root),
        "model": chosen,
        "target": config.get("target", "stay_not_worse"),
        "horizon": int(config["horizon"]),
        "feature_groups": ",".join(config.get("feature_groups", [])),
        "oot_lift": float(oot["lift"].mean()) if not oot.empty else np.nan,
        "oot_bps": float(oot["bps_forward"].mean()) if not oot.empty else np.nan,
        "oot_signals_per_week": float(oot["signals_per_week"].mean()) if not oot.empty else np.nan,
    }
    _append_journal(tables / "experiments.csv", journal_row)
    (artifacts / "journal.json").write_text(json.dumps(journal_row, indent=2), encoding="utf-8")
    return tables / "experiments.csv"


def run_summary(config_path: Path) -> Path:
    config_path = config_path.resolve()
    repo_root = _repo_root(config_path)
    config = load_yaml(config_path)
    frame = build_frame(config, repo_root)
    context_available = bool(columns_for_groups(["D"], frame))
    jobs: list[tuple[str, list[str] | None]] = [
        ("random", None),
        ("momentum", None),
        ("level", None),
        ("reversal", None),
        ("seasonality", None),
        ("logreg", ["A"]),
        ("logreg", ["A", "B"]),
        ("logreg", ["A", "B", "C"]),
        ("catboost", ["A", "B"]),
    ]
    if context_available:
        jobs.append(("logreg", ["A", "B", "C", "D"]))
        jobs.append(("catboost", ["A", "B", "C", "D"]))
    pieces: list[pd.DataFrame] = []
    for method, groups in jobs:
        method_config = dict(config)
        if groups is not None:
            method_config["feature_groups"] = groups
        _, metrics = run_method(frame, method_config, method)
        if metrics.empty:
            continue
        metrics["feature_groups"] = ",".join(groups or [])
        pieces.append(metrics)
    summary = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    output = repo_root / Path(config.get("reports_dir", "reports")) / "tables" / "ml_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    if not summary.empty:
        oot = summary[summary["split"] == "oot"]
        _append_journal(
            output.parent / "experiments.csv",
            {
                "exp_id": "ml_summary_h5",
                "git_hash": _git_hash(repo_root),
                "model": "matrix",
                "target": config.get("target", "stay_not_worse"),
                "horizon": int(config["horizon"]),
                "feature_groups": "matrix",
                "oot_lift": float(oot["lift"].mean()) if not oot.empty else np.nan,
                "oot_bps": float(oot["bps_forward"].mean()) if not oot.empty else np.nan,
                "oot_signals_per_week": float(oot["signals_per_week"].mean())
                if not oot.empty
                else np.nan,
            },
        )
    return output
