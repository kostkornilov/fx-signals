from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fx_signal.data import load_rates, load_yaml
from fx_signal.external import load_external_series
from fx_signal.features import columns_for_groups
from fx_signal.metrics import forward_bps
from fx_signal.splits import make_walk_forward_folds
from fx_signal.targets import target_column
from fx_signal.train import _split_frame, build_frame, run_method


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def build_data_audit(config: dict, repo_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for kind, key in (("target", "data_path"), ("cbr_context", "context_path")):
        if not config.get(key):
            continue
        path = _resolve(repo_root, config[key])
        if not path.exists():
            continue
        frame = load_rates(path)
        for series_id, group in frame.groupby("currency", sort=True):
            expected = len(pd.bdate_range(group["effective_date"].min(), group["effective_date"].max()))
            rows.append(
                {
                    "kind": kind,
                    "source": "cbr",
                    "series_id": series_id,
                    "start": group["effective_date"].min(),
                    "end": group["effective_date"].max(),
                    "observations": len(group),
                    "business_day_coverage": len(group) / expected if expected else np.nan,
                    "median_publication_lag_days": (
                        group["available_at"] - group["effective_date"]
                    ).dt.total_seconds().median()
                    / 86400,
                }
            )
    if config.get("external_path"):
        path = _resolve(repo_root, config["external_path"])
        if path.exists():
            frame = load_external_series(path)
            for (source, series_id), group in frame.groupby(["source", "series_id"], sort=True):
                expected = len(pd.bdate_range(group["event_date"].min(), group["event_date"].max()))
                rows.append(
                    {
                        "kind": "external",
                        "source": source,
                        "series_id": series_id,
                        "start": group["event_date"].min(),
                        "end": group["event_date"].max(),
                        "observations": len(group),
                        "business_day_coverage": len(group) / expected if expected else np.nan,
                        "median_publication_lag_days": (
                            group["available_at"] - group["event_date"]
                        ).dt.total_seconds().median()
                        / 86400,
                    }
                )
    return pd.DataFrame(rows)


def build_market_component_eda(config: dict, repo_root: Path) -> pd.DataFrame:
    """Quantify how much of each corridor move is the shared RUB component."""
    rates = load_rates(_resolve(repo_root, config["data_path"]))
    context = load_rates(_resolve(repo_root, config["context_path"]))
    target = rates.pivot(index="effective_date", columns="currency", values="rub_per_unit")
    market = context.pivot(index="effective_date", columns="currency", values="rub_per_unit")
    target_returns = target.pct_change(fill_method=None)
    market_returns = market.pct_change(fill_method=None)
    rows: list[dict] = []
    for corridor in target_returns:
        for driver in market_returns:
            pair = pd.concat(
                [target_returns[corridor], market_returns[driver]], axis=1, keys=["y", "x"]
            ).dropna()
            if pair.empty or pair["x"].var() == 0:
                continue
            beta = float(pair["y"].cov(pair["x"]) / pair["x"].var())
            residual = pair["y"] - beta * pair["x"]
            rows.append(
                {
                    "corridor": f"RUB->{corridor}",
                    "driver": f"{driver}/RUB",
                    "observations": len(pair),
                    "correlation": pair["y"].corr(pair["x"]),
                    "beta": beta,
                    "variance_explained_r2": 1.0 - residual.var() / pair["y"].var(),
                    "residual_volatility": residual.std(),
                }
            )
    return pd.DataFrame(rows)


def _paired_effect(
    group: pd.DataFrame,
    candidate_signal: str,
    baseline_signal: str,
    *,
    target_col: str,
    forward_col: str,
) -> dict[str, float]:
    target = group[target_col].astype(bool).to_numpy()
    candidate = group[candidate_signal].astype(bool).to_numpy()
    baseline = group[baseline_signal].astype(bool).to_numpy()
    benefits = forward_bps(group["rub_per_unit"], group[forward_col]).to_numpy()
    prevalence = target.mean()
    if not prevalence or not candidate.any() or not baseline.any():
        return {"delta_lift": np.nan, "delta_bps": np.nan}
    return {
        "delta_lift": float(
            target[candidate].mean() / prevalence - target[baseline].mean() / prevalence
        ),
        "delta_bps": float(np.nanmean(benefits[candidate]) - np.nanmean(benefits[baseline])),
    }


def _decision(row: pd.Series, *, min_signals_per_week: float) -> str:
    if row.get("signal_count", 0) == 0:
        return "unusable"
    if row.get("signals_per_week", 0.0) < min_signals_per_week:
        return "insufficient_frequency"
    if row.get("delta_lift", np.nan) > 0 and row.get("delta_bps", np.nan) >= 0:
        return "adopt"
    return "reject"


def run_research(config_path: Path) -> Path:
    config_path = config_path.resolve()
    repo_root = config_path.parent.parent
    config = load_yaml(config_path)
    output_dir = repo_root / Path(config.get("output_dir", "reports/research"))
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = build_data_audit(config, repo_root)
    audit.to_csv(output_dir / "data_audit.csv", index=False)
    component_eda = build_market_component_eda(config, repo_root)
    component_eda.to_csv(output_dir / "market_component_eda.csv", index=False)

    experiments = config["experiments"]
    baseline_name = str(config.get("baseline_experiment", experiments[0]["name"]))
    all_metrics: list[pd.DataFrame] = []
    comparisons: list[dict] = []
    for horizon in [int(value) for value in config.get("horizons", [5])]:
        horizon_config = {**config, "horizon": horizon}
        frame = build_frame(horizon_config, repo_root)
        scored: dict[str, pd.DataFrame] = {}
        for experiment in experiments:
            groups = experiment.get("feature_groups", [])
            if "F" in groups and not columns_for_groups(["F"], frame):
                continue
            method_config = {
                **horizon_config,
                "feature_groups": groups,
            }
            name = str(experiment["name"])
            method = str(experiment["method"])
            prediction, metrics = run_method(frame, method_config, method)
            metrics["experiment"] = name
            metrics["feature_groups"] = ",".join(groups)
            all_metrics.append(metrics)
            scored[name] = prediction

        if baseline_name not in scored:
            raise ValueError(f"Unknown baseline experiment: {baseline_name}")
        folds = make_walk_forward_folds(
            frame["effective_date"].min(),
            frame["effective_date"].max(),
            first_test_year=int(config.get("first_test_year", 2022)),
            oot_start=config.get("oot_start", "2025-09-01"),
        )
        target_col = target_column(str(config.get("target", "stay_not_worse")), horizon)
        baseline = scored[baseline_name][["eval_signal"]].rename(
            columns={"eval_signal": "baseline_signal"}
        )
        for name, prediction in scored.items():
            if name == baseline_name:
                continue
            joined = prediction.join(baseline).rename(columns={"eval_signal": "candidate_signal"})
            for split in ("y2022", "wf_oos", "oot"):
                subset = _split_frame(joined, folds, split)
                for currency, group in subset.dropna(subset=[target_col]).groupby("currency"):
                    stats = _paired_effect(
                        group,
                        "candidate_signal",
                        "baseline_signal",
                        target_col=target_col,
                        forward_col=f"forward_mean_h{horizon}",
                    )
                    candidate_count = int(group["candidate_signal"].sum())
                    span_weeks = max(
                        (group["effective_date"].max() - group["effective_date"].min()).days / 7,
                        1,
                    )
                    row = {
                        "horizon": horizon,
                        "split": split,
                        "corridor": f"RUB->{currency}",
                        "baseline": baseline_name,
                        "experiment": name,
                        "signal_count": candidate_count,
                        "signals_per_week": candidate_count / span_weeks,
                        **stats,
                    }
                    row["decision"] = _decision(
                        pd.Series(row),
                        min_signals_per_week=float(
                            config.get("decision_min_signals_per_week", 0.8)
                        ),
                    )
                    comparisons.append(row)

    metrics_frame = pd.concat(all_metrics, ignore_index=True)
    metrics_frame.to_csv(output_dir / "ablation_metrics.csv", index=False)
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(output_dir / "paired_comparisons.csv", index=False)
    _write_report(
        output_dir / "README.md", audit, component_eda, metrics_frame, comparison_frame
    )
    return output_dir / "README.md"


def _write_report(
    path: Path,
    audit: pd.DataFrame,
    component_eda: pd.DataFrame,
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> None:
    def table(frame: pd.DataFrame, include_index: bool = False) -> str:
        printable = frame.reset_index() if include_index else frame.copy()
        printable = printable.fillna("")
        headers = [str(column) for column in printable.columns]
        rows = [[str(value) for value in row] for row in printable.to_numpy()]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(lines)

    oot = metrics[metrics["split"].eq("oot")]
    summary = (
        oot.groupby("experiment", as_index=False)
        .agg(
            signal_count=("signal_count", "sum"),
            corridor_horizons_with_signal=(
                "signal_count",
                lambda values: int(values.gt(0).sum()),
            ),
            lift=("lift", "mean"),
            bps=("bps_forward", "mean"),
            signals_per_week=("signals_per_week", "mean"),
        )
        .round(3)
    )
    decisions = (
        comparisons[comparisons["split"].eq("oot")]
        .groupby(["experiment", "decision"])
        .size()
        .unstack(fill_value=0)
        if not comparisons.empty
        else pd.DataFrame()
    )
    text = "# Исследование дополнительных рядов\n\n"
    text += "## Покрытие данных\n\n" + table(audit) + "\n\n"
    text += "## Общий валютный фактор\n\n" + table(component_eda.round(4)) + "\n\n"
    text += "## OOT-результаты\n\n" + table(summary) + "\n\n"
    text += "## Решения относительно CBR-only baseline\n\n"
    text += table(decisions, include_index=True) if not decisions.empty else "Нет сравнений."
    eligible = summary[
        summary["lift"].ge(1.3)
        & summary["bps"].gt(0)
        & summary["signals_per_week"].between(1.0, 2.0)
    ]
    text += "\n\n## Вывод\n\n"
    if eligible.empty:
        text += (
            "Ни одна конфигурация не прошла полный quality gate: lift ≥ 1.3, положительная "
            "выгода и 1–2 сигнала в неделю. Высокий lift редких конфигураций нельзя считать "
            "подтверждением из-за недостаточного числа сигналов.\n"
        )
    else:
        text += "Quality gate прошли: " + ", ".join(eligible["experiment"].astype(str)) + ".\n"
    text += "\n\nПолные результаты: `ablation_metrics.csv`, `paired_comparisons.csv`.\n"
    text += "Сводный график: [figures/external_report.png](figures/external_report.png).\n"
    path.write_text(text, encoding="utf-8")
