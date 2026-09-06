from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fx_signal.backtest import walk_forward_predictions
from fx_signal.features import columns_for_groups
from fx_signal.indicators import ALL_SIGNAL_COLUMNS, add_research_indicators
from fx_signal.metrics import add_customer_outcomes, evaluate_method
from fx_signal.signal_snapshot import (
    DEFAULT_ALLOWED_SIGNALS,
    apply_meta_policy_to_predictions,
)
from fx_signal.splits import WalkForwardFold, make_walk_forward_folds, mask_test
from fx_signal.targets import add_targets, target_column
from fx_signal.train import _random_signal, _repo_root, build_frame

RULE_LABELS = {
    "signal_better_than_one_year_ago": "Правило: лучше, чем год назад",
    "signal_better_range_held": "Правило: диапазон удержался",
    "signal_better_than_30_day_average": "Правило: лучше 30-дневного среднего",
    "signal_momentum": "Правило: momentum",
    "signal_most_recent_changes_favourable": "Правило: недавние изменения хорошие",
    "signal_larger_than_usual_latest_improvement": "Правило: крупное последнее улучшение",
    "signal_level": "Правило: level",
    "signal_seasonality": "Правило: seasonality",
    "signal_most_recent_changes_unfavourable": "Правило: недавние изменения плохие",
    "signal_reversal": "Правило: reversal",
    "signal_less_than_one_year_ago": "Правило: хуже, чем год назад",
}

OOT_SUMMARY_COLUMNS = [
    "method",
    "lift_h1",
    "lar_h1",
    "lift_h5",
    "lar_h5",
    "lift_h10",
    "lar_h10",
    "signals_per_week",
    "cluster_rate",
    "corridors_with_lar_h1",
    "corridors_with_lar_h5",
    "corridors_with_lar_h10",
]
MODEL_LABELS = {"logreg": "LogReg", "catboost": "CatBoost"}


@dataclass(frozen=True)
class Candidate:
    method_id: str
    label: str
    method_type: str
    signal_col: str | None = None
    model_kind: str | None = None
    feature_groups: tuple[str, ...] = ()


def _stable_seed(seed: int, *parts: object) -> int:
    payload = ":".join([str(seed), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _candidates(config: dict) -> list[Candidate]:
    candidates = [Candidate("random", "Случайный день", "random")]
    for signal in config["rule_signals"]:
        if signal not in ALL_SIGNAL_COLUMNS:
            raise ValueError(f"Unknown rule signal: {signal}")
        candidates.append(
            Candidate(signal, RULE_LABELS.get(signal, signal), "rule", signal_col=signal)
        )
    for item in config["model_candidates"]:
        kind = str(item["kind"])
        groups = tuple(str(group) for group in item["feature_groups"])
        suffix = "_".join(group.lower() for group in groups)
        candidates.append(
            Candidate(
                f"{kind}_{suffix}",
                f"{MODEL_LABELS.get(kind, kind)} ({','.join(groups)})",
                "model",
                model_kind=kind,
                feature_groups=groups,
            )
        )
    ids = [candidate.method_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Backtest candidate ids must be unique")
    return candidates


def _prepare_frame(config: dict, repo_root: Path) -> pd.DataFrame:
    training_horizon = int(config["training_horizon"])
    build_config = {**config, "horizon": training_horizon}
    frame = build_frame(build_config, repo_root)
    frame = add_research_indicators(frame)
    end = pd.Timestamp(config["end"])
    frame = frame.loc[pd.to_datetime(frame["effective_date"]).le(end)].copy()
    for horizon in sorted(set(map(int, config["evaluation_horizons"])) | {training_horizon}):
        frame = add_targets(frame, horizon=horizon)
        frame = add_customer_outcomes(frame, horizon=horizon)
    return frame


def _folds(frame: pd.DataFrame, config: dict) -> list[WalkForwardFold]:
    folds = make_walk_forward_folds(
        frame["effective_date"].min(),
        frame["effective_date"].max(),
        first_test_year=int(config["first_test_year"]),
        oot_start=config["oot_start"],
    )
    if not folds or folds[-1].name != "oot":
        raise ValueError("Selected period must contain a non-empty final OOT interval")
    return folds


def _candidate_signal(
    frame: pd.DataFrame,
    candidate: Candidate,
    folds: list[WalkForwardFold],
    config: dict,
) -> tuple[pd.Series, pd.DataFrame]:
    if candidate.method_type == "rule":
        return frame[candidate.signal_col].eq(True), pd.DataFrame()
    if candidate.method_type == "random":
        rng = np.random.default_rng(_stable_seed(int(config["seed"]), candidate.method_id))
        return _random_signal(
            frame, rng, per_week=float(config["random_signals_per_week"])
        ), pd.DataFrame()

    training_horizon = int(config["training_horizon"])
    target_col = target_column(str(config["target"]), training_horizon)
    feature_cols = columns_for_groups(list(candidate.feature_groups), frame)
    if not feature_cols:
        raise ValueError(f"No features resolved for {candidate.method_id}")
    min_spw, max_spw = config["target_signals_per_week"]
    result = walk_forward_predictions(
        frame,
        model_kind=str(candidate.model_kind),
        feature_cols=feature_cols,
        target_col=target_col,
        horizon=training_horizon,
        folds=folds,
        threshold_grid=config["thresholds"],
        quantile_rates=config["quantile_rates"],
        target_signals_per_week=(float(min_spw), float(max_spw)),
    )
    _, final_signals = apply_meta_policy_to_predictions(
        frame,
        result,
        allowed_signals=tuple(config.get("allowed_signals", DEFAULT_ALLOWED_SIGNALS)),
        cooldown_days=int(config["cooldown_days"]),
        weekly_limit=int(config["weekly_limit"]),
    )
    return final_signals, result.thresholds


def _evaluate_candidate(
    frame: pd.DataFrame,
    candidate: Candidate,
    signals: pd.Series,
    thresholds: pd.DataFrame,
    folds: list[WalkForwardFold],
    config: dict,
) -> pd.DataFrame:
    work = frame.copy()
    work["_report_signal"] = signals.reindex(work.index).fillna(False).astype(bool)
    threshold_by_fold = (
        thresholds.set_index("fold")["threshold"].to_dict() if not thresholds.empty else {}
    )
    rows: list[pd.DataFrame] = []
    for fold in folds:
        test = work.loc[mask_test(work, fold)].copy()
        for horizon in map(int, config["evaluation_horizons"]):
            metric = evaluate_method(
                test,
                horizon=horizon,
                method=candidate.method_id,
                split=fold.name,
                target_col=target_column(str(config["target"]), horizon),
                signal_col="_report_signal",
                rng=np.random.default_rng(
                    _stable_seed(int(config["seed"]), candidate.method_id, fold.name, horizon)
                ),
                weekly_limit=int(config["weekly_limit"]),
                cooldown_days=int(config["cooldown_days"]),
                baseline_draws=int(config["baseline_draws"]),
            )
            metric.insert(0, "method_id", candidate.method_id)
            metric.insert(1, "method_label", candidate.label)
            metric.insert(2, "method_type", candidate.method_type)
            metric.insert(3, "feature_groups", ",".join(candidate.feature_groups))
            metric.insert(4, "training_horizon", int(config["training_horizon"]))
            metric["fold_kind"] = fold.split
            metric["train_end"] = fold.train_end
            metric["validation_start"] = fold.val_start
            metric["validation_end"] = fold.val_end
            metric["test_start"] = fold.test_start
            metric["test_end"] = fold.test_end
            metric["threshold"] = threshold_by_fold.get(fold.name, np.nan)
            rows.append(metric)
    return pd.concat(rows, ignore_index=True)


def build_oot_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    oot = metrics.loc[metrics["split"].eq("oot")].copy()
    rows: list[dict[str, object]] = []
    for label, group in oot.groupby("method_label", sort=False):
        row: dict[str, object] = {"method": label}
        for horizon in (1, 5, 10):
            part = group.loc[group["horizon"].eq(horizon)]
            row[f"lift_h{horizon}"] = part["lift"].mean()
            row[f"lar_h{horizon}"] = part["lift_at_risk"].mean()
            row[f"corridors_with_lar_h{horizon}"] = int(part["lift_at_risk"].notna().sum())
        h5 = group.loc[group["horizon"].eq(5)]
        row["signals_per_week"] = h5["signals_per_week"].mean()
        row["cluster_rate"] = h5["cluster_rate"].mean()
        rows.append(row)
    summary = pd.DataFrame(rows).reindex(columns=OOT_SUMMARY_COLUMNS)
    return summary.sort_values(
        ["corridors_with_lar_h5", "lift_h5", "corridors_with_lar_h10", "lift_h10"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)


def _method_summary(metrics: pd.DataFrame, primary_horizon: int = 5) -> pd.DataFrame:
    work = metrics.loc[metrics["horizon"].eq(primary_horizon)].copy()
    oos = work.loc[~work["split"].eq("oot")]
    oot = work.loc[work["split"].eq("oot")]
    rows = []
    for method_id, group in oos.groupby("method_id", sort=False):
        valid = group.loc[group["lift"].notna()]
        final = oot.loc[oot["method_id"].eq(method_id)]
        mean_lift = valid["lift"].mean()
        mean_value = valid["moment_advantage_bps"].mean()
        frequency = group["signals_per_week"].mean()
        passed = bool(
            pd.notna(mean_lift)
            and mean_lift >= 1.3
            and pd.notna(mean_value)
            and mean_value > 0
            and 1 <= frequency <= 2
        )
        rows.append(
            {
                "method_id": method_id,
                "method": group["method_label"].iloc[0],
                "oos_lift_mean": mean_lift,
                "oos_lift_worst": valid["lift"].min(),
                "oos_lift_13_share": valid["lift"].ge(1.3).mean() if len(valid) else np.nan,
                "oos_lar_mean": valid["lift_at_risk"].mean(),
                "oot_lift": final["lift"].mean(),
                "oot_lar": final["lift_at_risk"].mean(),
                "signals_per_week": frequency,
                "cluster_rate": group["cluster_rate"].mean(),
                "valid_slices": len(valid),
                "status": "PASS" if passed else "REVIEW",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["status", "oos_lar_mean", "oos_lift_mean"],
        ascending=[True, False, False],
        na_position="last",
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small Markdown table without an optional tabulate dependency."""
    values = frame.fillna("—").astype(str)
    header = "| " + " | ".join(values.columns) + " |"
    separator = "| " + " | ".join("---" for _ in values.columns) + " |"
    rows = [
        "| " + " | ".join(value.replace("|", "\\|") for value in row) + " |"
        for row in values.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def render_report(metrics: pd.DataFrame, config: dict, git_hash: str) -> str:
    summary = _method_summary(metrics)
    display = summary.head(10).copy()
    for column in [
        "oos_lift_mean", "oos_lift_worst", "oos_lar_mean", "oot_lift", "oot_lar",
        "signals_per_week", "cluster_rate",
    ]:
        display[column] = display[column].map(lambda value: "—" if pd.isna(value) else f"{value:.2f}")
    display["oos_lift_13_share"] = display["oos_lift_13_share"].map(
        lambda value: "—" if pd.isna(value) else f"{value:.0%}"
    )
    table = display.rename(
        columns={
            "method": "Метод", "oos_lift_mean": "OOS lift", "oos_lift_worst": "Worst lift",
            "oos_lift_13_share": "Срезы ≥1.3", "oos_lar_mean": "OOS LAR",
            "oot_lift": "OOT lift", "oot_lar": "OOT LAR", "signals_per_week": "Сигн./нед.",
            "cluster_rate": "Кучность", "valid_slices": "Срезов", "status": "Статус",
        }
    )[["Метод", "OOS lift", "Worst lift", "Срезы ≥1.3", "OOS LAR", "OOT lift", "OOT LAR", "Сигн./нед.", "Кучность", "Срезов", "Статус"]]
    passed = int(summary["status"].eq("PASS").sum())
    table_markdown = _markdown_table(table)
    return f"""# Воспроизводимый walk-forward backtest

## Итог

- Git commit: `{git_hash}`.
- Период оценки: с {int(config['first_test_year'])} года по {config['end']}; final OOT начинается {config['oot_start']}.
- ML обучается на таргете `h={config['training_horizon']}`; тот же поток сигналов проверяется на `h={"/".join(map(str, config['evaluation_horizons']))}`.
- Для frequency-matched random baseline используется {config.get('baseline_draws', 200)} детерминированных потоков на срез.
- Кандидатов, прошедших базовый gate (`lift ≥ 1.3`, выгода > 0, частота 1–2/нед.): **{passed}**.

## Краткая сводка (`h=5`)

{table_markdown}

OOS — среднее по отдельным `corridor × walk-forward fold`, final OOT в него не входит. LAR используется для ранжирования после базовых ограничений; он не заменяет lift.

## Интерпретация

- Полная детализация по методам, коридорам, периодам и горизонтам: [backtest_metrics.csv](backtest_metrics.csv).
- Компактная OOT-витрина для финального отчёта: [../tables/ml_summary_lift.csv](../tables/ml_summary_lift.csv).
- `REVIEW` означает, что хотя бы одно базовое условие не выполнено; отсутствие сигналов не считается доказательством качества.

## Ограничения

- Официальный курс ЦБ — proxy курса исполнения в приложении.
- Правила имеют фиксированные параметры из конфига; новый поиск правил в этом запуске не выполняется.
- CVaR на малом числе сигналов является описательной оценкой.
- Final OOT показывается отдельно и не используется для обучения или выбора порогов.
"""


def _git_hash(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_backtest_report(
    config_path: Path,
    *,
    first_test_year: int | None = None,
    oot_start: str | None = None,
    end: str | None = None,
) -> Path:
    config_path = config_path.resolve()
    repo_root = _repo_root(config_path)
    config = yaml.safe_load(config_path.read_text())
    if first_test_year is not None:
        config["first_test_year"] = first_test_year
    if oot_start is not None:
        config["oot_start"] = oot_start
    if end is not None:
        config["end"] = end

    frame = _prepare_frame(config, repo_root)
    folds = _folds(frame, config)
    pieces = []
    threshold_pieces = []
    candidates = _candidates(config)
    for position, candidate in enumerate(candidates, start=1):
        print(f"[{position}/{len(candidates)}] {candidate.method_id}", flush=True)
        signal, thresholds = _candidate_signal(frame, candidate, folds, config)
        pieces.append(_evaluate_candidate(frame, candidate, signal, thresholds, folds, config))
        if not thresholds.empty:
            audit = thresholds.copy()
            audit.insert(0, "method_id", candidate.method_id)
            threshold_pieces.append(audit)

    metrics = pd.concat(pieces, ignore_index=True)
    reports_dir = repo_root / config["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "backtest_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    build_oot_summary(metrics).to_csv(repo_root / config["oot_summary_path"], index=False)
    git_hash = _git_hash(repo_root)
    report_path = reports_dir / "REPORT.md"
    report_path.write_text(render_report(metrics, config, git_hash), encoding="utf-8")

    artifacts = repo_root / "artifacts" / "backtest"
    artifacts.mkdir(parents=True, exist_ok=True)
    if threshold_pieces:
        pd.concat(threshold_pieces, ignore_index=True).to_csv(
            artifacts / "thresholds.csv", index=False
        )
    (artifacts / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    data_path = repo_root / config["data_path"]
    context_path = repo_root / config["context_path"]
    manifest = {
        "git_commit": git_hash,
        "python": sys.version,
        "packages": {name: version(name) for name in ("pandas", "numpy", "scikit-learn", "catboost")},
        "data_sha256": {str(path.relative_to(repo_root)): _sha256(path) for path in (data_path, context_path)},
        "rows": len(metrics),
    }
    (artifacts / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return report_path
