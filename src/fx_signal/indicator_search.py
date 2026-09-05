from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fx_signal.data import load_yaml
from fx_signal.evaluation import evaluate_method, forward_bps, signals_per_week
from fx_signal.public_context import causal_context_join, load_public_context
from fx_signal.rule_catalog import RuleSpec, build_rule_catalog
from fx_signal.splits import make_walk_forward_folds, mask_test, mask_val
from fx_signal.targets import target_column
from fx_signal.train import _repo_root, build_frame


def purge_validation_tail(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if horizon <= 0:
        return frame.copy()
    pieces = []
    for _, group in frame.groupby("currency", sort=False):
        ordered = group.sort_values("effective_date")
        pieces.append(ordered.iloc[:-horizon] if len(ordered) > horizon else ordered.iloc[0:0])
    return pd.concat(pieces).sort_index() if pieces else frame.iloc[0:0].copy()


def _public_context_status(config: dict, repo_root: Path) -> str:
    configured = config.get("public_sources", [])
    path = Path(config.get("public_context_path", "data/raw/public/context.csv"))
    if not path.is_absolute():
        path = repo_root / path
    if path.exists():
        return f"loaded: {path}"
    if configured:
        return "not loaded: configured public sources were unavailable during this run"
    return "not configured"


def apply_communication_policy(
    frame: pd.DataFrame,
    signal: pd.Series,
    *,
    cooldown_prints: int,
    max_in_7d: int = 2,
) -> pd.Series:
    kept = pd.Series(False, index=frame.index)
    for _, group in frame.assign(_candidate=signal.fillna(False)).groupby("currency", sort=False):
        group = group.sort_values("effective_date")
        accepted_positions: list[int] = []
        accepted_dates: list[pd.Timestamp] = []
        for position, (index, row) in enumerate(group.iterrows()):
            if not bool(row["_candidate"]):
                continue
            current = pd.Timestamp(row["effective_date"])
            recent = [date for date in accepted_dates if date >= current - pd.Timedelta(6, unit="D")]
            if accepted_positions and position - accepted_positions[-1] <= cooldown_prints:
                continue
            if len(recent) >= max_in_7d:
                continue
            kept.loc[index] = True
            accepted_positions.append(position)
            accepted_dates.append(current)
    return kept


def _summary(
    frame: pd.DataFrame, signal: pd.Series, config: dict, split: str, *, detailed: bool = False
) -> tuple[dict, pd.DataFrame]:
    horizon = int(config["horizon"])
    target_col = target_column(str(config.get("target", "stay_not_worse")), horizon)
    work = frame.copy()
    work["_signal"] = signal.reindex(work.index).fillna(False)
    if detailed:
        metrics = evaluate_method(
            work,
            horizon=horizon,
            method="indicator_search",
            split=split,
            target_col=target_col,
            signal_col="_signal",
        )
    else:
        rows = []
        required = [target_col, f"forward_mean_h{horizon}", "_signal"]
        for currency, group in work.dropna(subset=required).groupby("currency", sort=False):
            target = group[target_col].astype(bool)
            chosen = group["_signal"].astype(bool)
            count = int(chosen.sum())
            prevalence = float(target.mean())
            hit = float(target[chosen].mean()) if count else np.nan
            bps = forward_bps(group["rub_per_unit"], group[f"forward_mean_h{horizon}"])
            rows.append({
                "corridor": f"RUB->{currency}", "signal_count": count,
                "lift": hit / prevalence if count and prevalence else np.nan,
                "bps_forward": float(bps[chosen].mean()) if count else np.nan,
                "signals_per_week": signals_per_week(group["effective_date"], chosen),
                "cluster_share": float((chosen & chosen.shift(1).eq(True)).sum() / count)
                if count else np.nan,
            })
        metrics = pd.DataFrame(rows)
    valid = metrics[metrics["signal_count"].gt(0)]
    total_signals = int(metrics["signal_count"].sum()) if not metrics.empty else 0
    positive_cells = float(valid["lift"].gt(1).mean()) if not valid.empty else 0.0
    row = {
        "signal_count": total_signals,
        "corridors_with_signals": int(metrics["signal_count"].gt(0).sum()),
        "mean_lift": float(valid["lift"].mean()) if not valid.empty else np.nan,
        "median_lift": float(valid["lift"].median()) if not valid.empty else np.nan,
        "mean_bps": float(valid["bps_forward"].mean()) if not valid.empty else np.nan,
        "mean_signals_per_week": float(metrics["signals_per_week"].mean()) if not metrics.empty else 0.0,
        "positive_share": positive_cells,
        "mean_cluster_share": float(valid["cluster_share"].mean()) if not valid.empty else np.nan,
    }
    return row, metrics


def _rank_key(row: dict) -> tuple:
    feasible = 0.8 <= row["mean_signals_per_week"] <= 2.5 and row["mean_bps"] > 0
    return (
        int(feasible),
        np.nan_to_num(row["median_lift"], nan=-1.0),
        np.nan_to_num(row["mean_bps"], nan=-1e9),
        -abs(row["mean_signals_per_week"] - 1.5),
    )


def _screen(frame: pd.DataFrame, specs: list[RuleSpec], config: dict, fold_name: str) -> pd.DataFrame:
    rows = []
    screening = config.get("screening", {})
    for spec in specs:
        summary, _ = _summary(frame, frame[spec.rule_id], config, "validation")
        passed = (
            summary["signal_count"] >= int(screening.get("min_signals", 50))
            and summary["corridors_with_signals"] >= int(screening.get("min_corridors", 4))
            and summary["median_lift"] > float(screening.get("min_median_lift", 1.0))
            and summary["mean_bps"] > float(screening.get("min_mean_bps", 0.0))
            and summary["positive_share"] >= float(screening.get("min_positive_share", 0.6))
        )
        rows.append({"fold": fold_name, **asdict(spec), **summary, "passed": passed})
    return pd.DataFrame(rows)


def _shortlist(screening: pd.DataFrame, config: dict) -> list[str]:
    cap_family = int(config.get("shortlist_per_family", 5))
    cap_total = int(config.get("shortlist_total", 20))
    pool = screening[screening["passed"]].copy()
    if pool.empty:
        pool = screening.copy()
    pool["_score"] = pool.apply(lambda row: _rank_key(row.to_dict()), axis=1)
    pool = pool.sort_values("_score", ascending=False)
    selected = pool.groupby("family", sort=False).head(cap_family)
    return selected.head(cap_total)["rule_id"].tolist()


def _candidate_signals(
    frame: pd.DataFrame,
    shortlist: list[str],
    specs: list[RuleSpec],
    *,
    filter_correlation: bool = False,
) -> dict[str, pd.Series]:
    candidates = {name: frame[name].fillna(False) for name in shortlist}
    family = {spec.rule_id: spec.family for spec in specs}
    level = [name for name in shortlist if family[name] == "level"][:5]
    other = [name for name in shortlist if family[name] != "level"][:8]
    for left in level:
        for right in other:
            candidates[f"and__{left}__{right}"] = candidates[left] & candidates[right]
    for i, left in enumerate(shortlist[:10]):
        for right in shortlist[i + 1 : 10]:
            corr = candidates[left].astype(float).corr(candidates[right].astype(float))
            if not filter_correlation or pd.isna(corr) or abs(corr) < 0.7:
                candidates[f"or__{left}__{right}"] = candidates[left] | candidates[right]
    if len(shortlist) >= 3:
        for size in (3, 5):
            names = shortlist[:size]
            if len(names) == size:
                candidates[f"vote2__top{size}"] = pd.concat(
                    [candidates[name] for name in names], axis=1
                ).sum(axis=1).ge(2)
    return candidates


def _select_policy(frame: pd.DataFrame, candidates: dict[str, pd.Series], config: dict) -> tuple[str, int, dict]:
    best: tuple[tuple, str, int, dict] | None = None
    for name, raw in candidates.items():
        for cooldown in config.get("cooldown_prints", [1, 2, 3, 5]):
            signal = apply_communication_policy(frame, raw, cooldown_prints=int(cooldown))
            summary, _ = _summary(frame, signal, config, "validation")
            candidate = (_rank_key(summary), name, int(cooldown), summary)
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best is None:
        raise RuntimeError("No indicator candidates were generated")
    _, name, cooldown, summary = best
    return name, cooldown, summary


def _discovery_effect(selected: pd.DataFrame, config: dict) -> dict[str, float]:
    horizon = int(config["horizon"])
    target_col = target_column(str(config.get("target", "stay_not_worse")), horizon)
    point_lifts, point_bps = [], []
    for _, group in selected.groupby("currency"):
        group = group.sort_values("effective_date")
        signal = group["selected_signal"].to_numpy(dtype=bool)
        target = group[target_col].to_numpy(dtype=bool)
        if signal.sum() and target.mean() > 0:
            point_lifts.append(float(target[signal].mean() / target.mean()))
            price = group["rub_per_unit"].to_numpy(dtype=float)
            future = group[f"forward_mean_h{horizon}"].to_numpy(dtype=float)
            point_bps.append(float((((future - price) / price * 1e4)[signal]).mean()))
    return {
        "lift_point": float(np.mean(point_lifts)) if point_lifts else np.nan,
        "bps_point": float(np.mean(point_bps)) if point_bps else np.nan,
    }


def run_indicator_search(config_path: Path) -> Path:
    config_path = config_path.resolve()
    repo_root = _repo_root(config_path)
    config = load_yaml(config_path)
    config["_public_context_status"] = _public_context_status(config, repo_root)
    frame = build_frame(config, repo_root)
    public_path = config.get("public_context_path")
    if public_path:
        resolved = Path(public_path)
        if not resolved.is_absolute():
            resolved = repo_root / resolved
        if resolved.exists():
            public = load_public_context(resolved)
            public["name"] = "ctx_" + public["name"].astype(str)
            frame = causal_context_join(frame, public)
    frame, specs = build_rule_catalog(frame)
    folds = make_walk_forward_folds(
        frame["effective_date"].min(), frame["effective_date"].max(),
        first_test_year=int(config.get("first_test_year", 2022)),
        oot_start=config.get("confirmation_start", "2025-09-01"),
    )
    screening_rows, selected_rows, metric_rows, prediction_rows = [], [], [], []
    for fold in folds:
        validation = purge_validation_tail(
            frame.loc[mask_val(frame, fold)].copy(), int(config["horizon"])
        )
        test = frame.loc[mask_test(frame, fold)].copy()
        if validation.empty or test.empty:
            continue
        screening = _screen(validation, specs, config, fold.name)
        screening_rows.append(screening)
        shortlist = _shortlist(screening, config)
        val_candidates = _candidate_signals(
            validation, shortlist, specs, filter_correlation=True
        )
        rule_name, cooldown, val_summary = _select_policy(validation, val_candidates, config)
        test_candidates = _candidate_signals(test, shortlist, specs)
        raw = test_candidates[rule_name]
        signal = apply_communication_policy(test, raw, cooldown_prints=cooldown)
        split = "confirmation" if fold.name == "oot" else fold.name
        summary, metrics = _summary(test, signal, config, split, detailed=True)
        metrics["selected_rule"] = rule_name
        metrics["cooldown_prints"] = cooldown
        metric_rows.append(metrics)
        selected_rows.append({
            "fold": fold.name, "reported_split": split, "selected_rule": rule_name,
            "cooldown_prints": cooldown, **{f"val_{k}": v for k, v in val_summary.items()},
            **{f"test_{k}": v for k, v in summary.items()},
        })
        test["selected_signal"] = signal
        test["search_fold"] = split
        prediction_rows.append(test)

    output_dir = repo_root / Path(config.get("reports_dir", "reports")) / "indicator_search"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(spec) for spec in specs]).to_csv(output_dir / "catalog.csv", index=False)
    screening_all = pd.concat(screening_rows, ignore_index=True) if screening_rows else pd.DataFrame()
    screening_all.to_csv(output_dir / "screening.csv", index=False)
    selected_table = pd.DataFrame(selected_rows)
    selected_table.to_csv(output_dir / "selected.csv", index=False)
    metrics_all = pd.concat(metric_rows, ignore_index=True) if metric_rows else pd.DataFrame()
    metrics_all.to_csv(output_dir / "metrics.csv", index=False)
    predictions = pd.concat(prediction_rows).sort_index() if prediction_rows else pd.DataFrame()
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    discovery = predictions[predictions["search_fold"].ne("confirmation")].copy()
    effect = _discovery_effect(discovery, config) if not discovery.empty else {}
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _write_report(output_dir / "REPORT.md", selected_table, metrics_all, effect, len(specs), config)
    return output_dir / "REPORT.md"


def run_indicator_report(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = load_yaml(config_path)
    repo_root = _repo_root(config_path)
    config["_public_context_status"] = _public_context_status(config, repo_root)
    output_dir = repo_root / Path(config.get("reports_dir", "reports")) / "indicator_search"
    predictions = pd.read_csv(
        output_dir / "predictions.csv",
        parse_dates=["effective_date", "available_at", "fetched_at"],
        low_memory=False,
    )
    selected = pd.read_csv(output_dir / "selected.csv")
    metrics = pd.read_csv(output_dir / "metrics.csv")
    catalog_size = len(pd.read_csv(output_dir / "catalog.csv"))
    discovery = predictions[predictions["search_fold"].ne("confirmation")]
    effect = _discovery_effect(discovery, config)
    report = output_dir / "REPORT.md"
    _write_report(report, selected, metrics, effect, catalog_size, config)
    return report


def _write_report(path: Path, selected: pd.DataFrame, metrics: pd.DataFrame, effect: dict,
                  catalog_size: int, config: dict) -> None:
    discovery = metrics[metrics["split"].ne("confirmation")] if not metrics.empty else metrics
    valid = discovery[discovery["signal_count"].gt(0)] if not discovery.empty else discovery
    mean_lift = float(effect.get("lift_point", np.nan))
    mean_bps = float(effect.get("bps_point", np.nan))
    mean_frequency = float(discovery["signals_per_week"].mean()) if not discovery.empty else 0.0
    positive_corridors = int(valid.groupby("corridor")["lift"].mean().gt(1).sum()) if not valid.empty else 0
    passed = (
        mean_lift >= 1.3 and mean_bps > 0 and positive_corridors >= 3
        and 0.8 <= mean_frequency <= 2.5
    )
    selected_lines = "\n".join(
        f"- {row.fold}: `{row.selected_rule}`, cooldown={row.cooldown_prints}"
        for row in selected.itertuples()
    ) or "- Нет выбранных правил."
    text = f"""# Indicator search report

## Итог

Проверено {catalog_size} одиночных правил и автоматически построенные интерпретируемые
ансамбли. Quality gate: **{'PASS' if passed else 'FAIL'}**. Отрицательный результат считается
валидным завершением перебора и не является основанием для продуктового запуска.

## Discovery outer folds

- Средний macro lift: {mean_lift:.3f}
- Средний эффект: {mean_bps:.1f} б.п.
- Средняя частота по коридорам: {mean_frequency:.3f} сигнала в неделю
- Коридоров со средним lift > 1: {positive_corridors}/5
- Публичный внешний контекст: {config.get('_public_context_status', 'unknown')}

## Выбранная внутри фолдов политика

{selected_lines}

Период после {config.get('confirmation_start', '2025-09-01')} показан только как повторно
использованный confirmation и не входит в discovery-оценки.
"""
    path.write_text(text, encoding="utf-8")
