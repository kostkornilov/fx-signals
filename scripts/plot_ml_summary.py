from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports" / "tables" / "ml_summary.csv"
DIAGNOSTICS = ROOT / "reports" / "tables" / "ml_diagnostics.csv"
OUTPUT = ROOT / "reports" / "figures"
LADDER = ["A", "A,B", "A,B,C", "A,B,C,D"]
SPLIT_NAMES = {
    "wf_2022": "2022\n(стресс-период)",
    "wf_2023": "2023",
    "wf_2024": "2024",
    "wf_2025": "янв.–авг. 2025",
    "oot": "сен. 2025 — сен. 2026\n(final OOT)",
}
METHOD_NAMES = {
    "random": "Случайный день",
    "momentum": "Правило: momentum",
    "level": "Правило: level",
    "reversal": "Правило: reversal",
    "seasonality": "Правило: seasonality",
}


def method_label(row: pd.Series) -> str:
    method = str(row["method"])
    if method in METHOD_NAMES:
        return METHOD_NAMES[method]
    groups = row.get("feature_groups")
    suffix = str(groups) if pd.notna(groups) and str(groups) else "—"
    return f"{method.upper()} ({suffix})"


def horizon_slice(data: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    work = data[data["evaluation_horizon"] == horizon].copy()
    work = work[work["fold"].isin(SPLIT_NAMES)]
    work["Метод"] = work.apply(method_label, axis=1)
    return work


def save_overview(data: pd.DataFrame) -> None:
    work = horizon_slice(data)
    aggregate = work.groupby(["fold", "Метод"], as_index=False).agg(
        lift=("lift", "mean"),
        advantage=("moment_advantage_bps", "mean"),
        lar=("lift_at_risk", "mean"),
        frequency=("signals_per_week", "mean"),
    )
    order = list(SPLIT_NAMES)
    method_order = list(dict.fromkeys(aggregate["Метод"]))
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharey=True)
    specs = [
        ("lift", "Lift относительно обычного дня", 1.0, ".2f"),
        ("advantage", "Выгода момента ±h, б.п.", 0.0, ".0f"),
        ("lar", "Lift-at-Risk (fixed)", 1.0, ".2f"),
        ("frequency", "Сигналов в неделю", None, ".2f"),
    ]
    for axis, (metric, title, reference, fmt) in zip(axes.ravel(), specs, strict=True):
        pivot = aggregate.pivot(index="Метод", columns="fold", values=metric).reindex(
            index=method_order, columns=order
        )
        sns.heatmap(
            pivot,
            annot=True,
            fmt=fmt,
            cmap="RdYlGn",
            center=reference,
            linewidths=0.5,
            cbar=False,
            ax=axis,
        )
        axis.set_title(title, fontsize=12, pad=12)
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_xticklabels([SPLIT_NAMES[item] for item in order], rotation=0, fontsize=9)
    fig.suptitle(
        "Правила и модели, оценка h=5 по метрикам metrics-ground.md",
        fontsize=16,
        y=1.01,
    )
    fig.text(
        0.5,
        -0.02,
        "Каждая ячейка — простое среднее по пяти валютным коридорам; пусто означает, что метод не сформировал сигналов.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT / "ml_methods_by_split.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_oot_corridors(data: pd.DataFrame) -> None:
    selected = horizon_slice(data)
    selected = selected[
        (selected["fold"] == "oot")
        & (
            selected["method"].isin(["random", "momentum", "level", "reversal", "seasonality"])
            | ((selected["method"] == "catboost") & (selected["feature_groups"] == "A,B"))
            | ((selected["method"] == "logreg") & (selected["feature_groups"] == "A,B,C"))
        )
    ].copy()
    selected["Коридор"] = selected["corridor"].str.replace("RUB->", "RUB→", regex=False)
    order = [
        "Случайный день",
        "Правило: momentum",
        "Правило: level",
        "Правило: reversal",
        "Правило: seasonality",
        "LOGREG (A,B,C)",
        "CATBOOST (A,B)",
    ]
    pivot = selected.pivot(index="Метод", columns="Коридор", values="lift").reindex(order)
    fig, axis = plt.subplots(figsize=(11, 7))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=1.0,
        linewidths=0.7,
        cbar_kws={"label": "Lift"},
        ax=axis,
    )
    axis.set_title("Final OOT, h=5: где метод лучше случайного дня", fontsize=15, pad=14)
    axis.set_xlabel("")
    axis.set_ylabel("")
    fig.text(
        0.5,
        0.01,
        "Зелёный и lift > 1 — лучше обычного дня; красный и lift < 1 — хуже. Период: сентябрь 2025 — сентябрь 2026.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUTPUT / "ml_oot_lift_by_corridor.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_requirements_gate(data: pd.DataFrame, diagnostics: pd.DataFrame, horizon: int) -> None:
    """Три требования кейса на одном полотне: точность, значимая выгода, целевая частота."""
    keys = ["experiment", "currency", "fold", "evaluation_horizon"]
    work = horizon_slice(data, horizon).merge(
        diagnostics[keys + ["moment_advantage_bps_ci_low"]], on=keys, how="left"
    )
    work["Lift ≥ 1.3"] = work["lift"] >= 1.3
    work["Выгода момента > 0\n(нижняя граница 95% CI)"] = work["moment_advantage_bps_ci_low"] > 0
    work["Частота 1–2 сигнала\nв неделю"] = work["signals_per_week"].between(1, 2)
    columns = list(work.columns[-3:])
    work["Все три требования\nодновременно"] = work[columns].all(axis=1)
    columns.append("Все три требования\nодновременно")
    share = work.groupby("Метод")[columns].mean().mul(100)
    share = share.reindex(share["Все три требования\nодновременно"].sort_values().index)
    fig, axis = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        share,
        annot=True,
        fmt=".0f",
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        linewidths=0.7,
        cbar_kws={"label": "Доля пройденных срезов, %"},
        ax=axis,
    )
    axis.set_title(
        f"Требования кейса на h={horizon}: доля пройденных срезов из 25\n"
        "(5 коридоров × 5 периодов walk-forward и OOT)",
        fontsize=14,
        pad=16,
    )
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.set_xticklabels(axis.get_xticklabels(), rotation=0, fontsize=9)
    fig.text(
        0.5,
        -0.04,
        "Неопределённая метрика считается непройденным требованием: отсутствие сигналов не является доказательством качества.\n"
        "Требования проверяются одновременно, поэтому последний столбец не равен произведению первых трёх.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT / f"ml_requirements_gate_h{horizon}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_feature_ladder(data: pd.DataFrame, horizon: int) -> None:
    """LogReg против CatBoost на всех четырёх ступенях групп признаков."""
    work = horizon_slice(data, horizon)
    work = work[(work["fold"] == "oot") & work["feature_groups"].isin(LADDER)]
    work["Коридор"] = work["corridor"].str.replace("RUB->", "RUB→", regex=False)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for axis, method in zip(axes, ("logreg", "catboost"), strict=True):
        pivot = work[work["method"] == method].pivot(
            index="feature_groups", columns="Коридор", values="lift"
        ).reindex(LADDER)
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=1.3,
            vmin=0,
            vmax=2.6,
            linewidths=0.7,
            cbar=False,
            ax=axis,
        )
        defined = int(pivot.notna().to_numpy().sum())
        axis.set_title(f"{method.upper()}: определён lift в {defined} из 20 клеток", fontsize=12)
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_yticks(np.arange(len(LADDER)) + 0.5)
        axis.set_yticklabels(LADDER, rotation=0)
    axes[0].set_ylabel("Накопительные группы признаков")
    fig.suptitle(
        f"Вклад групп признаков на одинаковых ступенях, final OOT (сен. 2025 — сен. 2026), h={horizon}\n"
        "Зелёный — lift не ниже требования 1.3, красный — ниже",
        fontsize=14,
    )
    fig.text(
        0.5,
        -0.03,
        "Пустая клетка — модель не выдала ни одного сигнала в этом коридоре, поэтому lift не определён.\n"
        "CatBoost на A и A,B,C обучен отдельно: легаси-матрица эти две ступени пропускала, "
        "и сравнение методов на них было невозможно.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT / f"ml_feature_ladder_h{horizon}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT)
    data["feature_groups"] = data["feature_groups"].replace({np.nan: ""})
    diagnostics = pd.read_csv(DIAGNOSTICS)
    save_overview(data)
    save_oot_corridors(data)
    # Все пять горизонтов показываются отдельно: усреднение по ним не предусмотрено метриками.
    for horizon in sorted(data["evaluation_horizon"].unique()):
        save_requirements_gate(data, diagnostics, horizon)
        save_feature_ladder(data, horizon)


if __name__ == "__main__":
    main()
