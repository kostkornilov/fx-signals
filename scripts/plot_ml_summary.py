from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports" / "tables" / "ml_summary.csv"
OUTPUT = ROOT / "reports" / "figures"
SPLIT_NAMES = {
    "y2022": "2022 (стресс-период)",
    "wf_oos": "2023 — авг. 2025\n(walk-forward test)",
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


def save_overview(data: pd.DataFrame) -> None:
    work = data.copy()
    work["Метод"] = work.apply(method_label, axis=1)
    aggregate = (
        work.groupby(["split", "Метод"], as_index=False)
        .agg(lift=("lift", "mean"), bps=("bps_forward", "mean"), frequency=("signals_per_week", "mean"))
    )
    order = list(SPLIT_NAMES)
    method_order = list(dict.fromkeys(aggregate["Метод"]))
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True)
    specs = [
        ("lift", "Lift относительно обычного дня", 1.0),
        ("bps", "Выгода относительно будущего среднего, б.п.", 0.0),
        ("frequency", "Сигналов в неделю", None),
    ]
    for axis, (metric, title, reference) in zip(axes, specs, strict=True):
        pivot = aggregate.pivot(index="Метод", columns="split", values=metric).reindex(
            index=method_order, columns=order
        )
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f" if metric != "bps" else ".0f",
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
        "Правила и модели на цели «в следующие 5 публикаций дешевле не станет»",
        fontsize=16,
        y=1.02,
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
    selected = data[
        (data["split"] == "oot")
        & (
            data["method"].isin(["random", "momentum", "level", "reversal", "seasonality"])
            | ((data["method"] == "catboost") & (data["feature_groups"] == "A,B"))
            | ((data["method"] == "logreg") & (data["feature_groups"] == "A,B,C"))
        )
    ].copy()
    selected["Метод"] = selected.apply(method_label, axis=1)
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
    axis.set_title("Final OOT: где метод лучше случайного дня", fontsize=15, pad=14)
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


def main() -> None:
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT)
    data["feature_groups"] = data["feature_groups"].replace({np.nan: ""})
    save_overview(data)
    save_oot_corridors(data)


if __name__ == "__main__":
    main()
