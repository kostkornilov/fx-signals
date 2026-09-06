from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports" / "research_external"
OUTPUT = INPUT / "figures"
SPLIT_ORDER = ("y2022", "wf_oos", "oot")
SPLIT_NAMES = {
    "y2022": "2022\n(стресс)",
    "wf_oos": "2023 — авг. 2025",
    "oot": "сен. 2025 — сен. 2026\n(final OOT)",
}
EXPERIMENT_NAMES = {
    "cbr_full_logreg": "LogReg без MOEX\n(A,B,C,E)",
    "all_external_logreg": "LogReg + MOEX\n(A–F)",
}


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(INPUT / "ablation_metrics.csv")
    comparisons = pd.read_csv(INPUT / "paired_comparisons.csv")
    audit = pd.read_csv(INPUT / "data_audit.csv")
    metrics["Модель"] = metrics["experiment"].map(EXPERIMENT_NAMES).fillna(metrics["experiment"])
    metrics["Период"] = pd.Categorical(
        metrics["split"].map(SPLIT_NAMES),
        categories=[SPLIT_NAMES[name] for name in SPLIT_ORDER],
        ordered=True,
    )
    metrics["Коридор"] = metrics["corridor"].str.replace("RUB->", "RUB→", regex=False)
    return metrics, comparisons, audit


def save_report(metrics: pd.DataFrame, comparisons: pd.DataFrame, audit: pd.DataFrame) -> Path:
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    _signals(axes[0, 0], metrics)
    _oot_lift(axes[0, 1], metrics)
    _walk_forward_lift(axes[1, 0], metrics)
    _gate(axes[1, 1], metrics)
    fig.suptitle(
        "Дополнительные ряды: LogReg на курсах ЦБ против LogReg с индексами MOEX\n"
        "IMOEX, RGBI и RVI входят в группу F с лагом публикации 1 день; "
        "это не курсы перевода, а фон российского рынка",
        fontsize=14,
        y=1.02,
    )
    fig.text(
        0.5,
        -0.02,
        "Пустая клетка и нулевая частота — модель не выдала сигналов, метрика не определена. "
        "Пунктир lift = 1.3 и зелёная полоса 1–2 сигнала в неделю — требования кейса.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    path = OUTPUT / "external_report.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _signals(axis, metrics: pd.DataFrame) -> None:
    work = (
        metrics.groupby(["Период", "Модель"], observed=True, as_index=False)["signals_per_week"]
        .mean()
    )
    sns.barplot(
        data=work,
        x="Период",
        y="signals_per_week",
        hue="Модель",
        ax=axis,
        palette="colorblind",
    )
    axis.axhspan(1, 2, color="#4c9f70", alpha=0.15, zorder=0)
    axis.set_ylabel("сигналов в неделю")
    axis.set_xlabel("")
    axis.set_title("Частота сигналов по периодам")
    axis.legend(fontsize=8, loc="upper right")


def _heatmap(axis, metrics: pd.DataFrame, split: str, title: str) -> None:
    work = metrics[metrics["split"].eq(split)].copy()
    models = [EXPERIMENT_NAMES[name] for name in EXPERIMENT_NAMES if name in set(work["experiment"])]
    corridors = sorted(work["Коридор"].unique())
    horizons = sorted(work["horizon"].unique())
    blocks = []
    ytick = []
    for model in models:
        part = work[work["Модель"].eq(model)]
        pivot = part.pivot_table(index="Коридор", columns="horizon", values="lift", aggfunc="mean")
        pivot = pivot.reindex(index=corridors, columns=horizons)
        blocks.append(pivot.to_numpy(dtype=float))
        ytick.extend([f"{model.splitlines()[0]} · {corridor}" for corridor in corridors])
    grid = np.vstack(blocks)
    sns.heatmap(
        grid,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=1.3,
        vmin=0,
        vmax=2.6,
        linewidths=0.5,
        cbar=False,
        ax=axis,
        xticklabels=[f"h={value}" for value in horizons],
        yticklabels=ytick,
    )
    axis.set_title(title)
    axis.set_xlabel("")


def _oot_lift(axis, metrics: pd.DataFrame) -> None:
    _heatmap(axis, metrics, "oot", "Lift на final OOT: пустые клетки — молчание")


def _walk_forward_lift(axis, metrics: pd.DataFrame) -> None:
    _heatmap(axis, metrics, "wf_oos", "Lift на 2023 — авг. 2025, где обе модели ещё говорят")


def _gate(axis, metrics: pd.DataFrame) -> None:
    work = metrics.copy()
    work["Lift ≥ 1.3"] = work["lift"] >= 1.3
    work["Выгода > 0"] = work["bps_forward"] > 0
    work["Частота 1–2 / нед."] = work["signals_per_week"].between(1, 2)
    columns = ["Lift ≥ 1.3", "Выгода > 0", "Частота 1–2 / нед."]
    rows = []
    for (split, model), part in work.groupby(["split", "Модель"], observed=True):
        row = {"Период": SPLIT_NAMES[split], "Модель": model}
        for column in columns:
            row[column] = 100 * part[column].mean()
        rows.append(row)
    table = pd.DataFrame(rows)
    table["метка"] = table["Модель"].str.replace("\n", " ", regex=False) + " · " + table["Период"].str.replace(
        "\n", " ", regex=False
    )
    share = table.set_index("метка")[columns]
    sns.heatmap(
        share,
        annot=True,
        fmt=".0f",
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        linewidths=0.6,
        cbar_kws={"label": "доля срезов, %"},
        ax=axis,
    )
    axis.set_title("Доля срезов, прошедших требование кейса")
    axis.set_xlabel("")
    axis.set_ylabel("")


def main() -> None:
    path = save_report(*_load())
    print(path)


if __name__ == "__main__":
    main()
