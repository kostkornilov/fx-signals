from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
WIDE_SRC = ROOT / "reports" / "ml_lar_threshold" / "oot_macro_wide.csv"
RULES_SRC = ROOT / "reports" / "ml_lar_threshold" / "research_rules_oot.csv"
ML_SRC = ROOT / "reports" / "ml_lar_threshold" / "metrics.csv"

RANGE_LABEL = "Правило: диапазон удержался"
CATBOOST_LABEL = "CatBoost (A,B,C,D)"
CORRIDOR_ORDER = ["RUB→AMD", "RUB→KGS", "RUB→KZT", "RUB→TJS", "RUB→UZS"]


def _summary_frame(wide: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "method": wide["label"],
            "lift_h1": wide["lift_h1"],
            "lar_h1": wide["lar_h1"],
            "lift_h5": wide["lift_h5"],
            "lar_h5": wide["lar_h5"],
            "lift_h10": wide["lift_h10"],
            "lar_h10": wide["lar_h10"],
            "signals_per_week": wide["frequency"],
            "cluster_rate": wide["cluster"],
            "corridors_with_lar_h1": wide["n_h1"],
            "corridors_with_lar_h5": wide["n_h5"],
            "corridors_with_lar_h10": wide["n_h10"],
        }
    )


def write_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = pd.read_csv(WIDE_SRC)
    columns = _summary_frame(wide)
    lar = columns.sort_values(
        [
            "corridors_with_lar_h5",
            "lar_h5",
            "corridors_with_lar_h10",
            "lar_h10",
            "corridors_with_lar_h1",
            "lar_h1",
        ],
        ascending=False,
        na_position="last",
    )
    lift = columns.sort_values(
        [
            "corridors_with_lar_h5",
            "lift_h5",
            "corridors_with_lar_h10",
            "lift_h10",
            "corridors_with_lar_h1",
            "lift_h1",
        ],
        ascending=False,
        na_position="last",
    )
    TABLES.mkdir(parents=True, exist_ok=True)
    lar.to_csv(TABLES / "ml_summary_lar.csv", index=False)
    lift.to_csv(TABLES / "ml_summary_lift.csv", index=False)
    return lar, lift


def _corridor_lift() -> pd.DataFrame:
    rules = pd.read_csv(RULES_SRC)
    range_held = rules[rules["indicator"] == "signal_better_range_held"].copy()
    range_held["Подход"] = RANGE_LABEL
    ml = pd.read_csv(ML_SRC)
    catboost = ml[
        (ml["split"] == "oot")
        & (ml["method"] == "catboost")
        & (ml["feature_groups"] == "A,B,C,D")
    ].copy()
    catboost["Подход"] = CATBOOST_LABEL
    catboost["evaluation_horizon"] = catboost["evaluation_horizon"]
    pieces = pd.concat(
        [
            range_held[["evaluation_horizon", "corridor", "Подход", "lift"]],
            catboost[["evaluation_horizon", "corridor", "Подход", "lift"]],
        ],
        ignore_index=True,
    )
    pieces["Коридор"] = pieces["corridor"].str.replace("RUB->", "RUB→", regex=False)
    return pieces


def save_lift_charts() -> None:
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    data = _corridor_lift()
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    palette = {RANGE_LABEL: "#4c9f70", CATBOOST_LABEL: "#4c72b0"}
    for horizon in (1, 5, 10):
        slice_ = data[data["evaluation_horizon"] == horizon].copy()
        fig, axis = plt.subplots(figsize=(10, 5.5))
        sns.barplot(
            data=slice_,
            x="Коридор",
            y="lift",
            hue="Подход",
            order=CORRIDOR_ORDER,
            hue_order=[RANGE_LABEL, CATBOOST_LABEL],
            palette=palette,
            ax=axis,
        )
        axis.axhline(1.0, color="#666666", linestyle=":", linewidth=1.1)
        axis.axhline(1.3, color="#333333", linestyle="--", linewidth=1.1)
        axis.set_ylabel("Lift")
        axis.set_xlabel("")
        ymax = max(1.5, float(slice_["lift"].max()) * 1.18)
        axis.set_ylim(0, ymax)
        axis.set_title(
            f"Final OOT, h={horizon}: lift по коридорам\n"
            "диапазон удержался vs CatBoost (A,B,C,D)",
            fontsize=13,
            pad=12,
        )
        axis.legend(title="", loc="upper right", frameon=True)
        axis.text(
            0.01,
            1.0 / ymax,
            "  случайный день",
            transform=axis.get_yaxis_transform(),
            va="bottom",
            fontsize=8,
            color="#666666",
        )
        axis.text(
            0.01,
            1.3 / ymax,
            "  ориентир 1.3",
            transform=axis.get_yaxis_transform(),
            va="bottom",
            fontsize=8,
            color="#333333",
        )
        fig.tight_layout()
        name = f"ml_top2_lift_h{horizon}.png"
        fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
        fig.savefig(TABLES / name, dpi=180, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    write_summaries()
    save_lift_charts()


if __name__ == "__main__":
    main()
