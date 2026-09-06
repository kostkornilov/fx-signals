from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports" / "tables" / "ml_summary.csv"
OUTPUT = ROOT / "reports" / "figures"
HORIZON = 5
LAR_GATE = 1.3
N_TOP = 7
N_BOTTOM = 5
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


def _oot() -> pd.DataFrame:
    data = pd.read_csv(INPUT)
    data["feature_groups"] = data["feature_groups"].fillna("")
    work = data[(data["fold"] == "oot") & (data["evaluation_horizon"] == HORIZON)].copy()
    work["Подход"] = work.apply(method_label, axis=1)
    work["Коридор"] = work["corridor"].str.replace("RUB->", "RUB→", regex=False)
    work["срез"] = work["Подход"] + " · " + work["Коридор"]
    return work


def _bars(axis, frame: pd.DataFrame, *, title: str, color: str) -> None:
    axis.barh(frame["срез"], frame["lift_at_risk"], color=color, height=0.65)
    axis.axvline(1.0, color="#333333", linestyle=":", linewidth=1.1)
    axis.axvline(LAR_GATE, color="#333333", linestyle="--", linewidth=1.2)
    axis.set_xlabel("Lift-at-Risk (fixed)")
    axis.set_title(title, fontsize=12)
    axis.invert_yaxis()
    for y, value in enumerate(frame["lift_at_risk"]):
        axis.text(value + 0.08, y, f"{value:.2f}", va="center", fontsize=9)


def main() -> None:
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    work = _oot()
    passed = work[work["lift_at_risk"] >= LAR_GATE].sort_values("lift_at_risk", ascending=False)
    top = passed.head(N_TOP).copy()
    median = (
        work.dropna(subset=["lift_at_risk"])
        .groupby("Подход", as_index=False)["lift_at_risk"]
        .median()
        .sort_values("lift_at_risk")
        .head(N_BOTTOM)
        .rename(columns={"Подход": "срез"})
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=False)
    _bars(
        axes[0],
        top,
        title=f"Топ-{N_TOP} срезов с LAR ≥ {LAR_GATE}\nfinal OOT, h={HORIZON}, основная матрица",
        color="#4c9f70",
    )
    _bars(
        axes[1],
        median,
        title=f"Топ-{N_BOTTOM} подходов с наименьшим медианным LAR\nпо пяти коридорам, final OOT, h={HORIZON}",
        color="#c44e52",
    )
    axes[0].set_xlim(0, top["lift_at_risk"].max() * 1.18)
    axes[1].set_xlim(0, max(LAR_GATE, median["lift_at_risk"].max()) * 1.25)
    fig.suptitle("Lift-at-Risk: лидеры и аутсайдеры", fontsize=14, y=1.02)
    fig.text(
        0.5,
        -0.04,
        "Пунктир — порог 1.3, точка — нейтраль 1. "
        "Эксперимент с индексами MOEX (LogReg A–F) на этом срезе не ранжируется: на OOT нет сигналов, LAR не определён. "
        "LogReg (A,B) и LogReg (A,B,C,D) тоже без сигналов во всех коридорах.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"ml_lar_leaders_h{HORIZON}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(path)
    print("TOP")
    print(top[["срез", "lift_at_risk", "lift", "signals_per_week"]].to_string(index=False))
    print("BOTTOM")
    print(median.to_string(index=False))


if __name__ == "__main__":
    main()
