from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    name: str
    split: str
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_walk_forward_folds(
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
    *,
    first_test_year: int = 2022,
    oot_start: str | pd.Timestamp = "2025-09-01",
) -> list[WalkForwardFold]:
    """Expanding yearly folds plus a final untouched OOT year.

    Train is everything before validation. Validation is the year before test.
    The last fold (`oot`) never shares its test window with earlier folds.
    """
    min_date = pd.Timestamp(min_date)
    max_date = pd.Timestamp(max_date) + pd.DateOffset(days=1)
    oot_start = pd.Timestamp(oot_start)
    folds: list[WalkForwardFold] = []

    year = first_test_year
    while True:
        test_start = pd.Timestamp(f"{year}-01-01")
        test_end = pd.Timestamp(f"{year + 1}-01-01")
        if test_start >= oot_start:
            break
        test_end = min(test_end, oot_start)
        if test_end <= test_start:
            break
        val_start = pd.Timestamp(f"{year - 1}-01-01")
        val_end = test_start
        train_end = val_start
        if train_end <= min_date + pd.DateOffset(months=6):
            year += 1
            continue
        folds.append(
            WalkForwardFold(
                name=f"wf_{year}",
                split="y2022" if year == 2022 else "wf_oos",
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        year += 1

    oot_end = max_date
    if oot_end > oot_start:
        folds.append(
            WalkForwardFold(
                name="oot",
                split="oot",
                train_end=pd.Timestamp("2024-09-01"),
                val_start=pd.Timestamp("2024-09-01"),
                val_end=oot_start,
                test_start=oot_start,
                test_end=oot_end,
            )
        )
    return folds


def _in_range(dates: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return dates.ge(start) & dates.lt(end)


def mask_train(
    frame: pd.DataFrame, fold: WalkForwardFold, purge_horizon: int
) -> pd.Series:
    """Rows strictly before validation, dropping the last `purge_horizon` prints.

    Labels look `horizon` steps ahead, so the tail of train is removed per currency.
    """
    dates = frame["effective_date"]
    in_train = dates.lt(fold.train_end)
    keep = pd.Series(False, index=frame.index)
    for _, group in frame.loc[in_train].groupby("currency", sort=False):
        if purge_horizon:
            group = group.iloc[:-purge_horizon] if len(group) > purge_horizon else group.iloc[0:0]
        keep.loc[group.index] = True
    return keep


def mask_val(frame: pd.DataFrame, fold: WalkForwardFold) -> pd.Series:
    return _in_range(frame["effective_date"], fold.val_start, fold.val_end)


def mask_test(frame: pd.DataFrame, fold: WalkForwardFold) -> pd.Series:
    return _in_range(frame["effective_date"], fold.test_start, fold.test_end)


def assert_no_overlap(frame: pd.DataFrame, fold: WalkForwardFold, purge_horizon: int) -> None:
    train = mask_train(frame, fold, purge_horizon)
    val = mask_val(frame, fold)
    test = mask_test(frame, fold)
    if (train & val).any() or (train & test).any() or (val & test).any():
        raise AssertionError(f"Fold {fold.name} has overlapping split masks")
