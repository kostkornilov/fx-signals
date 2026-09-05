import pandas as pd

from fx_signal.splits import (
    WalkForwardFold,
    assert_no_overlap,
    make_walk_forward_folds,
    mask_test,
    mask_train,
    mask_val,
)


def test_expanding_folds_end_with_oot() -> None:
    folds = make_walk_forward_folds(
        pd.Timestamp("2019-01-01"), pd.Timestamp("2026-09-02")
    )
    names = [fold.name for fold in folds]
    assert names[0] == "wf_2022"
    assert names[-1] == "oot"
    assert folds[-1].test_start == pd.Timestamp("2025-09-01")
    assert folds[0].split == "y2022"
    assert folds[0].train_end < folds[0].val_start or folds[0].train_end == folds[0].val_start


def test_purge_drops_train_tail_and_does_not_overlap_val() -> None:
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    frame = pd.DataFrame(
        {
            "currency": ["TJS"] * 20,
            "effective_date": dates,
            "rub_per_unit": range(20),
        }
    )
    fold = WalkForwardFold(
        name="unit",
        split="wf_oos",
        train_end=pd.Timestamp("2020-01-08"),
        val_start=pd.Timestamp("2020-01-08"),
        val_end=pd.Timestamp("2020-01-15"),
        test_start=pd.Timestamp("2020-01-15"),
        test_end=pd.Timestamp("2020-01-21"),
    )
    train = mask_train(frame, fold, purge_horizon=3)
    test = mask_test(frame, fold)
    assert_no_overlap(frame, fold, purge_horizon=3)
    assert frame.loc[train, "effective_date"].max() < pd.Timestamp("2020-01-08")
    # Three publication days dropped from the train tail.
    assert int(train.sum()) == 4
    assert mask_val(frame, fold).sum() == 7
    assert not (train & test).any()
