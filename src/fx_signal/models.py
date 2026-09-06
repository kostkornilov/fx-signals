from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fx_signal.metrics import fixed_window_lift_at_risk, forward_bps, signals_per_week

RankingMetric = Literal["lift", "lar"]
LAR_THRESHOLD_DRAWS = 50

SATURATION_SIGNALS_PER_WEEK = 1.0

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostClassifier = None


@dataclass
class FittedScorer:
    kind: str
    model: object
    feature_cols: list[str]
    calibrator: IsotonicRegression | None = None

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self._raw_proba(frame)
        if self.calibrator is None:
            return raw
        return np.clip(self.calibrator.predict(raw), 0.0, 1.0)

    def _raw_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.kind == "logreg":
            return self.model.predict_proba(_logreg_frame(frame, self.feature_cols))[:, 1]
        features = _boost_frame(frame, self.feature_cols)
        return self.model.predict_proba(features)[:, 1]


def _logreg_frame(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    numeric = frame.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce")
    currency = frame[["currency"]].astype(str)
    return pd.concat([numeric.reset_index(drop=True), currency.reset_index(drop=True)], axis=1)


def _boost_frame(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    features = frame.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce").copy()
    features["currency"] = frame["currency"].astype(str).to_numpy()
    return features


def fit_scorer(
    train: pd.DataFrame,
    val: pd.DataFrame,
    *,
    kind: str,
    feature_cols: list[str],
    target_col: str,
) -> FittedScorer:
    y_train = train[target_col].astype(int)
    if kind == "logreg":
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        model = Pipeline(
            [
                (
                    "prep",
                    _ColumnPrep(feature_cols, encoder),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=1000,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        model.fit(_logreg_frame(train, feature_cols), y_train)
    elif kind == "catboost":
        if CatBoostClassifier is None:
            raise ImportError("catboost is required for the catboost method")
        model = CatBoostClassifier(
            depth=3,
            iterations=200,
            learning_rate=0.05,
            loss_function="Logloss",
            auto_class_weights="SqrtBalanced",
            random_seed=0,
            verbose=False,
            od_type="Iter",
            od_wait=20,
            allow_writing_files=False,
        )
        fit_kwargs: dict = {
            "cat_features": ["currency"],
        }
        if not val.empty and val[target_col].nunique() > 1:
            fit_kwargs["eval_set"] = (
                _boost_frame(val, feature_cols),
                val[target_col].astype(int),
            )
        model.fit(
            _boost_frame(train, feature_cols),
            y_train,
            **fit_kwargs,
        )
    else:
        raise ValueError(f"Unknown model kind: {kind}")

    scorer = FittedScorer(kind=kind, model=model, feature_cols=feature_cols)
    if not val.empty and val[target_col].nunique() > 1:
        raw = scorer._raw_proba(val)
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw, val[target_col].astype(int))
        scorer.calibrator = calibrator
    return scorer


class _ColumnPrep(BaseEstimator, TransformerMixin):
    """Impute and scale numeric features, one-hot encode currency."""

    def __init__(self, feature_cols: list[str], encoder: OneHotEncoder) -> None:
        self.feature_cols = feature_cols
        self.encoder = encoder
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    def fit(self, frame: pd.DataFrame, y=None):
        numeric = frame[self.feature_cols]
        self.imputer.fit(numeric)
        scaled = self.imputer.transform(numeric)
        self.scaler.fit(scaled)
        self.encoder.fit(frame[["currency"]])
        return self

    def transform(self, frame: pd.DataFrame):
        numeric = self.scaler.transform(self.imputer.transform(frame[self.feature_cols]))
        cats = self.encoder.transform(frame[["currency"]])
        return np.hstack([numeric, cats])


def _saturating_score(
    weekly: float,
    metric: float,
    *,
    saturation: float = SATURATION_SIGNALS_PER_WEEK,
) -> float:
    if saturation <= 0:
        return metric
    return metric * min(weekly / saturation, 1.0)


def _macro_lar(
    val: pd.DataFrame,
    pred: np.ndarray,
    *,
    target_col: str,
    horizon: int,
) -> float:
    """Mean fixed-window LAR across currencies with at least one eligible signal."""
    work = val.copy()
    work["__threshold_signal__"] = pred
    rng = np.random.default_rng(0)
    scores: list[float] = []
    for _, group in work.groupby("currency", sort=False):
        result = fixed_window_lift_at_risk(
            group,
            horizon=horizon,
            target_col=target_col,
            signal_col="__threshold_signal__",
            baseline_draws=LAR_THRESHOLD_DRAWS,
            rng=rng,
        )
        score = result["lift_at_risk"]
        if isinstance(score, (int, float)) and math.isfinite(float(score)):
            scores.append(float(score))
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def select_threshold(
    val: pd.DataFrame,
    proba: np.ndarray,
    *,
    target_col: str,
    horizon: int,
    grid: list[float],
    quantiles: list[float],
    min_signals_per_week: float,
    max_signals_per_week: float,
    saturation_signals_per_week: float = SATURATION_SIGNALS_PER_WEEK,
    ranking_metric: RankingMetric = "lar",
) -> float:
    """Pick a send threshold by a saturating ranking metric at 1 signal/week.

    Default ``ranking_metric="lar"`` uses mean corridor Lift-at-Risk.
    ``"lift"`` uses author lift. Both are multiplied by ``min(signals_per_week, 1)``.

    ``min_signals_per_week`` is accepted for call-site compatibility and is not
    used: undershooting one signal per week is already inside the saturating
    score. Candidates above ``max_signals_per_week`` are dropped when any
    remaining candidate has positive mean bps.
    """
    del min_signals_per_week
    if ranking_metric not in ("lift", "lar"):
        raise ValueError("ranking_metric must be 'lift' or 'lar'")
    has_fact = val["has_fact"].fillna(False).to_numpy()
    target = val[target_col].astype(bool).to_numpy()
    dates = val["effective_date"]
    bps = forward_bps(val["rub_per_unit"], val[f"forward_mean_h{horizon}"]).to_numpy()
    candidates = set(grid)
    fact_proba = proba[has_fact]
    for quantile in quantiles:
        if len(fact_proba):
            candidates.add(float(np.quantile(fact_proba, 1.0 - quantile)))

    scored: list[tuple[float, float, float, float]] = []
    for threshold in sorted(candidates):
        pred = (proba >= threshold) & has_fact
        weekly = signals_per_week(dates, pd.Series(pred, index=val.index))
        if pred.sum() == 0:
            continue
        if ranking_metric == "lar":
            metric = _macro_lar(
                val,
                pred,
                target_col=target_col,
                horizon=horizon,
            )
            if not math.isfinite(metric):
                continue
        else:
            hit = float(target[pred].mean())
            prevalence = float(target.mean()) if len(target) else 0.0
            metric = hit / prevalence if prevalence else 0.0
        mean_bps = float(np.nanmean(bps[pred]))
        scored.append((threshold, weekly, metric, mean_bps))

    positive = [item for item in scored if item[3] > 0]
    capped = [item for item in positive if item[1] <= max_signals_per_week]
    pool = capped or positive
    if not pool:
        return 0.8

    def sort_key(item: tuple[float, float, float, float]) -> tuple[float, float]:
        _threshold, weekly, metric, mean_bps = item
        return (
            _saturating_score(
                weekly,
                metric,
                saturation=saturation_signals_per_week,
            ),
            mean_bps,
        )

    return max(pool, key=sort_key)[0]
