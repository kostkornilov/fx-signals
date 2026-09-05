from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fx_signal.metrics import forward_bps, signals_per_week

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
) -> float:
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
        hit = float(target[pred].mean())
        prevalence = float(target.mean()) if len(target) else 0.0
        lift = hit / prevalence if prevalence else 0.0
        mean_bps = float(np.nanmean(bps[pred]))
        scored.append((threshold, weekly, lift, mean_bps))

    feasible = [
        item
        for item in scored
        if min_signals_per_week <= item[1] <= max_signals_per_week and item[3] > 0
    ]
    pool = feasible or [item for item in scored if item[3] > 0] or scored
    if not pool:
        return 0.8
    return max(pool, key=lambda item: (item[2], item[3], -abs(item[1] - 1.5)))[0]
