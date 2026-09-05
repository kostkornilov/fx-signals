from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    family: str
    description: str
    params: dict[str, float | int | str]

    def record(self) -> dict:
        return asdict(self)


def _rolling_percentile(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window, min_periods=window).apply(
        lambda x: float((x <= x[-1]).mean()), raw=True
    )


def _rsi(price: pd.Series, window: int) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = -delta.clip(upper=0).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    result = 100 - 100 / (1 + rs)
    return result.where(loss.ne(0), 100.0).where(gain.ne(0), 0.0)


def _slope_z(price: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    slope = price.rolling(window, min_periods=window).apply(
        lambda y: float(np.polyfit(x, y, 1)[0]), raw=True
    )
    scale = price.pct_change().rolling(window, min_periods=window).std() * price
    return slope / scale.replace(0, np.nan)


def _per_currency(frame: pd.DataFrame, fn: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
    pieces = []
    for _, group in frame.groupby("currency", sort=False):
        pieces.append(fn(group.sort_values("effective_date").copy()))
    return pd.concat(pieces).sort_index()


def build_rule_catalog(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[RuleSpec]]:
    """Build a finite catalog of causal, human-readable rule signals."""
    specs: list[RuleSpec] = []

    def add(
        group: pd.DataFrame,
        family: str,
        name: str,
        signal: pd.Series,
        description: str,
        **params,
    ) -> None:
        rule_id = f"{family}__{name}"
        group[rule_id] = signal.astype("boolean")
        if not any(item.rule_id == rule_id for item in specs):
            specs.append(RuleSpec(rule_id, family, description, params))

    def price_rules(group: pd.DataFrame) -> pd.DataFrame:
        price = group["rub_per_unit"].astype(float)
        ret1 = price.pct_change()
        for window in (20, 60, 90, 120, 250):
            pct = _rolling_percentile(price, window)
            for threshold in (0.05, 0.10, 0.20):
                add(group, "level", f"pct_w{window}_p{int(threshold*100):02d}", pct.le(threshold),
                    f"Курс входит в нижние {threshold:.0%} за {window} публикаций",
                    window=window, threshold=threshold)
        for window in (20, 60, 120):
            low = price.rolling(window, min_periods=window).min()
            distance_bp = (price / low - 1) * 1e4
            for tolerance in (0, 10, 25, 50):
                add(group, "level", f"near_min_w{window}_bp{tolerance}",
                    distance_bp.le(tolerance),
                    f"Курс не дальше {tolerance} б.п. от минимума за {window} публикаций",
                    window=window, tolerance_bp=tolerance)
        for kind in ("sma", "ema"):
            for window in (10, 20, 60, 120):
                center = (price.rolling(window, min_periods=window).mean() if kind == "sma"
                          else price.ewm(span=window, min_periods=window, adjust=False).mean())
                std = price.rolling(window, min_periods=window).std()
                z = (price - center) / std.replace(0, np.nan)
                for threshold in (-0.5, -1.0, -1.5):
                    tag = str(abs(threshold)).replace(".", "p")
                    add(group, "level", f"z_{kind}_w{window}_m{tag}", z.le(threshold),
                        f"Курс ниже {kind.upper()}({window}) на {abs(threshold)} sigma",
                        kind=kind, window=window, threshold=threshold)
        for window in (10, 20, 60):
            mean = price.rolling(window, min_periods=window).mean()
            std = price.rolling(window, min_periods=window).std()
            for width in (1.0, 1.5, 2.0):
                tag = str(width).replace(".", "p")
                add(group, "level", f"boll_w{window}_k{tag}", price.le(mean-width*std),
                    f"Курс ниже нижней Bollinger band {window}/{width}",
                    window=window, width=width)

        falling = ret1.lt(0)
        streak = falling.groupby((~falling).cumsum()).cumsum()
        for length in (2, 3, 4, 5):
            add(group, "momentum", f"down_streak_{length}", streak.ge(length),
                f"Курс снижается не менее {length} публикаций подряд", length=length)
        for horizon in (1, 3, 5, 10, 20):
            ret = price.pct_change(horizon)
            for window in (60, 120):
                for quantile in (0.10, 0.20):
                    threshold = ret.rolling(window, min_periods=window).quantile(quantile)
                    add(group, "momentum", f"ret_h{horizon}_w{window}_p{int(quantile*100)}",
                        ret.le(threshold),
                        f"Падение за {horizon} публикаций в нижнем {quantile:.0%} диапазоне",
                        horizon=horizon, window=window, quantile=quantile)
        for window in (5, 10, 20, 60):
            slope = _slope_z(price, window)
            for threshold in (-0.5, -1.0):
                tag = str(abs(threshold)).replace(".", "p")
                add(group, "momentum", f"slope_w{window}_m{tag}", slope.le(threshold),
                    f"Нормированный нисходящий тренд за {window} публикаций",
                    window=window, threshold=threshold)
        for window in (5, 10, 14, 20):
            rsi = _rsi(price, window)
            for threshold in (20, 30, 40):
                add(group, "oversold", f"rsi_w{window}_p{threshold}", rsi.le(threshold),
                    f"RSI({window}) не выше {threshold}", window=window, threshold=threshold)
        for window in (10, 20, 60):
            low = price.rolling(window, min_periods=window).min()
            high = price.rolling(window, min_periods=window).max()
            stochastic = (price-low)/(high-low).replace(0, np.nan)
            for threshold in (0.10, 0.20):
                add(group, "oversold", f"stoch_w{window}_p{int(threshold*100)}",
                    stochastic.le(threshold),
                    f"Stochastic({window}) не выше {threshold:.0%}",
                    window=window, threshold=threshold)

        percentile60 = _rolling_percentile(price, 60)
        for horizon in (3, 5, 10):
            decline = price.pct_change(horizon).lt(0)
            add(group, "confirmation", f"stabilize_h{horizon}",
                percentile60.le(0.2) & decline & ret1.ge(0),
                f"Курс дешёвый после снижения за {horizon} публикаций и стабилизировался",
                horizon=horizon)
        vol5 = ret1.rolling(5, min_periods=5).std()
        vol20 = ret1.rolling(20, min_periods=20).std()
        ratio = vol5 / vol20.replace(0, np.nan)
        for regime, condition in (("calm", ratio.le(0.8)), ("normal", ratio.gt(0.8)&ratio.lt(1.2)),
                                  ("volatile", ratio.ge(1.2))):
            add(group, "regime", f"cheap_{regime}", percentile60.le(0.2)&condition,
                f"Дешёвый курс в режиме волатильности: {regime}", regime=regime)
        return group

    result = _per_currency(frame, price_rules)
    result, relative_specs = _add_relative_rules(result)
    specs.extend(relative_specs)
    return result, specs


def _add_relative_rules(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[RuleSpec]]:
    specs: list[RuleSpec] = []
    result = frame.copy()
    context_columns = [
        col for col in ("usd_ret_1", "eur_ret_1", "cny_ret_1", "residual_usd_1")
        if col in result.columns
    ]
    for column in context_columns:
        relative = result["ret_1"] - result[column] if column != "residual_usd_1" else result[column]
        for window in (60, 120):
            threshold = relative.groupby(result["currency"]).transform(
                lambda x, w=window: x.rolling(w, min_periods=w).quantile(0.1)
            )
            rule_id = f"relative__{column}_w{window}_p10"
            result[rule_id] = relative.le(threshold).astype("boolean")
            specs.append(RuleSpec(rule_id, "relative",
                f"Коридор аномально дешевеет относительно {column} за {window} публикаций",
                {"context": column, "window": window, "quantile": 0.1}))
    cheap = result.groupby("currency")["rub_per_unit"].transform(
        lambda x: _rolling_percentile(x, 60)
    ).le(0.2)
    for column in sorted(col for col in result.columns if col.startswith("ctx_")):
        change = result.groupby("currency")[column].pct_change(fill_method=None)
        for direction, quantile in (("low", 0.1), ("high", 0.9)):
            threshold = change.groupby(result["currency"]).transform(
                lambda x, q=quantile: x.rolling(120, min_periods=120).quantile(q)
            )
            condition = change.le(threshold) if direction == "low" else change.ge(threshold)
            rule_id = f"context__cheap_{column}_{direction}"
            result[rule_id] = (cheap & condition).astype("boolean")
            specs.append(RuleSpec(rule_id, "context",
                f"Дешёвый курс при экстремальном движении {column}: {direction}",
                {"context": column, "direction": direction, "window": 120}))
    return result, specs
