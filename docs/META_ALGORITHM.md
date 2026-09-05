# Meta-algorithm for push and indicator selection

## Summary

The system has one predictor of whether the current exchange-rate moment is worth communicating.
The meta-algorithm does not build another forecast and does not divide predictions into artificial
types such as `local_min` or `window_closing`.

The predictor decides **whether to send**. The active indicators decide **which fact to show**:

1. If the predictor score is below its fixed threshold, do not send a push.
2. If no indicator is active, do not send a push.
3. If one indicator is active, use it.
4. If several indicators are active, compare their factual effects on one common scale and select
   the largest effect after a clarity adjustment.

This algorithm does not estimate which text will create the most transfers. There is no customer
behaviour or conversion data. It selects the strongest clear explanation of a moment already
approved by the exchange-rate predictor.

## 1. Inputs

The meta-algorithm uses only:

- one market-model score;
- the frozen send threshold for that model;
- the seven indicator flags;
- the signed effect calculated by every indicator;
- a fixed clarity coefficient for every indicator;
- a market-data freshness flag.

It does not use customer intent, transfer probability, transfer amount, conversion, fees or bank
profit.

## 2. Common effect scale

Let:

- `q_t` be RUB per one unit of recipient currency;
- `u_t = 1 / q_t` be recipient-currency units per RUB;
- higher `u_t` be better for the recipient.

Every indicator returns an `effect` as a signed decimal change in `u` relative to its own factual
benchmark:

```text
effect = current_or_new_value / benchmark_value - 1
```

For example, `effect = 0.02` means 2% more recipient currency, while `effect = -0.02` means 2% less.
Using a dimensionless percentage allows annual, monthly and recent indicators to be compared even
though their raw calculations use different units and windows.

The seven effects are:

1. **Better than one year ago:** `u_t / u_year - 1`.
2. **A better range has held:** `minimum(u_t-2 ... u_t) / maximum(u_t-9 ... u_t-3) - 1`.
3. **A larger-than-usual latest improvement:** `u_t / u_t-1 - 1`.
4. **Today is better than the average for 30 days:** `u_t / average_30d(u) - 1`.
5. **Most recent changes were favourable:** `u_t / u_t-5 - 1`.
6. **Less recipient currency than one year ago:** `u_t / u_year - 1`, which is negative when the
   signal is active.
7. **Most recent changes were unfavourable:** `u_t / u_t-5 - 1`, which is negative when the signal
   is active.

The sign must be preserved in the rendered text. A negative effect must never be presented as a
benefit. Selection uses the absolute magnitude because both a strong improvement and a strong
deterioration can be important facts.

For user copy, the percentage may be converted into a concrete recipient amount for a fixed example
such as 10,000 RUB. The amount must be calculated from the same benchmark used by the indicator.

## 3. Clarity coefficients

Large but difficult facts should not automatically replace a slightly smaller fact that is much
easier to understand. The selection score therefore discounts each effect using a fixed clarity
coefficient:

```text
selection_score = abs(effect) * clarity_coefficient
```

The initial coefficients are:

- today versus the 30-day average: `1.00`;
- today versus one year ago, better or worse: `0.95`;
- a better range has held: `0.80`;
- a larger-than-usual latest improvement: `0.70`;
- recent favourable or unfavourable sequence: `0.65`.

These values are product rules, not learned conversion estimates. They should be changed only after
a comprehension test or an explicit product decision. Each coefficient must remain in `(0, 1]`.

## 4. Decision algorithm

```text
INPUT:
    market_score
    market_threshold
    fresh market-data flag
    active indicators with signed effects

1. If market data is stale, return NO_PUSH.
2. If market_score is missing or invalid, return NO_PUSH.
3. If market_score < market_threshold, return NO_PUSH.
4. Find all active and approved indicators.
5. If none are active, return NO_PUSH.
6. If exactly one is active, select it.
7. If two or more are active:
       a. require a finite effect for every active indicator;
       b. calculate abs(effect) * clarity_coefficient;
       c. select the indicator with the highest score;
       d. break an exact tie by clarity, then effect magnitude, then stable indicator order.
8. Render exactly one push and log the full decision.
```

An active indicator without an effect is an upstream data error. With several active indicators the
algorithm stays silent because it cannot compare them honestly. The single-indicator rule remains
simple: if only one approved fact is active, it is selected without needing a comparison.

## 5. Example

Assume the market-model score passes its threshold and two indicators are active:

```text
Today versus the 30-day average:
    effect  = 0.020
    clarity = 1.00
    score   = 0.0200

Today versus one year ago:
    effect  = 0.050
    clarity = 0.95
    score   = 0.0475
```

The annual indicator wins because its factual effect is much larger.

If the annual effect is only `0.0104`, its adjusted score is `0.00988`. A 30-day effect of `0.0100`
then wins because the simpler comparison is almost as strong before the clarity adjustment.

## 6. Output and audit fields

For every decision, store:

- whether a push should be sent;
- the selected indicator;
- the decision reason;
- model score and threshold;
- all active indicators;
- each active indicator's signed effect, clarity coefficient and selection score;
- the selected signed effect and final selection score.

The model score is used only for the send gate. It must not be multiplied into every indicator score:
at one timestamp every active indicator shares the same model score, so multiplication would not
change their order.

## 7. Limitations

This algorithm ranks the strength and clarity of factual exchange-rate explanations. It cannot know
which wording has the highest conversion or produces the most money transfers. Such a claim would
require customer-response data that the project does not have.

The clarity coefficients are starting assumptions. The exchange-rate predictor and its threshold
still require leakage-safe walk-forward validation. Indicator effects must also be calculated only
from information available at the decision time.

## Final recommendation

Use one predictor as the send/no-send gate. Do not pass a separate free-form `forecast_kind` into the
meta-algorithm. When several indicators are active, select the one with the largest
`abs(effect) * clarity_coefficient`, preserve the effect direction in the message, and keep the full
calculation in the decision log.
