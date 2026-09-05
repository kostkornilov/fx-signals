# Meta-algorithm for push and signal selection

## Summary

The meta-algorithm may use only exchange-rate history, the currently active indicators and the
market model's forecast. It has no information about a particular customer's intent, transfer
probability, transfer amount, conversion response or value to the bank.

This means that the algorithm can optimize **when the market moment is good enough for a push**, but
it cannot estimate which push text will create the most transfers or profit. All signal texts shown
at the same time share the same future exchange-rate path. Choosing one text instead of another does
not change the market outcome.

The recommended policy is therefore:

1. Use the model forecast as the send/no-send gate.
2. Keep only signals that are true, fresh and consistent with the model's forecast.
3. If no signal remains, do not send a push.
4. If one signal remains, select it.
5. If two or more signals remain, select the clearest explanation using a fixed priority.

The result should be described as the **best available explanation of a model-approved market
moment**, not the most profitable push.

## 1. Available information and hard limits

### Available inputs

At decision time `t`, the algorithm may use:

- the exchange-rate series available up to `t`;
- the current corridor and publication time;
- the values and active flags of all approved indicators;
- the market model's score, target, forecast direction and forecast horizon;
- signal freshness and the data needed to render a factual message.

All calculations must use only information that was available at `t`. Future rates may be used for
walk-forward evaluation, but never to make a live decision.

### Unavailable inputs

The algorithm does not know:

- whether a customer currently wants to transfer money;
- the probability that a customer will transfer after a push;
- the customer's expected transfer amount;
- whether one signal text converts better than another;
- fees, margin or incremental bank profit caused by a push;
- customer-level preferences or notification fatigue.

Therefore, a score produced from these inputs is not a conversion score or a profit score. In this
document it is called a **market-opportunity score**.

## 2. Separate timing from explanation

The decision has two different parts.

### Part A — Is the current market moment strong enough?

The predictive model answers this question. Depending on its trained target, its output may be a
probability of a favourable future-rate event, an expected forward rate change, or both.

The push becomes eligible only when:

- the model score passes a threshold fixed before deployment;
- the predicted direction is favourable for the intended transfer direction;
- the rate data and forecast are fresh;
- the threshold passed an out-of-time, walk-forward test;
- at least one approved signal can explain the moment truthfully.

The threshold controls market quality and alert frequency. It cannot be selected from conversion or
profit because those outcomes are not available.

### Part B — Which truthful signal should explain it?

The active indicators describe the same current market moment from different angles. For example,
today may be better than the 30-day average and better than one year ago at the same time.

The model score is normally the same for both texts, so it cannot tell us which text will produce
more transfers. Signal selection must instead optimize properties that can honestly be checked:

- the statement matches the model's direction and horizon;
- the statement is factually true at send time;
- the comparison is easy to understand;
- the text uses a concrete recipient-currency amount when possible;
- the chosen rule is deterministic and auditable.

## 3. Recommended decision algorithm

```text
INPUT:
    rate history available at time t
    model forecast M_t
    set of active signals A_t

1. Validate data freshness and the model version.
2. If M_t does not pass the frozen market threshold, return NO_PUSH.
3. Remove signals that are stale, false, unapproved or inconsistent with M_t.
4. If no signal remains, return NO_PUSH.
5. If one signal remains, select it.
6. If two or more signals remain:
       a. prefer a direct recipient-amount comparison;
       b. prefer a familiar reference period over a technical pattern;
       c. prefer the signal that best matches the forecast direction and horizon;
       d. break any remaining tie with a stable predefined order.
7. Render exactly one signal in the push.
8. Log the model score, threshold, active signals and selected signal.
```

Delivery limits and campaign competition may still exist elsewhere in the product, but they are not
part of this market-only algorithm because they require customer or communication data.

### Suggested clarity priority

For a favourable current moment, use this default order:

1. **Today is better than the average for 30 days.**
2. **Better than one year ago.**
3. **A better range has held.**
4. **A larger-than-usual latest improvement.**
5. **Most recent changes were favourable.**

The first two signals have simple reference points and can be expressed as the additional amount the
recipient gets for the same ruble amount. The other signals require more explanation about a range,
the usual size of a change or a sequence of observations.

For a model-approved deterioration or closing-window message, use:

1. **Most recent changes were unfavourable.**
2. **Less recipient currency than one year ago.**

The short-history signal comes first because it describes the current movement more directly. A
worsening signal must not be used with a favourable forecast if the two statements would create a
contradictory message. If the deployed model has no deterioration target, these two signals should
not independently open the send gate.

## 4. Variants for choosing between several active signals

### Variant A — Fixed clarity priority

Choose the first active signal in a predefined clarity order. The order should favour concrete
recipient amounts and familiar comparisons such as 30 days or one year.

**Advantages:** simple, stable, understandable and easy to audit.

**Limit:** it is a UX rule, not an estimate of conversion or profit.

**Recommendation:** use this as the default policy under the current data constraints.

### Variant B — Largest model contribution

If the predictive model is interpretable in terms of the same indicators, choose the active signal
with the largest positive contribution to the forecast.

**Advantages:** the message explains why the model considered the moment favourable.

**Limits:** correlated features can exchange contribution, attribution can be unstable, and the most
important model feature may produce a difficult message. This method finds the most faithful model
explanation, not the most profitable text.

**Use:** as a tie-breaker or diagnostic, not as the primary policy.

### Variant C — Best historical rate quality

For every signal, use walk-forward history to calculate its hit rate, average forward movement and
adverse tail when that signal was active. At a collision, choose the qualified signal with the best
predefined rate-quality score.

**Advantages:** uses only available market data and rejects indicators that usually describe weak
moments.

**Limits:** signals often activate in different market regimes. At one shared timestamp they all
receive the same future rate, so their historical averages do not show which text is better. This
variant mostly evaluates timing rules already covered by the predictive model.

**Use:** as an offline eligibility test; use cautiously as a selector.

### Variant D — Strongest normalized current evidence

Measure how far each active indicator is beyond its own activation threshold, normalize that value
using the signal's historical distribution, and choose the most exceptional signal.

**Advantages:** avoids directly comparing percentages, streak lengths and range levels that have
different units.

**Limits:** the statistically strongest fact is not necessarily the clearest fact. It also does not
estimate transfer response or profit.

**Use:** only after the clarity rule, as a deterministic tie-breaker between equally clear signals.

### Variant E — Forecast-aligned hybrid

First remove signals that do not match the model's direction or horizon. Then apply the fixed
clarity priority. If two signals have equal priority, use model contribution or normalized current
evidence as the final tie-breaker.

**Advantages:** keeps the text faithful to the forecast without presenting a technical signal when
a clearer explanation is available.

**Limit:** still selects an explanation, not a text with known financial impact.

**Recommendation:** this is the strongest later version if the model exposes reliable direction,
horizon and feature contributions.

## 5. Why “the most profitable signal” cannot be identified

Suppose two signals are active at the same time:

- today gives 3% more recipient currency than the 30-day average;
- today gives 8% more recipient currency than one year ago.

There is one current rate and one model forecast. Both messages refer to that same opportunity. Rate
history can tell us whether the moment later remained favourable, but it cannot tell us whether the
30-day wording or the one-year wording caused more transfers.

The following statements would therefore be unsupported:

- “Signal A has a higher conversion probability.”
- “Signal B creates more bank profit.”
- “This customer is more likely to respond to a short-history signal.”

Those questions require behavioural observations or a controlled message experiment, which are
outside the current input set. Until such data exists, the honest objective is **good market timing
plus the clearest truthful explanation**.

## 6. Offline evaluation with rate data only

Use expanding-window or walk-forward evaluation. At every historical decision point, calculate
features, signals and the model score using only earlier data, then observe the rate over the frozen
future horizon.

Evaluate the send gate with:

- alert frequency;
- hit rate for the model's exact target;
- average forward recipient-currency change;
- median forward change;
- worst-tail forward change;
- stability across time periods and currency corridors;
- signal freshness and the rate of contradictory explanations.

Evaluate the selector only for properties observable without customer data:

- every selected statement was true at decision time;
- the selected signal matched the forecast direction and horizon;
- identical inputs always produced the same choice;
- the policy did not compare incompatible raw signal units;
- one and only one message was selected.

Do not report conversion, incremental transfer volume or profit from this evaluation. These outcomes
cannot be derived from an exchange-rate time series.

## Final recommendation

Use the model only to decide whether the current moment is strong enough to communicate. When one
valid signal is active, select it. When several are active, use the forecast-aligned clarity policy:
remove contradictory signals, prefer a concrete recipient-amount comparison with a familiar period,
and use model contribution or normalized strength only as a tie-breaker.

This design fits the available data and stays honest about what it can predict. It chooses a good
market moment and a clear explanation; it does not claim to choose the push that will generate the
most transfers or money.
