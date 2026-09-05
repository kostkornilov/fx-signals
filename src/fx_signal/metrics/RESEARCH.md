# Metric research: FX transfer trigger

Research date: 5 September 2026.

## 1. Decision being evaluated

The system does not trade and does not forecast a customer's portfolio return. It makes a
selective communication decision: for a corridor and time `t`, send a factual message or stay
silent. The customer then chooses whether to make a real cross-border transfer.

That distinction matters. A useful metric must evaluate the final, thinned stream of messages at
the actual send threshold, not merely rank every day. It must also reflect that the two errors have
different costs:

- a false positive (the bank says that now is a good moment and a materially better rate appears
  soon) can cost the customer money and trust;
- a false negative loses a possible transfer or a chance to help, but does not directly make the
  customer worse off;
- too many individually correct messages can still exhaust the customer's communication budget;
- a historically good average can hide failure in a corridor or market regime.

The case fixes several constraints: `rub_per_unit` is lower-is-better; evaluation horizons are
1/3/5/10/20 fresh publications; the target rate is 1--2 pushes per customer per week across all
communications; the headline target is lift over a random day of at least 1.3; benefit in basis
points must be significantly positive; and the backtest must be walk-forward. See
[`DESCRIPTION.md`](../../../DESCRIPTION.md#целевая-метрика-и-что-считаем-успехом) and
[`CONTEXT.md`](../../../CONTEXT.md#сообщение-в-чате-1-уточнения-по-постановке).

## 2. Repository audit

The current baseline is useful for exploration but not yet a valid estimate of production
quality.

1. [`targets.py`](../targets.py) defines a hit as an exact local minimum in the *symmetric*
   interval `[t-h, t+h]`. The case clarification defines “good now” using the future `h`
   publications: the rate at send time must remain no worse. The symmetric label adds a past-side
   requirement that is not in the customer claim.
2. This mismatch mechanically favours the momentum rule: after three decreases, today's rate is
   already below recent past rates. It also makes the reversal rule fail by construction, because
   reversal fires one publication after the minimum. Reversal needs its own “rate subsequently
   rose” target.
3. [`lift.py`](lift.py) computes a full-history point estimate with no walk-forward tuning,
   uncertainty interval, cooldown, or final cross-corridor send policy.
4. The current `h=3` results (momentum lift about 3.00--3.32 and level lift about 1.44--1.90 across
   five corridors) are therefore diagnostic only. They must not be used as launch evidence.
5. CBR rates are a daily proxy for the bank's executable, intraday quote. That basis and latency
   gap cannot be resolved from the current data.

## 3. Main risks

### 3.1 Truth and trust risk

The notification's implied claim may not hold after it is sent. This is the highest-cost model
error because the bank caused an action and the counterfactual “wait” could have been better.
Raw accuracy is misleading here: most days are “do not send”, and always staying silent could be
highly accurate. Rare-event evaluation literature recommends precision/recall views over ROC
summaries when positives are scarce ([Saito and Rehmsmeier, 2015](https://doi.org/10.1371/journal.pone.0118432)).

### 3.2 Customer financial-harm risk

Binary hits treat a 1 bp miss and a 300 bp miss alike. They should not. Cost-sensitive learning
formalizes the idea that decisions should minimize expected cost rather than symmetric error
([Elkan, 2001](https://cseweb.ucsd.edu/~elkan/rescale.pdf)). Finance also uses expected shortfall
because averages or quantiles alone can hide the severity beyond a tail threshold; the Basel
market-risk framework explicitly relies on expected-shortfall models
([BCBS, 2019](https://www.bis.org/publications/201901-standards-minimum-capital-requirements-market-risk)).
This project is not subject to that trading-book rule; expected shortfall is borrowed only as a
useful tail-risk pattern.

### 3.3 Attention, fatigue, and opportunity-cost risk

FX messages compete with all other bank communications for a 1--2 per-week customer budget.
Repeated signals from one market move may crowd out more valuable messages. Push optimization
research also finds that optimizing only immediate response is myopic: too many irrelevant
notifications can lead users to disable them, while a production experiment achieved fewer sends
and higher open rate without reducing engagement
([O'Brien et al., 2022](https://arxiv.org/abs/2202.08812)). A recent randomized retail experiment
likewise frames frequency as a trade-off between short-term engagement, annoyance, unsubscribe,
and long-term spending
([Baek et al., 2026](https://www.columbia.edu/~wm2428/papers/customer_engagement.pdf)).

### 3.4 Model-selection and regime risk

FX relationships are non-stationary, signals are autocorrelated, and many combinations of
indicator, window, threshold, horizon, and corridor will be tried. Selecting the best full-history
result invites data-snooping. Finance research uses bootstrap reality checks to adjust for choosing
among many technical rules
([Sullivan, Timmermann and White, 1998](https://cepr.org/publications/dp1976)). Rolling-origin
evaluation preserves temporal order and evaluates genuine future observations
([Hyndman and Athanasopoulos, §5.10](https://otexts.com/fpp3/tscv.html)).

### 3.5 Basis, latency, compliance, and causal business risk

- **Basis/latency:** the CBR fixing can say “good” while the in-app executable quote, spread, or
  rate at push-open time does not.
- **Compliance:** an explanation can turn a factual trigger into an implicit promise about the
  future. This is a release gate, not something an offline scalar can make safe.
- **Causal business impact:** transfers after a push may have happened anyway or may only have
  moved forward from next week. Observational post-push conversion is therefore not uplift.
  Direct-marketing uplift methods require treatment/control outcomes and can optimize incremental
  profit rather than response alone
  ([Gubela et al., 2022](https://doi.org/10.1007/s11573-021-01068-3)).

## 4. Candidate metrics considered

### Metrics retained

The recommended scorecard has five primary lines:

1. lift at the production send budget;
2. average moment advantage in basis points;
3. tail customer regret in basis points;
4. useful-alert yield;
5. push frequency and budget use.

These cover truth, average benefit, magnitude of harm, useful output, and attention cost. Stability
is an evaluation requirement applied to all five metrics, not a separate metric. No weighted single
“mega-score” is recommended: arbitrary weights could allow excellent volume to compensate for
customer harm. Use constraints and Pareto comparison instead.

### Metrics rejected as primary

- **Accuracy:** dominated by the many silent days and ignores asymmetric errors.
- **AUROC:** measures ranking over thresholds the product will never use and can look good with a
  large true-negative pool. PR curves are useful during development, but the final decision is at
  a constrained send threshold.
- **F1:** assigns a symmetric trade-off to false pushes and missed opportunities and ignores bp
  magnitude.
- **Raw hit rate:** has no difficulty-adjusted comparison to a random day and can be increased by
  sending almost nothing.
- **Lift alone:** a tiny number of lucky alerts can have high lift; lift also ignores severity.
- **Mean bp benefit alone:** a positive average may coexist with rare, severe customer losses.
- **Sharpe ratio / maximum drawdown:** natural for repeated investment returns, but this product
  causes discrete customer payments, not a self-financing trading strategy.
- **Push-to-transfer conversion:** useful only in a randomized pilot. Offline it is unavailable;
  observationally it confounds intent with treatment effect.

## 5. Formal definitions

Let:

- `p[c,t]` be RUB paid per one unit of recipient currency; lower is better;
- `a[c,t]` be 1 only if the *final policy* sends after ranking, cross-corridor arbitration,
  cooldown, and budget enforcement;
- `h` be the next number of available rate observations, not calendar days;
- `epsilon` be a pre-registered tolerance for rounding/noise (zero for the initial CBR backtest);
- `S` be an out-of-time evaluation slice such as `(corridor, fold, regime)`.

Targets must match the message:

```text
good_now_hit[c,t,h] = 1 when p[c,t] <= min(p[c,t+1 : t+h]) * (1 + epsilon)

rate_rise_hit[c,t,h,delta] = 1 when
    p[c,t+h] / p[c,t] - 1 >= delta
```

`delta` is a pre-registered materiality threshold. Alternative message claims need separately
written labels before testing. Do not score all messages against `target_local_min_h`.

### Metric 1 — Lift at the production send budget

For a fixed message, horizon, slice, and deployed send budget:

```text
precision = sum(a * hit) / sum(a)
random_precision = mean(hit over eligible dates in the same corridor and OOT slice)
lift = precision / random_precision
```

For the closest operational comparator, repeatedly sample random dates with the same number of
sends and the same cooldown/calendar eligibility. The analytical mean above should agree with the
Monte Carlo baseline; the simulation supplies an uncertainty distribution.

Report alert count, precision, random precision, lift, and a one-sided 95% lower confidence bound
using a moving-block bootstrap over time. Selection target: point lift at least 1.3 and lower bound
above 1.0 in every corridor actually proposed for launch. The case headline remains lift >= 1.3
across several corridors and OOT folds.

Why it fits: it directly asks whether a sent claim is more truthful than spending the same scarce
slot on a random eligible day.

### Metric 2 — Average moment advantage

Keep the case-required average-window value:

```text
moment_advantage_bp[c,t,h] =
    10,000 * (mean(p[c,t-h : t+h]) / p[c,t] - 1)
```

Positive means the customer receives more foreign currency per RUB than at the surrounding
average. Report its mean and block-bootstrap 95% confidence interval over sent alerts; the lower
bound must be above zero.

Why it fits: lift measures how often a signal is right, while moment advantage measures the size of
the average customer benefit.

### Metric 3 — Tail customer regret

Use a forward-looking customer regret measure:

```text
regret_bp[c,t,h] =
    10,000 * max(p[c,t] / min(p[c,t+1 : t+h]) - 1, 0)
```

This estimates how much more RUB per unit the customer paid by acting at `t` rather than at the
best later observation in the decision window. Report mean regret and `ES95(regret)`, the average
regret among the worst 5% of sent alerts. If fewer than 100 alerts exist, report the maximum and
the mean of the worst five alerts as descriptive statistics; do not claim a stable ES95 estimate.

Why it fits: regret and ES95 expose the asymmetric downside that a binary hit or average benefit can
hide. A production threshold for ES95 needs the bank's loss/trust appetite and transfer-check
distribution; it should not be invented from public CBR data.

### Metric 4 — Useful-alert yield

Evaluate the final stream, not each indicator independently:

```text
useful_alerts_per_100_client_weeks =
    100 * sum(a * hit) / eligible_client_weeks
```

Maximize this only subject to the other quality gates and the Metric 5 frequency limit. An alert is
useful only if the final policy sent it and its message-specific truth condition was satisfied.

Why it fits: lift can be maximized by sending one message a year. Useful yield measures how much
verified customer help the system delivers, while Metric 5 separately measures the attention it
consumes.

### Metric 5 — Push frequency and budget use

Evaluate every sent push, whether correct or incorrect:

```text
pushes_per_client_week = sum(a) / eligible_client_weeks
over_budget_week_rate = weeks_above_limit / eligible_client_weeks
```

Report mean, median, and p90 pushes per client-week, zero-push weeks, `over_budget_week_rate`, and
`cluster_rate[d]`, the share of sends whose prior FX send was within `d` days. Apply these rules:

- total customer load (FX plus reserved/known other messages) no more than 2 per week;
- a pre-registered FX cooldown;
- `over_budget_week_rate = 0` after orchestration;
- the Metric 1 truth gate, Metric 2 value gate, and Metric 3 tail-risk gate.

Because client subscription data are absent, run scenarios for one and multiple subscribed
corridors and state the assumed budget left for FX. Do not add per-corridor rates: the organizer
clarified that the cap is per customer in total.

Why it fits: two incorrect pushes create zero useful alerts but still consume two communication
slots. Frequency therefore cannot be inferred from useful yield. Load and cluster rate diagnose
fatigue and repeated alerts from one movement.

### Stability requirement — Worst out-of-time slice

For every proposed launch corridor, produce predictions only in forward test folds. Recalculate all
five metrics in each slice. At minimum, compute the one-sided 95% lower bound of lift and the lower
bound of mean moment advantage:

```text
robust_lift_floor = min over launch slices LCB95(lift)
value_pass_rate = share of launch slices with LCB95(mean moment_advantage_bp) > 0
```

Also report the median and full slice distribution so the minimum is not interpreted without its
sample size. A corridor that fails is excluded or given a separate recalibration; strong corridors
must not average away a weak one.

Why it fits: it penalizes a rule that works only in one currency or one historical regime. It also
makes the meaning of “stable” auditable rather than relying on a pooled average.

## 6. Evaluation protocol

1. Freeze message-specific labels, candidate indicator families, horizons, tolerance, cooldown,
   and budget before looking at the final OOT period.
2. Use expanding or rolling walk-forward folds. Fit parameters and probability calibration only
   on data available before each test fold.
3. Purge at least `h` observations between fitting/calibration and evaluation wherever overlapping
   forward labels could leak into the next fold.
4. Within each training fold, choose the threshold that maximizes useful-alert yield while meeting
   truth, value, tail, and load constraints. Never tune the threshold on its test fold.
5. Merge indicators into the final cross-corridor stream, deduplicate the same market episode,
   apply cooldown, then compute all five metrics.
6. Use moving-block bootstrap intervals because nearby FX observations and alerts are dependent.
   Report the block rule, number of independent alert episodes, and raw send count.
7. Keep a final untouched OOT period. If many rule variants are compared, either correct for
   multiple testing / use a reality-check-style bootstrap or treat the final OOT result as the only
   confirmatory estimate.
8. Report separately by message, horizon, corridor, fold, and a pre-declared stress period; pooled
   totals are supplementary.

## 7. Offline-to-online boundary

The five metrics above decide whether the signal layer deserves a pilot. They cannot prove that a
push causes additional bank volume.

The pilot should randomize eligible customer-event opportunities into send and holdout, with a
stable assignment or appropriate clustered design to measure longer-term effects. The primary
business metric should be incremental net transfer volume per eligible customer over a window long
enough to include delayed transfers, not conversion per sent push. Secondary metrics are
incremental contribution margin, transfer timing/cannibalization at later windows, opt-out or loss
of push reachability, complaints, and the harmful-stale-open rate based on actual in-app quotes.
Qini/profit-uplift curves become relevant only after randomized treatment/control data exist.

## 8. Open inputs required from the bank

- executable quote and spread at send, open, and transfer time;
- distribution of transfer amount/check and acceptable bp/customer-RUB tail loss;
- delivery/open-delay distribution and dropped-notification data;
- customer corridor subscriptions and the communication budget left after other campaigns;
- definition of a new versus merely shifted transfer and the cannibalization window;
- compliance-approved message claims and tolerances;
- event volume needed for a powered clustered randomized pilot.

Until these exist, CBR-based benefit and regret are proxy metrics and must be labelled as such.
