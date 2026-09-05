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
is an evaluation requirement applied to all five metrics, not a separate metric. A weighted score
can help rank models that pass the five hard gates, but it must not let high lift compensate for a
frequency-policy violation or unacceptable customer harm. Section 9 develops such a secondary
ranking score, Lift-at-Risk.

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

## 9. Research for one integrated metric: Lift-at-Risk

### 9.1 What the integrated metric should and should not do

The case explicitly permits an integral metric for comparing fast and slow indicators, but keeps
lift, positive bp benefit, frequency, clustering, stability, and absence of leakage as separate
conditions. The organizer also clarified that lift is based on hit rate while bp benefit is a
separate mandatory condition. Therefore one scalar is useful for **ranking feasible policies**, not
for making unsafe policies look acceptable.

The desired scalar should have these properties:

- preserve the authors' definition of lift and its random-day comparator;
- reward larger customer benefit, not merely binary hits;
- penalize severe regret more than ordinary variation;
- have a neutral random-policy value of 1;
- support both a transparent fixed window and a customer-behavior-weighted window;
- be comparable only when horizon, message, corridor, randomization rules, and policy parameters
  are held fixed;
- leave notification frequency, minimum sample size, and absolute harm limits as hard gates.

“Lift-at-Risk” (`LAR`) is a project-specific name. The exact composite below is a proposal, not an
established financial statistic.

### 9.2 Similar approaches

**Mean--CVaR optimization in finance.** Portfolio optimization commonly balances expected return
against CVaR/expected shortfall. Rockafellar and Uryasev developed CVaR as an optimizable tail-loss
measure, and their work on general loss distributions covers empirical/discrete samples
([Rockafellar and Uryasev, 2000](https://doi.org/10.21314/jor.2000.038);
[Rockafellar and Uryasev, 2002](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=267256)).
The analogy here is useful but limited: an FX push is not a self-financing trade. We borrow the
expected-benefit minus tail-risk structure, not portfolio-return claims.

**Decision-curve net benefit.** Decision-curve analysis converts false-positive harm into the same
units as true-positive benefit and evaluates a policy at an action threshold. Its main lesson is
that an integrated metric needs an explicit, externally justified exchange rate between benefit
and harm; that exchange rate cannot be discovered from classification accuracy alone
([Vickers and Elkin, 2006](https://pmc.ncbi.nlm.nih.gov/articles/PMC2577036/)). This supports an
explicit regret-aversion parameter rather than a hidden arbitrary weight.

**Discounted gain in information retrieval.** Discounted cumulative gain gives less credit to
relevant results reached later in a ranked list
([Järvelin and Kekäläinen, 2002](https://doi.org/10.1145/582415.582418)). Time after a push is
similar to rank in one narrow respect: later outcomes can be less relevant to a customer decision.
However, the common logarithmic rank discount has no behavioral meaning for transfers, so it should
not be copied directly.

**Long-horizon push optimization.** Push research shows why an immediate-response objective is
incomplete: repeated or irrelevant notifications can damage future behavior. A model-based system
tested at Twitter sent fewer notifications with a higher open rate while maintaining engagement
([O'Brien et al., 2022](https://arxiv.org/abs/2202.08812)). This supports using the final,
budgeted push policy in evaluation, but it does not supply a universal FX discount rate.

**Survival/hazard weighting.** If transactions arrive with a constant rate `lambda`, the waiting
time to the next transaction has survival probability `S(t) = exp(-lambda*t)` and mean
`1/lambda`. This is the standard exponential inter-arrival model
([NIST Engineering Statistics Handbook](https://www.itl.nist.gov/div898/handbook/apr/section1/apr161.htm)).
It supplies a defensible first discount curve from transaction frequency. The constant-hazard
assumption is only a fallback: salary dates, recipient requests, seasonality, and customer
heterogeneity make real remittances non-memoryless.

Public remittance evidence supports using monthly and two-month scenarios, but not one universal
number. A World Bank survey of returned migrants reported 38.7% remitting monthly and 36.6% every
two months across six countries; the corresponding monthly figures were 19.7% for the Kyrgyz
Republic and 24.3% for Tajikistan. The report explicitly warns that its network/snowball sample is
not representative
([World Bank survey report, pp. 12--15](https://documents1.worldbank.org/curated/en/635911468252622201/pdf/531060NWP0Remi10Box345597B01PUBLIC1.pdf)).
These old survey values are reasonable sensitivity scenarios only. They are not estimates for the
bank's app users.

### 9.3 Candidate composite forms

Several simple forms were considered.

1. `lift * value / regret` is easy to explain but unstable when value or regret is near zero. It
   becomes infinite for a small sample with no observed regret.
2. `lift + alpha*value - beta*regret` mixes a ratio with basis points and changes if value is
   expressed in percent instead of bp.
3. Replacing the binary hit with a weighted fraction of favorable future days is smooth, but it is
   no longer the lift defined by the case authors.
4. `value - rho*CVaR(regret)` follows mean--CVaR logic, but discards the required truth comparison
   with a random day.
5. A log-additive score preserves lift and adds bp utility after explicit normalization:

```text
log(LAR) = log(lift) + incremental_risk_adjusted_value_bp / B
```

This fifth form is retained. Exponentiating keeps the published score positive and centered at 1.
It is monotone in lift and value, monotone decreasing in regret, and has no division by a noisy
near-zero regret estimate.

### 9.4 Common path quantities

For a signal at `t`, let `p[t]` be RUB per unit of recipient currency and let `p[t+j]` be the rate at
the `j`th later observation. Lower is better. For every `j = 1...h`, define:

```text
advantage_bp[t,j] = 10,000 * (p[t+j] / p[t] - 1)

loss_bp[t,j] = 10,000 * max(p[t] / p[t+j] - 1, 0)
```

`advantage_bp` is signed: positive means sending at `t` was better than waiting until `t+j`.
`loss_bp` is non-negative regret. The two formulas use different denominators deliberately and are
consistent with the existing value and regret definitions.

For non-negative relevance factors `d[j]`, define normalized value weights
`w[j] = d[j] / sum(d[1:h])`. A single signal then has:

```text
path_value_bp[t]  = sum(w[j] * advantage_bp[t,j])
path_regret_bp[t] = max(d[j] * loss_bp[t,j])
```

Value represents the typical waiting comparison. Regret deliberately keeps the worst relevant
later opportunity; averaging it could hide one large harmful miss. Across all sent messages:

```text
V = mean(path_value_bp)
R = CVaR95(path_regret_bp)
```

When fewer than 100 signals exist, `R` is too weakly estimated for a strong 5%-tail claim. Report
the maximum and worst-five mean alongside it.

### 9.5 Compare against a matched random policy

Absolute value is not enough. A rising or falling market can make all dates look favorable or
unfavorable. Generate random policies with the same corridor, fold, number of sends, eligible
calendar, cooldown, and weekly budget as the tested policy. Calculate the same quantities for the
signal (`V_s`, `R_s`) and random comparator (`V_0`, `R_0`):

For `M` random streams, define `V_0` as the mean of their `M` mean path values and `R_0` as the
mean of their `M` separately calculated CVaR values. Likewise, random hit rate is the mean hit rate
across streams. Do not pool all random messages before calculating CVaR: pooling answers a
different question and understates variation between feasible random policies.

```text
DeltaU_bp = (V_s - V_0) - rho * (R_s - R_0)
```

`rho >= 0` is regret aversion: how many bp of mean value the bank requires to accept one additional
bp of tail regret. Because the case says a bad message is worse than silence, `rho` should not be
silently set below 1. Use a pre-declared sensitivity range until the bank sets it.

The proposed score is:

```text
Lift_at_Risk = lift * exp(DeltaU_bp / B)
```

`B > 0` is the bp scale that determines how strongly economics adjusts lift. A matched random
policy has `lift = 1` and `DeltaU_bp = 0`, hence `LAR = 1`. A policy with better tail risk than
random receives a positive contribution because `R_s - R_0` is negative.

This formula does intentionally count both truth and magnitude. Lift answers how often the stated
claim holds; `DeltaU` answers how large the customer consequence is. Correlation between them is
expected, so uncertainty must be calculated for the complete score rather than by treating the
components as independent.

### 9.6 Variant A: fixed-window Lift-at-Risk (`LAR-F`)

Set:

```text
d[j] = 1
w[j] = 1/h
```

All next `h` observations matter equally. The per-message regret becomes the existing worst price
miss in the window. This variant is transparent, needs no customer behavior data, and should be the
default for the current hackathon backtest.

Its limitation is behavioral: a rate 20 days later is treated as equally relevant as tomorrow's
rate even if most customers would already have transferred.

### 9.7 Variant B: time-discounted Lift-at-Risk (`LAR-D`)

Let `Delta[j]` be elapsed **calendar time** from the push to later observation `j`, and let `S(t)` be
the probability that an eligible customer has not yet made the next transfer by time `t`. Set:

```text
d[j] = S(Delta[j]) / S(Delta[1])
w[j] = d[j] / sum(d[1:h])
```

The first observable alternative receives weight 1; later alternatives receive less weight.
Normalizing only the value weights keeps value in bp. Regret uses the unnormalized `d[j]`, so a
large miss tomorrow is not diluted merely because `h` is large.

Preferred production estimate: build an empirical survival curve from pre-treatment or holdout
customer data, starting at a clearly defined eligible moment, with right censoring. Estimate by
corridor and meaningful cohorts, shrinking sparse cohorts toward a global curve. This accounts for
payday and calendar effects without forcing a constant hazard.

Fallback when only a global transaction frequency is available:

```text
lambda = transactions / active eligible customer-days
T = 1 / lambda                         # mean inter-transaction time
d[j] = exp(-(Delta[j] - Delta[1]) / T)
```

The denominator must be active eligible customer-days, not population size or total calendar days.
Aggregate corridor volume alone cannot recover a per-customer waiting-time distribution.

For an illustrative monthly customer (`T = 30` days), the daily discount factor is about `0.967`
and the half-life is about `20.8` days. Relative to day 1, day 20 has weight about `0.53`. For a
two-month customer (`T = 60`), day 20 has weight about `0.73`. Thus a frequency-based discount makes
little difference at `h = 3/5` but can materially change `h = 20`.

Use calendar time for this discount even though the rate horizon is expressed in fresh
publications. Customers can transfer while no new CBR observation is published. With current proxy
data, use actual elapsed days between publications; with bank data, use executable quote time.

`LAR-D` reduces exactly to `LAR-F` when `lambda = 0`, and both variants are identical at `h = 1`.

### 9.8 Calibrating the two policy weights

Neither `rho` nor `B` should be tuned to maximize the final OOT result.

- Set `rho` from customer-risk policy or show at least `rho = 1, 2, 3`. `rho = 2` means one extra bp
  of tail regret requires two bp of incremental mean value.
- Set `B` using a stated materiality trade-off. If the bank says `U*` bp of incremental
  risk-adjusted value is as important as improving lift from `1.0` to `1.3`, use:

```text
B = U* / log(1.3)
```

For example, if `U* = 100 bp`, then `B` is about `381 bp`; a `+100 bp` `DeltaU` multiplies LAR by
`1.3`, and `-100 bp` divides it by `1.3`. The same choice can be made in RUB using a reference
transfer amount, but the amount and acceptable customer loss must come from the bank.

### 9.9 How to use and validate LAR

1. Apply the final deduplication, cooldown, and notification budget before scoring.
2. Require the original hard gates: authors' lift, statistically positive symmetric-window value,
   absolute regret cap, 1--2 signals per corridor-week, sample size, and walk-forward stability.
3. Among policies that pass, maximize the worst-corridor/worst-fold `LAR-F`; report median as
   secondary. A pooled mean must not hide a failing slice.
4. Report `LAR-D` as sensitivity analysis until a customer waiting-time curve is available. Then
   promote it only if rankings are stable across cohorts and discount estimates.
5. Use the same matched random draws for lift, value, and regret so the comparator is internally
   consistent.
6. Obtain confidence intervals by block-bootstrap resampling whole market episodes or weeks and
   recomputing lift, `V`, `R`, the random comparator, and LAR together.
7. Publish the components next to the scalar: signal count, hit rate, random hit rate, lift, `V_s`,
   `V_0`, `R_s`, `R_0`, `rho`, `B`, discount source, and LAR.

The main failure mode is false precision. `LAR = 1.27` is not meaningful unless the policy weights,
random comparator, sample size, and uncertainty interval are visible. The scalar makes trade-offs
consistent; it does not make the trade-offs objective.
