# Main metric: Lift-at-Risk

## Decision

Use **Lift-at-Risk** to rank models that already satisfy the basic safety and frequency rules.

Short names:

- `LAR-F`: fixed-window Lift-at-Risk;
- `LAR-D`: time-discounted Lift-at-Risk.

For the current dataset, use `LAR-F` as the main ranking metric because we do not have customer
transaction histories. Show `LAR-D` as a monthly/two-month sensitivity analysis. Once the bank
provides customer-level transaction intervals, `LAR-D` can become the main version.

Lift-at-Risk does not replace the underlying metrics. A model is not launchable if it breaks the
notification limit or an absolute customer-harm limit, regardless of its combined score.

## The final formula

$$
\Delta U_{\mathrm{bp}}
= \left(V_{\mathrm{signal}} - V_{\mathrm{random}}\right)
- \rho\left(R_{\mathrm{signal}} - R_{\mathrm{random}}\right)
$$

$$
\operatorname{LiftAtRisk}
= \operatorname{Lift}\exp\left(\frac{\Delta U_{\mathrm{bp}}}{B}\right)
$$

where $V$ is mean customer value in bp and $R$ is CVaR95 customer regret in bp.

Interpretation:

- `lift` measures how much more often our message is correct than a random message;
- value rewards economically better moments;
- tail regret penalizes the worst harmful messages;
- `rho` says how strongly we care about extra tail regret;
- `B` puts the bp adjustment on a comparable scale with lift;
- `LAR = 1` is the neutral random-policy level;
- higher is better.

The case's `1.3` target applies to the original lift, not automatically to LAR. `LAR > 1` means
better than matched random under the declared `rho`, `B`, and discount assumptions. A launch
threshold for LAR must be set separately.

The metric is relative to a matched random policy. This is important: if the whole market rises,
almost every day may look good. We only want credit for value or lower regret beyond what the same
number of random messages would have achieved.

## Step 1: calculate the authors' lift

For a fixed corridor, message, horizon, and out-of-time fold:

```text
hit_rate = correct sent messages / sent messages
random_hit_rate = correct eligible days / eligible days
lift = hit_rate / random_hit_rate
```

The random comparator should be simulated with the same number of pushes, eligible dates,
cooldown, and weekly budget. Lift keeps exactly the meaning required by the case authors.

## Step 2: calculate value and regret along the future path

Let `p[t]` be RUB paid for one unit of recipient currency. Lower is better. For each of the next
`h` rate observations:

```text
advantage_bp[t,j] = 10,000 * (p[t+j] / p[t] - 1)

loss_bp[t,j] = 10,000 * max(p[t] / p[t+j] - 1, 0)
```

Positive advantage means sending at `t` was better than waiting. Loss is positive only when a
better rate appeared later.

Given day-relevance factors `d[j]`, normalize the value weights:

```text
w[j] = d[j] / sum(d[1:h])
```

For each sent message:

```text
path_value_bp[t] = sum(w[j] * advantage_bp[t,j])

path_regret_bp[t] = max(d[j] * loss_bp[t,j])
```

Then aggregate all sent messages:

```text
signal_value_bp = mean(path_value_bp for sent messages)

signal_tail_regret_bp = CVaR95(path_regret_bp for sent messages)
```

CVaR95 is the average regret among the worst 5% of sent messages. With fewer than 100 messages,
also show maximum regret and the mean of the worst five; CVaR95 is then only descriptive.

Calculate `random_value_bp` and `random_tail_regret_bp` using frequency-matched random push streams.
With `M` random streams, use the mean of their `M` value estimates and the mean of their `M`
separately computed CVaR estimates. Do not pool all simulated messages before computing CVaR.

This forward path value is part of LAR. Continue to report the authors' separate moment-value
metric against the surrounding `±h` average as a mandatory check.

## Variant 1: fixed window (`LAR-F`)

Use equal relevance for every next observation:

```text
d[j] = 1
w[j] = 1/h
```

This means:

- value is the average advantage against each of the next `h` observations;
- regret is the worst missed price opportunity in those `h` observations;
- no customer behavior data or discount assumption is needed.

Use `LAR-F` for the current hackathon backtest. Its weakness is that a rate 20 days later counts as
just as relevant as tomorrow's rate.

## Variant 2: discounted window (`LAR-D`)

Let `S(days)` be the probability that a customer has not yet made the next transfer after the push.
Let `Delta[j]` be the actual number of calendar days between the push and future observation `j`:

```text
d[j] = S(Delta[j]) / S(Delta[1])
w[j] = d[j] / sum(d[1:h])
```

The first observable alternative has relevance 1. Each later rate matters less because there is a
growing chance that the customer's transfer decision has already happened.

### Best discount: bank transaction histories

Estimate `S(days)` from time-to-next-transfer data for eligible customers. Use pre-push or holdout
data, account for customers whose next transfer is not observed, and estimate separate curves for
large corridors or behavioral cohorts. Salary dates and recurring family payments will probably
make the real curve different from a smooth exponential decay.

### Fallback discount: global transaction frequency

If the only available number is transaction frequency, assume a constant transaction hazard:

```text
lambda = transactions / active eligible customer-days
mean_gap_days = 1 / lambda

d[j] = exp(-(Delta[j] - Delta[1]) / mean_gap_days)
```

Use active eligible customers in the denominator. Total cross-border volume divided by the whole
population is not a valid customer transaction rate.

An illustrative monthly customer has `mean_gap_days = 30`:

- daily discount factor: about `0.967`;
- discount half-life: about `20.8` days;
- day 20 relevance relative to day 1: about `0.53`.

For a two-month customer, day 20 relevance is about `0.73`. A
[World Bank survey report](https://documents1.worldbank.org/curated/en/635911468252622201/pdf/531060NWP0Remi10Box345597B01PUBLIC1.pdf)
found monthly and two-month remittance patterns in Kyrgyz and Tajik migrant samples, but the sample
is old and explicitly non-representative. Use `30` and `60` days as scenarios, not claimed customer
facts.

These scenarios assume no particular number of customers. They describe a per-eligible-customer
waiting-time curve; estimating total customer impact would additionally require the eligible
customer count and transfer amounts.

The discount must use calendar days, even though `h` counts fresh rate publications. Customers can
transfer during weekends and holidays when there is no new CBR observation.

## Choosing `rho`

`rho` is the regret-aversion weight:

```text
rho = 1: one extra bp of tail regret costs one bp of average value
rho = 2: one extra bp of tail regret requires two bp of average value
rho = 3: strongly conservative
```

The case says a bad signal is worse than silence, so do not silently use `rho < 1`. Until the bank
sets a risk appetite, publish results for `rho = 1, 2, 3` and require the model ranking to remain
reasonably stable.

## Choosing `B`

`B` controls how many bp have the same importance as a lift change. Choose a material amount `U*`:

> “`U*` bp of risk-adjusted customer value is as important as improving lift from 1.0 to 1.3.”

Then set:

```text
B = U* / log(1.3)
```

For example, if `U* = 100 bp`, then `B` is about `381 bp`. A `+100 bp` incremental utility changes
LAR by a factor of `1.3`; a `-100 bp` change divides it by `1.3`.

The bank should set `U*` using transfer checks and an acceptable RUB benefit/harm. Do not tune `B`
or `rho` on the final out-of-time period.

## Worked example

Suppose one policy has:

```text
lift = 1.40
signal value = 35 bp       random value = 5 bp
signal tail regret = 90 bp random tail regret = 60 bp
rho = 2
U* = 100 bp, so B = 381 bp
```

Then:

```text
DeltaU = (35 - 5) - 2 * (90 - 60) = -30 bp
LAR = 1.40 * exp(-30 / 381) = 1.29
```

The model's truth lift is good, but its extra tail regret reduces the combined score.

## How to use the metric

1. Generate predictions with the walk-forward backtest.
2. Merge duplicate triggers and apply the real cooldown and push budget.
3. Build many matched random push streams for the same corridor and fold.
4. Calculate lift, signal/random value, and signal/random CVaR regret.
5. Calculate `LAR-F`. Calculate `LAR-D` for frozen discount scenarios.
6. Recalculate the complete formula in a block bootstrap to get a confidence interval.
7. Compare models by the **lowest** LAR across proposed corridors and test folds, then use median LAR
   as a secondary comparison.

Always show these components beside LAR:

- number of messages;
- hit rate, random hit rate, and lift;
- signal and random value;
- signal and random tail regret;
- `rho`, `B`, horizon, and discount source;
- push frequency and over-budget rate;
- confidence interval.

## Hard gates that LAR cannot override

A model remains unacceptable if any of these fails:

- leakage-free walk-forward evaluation;
- the case's lift requirement;
- significantly positive surrounding-window moment value;
- the bank's absolute customer-regret limit;
- required message frequency and zero post-policy budget violations;
- enough independent messages to estimate the score;
- acceptable results in every proposed launch corridor.

Therefore LAR is “one metric to rank them all,” not “one number that can excuse every failure.”

## Recommendation

Use the following decision rule for this project:

```text
First: reject every policy that fails a hard gate.
Then: choose the policy with the highest worst-fold LAR-F.
Also: require the choice to remain sensible under LAR-D with 30- and 60-day scenarios.
Later: replace those scenarios with the bank's empirical next-transfer survival curve.
```

The derivation, alternatives, and sources are documented in [`RESEARCH.md`](RESEARCH.md).
