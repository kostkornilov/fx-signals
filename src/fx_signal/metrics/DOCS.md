# Metrics for the FX transfer signal

## What we need to measure

The model decides whether the bank should tell a customer that the current exchange rate is
interesting. A wrong message is more harmful than a missed message: the customer may transfer
money, see a better rate soon afterwards, and stop trusting the bank's notifications.

Four offline metrics:

1. **Lift:** how often our message is correct compared with a random day.
2. **Customer value and regret:** how much money the customer could gain or lose.
3. **Useful signals per week:** whether we produce enough good messages without sending too many.
4. **Stability:** whether the results remain good in new time periods and different currencies.

These metrics should be shown together. For example, a model with high lift is still unsafe if its
rare mistakes are very expensive for customers.

## Terms used below

`p[c,t]` is the number of rubles needed to buy one unit of the recipient's currency at time `t`.
A lower value is better for the sender.

`h` is the evaluation horizon. In the current daily CBR dataset, `h = 3` means the next three
available rate observations. These are usually the next three business or publication days.
Weekends and holidays without a new CBR observation are skipped. It does not mean exactly three
calendar days or 72 hours.

A **basis point**, abbreviated `bp`, is `0.01%`:

```text
1 bp   = 0.01%
100 bp = 1%
300 bp = 3%
```

## Define what a correct message means

Different messages make different claims, so they need different tests.

### “Now is a good time”

This message is correct if none of the next `h` rates is meaningfully better than the rate when the
message was sent:

```text
good_now_hit = 1 if the current rate is no worse than every rate in the next h observations
```

For example, if today's rate is 0.180 RUB/KZT and all three following observations are 0.180 or
higher, `good_now_hit = 1` for `h = 3`.

### “The favorable window is closing”

This message is correct if the recipient's currency becomes meaningfully more expensive after the
message:

```text
rate_rise_hit = 1 if the rate after h observations is higher than the current rate by at least delta
```

`delta` is the smallest increase that the team considers meaningful. It must be chosen before the
final test.

For example, if the rate rises from 0.180 to 0.183 RUB/KZT, the recipient's currency has become
about 1.67% more expensive. This confirms a “window is closing” message if `delta` is below 1.67%.

### Problem with the current target

The current [`targets.py`](../targets.py) checks whether today is the exact lowest rate in a window
that includes both past and future observations. This is not the final definition from the case.

It gives the momentum rule an unfair advantage because momentum already requires the current rate
to be below several past rates. It also gives the reversal rule zero hits because reversal happens
after the lowest point.

Therefore, the current lift results are preliminary. Each message must first receive the correct
forward-looking test described above.

## Metric 1: lift

Lift is the main metric required by the case authors:

```text
signal hit rate = correct sent messages / all sent messages
random hit rate = correct days / all eligible days
lift = signal hit rate / random hit rate
```

Example:

```text
Our signal is correct on 30% of sent days.
A random day is correct on 20% of days.
Lift = 30% / 20% = 1.5.
```

This means our signal is 1.5 times as likely to choose a correct day as random selection.

The target from the case is a stable lift of at least `1.3`. We should calculate lift only after
combining all indicators, removing duplicate signals, and applying the notification limit. The
random comparison must use the same currency, time period, and number of messages.

Every result should show:

- number of sent messages;
- signal hit rate;
- random hit rate;
- lift;
- a confidence interval showing how uncertain the result is.

A high lift based on five messages is much less convincing than the same lift based on hundreds of
messages.

## Metric 2: customer value and regret (CVaR)

Lift counts only whether a message was correct. It treats a tiny error and a very expensive error
as the same false positive. We must also measure the size of the difference in basis points.

### Average value of the selected moment

Compare the rate on the signal day with the average rate around that day:

```text
moment advantage in bp =
    10,000 * (average surrounding rate / signal-day rate - 1)
```

A positive result means the signal selected a better-than-average moment. The average result across
all sent messages should be greater than zero, with a confidence interval that does not include
zero.

### Customer regret after a message

Compare the signal-day rate with the best rate that appeared during the next `h` observations:

```text
regret in bp =
    10,000 * max(signal-day rate / best later rate - 1, 0)
```

Suppose a customer transfers ₽100,000 after our message:

- a 1 bp worse rate is approximately a ₽10 difference;
- a 300 bp worse rate is approximately a ₽3,000 difference.

Both are false positives, but the second is clearly more harmful. Report average regret, the worst
observed regret, and the average of the worst 5% of messages. If there are fewer than 100 messages,
report the worst five instead of making a strong statistical claim about the worst 5%.

The bank must decide what level of regret is acceptable using real transfer amounts and its risk
policy. We cannot choose that limit from public CBR data alone.

## Metric 3: useful signals per week

Lift can be made very high by sending only one message per year. That would not create a useful
product. Sending every day would create the opposite problem: notification fatigue.

Measure how many correct messages the final system produces:

```text
useful signals per 100 customer-weeks =
    100 * correct sent messages / eligible customer-weeks
```

Use this metric only while the following rules are satisfied:

- no customer receives more than two total bank notifications per week;
- only part of that total budget is reserved for FX messages;
- repeated signals from the same market movement are merged;
- a cooldown prevents messages from arriving too close together;
- lift and customer-value requirements still pass.

Also report the average number of FX messages per week, the percentage of weeks with no message,
and the percentage of messages sent within the cooldown period after a previous message.

The 1–2 message limit applies to each customer across all currencies and all bank communication. It
does not mean 1–2 messages for every currency.

Because the project has no customer data, calculate this metric for several simple scenarios, such
as a customer interested in one corridor and a customer interested in several corridors. Clearly
state how much of the weekly notification budget is assumed to be available for FX.

## Metric 4: stability

A model may work well on average while failing for one currency or one market period. To detect
this, evaluate it separately on future time periods that were not used to choose its parameters.

For every message type, horizon, currency, and test period, report lift and customer value. Then
show:

- the weakest lift result among currencies and test periods;
- the typical result, such as the median;
- how many test periods have customer value confidently above zero;
- the number of messages behind every result.

Do not hide a failing currency inside an average across all currencies. If a rule works for KZT but
not TJS, launch it only for KZT or calibrate a separate TJS rule.

## How to run a fair offline test

1. Define each message, its correct-answer rule, horizon, notification limit, and cooldown before
   the final test.
2. Use walk-forward evaluation: train on the past and test only on later observations.
3. Leave at least `h` observations between training labels and a test period when necessary, so
   future information cannot cross the boundary.
4. Choose model parameters and the send threshold using training data only.
5. Combine all indicators into the final message stream before calculating the four metrics.
6. Keep one final time period completely untouched until model selection is finished.
7. Calculate results separately by message, horizon, currency, and test period before showing any
   combined average.

## Metrics for a real customer pilot

The four offline metrics tell us whether the market signal is good enough to test. They cannot tell
us whether the notification causes more transfers.

For the pilot, randomly assign eligible customers or customer-events to two groups:

- **send group:** receives the FX message;
- **holdout group:** does not receive it.

The main pilot metric should be the additional transfer volume per eligible customer in the send
group compared with the holdout group. Use a long enough measurement period to detect transfers
that were merely moved from next week to this week.

Also measure profit after costs, later reduction in transfers, notification opt-outs, complaints,
and cases where the actual in-app rate became worse between sending and opening the message.

Do not use only “percentage of pushes followed by a transfer.” It has no holdout comparison and
counts customers who would have transferred without the message.

## Current conclusion

No baseline indicator is ready for launch yet. The current `h = 3` results show momentum lift of
about `3.00–3.32` and level lift of about `1.44–1.90` across five corridors, but they use the wrong
symmetric target, the whole history, and no confidence intervals.

The next step is to create message-specific targets and recompute lift, customer value, useful
signals per week, and stability using walk-forward test periods.

The supporting research and sources are in [`RESEARCH.md`](RESEARCH.md).
