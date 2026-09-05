# Seven new user-facing FX signals

Research date: 5 September 2026.

## Summary of findings

The seven proposed signals are:

1. **Better than one year ago** — a long-history signal. It compares today's reference recipient
   amount with the amount around the same date one year earlier.
2. **A better range has held** — a short-history signal. It detects a clear upward shift in the
   recipient amount that remains present for three publications.
3. **A larger-than-usual latest improvement** — a short-history signal. It detects when one
   favourable update is large compared with the corridor's recent normal movement.
4. **Today is better than the average for 30 days** — a short-history signal. It compares today's
   recipient amount with the average across the latest 30 calendar days.
5. **Most recent changes were favourable** — a short-history signal. It requires a meaningful net
   improvement and favourable movement in at least four of the latest five changes.
6. **Less recipient currency than one year ago** — a long-history worsening signal. It detects when
   the same rubles correspond to meaningfully less recipient currency than around this date last
   year.
7. **Most recent changes were unfavourable** — a short-history worsening signal. It requires a
   meaningful net decline and unfavourable movement in at least four of the latest five changes.

Signals 6 and 7 are the latest additions. The full set now contains two long-history and five
short-history signals.

These are new user-facing claims, not new names for the four current messages. The present signal
library already covers a strict consecutive decline, a low 90-publication percentile, a bounce from
a minimum, and holiday seasonality. Some low-level calculations such as returns can be reused; the
new part is the trigger definition, evidence shown to the user, and message truth test.

All message examples below are illustrative. Bracketed values must be filled from the observation
that caused the signal; none is a measured result from the current prototype.

Novelty was checked against the repository's case materials, current indicators, model features and
UX mock. The linked Miro board was not accessible in this environment, so its owner should make one
final duplicate check against these seven concepts.

The clearest language is not “the exchange rate rose” or “the exchange rate fell.” Those phrases
depend on how the pair is quoted. Use this stable mental model instead:

> For the same ruble amount, the recipient gets more or less local currency.

All seven signals describe only past and present observations. None says what will happen next.
They need a clean walk-forward backtest and a comprehension test with real remittance users.

## Selection method

The indicators were selected using five criteria:

- **Two-second meaning:** the main fact can be understood without knowing trading language.
- **Relevant outcome:** the message can be expressed as more or less recipient currency for the same
  rubles.
- **Different claim:** it is not simply another wording of the existing streak, percentile, bounce
  or holiday signals.
- **Data fit:** it can be calculated from fresh CBR publications without high, low or volume data.
- **Safe wording:** it states a historical comparison, not a forecast, guarantee or instruction.

The review in [FINANCE_INDICATORS_SUMMARY.md](FINANCE_INDICATORS_SUMMARY.md) found that moving
averages, ROC, RSI, MACD, range rules and volatility measures are common in trading. We expect their
plain evidence to be easier for this audience than their technical names; this must be tested. The
seven choices below use ideas from long-horizon ROC, channel confirmation, volatility normalization,
a rolling-average comparison and trend breadth, without exposing trading terms to the customer.

## Shared definition

Let:

- `q_t` be RUB per one unit of recipient currency at publication `t`; lower is better.
- `u_t = 1 / q_t` be recipient-currency units per RUB; higher is better.
- `A_t(S) = S × u_t` be the reference recipient amount for `S` rubles.

The model may calculate with `q_t`, but copy and charts should normally use `u_t` or `A_t`. A
“publication” means a new official rate, not a calendar day. The exact windows and thresholds below
are starting hypotheses. They must be chosen inside each walk-forward training fold, not after
looking at the test period.

CBR historical data gives an effective date, not a reliable intraday publication timestamp. CBR
says the exact posting time is not regulated and the rate normally takes effect on the next calendar
day. The prototype's `available_at = effective_date` convention is conservative for backtesting, but
the interface must show the time when the bank actually checked its source or live quote.
[CBR publication-time explanation](https://www.cbr.ru/Reception/TopicalMessage/Page/2661)
and [CBR effective-date explanation](https://www.cbr.ru/Reception/TopicalMessage/Page/2656)

## Signal 1 — Better than one year ago

**Type:** long-history comparison with a one-year lookback.

**Question answered:** For the same ruble amount, does the recipient get meaningfully more local
currency than around this date last year?

### Calculation

For effective date `d_t`, shift the date back by one calendar year (treat 29 February as 28 February
when needed), then choose `y(t)`: the latest publication on or before that date. Reject the
comparison if that publication is more than seven calendar days before the target date. Then:

`gain_year(t) = u_t / u_y(t) - 1`

The signal becomes eligible when:

- `gain_year(t)` crosses above a practical threshold calibrated for the corridor;
- the previous fresh observation has a valid annual comparison and was below that threshold;
- both observations pass the project's freshness and quality checks;
- the signal has not already been sent in the current episode.

Using the last publication on or before the target date avoids looking ahead when the exact date was
a weekend or holiday. This is a long-horizon Rate of Change comparison, but the user does not need
that name. It is also distinct from the current level signal: it compares with one clear past date,
not today's position inside a 90-publication distribution.

### User copy

Illustrative push title:

> More tenge than a year ago

Push body:

> At the latest CBR reference, the same rubles correspond to [gain]% more tenge than around this date last year.

Expanded card:

> Compared CBR effective rates for [current date] and [reference date]. This is a nominal currency comparison, not a measure of what the money can buy.

> The comparison does not predict the next rate. Your transfer uses the live rate shown below.

### Why it may be understandable

It uses one familiar reference point, one percentage and the recipient's currency. There is no
technical score or “top 10%” concept to decode. The exact two dates remain visible in the card.

### Main risks

- A one-year comparison says little about whether today is good relative to last week or next week.
- Two endpoints hide the path between them.
- Inflation may reduce what the extra recipient currency can buy.
- The condition can stay above the threshold for a long time.

Send only on the first threshold crossing, apply the shared cooldown, and keep the purchasing-power
caveat in the explanation. It must never be phrased as “the rate will continue to improve.”

## Signal 2 — A better range has held

**Type:** short-history confirmation, using ten fresh publications.

**Question answered:** Did the recipient amount move to a clearly better short-term range and remain
there, rather than touching it once?

### Calculation

Compare two non-overlapping blocks:

- the latest block: publications `t-2` to `t`;
- the earlier block: publications `t-9` to `t-3`.

The signal becomes eligible when:

- `gap = min(latest block) / max(earlier block) - 1` is above a calibrated practical margin;
- the signal is sent only on the first confirmed publication, followed by a cooldown.

This deliberately strict rule means that the two observed ranges do not overlap. It is a confirmed
short-term level shift, not “today is in the lowest 10% of a 90-publication window.” It can trigger
after one clear jump followed by two nearly flat updates, which the current consecutive-decline rule
would miss.

### User copy

Illustrative push:

> For 3 updates, the same rubles have corresponded to more tenge than in any of the previous 7 updates.

Expanded card:

> Even the lowest CBR reference amount in the latest 3 updates was [gain]% above the highest amount in the previous 7.

> This is a past comparison, not a forecast. The current transfer rate and recipient amount are shown below.

### Why it may be understandable

“Moved and stayed” is a simple story. It also explains why the bank waited for confirmation. The
user does not need to understand channels, breakouts, minimums or maximums; those details can remain
under “How we calculate this.”

### Main risks

- Confirmation arrives later, so part of the advantage may disappear while the system waits.
- The result can depend on the 3-versus-7 block boundary.
- A better short-term range may still be weak in long-term context.

The backtest should measure the price of waiting from the first shift to the third confirming
publication. The card should say “3 updates,” not “3 days.”

## Signal 3 — A larger-than-usual latest improvement

**Type:** fast short-history move, using the latest update and the previous ten changes.

**Question answered:** Was the latest favourable change large enough to stand out from recent
update-to-update noise?

### Calculation

Calculate the latest change in recipient value:

`move_1(t) = u_t / u_(t-1) - 1`

Calculate the normal recent move from the ten changes that were fully known before `t`:

`usual_10(t) = median(abs(u_i / u_(i-1) - 1)) for i = t-10 ... t-1`

The signal becomes eligible when:

- `move_1(t)` is favourable and above a calibrated practical-benefit floor;
- `move_1(t) >= 2 × max(usual_10(t), epsilon_c)`;
- the data is fresh and the observation passes the project's quality checks;
- this is the first qualifying observation in the episode, followed by a cooldown.

Here `epsilon_c` is a small corridor-specific floor fixed in training data. It prevents division by
zero when several published values are unchanged. Keep the two-times rule fixed; calibrate only the
practical-benefit and zero-denominator floors. The prototype already has a one-update return and a
20-update standard deviation, but it does not use this robust recent-move comparison as a
user-facing trigger.

### User copy

Illustrative push:

> At the latest CBR update, the same rubles correspond to [gain]% more tenge. This change was at least twice the usual recent change.

Expanded card:

> “Usual” means the median size of the previous 10 update-to-update changes. This public reference is not a locked transfer rate.

> The change has already happened. It does not tell us what the next rate will do.

### Why it may be understandable

The message states the practical direction first and says “twice the usual change” instead of
showing volatility, ATR or a standard deviation. Comprehension testing must confirm that users
understand this comparison.

### Main risks

- A data correction or unusual fixing can look like a real market move, so quality checks matter.
- A large change can reverse at the next update.
- This signal and a long-history signal may describe the same event.
- A ten-change benchmark can be unstable during a sudden change in market conditions.

Merge it with any long-history signal when both fire. Show the clearest single fact rather than two
notifications or two claims stacked in one push.

## Signal 4 — Today is better than the average for 30 days

**Type:** short-history level comparison, using the latest 30 calendar days.

**Question answered:** Does the same ruble amount correspond to meaningfully more recipient currency
today than on an average day in the recent 30-day period?

### Calculation

Let `D_30(t)` contain all valid publications with effective dates from `d_t - 29 days` through
`d_t`, including the current publication. Define:

`average_30d(t) = mean(u_i for i in D_30(t))`

`gain_vs_average_30d(t) = u_t / average_30d(t) - 1`

The signal becomes eligible when:

- `gain_vs_average_30d(t)` crosses above a practical corridor-specific threshold;
- the window contains at least 15 valid publications and covers at least 25 calendar days;
- the current observation is fresh and passes the project's data-quality checks;
- the signal has not already been sent in the current episode.

The calendar window matches the user-facing “30 days” claim. The minimum coverage prevents a short
or broken series from being described as a full month. This is simpler than the current 90-update
percentile: it compares today with one familiar recent average and reports the direct difference.

### User copy

Illustrative push title:

> Today is better than the average for 30 days

Push body:

> At the latest CBR reference, the same rubles correspond to [gain]% more tenge than the average of published rates over the last 30 days.

Expanded card:

> The comparison uses [count] valid CBR publications from [start date] through [current date]. Days without a new published rate do not add another value.

> This average describes the recent past and does not predict the next rate. Your transfer uses the live rate shown below.

### Why it may be understandable

“Today versus the last 30 days” is a familiar comparison. It uses one period, one average and one
percentage. The detailed card shows the exact dates and number of published rates, while the push
keeps the explanation short.

### Main risks

- A few unusual rates can pull the arithmetic average up or down.
- The result changes as old observations leave the rolling window.
- Today can be above the 30-day average but still be worse than last week.
- The condition can remain true for many updates, so threshold crossing and cooldown are required.
- It overlaps with the current 90-update level signal and must be compared against it directly.

Test whether users understand “average” more reliably than “lowest 10%.” Keep only the clearer rule
if the two triggers select almost the same market episodes.

## Signal 5 — Most recent changes were favourable

**Type:** short-history direction and magnitude signal, using the latest five changes.

**Question answered:** Has the recipient amount improved repeatedly in recent updates, even if the
path was not a perfect uninterrupted streak?

### Calculation

Use the five changes from publication `t-5` through publication `t`:

`favourable_count_5(t) = count(u_i > u_(i-1)) for i = t-4 ... t`

`gain_5(t) = u_t / u_(t-5) - 1`

The signal becomes eligible when:

- `favourable_count_5(t) >= 4`;
- `gain_5(t)` is above a calibrated practical-benefit floor;
- the condition has just changed from false to true;
- the data is fresh and the signal is outside the shared cooldown.

The count describes consistency, while the net-gain floor prevents four tiny favourable changes
from triggering an unimportant message. Unlike the current momentum signal, one flat or adverse
change is allowed. Unlike **A better range has held**, the latest block does not need to be entirely
above the earlier block. Unlike **A larger-than-usual latest improvement**, no single update must be
unusual.

### User copy

Illustrative push title:

> More tenge across recent updates

Push body:

> The same rubles now correspond to [gain]% more tenge than five CBR updates ago. [count] of the five changes moved in this direction.

Expanded card:

> We count published changes, not calendar days. A small move in the other direction can occur inside this comparison.

> This pattern has already happened. It does not mean that the next update will also be favourable.

### Why it may be understandable

“Four of five changes” is easier to explain than a moving average, RSI or trend-strength score. It
also teaches an important idea: a general improvement does not require every update to move in the
same direction.

### Main risks

- The five-change window and four-change rule are design choices that may not generalize.
- Counting direction ignores move size; the separate net-gain floor is therefore essential.
- One large adverse change can coexist with four small favourable changes.
- It may overlap with the current consecutive-decline signal or the confirmed-range signal.
- The signal can arrive after much of the improvement has already happened.

Compare it directly with the current strict momentum rule. If both fire, show only the explanation
that wins the comprehension test and has the better out-of-time customer-value evidence.

## Signal 6 — Less recipient currency than one year ago

**Type:** long-history worsening comparison with a one-year lookback.

**Question answered:** Does the same ruble amount correspond to meaningfully less recipient currency
than around this date last year?

### Calculation

For effective date `d_t`, find the one-year reference publication `y(t)` using the same safe date
matching as Signal 1: shift back one calendar year and take the latest publication on or before that
date, with a maximum seven-day gap. Then:

`loss_year(t) = 1 - u_t / u_y(t)`

The signal becomes eligible when:

- `loss_year(t)` crosses above a practical corridor-specific threshold;
- the previous fresh observation had a valid annual comparison below that threshold;
- both observations pass the project's freshness and quality checks;
- the signal has not already been sent in the current episode.

The crossing rule makes this a worsening event rather than a permanent “bad rate” label. It is the
negative counterpart of **Better than one year ago** and should use the same reference-date logic
and comparable thresholds.

### User copy

Illustrative push title:

> Less tenge than a year ago

Push body:

> At the latest CBR reference, the same rubles correspond to [loss]% less tenge than around this date last year.

Expanded card:

> Compared CBR effective rates for [current date] and [reference date]. This is a nominal currency comparison, not a measure of purchasing power.

> The comparison describes what has already changed. It does not predict whether the next rate will improve or worsen.

### Why it may be understandable

It uses one familiar reference date and states the practical result directly: the recipient gets
less for the same rubles. The user does not need to interpret whether an exchange-rate number moving
up or down is favourable.

### Main risks

- A one-year comparison uses only two endpoints and hides the path between them.
- Inflation and purchasing power are not included.
- The rate may already be improving over the latest few updates despite the weak annual comparison.
- A negative push can create anxiety without giving the user a useful action.
- The condition can remain true for a long period.

For a product focused on finding good transfer moments, use this first as an educational in-app
status or as a reason to suppress positive wording. Send it as a push only in a separately tested,
opted-in worsening-alert journey.

## Signal 7 — Most recent changes were unfavourable

**Type:** short-history worsening signal, using the latest five changes.

**Question answered:** Has the recipient amount worsened repeatedly across recent publications?

### Calculation

Use the five changes from publication `t-5` through publication `t`:

`unfavourable_count_5(t) = count(u_i < u_(i-1)) for i = t-4 ... t`

`loss_5(t) = 1 - u_t / u_(t-5)`

The signal becomes eligible when:

- `unfavourable_count_5(t) >= 4`;
- `loss_5(t)` is above a calibrated practical-loss floor;
- the condition has just changed from false to true;
- the data is fresh and the signal is outside the shared cooldown.

The count shows that worsening is broad rather than caused only by one update. The loss floor stops
four tiny movements from producing an unhelpful warning. This is different from the current
one-update reversal signal because it requires a material net decline across five changes.

### User copy

Illustrative push title:

> Less tenge across recent updates

Push body:

> The same rubles now correspond to [loss]% less tenge than five CBR updates ago. [count] of the five changes moved in this direction.

Expanded card:

> We count published changes, not calendar days. A small move in the other direction can occur inside this comparison.

> This recent pattern does not tell us what the next update will do. Check the current transfer rate before deciding.

### Why it may be understandable

The message uses a named short window, a direct loss and a simple count. It teaches that a worsening
trend can contain one small movement in the other direction without exposing a technical momentum
score.

### Main risks

- The five-change window and four-change rule are design choices that need out-of-time testing.
- One large favourable change can coexist with four small unfavourable changes.
- The warning may arrive after most of the deterioration has already happened.
- It may overlap with the existing reversal or window-closing message.
- Frequent negative alerts may increase notification fatigue or anxiety.

Use this as the short-history counterpart of Signal 6. If both worsening signals fire, show only one
claim. Prefer the short signal when the purpose is to explain recent dynamics and the long signal
when the purpose is historical context.

## UX research and product evidence

The Wise and Xe rate-alert documentation reviewed presents direct rates and user-set targets; it
does not expose named technical indicators. Wise also offers a 30-day tracker and daily updates. Xe
explicitly says that its alert uses an informational mid-market rate; the user must check the current
send rate.
[Wise rate-alert documentation](https://wise.com/help/articles/2932395/whats-the-mid-market-exchange-rate)
and [Xe rate-alert documentation](https://help.xe.com/hc/en-gb/articles/360019612078-Xe-Rate-Alerts)

The practical lesson is to keep the technical selection rule behind the interface. Every customer
message should answer:

1. What changed?
2. Compared with which exact publications or period?
3. What does that mean for the recipient?
4. When was the reference calculated?
5. Which live rate will actually be used?

Financial communication guidance supports clear language, visible key information, limited detail
and testing with the intended audience. It also warns that a message should help a customer make a
timely, informed decision rather than only meet a readability score.
[FCA consumer-understanding guidance](https://www.fca.org.uk/publications/good-and-poor-practice/consumer-understanding-good-practice-areas-improvement)

Apple describes a good notification as concise, valuable and understandable at a glance, and
advises against repeated notifications about the same event. It also warns against exposing
sensitive information on the lock screen. Therefore, use a general fact or percentage in the push
and show a personal transfer amount only after the app is opened.
[Apple notification guidance](https://developer.apple.com/design/human-interface-guidelines/notifications/)

## Copy rules for all seven signals

- Say “public reference rate” in the explanation; never imply that it is locked.
- For market-derived signals, lead with “more” or “less recipient currency for the same rubles.”
- Use the currency name and a named window; do not show pair notation such as `RUB/KZT` alone.
- Use “updates” or “published rates,” not calendar days.
- Show at most one main percentage in the compact card. Put the formula in an optional details view.
- Avoid “buy,” “sell,” “oversold,” “breakout,” “strong signal,” “best moment” and basis points.
- Do not use red/green or arrows as the only explanation of direction.
- If the signal is stale when opened, say so before showing any historical explanation.
- Show the current app quote, fees and recipient amount before the transfer action.

## Validation plan

### Offline signal test

For each candidate:

1. Calculate every feature using only information available at publication `t`.
2. Calibrate windows and thresholds inside each training fold and freeze them for its future test
   fold.
3. First verify that the literal message is true at send time. Then score the case's predeclared
   forward outcome over `h = 1, 3, 5, 10, 20` publications. A later reversal does not make the
   historical sentence false, but it can still make the signal outcome a miss for lift, value and
   regret.
4. Report lift over a matched random policy, average advantage, customer regret, useful alerts,
   frequency and clustering.
5. Report every corridor and out-of-time fold separately, including stress periods.
6. Merge signals from the same market episode before applying cooldown and the total customer
   communication limit.

The case target remains no more than two total bank messages per customer per week. These seven
signals are alternatives competing for that budget, not seven messages to send together.

### Comprehension test

Show the push and landing card to regular remittance users without first teaching the interface.
Ask them to explain, in their own words:

- whether the recipient gets more or less;
- which period is being compared;
- whether the message predicts tomorrow;
- whether the shown reference is the final transfer rate;
- what they would do if the signal has expired.

Test the main languages used by the intended corridors. Record correct interpretation and time to
answer, not only preference or click intent. The simplest winning copy is the one that preserves
truth while producing the fewest direction, timeframe and guarantee errors.

## Recommendation

Start model testing with **A better range has held**. It has the simplest story and naturally
supports the case's fast-versus-slow analysis: compare the first shift with the later confirmation
and measure the price of waiting.

Keep **Better than one year ago** as the simplest long-history baseline. It does not repeat the
current “low percentile” claim, but it is context, not a forecast or proof that today is the best
transfer moment.

Test **Today is better than the average for 30 days** as a direct short-history level signal. It is
easier to explain than a percentile and gives the user a familiar comparison period. Because it may
overlap with the current 90-update level signal, keep the one that produces fewer comprehension
errors and better out-of-time customer value.

Use **A larger-than-usual latest improvement** as the fast challenger. It is more distinct from the
existing streak than the earlier “steady improvement” idea, but it may overlap with a long-history
signal during a sharp move. Deduplicate them before applying the customer-level frequency limit.

Use **Most recent changes were favourable** as a smoother short-history challenger. Compare it with
the current strict momentum rule and **A better range has held**. Its “four of five” explanation is
simple, but it should survive only if users understand it faster and the out-of-time test shows a
meaningful customer benefit.

Treat **Less recipient currency than one year ago** and **Most recent changes were unfavourable** as
a separate worsening scenario. They are clear historical facts, but they are not reasons to label
the present moment profitable. First test them as in-app education or positive-push suppressors. A
worsening push needs its own opt-in, comprehension test and incremental-value experiment.

A user-set recipient-amount target is also a strong future UX option; Wise and Xe document similar
features. It is not one of the seven research signals because this repository has neither saved user
targets nor historical executable bank quotes. It should be evaluated separately when those inputs
exist.
