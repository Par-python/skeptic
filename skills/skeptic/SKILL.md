---
name: skeptic
description: Check whether a data result is real or would show up from nothing. Use before stating or trusting a claim like "beats baseline", "X% accurate", "catches N% of cases", "p < 0.05", "significant", "real signal", "correlated", "leads by N weeks", "outperforms", or a backtest win, and when the user asks to verify, audit, or sanity-check an analysis or a significance test.
license: MIT
compatibility: Needs Python 3 with numpy. Runs locally, no network access.
---

# skeptic

One question: **is this result real, or would you have gotten it from nothing?**

Two kinds of "nothing":
1. **A do-nothing baseline.** Always guess the majority class. Flag everyone. Always predict the last value.
2. **Data with no effect in it.** Run the user's own test on pure noise 200 times and count how often it says "significant". A correct test says it about 5% of the time at alpha 0.05.

The engine is `${CLAUDE_SKILL_DIR}/scripts/skeptic.py` (numpy only; `${CLAUDE_SKILL_DIR}` is the folder holding this SKILL.md). Do the math with it. Never judge calibration by reading code: run it. Reasoning finds the suspect; the noise run is the evidence.

## Size the effort to the stakes

- **Classifier claim, or a standard test on independent rows** (t-test, A/B with real randomization): one quick check, a few seconds. Don't build a pipeline.
- **Anything that searched first** (best lag, biggest jump, best threshold, best feature) **or runs on a time series**: full workflow below. This is where reading the code isn't enough.
- If the full run will take more than about a minute, say so before starting (`skeptic.eta`).

## Pick the check by the claim

| Claim | Check |
|---|---|
| "X% accurate", "beats the baseline" | `baseline_check(..., claim="accuracy")` |
| "catches N% of cases", "the flags are useful", a call or review list | `baseline_check(..., claim="detection")`. Flagging everyone catches 100%, so recall alone proves nothing; the flags must beat the base rate |
| "p < 0.05", "significant", "real signal", "correlated", "leads by N", permutation or shuffle test | `sweep` (null calibration across autocorrelation) + `power` |
| a classifier gated by a significance test | both |

An accuracy OVERCLAIM is not the same as "useless". If the card's notes say the flags have real lift, report that too.

## Workflow

1. **Find the exact test.** Locate the function that produces the p-value or verdict. Import it unchanged. Never edit the user's code to make it testable; write a thin wrapper in a separate scratch file. The wrapper must return a p-value: skeptic raises on NaN, bools, or anything outside [0, 1] rather than silently miscounting.
2. **Match the noise to the data.** Same length, same kind, same number of series. Measure autocorrelation on the real data with `estimate_phi`, and test phi 0, 0.5, 0.9 with `sweep`. Most broken nulls look fine at phi 0 and fail at 0.9. Real trend and price data is often near 0.9; add phi 1.0 (random walks) for levels that never settle.
   - Two-series tests (correlation, lead/lag): `sweep(test, n, series=2)` gives each trial two independent series.
   - Other generators: `poisson_counts(n, rate, phi)` daily counts, `random_walk(n)`, `permuted(data)` the user's own values reshuffled (warn: hides autocorrelation problems).
   - `ar1` is stationary noise that pulls back to its mean. It has no trend and no drift.
3. **Run it.** 200 noise trials is the default.
4. **Read the verdict honestly.**
   - MISCALIBRATED: the noise hit rate's 95% CI is entirely above alpha. The claim can't stand on this test.
   - CALIBRATED: the CI includes alpha. With 200 trials that only rules out rates above about 9% for a true 5%. Say "no sign of miscalibration at this resolution", and use 1000 trials if milder inflation would matter.
   - CONSERVATIVE: the CI is entirely below alpha. No false-alarm problem, but the test may barely fire. Go straight to power.
5. **If MISCALIBRATED, find the cause with a control.** Change one thing and rerun:
   - Was something picked on the real data first and then tested alone? Rerun with that choice made at random. If the control is calibrated, selection is the bug. Fix: put the search inside the statistic and use `surrogate_p(stat_fn, x, surrogates, ...)`, so every surrogate searches too (max vs max).
   - Does it only fail at high phi? The shuffle is breaking autocorrelation. Fix with surrogates that keep it: `phase_randomize`, `block_bootstrap` (block well above (1+phi)/(1-phi)), `ar1_surrogates`, or `prewhiten` both series and then shuffle. `circular_shift` runs hot on short, strongly autocorrelated data. Check whichever you pick.
   - Trends compared as levels? Compare changes instead.
6. **Re-check every fix twice: calibration AND power.** Plant a real effect of a plausible size (ideally the size the user claims) and run `power`. LOW POWER means the fix can't find what it's looking for. Surrogates built from data that contains the effect can swallow it (phase and AR(1) surrogates inherit a real step). If nothing is both calibrated and has power, say so plainly: "this test can't answer this question with this much data." That is a valid result.
7. **Hand back a receipt.** `skeptic.receipt(cards)` gives a markdown table. Paste it, then two or three plain sentences: what the claim was, whether it held, what to say instead.

## Honesty rules

- Report n and the 95% CI every time. Never round a MISCALIBRATED into "basically fine".
- Calibrated means: the CI of the noise hit rate includes alpha. That's the only definition.
- If you write the pass/fail bar, write it **before** running, and keep it. If a result makes you want to change the bar, report the original result first and label anything new as exploratory.
- Don't call a fix done until it was rerun on fresh seeds (`sweep(..., seed=1)`).
- The noise check tests the statistics, not the design. Also look at the data itself: is assignment really random, is there a time split, leakage, or duplicated rows? Report what you can't rule out from the file.
- Never modify the repo under test, and never publish code or data the user says belongs to someone else.

## Quick start

```python
import sys; sys.path.insert(0, "${CLAUDE_SKILL_DIR}/scripts")
import skeptic

# one-series test: my_test(series) -> p-value
cards = skeptic.sweep(my_test, n=104, name="my test")
# two-series test: lead_lag_test(x, y) -> p-value
cards += skeptic.sweep(lead_lag_test, n=104, series=2, phis=(0.0, 0.5, 0.9, 1.0), name="lead/lag")
cards.append(skeptic.power(my_test, lambda i: (series_with_effect(i),), name="finds a real effect"))
print(skeptic.receipt(cards))

# a fixed test: search inside every surrogate
best_r = lambda x, y: max(abs(np.corrcoef(x[:len(x)-k], y[k:])[0, 1]) for k in range(9))
p = skeptic.surrogate_p(best_r, x, skeptic.phase_randomize(x, 199), y)
```

CLI for classifiers: `python3 ${CLAUDE_SKILL_DIR}/scripts/skeptic.py baseline y_true.txt y_pred.txt --claimed [--claim detection] [--json]` (exit 1 on OVERCLAIM).

Worked examples: `${CLAUDE_SKILL_DIR}/examples/quickstart.py` (random-walk correlation, a 79% accurate model that loses to doing nothing, and a churn model that loses on accuracy but has real lift) and `${CLAUDE_SKILL_DIR}/examples/changepoint/` (pick the biggest jump then test it: 92% false positives, with pre-registered fixes, including the ones that failed).
