---
name: skeptic
description: Check whether a data result is real or would show up from nothing. Use before stating or trusting a claim like "beats baseline", "X% accurate", "p < 0.05", "significant", "real signal", "correlated", "outperforms", or a backtest win, and when the user asks to verify, audit, or sanity-check an analysis or a significance test.
---

# skeptic

One question: **is this result real, or would you have gotten it from nothing?**

Two kinds of "nothing":
1. **A do-nothing baseline.** Always guess the majority class. Always predict the last value.
2. **Data with no effect in it.** Run the user's own test on pure noise 200 times and count how often it says "significant". A correct test says it about 5% of the time at alpha 0.05.

The engine is `scripts/skeptic.py` (numpy only). Do the math with it. Never judge calibration by reading code: run it.

## When a claim shows up

Pick the check by the claim:

| Claim | Check |
|---|---|
| "model is X% accurate", "beats the baseline", "catches N% of cases" | `baseline_check` |
| "p < 0.05", "significant", "real signal", "correlated", permutation or shuffle test | `null_calibration` (via `sweep` for series) + `power` |
| both (a classifier gated by a significance test) | both |

## Workflow

1. **Find the exact test.** Locate the function that produces the p-value or verdict. Import it unchanged. Never edit the user's code to make it testable; write a thin wrapper in a separate scratch file instead.
2. **Match the noise to the data.** Same length, same kind (series, counts, two series). For time series, measure autocorrelation on the real data with `estimate_phi`, and always test phi 0, 0.5, 0.9 with `sweep`. Most broken nulls look fine at phi 0 and fail at 0.9. Real trend and price data is often near 0.9.
   - `ar1(n, phi)` series with drift, `random_walk(n)` levels, `poisson_counts(n, rate, phi)` daily counts, `permuted(data)` the user's own values reshuffled (warn: hides autocorrelation problems).
3. **Run it.** 200 noise trials is the default. If the test is slow, time one call first and say how long the full run will take.
4. **If MISCALIBRATED, find the cause with a control.** Change one thing and rerun:
   - Was something picked on the real data first (best lag, biggest jump, best threshold, best feature) and then tested alone? Rerun with that choice made at random. If the control is calibrated, selection is the bug. Fix: repeat the choice inside every shuffle and compare max to max.
   - Does it only fail at high phi? The shuffle is breaking autocorrelation. Fix: surrogates that keep it (block length well above the correlation time, phase randomization, AR fit on residuals).
   - Trends compared as levels? Compare changes instead.
5. **Re-check every fix twice: calibration AND power.** Plant a real effect of a plausible size with a signal generator and run `power`. A fix that never fires is not a fix. If nothing is both calibrated and has power, say so plainly: "this test can't answer this question with this much data." That is a valid result.
6. **Hand back a receipt.** `skeptic.receipt(cards)` gives a markdown table. Paste it, then two or three plain sentences: what the claim was, whether it held, what to change.

## Honesty rules

- Report n and the 95% CI every time. Never round a MISCALIBRATED into "basically fine".
- Calibrated means: the CI of the noise hit rate includes alpha. That's the only definition.
- If you write the pass/fail bar, write it **before** running, and keep it. If a result makes you want to change the bar, report the original result first and label anything new as exploratory.
- Don't call a fix done until it was rerun on fresh seeds.
- Never modify the repo under test, and never publish code or data the user says belongs to someone else.

## Quick start

```python
import sys; sys.path.insert(0, "<skill dir>/scripts")
import skeptic

cards = skeptic.sweep(my_test, n=104, name="my test")            # my_test(series) -> p-value
cards.append(skeptic.power(my_test, lambda i: (series_with_effect(i),), name="finds a real effect"))
print(skeptic.receipt(cards))
```

CLI for classifiers: `python scripts/skeptic.py baseline y_true.txt y_pred.txt --claimed` (exit 1 on OVERCLAIM).

Worked examples: `examples/quickstart.py` (random-walk correlation, 79% accuracy that loses to doing nothing) and `examples/changepoint/` (pick the biggest jump then test it: 92% false positives, with pre-registered fixes, including the ones that failed).
