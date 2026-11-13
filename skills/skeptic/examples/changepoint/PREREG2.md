# Changepoint stand-in: pre-registration 2

Written after PREREG.md's run, before this code ran. PREREG.md result: V2 and V3 calibrated at all phi,
but power at most 7% on a 1.5 sd step. FAIL under its rule. That result stands.

## Why (from explore_power.py, exploratory)
Surrogates were built from data containing the step. AR(1) fitted to a stepped series gives phi near 1,
so null surrogates wander as much as the real step. Power fell as the step grew. Also, z divided by the
whole-series sd, which a step inflates.

## New variant (only one)
V4 selection-aware, AR(1) bootstrap fitted on residuals:
1. Find the best step on the real data (same detector, same windows).
2. Subtract a level shift at that point. Fit mean, phi, sd on what's left.
3. Simulate 199 AR(1) series with that phi and sd. Rerun the detector in each.
4. Statistic: raw max |mean after - mean before| in data units (no division by series sd).

## Fresh seeds, same everything else
Noise seeds start at 20,000,000. Power seeds at 30,000,000. Same phi, sizes, counts, alpha.

## Pass rule: same as PREREG.md
CALIBRATED at all phi with point FPR at or below 0.075, and power at least 0.5 on a 1.5 sd step at phi 0.5.

## Prediction
Calibration may slip a bit, since removing the best step from pure noise lowers the phi estimate
(anti-conservative). Power should rise a lot. Either outcome gets reported.
