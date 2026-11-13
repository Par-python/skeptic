# Changepoint stand-in: pre-registration

Written before any of this code was run. Public example, written from scratch.

## The bug being reproduced
"Pick the most interesting spot, then test that spot as if you hadn't looked."
A detector scans a series for the biggest jump. A shuffle null is then run at that same spot.
The null never had to search, so it's too easy to beat.

## Setup
- Data with no effect: 104 weekly points, 50 + AR(1) noise (sd 10). phi in {0, 0.5, 0.9}. No step, ever.
- Detector: for every index c in [12, 92], z(c) = |mean(x[c:c+12]) - mean(x[c-12:c])| / sd(x). Pick the max.
- Statistic at c: mean(x[c:c+12]) - mean(x[c-12:c]), absolute value.
- 200 series per phi, 199 surrogates per test, alpha 0.05.

## Variants
- V0 naive: detector picks c. Block-shuffle (block 4). Statistic at the SAME c. **The bug.**
- C1 control: same as V0, but c is random instead of chosen. Isolates selection from shuffle choice.
- V1 selection-aware, block 4: rerun the detector inside every surrogate, compare max to max.
- V2 selection-aware, Fourier phase-randomized surrogates (keeps the autocorrelation).
- V3 selection-aware, AR(1) parametric bootstrap (fit mean, phi, sd; simulate).

## Predictions (stated, not assumed)
- V0 MISCALIBRATED at every phi.
- C1 roughly 5% at phi 0, inflated at phi 0.9 (block 4 breaks strong autocorrelation).
- V1 better than V0 but still inflated at phi 0.9.
- V2 and V3 close to 5% everywhere. Unknown which is better.

## Pass rule for "the fix"
A variant is the recommended fix only if, at all three phi:
- skeptic says CALIBRATED (Wilson 95% CI low end at or below 0.05), AND
- the point false-positive rate is at or below 0.075.

And it must still find real effects: power reported on a real step of 1.0 sd and 1.5 sd at a random
location in [24, 80], phi 0.5, 100 series each. Minimum 0.5 power at 1.5 sd.
If no variant passes both, that's reported as is. No re-tuning after seeing results without saying so.
