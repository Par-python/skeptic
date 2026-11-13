# Changelog

## 0.2.0

Packaged as a Claude Code plugin (`.claude-plugin/`, skill under `skills/skeptic/`), with an eval suite and CI.

### Fixed
- A test that returns NaN, a bool, a string, or a number outside [0, 1] now raises an error. Before, it was
  counted as "not significant", so a broken test could look calibrated.
- A test that never fires is now CONSERVATIVE instead of CALIBRATED, matching the stated definition
  (calibrated = the noise hit rate's 95% CI includes alpha).
- `ar1(n, phi=1.0)` returns a random walk. It used to return a constant series.
- Negative `phi` in `sweep` no longer crashes (negative seed).
- SKILL.md: "catches N% of cases" claims no longer go to an accuracy check, and `ar1` is no longer
  described as having drift.

### Added
- `baseline_check(..., claim="detection")`: judges flags by precision vs. base rate (lift, exact binomial
  test), since flagging everyone catches 100%. Every baseline card now reports precision, recall, lift, the
  confusion matrix and an accuracy CI, and an accuracy OVERCLAIM notes when the flags still have real lift.
- `sweep(..., series=2)` for two-series tests (correlation, lead/lag). `phi=1.0` gives random walks.
- Surrogates for building a fixed test: `phase_randomize`, `block_bootstrap`, `ar1_surrogates`,
  `circular_shift`, `prewhiten`, and `surrogate_p` (repeats any search inside every surrogate).
- `power` returns a verdict: POWER OK or LOW POWER (`min_power`, default 0.5).
- CALIBRATED cards say what the trial count can rule out (200 trials: rates above about 9% for a true 5%).
- `eta()` to estimate run time. CLI: `--claim`, `--positive`, `--json`.
- Unit tests (`tests/`), plugin eval suite (`evals/`), GitHub Actions CI.
- SKILL.md: sizes effort to the stakes, and checks the design (randomization, time splits, leakage),
  not just the statistics.

### Changed
- `baseline_check` picks the rarer class as the positive by default (was `1`).
- `sweep` uses new seeds, so its exact numbers differ from 0.1.
- The "random guessing" baseline was dropped; it can never beat the majority baseline.

## 0.1.0

First release.
