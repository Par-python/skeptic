# skeptic

**Is this result real, or would you have gotten it from nothing?**

A Claude skill (and a small numpy library) that checks data claims against "nothing":

- **A do-nothing baseline.** Your model is 79% accurate. Always guessing "no" is 84%.
- **Data with no effect.** Your test says p < 0.05. Run it on 200 series of pure noise. A correct test says "significant" about 5% of the time. Does yours?

## What it catches

From [`examples/quickstart.py`](skills/skeptic/examples/quickstart.py):

| Check | Result | Verdict |
|---|---|---|
| Pearson on raw trends | 78% "significant" on pure noise | **MISCALIBRATED** |
| Pearson on changes (fix) | 3.5% on pure noise | **CALIBRATED** |
| fix still finds a real link | 98% of the time | **POWER OK** |
| "79% accurate" model | 79% vs 84% for always guessing "no" | **OVERCLAIM** |
| "90.5% accurate" churn model | loses to "no" (92%), but flags are 4.9x richer in churners | **OVERCLAIM** on accuracy, **PASS** as a call list |

From [`examples/changepoint/`](skills/skeptic/examples/changepoint/): pick the biggest jump in a series, then test that spot with a shuffle.
On pure noise it says "real" **92%** of the time. Searching again inside every shuffle fixes calibration.
Getting a fix that is calibrated *and* still finds real jumps on short, autocorrelated data failed both
pre-registered attempts. Those failures are in the repo too.

## What's in the library

- **Checks:** `baseline_check` (accuracy or detection claims), `null_calibration`, `power`, `sweep` (noise at
  several autocorrelation levels, one or two series per trial).
- **Noise with no effect:** `ar1`, `random_walk`, `poisson_counts`, `permuted`.
- **Building a fixed test:** `phase_randomize`, `block_bootstrap`, `ar1_surrogates`, `circular_shift`,
  `prewhiten`, and `surrogate_p`, which repeats any search (best lag, biggest jump) inside every surrogate.
- **Verdicts:** MISCALIBRATED (fires too often on noise), CALIBRATED (the CI includes alpha), CONSERVATIVE
  (barely fires, so check power), POWER OK or LOW POWER, and PASS, FAIL or OVERCLAIM for baselines.
  A test that returns NaN or a non-p-value is an error, not a silent "not significant".

## Install

**Claude Code** (as a plugin, gets updates):

```bash
claude plugin install skeptic --marketplace Par-python/skeptic
```

Or add the marketplace first and install from it:

```bash
claude plugin marketplace add Par-python/skeptic
claude plugin install skeptic@par-python-skeptic
```

**Claude app (claude.ai):** download `skeptic-skill.zip` from the
[latest release](https://github.com/Par-python/skeptic/releases/latest) and upload it under
Settings > Capabilities > Skills. It holds the whole `skills/skeptic/` folder; the skill needs its script,
so don't upload `SKILL.md` alone.

**Manual:** copy `skills/skeptic/` into `~/.claude/skills/`.

Then ask Claude to check a result, or just make a claim. The skill triggers on things like
"p < 0.05", "beats baseline", "X% accurate", "leads by N weeks".

Requires Python 3 with numpy. The examples also use scipy and pandas.

## What it runs

- Only `skills/skeptic/scripts/skeptic.py` (numpy) and small wrapper scripts Claude writes in a scratch folder.
- No network access, no hooks, no MCP servers.
- It imports your test unchanged and never edits the code under test.

## Use as a library

```python
import sys; sys.path.insert(0, "skills/skeptic/scripts")
import skeptic

cards = skeptic.sweep(my_test, n=104, name="my test")                  # noise at autocorrelation 0, 0.5, 0.9
cards += skeptic.sweep(lead_lag_test, n=104, series=2, name="lead/lag")  # two-series tests
print(skeptic.receipt(cards))
```

```bash
python3 skills/skeptic/scripts/skeptic.py baseline y_true.txt y_pred.txt --claimed [--claim detection] [--json]
```

## Development

```bash
python3 -m pytest tests/                       # unit tests, numpy only
claude plugin validate --strict .              # manifest and skill checks
claude plugin eval . --scaffold --allow-tools Bash Write   # behavior evals: with vs. without the skill
```

The eval suite in `evals/` runs realistic requests with and without the skill: a best-lag time-series claim,
a churn model that loses to "always no" on accuracy, a clean A/B test that should pass, and two requests the
skill should stay out of. Evals call the model on your account, so they run on demand, not on every push.

## Rules it follows

- Every number comes with n and a 95% CI.
- Pass/fail bars are written before running and not moved after.
- A fix counts only if it is calibrated **and** still finds a planted real effect.
- Sometimes the honest answer is "this data can't answer this question."

See [CHANGELOG.md](CHANGELOG.md). MIT license.
