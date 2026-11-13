# skeptic

**Is this result real, or would you have gotten it from nothing?**

A Claude skill (and a small numpy library) that checks data claims against "nothing":

- **A do-nothing baseline.** Your model is 79% accurate. Always guessing "no" is 84%.
- **Data with no effect.** Your test says p < 0.05. Run it on 200 series of pure noise. A correct test says "significant" about 5% of the time. Does yours?

## What it catches

From `examples/quickstart.py`:

| Check | Result | Verdict |
|---|---|---|
| Pearson on raw trends | 78% "significant" on pure noise | **MISCALIBRATED** |
| Pearson on changes (fix) | 3.5% on pure noise | **CALIBRATED** |
| fix still finds a real link | 98% of the time | power |
| "79% accurate" model | 79% vs 84% for always guessing "no" | **OVERCLAIM** |

From `examples/changepoint/`: pick the biggest jump in a series, then test that spot with a shuffle.
On pure noise it says "real" **92%** of the time. Searching again inside every shuffle fixes calibration.
Getting a fix that is calibrated *and* still finds real jumps on short, autocorrelated data failed both
pre-registered attempts. Those failures are in the repo too.

## Install as a Claude skill

Copy the `skeptic/` folder into `~/.claude/skills/` (Claude Code) or upload it as a skill in Claude.
Then ask Claude to check a result, or just make a claim. The skill triggers on things like
"significant", "beats baseline", "real signal".

## Use as a library

```python
import skeptic   # scripts/skeptic.py, needs numpy (examples/quickstart.py also needs scipy)

cards = skeptic.sweep(my_test, n=104, name="my test")   # noise at autocorrelation 0, 0.5, 0.9
print(skeptic.receipt(cards))
```

```
python scripts/skeptic.py baseline y_true.txt y_pred.txt --claimed
```

## Rules it follows

- Every number comes with n and a 95% CI.
- Pass/fail bars are written before running and not moved after.
- A fix counts only if it is calibrated **and** still finds a planted real effect.
- Sometimes the honest answer is "this data can't answer this question."

MIT license.
