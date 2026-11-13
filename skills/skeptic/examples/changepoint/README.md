# Example: pick the biggest jump, then test it as if you hadn't looked

Public stand-in, written from scratch. Run: `python demo.py` (about 5 s), then `python demo2.py`.

## The bug
A detector scans a series for its biggest jump. A shuffle null then tests that same spot.
The real data got searched; the shuffles didn't. So noise beats its own null almost every time.

## What skeptic found (200 pure-noise series per row, alpha 0.05)

| Test | phi 0 | phi 0.5 | phi 0.9 |
|---|---|---|---|
| V0 naive (the bug) | **77%** | **94%** | **92%** |
| C1 same null, random index (control) | 5% | 4% | 12% |
| V1 search again in each shuffle, block shuffle | 4% | 19% | 30% |
| V2 search again, phase-randomized | 5% | 7% | 3% |
| V3 search again, AR(1) bootstrap | 3% | 7% | 5% |
| V4 search again, AR(1) fit after removing the step (PREREG2) | 6% | 5% | 18% |

The control isolates the cause. Same null at an index nobody chose is fine at phi 0, so selection is the bug.

## The fixes, honestly
- V2 and V3 are calibrated, but they find almost nothing real: 3% power on a 1.5 sd step.
  Their surrogates are built from data that already contains the step, so the step hides in its own null.
- V4 recovers power (44% at 1.5 sd) but over-fires at phi 0.9 (18%).
- **No variant passed the pre-registered rule.** Two tries, both reported. See PREREG.md, PREREG2.md, results*.json.

## What that means
With 104 weekly points and phi 0.9, there are only a handful of effectively independent points.
A real step and slow random wander look alike. At that length, no null here is both honest and useful.
That's a finding, not a failure of the tool: the right verdict is "this test can't answer this question
with this much data", not a recommended fix.
