"""Quickstart: two of the most common ways a data result lies. Run: python quickstart.py

1. "These two trends are correlated, p < 0.05."  Two random walks with no link at all.
2. "My model is 79% accurate."                     Always guessing 'no' does better.
3. "Our churn model is 90.5% accurate."            Also loses to 'no' (92%), but its flags are 5x richer
                                                    in churners than random. Wrong claim, useful model.
"""
import sys
from pathlib import Path
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import skeptic

N = 100


# ---- 1. correlation between two trend lines
def corr_p(x, y):
    """The test people actually run: Pearson correlation on the raw series."""
    return stats.pearsonr(x, y).pvalue


def corr_p_on_changes(x, y):
    """The fix: correlate week-to-week changes, not levels."""
    return stats.pearsonr(np.diff(x), np.diff(y)).pvalue


no_link = lambda i: (skeptic.random_walk(N, seed=2 * i), skeptic.random_walk(N, seed=2 * i + 1))

# a real link: y's changes follow x's changes
def linked(i):
    rng = np.random.default_rng(10_000 + i)
    dx = rng.normal(size=N)
    dy = 0.4 * dx + rng.normal(size=N)
    return np.cumsum(dx), np.cumsum(dy)


cards = [
    skeptic.null_calibration(corr_p, no_link, name="Pearson on raw trends"),
    skeptic.null_calibration(corr_p_on_changes, no_link, name="Pearson on changes (fix)"),
    skeptic.power(corr_p_on_changes, linked, name="fix still finds a real link"),
]

# ---- 2. accuracy vs doing nothing (19 cases, 3 positives; model never catches one)
y_true = [1] * 3 + [0] * 16
y_pred = [0] * 3 + [1] + [0] * 15
cards.append(skeptic.baseline_check(y_true, y_pred, claimed_beats=True, name="'79% accurate' model"))

# ---- 3. same OVERCLAIM on accuracy, but the flags have real lift: judge it as a detector instead
churn_true = [1] * 48 + [0] * 552                       # 600 customers, 8% churn
churn_pred = [1] * 17 + [0] * 31 + [1] * 26 + [0] * 526  # flags 43, 17 of them churned
cards.append(skeptic.baseline_check(churn_true, churn_pred, claimed_beats=True, name="'90.5% accurate' churn model"))
cards.append(skeptic.baseline_check(churn_true, churn_pred, claimed_beats=True, claim="detection",
                                    name="churn flags as a call list"))

print(skeptic.receipt(cards, title="quickstart receipt"))
