"""skeptic: is this result real, or would you have gotten it from nothing?

Checks a claim made from data against "nothing", two ways:
  baseline_check(y_true, y_pred, claimed_beats)  does the model beat doing nothing?
  null_calibration(test_fn, null_gen)            does your test say "significant" on data with no effect?
  power(test_fn, signal_gen)                     does it still find a real effect?
  sweep(test_fn, n, ...)                         null_calibration across autocorrelation levels
Every check returns a verdict card (a dict). receipt(cards) turns them into a markdown receipt.

CLI:
  python skeptic.py baseline y_true.txt y_pred.txt [--claimed]
  python skeptic.py receipt card1.json card2.json ...
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

__version__ = "0.1.0"


# ---------------------------------------------------------------- stats helpers
def wilson(k, n, z=1.96):
    """95% Wilson interval for k successes out of n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def estimate_phi(x):
    """Lag-1 autocorrelation of a series. Use it to pick a realistic null."""
    x = np.asarray(x, dtype=float)
    if x.size < 3 or x.std() == 0:
        return 0.0
    return float(np.clip(np.corrcoef(x[:-1], x[1:])[0, 1], -0.99, 0.99))


# ---------------------------------------------------------------- null generators
# Each returns one series with NO effect in it. Call with a seed so results are reproducible.
def ar1(n, phi=0.0, mean=0.0, sd=1.0, seed=0):
    """Stationary AR(1) noise. phi=0 is white noise; phi near 1 drifts slowly like real trend data."""
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + math.sqrt(1 - phi ** 2) * e[t]
    return mean + sd * x


def random_walk(n, sd=1.0, seed=0):
    """Cumulative sum of noise. Two independent random walks often look correlated. They aren't."""
    return np.cumsum(np.random.default_rng(seed).normal(scale=sd, size=n))


def poisson_counts(n, rate=5.0, phi=0.0, sd_log=0.3, seed=0):
    """Daily counts (mentions, sales) with an autocorrelated rate and no trend or spike."""
    rng = np.random.default_rng(seed)
    lam = np.exp(np.log(rate) + ar1(n, phi, sd=sd_log, seed=seed + 1))
    return rng.poisson(lam)


def permuted(data, seed=0):
    """Your own values in random order. Keeps the distribution, kills any order or link.
    Warning: also kills autocorrelation, so a test can look calibrated here and fail on ar1."""
    return np.random.default_rng(seed).permutation(np.asarray(data))


# ---------------------------------------------------------------- checks
def baseline_check(y_true, y_pred, claimed_beats=None, positive=1, name="claim"):
    """Compare accuracy to the best do-nothing baseline, and require catching at least one positive."""
    y_true, y_pred = list(y_true), list(y_pred)
    n = len(y_true)
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / n
    counts = Counter(y_true)
    majority_label, majority_n = counts.most_common(1)[0]
    baselines = {
        f"always '{majority_label}' (majority)": majority_n / n,
        f"always '{positive}'": counts.get(positive, 0) / n,
        "random guessing at base rate (expected)": sum((c / n) ** 2 for c in counts.values()),
    }
    best_name, best = max(baselines.items(), key=lambda kv: kv[1])
    n_pos = counts.get(positive, 0)
    tp = sum(t == positive and p == positive for t, p in zip(y_true, y_pred))
    recall = tp / n_pos if n_pos else None
    beats = acc > best and (recall is None or recall > 0)
    reasons = []
    if acc <= best:
        reasons.append(f"accuracy {acc:.2f} does not beat '{best_name}' at {best:.2f}")
    if recall == 0:
        reasons.append(f"caught 0 of {n_pos} positives (recall 0)")
    if n < 30:
        reasons.append(f"small sample: n={n}, one prediction moves accuracy by {1/n:.2f}")
    overclaim = bool(claimed_beats) and not beats
    return {
        "check": "baseline", "name": name, "n": n, "accuracy": round(acc, 3),
        "recall": None if recall is None else round(recall, 3),
        "baselines": {k: round(v, 3) for k, v in baselines.items()},
        "beats_baselines": beats, "claimed_beats": claimed_beats,
        "verdict": "OVERCLAIM" if overclaim else ("PASS" if beats else "FAIL (and not claimed)"),
        "reasons": reasons,
    }


def null_calibration(test_fn, null_gen, n_trials=200, alpha=0.05, name="test"):
    """Run test_fn on data where the null is true by construction.
    null_gen(i) returns a tuple of arguments for test_fn; test_fn returns a p-value.
    A calibrated test says 'significant' about alpha of the time. Flag it if clearly more."""
    ps = [float(test_fn(*null_gen(i))) for i in range(n_trials)]
    k = sum(p < alpha for p in ps)
    fpr = k / n_trials
    lo, hi = (round(v, 3) for v in wilson(k, n_trials))
    flagged = lo > alpha
    return {
        "check": "null_calibration", "name": name, "trials": n_trials, "alpha": alpha,
        "false_positives": k, "false_positive_rate": round(fpr, 3), "ci95": [lo, hi],
        "verdict": "MISCALIBRATED" if flagged else "CALIBRATED",
        "reasons": ([f"says 'significant' {fpr:.0%} of the time on data with no effect "
                     f"(should be about {alpha:.0%}); 95% CI {lo:.0%}-{hi:.0%}"] if flagged else []),
    }


def power(test_fn, signal_gen, n_trials=100, alpha=0.05, name="test"):
    """Fraction of datasets WITH a real, planted effect that the test detects."""
    ps = [float(test_fn(*signal_gen(i))) for i in range(n_trials)]
    k = sum(p < alpha for p in ps)
    lo, hi = wilson(k, n_trials)
    return {"check": "power", "name": name, "trials": n_trials, "detected": k,
            "power": round(k / n_trials, 3), "ci95": [round(lo, 3), round(hi, 3)]}


def sweep(test_fn, n, phis=(0.0, 0.5, 0.9), make_args=None, n_trials=200, alpha=0.05, name="test"):
    """null_calibration at several autocorrelation levels. Most broken nulls look fine at phi=0.
    make_args(series) turns one noise series into test_fn's arguments (default: (series,))."""
    make_args = make_args or (lambda s: (s,))
    cards = []
    for phi in phis:
        gen = lambda i, phi=phi: make_args(ar1(n, phi, seed=int(1e6 * phi) + i))
        cards.append(null_calibration(test_fn, gen, n_trials, alpha, name=f"{name} (phi={phi})"))
    return cards


# ---------------------------------------------------------------- output
def save(card, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(card, indent=1, default=lambda o: o.item()))
    return card


def receipt(cards, title="skeptic receipt"):
    """Markdown receipt: one line per check, then the reasons. Paste it next to the claim."""
    cards = [cards] if isinstance(cards, dict) else list(cards)
    lines = [f"## {title}", "", "| Check | Result | Verdict |", "|---|---|---|"]
    for c in cards:
        if c["check"] == "baseline":
            best = max(c["baselines"].values())
            res = f"accuracy {c['accuracy']:.0%} vs best do-nothing {best:.0%} (n={c['n']})"
        elif c["check"] == "null_calibration":
            lo, hi = c["ci95"]
            res = f"{c['false_positive_rate']:.1%} 'significant' on pure noise (CI {lo:.0%}-{hi:.0%}, {c['trials']} trials)"
        else:
            lo, hi = c["ci95"]
            res = f"finds the planted effect {c['power']:.0%} of the time (CI {lo:.0%}-{hi:.0%})"
        lines.append(f"| {c['name'].replace('|', '/')} | {res} | **{c.get('verdict', 'POWER')}** |")
    reasons = [f"- {c['name']}: {r}" for c in cards for r in c.get("reasons", [])]
    if reasons:
        lines += ["", "Why:"] + reasons
    lines += ["", f"_skeptic {__version__}. Calibrated means the 95% CI of the noise hit rate includes alpha._"]
    return "\n".join(lines)


# ---------------------------------------------------------------- CLI
def _read_labels(path):
    return [tok for tok in Path(path).read_text().replace(",", " ").split()]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd = argv.pop(0)
    if cmd == "baseline":
        claimed = "--claimed" in argv
        files = [a for a in argv if not a.startswith("--")]
        yt, yp = _read_labels(files[0]), _read_labels(files[1])
        pos = Counter(yt).most_common()[-1][0]  # rarer class is the positive one
        card = baseline_check(yt, yp, claimed_beats=claimed, positive=pos, name=Path(files[1]).stem)
        print(receipt(card))
        return 1 if card["verdict"] == "OVERCLAIM" else 0
    if cmd == "receipt":
        cards = []
        for f in argv:
            obj = json.loads(Path(f).read_text())
            cards += obj if isinstance(obj, list) else [obj]
        print(receipt(cards))
        return 0
    print(f"unknown command {cmd!r}; try --help")
    return 2


if __name__ == "__main__":
    sys.exit(main())
