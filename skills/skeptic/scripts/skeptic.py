"""skeptic: is this result real, or would you have gotten it from nothing?

Checks a claim made from data against "nothing", two ways:
  baseline_check(y_true, y_pred, claimed_beats)  does the model beat doing nothing?
  null_calibration(test_fn, null_gen)            does your test say "significant" on data with no effect?
  power(test_fn, signal_gen)                     does it still find a real effect?
  sweep(test_fn, n, series=1)                    null_calibration across autocorrelation levels
Building a fixed test:
  phase_randomize / block_bootstrap / circular_shift / ar1_surrogates   null copies that keep autocorrelation
  prewhiten(x)                                    strip AR(1) autocorrelation so a plain shuffle works again
  surrogate_p(stat_fn, x, surrogates)             p-value that repeats any search inside every surrogate
Every check returns a verdict card (a dict). receipt(cards) turns them into a markdown receipt.

CLI:
  python skeptic.py baseline y_true.txt y_pred.txt [--claimed] [--claim accuracy|detection] [--positive LABEL] [--json]
  python skeptic.py receipt card1.json card2.json ...
"""
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

__version__ = "0.2.0"


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


def binom_sf(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p). Exact, no scipy."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return min(1.0, sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1)))


def estimate_phi(x):
    """Lag-1 autocorrelation of a series. Use it to pick a realistic null."""
    x = np.asarray(x, dtype=float)
    if x.size < 3 or x.std() == 0:
        return 0.0
    return float(np.clip(np.corrcoef(x[:-1], x[1:])[0, 1], -0.99, 0.99))


def _as_p(value, trial):
    """A test must return a p-value in [0, 1]. Anything else silently skews the hit rate, so stop."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"trial {trial}: test returned a bool ({value}). Return the p-value, not the decision.")
    if isinstance(value, (str, bytes)):
        raise TypeError(f"trial {trial}: test returned a string ({value!r}), expected a p-value")
    try:
        p = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"trial {trial}: test returned {type(value).__name__}, expected a p-value") from None
    if not math.isfinite(p):
        raise ValueError(f"trial {trial}: test returned {p}. NaN or inf p-values would be counted as "
                         "'not significant' and make a broken test look calibrated. Fix the test or the data.")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"trial {trial}: test returned {p}, which is not a p-value. "
                         "Is it returning a statistic (t, z, r) instead?")
    return p


# ---------------------------------------------------------------- null generators
# Each returns one series with NO effect in it. Call with a seed so results are reproducible.
# A seed can be an int or a list of ints (e.g. [base, trial, series]).
def ar1(n, phi=0.0, mean=0.0, sd=1.0, seed=0):
    """Stationary AR(1) noise: no trend, pulls back toward its mean. phi=0 is white noise;
    phi near 1 wanders slowly like real trend data. phi=1 gives a random walk (never pulls back)."""
    if not -1.0 <= phi <= 1.0:
        raise ValueError(f"phi must be in [-1, 1], got {phi}")
    if phi == 1.0:
        return mean + random_walk(n, sd=sd, seed=seed)
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0]
    s = math.sqrt(1 - phi ** 2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + s * e[t]
    return mean + sd * x


def random_walk(n, sd=1.0, seed=0):
    """Cumulative sum of noise. Two independent random walks often look correlated. They aren't."""
    return np.cumsum(np.random.default_rng(seed).normal(scale=sd, size=n))


def poisson_counts(n, rate=5.0, phi=0.0, sd_log=0.3, seed=0):
    """Daily counts (mentions, sales) with an autocorrelated rate and no trend or spike."""
    rng = np.random.default_rng(seed)
    lam = np.exp(np.log(rate) + ar1(n, phi, sd=sd_log, seed=[1, *np.atleast_1d(seed).tolist()]))
    return rng.poisson(lam)


def permuted(data, seed=0):
    """Your own values in random order. Keeps the distribution, kills any order or link.
    Warning: also kills autocorrelation, so a test can look calibrated here and fail on ar1."""
    return np.random.default_rng(seed).permutation(np.asarray(data))


# ---------------------------------------------------------------- surrogates (for building a FIXED test)
# Each returns an array (n_surrogates, len(x)): copies of x with any link to other data destroyed,
# but with x's autocorrelation kept. Use them inside your test, then check the test with sweep + power.
def phase_randomize(x, n_surrogates=199, seed=0):
    """Same power spectrum (so same autocorrelation), random phases. Good default for stationary series.
    Weakness: if x contains a real step or trend, the surrogates inherit it, which can kill power."""
    x = np.asarray(x, dtype=float)
    n = x.size
    f = np.fft.rfft(x - x.mean())
    ph = np.random.default_rng(seed).uniform(0, 2 * np.pi, size=(n_surrogates, f.size))
    ph[:, 0] = 0
    if n % 2 == 0:
        ph[:, -1] = 0  # Nyquist term must stay real
    return x.mean() + np.fft.irfft(np.abs(f) * np.exp(1j * ph), n=n)


def block_bootstrap(x, block, n_surrogates=199, seed=0):
    """Circular block bootstrap: glue together random runs of `block` consecutive values.
    Keeps autocorrelation shorter than the block. Pick block well above the correlation time
    (for AR(1), about (1 + phi) / (1 - phi) points)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if not 1 <= block <= n:
        raise ValueError(f"block must be between 1 and {n}")
    rng = np.random.default_rng(seed)
    k = -(-n // block)
    starts = rng.integers(0, n, size=(n_surrogates, k))
    idx = (starts[:, :, None] + np.arange(block)) % n
    return x[idx.reshape(n_surrogates, -1)[:, :n]]


def circular_shift(x, n_surrogates=199, min_shift=None, seed=0):
    """Rotate x by a random offset. Keeps x almost exactly as it is and only breaks its alignment with the
    other series. Weakness: the wrap-around adds a jump, so with strong autocorrelation on short data it
    runs hot (measured: 12% false positives at phi 0.9, n=104, best-of-9-lags test). Check it with sweep."""
    x = np.asarray(x, dtype=float)
    n = x.size
    m = max(1, n // 10) if min_shift is None else min_shift
    if 2 * m >= n:
        raise ValueError("series too short for that min_shift")
    shifts = np.random.default_rng(seed).integers(m, n - m + 1, size=n_surrogates)
    return np.stack([np.roll(x, s) for s in shifts])


def ar1_surrogates(x, n_surrogates=199, seed=0):
    """Fit mean, phi and sd of x, then simulate fresh AR(1) series. Parametric, simple.
    Weakness: a real step or trend in x inflates the fitted phi, which can kill power."""
    x = np.asarray(x, dtype=float)
    n = x.size
    mu, sd, phi = x.mean(), x.std(), estimate_phi(x)
    e = np.random.default_rng(seed).normal(size=(n_surrogates, n))
    s = np.empty((n_surrogates, n))
    s[:, 0] = e[:, 0]
    c = math.sqrt(1 - phi ** 2)
    for t in range(1, n):
        s[:, t] = phi * s[:, t - 1] + c * e[:, t]
    return mu + sd * s


def prewhiten(x, phi=None):
    """Remove AR(1) autocorrelation: x[t] - phi * x[t-1], with phi fitted on x unless given.
    After prewhitening both series, an ordinary shuffle is roughly valid again. One point shorter."""
    x = np.asarray(x, dtype=float)
    phi = estimate_phi(x) if phi is None else phi
    return x[1:] - phi * x[:-1]


def surrogate_p(stat_fn, x, surrogates, *others, tail="greater"):
    """p-value of stat_fn(x, *others) against the same statistic on every surrogate of x.
    Put ANY search (best lag, biggest jump, best threshold) inside stat_fn, so the null searches too.
    tail: 'greater' (big is extreme), 'less', or 'abs' (compare |stat|)."""
    obs = float(stat_fn(np.asarray(x), *others))
    null = np.array([float(stat_fn(s, *others)) for s in surrogates])
    if tail == "abs":
        obs, null = abs(obs), np.abs(null)
    elif tail == "less":
        obs, null = -obs, -null
    elif tail != "greater":
        raise ValueError("tail must be 'greater', 'less' or 'abs'")
    return (int(np.sum(null >= obs)) + 1) / (null.size + 1)


# ---------------------------------------------------------------- checks
def baseline_check(y_true, y_pred, claimed_beats=None, positive=None, claim="accuracy", alpha=0.05,
                   name="claim"):
    """Does the model beat doing nothing?
    claim='accuracy'  ("X% accurate", "beats baseline"): accuracy must beat always guessing the majority
                      class, and the model must catch at least one positive.
    claim='detection' ("catches N% of fraud", "the flags are useful"): flagging everyone catches 100%, so
                      recall alone proves nothing. The flags must be richer in positives than the base rate
                      (lift > 1, one-sided binomial test at alpha).
    positive: the label that counts as a hit. Default: the rarer class in y_true."""
    if claim not in ("accuracy", "detection"):
        raise ValueError("claim must be 'accuracy' or 'detection'")
    y_true, y_pred = list(y_true), list(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true has {len(y_true)} labels, y_pred has {len(y_pred)}")
    n = len(y_true)
    if n == 0:
        raise ValueError("no labels")
    counts = Counter(y_true)
    if positive is None:
        positive = min(counts, key=lambda k: (counts[k], str(k)))
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / n
    majority_label, majority_n = counts.most_common(1)[0]
    baselines = {f"always '{majority_label}' (majority)": majority_n / n}
    if positive != majority_label:
        baselines[f"always '{positive}'"] = counts.get(positive, 0) / n
    best_name, best = max(baselines.items(), key=lambda kv: kv[1])

    n_pos = counts.get(positive, 0)
    flagged = sum(p == positive for p in y_pred)
    tp = sum(t == positive and p == positive for t, p in zip(y_true, y_pred))
    base_rate = n_pos / n
    recall = tp / n_pos if n_pos else None
    precision = tp / flagged if flagged else None
    lift = precision / base_rate if (precision is not None and base_rate) else None
    p_lift = binom_sf(tp, flagged, base_rate) if flagged and base_rate else None

    reasons, notes = [], []
    if claim == "accuracy":
        beats = acc > best and (recall is None or recall > 0)
        if acc <= best:
            reasons.append(f"accuracy {acc:.1%} does not beat {best_name} at {best:.1%}")
        if recall == 0:
            reasons.append(f"caught 0 of {n_pos} '{positive}' cases (recall 0)")
        if lift is not None and p_lift < alpha and not beats:
            notes.append(f"but its '{positive}' flags are {lift:.1f}x richer than the base rate "
                         f"({precision:.0%} vs {base_rate:.0%}, p={p_lift:.2g}). As a ranking or call list it may "
                         "still be useful: rerun with claim='detection' and report precision/recall, not accuracy")
    else:
        beats = tp > 0 and p_lift is not None and p_lift < alpha
        if not flagged:
            reasons.append(f"flagged nothing as '{positive}'")
        elif not beats:
            reasons.append(f"flags are {precision:.0%} '{positive}' vs {base_rate:.0%} base rate "
                           f"(lift {lift:.1f}x, p={p_lift:.2g}): no better than flagging at random")
        notes.append(f"flagging everyone would catch 100% at {base_rate:.0%} precision, so quote recall "
                     "together with precision")
    if n < 30:
        reasons.append(f"small sample: n={n}, one prediction moves accuracy by {1/n:.1%}")
    if n_pos and n_pos < 30:
        notes.append(f"only {n_pos} '{positive}' cases: recall CI {_ci_str(tp, n_pos)}"
                     + (f", precision CI {_ci_str(tp, flagged)}" if flagged else ""))

    overclaim = bool(claimed_beats) and not beats
    rnd = lambda v, d=3: None if v is None else round(v, d)
    return {
        "check": "baseline", "claim": claim, "name": name, "n": n, "positive": positive,
        "accuracy": round(acc, 3), "accuracy_ci95": [round(v, 3) for v in wilson(round(acc * n), n)],
        "baselines": {k: round(v, 3) for k, v in baselines.items()},
        "confusion": {"tp": tp, "fp": flagged - tp, "fn": n_pos - tp, "tn": n - n_pos - flagged + tp},
        "recall": rnd(recall), "precision": rnd(precision), "base_rate": round(base_rate, 3),
        "lift": rnd(lift, 2), "lift_p": None if p_lift is None else float(f"{p_lift:.3g}"),
        "beats_baselines": beats, "claimed_beats": claimed_beats,
        "verdict": "OVERCLAIM" if overclaim else ("PASS" if beats else "FAIL (and not claimed)"),
        "reasons": reasons, "notes": notes,
    }


def _ci_str(k, n):
    lo, hi = wilson(k, n)
    return f"{lo:.0%}-{hi:.0%}"


def null_calibration(test_fn, null_gen, n_trials=200, alpha=0.05, name="test"):
    """Run test_fn on data where the null is true by construction.
    null_gen(i) returns a tuple of arguments for test_fn; test_fn returns a p-value in [0, 1].
    Verdicts (95% Wilson CI of the false-positive rate vs alpha):
      MISCALIBRATED  CI entirely above alpha: false alarms. The result can't be trusted.
      CONSERVATIVE   CI entirely below alpha: safe from false alarms but wasting power, or not really testing.
      CALIBRATED     CI includes alpha. With n_trials=200 this only rules out rates above the CI top
                     (about 9% for a true 5%); use more trials to rule out milder inflation."""
    ps = [_as_p(test_fn(*null_gen(i)), i) for i in range(n_trials)]
    k = sum(p < alpha for p in ps)
    fpr = k / n_trials
    lo, hi = (round(v, 3) for v in wilson(k, n_trials))
    reasons, notes = [], []
    if lo > alpha:
        verdict = "MISCALIBRATED"
        reasons.append(f"says 'significant' {fpr:.1%} of the time on data with no effect "
                       f"(should be about {alpha:.0%}); 95% CI {lo:.1%}-{hi:.1%}")
    elif hi < alpha:
        verdict = "CONSERVATIVE"
        notes.append(f"says 'significant' only {fpr:.1%} of the time (CI {lo:.1%}-{hi:.1%}, below {alpha:.0%}). "
                     "No false-alarm problem, but check power: a test that rarely fires may never find anything")
    else:
        verdict = "CALIBRATED"
        notes.append(f"{n_trials} trials rule out false-positive rates above {hi:.1%}, not milder inflation")
    if len(set(ps)) == 1:
        notes.append(f"every trial returned p={ps[0]}: the test may not be looking at the data")
    return {
        "check": "null_calibration", "name": name, "trials": n_trials, "alpha": alpha,
        "false_positives": k, "false_positive_rate": round(fpr, 3), "ci95": [lo, hi],
        "verdict": verdict, "reasons": reasons, "notes": notes,
    }


def power(test_fn, signal_gen, n_trials=100, alpha=0.05, min_power=0.5, name="test"):
    """Fraction of datasets WITH a real, planted effect that the test detects.
    Verdict POWER OK if the point estimate reaches min_power, else LOW POWER."""
    ps = [_as_p(test_fn(*signal_gen(i)), i) for i in range(n_trials)]
    k = sum(p < alpha for p in ps)
    lo, hi = wilson(k, n_trials)
    pw = k / n_trials
    ok = pw >= min_power
    return {"check": "power", "name": name, "trials": n_trials, "detected": k, "alpha": alpha,
            "power": round(pw, 3), "ci95": [round(lo, 3), round(hi, 3)], "min_power": min_power,
            "verdict": "POWER OK" if ok else "LOW POWER",
            "reasons": [] if ok else [f"finds the planted effect only {pw:.0%} of the time "
                                      f"(want at least {min_power:.0%}): a 'no effect' result here means little"],
            "notes": []}


def sweep(test_fn, n, phis=(0.0, 0.5, 0.9), series=1, make_args=None, n_trials=200, alpha=0.05,
          name="test", seed=0):
    """null_calibration at several autocorrelation levels. Most broken nulls look fine at phi=0.
    series: how many independent noise series each trial gets (2 for correlation / lead-lag tests).
    make_args(*series) turns them into test_fn's arguments (default: pass them straight through).
    phi=1.0 means random walks."""
    make_args = make_args or (lambda *s: s)
    cards = []
    for j, phi in enumerate(phis):
        def gen(i, phi=phi, j=j):
            return make_args(*(ar1(n, phi, seed=[seed, j, i, s]) for s in range(series)))
        cards.append(null_calibration(test_fn, gen, n_trials, alpha, name=f"{name} (phi={phi})"))
    return cards


def eta(test_fn, args, n_calls):
    """Time one call of test_fn(*args) and return the estimated seconds for n_calls."""
    t = time.perf_counter()
    test_fn(*args)
    return (time.perf_counter() - t) * n_calls


# ---------------------------------------------------------------- output
def save(card, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(card, indent=1, default=lambda o: o.item()))
    return card


def _result_line(c):
    if c["check"] == "baseline":
        best = max(c["baselines"].values())
        res = f"accuracy {c['accuracy']:.1%} vs best do-nothing {best:.1%} (n={c['n']})"
        if c.get("lift") is not None:
            res += (f"; flags '{c['positive']}': precision {c['precision']:.0%} vs base rate {c['base_rate']:.0%}"
                    f" (lift {c['lift']:.1f}x), recall {c['recall']:.0%}")
        return res
    lo, hi = c["ci95"]
    if c["check"] == "null_calibration":
        return f"{c['false_positive_rate']:.1%} 'significant' on pure noise (CI {lo:.1%}-{hi:.1%}, {c['trials']} trials)"
    return f"finds the planted effect {c['power']:.0%} of the time (CI {lo:.0%}-{hi:.0%}, {c['trials']} trials)"


def receipt(cards, title="skeptic receipt"):
    """Markdown receipt: one line per check, then the reasons and notes. Paste it next to the claim."""
    cards = [cards] if isinstance(cards, dict) else list(cards)
    lines = [f"## {title}", "", "| Check | Result | Verdict |", "|---|---|---|"]
    for c in cards:
        lines.append(f"| {c['name'].replace('|', '/')} | {_result_line(c)} | **{c.get('verdict', 'POWER')}** |")
    reasons = [f"- {c['name']}: {r}" for c in cards for r in c.get("reasons", [])]
    notes = [f"- {c['name']}: {r}" for c in cards for r in c.get("notes", [])]
    if reasons:
        lines += ["", "Why:"] + reasons
    if notes:
        lines += ["", "Notes:"] + notes
    lines += ["", f"_skeptic {__version__}. Calibrated means the 95% CI of the noise hit rate includes alpha._"]
    return "\n".join(lines)


# ---------------------------------------------------------------- CLI
def _read_labels(path):
    return [tok for tok in Path(path).read_text().replace(",", " ").split()]


def _opt(argv, flag, default=None):
    if flag in argv:
        i = argv.index(flag)
        if i + 1 >= len(argv):
            raise SystemExit(f"{flag} needs a value")
        val = argv[i + 1]
        del argv[i:i + 2]
        return val
    return default


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd = argv.pop(0)
    if cmd == "baseline":
        claim = _opt(argv, "--claim", "accuracy")
        positive = _opt(argv, "--positive")
        claimed = "--claimed" in argv
        as_json = "--json" in argv
        files = [a for a in argv if not a.startswith("--")]
        if len(files) != 2:
            print("usage: skeptic.py baseline y_true.txt y_pred.txt [--claimed] [--claim accuracy|detection] "
                  "[--positive LABEL] [--json]")
            return 2
        yt, yp = _read_labels(files[0]), _read_labels(files[1])
        card = baseline_check(yt, yp, claimed_beats=claimed, positive=positive, claim=claim,
                              name=Path(files[1]).stem)
        print(json.dumps(card, indent=1) if as_json else receipt(card))
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
