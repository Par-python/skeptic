"""Public stand-in: 'pick the biggest jump, then test it as if you hadn't looked.'
Written from scratch. Runs PREREG.md. Usage: python demo.py"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import skeptic

N, W, B, BLOCK = 104, 12, 199, 4
SERIES, POWER_SERIES = 200, 100
LO, HI = W, N - W  # candidate changepoints


# ---------- data
def ar1(rng, n, phi, sd=10.0, mu=50.0):
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + np.sqrt(1 - phi ** 2) * e[t]
    return mu + sd * x


def noise(seed, phi):
    return ar1(np.random.default_rng(seed), N, phi)


def with_step(seed, phi, size_sd):
    rng = np.random.default_rng(seed)
    x = ar1(rng, N, phi)
    at = int(rng.integers(24, 81))
    x[at:] += size_sd * 10.0
    return x


# ---------- detector and statistic (vectorized over rows)
def jumps(X):
    """|mean(after W) - mean(before W)| / sd for every candidate c. X: (rows, N)."""
    X = np.atleast_2d(X)
    cs = np.concatenate([np.zeros((X.shape[0], 1)), np.cumsum(X, axis=1)], axis=1)
    c = np.arange(LO, HI + 1)
    after = (cs[:, c + W] - cs[:, c]) / W
    before = (cs[:, c] - cs[:, c - W]) / W
    return np.abs(after - before) / X.std(axis=1, keepdims=True)


def detect(x):
    return LO + int(np.argmax(jumps(x)[0]))


# ---------- surrogates: (B, N) arrays
def block_shuffle(x, rng, block=BLOCK):
    blocks = x.reshape(-1, block)
    return np.stack([blocks[rng.permutation(len(blocks))].ravel() for _ in range(B)])


def phase_randomize(x, rng):
    f = np.fft.rfft(x - x.mean())
    ph = rng.uniform(0, 2 * np.pi, size=(B, f.size))
    ph[:, 0] = 0
    ph[:, -1] = 0  # N even: keep Nyquist real
    return x.mean() + np.fft.irfft(np.abs(f) * np.exp(1j * ph), n=N)


def ar1_bootstrap(x, rng):
    mu, sd = x.mean(), x.std()
    z = (x - mu) / sd
    phi = float(np.clip(np.corrcoef(z[:-1], z[1:])[0, 1], -0.99, 0.99))
    e = rng.normal(size=(B, N))
    S = np.empty((B, N))
    S[:, 0] = e[:, 0]
    for t in range(1, N):
        S[:, t] = phi * S[:, t - 1] + np.sqrt(1 - phi ** 2) * e[:, t]
    return mu + sd * S


def p_value(obs, surr):
    return (np.sum(surr >= obs) + 1) / (len(surr) + 1)


# ---------- the tests (each returns a p-value)
def v0_naive(x, seed):
    """THE BUG: choose c on the real data, test only at c."""
    c = detect(x)
    surr = jumps(block_shuffle(x, np.random.default_rng(seed)))[:, c - LO]
    return p_value(jumps(x)[0, c - LO], surr)


def c1_random_index(x, seed):
    """Control: same null, at an index nobody chose."""
    rng = np.random.default_rng(seed)
    c = int(rng.integers(LO, HI + 1))
    surr = jumps(block_shuffle(x, rng))[:, c - LO]
    return p_value(jumps(x)[0, c - LO], surr)


def selection_aware(make_surr):
    def test(x, seed):
        surr = jumps(make_surr(x, np.random.default_rng(seed))).max(axis=1)  # search again in every surrogate
        return p_value(jumps(x)[0].max(), surr)
    return test


TESTS = {
    "V0 naive (the bug)": v0_naive,
    "C1 random index (control)": c1_random_index,
    "V1 selection-aware, block shuffle": selection_aware(block_shuffle),
    "V2 selection-aware, phase-randomized": selection_aware(phase_randomize),
    "V3 selection-aware, AR(1) bootstrap": selection_aware(ar1_bootstrap),
}


def main():
    out = {"calibration": {}, "power": {}}
    for phi in (0.0, 0.5, 0.9):
        for name, fn in TESTS.items():
            gen = lambda i, phi=phi: (noise(int(1e6 * phi) + i, phi), i)
            card = skeptic.null_calibration(fn, gen, n_trials=SERIES, name=f"{name} | phi={phi}")
            out["calibration"].setdefault(name, {})[f"phi={phi}"] = card
            print(f"phi={phi}  {name:40s} {card['false_positive_rate']:.1%}  "
                  f"CI {card['ci95'][0]:.0%}-{card['ci95'][1]:.0%}  {card['verdict']}", flush=True)
    for name in list(TESTS)[2:]:
        for size in (1.0, 1.5):
            gen = lambda i, size=size: (with_step(7_000_000 + i, 0.5, size), i)
            card = skeptic.power(TESTS[name], gen, n_trials=POWER_SERIES, name=f"{name} | step {size} sd")
            out["power"].setdefault(name, {})[f"step={size}sd"] = card
            print(f"power  {name:40s} step {size} sd: {card['power']:.0%}", flush=True)
    # pass rule, exactly as pre-registered
    verdicts = {}
    for name in list(TESTS)[2:]:
        cal = out["calibration"][name].values()
        ok_cal = all(c["ci95"][0] <= 0.05 and c["false_positive_rate"] <= 0.075 for c in cal)
        ok_pow = out["power"][name]["step=1.5sd"]["power"] >= 0.5
        verdicts[name] = {"calibrated_all_phi": ok_cal, "power_ok": ok_pow, "passes": ok_cal and ok_pow}
    out["pass_rule"] = verdicts
    print(json.dumps(verdicts, indent=1))
    Path(__file__).with_name("results.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item()))


if __name__ == "__main__":
    main()
