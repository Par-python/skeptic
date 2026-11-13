"""Runs PREREG2.md: V4, the AR(1) null fitted after removing the best step."""
import json
import sys
from pathlib import Path
import numpy as np
import demo
from demo import N, W, B, LO, HI

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import skeptic


def raw_jumps(X):
    X = np.atleast_2d(X)
    cs = np.concatenate([np.zeros((X.shape[0], 1)), np.cumsum(X, axis=1)], axis=1)
    c = np.arange(LO, HI + 1)
    return np.abs((cs[:, c + W] - cs[:, c]) / W - (cs[:, c] - cs[:, c - W]) / W)


def v4(x, seed):
    rng = np.random.default_rng(seed)
    j = raw_jumps(x)[0]
    c = LO + int(np.argmax(j))
    r = x.copy()
    r[c:] -= x[c:].mean() - x[:c].mean()          # remove the best level shift
    sd = r.std()
    z = (r - r.mean()) / sd
    phi = float(np.clip(np.corrcoef(z[:-1], z[1:])[0, 1], -0.99, 0.99))
    e = rng.normal(size=(B, N))
    S = np.empty((B, N))
    S[:, 0] = e[:, 0]
    for t in range(1, N):
        S[:, t] = phi * S[:, t - 1] + np.sqrt(1 - phi ** 2) * e[:, t]
    surr = raw_jumps(sd * S).max(axis=1)
    return (np.sum(surr >= j.max()) + 1) / (B + 1)


out = {"calibration": {}, "power": {}}
for phi in (0.0, 0.5, 0.9):
    gen = lambda i, phi=phi: (demo.noise(20_000_000 + int(1e6 * phi) + i, phi), i)
    card = skeptic.null_calibration(v4, gen, n_trials=demo.SERIES, name=f"V4 | phi={phi}")
    out["calibration"][f"phi={phi}"] = card
    print(f"phi={phi}  V4  {card['false_positive_rate']:.1%}  CI {card['ci95'][0]:.0%}-{card['ci95'][1]:.0%}  {card['verdict']}")
for size in (1.0, 1.5):
    gen = lambda i, size=size: (demo.with_step(30_000_000 + i, 0.5, size), i)
    card = skeptic.power(v4, gen, n_trials=demo.POWER_SERIES, name=f"V4 | step {size} sd")
    out["power"][f"step={size}sd"] = card
    print(f"power V4 step {size} sd: {card['power']:.0%}  CI {card['ci95'][0]:.0%}-{card['ci95'][1]:.0%}")
ok_cal = all(c["verdict"] == "CALIBRATED" and c["false_positive_rate"] <= 0.075 for c in out["calibration"].values())
ok_pow = out["power"]["step=1.5sd"]["power"] >= 0.5
out["pass_rule"] = {"calibrated_all_phi": ok_cal, "power_ok": ok_pow, "passes": ok_cal and ok_pow}
print(out["pass_rule"])
Path(__file__).with_name("results2.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item()))
