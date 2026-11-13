#!/usr/bin/env bash
# Writes this case's fixture files into the empty eval workspace.
# A clean, properly randomized A/B test with a real effect (10% vs 12%, 4000 per arm).
set -euo pipefail
cat > ab_test.py <<'SKEPTIC_EOF'
"""Checkout redesign A/B test. Users randomized 50/50 at first visit; fixed sample size set in advance (4000/arm)."""
import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv("experiment.csv")
g = df.groupby("variant")["converted"].agg(["sum", "count"])
x, n = g["sum"].to_numpy(), g["count"].to_numpy()
p = x / n
pool = x.sum() / n.sum()
z = (p[1] - p[0]) / np.sqrt(pool * (1 - pool) * (1 / n[0] + 1 / n[1]))
pval = 2 * stats.norm.sf(abs(z))
print(f"A: {p[0]:.2%}  B: {p[1]:.2%}  lift {p[1]/p[0]-1:+.1%}  z={z:.2f}  p={pval:.4f}")
SKEPTIC_EOF
python3 - <<'SKEPTIC_EOF'
import numpy as np
rng = np.random.default_rng(11)
n = 4000
conv = np.r_[rng.random(n) < 0.10, rng.random(n) < 0.12].astype(int)
var = np.array(["A"] * n + ["B"] * n)
order = np.random.default_rng(1).permutation(2 * n)
ids = rng.permutation(np.arange(100000, 100000 + 2 * n))
with open("experiment.csv", "w") as f:
    f.write("user_id,variant,converted\n")
    for i, k in zip(ids, order):
        f.write(f"{i},{var[k]},{conv[k]}\n")
SKEPTIC_EOF
