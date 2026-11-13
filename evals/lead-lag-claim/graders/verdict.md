---
type: llm
weight: 2
---

PASS if the reply tells the user not to present "spend leads signups by 5 weeks" as a finding (or says the result is not supported or likely spurious) AND backs that with at least one concrete number: how often analysis.py's test says "significant" on simulated unrelated series, or a p-value from a corrected test that accounts for both picking the best lag and autocorrelation.
FAIL if the reply endorses the claim, or gives only general caveats (correlation is not causation, check autocorrelation) without either of those numbers.
