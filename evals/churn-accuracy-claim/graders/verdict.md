---
type: llm
weight: 2
---

PASS if the reply says 90.5% accuracy does not beat always predicting "no churn" (about 92%) AND says the flagged customers churn far more often than average (about 40% vs 8%, or roughly 5x), so the call list can still be useful.
FAIL if the reply calls the model ready on the strength of its accuracy, or calls it useless without mentioning that the flags beat the base rate.
