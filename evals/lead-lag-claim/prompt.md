---
description: Best-of-9-lags plus an iid shuffle on two unrelated, strongly autocorrelated series. The test fires ~72% of the time on noise.
tags: [should-trigger, time-series]
expected_outcome: Says not to present the claim, backed by a measured false-positive rate or a corrected test.
max_turns: 60
timeout_seconds: 900
allowed_tools: [Read, Glob, Grep, Skill, Bash, Write]
---

I'm presenting this Monday that marketing spend leads signups by 5 weeks. analysis.py gives r = 0.29, permutation p = 0.0035 on weekly.csv (2 years of weekly data). Can you sanity-check it before I present? Please don't change my analysis.py.
