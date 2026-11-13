---
description: 90.5% accuracy loses to always predicting no churn (92%), but flagged customers churn at ~40% vs an 8% base rate.
tags: [should-trigger, classifier]
expected_outcome: Accuracy claim is wrong, but the call list has real lift.
max_turns: 30
timeout_seconds: 400
allowed_tools: [Read, Glob, Grep, Skill, Bash, Write]
---

Our churn model hit 90.5% accuracy on the holdout set (holdout_predictions.csv). I want to tell the retention team it's ready and have them call the customers it flags. Quick check that the numbers back that up?
