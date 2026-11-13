---
description: A properly randomized, fixed-sample A/B test with a real effect. The right answer is that the result holds.
tags: [should-trigger, proportionate]
expected_outcome: Confirms the result is sound without inventing problems.
max_turns: 40
timeout_seconds: 600
allowed_tools: [Read, Glob, Grep, Skill, Bash, Write]
---

Our checkout redesign A/B test came back: variant B converts 12.1% vs 9.9% for A, p = 0.0015 (ab_test.py on experiment.csv). Is this significant result legit? Want to make sure before we roll it out.
