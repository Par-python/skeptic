"""EXPLORATORY, not pre-registered: is near-zero power a bug, or was the step too small?"""
import numpy as np, demo
for phi in (0.0, 0.5):
    for size in (1.5, 3.0, 5.0):
        row = []
        for name in list(demo.TESTS)[2:]:
            k = sum(demo.TESTS[name](demo.with_step(9_000_000 + i, phi, size), i) < 0.05 for i in range(60))
            row.append(f"{name.split(',')[1].strip():18s} {k/60:4.0%}")
        # how big is the planted jump vs the biggest noise jump?
        z_true = np.mean([demo.jumps(demo.with_step(9_000_000 + i, phi, size))[0].max() for i in range(60)])
        z_null = np.mean([demo.jumps(demo.noise(9_500_000 + i, phi))[0].max() for i in range(60)])
        print(f"phi={phi} step={size}sd | max-z with step {z_true:.2f} vs noise {z_null:.2f} | " + " | ".join(row))
