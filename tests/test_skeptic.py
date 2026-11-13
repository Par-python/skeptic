"""Tests for skeptic. Run: python -m pytest tests/  (numpy only; no scipy needed)."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills" / "skeptic" / "scripts"))
import skeptic  # noqa: E402


# ---------------------------------------------------------------- helpers used by several tests
def t_test_p(x):
    """Two-sided one-sample test of mean 0 (normal approx). Calibrated on white noise, broken on AR(1)."""
    x = np.asarray(x)
    z = x.mean() / (x.std(ddof=1) / math.sqrt(x.size))
    return math.erfc(abs(z) / math.sqrt(2))


def corr_p(x, y):
    """Pearson r with a normal-approx p. Calibrated for independent white noise only."""
    r = np.corrcoef(x, y)[0, 1]
    z = math.atanh(np.clip(r, -0.999999, 0.999999)) * math.sqrt(len(x) - 3)
    return math.erfc(abs(z) / math.sqrt(2))


# ---------------------------------------------------------------- stats helpers
def test_wilson_matches_known_values():
    lo, hi = skeptic.wilson(10, 200)
    assert lo == pytest.approx(0.0274, abs=1e-3) and hi == pytest.approx(0.0896, abs=1e-3)
    assert skeptic.wilson(0, 0) == (0.0, 1.0)


def test_binom_sf():
    assert skeptic.binom_sf(0, 10, 0.3) == 1.0
    assert skeptic.binom_sf(11, 10, 0.3) == 0.0
    assert skeptic.binom_sf(10, 10, 0.5) == pytest.approx(0.5 ** 10)
    assert skeptic.binom_sf(1, 1, 0.2) == pytest.approx(0.2)


# ---------------------------------------------------------------- generators
def test_ar1_is_stationary_and_phi_is_right():
    x = skeptic.ar1(20_000, 0.9, seed=1)
    assert skeptic.estimate_phi(x) == pytest.approx(0.9, abs=0.02)
    assert x.std() == pytest.approx(1.0, abs=0.15)


def test_ar1_phi_one_is_a_random_walk_not_a_constant():
    x = skeptic.ar1(500, 1.0, seed=3)
    assert x.std() > 1.0                      # v0.1 returned a constant here
    assert np.allclose(x, skeptic.random_walk(500, seed=3))


def test_ar1_negative_phi_and_list_seeds_work():
    x = skeptic.ar1(5000, -0.5, seed=[0, 1, 2])
    assert skeptic.estimate_phi(x) == pytest.approx(-0.5, abs=0.05)
    with pytest.raises(ValueError):
        skeptic.ar1(10, 1.5)


def test_generators_are_reproducible():
    assert np.array_equal(skeptic.ar1(50, 0.5, seed=7), skeptic.ar1(50, 0.5, seed=7))
    assert np.array_equal(skeptic.poisson_counts(50, seed=7), skeptic.poisson_counts(50, seed=7))


# ---------------------------------------------------------------- p-value validation
@pytest.mark.parametrize("bad, err", [(float("nan"), ValueError), (2.5, ValueError), (-0.1, ValueError),
                                      (True, TypeError), ("0.01", TypeError), (None, TypeError)])
def test_null_calibration_rejects_non_p_values(bad, err):
    with pytest.raises(err):
        skeptic.null_calibration(lambda x: bad, lambda i: (skeptic.ar1(20, seed=i),), n_trials=5)


def test_numpy_scalars_are_accepted():
    card = skeptic.null_calibration(lambda x: np.float64(0.5), lambda i: (skeptic.ar1(20, seed=i),), n_trials=5)
    assert card["false_positives"] == 0


# ---------------------------------------------------------------- calibration verdicts
def test_correct_test_is_calibrated_on_white_noise():
    card = skeptic.null_calibration(t_test_p, lambda i: (skeptic.ar1(100, 0.0, seed=i),))
    assert card["verdict"] == "CALIBRATED"
    assert card["ci95"][0] <= 0.05 <= card["ci95"][1]


def test_naive_test_is_miscalibrated_on_autocorrelated_noise():
    card = skeptic.null_calibration(t_test_p, lambda i: (skeptic.ar1(100, 0.9, seed=i),))
    assert card["verdict"] == "MISCALIBRATED"
    assert card["reasons"]


def test_test_that_never_fires_is_conservative_not_calibrated():
    card = skeptic.null_calibration(lambda x: 1.0, lambda i: (skeptic.ar1(20, seed=i),))
    assert card["verdict"] == "CONSERVATIVE"
    assert any("same" in n or "p=1.0" in n for n in card["notes"])


def test_sweep_single_series():
    cards = skeptic.sweep(t_test_p, n=100, n_trials=150)
    assert [c["verdict"] for c in cards] == ["CALIBRATED", "MISCALIBRATED", "MISCALIBRATED"]


def test_sweep_two_series_and_random_walks():
    cards = skeptic.sweep(corr_p, n=100, series=2, phis=(0.0, 1.0), n_trials=150)
    assert cards[0]["verdict"] == "CALIBRATED"
    assert cards[1]["verdict"] == "MISCALIBRATED"        # spurious correlation between random walks
    assert cards[1]["false_positive_rate"] > 0.5


def test_sweep_series_are_independent():
    seen = []
    skeptic.sweep(lambda x, y: (seen.append((x, y)), 0.5)[1], n=30, series=2, n_trials=3, phis=(0.5,))
    for x, y in seen:
        assert not np.array_equal(x, y)


def test_sweep_is_reproducible():
    a = skeptic.sweep(t_test_p, n=50, n_trials=50)
    b = skeptic.sweep(t_test_p, n=50, n_trials=50)
    assert [c["false_positives"] for c in a] == [c["false_positives"] for c in b]


# ---------------------------------------------------------------- power
def test_power_verdicts():
    strong = skeptic.power(t_test_p, lambda i: (skeptic.ar1(100, seed=i) + 0.5,), n_trials=50)
    weak = skeptic.power(t_test_p, lambda i: (skeptic.ar1(100, seed=i) + 0.02,), n_trials=50)
    assert strong["verdict"] == "POWER OK" and strong["power"] > 0.9
    assert weak["verdict"] == "LOW POWER" and weak["reasons"]


# ---------------------------------------------------------------- surrogates
@pytest.mark.parametrize("make", [
    lambda x: skeptic.phase_randomize(x, 300, seed=1),
    lambda x: skeptic.block_bootstrap(x, 20, 300, seed=1),
    lambda x: skeptic.ar1_surrogates(x, 300, seed=1),
])
def test_surrogates_keep_autocorrelation(make):
    x = skeptic.ar1(400, 0.9, seed=5)
    s = make(x)
    assert s.shape == (300, 400)
    assert np.mean([skeptic.estimate_phi(r) for r in s]) == pytest.approx(0.9, abs=0.07)


def test_circular_shift_keeps_values_and_breaks_alignment():
    x = np.arange(100.0)
    s = skeptic.circular_shift(x, 50, seed=2)
    assert all(sorted(r) == sorted(x) for r in s)
    assert not any(np.array_equal(r, x) for r in s)


def test_phase_randomize_odd_length():
    s = skeptic.phase_randomize(skeptic.ar1(101, 0.5, seed=0), 10)
    assert s.shape == (10, 101) and np.all(np.isfinite(s))


def test_block_bootstrap_rejects_bad_block():
    with pytest.raises(ValueError):
        skeptic.block_bootstrap(np.zeros(10), 0)


def test_surrogate_p_tails_and_range():
    x = np.ones(20) * 5
    surr = np.zeros((99, 20))
    assert skeptic.surrogate_p(np.mean, x, surr) == pytest.approx(1 / 100)
    assert skeptic.surrogate_p(np.mean, x, surr, tail="less") == 1.0
    assert skeptic.surrogate_p(np.mean, -x, surr, tail="abs") == pytest.approx(1 / 100)
    with pytest.raises(ValueError):
        skeptic.surrogate_p(np.mean, x, surr, tail="both")


def _best_lag_r(x, y, max_lag=8):
    return max(abs(np.corrcoef(x[:len(x) - k], y[k:])[0, 1]) for k in range(max_lag + 1))


def _fixed_lead_lag_test(x, y, seed=0):
    """The recommended fix for 'pick the best lag, then test it': search inside every surrogate,
    with surrogates that keep autocorrelation."""
    return skeptic.surrogate_p(_best_lag_r, x, skeptic.phase_randomize(x, 99, seed=seed), y)


def _prewhitened_lead_lag_test(x, y, seed=0):
    a, b = skeptic.prewhiten(x), skeptic.prewhiten(y)
    rng = np.random.default_rng(seed)
    return skeptic.surrogate_p(_best_lag_r, a, np.stack([rng.permutation(a) for _ in range(99)]), b)


def test_selection_aware_surrogate_test_fixes_best_lag_bug_end_to_end():
    def naive(x, y):  # pick best lag, then iid-shuffle only at that lag
        k = int(np.argmax([abs(np.corrcoef(x[:len(x) - j], y[j:])[0, 1]) for j in range(9)]))
        obs = abs(np.corrcoef(x[:len(x) - k], y[k:])[0, 1])
        rng = np.random.default_rng(0)
        null = [abs(np.corrcoef(rng.permutation(x)[:len(x) - k], y[k:])[0, 1]) for _ in range(99)]
        return (sum(n >= obs for n in null) + 1) / 100

    bad = skeptic.sweep(naive, n=104, series=2, phis=(0.9,), n_trials=100)[0]
    assert bad["verdict"] == "MISCALIBRATED"
    for fix in (_fixed_lead_lag_test, _prewhitened_lead_lag_test):
        good = skeptic.sweep(fix, n=104, series=2, phis=(0.0, 0.9), n_trials=100)
        assert all(c["verdict"] != "MISCALIBRATED" for c in good), fix.__name__


def test_prewhiten_removes_autocorrelation():
    x = skeptic.ar1(5000, 0.9, seed=4)
    assert abs(skeptic.estimate_phi(skeptic.prewhiten(x))) < 0.05
    assert skeptic.prewhiten(x).size == 4999


# ---------------------------------------------------------------- baseline_check
def test_accuracy_overclaim_from_quickstart():
    y_true = [1] * 3 + [0] * 16
    y_pred = [0] * 3 + [1] + [0] * 15
    card = skeptic.baseline_check(y_true, y_pred, claimed_beats=True)
    assert card["verdict"] == "OVERCLAIM"
    assert card["positive"] == 1 and card["recall"] == 0


def _churn():
    yt = [1] * 48 + [0] * 552
    yp = [1] * 17 + [0] * 31 + [1] * 26 + [0] * 526
    return yt, yp


def test_useful_detector_below_majority_accuracy_gets_a_note_not_just_a_dismissal():
    card = skeptic.baseline_check(*_churn(), claimed_beats=True)
    assert card["verdict"] == "OVERCLAIM"                       # "90.5% accurate" really is an overclaim
    assert card["lift"] == pytest.approx(4.94, abs=0.01)
    assert any("claim='detection'" in n for n in card["notes"])


def test_detection_claim_passes_when_flags_beat_base_rate():
    card = skeptic.baseline_check(*_churn(), claimed_beats=True, claim="detection")
    assert card["verdict"] == "PASS"
    assert card["precision"] == pytest.approx(17 / 43, abs=1e-3)
    assert card["recall"] == pytest.approx(17 / 48, abs=1e-3)
    assert card["confusion"] == {"tp": 17, "fp": 26, "fn": 31, "tn": 526}


def test_detection_claim_fails_for_random_flags():
    rng = np.random.default_rng(0)
    yt = (rng.random(2000) < 0.05).astype(int).tolist()
    yp = (rng.random(2000) < 0.5).astype(int).tolist()      # flags half at random: high recall, no lift
    card = skeptic.baseline_check(yt, yp, claimed_beats=True, claim="detection")
    assert card["recall"] > 0.4
    assert card["verdict"] == "OVERCLAIM"


def test_baseline_input_errors():
    with pytest.raises(ValueError):
        skeptic.baseline_check([0, 1], [0])
    with pytest.raises(ValueError):
        skeptic.baseline_check([0], [0], claim="recall")


def test_baseline_multiclass_and_strings():
    card = skeptic.baseline_check(list("aabbbc"), list("aabbbc"), positive="c")
    assert card["verdict"] == "PASS" and card["accuracy"] == 1.0


# ---------------------------------------------------------------- receipt, save, CLI
def test_receipt_renders_every_card_type(tmp_path):
    cards = [skeptic.baseline_check(*_churn(), claimed_beats=True),
             skeptic.null_calibration(lambda x: 1.0, lambda i: (skeptic.ar1(20, seed=i),), n_trials=200),
             skeptic.power(t_test_p, lambda i: (skeptic.ar1(50, seed=i) + 1,), n_trials=20)]
    text = skeptic.receipt(cards)
    for word in ("OVERCLAIM", "CONSERVATIVE", "POWER OK", "lift", "Notes:"):
        assert word in text
    p = tmp_path / "c.json"
    skeptic.save(cards[0], p)
    assert json.loads(p.read_text())["verdict"] == "OVERCLAIM"


def test_receipt_handles_no_positives():
    card = skeptic.baseline_check([0, 0, 0], [0, 0, 0], positive=1)
    assert "accuracy" in skeptic.receipt(card)


def test_cli_baseline(tmp_path, capsys):
    yt, yp = _churn()
    (tmp_path / "t.txt").write_text(" ".join(map(str, yt)))
    (tmp_path / "p.txt").write_text(" ".join(map(str, yp)))
    args = [str(tmp_path / "t.txt"), str(tmp_path / "p.txt"), "--claimed"]
    assert skeptic.main(["baseline", *args]) == 1
    assert skeptic.main(["baseline", *args, "--claim", "detection"]) == 0
    capsys.readouterr()
    assert skeptic.main(["baseline", *args, "--json", "--positive", "1"]) == 1
    assert json.loads(capsys.readouterr().out)["positive"] == "1"
    assert skeptic.main(["baseline", args[0]]) == 2


def test_eta_returns_seconds():
    assert 0 <= skeptic.eta(lambda: None, (), 10) < 1
