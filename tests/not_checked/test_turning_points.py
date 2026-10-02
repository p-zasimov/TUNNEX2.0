# --- Test: test_turning_points. Unit tests for .\qmt_computations\core\turning_points.py
# Run with: pytest test_turning_points.py -v ---


# --- Modules ---
import pytest # type: ignore
from unittest.mock import MagicMock, patch

import math


# --- Module to test ---
from tunnex_2.qmt_computations.core.turning_points import find_turning_points # type: ignore


# --- Helpers and shared data ---
FUNC_PATH = "tunnex_2.qmt_computations.core.turning_points"


def _make_potential(irc_min=-1.0, irc_max=1.0, irc_ts=0.0,
                     energy_reagent=0.0, energy_product=0.0, energy_ts=1.0):
    """ Should build a MagicMock potential that returns fixed energies at reagent/product/TS x-values """
    potential = MagicMock()
    potential.irc_x_min = irc_min
    potential.irc_x_max = irc_max
    potential.irc_x_ts = irc_ts

    energy_map = {irc_min: energy_reagent, irc_max: energy_product, irc_ts: energy_ts}

    def _side_effect(x):
        return energy_map[x]

    potential.side_effect = _side_effect
    return potential


def _make_fake_brentq(left_value, right_value, irc_min=-1.0, irc_max=1.0, irc_ts=0.0):
    """ Should build a fake brentq function returning a fixed root depending on which interval it was given """
    def _fake_brentq(func, a, b, **kwargs):
        if (a, b) == (irc_min, irc_ts):
            return left_value
        if (a, b) == (irc_ts, irc_max):
            return right_value
        raise AssertionError(f"Unexpected brentq interval: ({a}, {b})")
    return _fake_brentq


# ===========================================================================
# --- find_turning_points - input validation ---
# ===========================================================================
class TestFindTurningPointsValidation:

    def test_raises_when_level_energy_below_reagent_energy(self):
        """ Verify that a level_energy below the reagent's energy raises a ValueError """
        potential = _make_potential(energy_reagent=0.0)
        with pytest.raises(ValueError):
            find_turning_points(-0.5, potential)


# ===========================================================================
# --- find_turning_points - exact barrier match ---
# ===========================================================================
class TestFindTurningPointsExactMatch:

    def test_returns_exact_match_when_level_energy_equals_ts_energy(self):
        """ Verify that a level_energy exactly matching the TS energy returns the EXACT_MATCH result """
        potential = _make_potential(energy_reagent=0.0, energy_product=0.0, energy_ts=1.0)
        result = find_turning_points(1.0, potential)
        expected = {"left": None, "right": None, "regime": "OVER THE BARRIER", "OVER_STATUS": "EXACT_MATCH"}
        assert result == expected, f"expected {expected}, got {result}"

    def test_returns_exact_match_within_tolerance(self):
        """ Verify that a level_energy within tol of the TS energy still counts as an exact match """
        potential = _make_potential(energy_reagent=0.0, energy_product=0.0, energy_ts=1.0)
        result = find_turning_points(1.0 + 1e-13, potential, tol=1e-12)
        assert result["OVER_STATUS"] == "EXACT_MATCH", f"expected {'EXACT_MATCH'}, got {result['OVER_STATUS']}"


# ===========================================================================
# --- find_turning_points - below-the-barrier regimes (NORMAL / ENDOTHERMIC) ---
# ===========================================================================
class TestFindTurningPointsBelowBarrier:

    def test_normal_regime_returns_two_turning_points(self):
        """ Verify that a level_energy at or above the product's energy yields both turning points, regime NORMAL """
        potential = _make_potential(energy_reagent=0.0, energy_product=0.2, energy_ts=1.0)
        fake_brentq = _make_fake_brentq(left_value=-0.3, right_value=0.6)
        with patch(FUNC_PATH + ".brentq", side_effect=fake_brentq):
            result = find_turning_points(0.5, potential)
        expected = {"left": -0.3, "right": 0.6, "regime": "NORMAL"}
        assert result == expected, f"expected {expected}, got {result}"

    def test_endothermic_regime_returns_only_left_turning_point(self):
        """ Verify that a level_energy below the product's energy yields only a left turning point, regime ENDOTHERMIC """
        potential = _make_potential(energy_reagent=0.0, energy_product=0.6, energy_ts=1.0)
        fake_brentq = _make_fake_brentq(left_value=-0.7, right_value=None)
        with patch(FUNC_PATH + ".brentq", side_effect=fake_brentq) as mock_brentq:
            result = find_turning_points(0.3, potential)
        expected = {"left": -0.7, "right": None, "regime": "ENDOTHERMIC REACTION"}
        assert result == expected, f"expected {expected}, got {result}"
        assert mock_brentq.call_count == 1, f"expected {1}, got {mock_brentq.call_count}"


# ===========================================================================
# --- find_turning_points - over-the-barrier, QUASI_NORMAL ---
# ===========================================================================
class TestFindTurningPointsQuasiNormal:

    def test_returns_quasi_normal_result_with_expected_fields(self):
        """ Verify that a level above TS with corrected energy still >= product's energy gives a QUASI_NORMAL result """
        potential = _make_potential(energy_reagent=0.0, energy_product=0.2, energy_ts=1.0)
        fake_brentq = _make_fake_brentq(left_value=-0.2, right_value=0.4)
        with patch(FUNC_PATH + ".brentq", side_effect=fake_brentq):
            result = find_turning_points(1.5, potential)
        assert result["regime"] == "OVER THE BARRIER", f"expected {'OVER THE BARRIER'}, got {result['regime']}"
        assert result["OVER_STATUS"] == "QUASI_NORMAL", f"expected {'QUASI_NORMAL'}, got {result['OVER_STATUS']}"
        assert result["left"] is None, f"expected {None}, got {result['left']}"
        assert result["right"] is None, f"expected {None}, got {result['right']}"
        assert result["quasi_left"] == -0.2, f"expected {-0.2}, got {result['quasi_left']}"
        assert result["quasi_right"] == 0.4, f"expected {0.4}, got {result['quasi_right']}"

    def test_quasi_level_equals_the_corrected_level_energy(self):
        """ Verify that QUASI_LEVEL equals max(0, 2 * energy_ts - level_energy) """
        potential = _make_potential(energy_reagent=0.0, energy_product=0.2, energy_ts=1.0)
        fake_brentq = _make_fake_brentq(left_value=-0.2, right_value=0.4)
        with patch(FUNC_PATH + ".brentq", side_effect=fake_brentq):
            result = find_turning_points(1.5, potential)
        expected_level = max(0, 2 * 1.0 - 1.5)
        assert result["QUASI_LEVEL"] == pytest.approx(expected_level), \
            f"expected {expected_level}, got {result['QUASI_LEVEL']}"

    def test_scale_factor_matches_formula_when_above_one(self):
        """ Verify that SCALE_FACTOR equals level_energy / (2 * energy_ts) when that ratio exceeds 1 """
        potential = _make_potential(energy_reagent=0.0, energy_product=0.2, energy_ts=1.0)
        fake_brentq = _make_fake_brentq(left_value=-0.1, right_value=0.1)
        with patch(FUNC_PATH + ".brentq", side_effect=fake_brentq):
            result = find_turning_points(3.0, potential)
        expected_scale = 3.0 / (2 * 1.0)
        assert result["SCALE_FACTOR"] == pytest.approx(expected_scale), \
            f"expected {expected_scale}, got {result['SCALE_FACTOR']}"

    def test_scale_factor_floored_at_one(self):
        """ Verify that SCALE_FACTOR never drops below 1, even when level_energy / (2 * energy_ts) is small """
        potential = _make_potential(energy_reagent=0.0, energy_product=0.2, energy_ts=1.0)
        fake_brentq = _make_fake_brentq(left_value=-0.1, right_value=0.1)
        with patch(FUNC_PATH + ".brentq", side_effect=fake_brentq):
            result = find_turning_points(1.01, potential)
        assert result["SCALE_FACTOR"] == pytest.approx(1.0), f"expected {1.0}, got {result['SCALE_FACTOR']}"

    def test_zero_protection_applied_when_ts_energy_is_near_zero(self):
        """ Verify that the zero-protection value substitutes for a near-zero (2 * energy_ts) denominator """
        potential = _make_potential(energy_reagent=-2.0, energy_product=-1.0, energy_ts=0.0)
        fake_brentq = _make_fake_brentq(left_value=-0.1, right_value=0.1)
        with patch(FUNC_PATH + ".brentq", side_effect=fake_brentq):
            result = find_turning_points(0.001, potential, zero_protection_val=1e-12)
        expected_scale = 0.001 / 1e-12
        assert result["SCALE_FACTOR"] == pytest.approx(expected_scale, rel=1e-9), \
            f"expected {expected_scale}, got {result['SCALE_FACTOR']}"


# ===========================================================================
# --- find_turning_points - over-the-barrier, QUASI_ENDOTHERMIC_REACTION ---
# ===========================================================================
class TestFindTurningPointsQuasiEndothermic:

    def test_returns_quasi_endothermic_result_with_expected_fields(self):
        """ Verify that a corrected energy below the product's energy gives a QUASI_ENDOTHERMIC_REACTION result """
        potential = _make_potential(energy_reagent=0.0, energy_product=0.6, energy_ts=1.0)
        fake_brentq = _make_fake_brentq(left_value=-0.4, right_value=0.9)
        with patch(FUNC_PATH + ".brentq", side_effect=fake_brentq):
            result = find_turning_points(1.8, potential)
        assert result["regime"] == "OVER THE BARRIER", f"expected {'OVER THE BARRIER'}, got {result['regime']}"
        assert result["OVER_STATUS"] == "QUASI_ENDOTHERMIC_REACTION", \
            f"expected {'QUASI_ENDOTHERMIC_REACTION'}, got {result['OVER_STATUS']}"
        assert result["quasi_left"] == -0.4, f"expected {-0.4}, got {result['quasi_left']}"
        assert result["quasi_right"] == 0.9, f"expected {0.9}, got {result['quasi_right']}"

    def test_quasi_level_equals_the_products_energy(self):
        """ Verify that QUASI_LEVEL is set to the product's energy in the quasi-endothermic branch """
        potential = _make_potential(energy_reagent=0.0, energy_product=0.6, energy_ts=1.0)
        fake_brentq = _make_fake_brentq(left_value=-0.4, right_value=0.9)
        with patch(FUNC_PATH + ".brentq", side_effect=fake_brentq):
            result = find_turning_points(1.8, potential)
        assert result["QUASI_LEVEL"] == 0.6, f"expected {0.6}, got {result['QUASI_LEVEL']}"

    def test_scale_factor_matches_formula(self):
        """ Verify that SCALE_FACTOR equals level_energy / (energy_ts + energy_product) """
        potential = _make_potential(energy_reagent=0.0, energy_product=0.6, energy_ts=1.0)
        fake_brentq = _make_fake_brentq(left_value=-0.4, right_value=0.9)
        with patch(FUNC_PATH + ".brentq", side_effect=fake_brentq):
            result = find_turning_points(1.8, potential)
        expected_scale = 1.8 / (1.0 + 0.6)
        assert result["SCALE_FACTOR"] == pytest.approx(expected_scale), \
            f"expected {expected_scale}, got {result['SCALE_FACTOR']}"

    def test_zero_protection_applied_when_energy_sum_is_near_zero(self):
        """ Verify that the zero-protection value substitutes when (energy_ts + energy_product) is near zero """
        potential = _make_potential(energy_reagent=-2.0, energy_product=0.5, energy_ts=-0.5)
        fake_brentq = _make_fake_brentq(left_value=-0.2, right_value=0.3)
        with patch(FUNC_PATH + ".brentq", side_effect=fake_brentq):
            result = find_turning_points(0.6, potential, zero_protection_val=1e-12)
        expected_scale = 0.6 / 1e-12
        assert result["SCALE_FACTOR"] == pytest.approx(expected_scale, rel=1e-9), \
            f"expected {expected_scale}, got {result['SCALE_FACTOR']}"
