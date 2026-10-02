# --- Test: test_potential. Unit tests for .\qmt_computations\core\potential.py
# Run with: pytest test_potential.py -v ---


# --- Modules ---
import pytest # type: ignore
from unittest.mock import MagicMock, patch
from contextlib import ExitStack

import numpy as np


# --- Module to test ---
from tunnex_2.qmt_computations.core.potential import Potential # type: ignore


# --- Helpers and shared data ---
FUNC_PATH = "tunnex_2.qmt_computations.core.potential"


def _make_data(e_x=None, e_y=None, zpve_x=None, zpve_y=None):
    """ Should build a self-consistent set of IRC arrays for building a Potential instance """
    e_x = e_x if e_x is not None else np.array([-1.0, 0.0, 1.0])
    e_y = e_y if e_y is not None else np.array([-10.0, -9.0, -10.5])
    zpve_x = zpve_x if zpve_x is not None else np.array([-1.0, 0.0, 1.0])
    zpve_y = zpve_y if zpve_y is not None else np.array([0.01, 0.015, 0.012])
    return e_x, e_y, zpve_x, zpve_y


def _make_potential(e0=-10.0, zpve0=0.01, scaling=1.0, **data_overrides):
    """ Should build a real Potential instance with default or overridden IRC data and reference energies """
    e_x, e_y, zpve_x, zpve_y = _make_data(**data_overrides)
    return Potential(e_x, e_y, zpve_x, zpve_y, E0=e0, ZPVE0=zpve0, potential_scaling_factor=scaling)


# ===========================================================================
# --- Potential.__init__ - interpolator construction ---
# ===========================================================================
class TestPotentialInitInterpolators:

    def test_calls_pchip_interpolator_with_electronic_energy_data(self):
        """ Verify that the E_interp interpolator is built from the E_x/E_y arrays with extrapolate disabled """
        e_x, e_y, zpve_x, zpve_y = _make_data()
        with patch(FUNC_PATH + ".PchipInterpolator") as mock_cls:
            mock_cls.return_value = MagicMock()
            Potential(e_x, e_y, zpve_x, zpve_y, E0=-10.0, ZPVE0=0.01, potential_scaling_factor=1.0)
        first_call = mock_cls.call_args_list[0]
        assert np.array_equal(first_call.args[0], e_x), f"expected {e_x}, got {first_call.args[0]}"
        assert np.array_equal(first_call.args[1], e_y), f"expected {e_y}, got {first_call.args[1]}"
        assert first_call.kwargs.get("extrapolate") is False, \
            f"expected {False}, got {first_call.kwargs.get('extrapolate')}"

    def test_calls_pchip_interpolator_with_zpve_data(self):
        """ Verify that the ZPVE_interp interpolator is built from the ZPVE_x/ZPVE_y arrays with extrapolate disabled """
        e_x, e_y, zpve_x, zpve_y = _make_data()
        with patch(FUNC_PATH + ".PchipInterpolator") as mock_cls:
            mock_cls.return_value = MagicMock()
            Potential(e_x, e_y, zpve_x, zpve_y, E0=-10.0, ZPVE0=0.01, potential_scaling_factor=1.0)
        second_call = mock_cls.call_args_list[1]
        assert np.array_equal(second_call.args[0], zpve_x), f"expected {zpve_x}, got {second_call.args[0]}"
        assert np.array_equal(second_call.args[1], zpve_y), f"expected {zpve_y}, got {second_call.args[1]}"
        assert second_call.kwargs.get("extrapolate") is False, \
            f"expected {False}, got {second_call.kwargs.get('extrapolate')}"


# ===========================================================================
# --- Potential.__init__ - offset and deviation ---
# ===========================================================================
class TestPotentialInitOffsetAndDeviation:

    def test_offset_matches_formula(self):
        """ Verify that offset equals (E_y[0] + ZPVE_y[0]) - (E0 + ZPVE0) """
        potential = _make_potential(e0=-10.0, zpve0=0.01, e_y=np.array([-10.0, -9.0, -10.5]),
                                     zpve_y=np.array([0.01, 0.015, 0.012]))
        expected = (-10.0 + 0.01) - (-10.0 + 0.01)
        assert potential.offset == pytest.approx(expected), f"expected {expected}, got {potential.offset}"

    def test_deviation_matches_formula(self):
        """ Verify that deviation equals abs(offset / (E0 + ZPVE0)) """
        potential = _make_potential(e0=-10.0, zpve0=0.01, e_y=np.array([-10.5, -9.0, -10.0]),
                                     zpve_y=np.array([0.01, 0.015, 0.012]))
        expected_offset = (-10.5 + 0.01) - (-10.0 + 0.01)
        expected_deviation = abs(expected_offset / (-10.0 + 0.01))
        assert potential.deviation == pytest.approx(expected_deviation), \
            f"expected {expected_deviation}, got {potential.deviation}"

    def test_warns_when_deviation_meets_or_exceeds_threshold(self):
        """ Verify that a warning is logged when the deviation reaches or exceeds 2% """
        with patch(FUNC_PATH + ".logger") as logger_mock:
            _make_potential(e0=-10.0, zpve0=0.0, e_y=np.array([-11.0, -9.0, -10.0]),
                             zpve_y=np.array([0.0, 0.0, 0.0]))
        logger_mock.warning.assert_called_once()

    def test_does_not_warn_when_deviation_below_threshold(self):
        """ Verify that no warning is logged when the deviation is below 2% """
        with patch(FUNC_PATH + ".logger") as logger_mock:
            _make_potential(e0=-10.0, zpve0=0.01, e_y=np.array([-10.0, -9.0, -10.5]),
                             zpve_y=np.array([0.01, 0.015, 0.012]))
        logger_mock.warning.assert_not_called()


# ===========================================================================
# --- Potential.__init__ - stored attributes ---
# ===========================================================================
class TestPotentialInitStoredAttributes:

    def test_stores_potential_scaling_factor(self):
        """ Verify that potential_scaling_factor is stored as given """
        potential = _make_potential(scaling=2.5)
        assert potential.potential_scaling_factor == 2.5, \
            f"expected {2.5}, got {potential.potential_scaling_factor}"

    def test_irc_x_min_equals_first_e_x_value(self):
        """ Verify that irc_x_min equals the first value of the electronic-energy IRC array """
        potential = _make_potential(e_x=np.array([-2.5, 0.0, 2.5]))
        assert potential.irc_x_min == -2.5, f"expected {-2.5}, got {potential.irc_x_min}"

    def test_irc_x_max_equals_last_e_x_value(self):
        """ Verify that irc_x_max equals the last value of the electronic-energy IRC array """
        potential = _make_potential(e_x=np.array([-2.5, 0.0, 2.5]))
        assert potential.irc_x_max == 2.5, f"expected {2.5}, got {potential.irc_x_max}"

    def test_irc_x_ts_equals_x_position_of_maximum_electronic_energy(self):
        """ Verify that irc_x_ts equals the IRC x-value at which E_y attains its maximum """
        potential = _make_potential(e_x=np.array([-1.0, 0.0, 1.0]), e_y=np.array([-10.0, -8.0, -10.5]))
        assert potential.irc_x_ts == 0.0, f"expected {0.0}, got {potential.irc_x_ts}"

    def test_irc_x_ts_picks_first_occurrence_of_the_maximum(self):
        """ Verify that irc_x_ts picks the first IRC x-value if the maximum electronic energy occurs more than once """
        potential = _make_potential(e_x=np.array([-1.0, 0.0, 1.0]), e_y=np.array([-8.0, -8.0, -10.0]))
        assert potential.irc_x_ts == -1.0, f"expected {-1.0}, got {potential.irc_x_ts}"


# ===========================================================================
# --- Potential.__call__ - normal behaviour ---
# ===========================================================================
class TestPotentialCallNormal:

    def test_returns_zero_at_irc_x_min(self):
        """ Verify that evaluating the potential at irc_x_min returns exactly zero (the baseline is subtracted) """
        potential = _make_potential()
        result = potential(potential.irc_x_min)
        assert result == pytest.approx(0.0, abs=1e-8), f"expected {0.0}, got {result}"

    def test_matches_manual_formula_at_a_given_point(self):
        """ Verify that the potential at an arbitrary point matches the interpolator-based formula """
        potential = _make_potential()
        x = 0.0
        expected = potential.potential_scaling_factor * (
            potential.E_interp(x) + potential.ZPVE_interp(x)
            - (potential.E_interp(potential.irc_x_min) + potential.ZPVE_interp(potential.irc_x_min)))
        result = potential(x)
        assert result == pytest.approx(expected), f"expected {expected}, got {result}"

    def test_scales_result_by_potential_scaling_factor(self):
        """ Verify that doubling potential_scaling_factor doubles the returned potential value """
        unscaled = _make_potential(scaling=1.0)
        scaled = _make_potential(scaling=2.0)
        x = 0.0
        assert scaled(x) == pytest.approx(2 * unscaled(x)), f"expected {2 * unscaled(x)}, got {scaled(x)}"

    def test_accepts_array_input_and_returns_array_output(self):
        """ Verify that calling the potential with an array of x-values returns an array of matching length """
        potential = _make_potential()
        xs = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        result = potential(xs)
        assert isinstance(result, np.ndarray), f"expected {np.ndarray}, got {type(result)}"
        assert len(result) == len(xs), f"expected {len(xs)}, got {len(result)}"

    def test_result_is_zero_specifically_at_the_reagent_end_of_the_array(self):
        """ Verify that the first element of an array evaluation at irc_x_min is zero """
        potential = _make_potential()
        xs = np.array([potential.irc_x_min, 0.0, potential.irc_x_max])
        result = potential(xs)
        assert result[0] == pytest.approx(0.0, abs=1e-8), f"expected {0.0}, got {result[0]}"
