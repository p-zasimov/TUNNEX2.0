# --- Test: test_kinetics. Unit tests for .\qmt_computations\core\kinetics.py
# Run with: pytest test_kinetics.py -v ---


# --- Modules ---
import pytest # type: ignore
from unittest.mock import MagicMock, patch
from contextlib import ExitStack

import numpy as np
from scipy.special import expit


# --- Module to test ---
from tunnex_2.qmt_computations.core.kinetics import ( # type: ignore
    transmission_prob,
    reaction_rate,
    half_life,
    _boltzmann_average,
    _temperature_averaging_finite_sum,
    _reaching_target_probability,
    _temperature_averaging_infinite_sum,
    _probabilities_for_integral,
    _integration_over_weighted_probs,
    _temperature_averaging_integral,
    temperature_averaging_tunneling,
    arrhenius_rate)


# --- Helpers and shared data ---
FUNC_PATH = "tunnex_2.qmt_computations.core.kinetics"


def _make_level(vib_energy=0.0, transmission_probability=0.5):
    """ Should build a MagicMock standing in for a single LevelResult row """
    level = MagicMock()
    level.vib_energy = vib_energy
    level.transmission_probability = transmission_probability
    return level


def _make_potential():
    """ Should build a bare MagicMock standing in for a potential object """
    return MagicMock()


def _patch_probability_chain(stack, turning_points=None, wkb_value=1.0, prob_sequence=None):
    """ Should patch find_turning_points, compute_wkb and transmission_prob together """
    turning_points = turning_points if turning_points is not None else {"left": -0.5, "right": 0.5}
    ftp = stack.enter_context(patch(FUNC_PATH + ".find_turning_points", return_value=turning_points))
    cwkb = stack.enter_context(patch(FUNC_PATH + ".compute_wkb", return_value=wkb_value))
    if prob_sequence is not None:
        tprob = stack.enter_context(patch(FUNC_PATH + ".transmission_prob", side_effect=prob_sequence))
    else:
        tprob = stack.enter_context(patch(FUNC_PATH + ".transmission_prob", return_value=0.5))
    return ftp, cwkb, tprob


# ===========================================================================
# --- transmission_prob - normal behaviour ---
# ===========================================================================
class TestTransmissionProbNormal:

    def test_zero_wkb_gives_one_half(self):
        """ Verify that a WKB integral of zero yields a transmission probability of 0.5 """
        result = transmission_prob(0.0)
        assert result == pytest.approx(0.5), f"expected {0.5}, got {result}"

    def test_matches_expit_formula(self):
        """ Verify that the transmission probability matches expit(-2 * wkb) """
        wkb = 1.75
        result = transmission_prob(wkb)
        expected = expit(-2 * wkb)
        assert result == pytest.approx(expected), f"expected {expected}, got {result}"

    def test_infinite_wkb_gives_zero(self):
        """ Verify that an infinite WKB integral yields a transmission probability of exactly 0.0 """
        result = transmission_prob(np.inf)
        assert result == 0.0, f"expected {0.0}, got {result}"

    def test_larger_wkb_gives_smaller_probability(self):
        """ Verify that the transmission probability decreases as the WKB integral increases """
        small = transmission_prob(0.5)
        large = transmission_prob(5.0)
        assert large < small, f"expected {large} < {small}"


# ===========================================================================
# --- reaction_rate - normal behaviour ---
# ===========================================================================
class TestReactionRateNormal:

    def test_matches_formula_with_patched_light_speed(self):
        """ Verify that reaction_rate follows freq * LIGHT_SPEED * 100 * prob """
        with patch(FUNC_PATH + ".LIGHT_SPEED", new=2.0):
            result = reaction_rate(freq=10.0, prob=0.5)
        expected = 10.0 * 2.0 * 100 * 0.5
        assert result == pytest.approx(expected), f"expected {expected}, got {result}"

    def test_zero_probability_gives_zero_rate(self):
        """ Verify that a zero transmission probability yields a zero reaction rate """
        with patch(FUNC_PATH + ".LIGHT_SPEED", new=1.0):
            result = reaction_rate(freq=100.0, prob=0.0)
        assert result == 0.0, f"expected {0.0}, got {result}"

    def test_scales_linearly_with_probability(self):
        """ Verify that doubling the probability doubles the reaction rate """
        with patch(FUNC_PATH + ".LIGHT_SPEED", new=1.0):
            single = reaction_rate(freq=50.0, prob=0.2)
            double = reaction_rate(freq=50.0, prob=0.4)
        assert double == pytest.approx(2 * single), f"expected {2 * single}, got {double}"


# ===========================================================================
# --- half_life - normal behaviour ---
# ===========================================================================
class TestHalfLifeNormal:

    def test_zero_reaction_rate_gives_infinity(self):
        """ Verify that a zero reaction rate yields an infinite half-life """
        result = half_life(0)
        assert np.isinf(result), f"expected {np.inf}, got {result}"

    def test_matches_log2_over_rate_formula(self):
        """ Verify that the half-life equals ln(2) divided by the reaction rate """
        result = half_life(2.0)
        expected = np.log(2) / 2.0
        assert result == pytest.approx(expected), f"expected {expected}, got {result}"

    def test_larger_rate_gives_smaller_half_life(self):
        """ Verify that a larger reaction rate yields a shorter half-life """
        slow = half_life(1.0)
        fast = half_life(10.0)
        assert fast < slow, f"expected {fast} < {slow}"


# ===========================================================================
# --- _boltzmann_average - normal behaviour ---
# ===========================================================================
class TestBoltzmannAverageNormal:

    def test_equal_probabilities_return_that_probability_regardless_of_weights(self):
        """ Verify that identical transmission probabilities across levels average to that same value """
        energies = np.array([0.0, 1.0, 2.0])
        probs = np.array([0.7, 0.7, 0.7])
        with patch(FUNC_PATH + ".K_B", new=1.0):
            result = _boltzmann_average(energies, probs, temperature=300.0)
        assert result == pytest.approx(0.7), f"expected {0.7}, got {result}"

    def test_matches_manual_weighted_average_with_patched_constants(self):
        """ Verify that the result matches a manually computed Boltzmann-weighted average """
        energies = np.array([0.0, 1.0])
        probs = np.array([1.0, 0.0])
        with patch(FUNC_PATH + ".K_B", new=1.0):
            result = _boltzmann_average(energies, probs, temperature=1.0)
        w0 = np.exp(0.0)
        w1 = np.exp(-1.0)
        expected = (w0 * 1.0 + w1 * 0.0) / (w0 + w1)
        assert result == pytest.approx(expected), f"expected {expected}, got {result}"

    def test_num_stab_mode_matches_default_mode(self):
        """ Verify that the numerically stable branch agrees with the default branch on the same input """
        energies = np.array([0.0, 0.5, 1.5])
        probs = np.array([0.9, 0.4, 0.1])
        with patch(FUNC_PATH + ".K_B", new=1.0):
            default_result = _boltzmann_average(energies, probs, temperature=2.0, num_stab=False)
            stable_result = _boltzmann_average(energies, probs, temperature=2.0, num_stab=True)
        assert stable_result == pytest.approx(default_result, rel=1e-9), \
            f"expected {default_result}, got {stable_result}"

    def test_lowest_energy_level_dominates_at_low_temperature(self):
        """ Verify that at very low temperature the average approaches the lowest-energy level's probability """
        energies = np.array([0.0, 10.0])
        probs = np.array([0.3, 0.9])
        with patch(FUNC_PATH + ".K_B", new=1.0):
            result = _boltzmann_average(energies, probs, temperature=1e-6)
        assert result == pytest.approx(0.3, abs=1e-3), f"expected close to {0.3}, got {result}"


# ===========================================================================
# --- _temperature_averaging_finite_sum - normal behaviour ---
# ===========================================================================
class TestTemperatureAveragingFiniteSumNormal:

    def test_forwards_extracted_arrays_to_boltzmann_average(self):
        """ Verify that vib_energy and transmission_probability arrays are extracted and forwarded """
        levels = [_make_level(vib_energy=0.0, transmission_probability=0.6),
                  _make_level(vib_energy=1.0, transmission_probability=0.2)]
        with patch(FUNC_PATH + "._boltzmann_average", return_value=0.42) as mock:
            result = _temperature_averaging_finite_sum(levels, temperature=300.0, num_stab=True)
        args = mock.call_args.args
        assert np.allclose(args[0], [0.0, 1.0]), f"expected {[0.0, 1.0]}, got {args[0]}"
        assert np.allclose(args[1], [0.6, 0.2]), f"expected {[0.6, 0.2]}, got {args[1]}"
        assert result == 0.42, f"expected {0.42}, got {result}"

    def test_forwards_temperature_and_num_stab_as_kwargs(self):
        """ Verify that temperature and num_stab are passed through to _boltzmann_average """
        levels = [_make_level()]
        with patch(FUNC_PATH + "._boltzmann_average", return_value=0.1) as mock:
            _temperature_averaging_finite_sum(levels, temperature=250.0, num_stab=False)
        kwargs = mock.call_args.kwargs
        assert kwargs["temperature"] == 250.0, f"expected {250.0}, got {kwargs['temperature']}"
        assert kwargs["num_stab"] is False, f"expected {False}, got {kwargs['num_stab']}"


# ===========================================================================
# --- _reaching_target_probability - input validation ---
# ===========================================================================
class TestReachingTargetProbabilityValidation:

    def test_raises_on_target_probability_zero(self):
        """ Verify that a target_probability of 0 raises a ValueError """
        with pytest.raises(ValueError):
            _reaching_target_probability(1000.0, _make_potential(), target_probability=0.0)

    def test_raises_on_target_probability_above_one(self):
        """ Verify that a target_probability above 1 raises a ValueError """
        with pytest.raises(ValueError):
            _reaching_target_probability(1000.0, _make_potential(), target_probability=1.5)

    def test_raises_on_negative_target_probability(self):
        """ Verify that a negative target_probability raises a ValueError """
        with pytest.raises(ValueError):
            _reaching_target_probability(1000.0, _make_potential(), target_probability=-0.1)


# ===========================================================================
# --- _reaching_target_probability - normal behaviour ---
# ===========================================================================
class TestReachingTargetProbabilityNormal:

    def test_stops_once_target_probability_is_reached(self):
        """ Verify that the loop stops as soon as the accumulated probability reaches the target """
        with ExitStack() as stack:
            _patch_probability_chain(stack, prob_sequence=[0.5, 0.99])
            probabilities, level_energy = _reaching_target_probability(
                1000.0, _make_potential(), target_probability=0.99)
        assert probabilities == [0.5, 0.99], f"expected {[0.5, 0.99]}, got {probabilities}"

    def test_returns_a_probability_per_evaluated_level(self):
        """ Verify that one probability entry is returned for each vibrational level evaluated """
        with ExitStack() as stack:
            _patch_probability_chain(stack, prob_sequence=[0.1, 0.4, 0.995])
            probabilities, _ = _reaching_target_probability(500.0, _make_potential(), target_probability=0.99)
        assert len(probabilities) == 3, f"expected {3}, got {len(probabilities)}"

    def test_stops_at_max_levels_and_logs_warning(self):
        """ Verify that the loop stops at max_levels and logs a warning when the target is not reached """
        with ExitStack() as stack:
            _patch_probability_chain(stack, prob_sequence=[0.1, 0.2, 0.3, 0.4])
            logger_mock = stack.enter_context(patch(FUNC_PATH + ".logger"))
            probabilities, _ = _reaching_target_probability(
                500.0, _make_potential(), target_probability=0.99, max_levels=4)
        assert len(probabilities) == 4, f"expected {4}, got {len(probabilities)}"
        logger_mock.warning.assert_called_once()

    def test_level_energy_corresponds_to_last_evaluated_level(self):
        """ Verify that the returned level energy corresponds to the vibrational index of the last evaluated level """
        with ExitStack() as stack:
            _patch_probability_chain(stack, prob_sequence=[0.99])
            _, level_energy = _reaching_target_probability(1200.0, _make_potential(), target_probability=0.99)
        from tunnex_2.constants_and_settings.constants_and_settings import CM_M1_TO_HARTREE # type: ignore
        expected_energy = 0.5 * 1200.0 * CM_M1_TO_HARTREE
        assert level_energy == pytest.approx(expected_energy), f"expected {expected_energy}, got {level_energy}"


# ===========================================================================
# --- _temperature_averaging_infinite_sum - input validation ---
# ===========================================================================
class TestTemperatureAveragingInfiniteSumValidation:

    def test_raises_when_final_probability_out_of_range(self):
        """ Verify that a ValueError is raised when the last recorded probability is outside (0, 1] """
        with pytest.raises(ValueError):
            _temperature_averaging_infinite_sum(
                1000.0, [-0.5], temperature=300.0, target_probability=0.99)


# ===========================================================================
# --- _temperature_averaging_infinite_sum - normal behaviour ---
# ===========================================================================
class TestTemperatureAveragingInfiniteSumNormal:

    def test_num_stab_mode_matches_default_mode(self):
        """ Verify that the numerically stable branch agrees with the default branch on the same input """
        probabilities = [0.1, 0.3, 0.6, 0.9]
        with patch(FUNC_PATH + ".K_B", new=1.0), patch(FUNC_PATH + ".CM_M1_TO_HARTREE", new=1.0):
            default_result = _temperature_averaging_infinite_sum(
                10.0, probabilities, temperature=50.0, target_probability=0.9, num_stab=False)
            stable_result = _temperature_averaging_infinite_sum(
                10.0, probabilities, temperature=50.0, target_probability=0.9, num_stab=True)
        assert stable_result == pytest.approx(default_result, rel=1e-6), \
            f"expected {default_result}, got {stable_result}"

    def test_result_lies_between_zero_and_one(self):
        """ Verify that the temperature-averaged probability is a valid probability in [0, 1] """
        probabilities = [0.2, 0.5, 0.8]
        with patch(FUNC_PATH + ".K_B", new=1.0), patch(FUNC_PATH + ".CM_M1_TO_HARTREE", new=1.0):
            result = _temperature_averaging_infinite_sum(
                5.0, probabilities, temperature=100.0, target_probability=0.8)
        assert 0.0 <= result <= 1.0, f"expected value in [0, 1], got {result}"

    def test_uses_last_probability_as_target_when_not_close_to_given_target(self):
        """ Verify that when probabilities[-1] differs from the given target, it is used as the new target instead """
        probabilities = [0.1, 0.2, 0.5]
        with patch(FUNC_PATH + ".K_B", new=1.0), patch(FUNC_PATH + ".CM_M1_TO_HARTREE", new=1.0):
            # Should not raise, since 0.5 is a valid probability even though it differs from 0.99
            result = _temperature_averaging_infinite_sum(
                5.0, probabilities, temperature=100.0, target_probability=0.99)
        assert 0.0 <= result <= 1.0, f"expected value in [0, 1], got {result}"


# ===========================================================================
# --- _probabilities_for_integral - input validation ---
# ===========================================================================
class TestProbabilitiesForIntegralValidation:

    def test_raises_on_grid_size_below_three(self):
        """ Verify that a grid_size below 3 raises a ValueError """
        with pytest.raises(ValueError):
            _probabilities_for_integral(1000.0, _make_potential(), grid_size=2)

    def test_raises_on_non_positive_frequency(self):
        """ Verify that a non-positive frequency raises a ValueError """
        with pytest.raises(ValueError):
            _probabilities_for_integral(0.0, _make_potential(), grid_size=5)


# ===========================================================================
# --- _probabilities_for_integral - normal behaviour ---
# ===========================================================================
class TestProbabilitiesForIntegralNormal:

    def test_returns_grid_of_requested_size(self):
        """ Verify that the returned energy and probability grids each have grid_size points """
        with ExitStack() as stack:
            stack.enter_context(patch(FUNC_PATH + "._reaching_target_probability",
                                       return_value=([0.5, 0.99], 5.0)))
            _patch_probability_chain(stack, prob_sequence=[0.1] * 10)
            energy_grid, probability_grid, zero_level, target_energy = _probabilities_for_integral(
                1000.0, _make_potential(), grid_size=10)
        assert len(energy_grid) == 10, f"expected {10}, got {len(energy_grid)}"
        assert len(probability_grid) == 10, f"expected {10}, got {len(probability_grid)}"

    def test_zero_level_matches_formula(self):
        """ Verify that zero_level equals 0.5 * frequency * CM_M1_TO_HARTREE """
        with ExitStack() as stack:
            stack.enter_context(patch(FUNC_PATH + "._reaching_target_probability",
                                       return_value=([0.5, 0.99], 5.0)))
            _patch_probability_chain(stack, prob_sequence=[0.1] * 5)
            with patch(FUNC_PATH + ".CM_M1_TO_HARTREE", new=1.0):
                _, _, zero_level, _ = _probabilities_for_integral(4.0, _make_potential(), grid_size=5)
        assert zero_level == pytest.approx(2.0), f"expected {2.0}, got {zero_level}"

    def test_doubles_target_energy_when_it_equals_zero_level(self):
        """ Verify that target_energy is doubled when it would otherwise equal zero_level """
        with ExitStack() as stack:
            with patch(FUNC_PATH + ".CM_M1_TO_HARTREE", new=1.0):
                zero_level = 0.5 * 4.0 * 1.0
                stack.enter_context(patch(FUNC_PATH + "._reaching_target_probability",
                                           return_value=([0.99], zero_level)))
                _patch_probability_chain(stack, prob_sequence=[0.1] * 5)
                _, _, _, target_energy = _probabilities_for_integral(4.0, _make_potential(), grid_size=5)
        assert target_energy == pytest.approx(2 * zero_level), f"expected {2 * zero_level}, got {target_energy}"


# ===========================================================================
# --- _integration_over_weighted_probs - normal behaviour ---
# ===========================================================================
class TestIntegrationOverWeightedProbsNormal:

    def test_returns_a_tuple_of_integral_and_last_probability(self):
        """ Verify that the function returns a 2-tuple: the integral value and the last probability in the grid """
        energy_grid = np.linspace(0.0, 1.0, 5)
        probability_grid = np.linspace(0.9, 0.1, 5)
        result = _integration_over_weighted_probs(energy_grid, probability_grid, 0.0, 1.0, temperature=300.0)
        assert len(result) == 2, f"expected {2}, got {len(result)}"
        assert result[1] == probability_grid[-1], f"expected {probability_grid[-1]}, got {result[1]}"

    def test_integral_value_is_finite_and_non_negative(self):
        """ Verify that the computed integral is a finite, non-negative number """
        energy_grid = np.linspace(0.0, 2.0, 8)
        probability_grid = np.linspace(0.99, 0.01, 8)
        integral_value, _ = _integration_over_weighted_probs(energy_grid, probability_grid, 0.0, 2.0, temperature=100.0)
        assert np.isfinite(integral_value), f"expected a finite value, got {integral_value}"
        assert integral_value >= 0.0, f"expected a non-negative value, got {integral_value}"


# ===========================================================================
# --- _temperature_averaging_integral - input validation ---
# ===========================================================================
class TestTemperatureAveragingIntegralValidation:

    def test_raises_when_final_probability_out_of_range(self):
        """ Verify that a ValueError is raised when the recorded probability is outside (0, 1] """
        with pytest.raises(ValueError):
            _temperature_averaging_integral(
                zero_level=0.0, target_energy=1.0, barrier_integral_and_prob=(0.1, -0.5),
                temperature=300.0, target_probability=0.99)


# ===========================================================================
# --- _temperature_averaging_integral - normal behaviour ---
# ===========================================================================
class TestTemperatureAveragingIntegralNormal:

    def test_matches_manual_formula_with_patched_kb(self):
        """ Verify that the default-mode result matches a manually computed formula """
        with patch(FUNC_PATH + ".K_B", new=1.0):
            result = _temperature_averaging_integral(
                zero_level=0.0, target_energy=1.0, barrier_integral_and_prob=(0.5, 0.9),
                temperature=2.0, target_probability=0.9)
            sum_boltzmann = 1.0 * 2.0
            prob_over_the_barrier_scaled = 0.9 * np.exp(-(1.0 - 0.0) / (1.0 * 2.0))
            expected = 0.5 / sum_boltzmann + prob_over_the_barrier_scaled
        assert result == pytest.approx(expected), f"expected {expected}, got {result}"

    def test_num_stab_mode_matches_default_mode(self):
        """ Verify that the numerically stable branch agrees with the default branch on the same input """
        with patch(FUNC_PATH + ".K_B", new=1.0):
            default_result = _temperature_averaging_integral(
                zero_level=0.0, target_energy=2.0, barrier_integral_and_prob=(0.3, 0.8),
                temperature=5.0, target_probability=0.8, num_stab=False)
            stable_result = _temperature_averaging_integral(
                zero_level=0.0, target_energy=2.0, barrier_integral_and_prob=(0.3, 0.8),
                temperature=5.0, target_probability=0.8, num_stab=True)
        assert stable_result == pytest.approx(default_result, rel=1e-6), \
            f"expected {default_result}, got {stable_result}"


# ===========================================================================
# --- temperature_averaging_tunneling - input validation ---
# ===========================================================================
class TestTemperatureAveragingTunnelingValidation:

    def test_raises_on_invalid_prob_aver_mode(self):
        """ Verify that an invalid prob_aver_mode raises a ValueError """
        with pytest.raises(ValueError):
            temperature_averaging_tunneling(1000.0, [], _make_potential(), 300.0, prob_aver_mode="bogus")


# ===========================================================================
# --- temperature_averaging_tunneling - dispatch behaviour ---
# ===========================================================================
class TestTemperatureAveragingTunnelingDispatch:

    def test_finite_sum_mode_delegates_to_finite_sum_helper(self):
        """ Verify that 'finite_sum' mode calls _temperature_averaging_finite_sum with the levels and temperature """
        levels = [_make_level()]
        with patch(FUNC_PATH + "._temperature_averaging_finite_sum", return_value=0.55) as mock:
            result = temperature_averaging_tunneling(
                1000.0, levels, _make_potential(), 310.0, prob_aver_mode="finite_sum")
        mock.assert_called_once_with(levels, 310.0, num_stab=False)
        assert result == 0.55, f"expected {0.55}, got {result}"

    def test_infinite_sum_mode_delegates_to_reaching_and_infinite_sum_helpers(self):
        """ Verify that 'infinite_sum' mode computes probabilities first, then temperature-averages them """
        potential = _make_potential()
        with ExitStack() as stack:
            reach_mock = stack.enter_context(patch(
                FUNC_PATH + "._reaching_target_probability", return_value=([0.5, 0.99], 1.0)))
            inf_mock = stack.enter_context(patch(
                FUNC_PATH + "._temperature_averaging_infinite_sum", return_value=0.33))
            result = temperature_averaging_tunneling(
                800.0, [], potential, 300.0, prob_aver_mode="infinite_sum")
        reach_mock.assert_called_once()
        inf_args = inf_mock.call_args.args
        assert inf_args[0] == 800.0, f"expected {800.0}, got {inf_args[0]}"
        assert inf_args[1] == [0.5, 0.99], f"expected {[0.5, 0.99]}, got {inf_args[1]}"
        assert result == 0.33, f"expected {0.33}, got {result}"

    def test_integral_mode_delegates_to_full_integral_chain(self):
        """ Verify that 'integral' mode chains the probability grid, integration and averaging helpers """
        potential = _make_potential()
        with ExitStack() as stack:
            prob_mock = stack.enter_context(patch(
                FUNC_PATH + "._probabilities_for_integral",
                return_value=(np.array([0.0, 1.0]), np.array([0.9, 0.1]), 0.0, 1.0)))
            integ_mock = stack.enter_context(patch(
                FUNC_PATH + "._integration_over_weighted_probs", return_value=(0.4, 0.1)))
            aver_mock = stack.enter_context(patch(
                FUNC_PATH + "._temperature_averaging_integral", return_value=0.6))
            result = temperature_averaging_tunneling(
                600.0, [], potential, 250.0, prob_aver_mode="integral")
        prob_mock.assert_called_once()
        integ_mock.assert_called_once()
        aver_mock.assert_called_once()
        assert result == 0.6, f"expected {0.6}, got {result}"


# ===========================================================================
# --- arrhenius_rate - input validation ---
# ===========================================================================
class TestArrheniusRateValidation:

    def test_raises_on_invalid_prob_aver_mode(self):
        """ Verify that an invalid prob_aver_mode raises a ValueError """
        with pytest.raises(ValueError):
            arrhenius_rate([], 1000.0, 200.0, 400.0, 50.0, _make_potential(), prob_aver_mode="bogus")


# ===========================================================================
# --- arrhenius_rate - normal behaviour (finite_sum mode) ---
# ===========================================================================
class TestArrheniusRateFiniteSum:

    def _run(self, T_min=200.0, T_max=300.0, T_step=50.0, prob_value=0.5, rate_value=10.0):
        """ Should run arrhenius_rate in finite_sum mode with the internal helpers patched """
        with ExitStack() as stack:
            fs_mock = stack.enter_context(patch(
                FUNC_PATH + "._temperature_averaging_finite_sum", return_value=prob_value))
            rate_mock = stack.enter_context(patch(
                FUNC_PATH + ".reaction_rate", return_value=rate_value))
            arr_mock = stack.enter_context(patch(
                FUNC_PATH + ".ArrheniusResult", side_effect=lambda *a: ("ArrheniusResult", a)))
            result = arrhenius_rate(
                [_make_level()], 1000.0, T_min, T_max, T_step, _make_potential(), prob_aver_mode="finite_sum")
        return fs_mock, rate_mock, arr_mock, result

    def test_produces_one_result_per_temperature_step(self):
        """ Verify that one ArrheniusResult is produced for each temperature from T_max down to T_min """
        _, _, _, result = self._run(T_min=200.0, T_max=300.0, T_step=50.0)
        assert len(result) == 3, f"expected {3}, got {len(result)}"

    def test_includes_t_min_even_if_not_an_exact_step(self):
        """ Verify that T_min is appended when the step size does not land exactly on it """
        _, _, _, result = self._run(T_min=210.0, T_max=300.0, T_step=50.0)
        temperatures = [r[1][0] for r in result]
        assert 210.0 in temperatures, f"expected {210.0} in {temperatures}"

    def test_calls_finite_sum_helper_for_each_temperature(self):
        """ Verify that _temperature_averaging_finite_sum is called once per temperature step """
        fs_mock, _, _, _ = self._run(T_min=200.0, T_max=300.0, T_step=50.0)
        assert fs_mock.call_count == 3, f"expected {3}, got {fs_mock.call_count}"

    def test_arrhenius_result_stores_temperature_and_its_inverse(self):
        """ Verify that each ArrheniusResult is constructed with the temperature and its reciprocal """
        _, _, _, result = self._run(T_min=200.0, T_max=200.0, T_step=50.0)
        args = result[0][1]
        assert args[0] == 200.0, f"expected {200.0}, got {args[0]}"
        assert args[1] == pytest.approx(1 / 200.0), f"expected {1 / 200.0}, got {args[1]}"

    def test_log_reaction_rate_is_log_of_reaction_rate_value(self):
        """ Verify that the stored log reaction rate equals the natural log of the computed reaction rate """
        _, _, _, result = self._run(T_min=200.0, T_max=200.0, T_step=50.0, rate_value=5.0)
        args = result[0][1]
        assert args[2] == pytest.approx(np.log(5.0)), f"expected {np.log(5.0)}, got {args[2]}"


# ===========================================================================
# --- arrhenius_rate - edge cases in the reaction-rate handling ---
# ===========================================================================
class TestArrheniusRateEdgeCases:

    def test_zero_reaction_rate_gives_negative_infinity_log(self):
        """ Verify that a zero reaction rate is logged as -inf instead of raising """
        with ExitStack() as stack:
            stack.enter_context(patch(FUNC_PATH + "._temperature_averaging_finite_sum", return_value=0.0))
            stack.enter_context(patch(FUNC_PATH + ".reaction_rate", return_value=0.0))
            arr_mock = stack.enter_context(patch(
                FUNC_PATH + ".ArrheniusResult", side_effect=lambda *a: ("ArrheniusResult", a)))
            result = arrhenius_rate(
                [_make_level()], 1000.0, 200.0, 200.0, 50.0, _make_potential(), prob_aver_mode="finite_sum")
        log_rate = result[0][1][2]
        assert np.isneginf(log_rate), f"expected {-np.inf}, got {log_rate}"

    def test_negative_reaction_rate_raises_value_error(self):
        """ Verify that a negative reaction rate value raises a ValueError instead of being logged """
        with ExitStack() as stack:
            stack.enter_context(patch(FUNC_PATH + "._temperature_averaging_finite_sum", return_value=0.5))
            stack.enter_context(patch(FUNC_PATH + ".reaction_rate", return_value=-1.0))
            with pytest.raises(ValueError):
                arrhenius_rate(
                    [_make_level()], 1000.0, 200.0, 200.0, 50.0, _make_potential(), prob_aver_mode="finite_sum")

    def test_stops_iterating_once_temperature_drops_to_zero_or_below(self):
        """ Verify that the temperature loop breaks once a non-positive temperature is reached """
        with ExitStack() as stack:
            stack.enter_context(patch(FUNC_PATH + "._temperature_averaging_finite_sum", return_value=0.5))
            stack.enter_context(patch(FUNC_PATH + ".reaction_rate", return_value=1.0))
            stack.enter_context(patch(
                FUNC_PATH + ".ArrheniusResult", side_effect=lambda *a: ("ArrheniusResult", a)))
            result = arrhenius_rate(
                [_make_level()], 1000.0, -50.0, 50.0, 50.0, _make_potential(), prob_aver_mode="finite_sum")
        temperatures = [r[1][0] for r in result]
        assert all(t > 0 for t in temperatures), f"expected all temperatures positive, got {temperatures}"


# ===========================================================================
# --- arrhenius_rate - normal behaviour (infinite_sum and integral modes) ---
# ===========================================================================
class TestArrheniusRateOtherModes:

    def test_infinite_sum_mode_precomputes_probabilities_once(self):
        """ Verify that 'infinite_sum' mode calls _reaching_target_probability exactly once before the loop """
        with ExitStack() as stack:
            reach_mock = stack.enter_context(patch(
                FUNC_PATH + "._reaching_target_probability", return_value=([0.5, 0.99], 1.0)))
            stack.enter_context(patch(FUNC_PATH + "._temperature_averaging_infinite_sum", return_value=0.4))
            stack.enter_context(patch(FUNC_PATH + ".reaction_rate", return_value=1.0))
            stack.enter_context(patch(
                FUNC_PATH + ".ArrheniusResult", side_effect=lambda *a: ("ArrheniusResult", a)))
            arrhenius_rate(
                [], 1000.0, 200.0, 300.0, 50.0, _make_potential(), prob_aver_mode="infinite_sum")
        reach_mock.assert_called_once()

    def test_integral_mode_precomputes_probability_grid_once(self):
        """ Verify that 'integral' mode calls _probabilities_for_integral exactly once before the loop """
        with ExitStack() as stack:
            prob_mock = stack.enter_context(patch(
                FUNC_PATH + "._probabilities_for_integral",
                return_value=(np.array([0.0, 1.0]), np.array([0.9, 0.1]), 0.0, 1.0)))
            stack.enter_context(patch(FUNC_PATH + "._integration_over_weighted_probs", return_value=(0.2, 0.1)))
            stack.enter_context(patch(FUNC_PATH + "._temperature_averaging_integral", return_value=0.3))
            stack.enter_context(patch(FUNC_PATH + ".reaction_rate", return_value=1.0))
            stack.enter_context(patch(
                FUNC_PATH + ".ArrheniusResult", side_effect=lambda *a: ("ArrheniusResult", a)))
            arrhenius_rate(
                [], 1000.0, 200.0, 300.0, 50.0, _make_potential(), prob_aver_mode="integral")
        prob_mock.assert_called_once()
