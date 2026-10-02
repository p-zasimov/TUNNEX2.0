# --- Test: test_kinetics. Unit tests for kinetics.py
# Run with: pytest test_kinetics.py -v ---


# --- Modules ---
import pytest # type: ignore
from unittest.mock import MagicMock, patch, call
from contextlib import ExitStack

import warnings
import numpy as np
from scipy.special import expit

from tunnex_2.constants_and_dataclasses.constants_and_settings import CM_M1_TO_HARTREE # type: ignore


# --- Module to test ---
from tunnex_2.qmt_computations.core.kinetics import ( # type: ignore
    transmission_prob,
    reaction_rate,
    half_life,
    _boltzmann_average,
    _temperature_averaging_finite_sum,
    _reaching_target_probability,
    _temperature_averaging_infinite_sum,
    _integrand_prob,
    _integration_over_weighted_probs,
    _temperature_averaging_integral,
    temperature_averaging_tunneling,
    arrhenius_rate)


# --- Helpers and shared data ---
FUNC_PATH = "tunnex_2.qmt_computations.core.kinetics"


def _make_level(vib_energy=0.0, transmission_probability=0.5):
    """ Should build an object standing in for a single LevelResult row """
    level = MagicMock()
    level.vib_energy = vib_energy
    level.transmission_probability = transmission_probability
    return level

def _make_potential():
    """ Should build an object standing in for a potential object """
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

def _patch_integrand_chain(stack, wkb_value=1.0, turning_points=None, k_b=1.0):
    """ Should patch K_B, find_turning_points, and compute_wkb for the tests """
    turning_points = turning_points if turning_points is not None else {"regime": "NORMAL",
        "left": -0.5, "right": 0.5}
    stack.enter_context(patch(FUNC_PATH + ".K_B", new=k_b))
    ftp = stack.enter_context(patch(FUNC_PATH + ".find_turning_points", return_value=turning_points))
    cwkb = stack.enter_context(patch(FUNC_PATH + ".compute_wkb", return_value=wkb_value))
    return ftp, cwkb

def _patch_integral_chain(stack, integral_value=0.5, k_b=1.0):
    """ Should patch K_B and _integration_over_weighted_probs for the tests """
    stack.enter_context(patch(FUNC_PATH + ".K_B", new=k_b))
    integ = stack.enter_context(patch(
        FUNC_PATH + "._integration_over_weighted_probs", return_value=integral_value))
    return integ

def _patch_integration_chain(stack, quad_result=(0.4, 1e-14, {})):
    """ Should patch quad and int_convergence_warning for tests """
    quad_mock = stack.enter_context(patch(FUNC_PATH + ".quad", return_value=quad_result))
    warn_mock = stack.enter_context(patch(FUNC_PATH + ".int_convergence_warning",
    return_value=False))
    return quad_mock, warn_mock

def _patch_averaging_helpers(stack, prob=0.5, reach_value=None):
    """ Should patch the probability search and all three temperature-averaging helpers """
    reach_value = reach_value if reach_value is not None else ([0.5, 0.99], 0.2, 1.0)
    result = {
        "reach": stack.enter_context(patch(
            FUNC_PATH + "._reaching_target_probability", return_value=reach_value)),
        "finite": stack.enter_context(patch(
            FUNC_PATH + "._temperature_averaging_finite_sum", return_value=prob)),
        "infinite": stack.enter_context(patch(
            FUNC_PATH + "._temperature_averaging_infinite_sum", return_value=prob)),
        "integral": stack.enter_context(patch(
            FUNC_PATH + "._temperature_averaging_integral", return_value=prob))}
    return result

def _patch_arrhenius_chain(stack, prob=0.5, rate=10.0):
    """ Should patch the averaging helpers, reaction_rate and ArrheniusResult for the tests """
    mocks = _patch_averaging_helpers(stack, prob=prob)
    mocks["rate"] = stack.enter_context(patch(FUNC_PATH + ".reaction_rate", return_value=rate))
    mocks["result"] = stack.enter_context(patch(
        FUNC_PATH + ".ArrheniusResult", side_effect=lambda *a: ("ArrheniusResult", a)))
    return mocks

def _run_arrhenius(mode="finite_sum", T_min=200.0, T_max=300.0, T_step=50.0,
    prob=0.5, rate=10.0, rate_side_effect=None, **kwargs):
    """ Should run arrhenius_rate with all the internal helpers patched """
    potential = _make_potential()
    levels = [_make_level()]
    with ExitStack() as stack:
        mocks = _patch_arrhenius_chain(stack, prob=prob, rate=rate)
        if rate_side_effect is not None:
            mocks["rate"].side_effect = rate_side_effect
        result = arrhenius_rate(levels, 1000.0, T_min, T_max, T_step, potential,
        prob_aver_mode=mode, **kwargs)
    return mocks, result, potential, levels

def _arrhenius_args(result):
    """ Should extract the constructor arguments from the stubbed ArrheniusResult objects """
    value = [r[1] for r in result]
    return value


# ===========================================================================
# --- transmission_prob - normal behaviour and input validation ---
# ===========================================================================
class TestTransmissionProb:

    def test_zero_wkb_gives_one_half(self):
        """ Verify that a WKB integral of zero yields a transmission probability of 0.5 """
        result = transmission_prob(0.0)
        assert np.isclose(result, 0.5), f"expected {0.5}, got {result}"

    def test_matches_expit_formula(self):
        """ Verify that the transmission probability matches expit(-2 * wkb) """
        wkb = 1.75
        result = transmission_prob(wkb)
        expected = expit(-2 * wkb)
        assert np.isclose(result, expected), f"expected {expected}, got {result}"

    def test_infinite_wkb_gives_zero(self):
        """
        Verify that an infinite WKB integral yields
        a transmission probability of exactly 0.0 and 1.0
        """
        result_p = transmission_prob(np.inf)
        result_m = transmission_prob(-np.inf)
        assert result_p == 0.0, f"expected {0.0}, got {result_p}; test #1"
        assert result_m == 1.0, f"expected {1.0}, got {result_m}; test #2"

    def test_larger_wkb_gives_smaller_probability(self):
        """
        Verify that the transmission probability decreases
        as the WKB integral increases
        """
        small = transmission_prob(0.5)
        large = transmission_prob(5.0)
        assert large < small, f"expected {large} < {small}"


# ===========================================================================
# --- reaction_rate - normal behaviour and input validation ---
# ===========================================================================
class TestReactionRate:

    def test_matches_formula_with_patched_light_speed(self):
        """ Verify that reaction_rate follows freq * LIGHT_SPEED * 100 * prob """
        with patch(FUNC_PATH + ".LIGHT_SPEED", new=2.0):
            result = reaction_rate(freq=10.0, prob=0.5)
        expected = 10.0 * 2.0 * 100 * 0.5
        assert np.isclose(result, expected), f"expected {expected}, got {result}"

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
        assert np.isclose(double, 2 * single), f"expected {2 * single}, got {double}"


# ===========================================================================
# --- half_life - normal behaviour and input validation ---
# ===========================================================================
class TestHalfLife:

    def test_zero_reaction_rate_gives_infinity(self):
        """ Verify that a zero reaction rate yields an infinite half-life """
        result = half_life(0)
        assert np.isinf(result), f"expected {np.inf}, got {result}"

    def test_matches_log2_over_rate_formula(self):
        """ Verify that the half-life equals np.log(2) divided by the reaction rate """
        result = half_life(2.0)
        expected = np.log(2) / 2.0
        assert np.isclose(result, expected), f"expected {expected}, got {result}"

    def test_larger_rate_gives_smaller_half_life(self):
        """ Verify that a larger reaction rate yields a shorter half-life """
        slow = half_life(1.0)
        fast = half_life(10.0)
        assert fast < slow, f"expected {fast} < {slow}"


# ===========================================================================
# --- _boltzmann_average - normal behaviour and input validation ---
# ===========================================================================
class TestBoltzmannAverage:

    def test_equal_probabilities_return_that_probability_regardless_of_weights(self):
        """
        Verify that identical transmission probabilities
        across levels average to that same value
        """
        energies = np.array([0.0, 1.0, 2.0])
        probs = np.array([0.7, 0.7, 0.7])
        with patch(FUNC_PATH + ".K_B", new=1.0):
             result = _boltzmann_average(energies, probs, temperature=300.0)
        assert np.isclose(result, 0.7), f"expected {0.7}, got {result}"

    def test_matches_manual_weighted_average_with_patched_constants(self):
        """ Verify that the result matches a manually computed Boltzmann-weighted average """
        energies = np.array([0.0, 1.0])
        probs = np.array([1.0, 0.0])
        with patch(FUNC_PATH + ".K_B", new=1.0):
             result = _boltzmann_average(energies, probs, temperature=1.0)
        w0 = np.exp(0.0)
        w1 = np.exp(-1.0)
        expected = (w0 * 1.0 + w1 * 0.0) / (w0 + w1)
        assert np.isclose(result, expected), f"expected {expected}, got {result}"

    def test_lowest_energy_level_dominates_at_low_temperature(self):
        """
        Verify that at very low temperature the average approaches
        the lowest-energy level's probability
        """
        energies = np.array([0.0, 10.0])
        probs = np.array([0.3, 0.9])
        with patch(FUNC_PATH + ".K_B", new=1.0):
             result = _boltzmann_average(energies, probs, temperature=1e-6)
        assert np.isclose(result, 0.3, atol=1e-3), f"expected close to {0.3}, got {result}"


# =================================================================================
# --- _temperature_averaging_finite_sum - normal behaviour and input validation ---
# =================================================================================
class TestTemperatureAveragingFiniteSum:

    def test_forwards_extracted_arrays_to_boltzmann_average(self):
        """
        Verify that vib_energy and transmission_probability arrays
        are extracted and forwarded
        """
        levels = [_make_level(vib_energy=0.0, transmission_probability=0.6),
                  _make_level(vib_energy=1.0, transmission_probability=0.2)]
        with patch(FUNC_PATH + "._boltzmann_average", return_value=0.42) as mock:
             result = _temperature_averaging_finite_sum(levels, temperature=300.0)
        args = mock.call_args.args
        assert np.allclose(args[0], [0.0, 1.0]), f"expected {[0.0, 1.0]}, got {args[0]}"
        assert np.allclose(args[1], [0.6, 0.2]), f"expected {[0.6, 0.2]}, got {args[1]}"
        assert result == 0.42, f"expected {0.42}, got {result}"

    def test_forwards_temperature_and_num_stab_as_kwargs(self):
        """ Verify that temperature is passed through to _boltzmann_average """
        levels = [_make_level()]
        with patch(FUNC_PATH + "._boltzmann_average", return_value=0.1) as mock:
             _temperature_averaging_finite_sum(levels, temperature=250.0)
        kwargs = mock.call_args.kwargs
        assert kwargs["temperature"] == 250.0, f"expected {250.0}, got {kwargs["temperature"]}"


# ============================================================================
# --- _reaching_target_probability - normal behaviour and input validation ---
# ============================================================================
class TestReachingTargetProbability:

    def test_stops_once_target_probability_is_reached(self):
        """ Verify that the loop stops as soon as the accumulated probability reaches the target """
        with ExitStack() as stack:
            _patch_probability_chain(stack, prob_sequence=[0.5, 0.99])
            probabilities, _, _ = _reaching_target_probability(
            1000.0, _make_potential(), target_probability=0.99)
        assert probabilities == [0.5, 0.99], f"expected {[0.5, 0.99]}, got {probabilities}"

    def test_returns_a_probability_per_evaluated_level(self):
        """ Verify that one probability entry is returned for each vibrational level evaluated """
        with ExitStack() as stack:
            _patch_probability_chain(stack, prob_sequence=[0.1, 0.4, 0.995])
            probabilities, _, _ = _reaching_target_probability(500.0,
            _make_potential(), target_probability=0.99)
        assert len(probabilities) == 3, f"expected {3}, got {len(probabilities)}"

    def test_stops_at_max_levels_and_logs_warning(self):
        """
        Verify that the loop stops at max_levels
        and logs a warning when the target probability is not reached
        """
        with ExitStack() as stack:
            _patch_probability_chain(stack, prob_sequence=[0.1, 0.2, 0.3, 0.4])
            logger_mock = stack.enter_context(patch(FUNC_PATH + ".logger"))
            probabilities, _, _ = _reaching_target_probability(
            500.0, _make_potential(), target_probability=0.99, max_levels=4)
        assert len(probabilities) == 4, f"expected {4}, got {len(probabilities)}"
        logger_mock.warning.assert_called_once()
        # it should be called once

    def test_level_energy_corresponds_to_last_evaluated_level(self):
        """
        Verify that the returned level energy corresponds
        to the vibrational index of the last evaluated level
        """
        with ExitStack() as stack:
            _patch_probability_chain(stack, prob_sequence=[0.99])
            _, _, target_energy = _reaching_target_probability(1200.0, _make_potential(), target_probability=0.99)
        expected_energy = 0.5 * 1200.0 * CM_M1_TO_HARTREE
        assert np.isclose(target_energy, expected_energy), f"expected {expected_energy}, got {target_energy}"
    
    def test_raises_on_target_probability_zero(self):
        """ Verify that a target_probability of 0 raises an error. Expecting a ValueError """
        with pytest.raises(ValueError, match = "Expected target_probability"):
            _reaching_target_probability(1000.0, _make_potential(), target_probability=0.0)

    def test_raises_on_target_probability_one(self):
        """ Verify that a target_probability of 1 raises an error. Expecting a ValueError """
        with pytest.raises(ValueError, match = "Expected target_probability"):
            _reaching_target_probability(1000.0, _make_potential(), target_probability=1.0)

    def test_raises_on_target_probability_above_one(self):
        """ Verify that a target_probability above 1 raises an error. Expecting a ValueError """
        with pytest.raises(ValueError, match = "Expected target_probability"):
            _reaching_target_probability(1000.0, _make_potential(), target_probability=1.5)

    def test_raises_on_negative_target_probability(self):
        """ Verify that a negative target_probability raises an error. Expecting a ValueError """
        with pytest.raises(ValueError, match = "Expected target_probability"):
            _reaching_target_probability(1000.0, _make_potential(), target_probability=-0.1)

    def test_raises_on_max_levels_smaller_than_four(self):
        """ Verify that max_levels < 4 raises an error. Expecting a ValueError """
        with pytest.raises(ValueError, match = "Expected max_levels"):
            _reaching_target_probability(1000.0, _make_potential(), max_levels=3)


# ===================================================================================
# --- _temperature_averaging_infinite_sum - normal behaviour and input validation ---
# ===================================================================================
class TestTemperatureAveragingInfiniteSum:

    def test_result_lies_between_zero_and_one(self):
        """ Verify that the temperature-averaged probability is a valid probability in [0, 1] """
        probabilities = [0.2, 0.5, 0.8]
        with patch(FUNC_PATH + ".K_B", new=1.0), patch(FUNC_PATH + ".CM_M1_TO_HARTREE", new=1.0):
             result = _temperature_averaging_infinite_sum(5.0, probabilities,
                 temperature=100.0, target_probability=0.8)
        assert 0.0 <= result <= 1.0, f"expected value in [0, 1], got {result}"

    def test_uses_last_probability_as_target_when_not_close_to_given_target(self):
        """
        Verify that when the reached probability differs from the given target,
        it is used as the new target instead
        """
        target_probability = 0.99
        probabilities = [0.1, 0.2, 0.5]
        with patch(FUNC_PATH + ".K_B", new=1.0), patch(FUNC_PATH + ".CM_M1_TO_HARTREE", new=1.0):
             result = _temperature_averaging_infinite_sum(5.0, probabilities,
                 temperature=100.0, target_probability=target_probability)
        assert result < target_probability, f"expected value < {target_probability}, got {result}"
    
    def test_raises_when_final_probability_out_of_range(self):
        """
        Verify that a target_probability beyond the (0, 1) range
        raises an error. Expecting a ValueError
        """
        with pytest.raises(ValueError, match = "Expected target_probability"):
            _temperature_averaging_infinite_sum(1000.0, [-0.5],
                temperature=300.0, target_probability=0.99)


# ============================================================================
# --- _integrand_prob - normal behaviour and input validation ---
# ============================================================================
class TestIntegrandProb:

    def test_turning_points_use_unshifted_energy(self):
        """
        Verify that find_turning_points receives the absolute energy,
        not energy - zero_level
        """
        potential = _make_potential()
        with ExitStack() as stack:
            ftp, _ = _patch_integrand_chain(stack)
            _integrand_prob(5.0, 2.0, potential, 300.0)
        ftp.assert_called_once_with(5.0, potential)
        # it should be called once

    def test_zero_level_only_affects_the_boltzmann_factor(self):
        """
        Verify that shifting the energy and the zero level together
        leaves the result unchanged (the WKB part is mocked)
        """
        with ExitStack() as stack:
            _patch_integrand_chain(stack, wkb_value=0.7)
            first = _integrand_prob(3.0, 1.0, _make_potential(), 1.0)
            second = _integrand_prob(103.0, 101.0, _make_potential(), 1.0)
        assert np.isclose(first, second), f"expected {first}, got {second}"

    @pytest.mark.parametrize("wkb", [-3.0, -0.5, 0.0, 0.5, 3.0, 20.0])
    def test_matches_analytic_formula(self, wkb):
        """ Verify that the result equals expit(-2 * wkb) * exp(-(E - E0) / (k_B * T)) """
        energy, zero_level, k_b, temperature = 3.0, 1.0, 2.0, 0.5
        with ExitStack() as stack:
            _patch_integrand_chain(stack, wkb_value=wkb, k_b=k_b)
            result = _integrand_prob(energy, zero_level, _make_potential(), temperature)
        expected = expit(-2.0 * wkb) * np.exp(-(energy - zero_level) / (k_b * temperature))
        assert np.isclose(result, expected, rtol=1e-12), f"expected {expected}, got {result}"

    def test_energy_at_zero_level_returns_pure_probability(self):
        """ Verify that the Boltzmann factor is exactly 1 when energy equals zero_level """
        with ExitStack() as stack:
            _patch_integrand_chain(stack, wkb_value=0.4)
            result = _integrand_prob(1.0, 1.0, _make_potential(), 1.0)
        expected = expit(-0.8)
        assert np.isclose(result, expected), f"expected {expected}, got {result}"

    def test_infinite_wkb_gives_zero_without_warnings(self):
        """ Verify that an infinite WKB integral (endothermic regime) gives 0.0 and no warnings """
        with ExitStack() as stack:
            _patch_integrand_chain(stack, wkb_value=np.inf)
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                # any RuntimeWarning is a test failure
                result = _integrand_prob(2.0, 1.0, _make_potential(), 1.0)
        assert result == 0.0, f"expected {0.0}, got {result}"

    def test_huge_wkb_gives_zero_without_log_of_zero(self):
        """ Verify that a huge WKB integral underflows to exactly 0.0 """
        with ExitStack() as stack:
            _patch_integrand_chain(stack, wkb_value=1000.0)
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                result = _integrand_prob(2.0, 1.0, _make_potential(), 1.0)
        assert result == 0.0, f"expected {0.0}, got {result}"

    def test_negative_infinite_wkb_gives_pure_boltzmann_factor(self):
        """
        Verify that W = -inf gives P = 1,
        i.e. just the Boltzmann factor, without warnings
        """
        with ExitStack() as stack:
            _patch_integrand_chain(stack, wkb_value=-np.inf)
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                result = _integrand_prob(2.0, 1.0, _make_potential(), 1.0)
        expected = np.exp(-1.0)
        assert np.isclose(result, expected), f"expected {expected}, got {result}"

    def test_negative_wkb_approaches_boltzmann_factor(self):
        """ Verify that a strongly negative W (over the barrier) gives P close to 1 """
        with ExitStack() as stack:
            _patch_integrand_chain(stack, wkb_value=-50.0)
            result = _integrand_prob(2.0, 1.0, _make_potential(), 1.0)
        expected = np.exp(-1.0)
        assert np.isclose(result, expected, rtol=1e-12), f"expected {expected}, got {result}"

    def test_nan_wkb_propagates_nan(self):
        """ Verify that a nan WKB integral is not silently turned into a number """
        with ExitStack() as stack:
            _patch_integrand_chain(stack, wkb_value=np.nan)
            result = _integrand_prob(2.0, 1.0, _make_potential(), 1.0)
        assert np.isnan(result), f"expected {np.nan}, got {result}"

    @pytest.mark.parametrize("wkb", [-10.0, -1.0, 0.0, 1.0, 10.0, np.inf])
    @pytest.mark.parametrize("energy", [1.0, 1.5, 5.0, 50.0])
    def test_result_lies_between_zero_and_one(self, wkb, energy):
        """ Verify that the weighted probability is a valid value in [0, 1] """
        with ExitStack() as stack:
            _patch_integrand_chain(stack, wkb_value=wkb)
            result = _integrand_prob(energy, 1.0, _make_potential(), 1.0)
        assert 0.0 <= result <= 1.0, f"expected value in [0, 1], got {result}"

    def test_larger_energy_gives_smaller_result_at_fixed_wkb(self):
        """ Verify that the Boltzmann weight decreases with energy when the WKB integral is fixed """
        with ExitStack() as stack:
            _patch_integrand_chain(stack, wkb_value=1.0)
            low = _integrand_prob(2.0, 1.0, _make_potential(), 1.0)
            high = _integrand_prob(4.0, 1.0, _make_potential(), 1.0)
        assert high < low, f"expected {high} < {low}"

    def test_higher_temperature_gives_larger_result(self):
        """ Verify that a higher temperature increases the Boltzmann weight at fixed energy """
        with ExitStack() as stack:
            _patch_integrand_chain(stack, wkb_value=1.0)
            cold = _integrand_prob(2.0, 1.0, _make_potential(), 0.5)
            hot = _integrand_prob(2.0, 1.0, _make_potential(), 2.0)
        assert hot > cold, f"expected {hot} > {cold}"


# ================================================================================
# --- _integration_over_weighted_probs - normal behaviour and input validation ---
# ================================================================================
class TestIntegrationOverWeightedProbs:

    def test_integrates_between_zero_level_and_target_energy(self):
        """
        Verify that quad is called with zero_level
        and target_energy as integration bounds
        """
        with ExitStack() as stack:
            quad_mock, _ = _patch_integration_chain(stack)
            _integration_over_weighted_probs(0.2, 1.5, _make_potential(), 300.0)
        args = quad_mock.call_args.args
        assert args[1] == 0.2, f"expected {0.2}, got {args[1]}; test #1"
        assert args[2] == 1.5, f"expected {1.5}, got {args[2]}; test #2"

    def test_integrand_forwards_all_arguments_to_integrand_prob(self):
        """
        Verify that the function given to quad calls _integrand_prob
        with the integration variable, zero_level, potential and temperature
        """
        potential = _make_potential()
        captured_func = {}
        def _capture_quad(func, a, b, **kwargs):
            captured_func["func"] = func
            return (0.0, 0.0, {})
        with ExitStack() as stack:
            stack.enter_context(patch(FUNC_PATH + ".quad", side_effect=_capture_quad))
            stack.enter_context(patch(FUNC_PATH + ".int_convergence_warning",
                                      return_value=False))
            integrand_mock = stack.enter_context(patch(FUNC_PATH + "._integrand_prob",
            return_value=0.3))
            _integration_over_weighted_probs(0.2, 1.5, potential, 310.0)
            value = captured_func["func"](0.9)
        integrand_mock.assert_called_once_with(0.9, 0.2, potential, 310.0)
        assert value == 0.3, f"expected {0.3}, got {value}"

    def test_returns_the_integrated_value_as_a_plain_number(self):
        """ Verify that the function returns the first element of the quad result, not a tuple """
        with ExitStack() as stack:
            _patch_integration_chain(stack, quad_result=(0.4, 1e-14, {}))
            result = _integration_over_weighted_probs(0.0, 1.0, _make_potential(), 300.0)
        assert result == 0.4, f"expected {0.4}, got {result}"
        assert np.ndim(result) == 0, f"expected a scalar, got ndim={np.ndim(result)}"

    def test_no_warning_is_logged_on_clean_integration(self):
        """ Verify that nothing is logged when quad returns no message """
        with ExitStack() as stack:
            stack.enter_context(patch(FUNC_PATH + ".quad", return_value=(0.4, 1e-14, {})))
            logger_mock = stack.enter_context(patch(FUNC_PATH + ".logger"))
            _integration_over_weighted_probs(0.0, 1.0, _make_potential(), 300.0)
        logger_mock.warning.assert_not_called()

    def test_result_is_finite_and_non_negative_for_growing_probability(self):
        """ Verify that a realistic P(E) growing with energy gives a finite, non-negative integral """
        integrand = lambda energy, zero_level, potential, temperature: (
            expit(4.0 * (energy - zero_level) - 2.0) * np.exp(-(energy - zero_level) / temperature))
        with ExitStack() as stack:
            stack.enter_context(patch(FUNC_PATH + "._integrand_prob", side_effect=integrand))
            stack.enter_context(patch(FUNC_PATH + ".logger"))
            result = _integration_over_weighted_probs(0.0, 5.0, _make_potential(), 1.0)
        assert np.isfinite(result), f"expected a finite value, got {result}; test #1"
        assert result >= 0.0, f"expected a non-negative value, got {result}; test #2"

    def test_integral_is_bounded_by_the_boltzmann_integral(self):
        """ Verify that for 0 <= P(E) <= 1 the integral does not exceed the one with P(E) = 1 """
        zero_level, target_energy, temperature = 0.0, 3.0, 1.0
        integrand = lambda energy, zl, potential, temp: (
            expit(2.0 * (energy - zl) - 1.0) * np.exp(-(energy - zl) / temp))
        with ExitStack() as stack:
            stack.enter_context(patch(FUNC_PATH + "._integrand_prob", side_effect=integrand))
            stack.enter_context(patch(FUNC_PATH + ".logger"))
            result = _integration_over_weighted_probs(zero_level, target_energy,
            _make_potential(), temperature)
        upper_bound = temperature * (1.0 - np.exp(-(target_energy - zero_level) / temperature))
        assert 0.0 <= result <= upper_bound, f"expected value in [0, {upper_bound}], got {result}"


# ===============================================================================
# --- _temperature_averaging_integral - normal behaviour and input validation ---
# ===============================================================================
class TestTemperatureAveragingIntegral:

    def test_forwards_arguments_to_the_integration(self):
        """ Verify that the arguments are forwarded to the integration """
        potential = _make_potential()
        with ExitStack() as stack:
            integ = _patch_integral_chain(stack)
            _temperature_averaging_integral(0.2, 1.5, [0.9],
            potential, 310.0, target_probability=0.9)
        integ.assert_called_once_with(0.2, 1.5, potential, 310.0)
        # it should be called once

    @pytest.mark.parametrize("target_energy", [0.5, 0.3])
    def test_invalid_target_energy_is_replaced_by_twice_zero_level(self, target_energy):
        """
        Verify that target_energy <= zero_level is replaced
        by 2 * zero_level before the integration
        """
        potential = _make_potential()
        with ExitStack() as stack:
            integ = _patch_integral_chain(stack)
            _temperature_averaging_integral(0.5, target_energy, [0.9],
            potential, 300.0, target_probability=0.9)
        integ.assert_called_once_with(0.5, 1.0, potential, 300.0)
        # it should be called once

    def test_replaced_target_energy_is_used_in_the_tail(self):
        """ Verify that the Boltzmann tail uses the replaced target energy """
        zero_level, temperature = 0.5, 2.0
        with ExitStack() as stack:
            _patch_integral_chain(stack, integral_value=0.4, k_b=1.0)
            result = _temperature_averaging_integral(zero_level, 0.1, [0.9],
            _make_potential(), temperature, target_probability=0.9)
        kT = 1.0 * temperature
        expected = 0.4 / kT + 0.95 * np.exp(-(2 * zero_level - zero_level) / kT)
        assert np.isclose(result, expected, rtol=1e-12), f"expected {expected}, got {result}"

    def test_uses_last_probability_when_target_was_not_reached(self):
        """
        Verify that when the last probability differs from the given target,
        it defines the tail
        """
        with ExitStack() as stack:
            _patch_integral_chain(stack, integral_value=0.5, k_b=1.0)
            result = _temperature_averaging_integral(0.0, 1.0, [0.2, 0.5],
            _make_potential(), 2.0, target_probability=0.99)
        expected = 0.5 / 2.0 + 0.75 * np.exp(-0.5)
        assert np.isclose(result, expected, rtol=1e-12), f"expected {expected}, got {result}"

    def test_uses_given_target_when_last_probability_matches_it(self):
        """ Verify that the given target defines the tail when the last probability reached it """
        with ExitStack() as stack:
            _patch_integral_chain(stack, integral_value=0.5, k_b=1.0)
            result = _temperature_averaging_integral(0.0, 1.0, [0.2, 0.99],
            _make_potential(), 2.0, target_probability=0.99)
        expected = 0.5 / 2.0 + 0.995 * np.exp(-0.5)
        assert np.isclose(result, expected, rtol=1e-12), f"expected {expected}, got {result}"

    @pytest.mark.parametrize("integral, zero_level, target_energy, temperature, last_prob", [
        (0.5, 0.0, 1.0, 2.0, 0.9), (0.01, 0.2, 2.0, 0.5, 0.99), (1e-9, 0.1, 3.0, 1.0, 0.6)])
    def test_matches_manual_formula(self, integral, zero_level, target_energy, temperature, last_prob):
        """ Verify that result = I / kT + 0.5 * (1 + P_t) * exp(-(E_t - E_0) / kT) """
        k_b = 2.0
        with ExitStack() as stack:
            _patch_integral_chain(stack, integral_value=integral, k_b=k_b)
            result = _temperature_averaging_integral(zero_level, target_energy, [last_prob],
            _make_potential(), temperature, target_probability=last_prob)
        kT = k_b * temperature
        expected = integral / kT + 0.5 * (1 + last_prob) * np.exp(-(target_energy - zero_level) / kT)
        assert np.isclose(result, expected, rtol=1e-12), f"expected {expected}, got {result}"

    @pytest.mark.parametrize("last_probability", [-0.5, 0.0, 1.0, 1.5])
    def test_raises_when_last_probability_out_of_range(self, last_probability):
        """
        Verify that a target_probability beyond the (0, 1) range
        raises an error. Expecting a ValueError
        """
        with ExitStack() as stack:
            integ = _patch_integral_chain(stack)
            with pytest.raises(ValueError, match = "Expected target_probability"):
                _temperature_averaging_integral(0.1, 0.5, [last_probability],
                _make_potential(), 300.0, target_probability=0.99)
        integ.assert_not_called()
        # it should not be called


# ===============================================================================
# --- temperature_averaging_tunneling - normal behaviour and input validation ---
# ===============================================================================
class TestTemperatureAveragingTunneling:

    def test_default_mode_is_finite_sum(self):
        """ Verify that finite_sum is used when prob_aver_mode is not given """
        levels = [_make_level()]
        with ExitStack() as stack:
            mocks = _patch_averaging_helpers(stack, prob=0.55)
            result = temperature_averaging_tunneling(1000.0, levels, _make_potential(), 310.0)
        mocks["finite"].assert_called_once_with(levels, 310.0)
        assert result == 0.55, f"expected {0.55}, got {result}"

    @pytest.mark.parametrize("mode, selected, uses_search", [
        ("finite_sum", "finite", False),
        ("infinite_sum", "infinite", True),
        ("integral", "integral", True)])
    def test_only_the_selected_helper_is_called(self, mode, selected, uses_search):
        """
        Verify that each mode calls its own helper only,
        and that the probability search is skipped for finite_sum
        """
        with ExitStack() as stack:
            mocks = _patch_averaging_helpers(stack)
            temperature_averaging_tunneling(800.0, [_make_level()], _make_potential(), 300.0,
                prob_aver_mode=mode)
        for name in ("finite", "infinite", "integral"):
            assert mocks[name].called == (name == selected), f"unexpected call state for {name}; test #1"
        assert mocks["reach"].called == uses_search, f"expected search called={uses_search}; test #2"

    @pytest.mark.parametrize("mode", ["bogus", "Finite_Sum", "", None])
    def test_raises_on_invalid_prob_aver_mode(self, mode):
        """ Verify that an invalid =prob_aver_mode raises an error. Expecting a ValueError """
        with pytest.raises(ValueError, match = "Expected prob_aver_mode"):
            temperature_averaging_tunneling(1000.0, [], _make_potential(),
            300.0, prob_aver_mode=mode)


# ===========================================================================
# --- arrhenius_rate - normal behaviour and input validation ---
# ===========================================================================
class TestArrheniusRate:

    def test_grid_runs_from_t_max_down_to_t_min(self):
        """ Verify that temperatures go from T_max down to T_min with the given step """
        _, result, _, _ = _run_arrhenius(mode="infinite_sum", T_min=-50.0, T_max=300.0, T_step=50.0)
        # the negative temperature is used to cover the cycle
        temperatures = [a[0] for a in _arrhenius_args(result)]
        expected = [300.0, 250.0, 200.0, 150.0, 100.0, 50.0]
        assert temperatures == expected, f"expected {1}, got {temperatures}"

    def test_includes_t_min_even_if_not_an_exact_step(self):
        """ Verify that T_min is appended when the step does not land exactly on it """
        _, result, _, _ = _run_arrhenius(mode="integral", T_min=210.0, T_max=300.0, T_step=50.0)
        temperatures = [a[0] for a in _arrhenius_args(result)]
        expected = [300.0, 250.0, 210.0]
        assert temperatures == expected, f"expected {expected}, got {temperatures}"

    def test_step_larger_than_the_range_gives_only_the_endpoints(self):
        """ Verify that a huge step leaves just T_max and T_min """
        _, result, _, _ = _run_arrhenius(T_min=200.0, T_max=300.0, T_step=500.0)
        temperatures = [a[0] for a in _arrhenius_args(result)]
        expected = [300.0, 200.0]
        assert temperatures == expected, f"expected {expected}, got {temperatures}"

    def test_default_mode_is_finite_sum(self):
        """ Verify that finite_sum is used when prob_aver_mode is not given """
        with ExitStack() as stack:
            mocks = _patch_arrhenius_chain(stack)
            arrhenius_rate([_make_level()], 1000.0, 200.0, 300.0, 50.0, _make_potential())
        assert mocks["finite"].call_count == 3, f"expected {3}, got {mocks['finite'].call_count}"
        mocks["reach"].assert_not_called()
        # it should not be called

    def test_arrhenius_result_stores_temperature_and_its_inverse(self):
        """ Verify that each ArrheniusResult gets the temperature and its reciprocal """
        _, result, _, _ = _run_arrhenius()
        for args in _arrhenius_args(result):
            assert np.isclose(args[1], 1.0 / args[0]), f"expected {1.0 / args[0]}, got {args[1]}"

    def test_log_reaction_rate_is_log_of_each_reaction_rate_value(self):
        """
        Verify that the stored log rate is
        the natural log of the rate computed for that temperature
        """
        _, result, _, _ = _run_arrhenius(rate_side_effect=[5.0, 7.0, 9.0])
        logs = [a[2] for a in _arrhenius_args(result)]
        expected = [np.log(5.0), np.log(7.0), np.log(9.0)]
        assert np.allclose(logs, expected), f"expected {expected}, got {logs}"

    def test_reaction_rate_gets_frequency_and_averaged_probability(self):
        """
        Verify that reaction_rate is called with the frequency
        and the averaged probability per temperature
        """
        mocks, _, _, _ = _run_arrhenius(prob=0.37)
        expected_calls = [call(1000.0, 0.37)] * 3
        assert mocks["rate"].call_args_list == expected_calls, f"expected {expected_calls}, "
        f"got {mocks['rate'].call_args_list}"

    def test_zero_reaction_rate_gives_negative_infinity_log(self):
        """ Verify that a zero reaction rate is stored as -inf instead of raising """
        _, result, _, _ = _run_arrhenius(rate=0.0)
        assert len(result) == 3, f"expected {3}, got {len(result)}"
        for args in _arrhenius_args(result):
            assert np.isneginf(args[2]), f"expected {-np.inf}, got {args[2]}"

    @pytest.mark.parametrize("bad_rate", [-1.0, np.inf, -np.inf, np.nan])
    def test_invalid_reaction_rate_raises_value_error(self, bad_rate):
        """
        Verify that a negative, infinite or nan reaction rate
        raises an error. Expecting a ValueError
        """
        with pytest.raises(ValueError, match = "Invalid reaction rate"):
            _run_arrhenius(rate=bad_rate)

    @pytest.mark.parametrize("mode", ["bogus", "Integral", "", None])
    def test_raises_on_invalid_prob_aver_mode(self, mode):
        """ Verify that an invalid prob_aver_mode raises an error. Expecting a ValueError """
        with pytest.raises(ValueError, match = "Expected prob_aver_mode"):
            arrhenius_rate([], 1000.0, 200.0, 400.0, 50.0,
            _make_potential(), prob_aver_mode=mode)