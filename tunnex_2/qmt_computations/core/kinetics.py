# --- Modules ---
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import quad
from scipy.special import logsumexp, expit

from tunnex_2.constants_and_dataclasses.constants_and_settings import ( # type: ignore
    LIGHT_SPEED,
    K_B,
    CM_M1_TO_HARTREE)
from tunnex_2.constants_and_dataclasses.dataclasses import ArrheniusResult  # type: ignore

from tunnex_2.irc_computations.ancillary_functions.interface import setup_logger  # type: ignore

from tunnex_2.qmt_computations.core.turning_points import find_turning_points  # type: ignore
from tunnex_2.qmt_computations.core.wkb import compute_wkb  # type: ignore


# --- Defining the logging function ---
logger = setup_logger()


# --- Defining the function to compute the transmission probability ---
def transmission_prob(wkb: float) -> float:
    if np.isinf(wkb):
        return 0.0
    prob = expit(-2 * wkb) # expit(x) = 1/(1+exp(-x))
    return prob  # A transmission probability [dimensionless]


# --- Defining the function to compute the reaction rate ---
def reaction_rate(freq: float, prob: float) -> float:
    return freq * LIGHT_SPEED * 100 * prob  # A reaction rate in seconds-1


# --- Defining the function to compute the half-life ---
def half_life(reaction_rate: float) -> float:
    if reaction_rate == 0:
        return np.inf
    return np.log(2) / reaction_rate  # A half-life in seconds


# --- Defining the average transmission probability (block: finite sum of vibrational levels) ---
def _boltzmann_average(energies: np.ndarray, tunneling_probabilities: np.ndarray,
    temperature: float, num_stab: bool = False) -> float:
    
    E0 = energies[0]
    log_boltzmann = -(energies - E0) / (K_B * temperature)

    if not num_stab:
        boltzmann_factor = np.exp(log_boltzmann)
        sum_boltzmann = np.sum(boltzmann_factor)

        if np.isclose(sum_boltzmann, 0.0, atol=1e-12):
            raise ValueError(f"Expected non-zero Boltzmann sum, got {sum_boltzmann:.3g}.")
        # It should always be bigger than one, but, anyway, zero division check is useful in any place where it may occur

        result = np.sum(boltzmann_factor * tunneling_probabilities) / sum_boltzmann
    else: # more numerical stable computations (experimental)
        log_numerator = logsumexp(log_boltzmann + np.log(tunneling_probabilities))
        log_denominator = logsumexp(log_boltzmann)
        result = np.exp(log_numerator - log_denominator)

    return result # A temperature averaged probability [dimensionless]


# --- Returning the result for a temperature averaging (block: finite sum of vibrational levels) ---
def _temperature_averaging_finite_sum(levels: object, temperature: float, num_stab: bool = False) -> float:
    vib_energies = np.array([level.vib_energy for level in levels])
    transmission_probs = np.array([level.transmission_probability for level in levels])
    result = _boltzmann_average(vib_energies, transmission_probs, temperature=temperature, num_stab=num_stab)
    return result # A temperature averaged probability [dimensionless]


# --- Computing the transmission probabilities for all levels up to the target probability ---
# It is very computationally expencive, thus, separated
def _reaching_target_probability(frequency: float, potential: object,
    target_probability: float = 0.99, max_levels: int = 1000) -> tuple[list[float], float]:

    probabilities = []
    prob = 0.0
    i = 0
    level_energy = None

    if not 0 < target_probability <= 1:
        raise ValueError(f"Expected target_probability in [0, 1]. Got {target_probability:.2f}")

    while prob < target_probability and i < max_levels:
        level_energy = (i + 0.5) * frequency * CM_M1_TO_HARTREE
        i += 1

        turning_points = find_turning_points(level_energy, potential)
        wkb = compute_wkb(turning_points, level_energy, potential)
        prob = transmission_prob(wkb)
        probabilities.append(prob)

    if prob < target_probability and i >= max_levels:
        logger.warning("Maximum number of vibrational levels (%d) reached before "
                       "the target transmission probability was achieved: "
                       "target=%.4f, achieved=%.4f.", max_levels, target_probability, prob)

    # The loop was not executed, so return the energy of the first vibrational level
    if level_energy is None:
        level_energy = (i + 0.5) * frequency * CM_M1_TO_HARTREE
        probabilities.append(prob)

    return probabilities, level_energy # A list of transmission probabilities [dimensionless], level energy [Hartree]


# --- Returning the result for a temperature averaging (block: infinite sum of vibrational levels) ---
def _temperature_averaging_infinite_sum(frequency: float, probabilities: list[float],
    temperature: float, target_probability: float = 0.99, num_stab: bool = False) -> float:

    if not np.isclose(probabilities[-1], target_probability, atol=1e-12):
        target_probability = probabilities[-1]

    if not 0 < target_probability <= 1:
        raise ValueError(f"Expected target_probability in [0, 1]. Got {target_probability:.2f}") 
    
    if not num_stab:
        # --- Defining the population redusement factor (q) based on Boltzmann's equations
        # and the number of vibrational levels under the barrier (n) ---
        q = np.exp(-(frequency * CM_M1_TO_HARTREE) / (K_B * temperature))
        n = len(probabilities)
        sum_prob_fin = 0

        # --- Counting the Boltzmann's factors and Boltzmann's weighted probabilities for the levels under the barrier ---
        for i, probability in enumerate(probabilities):
            boltzmann_factor = q ** i
            sum_prob_fin += boltzmann_factor * probability

        # --- Counting the rest over the barrier levels (up to infinity) and temerature averaged probability.
        # Assuming that for these levels the transmission probability is over_the_barrier_probability ---
        sum_boltzmann_rest = q ** n / (1.0 - q)
        sum_boltzmann_inf = 1.0 / (1.0 - q)
        result = (sum_prob_fin + target_probability * sum_boltzmann_rest) / sum_boltzmann_inf
    else: # more numerical stable computations (experimental)
        log_q = -frequency * CM_M1_TO_HARTREE / (K_B * temperature)
        n = len(probabilities)

        # According to the properties of logarithms: log(q ** i) = i * log(q)
        indices = np.arange(n)
        log_boltzmann = indices * log_q

        # Sum of Boltzmann-weighted probabilities: sum(q ** i * probabilities[i])
        log_probabilities = np.log(probabilities)
        log_sum_prob_fin = logsumexp(log_boltzmann + log_probabilities)

        # Remaining infinite tail: q ** n / (1 - q); np.log1p calculates log(1 + x)
        log_sum_boltzmann_rest = (n * log_q - np.log1p(-np.exp(log_q)))

        # Infinite Boltzmann sum: 1 / (1 - q); np.log1p calculates log(1 + x)
        log_sum_boltzmann_inf = (-np.log1p(-np.exp(log_q)))

        # Numerator: sum_prob_fin + target_probability * sum_boltzmann_rest
        log_numerator = logsumexp([log_sum_prob_fin, np.log(target_probability) + log_sum_boltzmann_rest])

        # Average probability
        result =  np.exp(log_numerator - log_sum_boltzmann_inf)

    return result # A temperature averaged probability [dimensionless]


# --- Computing the weighted probabilities for temperature averaging (block: integration from the first vibrational level to +inf) ---
# Assuming that the vibrational energy levels form a continuous distribution, the density of states p(E) is constant, i.e., p(E) = 1
# It is very computationally expencive, thus, separated
def _probabilities_for_integral(frequency: float, potential: object, target_probability: float = 0.99,
    max_levels: int = 1000, grid_size: int = 100) -> tuple[np.ndarray, np.ndarray, float, float]:

    if grid_size < 3:
        raise ValueError(f"Energy grid size must be at least 3. Got {grid_size}.")

    if frequency <= 0:
        raise ValueError(f"Expected a positive frequency. Got {frequency:.2f}.")

    _, target_energy = _reaching_target_probability(frequency, potential,
    target_probability=target_probability, max_levels=max_levels)

    zero_level = 0.5 * frequency * CM_M1_TO_HARTREE

    if np.isclose(target_energy, zero_level, atol=1e-12):
        target_energy = 2 * zero_level

    energy_grid = np.linspace(zero_level, target_energy, grid_size)

    probability_grid = np.empty_like(energy_grid)

    for i, energy_point in enumerate(energy_grid):
        turning_points = find_turning_points(energy_point, potential)
        wkb = compute_wkb(turning_points, energy_point, potential)
        probability_grid[i] = transmission_prob(wkb)
    
    return energy_grid, probability_grid, zero_level, target_energy
    # An array of energy points, tunneling probabilities, and zero-level data


# --- Returning the integral of the interpolation function based on the computed weighted probabilities
# for temperature averaging (block: integration from the first vibrational level to +inf) ---
# Assuming that the vibrational energy levels form a continuous distribution, the density of states p(E) is constant, i.e., p(E) = 1
def _integration_over_weighted_probs(energy_grid: np.ndarray, probability_grid: np.ndarray,
    zero_level: float, target_energy: float, temperature: float) -> tuple[float, float]:
    
    # Here is our function of the probability distribution to integrate.
    # Dealing with the logarithm inperpolation, assuming that the probability behaves like an exponent:
    # The exponential behaviour is expected from the P[(E] = 1 / (1 + exp(2*theta[E]))
    log_probability_interpolator = PchipInterpolator(energy_grid, np.log(probability_grid), extrapolate=False)

    def _integrand(energy):
        log_probability = log_probability_interpolator(energy)
        log_boltzmann = -(energy - zero_level) / (K_B * temperature)
        return np.exp(log_probability + log_boltzmann)

    integr_res = quad(_integrand, zero_level, target_energy, full_output=1)

    # The result gives an integrated value (result[0]) and a bunch of integration details.
    # We are interested only in an integrated value (result[0])

    return integr_res[0], probability_grid[-1] # An integral of type: P = int[zero_level,
    # target_energy](P(E) * exp(-(E-zero_level)/(kT))dE].
    # It is proportional to the temperature averaged transmission probability in the [zero_level, target_energy] range

# --- Returning the result for a temperature averaging (block: integration from the first vibrational level to +inf) ---
# Assuming that the vibrational energy levels form a continuous distribution, the density of states p(E) is constant, i.e., p(E) = 1
def _temperature_averaging_integral(zero_level: float, target_energy: float,
    barrier_integral_and_prob: tuple[float, float], temperature: float,
    target_probability: float = 0.99, num_stab: bool = False) -> float:
    
    if not np.isclose(barrier_integral_and_prob[-1], target_probability, atol=1e-12):
        target_probability = barrier_integral_and_prob[-1]
    
    if not 0 < target_probability <= 1:
        raise ValueError(f"Expected target_probability in [0, 1]. Got {target_probability:.2f}")

    sum_boltzmann = K_B * temperature
    
    if not num_stab:
        prob_over_the_barrier_scaled = target_probability * np.exp(-(target_energy - zero_level) / (K_B * temperature))
        # It is simply an integral of type: P = int[barrier_height,
        # +inf](target_probability * exp(-(E)/(kT))dE] = 0.5 * kT * exp(-(barrier_height)/(kT)).
        # It is already scaled by the Boltzmann's sum: P = target_probability * exp(-(barrier_height)/(kT)).
        # Assuming that for over the barrier regime the transmission probability is target_probability

        result = barrier_integral_and_prob[0] / sum_boltzmann + prob_over_the_barrier_scaled
    else:
        log_over_barrier_term = (np.log(target_probability) - (target_energy - zero_level) / (K_B * temperature))
        log_integral_term = (np.log(barrier_integral_and_prob[0]) - np.log(sum_boltzmann))
        log_result = logsumexp([log_integral_term, log_over_barrier_term])
        result = np.exp(log_result)

    return result # A temperature averaged probability [dimensionless]


# --- Returning the result for a temperature averaging (block: orchestration) ---
def temperature_averaging_tunneling(frequency: float, level_result_data: object, potential: object,
    temperature: float, prob_aver_mode: str = "finite_sum", target_probability: float = 0.99,
    max_levels: int = 1000, grid_size: int = 100, num_stab: bool = False) -> float:

    allowed_modes = ("finite_sum", "infinite_sum", "integral")
    if prob_aver_mode not in allowed_modes:
        raise ValueError(f"Expected prob_aver_mode in {allowed_modes}. Got {prob_aver_mode}.")
    
    if prob_aver_mode == "finite_sum":
        prob_aver = _temperature_averaging_finite_sum(level_result_data, temperature, num_stab=num_stab)
    elif prob_aver_mode == "infinite_sum":
        probabilities, _ = _reaching_target_probability(frequency, potential,
            target_probability=target_probability, max_levels=max_levels)
        prob_aver = _temperature_averaging_infinite_sum(frequency, probabilities, temperature,
            target_probability=target_probability, num_stab=num_stab)
    else:
        energy_grid, probability_grid, zero_level, target_energy = _probabilities_for_integral(
            frequency, potential, target_probability=target_probability, max_levels=max_levels, grid_size=grid_size)
        barrier_integral_and_prob = _integration_over_weighted_probs(energy_grid, probability_grid,
            zero_level, target_energy, temperature)
        prob_aver = _temperature_averaging_integral(zero_level, target_energy, barrier_integral_and_prob,
            temperature, target_probability=target_probability, num_stab=num_stab)
    
    return prob_aver


# --- Defining the function to compute the average reaction rate for different temperatures ---
def arrhenius_rate(levels: object, freq: float, T_min: float, T_max: float, T_step: float,
    potential: object, prob_aver_mode: str = "finite_sum", target_probability: float = 0.99,
    max_levels: int = 1000, grid_size: int = 100, num_stab: bool = False) -> list[object]:

    allowed_modes = ("finite_sum", "infinite_sum", "integral")
    if prob_aver_mode not in allowed_modes:
        raise ValueError(f"Expected prob_aver_mode in {allowed_modes}. Got {prob_aver_mode}.")

    if prob_aver_mode == "infinite_sum":
        probabilities, _ = _reaching_target_probability(freq, potential,
            target_probability=target_probability, max_levels=max_levels)
    
    if prob_aver_mode == "integral":
        energy_grid, probability_grid, zero_level, target_energy = _probabilities_for_integral(freq, potential,
            target_probability=target_probability, max_levels=max_levels, grid_size=grid_size)
    
    arrhenius_matrix = []

    temperatures = list(np.arange(T_max, T_min, -T_step))

    if not np.isclose(temperatures[-1], T_min, atol=1e-12):
        temperatures.append(T_min)

    for arrhenius_T in temperatures:

        if arrhenius_T <= 0:
            break      
        
        if prob_aver_mode == "finite_sum":
            arrhenius_prob = _temperature_averaging_finite_sum(levels, arrhenius_T, num_stab=num_stab)
        elif prob_aver_mode == "infinite_sum":
            arrhenius_prob = _temperature_averaging_infinite_sum(freq, probabilities, arrhenius_T,
                target_probability=target_probability, num_stab=num_stab)
        else:
            barrier_integral_and_prob = _integration_over_weighted_probs(energy_grid, probability_grid,
                zero_level, target_energy, arrhenius_T)
            arrhenius_prob = _temperature_averaging_integral(zero_level, target_energy, barrier_integral_and_prob,
                arrhenius_T, target_probability=target_probability, num_stab=num_stab)

        arrhenius_rate_value = reaction_rate(freq, arrhenius_prob)

        if np.isfinite(arrhenius_rate_value) and arrhenius_rate_value > 0:
            log_arrhenius_rate_value = np.log(arrhenius_rate_value)
        elif arrhenius_rate_value == 0:
            log_arrhenius_rate_value = -np.inf
        else:
            raise ValueError(f"Invalid reaction rate: {arrhenius_rate_value}. Please check your data.")

        arrhenius_matrix.append(ArrheniusResult(arrhenius_T, 1 / arrhenius_T, log_arrhenius_rate_value))

    return arrhenius_matrix