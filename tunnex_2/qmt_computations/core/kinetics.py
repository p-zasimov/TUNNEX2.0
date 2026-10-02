# --- Modules ---
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import quad
from tunnex_2.qmt_computations.constants_and_models.constants import light_speed, k_b, cm_m1_to_Hartree  # type: ignore
from tunnex_2.qmt_computations.constants_and_models.results import ArrheniusResult  # type: ignore
from tunnex_2.qmt_computations.core.turning_points import find_turning_points  # type: ignore
from tunnex_2.qmt_computations.core.wkb import compute_wkb  # type: ignore


# --- Defining the function to compute the transmission probability ---
def transmission_prob(wkb: float) -> float:
    if np.isinf(wkb):
        return 0.0
    prob = 1 / (1 + np.exp(2 * wkb))
    return prob  # A transmission probability [dimensionless]


# --- Defining the function to compute the reaction rate ---
def reaction_rate(freq: float, prob: float) -> float:
    return freq * light_speed * prob  # A reaction rate in seconds-1


# --- Defining the function to compute the half-life ---
def half_life(reaction_rate: float) -> float:
    if reaction_rate == 0:
        return np.inf
    return np.log(2) / reaction_rate  # A half-life in seconds


# --- Defining the average transmission probability (block: finite sum of vibrational levels) ---
def _boltzmann_average(energies: np.ndarray, tunneling_probabilities: np.ndarray, temperature: float) -> float:

    E0 = energies[0]

    boltzmann_factor = np.exp(-(energies - E0) / (k_b * temperature))

    sum_boltzmann = np.sum(boltzmann_factor)

    if sum_boltzmann == 0:
        raise ZeroDivisionError(f"Got zero value for Boltzmann sum during the temperature averaging. Please check your data.")
    # It should always be bigger than one, but, anyway, zero division check is useful in any place where it may occur

    return np.sum(boltzmann_factor * tunneling_probabilities) / sum_boltzmann # A temperature averaged probability [dimensionless]


# --- Returning the result for a temperature averaging (block: finite sum of vibrational levels) ---
def temperature_averaging_finite_sum(levels: object, temperature: float) -> float:
    vib_energies = np.array([level.vib_energy for level in levels])
    transmission_probs = np.array([level.transmission_probability for level in levels])

    return _boltzmann_average(vib_energies, transmission_probs, temperature) # A temperature averaged probability [dimensionless]


# --- Computing the transmission probabilities for all levels not over the barrier (block: infinite sum of vibrational levels) ---
def probabilities_for_infinite_sum(frequency : float, potential: object, max_levels: int = 1000) -> list[float]:

    probabilities = []
    regime = None
    i = 0

    while regime != "OVER THE BARRIER" and i < max_levels:
        level_energy = (i + 0.5) * frequency * cm_m1_to_Hartree
        i += 1

        turning_points = find_turning_points(level_energy, potential)
        regime = turning_points["regime"]

        if regime == "OVER THE BARRIER":
            break

        wkb = compute_wkb(turning_points, level_energy, potential)
        prob = transmission_prob(wkb)
        probabilities.append(prob)

    if regime != "OVER THE BARRIER" and i >= max_levels:
        print(f"Warning: maximum number of vibrational levels ({max_levels}) was reached.\n")

    return probabilities # A list of transmission probabilities [dimensionless]


# --- Returning the result for a temperature averaging (block: infinite sum of vibrational levels) ---
def temperature_averaging_infinite_sum(frequency : float, probabilities: list, temperature: float) -> float:

    # --- Defining the population redusement factor (q) based on Boltzmann's equations
    # and the number of vibrational levels under the barrier (n) ---
    q = np.exp(-(frequency * cm_m1_to_Hartree) / (k_b * temperature))
    n = len(probabilities)

    sum_boltzmann_fin = 0
    sum_prob_fin = 0

    # --- Counting the Boltzmann's factors and Boltzmann's weighted probabilities for the levels under the barrier ---
    for i, probability in enumerate(probabilities):
        boltzmann_factor = q ** i
        sum_boltzmann_fin += boltzmann_factor
        sum_prob_fin += boltzmann_factor * probability

    # --- Counting the rest over the barrier levels (up to infinity) and temerature averaged probability.
    # Assuming that for these levels the transmission probability is 0.5 ---
    sum_boltzmann_rest = q ** n / (1.0 - q)
    sum_boltzmann_inf = 1.0 / (1.0 - q)

    return (sum_prob_fin + 0.5 * sum_boltzmann_rest) / sum_boltzmann_inf # A temperature averaged probability [dimensionless]


# --- Computing the weighted probabilities for temperature averaging (block: integration from the first vibrational level to +inf) ---
# Assuming that the vibrational energy levels form a continuous distribution, the density of states p(E) is constant, i.e., p(E) = 1
def probabilities_for_integral(frequency : float, barrier_height : float, potential : object, grid_size : int = 100) -> tuple[np.ndarray | None, np.ndarray | None, float]:

    zero_level = 0.5 * frequency * cm_m1_to_Hartree

    if zero_level >= barrier_height:
        return None, None, zero_level
        # It means that the first energy level is bigger than the barrier, thus, returning 0.5 * kT
        # via returning tuple[None, None, float] here

    if grid_size < 2:
        raise ValueError(f"Energy grid size must be at least 2. Got {grid_size}.")

    energy_grid = np.linspace(zero_level, barrier_height, grid_size)

    probability_grid = np.empty_like(energy_grid)

    for i, energy_point in enumerate(energy_grid):
        turning_points = find_turning_points(energy_point, potential)
        wkb = compute_wkb(turning_points, energy_point, potential)
        probability_grid[i] = transmission_prob(wkb)
    
    return energy_grid, probability_grid, zero_level # An array of energy points, tunneling probabilities, and zero-level data


# --- Returning the integral of the interpolation function based on the computed weighted probabilities
# for temperature averaging (block: integration from the first vibrational level to +inf) ---
# Assuming that the vibrational energy levels form a continuous distribution, the density of states p(E) is constant, i.e., p(E) = 1
def integration_for_integral(energy_grid : np.ndarray, probability_grid : np.ndarray, zero_level : float, barrier_height : float, temperature: float) -> float:

    if energy_grid is None and probability_grid is None: # It means that the first energy level is bigger than the barrier, thus, returning 0.5 * kT
        sum_boltzmann = k_b * temperature
        return 0.5 * sum_boltzmann

    weighted_probability_grid = probability_grid * np.exp(-(energy_grid - zero_level) / (k_b * temperature))

    integrand = PchipInterpolator(energy_grid, weighted_probability_grid, extrapolate=False)

    integr_res = quad(integrand, zero_level, barrier_height, full_output=1)
    # The result gives an integrated value (result[0]) and a bunch of integration details.
    # We are interested only in an integrated value (result[0])

    return integr_res[0] # An integral of type: P = int[zero_level, barrier_height](P(E) * exp(-(E-zero_level)/(kT))dE].
    #It is proportional to the temperature averaged transmission probability in the [zero_level, barrier_height] range

# --- Returning the result for a temperature averaging (block: integration from the first vibrational level to +inf) ---
# Assuming that the vibrational energy levels form a continuous distribution, the density of states p(E) is constant, i.e., p(E) = 1
def temperature_averaging_integral(zero_level : float, barrier_height : float, barrier_integral : float, temperature: float) -> float:

    prob_over_the_barrier_scaled = 0.5 * np.exp(-(barrier_height - zero_level) / (k_b * temperature))
    # It is simply an integral of type: P = int[barrier_height, +inf](0.5 * exp(-(E)/(kT))dE] = 0.5 * kT * exp(-(barrier_height)/(kT)).
    # It is already scaled by the Boltzmann's sum: P = 0.5 * exp(-(barrier_height)/(kT)).
    # Assuming that for over the barrier regime the transmission probability is 0.5 ---

    sum_boltzmann = k_b * temperature
    result = barrier_integral / sum_boltzmann + prob_over_the_barrier_scaled

    return result # A temperature averaged probability [dimensionless]

    
# --- Defining the function to compute the average reaction rate for different temperatures ---
def arrhenius_rate(
        levels: object,
        freq: float,
        T_min: float,
        T_max: float,
        T_step: float,
        inf_sum_probs : list | None = None,
        zero_level : float | None = None,
        barrier_height : float | None = None,
        energy_grid : np.ndarray | None = None,
        probability_grid : np.ndarray | None = None) -> list[object]:

    arrhenius_matrix = []

    temperatures = list(np.arange(T_max, T_min, -T_step))

    if not np.isclose(temperatures[-1], T_min):
        temperatures.append(T_min)

    for arrhenius_T in temperatures:

        if arrhenius_T <= 0:
            break

        if inf_sum_probs is not None:
            arrhenius_prob = temperature_averaging_infinite_sum(freq, inf_sum_probs, arrhenius_T) # Infinite sum mode

        elif zero_level is not None:
            barrier_integr = integration_for_integral(energy_grid, probability_grid, zero_level, barrier_height, arrhenius_T) # Integral mode
            arrhenius_prob = temperature_averaging_integral(zero_level, barrier_height, barrier_integr, arrhenius_T)

        else:
            arrhenius_prob = temperature_averaging_finite_sum(levels, arrhenius_T) # Finite sum mode


        arrhenius_rate_value = reaction_rate(freq, arrhenius_prob)

        if np.isfinite(arrhenius_rate_value) and arrhenius_rate_value > 0:
            log_arrhenius_rate_value = np.log(arrhenius_rate_value)
        elif arrhenius_rate_value == 0:
            log_arrhenius_rate_value = -np.inf
        else:
            raise ValueError(f"Invalid reaction rate: {arrhenius_rate_value}. Please check your data.")

        arrhenius_matrix.append(ArrheniusResult(arrhenius_T, 1 / arrhenius_T, log_arrhenius_rate_value))

    return arrhenius_matrix
