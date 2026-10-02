# --- Modules ---
import numpy as np
from scipy.integrate import quad
from scipy.special import softmax, logsumexp, expit, log_expit

from tunnex_2.constants_and_dataclasses.constants_and_settings import ( # type: ignore
    TUNN_PROB_AVERAGING_ALLOWED_MODES,
    LIGHT_SPEED,
    K_B,
    CM_M1_TO_HARTREE)
from tunnex_2.constants_and_dataclasses.dataclasses import ArrheniusResult  # type: ignore

from tunnex_2.irc_computations.ancillary_functions.interface import setup_logger  # type: ignore

from tunnex_2.qmt_computations.core.turning_points import find_turning_points  # type: ignore
from tunnex_2.qmt_computations.core.wkb import compute_wkb, int_convergence_warning  # type: ignore


# --- Defining the logging function ---
logger = setup_logger()


# --- Defining the function to compute the transmission probability ---
def transmission_prob(wkb: float) -> float:
    if np.isposinf(wkb):
        return 0.0
    if np.isneginf(wkb):
        return 1.0
    prob = expit(-2.0 * wkb)  # expit(x) = 1/(1+exp(-x))
    return prob  # A transmission probability [dimensionless]


# --- Defining the function to compute the reaction rate ---
def reaction_rate(freq: float, prob: float) -> float:
    return freq * LIGHT_SPEED * 100 * prob  # A reaction rate in seconds^-1


# --- Defining the function to compute the half-life ---
def half_life(reaction_rate: float) -> float:
    if reaction_rate == 0:
        return np.inf
    return np.log(2) / reaction_rate  # A half-life in seconds


# --- Defining the average transmission probability (block: finite sum of vibrational levels) ---
def _boltzmann_average(energies: np.ndarray, tunneling_probabilities: np.ndarray,
    temperature: float) -> float:
    
    # --- Defining the input parameters ---
    E0 = energies[0]
    log_boltzmann_factor = -(energies - E0) / (K_B * temperature)

    # --- Computing the result using numerically stable functions ---
    boltzmann_factors_norm = softmax(log_boltzmann_factor)
    # boltzmann_factor_norm = np.exp(log_boltzmann_factor) / sum_boltzmann
    # sum_boltzmann = np.sum(boltzmann_factors)
    # (it is equal to one after the normalization)
    result = np.dot(boltzmann_factors_norm, tunneling_probabilities)
    # result = sum(boltzmann_factors_norm * tunneling_probabilities)
    # Since both boltzmann_factors_norm and tunneling_probabilities are within [0, 1],
    # the numerical precision is reasonable to avoid the usage of logsumexp
    # all the factors close to zero should just vanish
    
    return result
    # A temperature averaged probability [dimensionless]


# --- Returning the result for a temperature averaging (block: finite sum of vibrational levels) ---
def _temperature_averaging_finite_sum(levels: object, temperature: float) -> float:
    vib_energies = np.array([level.vib_energy for level in levels])
    transmission_probs = np.array([level.transmission_probability for level in levels])
    result = _boltzmann_average(vib_energies, transmission_probs, temperature=temperature)
    return result
    # A temperature averaged probability [dimensionless]


# --- Computing the transmission probabilities for all levels up to the target probability.
# It is a computationally expencive procedure, therefore,
# these computations are collected in a separate block ---
def _reaching_target_probability(frequency: float, potential: object,
    target_probability: float = 0.99, max_levels: int = 1000) -> tuple[list[float], float, float]:
    
    # --- Setting the input parameters ---
    probabilities = []
    prob = 0.0
    i = 0
    level_energy = None
    
    # --- The tunneling probability should be in (0, 1) range ---
    if not 0 < target_probability < 1:
        raise ValueError(f"Expected target_probability in (0, 1) range. Got {target_probability:.2f}")

    # --- The bare reasonable minimum is (0, 1, 2, 3) vibrational levels ---
    if max_levels < 4:
        raise ValueError(f"Expected max_levels >= 4. Got {max_levels}")

    # --- The data validation check guarantees that the loop with be executed at least once ---
    
    while prob < target_probability and i < max_levels:
        level_energy = (i + 0.5) * frequency * CM_M1_TO_HARTREE
        turning_points = find_turning_points(level_energy, potential)
        wkb = compute_wkb(turning_points, level_energy, potential)
        prob = transmission_prob(wkb)
        probabilities.append(prob)
        i += 1

    # --- Logging the case when the loop was terminated before reaching the target probability ---
    if prob < target_probability and i >= max_levels:
        logger.warning("Maximum number of vibrational levels (%d) reached before "
                       "the target transmission probability was achieved: "
                       "target=%.4f, achieved=%.4f.", max_levels, target_probability, prob)
    
    # --- Computing the zero energy level for later use ---
    zero_level = 0.5 * frequency * CM_M1_TO_HARTREE
    
    return probabilities, zero_level, level_energy
    # A list of transmission probabilities [dimensionless], level energy [Hartree],
    # and zero-level energy [Hartree]


# --- Returning the result for a temperature averaging (block: infinite sum of vibrational levels) ---
def _temperature_averaging_infinite_sum(frequency: float, probabilities: list[float],
    temperature: float, target_probability: float = 0.99) -> float:

    # --- If the target probability was not reached during the creation
    # of the list of probabilities, using the last reached value as a target probability.
    # This case should be already logged by the _reaching_target_probability function ---
    if not np.isclose(probabilities[-1], target_probability, atol=1e-12):
        target_probability = probabilities[-1]
    
    # --- The tunneling probability should be in (0, 1) range ---
    if not 0 < target_probability < 1:
        raise ValueError(f"Expected target_probability in (0, 1) range. Got {target_probability:.2f}") 
      
    # --- Computing the result using numerically stable functions ---
    n = len(probabilities)
    # it determines the range of energy levels
    # where the tunneling probabilities are computed explicitly
    log_q = -frequency * CM_M1_TO_HARTREE / (K_B * temperature)
    # it corresponds to q = np.exp(-(frequency * CM_M1_TO_HARTREE) / (K_B * temperature))
    one_minus_q = -np.expm1(log_q)
    # 1 - q is computed using a numerically stable function: np.expm1(x) = e(x) - 1
    # the sum_boltzmann of the infinite number of equidistant (q) levels is 1 / (1 - q)
    # the series should always converge, since frequency > 0 and T > 0
    boltzmann_factor_norm = one_minus_q * np.exp(np.arange(n) * log_q)
    # each boltzmann_factor is computed as q ** i, the normalization requires them
    # to be multiplied by (1 - q)
    # overflow is not possible here, since log_q < 0 (always), while underflow
    # does not affect the result (because of the small weight of the term).
    # The same applies to boltzmann_factor_tail_norm
    boltzmann_factor_tail_norm = np.exp(n * log_q)
    # as a sum of an infinite geometric series, boltzmann_factor_tail is q ** n / (1 - q)
    # after the normalization boltzmann_factor_tail_norm is q ** n
    probability_tail = 0.5 * (1 + target_probability)
    # it is a compromise between the actually achieved result (target_probability) and
    # the probability at the infinite energy (1). The former is good for a function continuity,
    # however, if the actually reached probability is small (e.g. 0.6), it may be an inappropriate
    # description of the probabilities at higher energies which go to unity. The probability of 1
    # is a true limit of the transmission probability at the infinite energy, but it may create a noticeable
    # function discontinuity. Therefore, the value of 0.5 * (1 + target_probability) is used.
    # for most of the cases, where target_probability is big enough (e.g. 0.99), it should not make
    # a significant difference
    result = np.dot(boltzmann_factor_norm, probabilities) + probability_tail * boltzmann_factor_tail_norm
    # result = sum(boltzmann_factors_norm * tunneling_probabilities) + 
    # + probability_tail * boltzmann_factor_tail_norm

    return result
    # A temperature averaged probability [dimensionless]


# --- Computing the weighted probabilities for temperature averaging (block: integration from the first vibrational level to +inf).
# Assuming that the vibrational energy levels form a continuous distribution, the density of states p(E) is constant, i.e., p(E) = 1.
# It is a computationally expencive procedure, therefore,
# these computations are collected in a separate block ---
def _integrand_prob(energy: float, zero_level: float, potential: object, temperature: float) -> float:
    
    # --- Computing the level energy ---
    level_energy = energy - zero_level # it is always > 0

    # --- Computing the logarithm of a Boltzmann factor ---
    log_boltzmann = -level_energy / (K_B * temperature)

    # --- Computing the logarithm of a tunneling probability ---
    turning_points = find_turning_points(energy, potential)
    wkb = compute_wkb(turning_points, energy, potential)
    log_prob = log_expit(-2.0 * wkb)
    # the input energies for probabilities should not be shifted by the zero_level value
    # expit(x) = 1/(1+exp(-x)); log_expit = log(expit(x))
    
    # --- Computing the weighted tunneling probability ---
    result = np.exp(log_prob + log_boltzmann)
    
    return result
    # A weighted probability for a given energy [dimensionless]


# --- Returning the integral based on the computed weighted probabilities
# for temperature averaging (block: integration from the first vibrational level to +inf)
# Assuming that the vibrational energy levels form a continuous distribution,
# the density of states p(E) is constant, i.e., p(E) = 1 ---
def _integration_over_weighted_probs(zero_level: float, target_energy: float,
    potential: object, temperature: float) -> float:
    
    # --- Defining the function for the integration ---
    integrand_func_prob = lambda level_energy: _integrand_prob(level_energy,
        zero_level, potential, temperature)
    
    # --- Computing the integral and logging the integration warnings ---
    result = quad(integrand_func_prob, zero_level, target_energy, full_output=1)
    int_convergence_warning(result, logger)
    ans = result[0]
    # The result gives an integrated value (result[0]) and a bunch of integration details.
    # Storing only the former value
    
    return ans
    # An integral of type: P = int[zero_level, target_energy](P(E) * exp(-(E-zero_level)/(kT))dE].
    # It is proportional to the temperature averaged transmission probability
    # in the [zero_level, target_energy] range


# --- Returning the result for a temperature averaging (block: integration from the first vibrational level to +inf).
# Assuming that the vibrational energy levels form a continuous distribution,
# the density of states p(E) is constant, i.e., p(E) = 1 ---
def _temperature_averaging_integral(zero_level: float, target_energy: float,
    probabilities: list[float], potential: object, temperature: float,
    target_probability: float = 0.99) -> float:
    
    # --- If the target probability was not reached during the creation
    # of the list of probabilities, using the last reached value as a target probability.
    # This case should be already logged by the _reaching_target_probability function ---
    if not np.isclose(probabilities[-1], target_probability, atol=1e-12):
        target_probability = probabilities[-1]
    
    # --- The tunneling probability should be in (0, 1) range ---
    if not 0 < target_probability < 1:
        raise ValueError(f"Expected target_probability in (0, 1) range. Got {target_probability:.2f}") 

    # --- Protecting the code from the wrong integration limits (zero_level is always > 0) ---
    if target_energy <= zero_level:
        target_energy = 2 * zero_level
    
    # --- Computing the Boltzmann's sum ---
    sum_boltzmann = K_B * temperature
    
    # --- Computing the transmission probability integral and probability_tail ---
    integr_prob = _integration_over_weighted_probs(zero_level, target_energy, potential, temperature)
    probability_tail = 0.5 * (1 + target_probability)
    # it is a compromise between the actually achieved result (target_probability) and
    # the probability at the infinite energy (1). The former is good for a function continuity,
    # however, if the actually reached probability is small (e.g. 0.6), it may be an inappropriate
    # description of the probabilities at higher energies which go to unity. The probability of 1
    # is a true limit of the transmission probability at the infinite energy, but it may create a noticeable
    # function discontinuity. Therefore, the value of 0.5 * (1 + target_probability) is used.
    # for most of the cases, where target_probability is big enough (e.g. 0.99), it should not make
    # a significant difference
    
    # --- Computing the temperature averaged transmission probability ---
    log_boltzmann_tail = (np.log(probability_tail) - (target_energy - zero_level) / (K_B * temperature))
    # It corresponds to the energies higher than the target energy up to +inf.
    # The formula is P_over = probability_tail * exp(- (target_energy - zero_level) / (K_B * temperature))
    log_integral_term = (np.log(integr_prob) - np.log(sum_boltzmann))
    # It corresponds to the energies from the zero level to target energy.
    # The formula is P_under = integr_prob / (K_B * temperature)
    log_result = logsumexp([log_integral_term, log_boltzmann_tail])
    result = np.exp(log_result)
    # The formula is P = P_under + P_over

    return result
    # A temperature averaged probability [dimensionless]


# --- Returning the result for a temperature averaging (block: orchestration) ---
def temperature_averaging_tunneling(frequency: float, level_result_data: object, potential: object,
    temperature: float, prob_aver_mode: str = "finite_sum", target_probability: float = 0.99,
    max_levels: int = 1000) -> float:

    # --- Checking the input data ---
    allowed_modes = TUNN_PROB_AVERAGING_ALLOWED_MODES.values()
    if prob_aver_mode not in allowed_modes:
        raise ValueError(f"Expected prob_aver_mode in {allowed_modes}. Got {prob_aver_mode}.")

    # --- Computing the probabilities ---
    if prob_aver_mode == "finite_sum":
        prob_aver = _temperature_averaging_finite_sum(level_result_data, temperature)
    elif prob_aver_mode == "infinite_sum":
        probabilities, _, _ = _reaching_target_probability(frequency, potential,
            target_probability=target_probability, max_levels=max_levels)
        prob_aver = _temperature_averaging_infinite_sum(frequency, probabilities, temperature,
            target_probability=target_probability)
    else: # prob_aver_mode == "integral":
        probabilities, zero_level, target_energy = _reaching_target_probability(frequency, potential,
            target_probability=target_probability, max_levels=max_levels)
        prob_aver = _temperature_averaging_integral(zero_level, target_energy,
            probabilities, potential, temperature, target_probability=target_probability)
    
    return prob_aver


# --- Defining the function to compute the average reaction rate for different temperatures ---
def arrhenius_rate(levels: object, freq: float, T_min: float, T_max: float, T_step: float,
    potential: object, prob_aver_mode: str = "finite_sum", target_probability: float = 0.99,
    max_levels: int = 1000) -> list[object]:

    # --- Checking the input data ---
    allowed_modes = TUNN_PROB_AVERAGING_ALLOWED_MODES.values()
    if prob_aver_mode not in allowed_modes:
        raise ValueError(f"Expected prob_aver_mode in {allowed_modes}. Got {prob_aver_mode}.")

    if prob_aver_mode != "finite_sum":
        probabilities, zero_level, target_energy = _reaching_target_probability(freq, potential,
            target_probability=target_probability, max_levels=max_levels)
    
    arrhenius_matrix = []

    temperatures = list(np.arange(T_max, T_min, -T_step))

    if not temperatures or not np.isclose(temperatures[-1], T_min, atol=1e-12):
        temperatures.append(T_min)

    # --- Computing the Arrhenius data ---
    for arrhenius_T in temperatures:

        if arrhenius_T <= 0:
            break      
        
        if prob_aver_mode == "finite_sum":
            arrhenius_prob = _temperature_averaging_finite_sum(levels, arrhenius_T)
        elif prob_aver_mode == "infinite_sum":
            arrhenius_prob = _temperature_averaging_infinite_sum(freq, probabilities, arrhenius_T,
            target_probability=target_probability)
        else:
            arrhenius_prob = _temperature_averaging_integral(zero_level, target_energy,
            probabilities, potential, arrhenius_T, target_probability=target_probability)

        arrhenius_rate_value = reaction_rate(freq, arrhenius_prob)

        if np.isfinite(arrhenius_rate_value) and arrhenius_rate_value > 0:
            log_arrhenius_rate_value = np.log(arrhenius_rate_value)
        elif arrhenius_rate_value == 0:
            log_arrhenius_rate_value = -np.inf
        else:
            raise ValueError(f"Invalid reaction rate: {arrhenius_rate_value}. Please check your data.")

        arrhenius_matrix.append(ArrheniusResult(arrhenius_T, 1 / arrhenius_T, log_arrhenius_rate_value))

    return arrhenius_matrix