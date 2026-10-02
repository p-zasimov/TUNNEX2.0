# --- Modules ---
import numpy as np
from tunnex_2.qmt_computations.constants_and_models.constants import cm_m1_to_Hartree, hour, day, year, grid_size  # type: ignore
from tunnex_2.qmt_computations.constants_and_models.results import LevelResult, TemperatureAverResult  # type: ignore
from tunnex_2.qmt_computations.core.turning_points import find_turning_points  # type: ignore
from tunnex_2.qmt_computations.core.wkb import compute_wkb  # type: ignore
from tunnex_2.qmt_computations.core.kinetics import (  # type: ignore
    transmission_prob,
    reaction_rate,
    half_life,
    temperature_averaging_finite_sum,
    probabilities_for_infinite_sum,
    temperature_averaging_infinite_sum,
    probabilities_for_integral,
    integration_for_integral,
    temperature_averaging_integral,
    arrhenius_rate,
)
from tunnex_2.qmt_computations.core.potential import find_potential_maximum  # type: ignore


# --- Defining the pipeline function to compute the QMT half-lives ---
def qmt_computations_pipeline(params: object, potential: object, prob_aver_mode : str = 'finite_sum') -> tuple[np.array, list[object], object, object]:
    
    # --- Checking the parameters ---
    if prob_aver_mode not in ('finite_sum', 'infinite_sum', 'integral'):
        raise ValueError(f"Expected 'finite_sum', 'infinite_sum', 'integral' as a mode. Got {prob_aver_mode}.")

    # --- Storing the interpolated ZPVE-corrected IRC  ---
    x_grid = np.linspace(potential.irc_x_min, potential.irc_x_max, grid_size)
    dtype = np.dtype([("irc_value", float), ("energy_value", float)])
    interpolated_irc_grid = np.array(list(zip(x_grid, potential(x_grid))), dtype=dtype)

    # --- Filling the matrix of the results for the given vibrational levels ---
    level_result_data = []

    for i in range(params.the_upper_level_to_compute + 1):
        # --- Computing vibrational energy levels ---
        level_energy = (i + 0.5) * params.freq * cm_m1_to_Hartree

        # --- Defining the function which computes the turning points ---
        turning_points = find_turning_points(level_energy, potential)

        # --- Computing the WKB-integrals ---
        wkb = compute_wkb(turning_points, level_energy, potential)

        # --- Computing the transmission probabilities ---
        prob = transmission_prob(wkb)

        # --- Computing the reaction rate in seconds-1 ---
        reaction_rate_value = reaction_rate(params.freq, prob)

        # --- Computing the half-life in seconds ---
        half_life_value = half_life(reaction_rate_value)

        # --- Taking the computed vibrational energy level indexes and levels and appending them with the computed parameters ---
        level_result_data.append(
            LevelResult(
                i,
                level_energy,
                turning_points["left"],
                turning_points["right"],
                wkb,
                prob,
                reaction_rate_value,
                half_life_value,
                half_life_value * hour,
                half_life_value * day,
                half_life_value * year,
            )
        )

    # --- Computing the temperature-averaged transmission probabilities ---
    if prob_aver_mode == 'finite_sum':
        prob_aver = temperature_averaging_finite_sum(level_result_data, params.T)

    elif prob_aver_mode == 'infinite_sum':
        probabilities_finite = probabilities_for_infinite_sum(params.freq, potential)
        prob_aver = temperature_averaging_infinite_sum(params.freq, probabilities_finite, params.T)
    else:
        barrier_height = max(potential(potential.irc_x_ts), find_potential_maximum(potential)[1])
        energy_grid, probability_grid, zero_level = probabilities_for_integral(params.freq, barrier_height, potential)
        barrier_integr = integration_for_integral(energy_grid, probability_grid, zero_level, barrier_height, params.T)
        prob_aver = temperature_averaging_integral(zero_level, barrier_height, barrier_integr, params.T)
    
    # --- Computing the temperature-averaged reaction rate in seconds-1 ---
    reaction_rate_aver = reaction_rate(params.freq, prob_aver)

    # --- Computing the temparature-averaged half-life in seconds ---
    half_life_aver = half_life(reaction_rate_aver)

    # --- Storing the temparature-averaged parameters ---
    results_aver = TemperatureAverResult(
        prob_aver,
        reaction_rate_aver,
        half_life_aver,
        half_life_aver * hour,
        half_life_aver * day,
        half_life_aver * year,
    )

    # --- Storing the arrhenius graph parameters (tunneling only!) ---
    if prob_aver_mode == 'finite_sum':
        arrhenius_rate_vals = arrhenius_rate(
            level_result_data,
            params.freq,
            params.T_arrhenius_min,
            params.T_arrhenius_max,
            params.T_arrhenius_step,
        )
    elif prob_aver_mode == 'infinite_sum':
        arrhenius_rate_vals = arrhenius_rate(
            level_result_data,
            params.freq,
            params.T_arrhenius_min,
            params.T_arrhenius_max,
            params.T_arrhenius_step,
            probabilities_finite,
        )
    else:
        arrhenius_rate_vals = arrhenius_rate(
            level_result_data,
            params.freq,
            params.T_arrhenius_min,
            params.T_arrhenius_max,
            params.T_arrhenius_step,
            None, # It was set to None to avoid the lauch of the 'infinite_sum' mode
            zero_level,
            barrier_height,
            energy_grid,
            probability_grid,
        )

    return interpolated_irc_grid, level_result_data, results_aver, arrhenius_rate_vals
