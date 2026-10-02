# --- Modules ---
from pathlib import Path
from tunnex_2.irc_computations.constants_and_dataclasses.constants_and_settings import ( # type: ignore
    command_file_path,
    run_software_key_marker,
    corr_analysis_bool,
)
from tunnex_2.irc_computations.ancillary_functions.middle_level_functions import ( # type: ignore
    ts_optimization,
    eckart_react_prod_optimization,
    eckart_potential_results,
    opt_file_parsing,
    attempt_freq_and_levels,
    tunnex_input_writing,
)
from tunnex_2.irc_computations.ancillary_functions.interface import file_check # type: ignore
from tunnex_2.irc_computations.constants_and_dataclasses.dataclasses import MakeProjConfig # type: ignore


# --- Defining the function to compute the IRC data using Eckart potential ---
# --- For formulas please check: Johnston, H. S., & Heicklen, J. (1962). J. Phys. Chem., 66(3), 532-533 ---
def pipeline_eckart_potential(configs : MakeProjConfig) -> tuple[Path, Path]:

    # --- Defining some developer-useful features ---
    command_file = command_file_path
    run_software_key = run_software_key_marker
    corr_analysis = corr_analysis_bool

    # --- Defining the configurations of the pipeline_make_projections computations ---
    ts_guess_input = configs.ts_input_file
    prog_mode = configs.prog_mode
    react_guess_input = configs.react_input_file
    prod_guess_input = configs.prod_input_file
    ts_output_file_computed = configs.ts_computed_file
    react_output_file_computed = configs.react_computed_file
    prod_output_file_computed = configs.prod_computed_file

    # --- Optimization of the transition state geometry ---
    if ts_output_file_computed is None:
        ts_output_file = ts_optimization(ts_guess_input, command_file, prog_mode, run_software_key)
    else:
        ts_output_file = file_check(ts_output_file_computed)

    # --- Optimization of the reactant and product geometries ---
    if react_output_file_computed is None or prod_output_file_computed is None:
        react_output_file, prod_output_file = eckart_react_prod_optimization(react_guess_input, prod_guess_input, command_file, prog_mode, run_software_key)
    else:
        react_output_file = file_check(react_output_file_computed)
        prod_output_file = file_check(prod_output_file_computed)

    # --- Output file parsing for the optimized transition state, reactant, and product ---
    energy_points = opt_file_parsing(ts_output_file, react_output_file, prod_output_file, prog_mode)

    # --- Computing Eckart potential ---
    irc_data = eckart_potential_results(ts_output_file, energy_points, prog_mode)

    # --- Computation of the attempt frequencies and the number of vibrational levels for the temperature averaging ---
    potential_scal_factor_react, potential_scal_factor_prod = 1.0, 1.0
    attempt_freq_and_levels_data = attempt_freq_and_levels(ts_output_file, react_output_file, prod_output_file, irc_data.electronic_energies, potential_scal_factor_react, potential_scal_factor_prod, corr_analysis, prog_mode)

    # --- Writing the input files for the TUNNEX 2.0 computations (using the default potential_scal_factors here) ---
    tunnex_forward, tunnex_backward = tunnex_input_writing(ts_guess_input, irc_data, energy_points, attempt_freq_and_levels_data, potential_scal_factor_react, potential_scal_factor_prod)
    
    return tunnex_forward, tunnex_backward