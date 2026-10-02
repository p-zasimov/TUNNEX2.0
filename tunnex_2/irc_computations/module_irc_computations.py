# --- Modules ---
from pathlib import Path
from tunnex_2.irc_computations.constants_and_dataclasses.constants_and_settings import ( # type: ignore
    command_file_path,
    run_software_key_marker,
    corr_analysis_bool,
)
from tunnex_2.irc_computations.ancillary_functions.middle_level_functions import ( # type: ignore
    ts_optimization,
    irc_computations,
    irc_file_parsing,
    react_prod_optimization,
    opt_file_parsing,
    attempt_freq_and_levels,
    potential_scal_factor_comp,
    tunnex_input_writing,
)
from tunnex_2.irc_computations.ancillary_functions.interface import file_check # type: ignore
from tunnex_2.irc_computations.constants_and_dataclasses.dataclasses import MakeProjConfig # type: ignore


# --- Defining the function to compute the IRC data using Gaussian or ORCA ---
def pipeline_irc_computations(configs : MakeProjConfig) -> tuple[Path, Path]:

    # --- Defining some developer-useful features ---
    command_file = command_file_path
    run_software_key = run_software_key_marker
    corr_analysis = corr_analysis_bool

    # --- Defining the configurations of the pipeline_make_projections computations ---
    ts_guess_input = configs.ts_input_file
    prog_mode = configs.prog_mode
    proj_freq = configs.proj_freq
    calc_all = configs.calc_all
    hybrid_mode = configs.hybrid_mode
    ts_output_file_computed = configs.ts_computed_file
    irc_output_file_computed = configs.irc_computed_file
    react_output_file_computed = configs.react_computed_file
    prod_output_file_computed = configs.prod_computed_file

    # --- Optimization of the transition state geometry ---
    if ts_output_file_computed is None:
        ts_output_file = ts_optimization(ts_guess_input, command_file, prog_mode, run_software_key)
    else:
        ts_output_file = file_check(ts_output_file_computed)

    # --- Computation of the IRC pathway in both directions (both electronic energies and ZPVEs are available for Gaussian, only electronic energies are available for ORCA) ---
    if irc_output_file_computed is None:
        irc_output_file = irc_computations(ts_guess_input, ts_output_file, command_file, prog_mode, run_software_key, proj_freq, calc_all)
    else:
        irc_output_file = file_check(irc_output_file_computed)

    # --- IRC output file parsing and analyzing the heat effect of the reaction ---
    irc_data = irc_file_parsing(irc_output_file, ts_output_file, prog_mode, proj_freq)

    # --- Optimization of the reactant and product geometries ---
    if react_output_file_computed is None or prod_output_file_computed is None:
        react_output_file, prod_output_file = react_prod_optimization(ts_guess_input, irc_output_file, command_file, prog_mode, run_software_key)
    else:
        react_output_file = file_check(react_output_file_computed)
        prod_output_file = file_check(prod_output_file_computed)

    # --- Output file parsing for the optimized transition state, reactant, and product ---
    energy_points = opt_file_parsing(ts_output_file, react_output_file, prod_output_file, prog_mode)

    # --- Determining the potential scaling factor (available for a hybrid_mode, default value is 1.0) ---
    if hybrid_mode:
        potential_scal_factor_react, potential_scal_factor_prod  = potential_scal_factor_comp(irc_data, energy_points)
    else:
        potential_scal_factor_react, potential_scal_factor_prod  = 1.0, 1.0

    # --- Computation of the attempt frequencies and the number of vibrational levels for the temperature averaging ---
    attempt_freq_and_levels_data = attempt_freq_and_levels(ts_output_file, react_output_file, prod_output_file, irc_data.electronic_energies, potential_scal_factor_react, potential_scal_factor_prod, corr_analysis, prog_mode)

    # --- Writing the input files for the TUNNEX 2.0 computations ---
    tunnex_forward, tunnex_backward = tunnex_input_writing(ts_guess_input, irc_data, energy_points, attempt_freq_and_levels_data, potential_scal_factor_react, potential_scal_factor_prod)
    
    return tunnex_forward, tunnex_backward