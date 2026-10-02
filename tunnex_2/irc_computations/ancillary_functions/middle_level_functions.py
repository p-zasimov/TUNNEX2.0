# --- Modules ---
import math
from pathlib import Path
from tunnex_2.irc_computations.constants_and_dataclasses.constants_and_settings import (# type: ignore
    kJ_mol_m1_to_Hartree,
    curve_smoothness_ratio_tol)
from tunnex_2.irc_computations.constants_and_dataclasses.dataclasses import ( # type: ignore
    IRCData,
    EnergyPointsData,
    AttemptFreqLevelNum,
    TunnexInputSettings,
 )
from tunnex_2.irc_computations.ancillary_functions.attempt_freq_calc import ( # type: ignore
    attempt_freq_calc,
    write_freq_corr,
    compute_number_of_vib_levels,
    gauss_mode_coordinate_reader,
    orca_mode_coordinate_reader,
    curve_smoothness_check,
)
from tunnex_2.irc_computations.ancillary_functions.eckart_potential_functions import ( # type: ignore
    eckart_potential_parameters,
    eckart_potential_data,
)
from tunnex_2.irc_computations.ancillary_functions.interface import( # type: ignore
    run_software,
    gauss_error_check,
    gauss_out_filename,
    orca_error_check,
    orca_out_filename,
    file_check,
)
from tunnex_2.irc_computations.ancillary_functions.tunnex_input import ( # type: ignore
    write_input_head,
    writing_el_energy,
    writing_zpve_energy,
)
from tunnex_2.irc_computations.gaussian.file_parsing import ( # type: ignore
    reading_gauss_irc,
    reading_gauss_struct_file,
)
from tunnex_2.irc_computations.gaussian.file_writing import ( # type: ignore
    create_gauss_input_file,
    create_gauss_irc_input,
    create_gauss_react_prod_input,
)
from tunnex_2.irc_computations.orca.file_parsing import ( # type: ignore
    reading_orca_irc,
    reading_orca_struct_file,
)
from tunnex_2.irc_computations.orca.file_writing import ( # type: ignore
    create_orca_input_file,
    create_orca_irc_input,
    create_orca_react_prod_input,
)
from tunnex_2.irc_computations.gaussian import gauss_patterns # type: ignore
from tunnex_2.irc_computations.orca import orca_patterns # type: ignore


# --- Defining the function for the optimization of the transition state geometry ---
def ts_optimization(ts_guess_input : str | Path, command_file : str | Path, mode : str, run_software_key: bool = True) -> Path:

    ts_guess_input_path = file_check(ts_guess_input)
    ts_input_file = ts_guess_input_path.with_stem(ts_guess_input_path.stem + "_ts")

    if mode == 'gaussian':
        create_gauss_input_file(ts_guess_input_path, ts_input_file, None, gauss_patterns.ts_optimization_line)
    elif mode == 'orca':
        create_orca_input_file(ts_guess_input_path, ts_input_file, None, orca_patterns.ts_optimization_block)
    else:
        raise ValueError(f"Expected 'gaussian' or 'orca' as program mode. Got '{mode}'.")

    if run_software_key:
        run_software(ts_input_file, command_file)

    if mode == 'gaussian':
        ts_output_file = gauss_out_filename(ts_input_file)
        gauss_error_check(ts_output_file)
    elif mode == 'orca':
        ts_output_file = orca_out_filename(ts_input_file, ".out")
        orca_error_check(ts_output_file)
    else:
        raise ValueError(f"Expected 'gaussian' or 'orca' as program mode. Got '{mode}'.")

    return ts_output_file


# --- Defining the function for the IRC computations ---
def irc_computations(ts_guess_input : str | Path, ts_output_file : Path, command_file : str | Path, mode : str, run_software_key: bool = True, proj_freq: bool=True, calc_all: bool=True) -> Path:

    ts_guess_input_path = file_check(ts_guess_input)

    if mode == 'gaussian':
        irc_input_file = create_gauss_irc_input(ts_guess_input_path, ts_output_file, proj_freq, calc_all)
    elif mode == 'orca':
        irc_input_file = create_orca_irc_input(ts_guess_input_path, ts_output_file)
    else:
        raise ValueError(f"Expected 'gaussian' or 'orca' as program mode. Got '{mode}'.")

    if run_software_key:
        run_software(irc_input_file, command_file)

    if mode == 'gaussian':
        irc_output_file = gauss_out_filename(irc_input_file)
        gauss_error_check(irc_output_file)
    elif mode == 'orca':
        irc_output_file = orca_out_filename(irc_input_file, ".xyz")
        orca_error_check(irc_output_file)
    else:
        raise ValueError(f"Expected 'gaussian' or 'orca' as program mode. Got '{mode}'.")

    return irc_output_file


# --- Defining the function for IRC output file parsing and analyzing the heat effect of the reaction ---
def irc_file_parsing(irc_output_file : Path, ts_output_file : Path, mode : str, proj_freq : bool = True) -> IRCData:

    if mode == 'gaussian':
        irc_data = reading_gauss_irc(irc_output_file, proj_freq)
    elif mode == 'orca':
        irc_data = reading_orca_irc(irc_output_file, ts_output_file)
    else:
        raise ValueError(f"Expected 'gaussian' or 'orca' as program mode. Got '{mode}'.")

    if irc_data.electronic_energies is not None:
        curve_smoothness_check(irc_data.electronic_energies, 'electronic energy', curve_smoothness_ratio_tol)
    if irc_data.zpve_energies_forward is not None:
        curve_smoothness_check(irc_data.zpve_energies_forward, 'ZPVE', curve_smoothness_ratio_tol)
    if irc_data.zpve_energies_reverse is not None:
        curve_smoothness_check(irc_data.zpve_energies_reverse, 'ZPVE', curve_smoothness_ratio_tol)

    react_energy = min(irc_data.electronic_energies, key=lambda x: x[0])[1]
    prod_energy = max(irc_data.electronic_energies, key=lambda x: x[0])[1]
    electronic_energy_difference = - (prod_energy - react_energy) / kJ_mol_m1_to_Hartree
    # The difference is multiplied by -1 because the program deals with the computed electronic energies which are negative
    
    if electronic_energy_difference > 0:
        print(f"Please note that the forward reaction is exothermic by {abs(electronic_energy_difference):.1f} kJ mol-1.\n")
    elif electronic_energy_difference < 0:
        print(f"Please note that the forward reaction is endothermic by {abs(electronic_energy_difference):.1f} kJ mol-1.\n")
    else:
        print(f"The energy of reactant is equal to one of the product.\n")

    return irc_data


# --- Defining the function for the optimization of the reactant and product geometries ---
def react_prod_optimization(ts_guess_input : str | Path, irc_output_file : Path, command_file : str | Path, mode : str, run_software_key: bool = True) -> tuple[Path, Path]:

    ts_guess_input_path = file_check(ts_guess_input)

    if mode == 'gaussian':
        react_input_file, prod_input_file = create_gauss_react_prod_input(irc_output_file, ts_guess_input_path)
    elif mode == 'orca':
        react_input_file, prod_input_file = create_orca_react_prod_input(ts_guess_input_path)
    else:
        raise ValueError(f"Expected 'gaussian' or 'orca' as program mode. Got '{mode}'.")

    if run_software_key:
        run_software(react_input_file, command_file)
        run_software(prod_input_file, command_file)

    if mode == 'gaussian':
        react_output_file = gauss_out_filename(react_input_file)
        prod_output_file = gauss_out_filename(prod_input_file)
        gauss_error_check(react_output_file)
        gauss_error_check(prod_output_file)
    elif mode == 'orca':
        react_output_file = orca_out_filename(react_input_file, ".out")
        prod_output_file = orca_out_filename(prod_input_file, ".out")
        orca_error_check(react_output_file)
        orca_error_check(prod_output_file)
    else:
        raise ValueError(f"Expected 'gaussian' or 'orca' as program mode. Got '{mode}'.")

    return react_output_file, prod_output_file


# --- Defining the function for output file parsing for the optimized transition state, reactant, and product energies ---
def opt_file_parsing(ts_output_file : Path, react_output_file : Path, prod_output_file : Path, mode : str) -> EnergyPointsData:

    ts_output_file_path = file_check(ts_output_file)
    react_output_file_path = file_check(react_output_file)
    prod_output_file_path = file_check(prod_output_file)

    if mode == 'gaussian':
        ts_el_energy, ts_zpve_energy = reading_gauss_struct_file(ts_output_file_path, mode='ts')
        react_el_energy, react_zpve_energy = reading_gauss_struct_file(react_output_file_path, mode='react')
        prod_el_energy, prod_zpve_energy = reading_gauss_struct_file(prod_output_file_path, mode='prod')
    elif mode == 'orca':
        ts_el_energy, ts_zpve_energy = reading_orca_struct_file(ts_output_file_path, mode='ts')
        react_el_energy, react_zpve_energy = reading_orca_struct_file(react_output_file_path, mode='react')
        prod_el_energy, prod_zpve_energy = reading_orca_struct_file(prod_output_file_path, mode='prod')
    else:
        raise ValueError(f"Expected 'gaussian' or 'orca' as program mode. Got '{mode}'.")

    energy_points = EnergyPointsData(
    transition_state_el_energy=ts_el_energy,
    transition_state_zpve=ts_zpve_energy,
    reactant_el_energy=react_el_energy,
    reactant_zpve=react_zpve_energy,
    product_el_energy=prod_el_energy,
    product_zpve=prod_zpve_energy,
    )
    
    return energy_points


# --- Defining the function for the computation of the attempt frequencies (# attempt_freq_xxx_corr is extra information
# which could be useful during the attempt frequency analysis) ---
def attempt_freq_and_levels(ts_output_file : Path, react_output_file : Path, prod_output_file : Path, data_el_energy : list[list[float]], potential_scaling_factor_react : float, potential_scaling_factor_prod : float, corr_analysis : bool, mode : str) -> AttemptFreqLevelNum:

    ts_output_file_path = file_check(ts_output_file)
    react_output_file_path = file_check(react_output_file)
    prod_output_file_path = file_check(prod_output_file)

    attempt_freq_react, attempt_freq_react_corr = attempt_freq_calc(ts_output_file_path, react_output_file_path, 'reactant', mode)
    attempt_freq_prod, attempt_freq_product_corr = attempt_freq_calc(ts_output_file_path, prod_output_file_path, 'product', mode)

    if corr_analysis:
        react_corr_file = react_output_file.with_stem(react_output_file.stem + "_corr").with_suffix(".txt")
        prod_corr_file = prod_output_file.with_stem(prod_output_file.stem + "_corr").with_suffix(".txt")
        write_freq_corr(react_corr_file, attempt_freq_react_corr)
        write_freq_corr(prod_corr_file, attempt_freq_product_corr)

    number_of_levels_react = compute_number_of_vib_levels(attempt_freq_react, data_el_energy, potential_scaling_factor_react, 'reactant')
    number_of_levels_product = compute_number_of_vib_levels(attempt_freq_prod, data_el_energy, potential_scaling_factor_prod, 'product')

    attempt_freq_and_levels_data = AttemptFreqLevelNum(
    att_freq_react = attempt_freq_react,
    att_freq_react_corr = attempt_freq_react_corr,
    att_freq_prod = attempt_freq_prod,
    att_freq_prod_corr = attempt_freq_product_corr,
    num_levels_react = number_of_levels_react,
    num_levels_prod = number_of_levels_product)

    return attempt_freq_and_levels_data


# --- Defining the function for the potential scaling factor computations ---
def potential_scal_factor_comp(irc_data : IRCData, energy_points: EnergyPointsData) -> tuple[float, float]:
    irc_el_energy_react = min(irc_data.electronic_energies, key=lambda irc: irc[0])[1]
    irc_el_energy_ts = min(irc_data.electronic_energies, key=lambda irc: abs(irc[0]))[1]
    irc_el_energy_prod = max(irc_data.electronic_energies, key=lambda irc: irc[0])[1]

    if irc_data.zpve_energies_reverse is not None:
        irc_zpve_energy_react = min(irc_data.zpve_energies_reverse, key=lambda x: x[0])[1]
    else:
        irc_zpve_energy_react = 0.0

    irc_zpve_energy_ts = energy_points.transition_state_zpve[1] # Neither Gaussian nor ORCA computes this value during the IRC scan

    if irc_data.zpve_energies_forward is not None:
        irc_zpve_energy_prod = max(irc_data.zpve_energies_forward, key=lambda x: x[0])[1]
    else:
        irc_zpve_energy_prod = 0.0

    energy_points_react = (energy_points.transition_state_el_energy[1] + energy_points.transition_state_zpve[1] - energy_points.reactant_el_energy[1] - energy_points.reactant_zpve[1])
    irc_react = (irc_el_energy_ts + irc_zpve_energy_ts - irc_el_energy_react - irc_zpve_energy_react)
    if not math.isclose(irc_react, 0.0, abs_tol=1e-10):
        potential_scal_factor_react = abs(energy_points_react / irc_react) # Potential_scal_factor cannot be negative, thus, using abs()
    else:
        potential_scal_factor_react = 1.0 # Not changing the result if the barrier is very flat
        print(f"Very small barrier ({irc_react:.3g} Hartree) for a forward reaction. Setting potential_scal_factor_react as {potential_scal_factor_react}.")

    energy_points_prod = (energy_points.transition_state_el_energy[1] + energy_points.transition_state_zpve[1] - energy_points.product_el_energy[1] - energy_points.product_zpve[1])
    irc_prod = (irc_el_energy_ts + irc_zpve_energy_ts - irc_el_energy_prod - irc_zpve_energy_prod)
    if not math.isclose(irc_prod, 0.0, abs_tol=1e-10):
        potential_scal_factor_prod = abs(energy_points_prod / irc_prod) # Potential_scal_factor cannot be negative, thus, using abs()
    else:
        potential_scal_factor_prod = 1.0 # Not changing the result if the barrier is very flat
        print(f"Very small barrier ({irc_prod:.3g} Hartree) for a backward reaction. Setting potential_scal_factor_prod as {potential_scal_factor_prod}.")

    return potential_scal_factor_react, potential_scal_factor_prod


# --- Defining the function for writing the input files for the TUNNEX 2.0 computations ---
def tunnex_input_writing(ts_guess_input: str | Path, irc_data, energy_points : EnergyPointsData, attempt_freq_and_levels_data: AttemptFreqLevelNum, potential_scal_factor_react : float = 1.0, potential_scal_factor_prod : float = 1.0) -> tuple[Path, Path]:

    ts_guess_input_path = file_check(ts_guess_input)

    tunnex_settings_react = TunnexInputSettings(
    freq=attempt_freq_and_levels_data.att_freq_react,
    E0=energy_points.reactant_el_energy[1],
    ZPVE0=energy_points.reactant_zpve[1],
    number_of_levels=attempt_freq_and_levels_data.num_levels_react,
    potential_scaling_factor = potential_scal_factor_react,
    )

    tunnex_settings_prod = TunnexInputSettings(
    freq=attempt_freq_and_levels_data.att_freq_prod,
    E0=energy_points.product_el_energy[1],
    ZPVE0=energy_points.product_zpve[1],
    number_of_levels=attempt_freq_and_levels_data.num_levels_prod,
    potential_scaling_factor = potential_scal_factor_prod,
    )

    tunnex_input_forward_file = write_input_head('_forward', ts_guess_input_path, tunnex_settings_react)
    tunnex_input_backward_file = write_input_head('_backward', ts_guess_input_path, tunnex_settings_prod)

    writing_el_energy(tunnex_input_forward_file, irc_data.electronic_energies, reverse_key=False)
    writing_el_energy(tunnex_input_backward_file, irc_data.electronic_energies, reverse_key=True)

    writing_zpve_energy(tunnex_input_forward_file, irc_data.electronic_energies, irc_data.zpve_energies_forward, irc_data.zpve_energies_reverse, energy_points.transition_state_zpve, reverse_key=False)
    writing_zpve_energy(tunnex_input_backward_file, irc_data.electronic_energies, irc_data.zpve_energies_forward, irc_data.zpve_energies_reverse, energy_points.transition_state_zpve, reverse_key=True)

    return tunnex_input_forward_file, tunnex_input_backward_file


# --- Defining the function for the optimization of the transition state geometry ---
def eckart_react_prod_optimization(react_guess_input : str | Path, prod_guess_input : str | Path, command_file : str | Path, mode : str, run_software_key: bool = True) -> tuple[Path, Path]:

    react_guess_input_path = file_check(react_guess_input)
    react_input_file = react_guess_input_path.with_stem(react_guess_input_path.stem + "_react")

    prod_guess_input_path = file_check(prod_guess_input)
    prod_input_file = prod_guess_input_path.with_stem(prod_guess_input_path.stem + "_prod")

    if mode == 'gaussian':
        create_gauss_input_file(react_guess_input_path, react_input_file, None, gauss_patterns.method_tail_react_prod)
        create_gauss_input_file(prod_guess_input_path, prod_input_file, None, gauss_patterns.method_tail_react_prod)
    elif mode == 'orca':
        create_orca_input_file(react_guess_input_path, react_input_file, None, orca_patterns.min_optimization_line)
        create_orca_input_file(prod_guess_input_path, prod_input_file, None, orca_patterns.min_optimization_line)
    else:
        raise ValueError(f"Expected 'gaussian' or 'orca' as program mode. Got '{mode}'.")

    if run_software_key:
        run_software(react_input_file, command_file)
        run_software(prod_input_file, command_file)

    if mode == 'gaussian':
        react_output_file = gauss_out_filename(react_input_file)
        gauss_error_check(react_output_file)
        prod_output_file = gauss_out_filename(prod_input_file)
        gauss_error_check(prod_output_file)
    elif mode == 'orca':
        react_output_file = orca_out_filename(react_input_file, ".out")
        orca_error_check(react_output_file)
        prod_output_file = orca_out_filename(prod_input_file, ".out")
        orca_error_check(prod_output_file)
    else:
        raise ValueError(f"Expected 'gaussian' or 'orca' as program mode. Got '{mode}'.")

    return react_output_file, prod_output_file


# --- Defining the function to store the datapoints of the Eckart potential function ---
def eckart_potential_results(ts_output_file: str | Path, energy_points : EnergyPointsData, mode : str) -> IRCData:

    # --- Parsing the transition state frequency (imaginary one) ---
    ts_output_file_path = Path(ts_output_file)

    if mode == 'gaussian':
        freq_and_modes, _ = gauss_mode_coordinate_reader(ts_output_file_path)
    elif mode == 'orca':
        freq_and_modes, _ = orca_mode_coordinate_reader(ts_output_file_path)
    else:
        raise ValueError(f"Expected 'gaussian' or 'orca' as program mode. Got '{mode}'.")

    ts_freq_cm_m1 = min(freq_and_modes, key=lambda freq: freq[0])[0]

    # --- Computing the barrier heights for Eckart potential ---
    ts_el_energy = energy_points.transition_state_el_energy[1]
    ts_zpve = energy_points.transition_state_zpve[1]

    react_el_energy = energy_points.reactant_el_energy[1]
    react_zpve = energy_points.reactant_zpve[1]

    prod_el_energy = energy_points.product_el_energy[1]
    prod_zpve = energy_points.product_zpve[1]

    dV1_Hartree = (ts_el_energy + ts_zpve) - (react_el_energy + react_zpve)
    dV2_Hartree = (ts_el_energy + ts_zpve) - (prod_el_energy + prod_zpve)

    # --- Computing the intermediate parameters for Eckart potential ---
    A_Hartree, B_Hartree, L_sqrt_amu_bohr = eckart_potential_parameters(ts_freq_cm_m1, dV1_Hartree, dV2_Hartree)

    # --- Computing Eckart potential ---
    irc_data = eckart_potential_data(A_Hartree, B_Hartree, L_sqrt_amu_bohr)

    if irc_data.electronic_energies is not None:
        curve_smoothness_check(irc_data.electronic_energies, 'electronic energy', curve_smoothness_ratio_tol)
    if irc_data.zpve_energies_forward is not None:
        curve_smoothness_check(irc_data.zpve_energies_forward, 'ZPVE', curve_smoothness_ratio_tol)
    if irc_data.zpve_energies_reverse is not None:
        curve_smoothness_check(irc_data.zpve_energies_reverse, 'ZPVE', curve_smoothness_ratio_tol)


    react_energy = min(irc_data.electronic_energies, key=lambda x: x[0])[1]
    prod_energy = max(irc_data.electronic_energies, key=lambda x: x[0])[1]
    electronic_energy_difference = - (prod_energy - react_energy) / kJ_mol_m1_to_Hartree
    # The difference is multiplied by -1 because the program deals with the computed electronic energies which are negative
    
    if electronic_energy_difference > 0:
        print(f"Please note that the forward reaction is exothermic by {abs(electronic_energy_difference):.1f} kJ mol-1.\n")
    elif electronic_energy_difference < 0:
        print(f"Please note that the forward reaction is endothermic by {abs(electronic_energy_difference):.1f} kJ mol-1.\n")
    else:
        print(f"The energy of reactant is equal to one of the product.\n")

    return irc_data