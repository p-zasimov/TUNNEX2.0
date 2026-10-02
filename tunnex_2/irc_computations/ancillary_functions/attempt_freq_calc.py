# --- Modules ---
import math
from pathlib import Path
from tunnex_2.irc_computations.gaussian import gauss_patterns # type: ignore
from tunnex_2.irc_computations.orca import orca_patterns # type: ignore
from tunnex_2.irc_computations.orca.file_parsing import orca_mass_extraction # type: ignore
from tunnex_2.irc_computations.constants_and_dataclasses.constants_and_settings import cm_m1_to_Hartree # type: ignore


# --- Collecting frequencies and normal mode vectors (Gaussian) ---
def gauss_mode_coordinate_reader(filename: str | Path) -> tuple[list[list[float]], tuple[float, ...]]:

    # --- Reading a file and searching for frequencies and masses ---
    with open(filename, encoding="utf-8") as f:
        text = f.read()

    block_match = gauss_patterns.pattern_mode_vectors.search(text)
    if block_match is None:
        raise ValueError(f"The normal coordinates were not found. Please check '{filename}'.")

    vib_masses = tuple(float(mass) for mass in gauss_patterns.pattern_vibr_mass.findall(text))
    if not vib_masses:
        raise ValueError(f"The atomic masses were not found. Please check '{filename}'.")

    freq_and_modes = []

    # --- Reading frequencies and normal mode vectors ---
    lines = iter(block_match.group(1).splitlines())

    for line in lines:

        if "Frequencies --" not in line:
            continue

        _, freq = line.split("--", maxsplit=1)
        modes = [[float(x)] for x in freq.split()]
        if len(modes) not in (1, 3):
            raise ValueError(f"One or three normal modes were expected in a frequency block. Please check '{filename}'.")

        # --- Skipping the lines to line 'Atom  AN...' and going to the next line ---
        coordinate_table = False

        for line in lines:
            if "Atom  AN" in line:
                coordinate_table = True
                break

        if not coordinate_table:
            raise ValueError(f"Normal mode vector table was not found. Please check '{filename}'.")

        coordinates_found = False

        # --- Saving normal mode vectors ---
        for line in lines:
            if len(modes) == 1:
                coordinate_match = gauss_patterns.pattern_mode_coordinates_one_mode.match(line)
            else:
                coordinate_match = gauss_patterns.pattern_mode_coordinates.match(line)

            if coordinate_match is None:
                break

            coordinates_found = True

            coordinates = [float(value) for value in coordinate_match.groups()[2:]]

            expected_coordinates = 3 * len(modes)

            if len(coordinates) != expected_coordinates:
                raise ValueError(f"Expected {expected_coordinates} normal-mode coordinates, got {len(coordinates)}. Please check '{filename}'.")

            for mode_index, frequency in enumerate(modes):
                start = 3 * mode_index
                frequency.extend(coordinates[start:start + 3])

        if not coordinates_found:
            raise ValueError(f"Normal mode coordinates were not found. Please check '{filename}'.")

        freq_and_modes.extend(modes)

    return freq_and_modes, vib_masses


# --- Collecting frequencies and normal mode vectors (ORCA) ---
def orca_mode_coordinate_reader(filename: str | Path) -> tuple[list[list[float]], tuple[float, ...]]:

    # --- Reading a file and searching for frequencies and normal modes ---
    with open(filename, encoding="utf-8") as f:
        text = f.read()

    vibr_blocks = orca_patterns.pattern_vibr_block.findall(text)
    if not vibr_blocks:
        raise ValueError(f"Frequencies were not found. Please check '{filename}'.")
    
    last_vibr_block = vibr_blocks[-1]
    freqs = [float(x) for x in orca_patterns.pattern_vibr_frequencies.findall(last_vibr_block)]
    if not freqs:
        raise ValueError(f"Vibrational frequencies were not found in the frequency block. Please check '{filename}'.")

    matches = orca_patterns.pattern_normal_modes.findall(text)
    if not matches:
        raise ValueError(f"Normal modes were not found. Please check '{filename}'.")
    
    normal_modes_text = matches[-1]

    # --- Filling the values of normal modes ---
    matrix_mode_coordinates = []
    columns = None

    for line in normal_modes_text.splitlines():
        fields = line.split()
        if not fields:
            continue

        # --- Reading the heades ---
        if all(field.isdigit() for field in fields):
            columns = [int(field) for field in fields]
            while len(matrix_mode_coordinates) <= max(columns):
                matrix_mode_coordinates.append([])
            continue

        # --- Reading the data ---
        if fields[0].isdigit() and columns is not None:
            values = [float(x) for x in fields[1:]]

            for column, value in zip(columns, values):
                matrix_mode_coordinates[column].append(value)

    vib_masses = orca_mass_extraction(filename)

    if len(freqs) != len(matrix_mode_coordinates):
        raise ValueError(f"Number of frequencies ({len(freqs)}) does not match number of normal modes ({len(matrix_mode_coordinates)}). Please check '{filename}'.")

    freq_and_modes = [[freq, *mode] for freq, mode in zip(freqs, matrix_mode_coordinates) if abs(freq) >= 1e-2]

    return freq_and_modes, vib_masses


# --- Scaling the coordinates of the vibrational mode vectors ---
def _mass_scaling(modes: list[list[float]], masses: tuple[float, ...]) -> tuple[float, list[tuple[float, float]]]:
    modes_scaled = []

    for row in modes:
        scaled_coordinates = [coordinate * math.sqrt(masses[i // 3]) for i, coordinate in enumerate(row[1:])]
        modes_scaled_row = [row[0]] + scaled_coordinates
        modes_scaled.append(modes_scaled_row)

    return modes_scaled


# --- Defining the function to determine the attemt frequency ---
def attempt_freq_calc(ts_file_out: Path, minimum_file_out: Path, species: str, prog_key: str = 'gaussian') -> float | list[list[float, float]]:

    readers = {"gaussian": gauss_mode_coordinate_reader, "orca": orca_mode_coordinate_reader}
    try:
        reader = readers[prog_key]
    except KeyError:
        raise ValueError(f"Unsupported program key ('{prog_key!r}'). Expected 'gaussian' or 'orca'.") from None
    reader = readers[prog_key]

    if species not in ('reactant', 'product'):
        raise ValueError(f"Unsupported species type ('{species!r}'). Expected 'reactant' or 'product'.")

    # --- Reading the negative transtion state frequency (reaction coordinate) and normalizing it ---
    ts_modes, ts_masses = reader(ts_file_out)
    ts_modes_scaled = _mass_scaling(ts_modes, ts_masses)
    ts_mode_scaled = ts_modes_scaled[0]
    ts_mode_norm = math.sqrt(sum(x ** 2 for x in ts_mode_scaled[1:]))
    if ts_mode_norm <= 1e-12:
        raise ValueError(f"The frequency normalization factor is smaller than 1e-12. Please check '{ts_file_out}'.")

    correlations = []

    # --- Reading the frequencies and normal mode shift vectors at the minimum ---
    minimum_modes, minimum_masses = reader(minimum_file_out)
    minimum_modes_scaled = _mass_scaling(minimum_modes, minimum_masses)

    for minimum_mode_scaled in minimum_modes_scaled:

        freq = minimum_mode_scaled[0]

        # --- Normalizing the mode shift-vectors at the minimum ---
        minimum_mode_norm = math.sqrt(sum(x ** 2 for x in minimum_mode_scaled[1:]))
        if minimum_mode_norm <= 1e-12:
                raise ValueError(f"The frequency normalization factor is smaller than 1e-12. Please check '{minimum_file_out}'.")
        
        # --- Computing the scalar products of minimum normal mode shift-vectors and one mode of the transition state ---
        if len(ts_mode_scaled[1:]) != len(minimum_mode_scaled[1:]):
                raise ValueError(f"Transition state and minimum normal modes have different numbers of coordinates. Please check '{ts_file_out}' and '{minimum_file_out}'.")
        
        correlation = abs(sum(x * y for x, y in zip(ts_mode_scaled[1:], minimum_mode_scaled[1:]))) / (ts_mode_norm * minimum_mode_norm)
        correlations.append((freq, correlation))

    if not correlations:
        raise ValueError(f"No valid normal modes were found. Please check {minimum_file_out}.")

    # --- Picking the frequency ---
    attempt_freq, max_correlation = max(correlations, key=lambda x: x[1])

    print(f'For {species} the attempt frequency is {attempt_freq:.2f} cm-1 which correlates with the IRC coordinate as {max_correlation:.2f}. Please check this frequency manually.\n')

    return attempt_freq, correlations


# --- Defining the function to write the correlation of reactant (product) frequencies with the IRC ---
def write_freq_corr(filename: str | Path, attempt_freq_species_corr: list[tuple[float, float]]) -> None:

    att_freq_aver = sum(freq * corr for freq, corr in attempt_freq_species_corr) / sum(corr for _, corr in attempt_freq_species_corr)

    with open(filename, "w", encoding="utf-8") as f_out:
        f_out.write(f"{'Frequency, cm-1':>15} | {'Correlation with the IRC':>25}\n")
        f_out.write("-" * 15 + "-+-" + "-" * 25 + "\n")
        for frequency, correlation in attempt_freq_species_corr:
            f_out.write(f"{frequency:15.2f} | {correlation:25.5f}\n")
        f_out.write(f"\nThe correlation-weighted average attempt frequency is {att_freq_aver:.2f} cm-1\n")

    return None


# --- Computing the number of vibrational levels for temperature averaging ---
def compute_number_of_vib_levels(frequency : float, data_el_energy : list[tuple[float, float]], potential_scaling_factor : float = 1.0, direction: str = 'reactant') -> int:

    if direction not in ("reactant", "product"):
        raise ValueError(f"Expected 'reactant' or 'product' in direction. Got '{direction}'.")

    if frequency <= 0:
        raise ValueError(f"Frequency must be positive. Got {frequency} cm-1.")
    
    ts_energy = max(data_el_energy, key=lambda x: x[1])

    if direction == 'reactant':
        point_energy = min(data_el_energy, key=lambda x: x[0])
    else:
        point_energy = max(data_el_energy, key=lambda x: x[0])

    # --- Because the number of gaps is N, the number of level should be N + 1. But the first level is started from 0.5 * freq, so, N + 1 - 0.5 = N + 0.5 ---
    number_of_levels = math.ceil(potential_scaling_factor * (ts_energy[1] - point_energy[1]) / (frequency * cm_m1_to_Hartree) + 0.5)

    return max(0, number_of_levels - 1) # a) This value cannot be negative. b) Since we adding 1 after, here we subtract 1.


# --- Defining the function to check electronic eenrgies or ZPVE data for sudden changes in slope ---
def curve_smoothness_check(energy_data: list[tuple[float, float]], mode : str, smoothness_ratio_tol : float, irc_spacing_tol : float = 1e-10, derivative_zero_tol : float = 1e-10) -> None:

    if len(energy_data) < 3:
        return None
    
    for i in range(len(energy_data) - 2):

        irc_0, energy_0 = energy_data[i]
        irc_1, energy_1 = energy_data[i + 1]
        irc_2, energy_2 = energy_data[i + 2]

        delta_irc_1 = irc_1 - irc_0
        delta_irc_2 = irc_2 - irc_1

        if math.isclose(delta_irc_1, 0.0, abs_tol=irc_spacing_tol):
            raise ValueError(f"Duplicate IRC values: {irc_0} and {irc_1}. Please check your data.")

        if math.isclose(delta_irc_2, 0.0, abs_tol=irc_spacing_tol):
            raise ValueError(f"Duplicate IRC values: {irc_1} and {irc_2}. Please check your data.")

        derivative = (energy_1 - energy_0) / delta_irc_1
        derivative_is_zero =  math.isclose(derivative, 0.0, abs_tol=derivative_zero_tol)

        derivative_next = (energy_2 - energy_1) / delta_irc_2
        derivative_next_is_zero =  math.isclose(derivative_next, 0.0, abs_tol=derivative_zero_tol)

        if derivative_is_zero and derivative_next_is_zero:
            continue
        if derivative_is_zero or derivative_next_is_zero:
            print(f"A potential {mode} slope anomaly at IRC={irc_1}. Derivatives: {derivative:.3g} and {derivative_next:.3g}. Please check your data.\n")
            continue

        derivatives_ratio = abs(derivative_next / derivative)
        
        if derivatives_ratio > smoothness_ratio_tol:
            print(f"A potential {mode} outbreak was found at IRC={irc_1}. Absolute slope ratio={derivatives_ratio:.3g}, tolerance={smoothness_ratio_tol}. Please check your data.\n")

        if 1 / derivatives_ratio > smoothness_ratio_tol:
            print(f"A potential {mode} outbreak was found at IRC={irc_1}. Absolute inverse slope ratio={1 / derivatives_ratio:.3g}, tolerance={smoothness_ratio_tol}. Please check your data.\n")

    return None