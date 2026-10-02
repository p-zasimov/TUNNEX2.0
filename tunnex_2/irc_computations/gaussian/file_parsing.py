# --- Modules ---
from pathlib import Path
from tunnex_2.irc_computations.gaussian import gauss_patterns # type: ignore
from tunnex_2.irc_computations.constants_and_dataclasses.constants_and_settings import cm_m1_to_Hartree # type: ignore
from tunnex_2.irc_computations.constants_and_dataclasses.dataclasses import IRCData # type: ignore


# --- Defining the auxiliary function to extract IRC coordinates and ZPVEs from the IRC output file
# (ZPVEs are obtained via the summation of positive frequencies) ---
def _gauss_freq_extraction(irc_direction: str, filename: Path, sign: int) -> list[tuple[float, float]]:

    zpve_energies_direction = []
    positive_frequency_sum = 0.0

    for line in irc_direction.splitlines():
        if "Frequencies --" in line:
            positive_frequency_sum += sum(freq for freq in map(float, gauss_patterns.pattern_freq.findall(line)) if freq > 0.0)
            continue

        elif "NET REACTION COORDINATE UP TO THIS POINT" in line:
            match = gauss_patterns.pattern_irc_coordinate.search(line).group(1)
            if match is None:
                raise ValueError(f"Could not extract IRC coordinate from line: {line!r}. Please check '{filename}'.")
            irc_coord = sign * float(match)

            zpve_point = 0.5 * positive_frequency_sum * cm_m1_to_Hartree
            zpve_energies_direction.append((irc_coord, zpve_point))
            positive_frequency_sum = 0.0

    return zpve_energies_direction


# --- Defining the function to read the IRC coordinates, electronic energies, and ZPVEs from the IRC output file ---
def reading_gauss_irc(irc_file_out: Path, proj_freq: bool=True) -> IRCData:

    # --- Opening the file and reading the data ---
    with open(irc_file_out, encoding="utf-8") as f_irc:
        text = f_irc.read()       
        marker = gauss_patterns.irc_split_marker_forward
        sections = text.split(marker, maxsplit=1)

        if len(sections) != 2:
            raise ValueError(f"Cannot find '{marker}' in '{irc_file_out}'. Please check '{irc_file_out}'.")
        
        forward, reverse = sections

    # --- Extracting the transition state energy ---
    match = gauss_patterns.pattern_ts_energy.search(text)

    if match:
        ts_el_energy = float(match.group(1))
        text_irc_energies = text[match.end():]
    else:
        raise ValueError(f"The transition state energy was not found. Please check '{irc_file_out}'.")

    # --- Extracting the IRC values and electronic energies ---
    data_el_energy = [(float(irc), ts_el_energy + float(relative_el_energy)) for relative_el_energy, irc in gauss_patterns.pattern_el_energy.findall(text_irc_energies)]

    # --- Extracting ZPVEs ---
    if proj_freq:
        zpve_energies_forward = _gauss_freq_extraction(forward, irc_file_out, sign=1)
        zpve_energies_reverse = _gauss_freq_extraction(reverse, irc_file_out, sign=-1)
    else:
        zpve_energies_forward = None
        zpve_energies_reverse = None

    return IRCData(electronic_energies=data_el_energy, zpve_energies_forward=zpve_energies_forward, zpve_energies_reverse=zpve_energies_reverse)


# --- Defining the function to store the frequencies of the transition state and minima (reactant, product) ---
def reading_gauss_struct_file(opt_file_out: Path, mode: str='ts') -> tuple[tuple[float, float], tuple[float, float]]:

    # --- Checking if we correctly defined the mode ---
    mode_positions = {"ts": 0.0, "react": float("-inf"), "prod": float("inf")}
    if mode not in mode_positions:
        raise ValueError(f"The inserted mode ('{mode}') is incorrect. Please use 'react', 'prod', or 'ts'.")
    position = mode_positions[mode]

    # --- Opening the file and reading the data ---
    with open(opt_file_out, encoding="utf-8") as f_in:

        el_energy, zpve_energy = None, None
        positive_frequency_sum = 0.0

        # --- Searching for the electronic energy and ZPVE ---
        for line in f_in:
            match_el_energy = gauss_patterns.pattern_el_energy_single_point.search(line)

            if match_el_energy:
                el_energy = (position, float(match_el_energy.group(1)))

            elif "Frequencies --" in line:
                nums = gauss_patterns.pattern_freq.findall(line)
                positive_frequency_sum += sum(freq for freq in map(float, nums) if freq > 0.0)

            elif "- Thermochemistry -" in line:
                if positive_frequency_sum == 0.0:
                    print(f"No frequencies was found for a reactant (product) or a transition state.",
                          f"Please note that ZPVE values may be incorrect.")
                zpve_energy = (position, 0.5 * positive_frequency_sum * cm_m1_to_Hartree)
                positive_frequency_sum = 0.0

        if el_energy is None:
            raise ValueError(f"The electronic energy of a reactant (product) or a transition state was not found. Please check '{opt_file_out}'.")

        if zpve_energy is None:
            raise ValueError(f"The ZPVE of a reactant (product) or a transition state was not found. Please check '{opt_file_out}'.")
    
    return el_energy, zpve_energy