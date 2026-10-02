# --- Modules ---
from pathlib import Path
import numpy as np
from tunnex_2.irc_computations.orca import orca_patterns # type: ignore
from tunnex_2.irc_computations.constants_and_dataclasses.constants_and_settings import angstrom_to_bohr_radius # type: ignore
from tunnex_2.irc_computations.constants_and_dataclasses.dataclasses import IRCData # type: ignore


# --- Defining the auxiliary function to extract IRC coordinates and ZPVEs from the IRC output file
# (No method of computing projected frequencies is available in ORCA by now [01.09.2026]) ---
def _orca_freq_extraction(irc_direction: str, sign: int) -> None:
    return None


# --- Defining the function to read the IRC geometries and electronic energies from the IRC file ---
def _orca_struct_extraction(irc_file_xyz: Path) -> tuple[np.ndarray, np.ndarray]:

    with open(irc_file_xyz, encoding="utf-8") as f_irc:
        text = f_irc.read()

    geometry_blocks = orca_patterns.pattern_irc_coordinates_block.findall(text)
    if not geometry_blocks:
        raise ValueError(f"No geometries were found in '{irc_file_xyz}'. Please check '{irc_file_xyz}'.")

    energies = []
    geometries_results = []

    for energy, geometry in geometry_blocks:
        energies.append(float(energy))

        coordinates = (orca_patterns.pattern_irc_geometry.findall(geometry))
        if not coordinates:
            raise ValueError(f"No atomic coordinates found for IRC point with energy {energy}. Please check '{irc_file_xyz}'.")

        geometry_coordinates = [[float(x) for x in row[1:]] for row in coordinates]
        if any(len(atom) != 3 for atom in geometry_coordinates):
            raise ValueError(f"Invalid Cartesian coordinates for IRC point with energy {energy}. Please check '{irc_file_xyz}'.")

        geometries_results.append(geometry_coordinates)

    energies = np.array(energies, dtype=float)
    geometries_results = np.array(geometries_results, dtype=float)

    if geometries_results.ndim != 3 or geometries_results.shape[-1] != 3:
            raise ValueError(f"Unexpected geometry array shape: {geometries_results.shape}. Please check '{irc_file_xyz}'.")

    return energies, geometries_results


# --- Defining the function to extract the atomic mass ---
def orca_mass_extraction(filename: Path) -> np.ndarray:

    with open(filename, encoding="utf-8") as f_ts:
        text = f_ts.read()

    match = orca_patterns.pattern_mass_block.search(text)

    if match is None:
        raise ValueError(f"Cannot find the atomic mass block in '{filename}'. Please check '{filename}'.")

    mass_block = match.group(1)

    atom_masses = np.array([float(m) for m in orca_patterns.pattern_atom_mass.findall(mass_block)], dtype=float)

    if atom_masses.size == 0:
        raise ValueError(f"Atomic mass block was found, but no atomic masses could be extracted from '{filename}'.",
                         f"Please check '{filename}'.")

    return atom_masses


# --- Defining the function to correct the geometries according to their center of masses ---
def _center_of_mass_correction(geometry: np.ndarray, atom_masses: np.ndarray, filename: Path) -> np.ndarray:

    if geometry.ndim != 2 or geometry.shape[1] != 3:
        raise ValueError(f"Geometry must have shape (n_atoms, 3), got {geometry.shape}. Please check '{filename}'.")

    if atom_masses.ndim != 1:
        raise ValueError(f"Atom masses must have shape (n_atoms,), got {atom_masses.shape}. Please check '{filename}'.")

    if geometry.shape[0] != atom_masses.size:
        raise ValueError(f"Number of atoms in geometry ({geometry.shape[0]}) does not match number of masses ({atom_masses.size}).",
                         f"Please check '{filename}'.")

    center_of_mass = (np.sum(geometry * atom_masses[:, None], axis=0) / np.sum(atom_masses))

    return geometry - center_of_mass


# --- Defining the function to account for the rotation and compute the distance between the structures (as amu ** 0.5 * bohr) ---
def _struct_distance(geometry_1: np.ndarray, geometry_2: np.ndarray, atom_masses: np.ndarray, filename: Path) -> float:

    if geometry_1.shape != geometry_2.shape:
        raise ValueError(f"Geometry shapes do not match: {geometry_1.shape} vs {geometry_2.shape}. Please check '{filename}'.")

    if geometry_1.ndim != 2 or geometry_1.shape[1] != 3:
        raise ValueError(f"Geometries must have shape (n_atoms, 3), got {geometry_1.shape}. Please check '{filename}'.")

    if atom_masses.ndim != 1:
        raise ValueError(f"Atom masses must have shape (n_atoms,), got {atom_masses.shape}. Please check '{filename}'.")

    if geometry_1.shape[0] != atom_masses.size:
        raise ValueError(f"Number of atoms ({geometry_1.shape[0]}) does not match number of masses ({atom_masses.size}).", 
                         f"Please check '{filename}'.")

    # --- Computing the mass-weighted covariance matrix ---
    H = geometry_2.T @ (atom_masses[:, None] * geometry_1)

    # --- SVD decomposition ---
    U, _, Vt = np.linalg.svd(H)

    # --- Elliminating the reflection (if there is one, d will be -1,
    # implying an application of n additional reflection to cancel it) ---
    det = np.linalg.det(Vt.T @ U.T)
    if np.isclose(det, 0.0):
        raise ValueError(f"Degenerate rotation matrix encountered. Please check '{filename}'.")
    d = 1.0 if det >= 0.0 else -1.0
    D = np.diag([1.0, 1.0, d])

    # --- Building the optimal rotation matrix ---
    R = Vt.T @ D @ U.T

    # --- Rotating the geometry ---
    geometry_2_rotated = geometry_2 @ R.T

    # --- Computing the IRC values ---
    structural_distance = np.sqrt(np.sum(atom_masses[:, None] * (geometry_2_rotated - geometry_1) ** 2)) * angstrom_to_bohr_radius

    return structural_distance


# --- Defining the function to read the IRC coordinates, electronic energies, and ZPVEs from the IRC output file ---
def reading_orca_irc(irc_file_xyz: str | Path, ts_output_file: str | Path, proj_freq: bool=True) -> IRCData:

    irc_file_xyz_path = Path(irc_file_xyz)
    ts_output_file_path = Path(ts_output_file)

    # --- Extracting the atomic masses, list of the electronic energies, and coordinates ---
    atom_masses = orca_mass_extraction(ts_output_file_path)
    data_irc_el_energy, struct_coordinates = _orca_struct_extraction(irc_file_xyz_path)

    # --- Performing center of mass (CoM) correction ---
    com_corrected = [_center_of_mass_correction(geometry, atom_masses, irc_file_xyz_path) for geometry in struct_coordinates]

    # --- Computing the IRC steps ---
    irc_steps = [_struct_distance(com_corrected[i], com_corrected[i+1], atom_masses, irc_file_xyz_path) for i in range(len(com_corrected) - 1)]

    # --- Accumulating the IRC steps along the path ---
    data_irc = np.concatenate([[0.0], np.cumsum(irc_steps)])

    # --- Merging IRC coordinates with the electronic energies ---
    data_el_energy = np.column_stack((data_irc, data_irc_el_energy))

    # --- Correcting the IRC coordinates in order to make the TS at IRC = 0.0 and transforming the data to list[float] ---
    data_el_energy[:, 0] -= data_el_energy[np.argmax(data_el_energy[:, 1]), 0]
    data_el_energy = [(float(irc), float(energy)) for irc, energy in data_el_energy]

    # --- Extracting ZPVEs (No method of computing projected frequencies is available in ORCA by now [01.09.2026]) ---
    if proj_freq:
        zpve_energies_forward = _orca_freq_extraction('forward', sign=1) # None
        zpve_energies_reverse = _orca_freq_extraction('reverse', sign=-1) # None
    else:
        zpve_energies_forward = None
        zpve_energies_reverse = None

    return IRCData(electronic_energies=data_el_energy, zpve_energies_forward=zpve_energies_forward, zpve_energies_reverse=zpve_energies_reverse)

    
# --- Defining the function to store the frequencies of the transition state and minima (reactant, product) ---
def reading_orca_struct_file(opt_file_out: Path, mode: str='ts') -> tuple[tuple[float, float], tuple[float, float]]:

    # --- Checking if we correctly defined the mode ---
    mode_positions = {"ts": 0.0, "react": float("-inf"), "prod": float("inf")}
    if mode not in mode_positions:
        raise ValueError(f"The inserted mode ('{mode}') is incorrect. Please use 'react', 'prod', or 'ts'.")
    position = mode_positions[mode]

    # --- Opening the file and reading the data ---
    with open(opt_file_out, encoding="utf-8") as f_in:
        text = f_in.read()

    # --- Searching for the electronic energy and ZPVE ---
    matches = orca_patterns.pattern_el_energy_zpve.findall(text)
    if not matches:
        raise ValueError(f"The electronic energy of the species was not found. Please check '{opt_file_out}'.")
    
    el_energy = (position, float(matches[-1][0]))
    zpve_energy = (position, float(matches[-1][1]))

    return el_energy, zpve_energy