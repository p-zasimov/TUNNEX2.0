# --- Constants --- 
cm_m1_to_Hartree = 1 / 219474.63136320  # cm-1 to Hartree
kJ_mol_m1_to_Hartree = 1 / 2625.49963948  # kJ mol-1 to Hartree
angstrom_to_bohr_radius = 1.8897259886 # Angstrom to Bohr radius


# --- Constants (Eckart potential) --- 
cm_m1_to_Hz = 2.99793e+10 # cm-1 to Hz (or sec-1)
Hartree_to_J = 4.3597447222e-18 # Hartree to J
amu_to_kg = 1.66053906893e-27 # Atomic mass unit (amu, also known as Dalton) to kg
bohr_radius_to_m = 5.29177210544e-11 # Bohr radius to meter


# --- Settings (non-numbers) ---
corr_analysis_bool = True # Useful tool for checking the correlation between the reaction coordinate and reactant frequencies.
# Helpful to determine the attempt frequency
command_file_path = "command_settings.txt" # Where to store the command line
run_software_key_marker = False # The key defining whether to compute the files or not


# --- Settings (numbers) ---
curve_smoothness_ratio_tol = 100.0 # Defines the sensitivity of the code to the ZPVE data smoothness