# --- Modules ---
import re


# --- Gaussian file parcing patterns (Cartesian Gaussian geometry is expected in all cases) ---
pattern_computational_method = re.compile(r'^(#\S+)\s+(\S+).*$')
# It searches for a method line in a gaussian input file. The input line should be '#p METHOD/BASIS ...'
ts_optimization_line = "opt=(ts,calcfc,noeigen) freq=NoRaman"
# A line to add to the input-file for the optimization of the geometry of the transition state
pattern_comment = 'Comment'
# A line defining the comment section


pattern_optimized_ts_geometry = re.compile(r"Standard orientation:\s*-+\s*Center.*?\s*-+\s*(.*?)\s*-+\s*Rotational constants", re.DOTALL)
# It searches the blocks containing the optimized geometry
pattern_geometry_line_output = re.compile(
r'^\s*(\d+)\s+'
r'(\d+)\s+'
r'(\d+)\s+'
r'([-+]?\d*\.\d+)\s+'
r'([-+]?\d*\.\d+)\s+'
r'([-+]?\d*\.\d+)\s*$',
re.MULTILINE)
# It searches the pattern of type '1 6 0 0.000000 0.635864 0.000000' in the output files
pattern_geometry_line_input = re.compile(
    r"^\s*([A-Z][a-z]?|\d+)\s+"
    r"([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s*$",
    re.MULTILINE)
# It searches the pattern of type 'C 0.000000 1.234567 -0.123456' in the input files


irc_split_marker_forward = "Calculation of FORWARD path complete."
# It is a marker to split the IRC file to the forward (before the marker) and reverse (after the marker) irc fragments
irc_split_marker_reverse = "Calculation of REVERSE path complete."
# It is a marker to identify the end the irc-computation in the reverse direction 
pattern_ts_energy = re.compile(r"Energies reported relative to the TS energy of\s+([-+]?\d+\.\d+)")
# It searches the energy of the transition state in the irc output file
pattern_el_energy = re.compile(r"^\s*\d+\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)", re.MULTILINE)
# It searches the IRC lines in the electronic irc table at the end of the irc output-file
pattern_freq = re.compile(r"-?\d+\.\d+")
# It is a pattern to search the frequencies in the 'Frequencies --  -2091.8757    36.5420    306.8203' line
pattern_irc_coordinate = re.compile(r"=\s*(-?\d+\.\d+)")
# It is a pattern to search the irc-coordinate in the 'NET REACTION COORDINATE UP TO THIS POINT = 0.42130' line


pattern_geometry_irc_line_output = re.compile(
r'^\s*\d+\s+(\d+)\s+'
r'([-+]?\d*\.\d+)\s+'
r'([-+]?\d*\.\d+)\s+'
r'([-+]?\d*\.\d+)\s*$',
re.MULTILINE)
# It is a pattern to find the geometry of the species (reactant and product) in the irc output-file
pattern_react_prod_geometry = re.compile(r"\*{4}\s+End of Projected Frequency Analysis\s+\*{4}")
# This pattern opens the geometry table
method_tail_react_prod = 'Opt freq=NoRaman'
# This pattern is used for the optimization of the reactant (product) geometries

pattern_el_energy_single_point = re.compile(r"SCF Done:\s+E\([^)]+\)\s*=\s*(-?\d+\.\d+)")
# It searches for the electronic energy value in the geometry optimization output file

pattern_mode_coordinates = re.compile(
    r'^\s*'
    r'(\d+)\s+'                      # Atom
    r'(\d+)\s+'                      # AN
    r'([-+]?\d*\.\d+)\s+'            # X1
    r'([-+]?\d*\.\d+)\s+'            # Y1
    r'([-+]?\d*\.\d+)\s+'            # Z1
    r'([-+]?\d*\.\d+)\s+'            # X2
    r'([-+]?\d*\.\d+)\s+'            # Y2
    r'([-+]?\d*\.\d+)\s+'            # Z2
    r'([-+]?\d*\.\d+)\s+'            # X3
    r'([-+]?\d*\.\d+)\s+'            # Y3
    r'([-+]?\d*\.\d+)\s*$'           # Z3
)
# It searches for the vector coordinates of the corresponding vibrational modes. It is needed to compute the attempt frequency
pattern_mode_coordinates_one_mode = re.compile(
    r'^\s*'
    r'(\d+)\s+'                      # Atom
    r'(\d+)\s+'                      # AN
    r'([-+]?\d*\.\d+)\s+'            # X1
    r'([-+]?\d*\.\d+)\s+'            # Y1
    r'([-+]?\d*\.\d+)\s*$'           # Z1
)
# It searches for the vector coordinates of the corresponding vibrational modes. It is needed to compute the attempt frequency
pattern_mode_vectors = re.compile(
    r"and normal coordinates:(.*?)-+\s*\n\s*- Thermochemistry -",
    re.DOTALL,
)
# It searches for the block containing frequencies and vector coordinates. It is needed to compute the attempt frequency
pattern_vibr_mass = re.compile(r"Atom\s+\d+\s+has atomic number\s+\d+\s+and mass\s+([0-9]+(?:\.[0-9]+)?)")
# It searches for atomic masses to scale the vibrational shift vectors