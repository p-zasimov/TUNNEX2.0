# --- Modules ---
import re


# --- ORCA file parcing patterns ---
pattern_computational_method = re.compile(r'^\s*!\s+(\S+)\s+(\S+)')
# It searches for a method line in an orca input file. The input line should be '! METHOD BASIS ...'
pattern_ts_optimization = re.compile(r"(?ms)^%geom\b.*?^end\s*$")
# It searches for a specific block which contains the details of the transition state optimization
ts_optimization_block = r"""OptTS Freq

%geom
Calc_Hess true
NumHess true
end
"""
# A block to add to the input-file for the optimization of the geometry of the transition state


pattern_optimized_ts_geometry = re.compile(
    r'CARTESIAN COORDINATES \(ANGSTROEM\)\n'
    r'-+\n'
    r'((?:[ \t]*(?:[A-Za-z]{1,2}|\d+)(?:[ \t]+-?\d+\.\d+){3}[ \t]*\n?)+)'
)
# It searches for the blocks containing the optimized geometry
pattern_geometry_line_output = re.compile(
    r"^\s*([A-Z][a-z]?|\d+)\s+"
    r"([-+]?\d+(?:\.\d+)?)\s+"
    r"([-+]?\d+(?:\.\d+)?)\s+"
    r"([-+]?\d+(?:\.\d+)?)\s*$",
    re.MULTILINE
)
# It searches the pattern of type 'С 0.000000 0.635864 0.000000' in the output files
pattern_geometry_line_input = re.compile(
    r"^\s*([A-Z][a-z]?|\d+)\s+"
    r"([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s*$",
    re.MULTILINE)
# It searches the pattern of type 'C 0.000000 1.234567 -0.123456' in the input files


pattern_irc_coordinates_block = re.compile(
    r"E\s+([-+]?\d+(?:\.\d+)?)\s*\n"
    r"("
    r"(?:"
    r"[ \t]*(?:[A-Za-z]{1,2}|\d+)"
    r"[ \t]+[-+]?\d+(?:\.\d+)?"
    r"[ \t]+[-+]?\d+(?:\.\d+)?"
    r"[ \t]+[-+]?\d+(?:\.\d+)?"
    r"[ \t]*\n?"
    r")+"
    r")"
)
# It searches the coordinates block in the irc output file
pattern_irc_geometry = re.compile(
    r"^\s*((?:[A-Za-z]{1,2}|\d+))\s+"
    r"([-+]?\d+(?:\.\d+)?)\s+"
    r"([-+]?\d+(?:\.\d+)?)\s+"
    r"([-+]?\d+(?:\.\d+)?)\s*$",
    re.MULTILINE
)
# It searches the coordinates in the coordinates block in the irc output file


pattern_mass_block = re.compile(
    r'CARTESIAN COORDINATES \(A\.U\.\)\n'
    r'-+\n'
    r'[ \t]*NO[ \t]+LB[ \t]+ZA[ \t]+FRAG[ \t]+MASS[ \t]+X[ \t]+Y[ \t]+Z[ \t]*\n'
    r'((?:'
    r'[ \t]*\d+[ \t]+'
    r'(?:[A-Za-z]{1,2}|\d+)[ \t]+'
    r'[-+]?\d+(?:\.\d+)?[ \t]+'
    r'\d+[ \t]+'
    r'[-+]?\d+(?:\.\d+)?[ \t]+'
    r'[-+]?\d+(?:\.\d+)?[ \t]+'
    r'[-+]?\d+(?:\.\d+)?[ \t]+'
    r'[-+]?\d+(?:\.\d+)?[ \t]*'
    r'\n?'
    r')+)'
)
# It searches the mass block in the ts output file
pattern_atom_mass = re.compile(
        r'\d+\s+(?:[A-Za-z]{1,2}|\d+)\s+-?\d+\.\d+\s+\d+\s+(-?\d+\.\d+)\s+-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+'
    )
# It searches the masses in the mass block in the ts output file


pattern_charge_mult = re.compile(r'^\* xyz.*$')
# It serches for the charge and spin of a molecule
min_optimization_line = "Opt Freq\n"
# A line to add to the input-file for the optimization of the geometry of the transition state


pattern_el_energy_zpve = re.compile(
    r'Electronic energy\s+\.\.\.\s+([-+]?\d+\.\d+)\s+Eh\s+'
    r'Zero point energy\s+\.\.\.\s+([-+]?\d+\.\d+)\s+Eh'
)
# It searches for the electronic energy and ZPVE in output files


pattern_vibr_block = re.compile(
    r'-+\s*VIBRATIONAL FREQUENCIES\s*-+'
    r'(.*?)'
    r'-+\s*NORMAL MODES\s*-+',
    re.DOTALL
)
# It searches for the vibrational frequencies block in output files
pattern_vibr_frequencies = re.compile(
    r'^\s*\d+:\s+([-+]?\d+(?:\.\d+)?)\s+cm\*\*-1',
    re.MULTILINE
)
# It searches for the vibrational frequencies in output files
pattern_normal_modes = re.compile(
    r'------------\s*'
    r'NORMAL MODES\s*'
    r'------------\s*'
    r'.*?'
    r'Thus, these vectors are normalized but \*not\* orthogonal\s*'
    r'(.*?)'
    r'\s*-----------\s*'
    r'IR SPECTRUM',
    re.DOTALL
)
# It searches for the normal modes in output files


irc_parameters_block = f"""IRC

%irc
    MaxIter 100
    PrintLevel 1
    Direction both
    InitHess calc_anfreq
    hessMode 0
    Scale_Init_Displ 0.10
    Scale_Displ_SD   0.10
    Adapt_Scale_Displ false
end
"""
# It is needed to be put inside the IRC input-file for ORCA