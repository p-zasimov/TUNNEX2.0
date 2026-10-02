# --- Modules ---
from pathlib import Path
from tunnex_2.irc_computations.orca import orca_patterns # type: ignore


# --- Defining the auxiliary function to write the input file for a reactant (product) or the IRC input file ---
def create_orca_input_file(orca_ts_guess_input: Path, orca_species_output: Path, species_geometry: str | None, method_tail: str) -> None:

    try:
        with open(orca_ts_guess_input, "r", encoding="utf-8") as f_in, \
            open(orca_species_output, "w", encoding="utf-8") as f_species_out:

            geometry_found = species_geometry is None
            method_found = False
            inside_geom_block = False

            for line in f_in:
                stripped = line.strip()
                if stripped:

                    if stripped == r'%geom':
                        inside_geom_block = True
                        continue
                
                    if inside_geom_block:
                        if stripped == "end":
                            inside_geom_block = False
                        continue

                    match_method = orca_patterns.pattern_computational_method.match(stripped)

                    if match_method:
                        if method_found:
                            raise ValueError(f"Multiple computational method lines were found.",
                                             f"Please check the {orca_ts_guess_input} file.")
                        method_found = True
                        method, basis = match_method.groups()
                        f_species_out.write(f"! {method} {basis} {method_tail}\n")
                        continue

                    if species_geometry is not None:
                        match_geometry_line = orca_patterns.pattern_geometry_line_input.match(stripped)
                        if match_geometry_line:
                            geometry_found = True
                            break

                    f_species_out.write(line)

            if not geometry_found:
                raise ValueError(f"The geometry of the species was not found. Please check the {orca_ts_guess_input} file.")
            if not method_found:
                raise ValueError(f"The computational method was not found. Please check the {orca_ts_guess_input} file.")
            if inside_geom_block:
                raise ValueError(f"The %geom block was not closed with 'end'. Please check the {orca_ts_guess_input} file.")

            if species_geometry is not None:
                f_species_out.write(species_geometry)
                f_species_out.write("\n*\n\n\n")
            else:
                f_species_out.write("\n\n")

    except ValueError:
        orca_species_output.unlink(missing_ok=True)
        raise

    return None


# --- Defining the auxiliary function to read the optimized transition state geometry (assuming that the optimization is converged) ---
def _orca_extract_geom_from_opt_file(filename: Path) -> str:

    with open(filename, "r", encoding="utf-8") as f_in:
        text = f_in.read()

    geometry_blocks = orca_patterns.pattern_optimized_ts_geometry.findall(text)
    if not geometry_blocks:
        raise ValueError(f"The geometry of the transition state was not found. Please check {filename}.")

    last_geometry = geometry_blocks[-1]

    geometry_matches = (orca_patterns.pattern_geometry_line_output.findall(last_geometry))

    return "\n".join(" ".join(row) for row in geometry_matches)


# --- Defining the auxiliary function to take the reactant (product) geometry ---
def _orca_extract_geom_from_irc(filename: Path) -> str:

    with open(filename, "r", encoding="utf-8") as f_in:
        text = f_in.read()

    geometry_matches = orca_patterns.pattern_geometry_line_output.findall(text)
    if not geometry_matches:
        raise ValueError(f"The geometry of the reactant (product) was not found. Please check {filename}.")

    return "\n".join(" ".join(row) for row in geometry_matches)


# --- Defining the function to create the IRC input files
# (No method of computing projected frequencies is available in ORCA by now [01.09.2026]) ---
def create_orca_irc_input(orca_ts_guess_input: str | Path, orca_optimized_ts_file: Path)  -> Path:
    
    # --- Creating the filename of a new file ---
    input_file = Path(orca_ts_guess_input)
    orca_irc_input = input_file.with_stem(input_file.stem + "_irc")
    method_tail_irc = orca_patterns.irc_parameters_block

    # --- Opening the output files and reading the optimized geometry ---
    ts_geometry = _orca_extract_geom_from_opt_file(orca_optimized_ts_file)
    create_orca_input_file(orca_ts_guess_input, orca_irc_input, ts_geometry, method_tail_irc)

    return orca_irc_input


# --- Defining the function to create the file for the geometry optimization of the reactant (product) ---
def create_orca_react_prod_input(orca_ts_guess_input: str | Path) -> tuple[Path, Path]:

    # --- Creating the filenames of new files ---
    ts_guess_path = Path(orca_ts_guess_input)
    orca_react_input = ts_guess_path.with_stem(ts_guess_path.stem + "_react")
    orca_prod_input = ts_guess_path.with_stem(ts_guess_path.stem + "_prod")

    # --- Taking the reactant and product geometries ---
    react_irc_name = ts_guess_path.with_name(ts_guess_path.stem + "_irc_IRC_B.xyz")
    react_geom = _orca_extract_geom_from_irc(react_irc_name)
    prod_irc_name = ts_guess_path.with_name(ts_guess_path.stem + "_irc_IRC_F.xyz")
    prod_geom = _orca_extract_geom_from_irc(prod_irc_name)

    # --- Writing new files ---
    create_orca_input_file(orca_ts_guess_input, orca_react_input, react_geom, orca_patterns.min_optimization_line)
    create_orca_input_file(orca_ts_guess_input, orca_prod_input, prod_geom, orca_patterns.min_optimization_line)
          
    return orca_react_input, orca_prod_input


###################################################################################################
# --- Defining the function to create the optimization input file for the input file provided by a user (OLD FUNCTION, NOT USED NOW) ---
def _create_orca_ts_input_old(orca_ts_guess_input: str | Path, orca_ts_opt_input : str | Path) -> None:
    
    # --- Writing a new file ---
    with open(orca_ts_guess_input, "r", encoding="utf-8") as f_in, \
            open(orca_ts_opt_input, "w", encoding="utf-8") as f_out:

        inside_geom_block = False
        
        for line in f_in:
            stripped = line.strip()

            if stripped == r'%geom':
                inside_geom_block = True
                continue
            
            if inside_geom_block:
                if stripped == r'end':
                    inside_geom_block = False
                continue

            match = orca_patterns.pattern_computational_method.match(stripped)

            if match:
                method, basis = match.groups()
                f_out.write(f"! {method} {basis} {orca_patterns.ts_optimization_line}\n\n")
                f_out.write(f"{orca_patterns.block_ts_optimization}\n")
            elif stripped:
                    f_out.write(line)

        if inside_geom_block:
            raise ValueError(r"The %geom block was not closed with 'end'. Please check your input file.")

    return None
###################################################################################################