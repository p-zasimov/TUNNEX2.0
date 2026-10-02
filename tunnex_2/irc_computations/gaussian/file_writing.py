# --- Modules ---
from pathlib import Path
from tunnex_2.irc_computations.gaussian import gauss_patterns # type: ignore


# --- Defining the auxiliary function to write the input file for a transition state, reactant (product) or the IRC input file.
# If the last three parameters are None, then we are making the transition state input file from an initial guess ---
def create_gauss_input_file(gauss_ts_guess_input: Path, gauss_species_output: Path, species_geometry: str | None, method_tail: str) -> None:

    try:
        with open(gauss_ts_guess_input, "r", encoding="utf-8") as f_in, \
            open(gauss_species_output, "w", encoding="utf-8") as f_species_out:

            geometry_found = species_geometry is None
            method_found = False

            for line in f_in:
                stripped = line.strip()
                if stripped:

                    match_method = gauss_patterns.pattern_computational_method.match(stripped)

                    if match_method:
                        if method_found:
                            raise ValueError(f"Multiple computational method lines were found. Please check '{gauss_ts_guess_input}'.")
                        method_found = True
                        prefix, method_and_basis = match_method.groups()
                        f_species_out.write(f"{prefix} {method_and_basis} {method_tail}\n\n")
                        continue

                    if species_geometry is not None:
                        match_geometry_line = gauss_patterns.pattern_geometry_line_input.match(stripped)
                        if match_geometry_line:
                            geometry_found = True
                            break

                    f_species_out.write(line)

                    if stripped == gauss_patterns.pattern_comment:
                        f_species_out.write("\n")

            if not geometry_found:
                raise ValueError(f"The geometry of the species was not found. Please check '{gauss_ts_guess_input}'.")
            if not method_found:
                raise ValueError(f"The computational method was not found. Please check '{gauss_ts_guess_input}'.")
            
            if species_geometry is not None:
                f_species_out.write(species_geometry)
                f_species_out.write("\n\n\n\n")
            else:
                f_species_out.write("\n\n")

    except ValueError:
        gauss_species_output.unlink(missing_ok=True)
        raise

    return None


# --- Defining the auxiliary function to read the optimized transition state geometry (assuming that the optimization is converged) ---
def _gauss_extract_geom_from_opt_file(file: Path) -> str:

    with open(file, "r", encoding="utf-8") as f_in:
        text = f_in.read()

    geometry_blocks = gauss_patterns.pattern_optimized_ts_geometry.findall(text)
    
    if not geometry_blocks:
        raise ValueError(f"The geometry of the transition state was not found. Please check '{file}'.")

    last_geometry = geometry_blocks[-1]

    geometry_lines = [f"{atom} {x} {y} {z}" for _, atom, _, x, y, z in gauss_patterns.pattern_geometry_line_output.findall(last_geometry)]
    if not geometry_lines:
            raise ValueError(f"The transition-state geometry block was found, but no geometry coordinates could be extracted.",
                             f"Please check '{file}'.")

    return "\n".join(geometry_lines)


# --- Defining the auxiliary function to take the reactant (product) geometry ---
def _gauss_extract_geom_from_irc(file: Path, marker: str) -> str:
  
    with open(file, "r", encoding="utf-8") as f_in:
        text = f_in.read()
        end = text.find(marker)
        if end == -1:
            raise ValueError(f"'{marker}' was not found. Please check '{file}'.")

        last_match = None

        for geometry_match in gauss_patterns.pattern_react_prod_geometry.finditer(text, 0, end):
            last_match = geometry_match

        if last_match is None:
            raise ValueError(f"The geometry block of a reactant (product) was not found. Please check '{file}'.")

        geometry_block = text[last_match.end():end]

        geometry_lines = [f"{atom} {x} {y} {z}" for atom, x, y, z in gauss_patterns.pattern_geometry_irc_line_output.findall(geometry_block)]
        if not geometry_lines:
            raise ValueError(f"The geometry of a reactant (product) was not found in the geometry block. Please check '{file}'.")

    return "\n".join(geometry_lines)


# --- Defining the function to create the IRC input files ---
def create_gauss_irc_input(gauss_ts_guess_input: str | Path, gauss_optimized_ts_file: Path, proj_freq: bool=True, calc_all: bool=True, max_points: int=50, max_cycle: int=40, step_size: float=-10)  -> Path:
    
    # --- Creating the filename of a new file ---
    input_file = Path(gauss_ts_guess_input)
    gauss_irc_input = input_file.with_stem(input_file.stem + "_irc")

    # --- Specifies that the force constants be computed at every point ('calcall') or only at the first point ('calcfc').
    # The first one (default) is slower, but more accurate for projected frequencies ---
    calc_type = "calcall" if calc_all else "calcfc"
    method_tail_irc = f"irc=({calc_type},maxpoints={max_points},maxcycle={max_cycle},stepsize={step_size})"

    # --- Choosing whether one should compute the projected frequencies. Computing them by default ---
    if proj_freq:
        method_tail_irc += " iop(1/73=2)"

    # --- Opening the output files and reading the optimized geometry ---
    ts_geometry = _gauss_extract_geom_from_opt_file(gauss_optimized_ts_file)
    create_gauss_input_file(gauss_ts_guess_input, gauss_irc_input, ts_geometry, method_tail_irc)

    return gauss_irc_input


# --- Defining the function to create the file for the geometry optimization of a reactant (product) ---
def create_gauss_react_prod_input(irc_file_out: Path, gauss_ts_guess_input: Path) -> tuple[str, str]:

    # --- Creating the filenames of new files ---
    ts_guess_path = Path(gauss_ts_guess_input)
    gauss_react_input = ts_guess_path.with_stem(ts_guess_path.stem + "_react")
    gauss_prod_input = ts_guess_path.with_stem(ts_guess_path.stem + "_prod")

    # --- Taking the reactant and product geometries ---
    react_geom = _gauss_extract_geom_from_irc(irc_file_out, gauss_patterns.irc_split_marker_reverse)
    prod_geom = _gauss_extract_geom_from_irc(irc_file_out, gauss_patterns.irc_split_marker_forward)
        
    # --- Writing the input files for the reactant and product ---
    create_gauss_input_file(gauss_ts_guess_input, gauss_react_input, react_geom, gauss_patterns.method_tail_react_prod)
    create_gauss_input_file(gauss_ts_guess_input, gauss_prod_input, prod_geom, gauss_patterns.method_tail_react_prod)

    return gauss_react_input, gauss_prod_input


###################################################################################################
# --- Defining the function to create the optimization input file for the input file provided by a user (OLD FUNCTION, NOT USED NOW) ---
def _create_gauss_ts_input_old(gauss_ts_guess_input: str) -> Path:
    
    # --- Creating the filename of a new file ---
    input_file = Path(gauss_ts_guess_input)
    if not input_file.is_file():
        raise FileNotFoundError(f"Gaussian input file was not found: {input_file}")

    gauss_ts_opt_input = input_file.with_stem(input_file.stem + "_ts")

    # --- Writing a new file ---
    with open(input_file, "r", encoding="utf-8") as f_in, \
            open(gauss_ts_opt_input, "w", encoding="utf-8") as f_out:

        method_found = False

        for line in f_in:
            match = gauss_patterns.pattern_computational_method.match(line.strip())
            if match:
                if method_found:
                    gauss_ts_opt_input.unlink(missing_ok=True)
                    raise ValueError("Multiple computational method lines were found. Please check your data.")

                method_found = True
                prefix, method_and_basis = match.groups()
                f_out.write(f"{prefix} {method_and_basis} {gauss_patterns.ts_optimization_line}\n")
            else:
                f_out.write(line)

    if not method_found:
        gauss_ts_opt_input.unlink(missing_ok=True)
        raise ValueError("The computational method was not found. Please check your data.")

    return gauss_ts_opt_input
###################################################################################################