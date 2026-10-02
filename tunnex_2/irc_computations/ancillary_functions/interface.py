# --- Modules ---
import shlex
import subprocess
from pathlib import Path


# --- Defining the function which checks if the file exist ---
def file_check(filename: str | Path) -> Path:
    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"File '{path}' was not found. Please check '{path}'.")
    return path


# --- Defining the function to read the command to execute the quantum chemistry program (Gaussian, ORCA etc.) at the given computer ---
def _insert_command(command_file: str | Path) -> str:
    command_file = Path(command_file)

    if command_file.is_file() and command_file.stat().st_size > 0:
        return command_file.read_text(encoding="utf-8").strip()

    command_line = input("Please specify the command used to run Gaussian or ORCA:\n").strip()
    command_file.write_text(command_line, encoding="utf-8")
    return command_line


# --- Defining the function to run the Gaussian, ORCA or another quantum chemistry program ---
def run_software(filename: str | Path, command_file: str | Path) -> None:
    file_check(filename)
    command_line = _insert_command(Path(command_file))
    subprocess.run([*shlex.split(command_line), str(filename)], check=True)
    return None


# --- Defining the function to read Gaussian output files ---
def gauss_out_filename(filename: str | Path) -> Path:
    return Path(filename).with_suffix(".out")


# --- Defining the function to read Orca output files ---
def orca_out_filename(filename: str | Path, suffix: str) -> Path:
    if suffix.lower() == ".xyz":
        return Path(filename).with_stem(Path(filename).stem + "_IRC_Full_trj").with_suffix(suffix)
    else:
        return Path(filename).with_suffix(suffix)


# --- Checking Gaussian output for abnormal termination ---
def gauss_error_check(filename: str | Path) -> None:

    path_filename = file_check(filename)
    text = path_filename.read_text(encoding="utf-8")

    marker = "Error termination via"
    pos = text.find(marker)
    if pos == -1:
        return None

    raise RuntimeError(f"Gaussian terminated with an error in '{filename}':\n\n"f"{text[pos:]}")


# --- Checking ORCA output for normal termination ---
def _orca_normal_term_check(file: str | Path) -> None:
    text = file.read_text(encoding="utf-8")
    if "****ORCA TERMINATED NORMALLY****" not in text:
        raise RuntimeError(f"ORCA terminated with an error in '{file}'. Please check '{file}'.")
    return None


# --- Checking ORCA output for existing and normal termination ---
def orca_error_check(filename: str | Path) -> None:

    suffix_list = (".xyz", ".out")

    path_filename = file_check(filename)
    suffix = path_filename.suffix.lower()

    if suffix == suffix_list[0]:
        new_filename = path_filename.with_stem(path_filename.stem.removesuffix("_IRC_Full_trj")).with_suffix(".out")
        new_path_filename = file_check(new_filename)
        _orca_normal_term_check(new_path_filename)

    elif suffix == suffix_list[1]:
        _orca_normal_term_check(path_filename)

    else:
        raise ValueError(f"Found the undocumented suffix '{suffix}' in '{filename}'. Expected '.xyz' or '.out'.")

    return None