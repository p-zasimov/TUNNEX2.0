# --- Modules ---
from pathlib import Path
import numpy as np
from tunnex_2.qmt_computations.constants_and_models.constants import Hartree_to_kJ_mol_m1  # type: ignore


# --- Parameters of the output-file format ---
char_num_val = 28   # Number of characters per section (in general)
char_num_T = 4      # Number of characters per section (temperature)
char_num_offset = 8 # Number of characters per section (offset)
round_num_val = 6   # Number of digits for rounding (in general)
round_num_T = 2     # Number of digits for rounding a temperature


# --- Checking the format of the input-value ---
def _fmt(val: float, char_num: int, round_num: int, is_int: bool = False) -> str:
    if is_int:
        return f"{val:^{char_num}.2f}"
    elif val is None:
        return f"{'No turning point':^{char_num}}"
    elif np.isinf(val):
        return f"{'+inf':^{char_num}}"
    else:
        return f"{val:^{char_num}.{round_num}e}"


# --- Printing the vibrational level indexes and relative parameters (part 1) ---
def _section_level_results_1(data: object) -> str:

    lines = []
    lines.append("#" * 165)
    lines.append(
        "The vibrational level indexes, energies, corresponding turning points (tp), and WKB integrals".center(
            165
        )
    )
    lines.append("#" * 165)
    lines.append("")

    header = (
        f"{'N':^{char_num_val}} | {'Vibrational energy, kJ mol-1':^{char_num_val}} | "
        f"{'tp (left), sqrt(amu) * bohr':^{char_num_val}} | {'tp (right), sqrt(amu) * bohr':^{char_num_val}} | {'WKB integral, sqrt(Hartree) * amu * bohr':^{char_num_val}}"
    )
    lines.append(header)
    lines.append("-" * 165)

    for level in data:

        lines.append(
            f"{level.vib_index:^{char_num_val}} | "
            f"{level.vib_energy * Hartree_to_kJ_mol_m1:^{char_num_val}.{round_num_val}e} | "
            f"{_fmt(level.turning_point_left, char_num_val, round_num_val)} | "
            f"{_fmt(level.turning_point_right, char_num_val, round_num_val)} | "
            f"{_fmt(level.wkb, char_num_val, round_num_val)}"
        )

    lines.append("")

    return "\n".join(lines)


# --- Printing the vibrational level indexes and relative parameters (part 2) ---
def _section_level_results_2(data: object) -> str:

    lines = []
    lines.append("#" * 242)
    lines.append(
        "The vibrational level indexes, energies, corresponding transmission probabilities (P), reaction rate (k), and tunneling half-lives (t_1/2)".center(
            242
        )
    )
    lines.append("#" * 242)
    lines.append("")

    header = (
        f"{'N':^{char_num_val}} | {'Vibrational energy, kJ mol-1':^{char_num_val}} | "
        f"{'P':^{char_num_val}} | {'k, s-1':^{char_num_val}} | "
        f"{'t_1/2, seconds':^{char_num_val}} | {'t_1/2, hours':^{char_num_val}} | "
        f"{'t_1/2, days':^{char_num_val}} | {'t_1/2, years':^{char_num_val}}"
    )
    lines.append(header)
    lines.append("-" * 242)

    for level in data:

        lines.append(
            f"{level.vib_index:^{char_num_val}} | "
            f"{level.vib_energy * Hartree_to_kJ_mol_m1:^{char_num_val}.{round_num_val}e} | "
            f"{_fmt(level.transmission_probability, char_num_val, round_num_val)} | "
            f"{_fmt(level.reaction_rate, char_num_val, round_num_val)} | "
            f"{_fmt(level.half_s, char_num_val, round_num_val)} | "
            f"{_fmt(level.half_h, char_num_val, round_num_val)} | "
            f"{_fmt(level.half_d, char_num_val, round_num_val)} | "
            f"{_fmt(level.half_y, char_num_val, round_num_val)}"
        )

    lines.append("")

    return "\n".join(lines)


# --- Printing the temperature-averaged parameters ---
def _section_temperature_aver_results(data: object, temperature: float) -> str:

    lines = []
    lines.append("#" * 242)
    lines.append(
        f"The average transmission probabilities (P), reaction rate (k), and tunneling half-lives (t_1/2) for T = {temperature:^{char_num_T}.{round_num_T}f} K".center(
            242
        )
    )
    lines.append("#" * 242)
    lines.append("")

    header = (
        f"{'Temperature':^{char_num_val}} | {'T, K':^{char_num_val}} | {'P':^{char_num_val}} | {'k, s-1':^{char_num_val}} | "
        f"{'t_1/2, seconds':^{char_num_val}} | {'t_1/2, hours':^{char_num_val}} | "
        f"{'t_1/2, days':^{char_num_val}} | {'t_1/2, years':^{char_num_val}}"
    )
    lines.append(header)
    lines.append("-" * 242)

    lines.append(
        f"{'averaged data':^{char_num_val}} | "
        f"{temperature:^{char_num_val}.{round_num_T}f} | "
        f"{_fmt(data.transmission_probability, char_num_val, round_num_val)} | "
        f"{_fmt(data.reaction_rate, char_num_val, round_num_val)} | "
        f"{_fmt(data.half_s, char_num_val, round_num_val)} | "
        f"{_fmt(data.half_h, char_num_val, round_num_val)} | "
        f"{_fmt(data.half_d, char_num_val, round_num_val)} | "
        f"{_fmt(data.half_y, char_num_val, round_num_val)}"
    )
    lines.append("")

    return "\n".join(lines)


# --- Printing the Arrhenius plot data (tunneling only!) ---
def _section_arrhenius(data: object) -> str:
    lines = []

    lines.append("#" * 90)
    lines.append("Arrhenius plot data (tunneling only!)".center(90))
    lines.append("#" * 90)
    lines.append("")

    header = f"{'T, K':^{char_num_val}} | {'1/T, K':^{char_num_val}} | {'ln(k), s-1':^{char_num_val}}"
    lines.append(header)
    lines.append("-" * 90)

    for row in data:
        lines.append(
            f"{_fmt(row.arrhenius_temperature, char_num_val, round_num_val)} | "
            f"{_fmt(row.arrhenius_temperature_inv, char_num_val, round_num_val)} | "
            f"{_fmt(row.log_reaction_rate, char_num_val, round_num_val)}"
        )

    lines.append("")

    return "\n".join(lines)


# --- Printing the interpolated and ZPVE-corrected IRC-curve ---
def _section_irc_interpol(data: object) -> str:
    lines = []

    lines.append("#" * 101)
    lines.append(
        "The ZPVE-corrected IRC-surface approximated by a cubic spline after offsetting and potential scaling".center(
            101
        )
    )
    lines.append("#" * 101)
    lines.append("")

    header = f"{'IRC, sqrt(amu) * bohr':^{char_num_val}} | {'Energy, kJ mol-1':^{char_num_val}}"
    lines.append(header)
    lines.append("-" * 60)

    for row in data:
        lines.append(
            f"{row['irc_value']:^{char_num_val}.{round_num_val}f} | {row['energy_value'] * Hartree_to_kJ_mol_m1:^{char_num_val}.{round_num_val}f}"
        )

    lines.append("")

    return "\n".join(lines)


# Printing the offset, defined as E(IRC_min) + ZPVE(IRC_min) − E0 − ZPVE0
def _section_offset(offset: float) -> str:
    lines = []

    lines.append("#" * 80)
    lines.append("Offset is defined as E(IRC_min) + ZPVE(IRC_min) − E0 − ZPVE0")
    lines.append("#" * 80)
    lines.append("")
    lines.append(
        f"{f'offset = {offset * Hartree_to_kJ_mol_m1:^{char_num_offset}.{round_num_val}f} kJ mol-1':^80}"
    )
    lines.append("")

    return "\n".join(lines)


def build_report(
    data_level: object,
    data_temperature_averaged: object,
    temperature: float,
    data_arrhenius: object,
    data_irc: object,
    offset: float,
) -> str:
    parts = []

    parts.append(_section_level_results_1(data_level))
    parts.append(_section_level_results_2(data_level))
    parts.append("")
    parts.append(
        _section_temperature_aver_results(data_temperature_averaged, temperature)
    )
    parts.append("")
    parts.append(_section_arrhenius(data_arrhenius))
    parts.append("")
    parts.append(_section_irc_interpol(data_irc))
    parts.append(_section_offset(offset))

    return "\n\n".join(parts)


# --- Opening a new file in a write mode and writing the data ---
def write_output(path: Path, report: str) -> None:
    path.write_text(report, encoding="utf-8")
    return None