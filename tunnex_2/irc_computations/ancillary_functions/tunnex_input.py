# --- Modules ---
from pathlib import Path
from tunnex_2.irc_computations.constants_and_dataclasses.dataclasses import TunnexInputSettings # type: ignore


# --- Writing the standard setting for three header lines ---
def write_input_head(
    direction: str,
    filename: Path,
    tunnex_settings: TunnexInputSettings,
) -> Path:

    filename_out = filename.with_stem(filename.stem + direction).with_suffix(".txt")

    header = f"""\
freq={tunnex_settings.freq:.2f}; T={tunnex_settings.T}
E0={tunnex_settings.E0:.6f}; ZPVE0={tunnex_settings.ZPVE0:.6f}; potential_scaling_factor={tunnex_settings.potential_scaling_factor}; N={tunnex_settings.number_of_levels}
T_arrhenius_min={tunnex_settings.T_arrhenius_min}; T_arrhenius_max={tunnex_settings.T_arrhenius_max}; T_step={tunnex_settings.T_step}

# freq in cm-1, T in K; E0, ZPVE0 in Hartree; potential_scaling_factor and N are dimensionless
# IRC in sqrt(amu) * bohr; E, ZPVE in Hartree
"""
    
    with open(filename_out, "w", encoding="utf-8") as f:
        f.write(header)

    return filename_out


# --- Writing the electronic energy to the file ---
def writing_el_energy(filename: str | Path, el_energy: list[list[float]], reverse_key: bool = False) -> None:

    el_energy = sorted(el_energy, key=lambda row: row[0], reverse=reverse_key)

    irc_sign = -1 if reverse_key else 1

    lines = [
        "\n\n",
        "IRC; E\n\n",
        *(f"{irc_sign * irc:.6f}; {energy:.6f}\n" for irc, energy in el_energy),
        "\nEND; E\n",
    ]
    
    with open(filename, "a", encoding="utf-8") as f:
        f.writelines(lines)

    return None


# --- Writing the ZPVE to the file ---
def writing_zpve_energy(filename: str | Path, el_energy : list[list[float]], zpve_forward : list[tuple[float, float]] | None, zpve_reverse : list[tuple[float, float]] | None, ts_zpve_energy: tuple[float, float], reverse_key: bool = False) -> None:

    irc_sign = -1 if reverse_key else 1

    with open(filename, "a", encoding="utf-8") as f:
        if zpve_forward is None:
            zpve_data = [(el_energy[0][0], 0.0), (el_energy[-1][0], 0.0)]
        else:
            zpve_data = [*zpve_reverse, ts_zpve_energy, *zpve_forward]

        zpve_data.sort(key=lambda row: row[0], reverse=reverse_key)

        irc_sign = -1 if reverse_key else 1

        lines = [
            "\n\n",
            "IRC; ZPVE\n\n",
            *(f"{irc_sign * irc:.6f}; {energy:.6f}\n" for irc, energy in zpve_data),
            "\nEND; ZPVE\n",
        ]

        with open(filename, "a", encoding="utf-8") as f:
            f.writelines(lines)

    return None