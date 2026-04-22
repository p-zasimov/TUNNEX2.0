# (1) *** DEFINING CONSTANTS AND READING INPUT-FILE  ***
# --- Importing NumPy, Path, and SciPy ---
import numpy as np
from pathlib import Path
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.integrate import quad


# --- Defining constants ---
revelo = 1822.8886259874  # A unified atomic mass unit (dalton) in electron mass (atomic units)
k_b = 3.166811563e-6  # Boltzmann constant in Hartree K-1
light = 29979245800  # Speed of light in cm s-1
cm_m1_to_Hartree = 1 / 219474.63136320  # cm-1 to Hartree
Hartree_to_kJ_mol_m1 = 2625.49963947  # Hartree to kJ mol-1

# --- Defining the folder path and file names ---
folder_path = Path(
    r"D:\python_projects\TUNNEX_2\v0p5_and_older_versions"
)  # Folder path

input_file_name = "input_v0p5.txt"  # Input-file name
output_file_name = "output_v0p5.txt"  # Output-file name

input_file_path = folder_path / input_file_name
output_file_path = folder_path / output_file_name

# By the convention suggesting that the reagent is on the left side, and the product is on the right side
# Also suggesting that in the input file, IRCs increase from top to bottom, both for the electronic energy and ZPVE
# If it is not the case, please fix your input file, or own the errors and weird results

# The program also uses the input electronic energy IRC-matrix to estimate the IRC coordinate of the transition state
# (it should have a maximum of an electronic energy)


# --- Opening an input-file and read all lines ---
with open(input_file_path, "r") as f_in:
    lines = f_in.readlines()


# --- Defining the function to read the lines and reading the lines ---
def parse_line(line):
    return {
        key.strip(): float(value)
        for key, value in (item.split("=") for item in line.strip().split(";"))
    }


params0 = parse_line(lines[0])
params1 = parse_line(lines[1])
params2 = parse_line(lines[2])


# --- Reading the first line (the attempt frequency [freq, in cm-1]
# and temperature [T; K]) ---
freq, T = params0["freq"], params0["T"]


# --- Reading the second line (the reagent's electronic energy [E0, Hartree],
# zero-point vibrational Energy [ZPVE0, Hartree],potential_scaling_factor [potential_scaling_factor])
# and the_upper_level_to_compute [N] ---
E0, ZPVE0, potential_scaling_factor, the_upper_level_to_compute = (
    params1["E0"],
    params1["ZPVE0"],
    params1["potential_scaling_factor"],
    int(params1["N"]),
)

# Potential_scaling_factor means a scaling factor for the final ZPVE-corrected IRC curve after offsetting. The default value is 1.0
# If the_upper_level_to_compute = 3, for example, then the code will compute the tunneling probabilities for the
# vibrational energy levels of v = 0, 1, 2, 3


# --- Reading the third line (minimum and maximum temperatures and a step [T; K] for an Arrhenius plot) ---
T_arrhenius_min, T_arrhenius_max, T_step = (
    params2["T_arrhenius_min"],
    params2["T_arrhenius_max"],
    params2["T_step"],
)


# --- Defining the function to validate the input arguments ---
def validate_positive(**kwargs):
    for name, value in kwargs.items():
        if name == "the_upper_level_to_compute":
            if value < 0:
                raise ValueError(f"{name} should be >= 0")
        elif value <= 0:
            raise ValueError(f"{name} should be > 0")


# --- Validating the input arguments ---
validate_positive(
    freq=freq,
    T=T,
    potential_scaling_factor=potential_scaling_factor,
    the_upper_level_to_compute=the_upper_level_to_compute,
    T_arrhenius_min=T_arrhenius_min,
    T_arrhenius_max=T_arrhenius_max,
    T_step=T_step,
)


# --- Defining the function to read the IRC_matrices [Nx2], [Kx2] and putting them in two separate lists ---
def read_IRC(start_tag, end_tag):
    col_x = []
    col_y = []
    reading = False
    comp_irc = float("-inf")

    for line in lines:
        line = line.strip()

        if line == start_tag:
            reading = True
            continue
        if line == end_tag:
            break

        if reading and line:
            values = [float(x) for x in line.split(";")]

            if values[0] <= comp_irc:
                raise ValueError("The set is not monotonically IRC-increasing")

            comp_irc = values[0]
            col_x.append(values[0])
            col_y.append(values[1])

    return col_x, col_y


# --- Reading the IRC_E [IRC, amu^0.5 * bohr]; electronic energy [E; Hartree] values
# and putting them in two separate lists ---
E_x, E_y = read_IRC("IRC; E", "END; E")


# --- Reading the IRC_ZPVE [IRC, amu^0.5 * bohr]; zero-point vibrational Energy [ZPVE; Hartree] values
# and putting them in two separate lists ---
ZPVE_x, ZPVE_y = read_IRC("IRC; ZPVE", "END; ZPVE")


# --- Computing the position of the transition state for future computations of turning points ---
idx_max_ts = int(np.argmax(E_y))
IRC_max_ts = E_x[idx_max_ts]


# (2) *** APPROXIMATION OF THE IRC-CURVES WITH MONOTONIC CUBIC SPLINES  ***
# --- Defining the interpolation function ---
E_interp = PchipInterpolator(E_x, E_y, extrapolate=False)
ZPVE_interp = PchipInterpolator(ZPVE_x, ZPVE_y, extrapolate=False)


# --- Computing an offset ---
if (
    abs(E_x[0] - ZPVE_x[0]) < 0.000001 and abs(E_x[-1] - ZPVE_x[-1]) < 0.000001
):  # Checking if IRC and ZPVE matrices start and end with the same values.
    # The float numbers cannot be compared directly, thus, the inequation is used
    offset = (E_y[0] + ZPVE_y[0]) - (E0 + ZPVE0)
    deviation = abs(
        offset / (E0 + ZPVE0)
    )  # Checking if the IRC is fully computed in the reagent's region
    if deviation >= 0.02:
        print(
            f"Warning: the deviation between the reagent[IRC] and submitted reagent's energy is {deviation * 100:.{1}f} %. More detailed IRC-computations in the region of the reagent may be necessary"
        )
else:
    raise ValueError(
        "IRC and ZPVE matrices should start or end with a different IRC-value. Please check the data"
    )


# --- Defining the final IRC-potential-curve, which was ZPVE-corrected and interpolated with a monotonic cubic spline ---
def final_potential(
    x,
    E_interp=E_interp,
    ZPVE_interp=ZPVE_interp,
    E_reagent=E_y,
    ZPVE_reagent=ZPVE_y,
    potential_scaling_factor_y=potential_scaling_factor,
):
    if not np.isfinite(E_interp(x)):
        raise ValueError("The value for the extrapolation is beyond the IRC-interval")
    else:
        final_potential_result = potential_scaling_factor_y * (
            E_interp(x) + ZPVE_interp(x) - (E_reagent[0] + ZPVE_reagent[0])
        )
    return final_potential_result


# --- Saving the ZPVE-corrected IRC-curve after the interpolation with a monotonic cubic spline ---
grid_size = 100
dx = (E_x[-1] - E_x[0]) / grid_size
IRC_cubic_spline = [
    (E_x[0] + i * dx, final_potential(E_x[0] + i * dx)) for i in range(grid_size + 1)
]


# (3) *** COMPUTING THE VIBRATIONAL ENERGY LEVELS AND TURNING POINTS  ***
# --- Computing vibrational energy levels ---
vib_level_energies = [
    (i + 0.5) * freq * cm_m1_to_Hartree
    for i in range(int(the_upper_level_to_compute) + 1)
]


# --- Defining the function which computes the turning points ---
def turning_points(level, E_x=E_x, func=final_potential, IRC_max_ts=IRC_max_ts):
    if level < func(E_x[0]):
        raise ValueError(
            "The energy level is below the reagent's energy. Please check your data"
        )
    # Because the reagent's energy is set to be zero after offset, and vibrational energy levels
    # are positive, this error means the problem with your data
    elif level < func(E_x[-1]):
        return (
            "Tunneling is not possible",
            "One turning point",
        )
    # It means that the energy is not smaller than the reagent's
    # energy, but smaller than the product's energy. From this level, tunneling is not possible, but the vibrational excitation can
    # make it possible
    elif level >= func(IRC_max_ts):
        return (
            "Over the barrier",
            "No turning points",
        )
    # The vibrational level's energy is bigger than the energy of the transition state in this case.
    # The resulting integral should be zero
    # In other cases, we should have two turning points
    else:
        func_target = lambda x: func(x) - level
        turning_point_left = brentq(func_target, E_x[0], IRC_max_ts)
        turning_point_right = brentq(func_target, IRC_max_ts, E_x[-1])
        return turning_point_left, turning_point_right


# --- Taking the computed vibrational energy level indexes and levels and appending them with turning points ---
# Filling the final output-matrix
output_values = []

for i, level in enumerate(vib_level_energies):
    row = [i, level]
    for turning_point in turning_points(level):
        row.append(turning_point)
    output_values.append(row)


# (4) *** COMPUTING THE WKB-INTEGRALS  ***
# --- Dealing with the cases, when tunneling is either not possible, or the energy level of a vibrational level is over the barrier ---
def wkb_integral(turning_point_left, turning_point_right, level, func):
    if turning_point_left == "Tunneling is not possible":
        return "+inf"
    elif turning_point_left == "Over the barrier":
        return 0
    else:  # Implementation of the wkb_function function implies that V(x) - E >= 0 for a given x.
        # However, for turning points, the binary search could converge to V(x) slightly smaller than E (< tol, the default tol=1e-12).
        # Thus, zero is returned for the cases where wkb_function_square <= 0
        wkb_function_int = lambda x: np.sqrt(
            np.maximum(0, 2 * revelo * (func(x) - level))
        )
        answer = quad(
            wkb_function_int, turning_point_left, turning_point_right, full_output=1
        )  # The default tolerance of integration is tol=1.49e-8. However, the other limiting factor,
        # the maximum number of steps is only max_iter=50. It is often exceeded, but I turned off the notifications about this.
    return answer[0]


# --- Computing the WKB-integrals ---
for row in output_values:
    level, turning_point_left, turning_point_right = row[1], row[2], row[3]
    wkb_integral_value = wkb_integral(
        turning_point_left, turning_point_right, level, final_potential
    )
    row.append(wkb_integral_value)


# (5) *** COMPUTING THE TRANSMISSON PROBABILITIES AND HALF-LIVES  ***
# --- Defining the function to compute the transmission probabilities and half-lives ---
def transmission_prob_and_half(wkb_integral_value, ts_freq, light_speed):
    if wkb_integral_value == "+inf":
        return 0, 0, "+inf", "+inf", "+inf", "+inf"
    else:
        t_prob = 1 / (1 + np.exp(2 * wkb_integral_value))  # A transmission probability
        reaction_rate = ts_freq * light_speed * t_prob  # A reaction rate in seconds-1
        half_s = np.log(2) / (reaction_rate)  # A half-life in seconds
        half_h = half_s / 3600  # A half-life in hours
        half_d = half_s / 86400  # A half-life in days
        half_y = half_s / 31536000  # A half-life in years
        return t_prob, reaction_rate, half_s, half_h, half_d, half_y


# --- Adding the computed value to the resulting matrix ---
for row in output_values:
    transmission_probability = transmission_prob_and_half(row[4], freq, light)
    for elem in transmission_probability:
        row.append(elem)


# (6) *** ANALYZING THE BOLTZMANN DISTRIBUTION FOR A GIVEN TEMPERATURE AND AVERAGING THE TRANSMISSON PROBABILITY ***
# --- Defining the average transmission probability ---
def boltzmann_average(matrix, ts_freq, temperature, boltzmann_constant, light_speed):
    sum_boltzmann = 0
    transmission_prob_average = 0
    for row in matrix:
        boltzmann_factor = np.exp(
            -(row[1] - matrix[0][1]) / (boltzmann_constant * temperature)
        )
        transmission_prob_average += boltzmann_factor * row[5]
        sum_boltzmann += boltzmann_factor
    return transmission_prob_average / sum_boltzmann


# --- Defining the average transmission probability ---
def transmission_prob_boltzmann(
    matrix, ts_freq, temperature, boltzmann_constant, light_speed
):
    transmission_probability_average = boltzmann_average(
        matrix, ts_freq, temperature, boltzmann_constant, light_speed
    )
    if transmission_probability_average == 0:
        reaction_rate_average = 0
        half_s_average = "+inf"
        half_h_average = "+inf"
        half_d_average = "+inf"
        half_y_average = "+inf"
    else:
        reaction_rate_average = (
            ts_freq * light_speed * transmission_probability_average
        )  # A reaction rate in seconds-1
        half_s_average = np.log(2) / (reaction_rate_average)  # A half-life in seconds
        half_h_average = half_s_average / 3600  # A half-life in hours
        half_d_average = half_s_average / 86400  # A half-life in days
        half_y_average = half_s_average / 31536000  # A half-life in years
    return (
        transmission_probability_average,
        reaction_rate_average,
        half_s_average,
        half_h_average,
        half_d_average,
        half_y_average,
    )


transmission_probability_average_result = transmission_prob_boltzmann(
    output_values, freq, T, k_b, light
)


# (7) *** COMPUTING THE DATA FOR AN ARRHENIUS PLOT ***
# --- Computing the average reaction rate ---
def reaction_rate_arr(matrix, ts_freq, temperature, boltzmann_constant, light_speed):
    transmission_probability_average = boltzmann_average(
        matrix, ts_freq, temperature, boltzmann_constant, light_speed
    )
    if transmission_probability_average == 0:
        reaction_rate_average = 0
    else:
        reaction_rate_average = (
            freq * light * transmission_probability_average
        )  # A reaction rate in seconds-1
    return reaction_rate_average


# --- Computing the average reaction rate for different temperatures ---
arrhenius_matrix = []

for arrhenius_T in range(
    int(T_arrhenius_max), int(T_arrhenius_min) - int(T_step), -int(T_step)
):
    if arrhenius_T <= 0:
        break
    else:
        arrhenius_matrix_row = [arrhenius_T, 1 / arrhenius_T]
        arrhenius_result = reaction_rate_arr(
            output_values, freq, arrhenius_T, k_b, light
        )
        if arrhenius_result > 0:
            arrhenius_matrix_row.append(np.log(arrhenius_result))
        else:
            arrhenius_matrix_row.append("-inf")
        arrhenius_matrix.append(arrhenius_matrix_row)


# (8) *** SAVING THE RESULTS IN A NEW FILE ***
# --- Opening a new file in a 'w' (write) mode and writing the data ---
with open(output_file_path, "w") as f_out:
    # Printing the tunneling half-lives and related values
    f_out.write("#" * 165)
    f_out.write(f"\n")
    f_out.write(
        f'{"The vibrational level indexes, energies, corresponding turning points (tp), and WKB integrals":^165}\n'
    )
    f_out.write("#" * 165)
    f_out.write(f"\n\n")
    n, k = 28, 6
    f_out.write(
        f"{'N':^{n}} | {'Vibrational energy, kJ mol-1':^{n}} | {'tp (left), sqrt(amu) * bohr':^{n}} | {'tp (right), sqrt(amu) * bohr':^{n}} | {'WKB integral, sqrt(Hartree) * amu * bohr':^{n}}\n"
    )
    f_out.write("-" * 165)
    f_out.write(f"\n")
    for row in output_values:
        c_off = 6
        for i in range(len(row) - c_off):
            if i == 0:
                f_out.write(f"{row[i]:^{n}.{0}f} | ")
            elif i == 1:
                f_out.write(f"{row[i] * Hartree_to_kJ_mol_m1:^{n}.{k}f} | ")
            elif i == len(row) - 1 - c_off:
                if type(row[i]) == str:
                    f_out.write(f"{row[i]:^{n}}\n")
                else:
                    f_out.write(f"{row[i]:^{n}.{k}f}\n")
            else:
                if type(row[i]) == str:
                    f_out.write(f"{row[i]:^{n}} | ")
                else:
                    f_out.write(f"{row[i]:^{n}.{k}f} | ")
    f_out.write(f"\n\n")

    f_out.write("#" * 242)
    f_out.write(f"\n")
    f_out.write(
        f'{"The vibrational level indexes, energies, corresponding transmission probabilities (P), reaction rate (k), and tunneling half-lives (t_1/2)":^242}\n'
    )
    f_out.write("#" * 242)
    f_out.write(f"\n\n")
    f_out.write(
        f"{'N':^{n}} | {'Vibrational energy, kJ mol-1':^{n}} | {'P':^{n}} | {'k, s-1':^{n}} | {'t_1/2, seconds':^{n}} | {'t_1/2, hours':^{n}} | {'t_1/2, days':^{n}} | {'t_1/2, years':^{n}}\n"
    )
    f_out.write("-" * 242)
    f_out.write(f"\n")
    for row in output_values:
        for i in range(len(row)):
            if i == 0:
                f_out.write(f"{row[i]:^{n}.{0}f} | ")
            elif i == 1:
                f_out.write(f"{row[i] * Hartree_to_kJ_mol_m1:^{n}.{k}e} | ")
            elif i == len(row) - 1:
                if type(row[i]) == str:
                    f_out.write(f"{row[i]:^{n}}\n")
                else:
                    f_out.write(f"{row[i]:^{n}.{k}e}\n")
            elif i >= 2 and i <= 4:
                pass
            else:
                if type(row[i]) == str:
                    f_out.write(f"{row[i]:^{n}} | ")
                else:
                    f_out.write(f"{row[i]:^{n}.{k}e} | ")
    f_out.write(f"\n\n\n\n")

    # Printing the average tunneling half-lives and related values
    f_out.write("#" * 217)
    f_out.write(f"\n")
    f_out.write(
        f'{f"The average transmission probabilities (P), reaction rate (k), and tunneling half-lives (t_1/2) for T = {T:^{4}.{2}f} K":^217}\n'
    )
    f_out.write("#" * 217)
    f_out.write(f"\n\n")
    f_out.write(
        f"{'T, K':^{n}} | {'P':^{n}} | {'k, s-1':^{n}} | {'t_1/2, seconds':^{n}} | {'t_1/2, hours':^{n}} | {'t_1/2, days':^{n}} | {'t_1/2, years':^{n}}\n"
    )
    f_out.write("-" * 217)
    f_out.write(f"\n")
    f_out.write(f"{T:^{n}.{2}f} | ")
    for i in range(len(transmission_probability_average_result)):
        if i == len(transmission_probability_average_result) - 1:
            if type(transmission_probability_average_result[i]) == str:
                f_out.write(f"{transmission_probability_average_result[i]:^{n}}")
            else:
                f_out.write(f"{transmission_probability_average_result[i]:^{n}.{k}e}")
        else:
            if type(transmission_probability_average_result[i]) == str:
                f_out.write(f"{transmission_probability_average_result[i]:^{n}} | ")
            else:
                f_out.write(
                    f"{transmission_probability_average_result[i]:^{n}.{k}e} | "
                )
    f_out.write(f"\n\n\n\n\n")

    # Printing the Arrhenius plot data (tunneling only!)
    f_out.write("#" * 90)
    f_out.write(f"\n")
    f_out.write(f'{"Arrhenius plot data (tunneling only!)":^90}\n')
    f_out.write("#" * 90)
    f_out.write(f"\n\n")
    f_out.write(f"{'T, K':^{n}} | {'1/T, K':^{n}}| {'ln(k), s-1':^{n}}\n")
    f_out.write("-" * 90)
    f_out.write(f"\n")
    for row in arrhenius_matrix:
        for i in range(len(row)):
            if i == 0:
                f_out.write(f"{row[i]:^{n}.{2}f} |")
            elif i == len(row) - 1:
                if type(row[i]) == str:
                    f_out.write(f"{row[i]:^{n}}")
                else:
                    f_out.write(f"{row[i]:^{n}.{k}e}")
            else:
                f_out.write(f"{row[i]:^{n}.{k}e} |")
        f_out.write(f"\n")
    f_out.write(f"\n\n\n\n")

    # Printing the IRC_cubic_spline_matrix
    f_out.write("#" * 101)
    f_out.write(f"\n")
    f_out.write(
        f'{"The ZPVE-corrected IRC-surface approximated by a cubic spline after offsetting and potential scaling":^90}\n'
    )
    f_out.write("#" * 101)
    f_out.write(f"\n\n")
    f_out.write(f"{'IRC, sqrt(amu) * bohr':^{n}} | {'Energy, kJ mol-1':^{n}}\n")
    f_out.write("-" * 60)
    f_out.write(f"\n")

    for row in IRC_cubic_spline:
        f_out.write(f"{row[0]:^{n}.{k}f} | {row[1] * Hartree_to_kJ_mol_m1:^{n}.{k}}\n")
    f_out.write(f"\n")

    # Printing the offset, defined as E(IRC_min) + ZPVE(IRC_min) - E0 - ZPVE0
    f_out.write("#" * 80)
    f_out.write(f"\n")
    f_out.write("Offset is defined as E(IRC_min) + ZPVE(IRC_min) - E0 - ZPVE0\n")
    f_out.write("#" * 80)
    f_out.write(f"\n\n")
    f_out.write(
        f'{f"offset = {offset * Hartree_to_kJ_mol_m1:^{8}.{k}f} kJ mol-1":^80}\n'
    )


print(f"The results were succesfully stored in a file: {output_file_name}")
