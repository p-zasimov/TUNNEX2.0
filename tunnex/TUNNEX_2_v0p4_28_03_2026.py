# (1) *** DEFINING CONSTANTS AND READING INPUT-FILE  ***

# Import os and math
import os
import math
# from .constants import *

# --- Defining constants ---
revelo = 1822.8886259874  # A unified atomic mass unit (dalton) in electron mass (atomic units)
k_b = 3.166811563e-6  # Boltzmann constant in Hartree K-1
light = 29979245800  # Speed of light in cm s-1
cm_m1_to_Hartree = 1 / 219474.63136320  # cm-1 to Hartree
Hartree_to_kJ_mol_m1 = 2625.49963947  # Hartree to kJ mol-1

# Defining the folder path and file names
folder_path = r"D:\python_projects\TUNNEX_2\\"  # Folder path

# Checking if the folder_path exists, if not, make it
# if not os.path.exists(folder_path):
#    os.makedirs(folder_path)

input_file_name = "WKB_100927b_cypro13CH_AE-CCSD_T_pVTZ_B3PW91_cc-pVTZ_tight_input.txt"  # Input-file name
output_file_name = "WKB_100927b_cypro13CH_AE-CCSD_T_pVTZ_B3PW91_cc-pVTZ_tight_output.txt"  # Output-file name

input_file_path = os.path.join(folder_path, input_file_name)
output_file_path = os.path.join(folder_path, output_file_name)

# By the convention suggesting that the reagent is on the left side, and the product is on the right side
# Also suggesting that in the input file, IRCs increase from top to bottom, both for the electronic energy and ZPVE
# If it is not the case, please fix your input file, or own the errors and weird results

# The program also uses the input electronic energy IRC-matrix to estimate the IRC coordinate of the transition state
# (it should have a maximum of an electronic energy)


# --- Open an input-file and read all lines ---

with open(input_file_path, "r") as f_in:
    lines = f_in.readlines()


# --- Defining the function to read the input parameters ---
def read_parameters(line_number, column_number):
    line_n = lines[line_number].strip()
    parts_n = line_n.split(";")
    return float(parts_n[column_number].split("=")[1])


# --- Reading the first line (the attempt frequency [freq, in cm-1]
# and temperature [T; K]) ---
freq, T = read_parameters(0, 0), read_parameters(0, 1)
if freq <= 0:
    raise ValueError("Frequency should be > 0")
if T <= 0:
    raise ValueError("Temperature should be > 0")

# --- Reading the second line (the reagent's electronic energy [E0, Hartree],
# zero-point vibrational Energy [ZPVE0, Hartree],potential_scaling_factor [potential_scaling_factor])
# and the_upper_level_to_compute [N] ---
E0, ZPVE0, potential_scaling_factor, the_upper_level_to_compute = (
    read_parameters(1, 0),
    read_parameters(1, 1),
    read_parameters(1, 2),
    int(read_parameters(1, 3)),
)
if potential_scaling_factor <= 0:
    raise ValueError("Potential scaling factor should be > 0")
if the_upper_level_to_compute < 0:
    raise ValueError("N should be >= 0")

# Potential_scaling_factor means a scaling factor for the final ZPVE-corrected IRC curve after offsetting. The default value is 1.0
# If the_upper_level_to_compute = 3, for example, then the code will compute the tunneling probabilities for the
# vibrational energy levels of v = 0, 1, 2, 3

# --- Reading the third line (minimum and maximum temperatures and a step [T; K] for an Arrhenius plot)
T_arrhenius_min, T_arrhenius_max, T_step = (
    read_parameters(2, 0),
    read_parameters(2, 1),
    read_parameters(2, 2),
)

if T_arrhenius_min <= 0 or T_arrhenius_max <= 0 or T_step <= 0:
    raise ValueError(
        "The border temperatures and step for an Arrhenius graph should be > 0"
    )


# --- Defining the function to read the IRC_matrices [Nx2], [Kx2] ---
def read_IRC(START, END):  # Searching where the matrix starts and ends
    start_index = None
    end_index = None
    for i, line in enumerate(lines):
        if line.strip() == START:
            start_index = i + 1
            break
    for i, line in enumerate(lines):
        if line.strip() == END:
            end_index = i - 1
            break

    IRC_matrix = []  # Filling the matrix

    if start_index is not None:
        comp_irc = float("-inf")
        for line in lines[start_index:end_index]:
            line = line.strip()
            if line:  # Leaving empty lines
                values = [float(x.strip()) for x in line.split(";")]
                if (
                    values[0] > comp_irc
                ):  # Checking if set increases monotonically for the IRC-values
                    IRC_matrix.append(values)
                    comp_irc = values[0]
                else:
                    raise ValueError("The set is not monotonically IRC-increasing")

    return IRC_matrix


# --- Reading the IRC_E [IRC, amu^0.5 * bohr]; electronic energy [E; Hartree] values
# and putting them in the IRC_E_matrix [Nx2] ---
IRC_E_matrix = read_IRC("IRC; E", "END; E")

# --- Reading the IRC_ZPVE [IRC, amu^0.5 * bohr]; zero-point vibrational Energy [ZPVE; Hartree] values
# and putting them in the IRC_ZPVE_matrix [Kx2] ---
IRC_ZPVE_matrix = read_IRC("IRC; ZPVE", "END; ZPVE")


# (2) *** APPROXIMATION OF THE IRC-CURVES WITH MONOTONIC CUBIC SPLINES  ***


# --- Defining the function to approximate the data with a monotone cubic interpolation ---
def cubic_spline(matrix):
    if len(matrix) < 2:
        raise ValueError("The number of points is not enough to plot a spline")

    # Compute consecutive differences and slopes
    dxs = [matrix[i + 1][0] - matrix[i][0] for i in range(len(matrix) - 1)]
    dys = [matrix[i + 1][1] - matrix[i][1] for i in range(len(matrix) - 1)]
    ms = [dy / dx for dx, dy in zip(dxs, dys)]

    # Compute first-degree coefficients (c1s)
    c1s = [ms[0]]
    for i in range(len(ms) - 1):
        m, m_next = ms[i], ms[i + 1]
        if m * m_next <= 0:
            c1s.append(0.0)
        else:
            dx, dx_next = dxs[i], dxs[i + 1]
            common = dx + dx_next
            c1s.append(
                3 * common / ((common + dx_next) / m + (common + dx) / m_next)
            )  # It uses a weighted harmonic mean to prevent overshooting and make the curve smoother.
            # It also used cross-weighting, implying that the weight of the derivative
            # computed at the smaller interval has a bigger final weight as the more local, and, thus, more trustworthy value.
            # A scheme with the (common + dx) weight is also better to be used as a simple reciprocal 1/dx as a weight,
            # because in this case, the minimum weight for a single derivative cannot be smaller than 1/3. It makes the function
            # more stable by preventing overshooting if some very local derivative has a big value.
            # It also automatically limits [(c1s[i]/ms[i])^2 + (c1s[i+1]/ms[i])^2] <= 9 which is required for a strict monotonicity.
    c1s.append(ms[-1])

    # Compute second- and third-degree coefficients (c2s, c3s)
    c2s, c3s = [], []
    for i in range(len(c1s) - 1):
        c1, m = c1s[i], ms[i]
        inv_dx = 1 / dxs[i]
        common = c1 + c1s[i + 1] - 2 * m
        c2s.append(
            (m - c1 - common) * inv_dx
        )  # The c2s and c3s coefficients can be derived from a system of equations
        # which set boundary conditions for a monotonic cubic spline P(x) (4 equations): P(x[i]) = y[i], P(x[i+1]) = y[i+1],
        # P'(x[i]) = c1s[i], and P'(x[i+1]) = c1s[i+1]
        c3s.append(common * inv_dx * inv_dx)

    for i in range(len(matrix)):
        if i == len(matrix) - 1:
            matrix[i].append(c1s[i])
            for _ in range(2):
                matrix[i].append(
                    0
                )  # Putting zeros instead of coefficients to fill the matrix
        else:
            matrix[i].append(c1s[i])
            matrix[i].append(c2s[i])
            matrix[i].append(c3s[i])
    return matrix


# --- Defining the function to search for the lower border index (int) of the IRC-interval, where x is located ---
def binary_search_idx(x, matrix):
    # Clamp x to range
    if x < matrix[0][0] or x > matrix[len(matrix) - 1][0]:
        raise ValueError("The value is beyond the IRC-interval")
    elif x == matrix[0]:
        i = 0
    elif x >= matrix[len(matrix) - 1][0]:
        i = len(matrix) - 2
    else:  # Binary search for interval
        low, high = 0, len(matrix) - 2
        while low <= high:
            mid = (low + high) // 2
            if matrix[mid][0] < x:
                low = mid + 1
            else:
                high = mid - 1
        i = max(0, high)
    return i


# --- Defining the function which can find the interpolated y-value for a given x ---
def cubic_spline_value(x, matrix):
    binary_search_x = binary_search_idx(x, matrix)
    dx = x - matrix[binary_search_x][0]
    answer = matrix[binary_search_x][1] + dx * (
        matrix[binary_search_x][2]
        + dx * (matrix[binary_search_x][3] + dx * matrix[binary_search_x][4])
    )
    # d_answer = matrix[binary_search_x][2] + dx * (2 * matrix[binary_search_x][3] + dx * 3 * matrix[binary_search_x][4]) # It yield the first derivative
    # (maybe it will be needed in the future)
    # int_answer = dx * matrix[binary_search_x][1] + dx ** 2 * (
    #    matrix[binary_search_x][2] / 2
    #    + dx * (matrix[binary_search_x][3] / 3 + dx * matrix[binary_search_x][4] / 4)) # It yield the first anti-derivative (ignoring the constant).
    # (maybe it will be needed in the future)
    return answer


# --- Computing the spline coefficients ---
E_cubic_spline = cubic_spline(IRC_E_matrix)
ZPVE_cubic_spline = cubic_spline(IRC_ZPVE_matrix)


# (3) *** DEFINING THE INTERPOLATION-FUNCTION  ***


# --- Computing an offset ---
if abs(IRC_E_matrix[0][0] - IRC_ZPVE_matrix[0][0]) < 0.000001:
    offset = (IRC_E_matrix[0][1] + IRC_ZPVE_matrix[0][1]) - (E0 + ZPVE0)
    deviation = abs(
        offset / (E0 + ZPVE0)
    )  # Checks if the IRC is fully computed in the reagent's region
    if deviation >= 0.02:
        print(
            f"Warning: the deviation between the reagent[IRC] and submitted reagent's energy is {deviation * 100:.{1}f} %. More detailed IRC-computations in the region of the reagent may be necessary"
        )
else:
    raise ValueError("IRC and ZPVE matrices start with a different IRC-value")


# --- Defining the final IRC-potential-curve, which was ZPVE-corrected and interpolated with a natural cubic spline ---
def final_potential(
    x,
    E_cubic_spline_matrix=E_cubic_spline,
    ZPVE_cubic_spline_matrix=ZPVE_cubic_spline,
    E_reagent=E0,
    ZPVE_reagent=ZPVE0,
    offset_y=offset,
    potential_scaling_factor_y=potential_scaling_factor,
):
    final_potential_result = potential_scaling_factor_y * (
        cubic_spline_value(x, E_cubic_spline_matrix)
        + cubic_spline_value(x, ZPVE_cubic_spline_matrix)
        - (E_reagent + ZPVE_reagent)
        - offset_y
    )
    return final_potential_result


# Creating short-forms of final_potential, final_potential_first_derivative, and final_potential_integral
# in order to avoid mentioning E_cubic_spline_matrix, ..., potential_scaling_factor every time

short_final_potential = lambda a: final_potential(
    a, E_cubic_spline, ZPVE_cubic_spline, E0, ZPVE0, offset, potential_scaling_factor
)


# --- Saving the ZPVE-corrected IRC-curve after the interpolation with a cubic spline ---
IRC_cubic_spline = []

for elem in IRC_E_matrix:
    row = []
    row.append(elem[0])
    row.append(short_final_potential(elem[0]))
    IRC_cubic_spline.append(row)


# (4) *** COMPUTING THE VIBRATIONAL ENERGY LEVELS AND TURNING POINTS  ***


# --- Computing vibrational energy levels ---
vib_level_energies = [
    (i + 0.5) * freq * cm_m1_to_Hartree
    for i in range(int(the_upper_level_to_compute) + 1)
]


# --- Defining the function to search for the turning point (float) in a given interval ---
def turning_point_binary_search(
    y_target,
    E_cubic_spline_matrix=E_cubic_spline,
    ZPVE_cubic_spline_matrix=ZPVE_cubic_spline,  # ZPVE_cubic_spline_matrix does not do anything here, because the
    # short_final_potential is used instead of the regular final_potential function
    left_or_right="left",
    tol=1e-8,
    max_iter=50,
):
    if (
        left_or_right == "left"
    ):  # The (reagent, transition state) interval for a left turning point
        left = E_cubic_spline_matrix[0][0]
        right = E_cubic_spline_matrix[
            max(
                range(len(E_cubic_spline_matrix) - 1),
                key=lambda i: E_cubic_spline_matrix[i][1],
            )
        ][0]
    else:  # The (transition state, product) interval for a right turning point
        left = E_cubic_spline_matrix[
            max(
                range(len(E_cubic_spline_matrix) - 1),
                key=lambda i: E_cubic_spline_matrix[i][1],
            )
        ][0]
        right = E_cubic_spline_matrix[len(E_cubic_spline_matrix) - 1][0]

    func_target_left = short_final_potential(left) - y_target
    func_target_right = short_final_potential(right) - y_target

    # Checks that the solution actually exists
    if func_target_left * func_target_right > 0:
        raise ValueError("The solution is not guaranteed")

    for _ in range(max_iter):
        mid = (left + right) / 2
        func_target_mid = short_final_potential(mid) - y_target

        if abs(func_target_mid) < tol:
            return mid

        # Choosing the half of the interval
        if func_target_left * func_target_mid < 0:
            right = mid
            func_target_right = func_target_mid
        else:
            left = mid
            func_target_left = func_target_mid

    # If not converged for the max_iter number of iterations, return the best approximation
    return (left + right) / 2


# --- Defining the function which computes the turning points ---
def turning_points(
    y_target,
    E_cubic_spline_matrix=E_cubic_spline,
    ZPVE_cubic_spline_matrix=ZPVE_cubic_spline,
    tol=1e-8,
    max_iter=50,
):

    IRC_max_ts = E_cubic_spline_matrix[
        max(
            range(len(E_cubic_spline_matrix) - 1),
            key=lambda i: E_cubic_spline_matrix[i][1],
        )
    ][0]

    if y_target < short_final_potential(IRC_E_matrix[0][0]):
        raise ValueError(
            "The energy level is below the reagent's energy. Please check your data"
        )
    # Because the reagent's energy is set to be zero after offset, and vibrational energy levels
    # are positive, this error means the problem with your data
    elif y_target < short_final_potential(IRC_E_matrix[len(IRC_E_matrix) - 1][0]):
        return (
            "Tunneling is not possible",
            "One turning point",
        )
    # It means that the energy is not smaller than the reagent's
    # energy, but smaller than the product's energy. From this level, tunneling is not possible, but the vibrational excitation can
    # make it possible
    elif y_target >= short_final_potential(IRC_max_ts):
        return (
            "Over the barrier",
            "No turning points",
        )
    # The vibrational level's energy is bigger than the energy of the transition state in this case.
    # The resulting integral should be zero
    # In other cases, we should have two turning points
    else:
        turning_point_left = turning_point_binary_search(
            y_target,
            E_cubic_spline_matrix,
            ZPVE_cubic_spline_matrix,
            "left",
            tol,
            max_iter,
        )
        turning_point_right = turning_point_binary_search(
            y_target,
            E_cubic_spline_matrix,
            ZPVE_cubic_spline_matrix,
            "right",
            tol,
            max_iter,
        )
        return turning_point_left, turning_point_right


# --- Taking the computed vibrational energy level indexes and levels and appending them with turning points ---
# Filling the final output-matrix
output_values = []

for level in vib_level_energies:
    row = []
    row.append(vib_level_energies.index(level))
    row.append(level)
    for turning_point in turning_points(
        level,
        E_cubic_spline,
        ZPVE_cubic_spline,
    ):
        row.append(turning_point)
    output_values.append(row)


# (5) *** COMPUTING THE WKB-INTEGRALS  ***


# --- Defining the wkb_function(x) as sqrt(2 * m * (V(x) - E)) to compute an integral ---
def wkb_function(
    x,
    level,
    E_cubic_spline_matrix=E_cubic_spline,
    ZPVE_cubic_spline_matrix=ZPVE_cubic_spline,
):
    wkb_function_square = (
        2 * revelo * (short_final_potential(x) - level)
    )  # wkb_function_square(x) = 2 * m * (V(x) - E)
    if wkb_function_square <= 0:
        answer = 0
    else:
        answer = wkb_function_square**0.5
    return answer


# Implementation of this function implies that V(x) - E >= 0 for a given x.
# However, for turning points, the binary search could converge to V(x) slightly smaller than E (< tol, the default tol=1e-8).
# Thus, zero is returned for the cases where wkb_function_square <= 0


# --- Defining the adaptive_simpson quadrature to compute an integral numerically using an adaptive parabolic or cubic approximation ---
# Improving the tol value (tol=1e-6 is the default) barely affects the integrals,
# but significantly increases the program performance time
def adaptive_simpson(
    turning_point_left,
    turning_point_right,
    level,
    func,
    mode_v="cube",
    tol=1e-6,
    max_recursion=50,
):
    # Approximating the desired integral by a parabolic integral
    def simpson_quad(turning_point_left, turning_point_right, level, func):
        turning_point_mid = (turning_point_left + turning_point_right) / 2
        func_left = func(turning_point_left, level)
        func_right = func(turning_point_right, level)
        func_mid = func(turning_point_mid, level)
        return (
            (turning_point_right - turning_point_left)
            / 6
            * (func_left + 4 * func_mid + func_right)
        )

    # Approximating the desired integral by a cubic integral. If "adaptive_simpson" was chosen, "simpson_cube" is default
    def simpson_cube(turning_point_left, turning_point_right, level, func):
        turning_point_1_3 = 2 * turning_point_left / 3 + turning_point_right / 3
        turning_point_2_3 = turning_point_left / 3 + 2 * turning_point_right / 3
        func_left = func(turning_point_left, level)
        func_right = func(turning_point_right, level)
        func_1_3 = func(turning_point_1_3, level)
        func_2_3 = func(turning_point_2_3, level)
        return (
            (turning_point_right - turning_point_left)
            / 8
            * (func_left + 3 * func_1_3 + 3 * func_2_3 + func_right)
        )

    if mode_v == "quad":
        int_mode_f = simpson_quad
    elif mode_v == "cube":
        int_mode_f = simpson_cube
    else:
        raise ValueError("The integration method is not defined")

    # A recursive implementation of adaptive Simpson's rule
    def recursive(
        turning_point_left,
        turning_point_right,
        level,
        func,
        int_mode,
        tol,
        depth,
    ):
        turning_point_mid = (turning_point_left + turning_point_right) / 2
        S = int_mode_f(
            turning_point_left,
            turning_point_right,
            level,
            func,
        )
        S_left = int_mode_f(
            turning_point_left,
            turning_point_mid,
            level,
            func,
        )
        S_right = int_mode_f(
            turning_point_mid,
            turning_point_right,
            level,
            func,
        )
        if depth <= 0:
            return S_left + S_right + (S_left + S_right - S) / 15  # Maximum depth
        if abs(S_left + S_right - S) < 15 * tol:
            return S_left + S_right + (S_left + S_right - S) / 15
        else:
            # Split into two parts recursively
            return recursive(
                turning_point_left,
                turning_point_mid,
                level,
                func,
                int_mode,
                tol / 2,
                depth - 1,
            ) + recursive(
                turning_point_mid,
                turning_point_right,
                level,
                func,
                int_mode,
                tol / 2,
                depth - 1,
            )

    return recursive(
        turning_point_left,
        turning_point_right,
        level,
        func,
        int_mode_f,
        tol,
        max_recursion,
    )


# --- Defining the adaptive_gauss_kronrod quadrature to compute an integral numerically using an adaptive parabolic approximation ---
# Improving the tol value (tol=1e-6 is the default) barely affects the integrals,
# but significantly increases the program performance time
def adaptive_gauss_kronrod(  # Here "mode_v" does not do anything. It is needed for compatibility with "adaptive_simpson"
    turning_point_left,
    turning_point_right,
    level,
    func,
    mode_v="gauss_kronrod",
    tol=1e-6,
    max_recursion=50,
):
    # Gauss's nodes and weights
    gauss_nodes = [
        0.000000000000000,
        0.405845151377397,
        -0.405845151377397,
        0.741531185599394,
        -0.741531185599394,
        0.949107912342759,
        -0.949107912342759,
    ]

    gauss_weights = [
        0.417959183673469,
        0.381830050505119,
        0.381830050505119,
        0.279705391489277,
        0.279705391489277,
        0.129484966168870,
        0.129484966168870,
    ]

    # Kronrod's nodes and weights
    kronrod_nodes = [
        0.000000000000000,
        0.207784955007898,
        -0.207784955007898,
        0.405845151377397,
        -0.405845151377397,
        0.586087235467691,
        -0.586087235467691,
        0.741531185599394,
        -0.741531185599394,
        0.864864423359769,
        -0.864864423359769,
        0.949107912342759,
        -0.949107912342759,
        0.991455371120813,
        -0.991455371120813,
    ]

    kronrod_weights = [
        0.209482141084728,
        0.204432940075298,
        0.204432940075298,
        0.190350578064785,
        0.190350578064785,
        0.169004726639267,
        0.169004726639267,
        0.140653259715525,
        0.140653259715525,
        0.104790010322250,
        0.104790010322250,
        0.063092092629979,
        0.063092092629979,
        0.022935322010529,
        0.022935322010529,
    ]

    # Defining the numerical integration using the Gauss-Kronrod method
    def gauss_kronrod_integral(
        turning_point_left, turning_point_right, level, func, n=7
    ):
        # Choosing the integration method
        if n == 7:
            nodes = gauss_nodes
            weights = gauss_weights
        elif n == 15:
            nodes = kronrod_nodes
            weights = kronrod_weights
        else:
            raise ValueError(
                "Currently, only 7 or 15-point Gauss-Kronrod rules are supported"
            )

        # Scaling nodes and weights for an arbitrary interval [a, b] and computing the integral
        integral = 0.0
        for i in range(len(nodes)):
            x = (
                0.5 * (nodes[i] + 1) * (turning_point_right - turning_point_left)
                + turning_point_left
            )
            w = 0.5 * weights[i] * (turning_point_right - turning_point_left)
            integral += w * func(x, level)

        return integral

    # A recursive implementation of adaptive Gauss-Kronrod
    def recursive(
        turning_point_left,
        turning_point_right,
        level,
        func,
        tol,
        depth,
    ):
        turning_point_mid = (turning_point_left + turning_point_right) / 2
        S_gauss = gauss_kronrod_integral(
            turning_point_left, turning_point_right, level, func, 7
        )
        S_kronrod = gauss_kronrod_integral(
            turning_point_left, turning_point_right, level, func, 15
        )
        if depth <= 0:
            return 0.5 * S_kronrod + 0.5 * S_gauss  # Maximum depth
        if abs(S_gauss - S_kronrod) < tol:
            return 0.5 * S_kronrod + 0.5 * S_gauss
        else:
            # Split into two parts recursively
            return recursive(
                turning_point_left,
                turning_point_mid,
                level,
                func,
                tol / 2,
                depth - 1,
            ) + recursive(
                turning_point_mid,
                turning_point_right,
                level,
                func,
                tol / 2,
                depth - 1,
            )

    return recursive(
        turning_point_left,
        turning_point_right,
        level,
        func,
        tol,
        max_recursion,
    )


# --- Dealing with the cases, when tunneling is either not possible, or the energy level of a vibrational level is over the barrier ---
def wkb_integral(
    turning_point_left,
    turning_point_right,
    level,
    func,
    method="adaptive_gauss_kronrod",
    mode="cube",
    tol=1e-6,
    max_recursion=50,
):
    if turning_point_left == "Tunneling is not possible":
        return "+inf"
    elif turning_point_left == "Over the barrier":
        return 0
    else:
        answer = method(
            turning_point_left,
            turning_point_right,
            level,
            func,
            mode,
            tol,
            max_recursion,
        )
    return answer


# --- Computing the wkb-integral and adding them to the matrix ---
method_value = adaptive_gauss_kronrod  # It can be either "adaptive_gauss_kronrod" or "adaptive_simpson"
# The "adaptive_gauss_kronrod" method is the default

# If "adaptive_simpson" was chosen:
mode_value = "cube"  # Using either a parabolic "quad" or cubic "cube" integral for "adaptive_simpson". The "cube" is the default

for row in output_values:
    level, turning_point_left, turning_point_right = row[1], row[2], row[3]
    wkb_integral_value = wkb_integral(
        turning_point_left,
        turning_point_right,
        level,
        wkb_function,
        method_value,
        mode_value,
    )
    row.append(wkb_integral_value)


# (6) *** COMPUTING THE TRANSMISSON PROBABILITIES AND HALF-LIVES  ***


# --- Defining the function to compute the transmission probabilities and half-lives ---
def transmission_prob_and_half(wkb_integral_value, ts_freq, light_speed):
    if wkb_integral_value == "+inf":
        return 0, 0, "+inf", "+inf", "+inf", "+inf"
    else:
        t_prob = 1 / (
            1 + math.exp(1) ** (2 * wkb_integral_value)
        )  # A transmission probability
        reaction_rate = ts_freq * light_speed * t_prob  # A reaction rate in seconds-1
        half_s = math.log(2) / (reaction_rate)  # A half-life in seconds
        half_h = half_s / 3600  # A half-life in hours
        half_d = half_s / 86400  # A half-life in days
        half_y = half_s / 31536000  # A half-life in years
        return t_prob, reaction_rate, half_s, half_h, half_d, half_y


# --- Adding the computed value to the resulting matrix ---
for row in output_values:
    transmission_probability = transmission_prob_and_half(row[4], freq, light)
    for elem in transmission_probability:
        row.append(elem)


# (7) *** ANALYZING THE BOLTZMANN DISTRIBUTION FOR A GIVEN TEMPERATURE AND AVERAGING THE TRANSMISSON PROBABILITY ***


# --- Defining the average transmission probability ---
def transmission_prob_boltzmann(
    matrix, ts_freq, temperature, boltzmann_constant, light_speed
):
    sum_boltmann = 0
    for row in matrix:
        sum_boltmann += math.exp(1) ** (
            -(row[1] - matrix[0][1]) / (boltzmann_constant * temperature)
        )
    transmission_probability_average = 0
    for row in matrix:
        boltmann_factor = (
            math.exp(1)
            ** (-(row[1] - matrix[0][1]) / (boltzmann_constant * temperature))
            / sum_boltmann
        )
        transmission_probability_average += (
            boltmann_factor
            * transmission_prob_and_half(row[4], ts_freq, light_speed)[0]
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
        half_s_average = math.log(2) / (reaction_rate_average)  # A half-life in seconds
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


# (8) *** COMPUTING THE DATA FOR AN ARRHENIUS PLOT ***


# --- Computing the average reaction rate ---
def reaction_rate_arr(matrix, ts_freq, temperature, boltzmann_constant, light_speed):
    sum_boltmann = 0
    for row in matrix:
        sum_boltmann += math.exp(1) ** (
            -(row[1] - matrix[0][1]) / (boltzmann_constant * temperature)
        )
    transmission_probability_average = 0
    for row in matrix:
        boltmann_factor = (
            math.exp(1)
            ** (-(row[1] - matrix[0][1]) / (boltzmann_constant * temperature))
            / sum_boltmann
        )
        transmission_probability_average += (
            boltmann_factor
            * transmission_prob_and_half(row[4], ts_freq, light_speed)[0]
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
        arrhenius_matrix_row = []
        arrhenius_matrix_row.append(arrhenius_T)
        arrhenius_matrix_row.append(1 / arrhenius_T)
        arrhenius_result = reaction_rate_arr(
            output_values, freq, arrhenius_T, k_b, light
        )
        if arrhenius_result > 0:
            arrhenius_matrix_row.append(
                math.log(
                    reaction_rate_arr(output_values, freq, arrhenius_T, k_b, light)
                )
            )
        else:
            arrhenius_matrix_row.append("-inf")
        arrhenius_matrix.append(arrhenius_matrix_row)


# (9) *** SAVING THE RESULTS IN A NEW FILE ***


# Opening a new file in a 'w' (write) mode and writing the data
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
