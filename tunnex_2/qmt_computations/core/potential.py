# --- Comments ---
# By the convention suggesting that the reagent is on the left hand side, and the product is on the right hand side
# Also suggesting that in the input file, IRCs increase from top to bottom, both for the electronic energy and ZPVE
# If it is not the case, please fix your input file, or own the errors and weird results

# The program also uses the input electronic energy IRC-matrix to estimate the IRC coordinate of the transition state
# (it should have a maximum of an electronic energy)


# --- Modules ---
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar


# --- Defining the class Potential ---
class Potential:
    def __init__(self, E_x, E_y, ZPVE_x, ZPVE_y, E0, ZPVE0, potential_scaling_factor):
        # --- Defining the interpolation function, monotonic cubic spline ---
        self.E_interp = PchipInterpolator(E_x, E_y, extrapolate=False)
        self.ZPVE_interp = PchipInterpolator(ZPVE_x, ZPVE_y, extrapolate=False)

        self.offset = (E_y[0] + ZPVE_y[0]) - (E0 + ZPVE0)
        # Checking if the IRC is fully computed in the reagent's region
        self.deviation = abs(self.offset / (E0 + ZPVE0))
        # Checking if the IRC is fully computed in the reagent's region
        if self.deviation >= 0.02:
            print(
                f"Warning: the deviation between the reagent[IRC] and submitted reagent's energy is {self.deviation * 100:.{1}f} %.",
                f"More detailed IRC-computations in the region of the reagent may be necessary.\n")
        self.potential_scaling_factor = potential_scaling_factor

        # --- Defining the irc_x_min, irc_x_max, and position of the transition state (irc_x_ts)
        # for future computations of turning points ---
        self.irc_x_min = E_x[0]
        self.irc_x_max = E_x[-1]
        self.irc_x_ts = E_x[np.argmax(E_y)]

    # --- Computing the potential for a given x ---
    def __call__(self, x):
        result = (
            self.E_interp(x)
            + self.ZPVE_interp(x)
            - (self.E_interp(self.irc_x_min) + self.ZPVE_interp(self.irc_x_min))
        )
        return self.potential_scaling_factor * result


# --- Defining the function which searched for the maximum of the potential ---
def find_potential_maximum(potential: object) -> tuple[float, float]:
    result = minimize_scalar(lambda x: -potential(x), bounds=(potential.irc_x_min, potential.irc_x_max), method="bounded")

    max_energy = -result.fun
    irc_x_max_energy = result.x

    return irc_x_max_energy, max_energy