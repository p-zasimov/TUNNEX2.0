# --- Modules ---
import numpy as np
from scipy.interpolate import PchipInterpolator

from tunnex_2.irc_computations.ancillary_functions.interface import setup_logger # type: ignore


# --- Defining the logging function ---
logger = setup_logger()


# --- Defining the class Potential ---
class Potential:

    # --- Defining the class methods ---
    def __init__(self, E_x: np.ndarray, E_y: np.ndarray, ZPVE_x: np.ndarray, ZPVE_y: np.ndarray,
        E0: float, ZPVE0: float, potential_scaling_factor: float) -> None:
        
        # --- Defining the interpolation function, monotonic cubic spline ---
        self.E_interp = PchipInterpolator(E_x, E_y, extrapolate=False)
        self.ZPVE_interp = PchipInterpolator(ZPVE_x, ZPVE_y, extrapolate=False)

        self.offset = (E_y[0] + ZPVE_y[0]) - (E0 + ZPVE0)
        # Checking if the IRC is fully computed in the reagent's region
        self.deviation = abs(self.offset / (E0 + ZPVE0))
        # Checking if the IRC is fully computed in the reagent's region
        
        if self.deviation >= 0.02:
            logger.warning("The deviation between the reagent[IRC] and submitted reagent's energy is %.1f%%. "
                "More detailed IRC computations in the region of the reagent may be necessary.", self.deviation * 100)

        self.potential_scaling_factor = potential_scaling_factor

        # --- Defining the irc_x_min, irc_x_max, and position of the transition state (irc_x_ts)
        # for future computations of turning points ---
        self.irc_x_min = E_x[0]
        self.irc_x_max = E_x[-1]
        self.irc_x_ts = E_x[np.argmax(E_y)]

    # --- Computing the potential for a given x ---
    def __call__(self, x: float) -> float:
        
        result = (self.E_interp(x) + self.ZPVE_interp(x) -
            (self.E_interp(self.irc_x_min) + self.ZPVE_interp(self.irc_x_min)))
        
        return self.potential_scaling_factor * result


# --- Important information ---
# By the convention suggesting that the reagent is on the left hand side, and the product is on the right hand side
# Also suggesting that in the input file, IRCs increase from top to bottom, both for the electronic energy and ZPVE
# If it is not the case, please fix your input file, or own the errors and weird results

# The program also uses the input electronic energy IRC-matrix to estimate the IRC coordinate of the transition state
# (it should have a maximum of an electronic energy)