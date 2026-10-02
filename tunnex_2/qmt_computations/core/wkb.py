# --- Modules ---
import numpy as np
from scipy.integrate import quad

from tunnex_2.constants_and_dataclasses.constants_and_settings import REVELO  # type: ignore


# --- Defining the function for the integration ---
def _integrand(x: float, potential: object, level_value: float) -> float:
    result = np.sqrt(np.maximum(0, 2 * REVELO * (potential(x) - level_value)))
    return result
    # Implementation of the wkb_function function implies that V(x) - E >= 0 for a given x.
    # However, for turning points, the binary search could converge to V(x) slightly smaller than E
    # (the default turning point search tolerance is tol=1e-12).
    # Thus, negative values are clamped to zero for the cases where V(x) - E < 0.


# --- Defining the function to compute the WKB-integral ---
def compute_wkb(turning_points: dict, level: float, potential: object, zero_tol: float = 1e-12) -> float:

    allowed_regimes = ("NORMAL", "ENDOTHERMIC REACTION", "OVER THE BARRIER")
    regime = turning_points["regime"]

    if regime not in allowed_regimes:
        raise ValueError(f"Expected tunneling regimes: {allowed_regimes}. Got {regime}.")

    if regime == "OVER THE BARRIER":

        allowed_over_the_barrier = ("QUASI_NORMAL", "QUASI_ENDOTHERMIC_REACTION", "EXACT_MATCH")
        over_status = turning_points["OVER_STATUS"]

        if over_status not in allowed_over_the_barrier:
            raise ValueError(f"Expected over the barrier tunneling regimes: {allowed_over_the_barrier}. "
                             f"Got {over_status}.")
        
        if over_status == "EXACT_MATCH":
            return 0.0   # The energy level is exactly the barrier, thus, returning zero

        integrand_func_over = lambda x: _integrand(x, potential, turning_points["QUASI_LEVEL"])
        result = quad(integrand_func_over, turning_points["quasi_left"], turning_points["quasi_right"], full_output=1)
        
        if np.isclose(result[0], 0.0, atol=zero_tol):
            ans = -turning_points["SCALE_FACTOR"] * zero_tol # it might be useful for very flat barriers
        else:
            ans = -turning_points["SCALE_FACTOR"] * result[0]
        return ans # The energy level is over the barrier, thus, returning the negative value
    
    elif regime == "ENDOTHERMIC REACTION":
        return np.inf  # regime == "ENDOTHERMIC REACTION", thus, returning "+inf"

    else: # regime == "NORMAL"
        integrand_func_normal = lambda x: _integrand(x, potential, level)
        result = quad(integrand_func_normal, turning_points["left"], turning_points["right"], full_output=1)
        ans = result[0]
        # The result gives an integrated value (result[0]) and a bunch of integration details. Storing only the former value
    return ans
