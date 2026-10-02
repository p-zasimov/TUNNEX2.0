# --- Modules ---
import numpy as np
from scipy.integrate import quad
from tunnex_2.qmt_computations.constants_and_models.constants import revelo  # type: ignore


# --- Defining the function to compute the WKB-integral ---
def compute_wkb(turning_points: dict, level: float, potential: object) -> float:
    if turning_points["regime"] != 'NORMAL':
        if turning_points["regime"] == 'OVER THE BARRIER':
            return 0.0   # The energy level is over the barrier, thus, returning zero
        return (np.inf)  # The only option here is tunneling not possible ("ENDOTHERMIC REACTION"), thus, returning "+inf"

    def _integrand(x: float) -> float:
        return np.sqrt(np.maximum(0, 2 * revelo * (potential(x) - level)))

    # Implementation of the wkb_function function implies that V(x) - E >= 0 for a given x.
    # However, for turning points, the binary search could converge to V(x) slightly smaller than E
    # (the default turning point search tolarance is tol=1e-12).
    # Thus, zero is returned for the cases where V(x) - E < 0.

    result = quad(_integrand, turning_points["left"], turning_points["right"], full_output=1)
    # The result gives an integrated value (result[0]) and a bunch of integration details.
    # We are interested only in an integrated value (result[0])
    
    return result[0]
