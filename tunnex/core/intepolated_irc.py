import numpy as np
from tunnex_2.models.results import InterpolatedIRC  # type: ignore


# --- Saving the ZPVE-corrected IRC-curve after the interpolation with a monotonic cubic spline ---
def build_interpolated_irc(
    x_min: float, x_max: float, potential: object, grid_size: int
) -> list:
    dx = (x_max - x_min) / grid_size
    irc_matrix = [
        InterpolatedIRC(x_min + dx * i, potential(x_min + dx * i))
        for i in range(grid_size + 1)
    ]
    return irc_matrix
