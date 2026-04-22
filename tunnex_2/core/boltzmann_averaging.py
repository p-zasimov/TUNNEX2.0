from tunnex_2.constants import k_b  # type: ignore
import numpy as np


# --- Defining the average transmission probability ---
def boltzmann_average(vib_levels: list, transmission_probs: list, T: float) -> float:
    sum_boltzmann = 0
    transmission_prob_average = 0

    E0 = vib_levels[0]

    for i, vib_level in enumerate(vib_levels):
        boltzmann_factor = np.exp(-(vib_level - E0) / (k_b * T))
        transmission_prob_average += boltzmann_factor * transmission_probs[i]
        sum_boltzmann += boltzmann_factor

    return transmission_prob_average / sum_boltzmann
