# --- Modules ---
import math
from scipy.optimize import brentq


# --- Defining the function which computes the turning points ---
def find_turning_points(level_energy: float, potential: object,
    zero_protection_val: float = 1e-12, tol: float = 1e-12) -> dict:
    
    # Defining the coordinates and energies
    irc_reagent = potential.irc_x_min
    energy_reagent = potential(irc_reagent)
    irc_product = potential.irc_x_max
    energy_product = potential(irc_product)
    irc_ts = potential.irc_x_ts
    energy_ts = potential(irc_ts)

    # Checking the negative level energies
    if level_energy < energy_reagent:
        raise ValueError(f"The energy level ({level_energy}) is below the reagent's energy ({energy_reagent}). "
            f"Please check your data.")
    # Because the reagent's energy is set to be zero after offset, and vibrational energy levels
    # are positive, this error means the problem with your data

    # We are exactly at the barrier
    if math.isclose(level_energy, energy_ts, abs_tol=tol):
        result = {"left": None, "right": None, "regime": "OVER THE BARRIER", "OVER_STATUS": "EXACT_MATCH"}
        return result
    
    # We are above the barrier
    if level_energy > energy_ts:

        # Reflecting the level energy over the barrier height, but not allowing it to be negative
        level_energy_corrected = max(0, 2 * energy_ts - level_energy)
        func_target_corr = lambda x: potential(x) - level_energy_corrected
        quasi_turning_point_left = brentq(func_target_corr, irc_reagent, irc_ts)

        # For a quasiendothermic reaction, computing the maximum possible integral
        # and then scale it linearly if level_energy > energy_ts + energy_product
        if level_energy_corrected < energy_product:
            func_target_prod = lambda x: potential(x) - energy_product
            quasi_turning_point_left = brentq(func_target_prod, irc_reagent, irc_ts)
            quasi_turning_point_right = brentq(func_target_prod, irc_ts, irc_product)
            
            # Zero protection procedure
            energy_sum = energy_ts + energy_product
            if math.isclose(energy_sum, 0.0, abs_tol=zero_protection_val):
                energy_sum = zero_protection_val
            
            # Estimating the scaling factor
            scale_factor = max(1, level_energy / (energy_sum))
            result = {"left": None, "right": None, "regime": "OVER THE BARRIER",
                      "OVER_STATUS": "QUASI_ENDOTHERMIC_REACTION",
                      "quasi_left": quasi_turning_point_left, "quasi_right": quasi_turning_point_right,
                      "QUASI_LEVEL": energy_product, "SCALE_FACTOR": scale_factor}
            return result
        
        quasi_turning_point_left = brentq(func_target_corr, irc_reagent, irc_ts)
        quasi_turning_point_right = brentq(func_target_corr, irc_ts, irc_product)

        # Zero protection procedure
        energy_sum = 2 * energy_ts
        if math.isclose(energy_sum, 0.0, abs_tol=zero_protection_val):
            energy_sum = zero_protection_val
        scale_factor = max(1, level_energy / (energy_sum))

        result = {"left": None, "right": None, "regime": "OVER THE BARRIER", "OVER_STATUS": "QUASI_NORMAL",
                  "quasi_left": quasi_turning_point_left, "quasi_right": quasi_turning_point_right,
                  "QUASI_LEVEL": level_energy_corrected, "SCALE_FACTOR": scale_factor}
        return result
    
    # The vibrational level's energy is bigger than the energy of the transition state in this case.
    # The resulting integral should be zero or negative

    func_target = lambda x: potential(x) - level_energy

    turning_point_left = brentq(func_target, irc_reagent, irc_ts)

    if level_energy < energy_product:
        result = {"left": turning_point_left, "right": None, "regime": "ENDOTHERMIC REACTION"}
        return result
    # It means that the energy is not smaller than the reagent's energy, but smaller than the product's energy.
    # From this level, tunneling is not possible, but the vibrational excitation can make it possible

    # In other cases, we should have two turning points
    turning_point_right = brentq(func_target, irc_ts, irc_product)
    result = {"left": turning_point_left, "right": turning_point_right, "regime": "NORMAL"}
    
    return result