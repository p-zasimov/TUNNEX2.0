from dataclasses import dataclass


@dataclass
class InputParams:
    freq: float  # The attempt frequency [freq, in cm-1]
    T: float  # Temperature [T; K]

    E0: float  # The reagent's electronic energy [E0, Hartree]
    ZPVE0: float  # The reagent's zero-point vibrational Energy [ZPVE0, Hartree]
    potential_scaling_factor: (
        float  # Potential_scaling_factor [potential_scaling_factor]
    )
    the_upper_level_to_compute: int  # The_upper_level_to_compute [N]
    T_arrhenius_min: float  # Minimum temperature [T; K] for an Arrhenius plot
    T_arrhenius_max: float  # Maximum temperature [T; K] for an Arrhenius plot
    T_arrhenius_step: float  # Step [T; K] for an Arrhenius plot
