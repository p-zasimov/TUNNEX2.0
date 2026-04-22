from dataclasses import dataclass
from enum import Enum


class Regime(Enum): # The tunneling regime for the turning points computations
    NORMAL = "Normal"
    NO_TUNNELING = "Tunneling is not possible"
    OVER_BARRIER = "Over the barrier"


@dataclass # Which values tu return as a result of the turning points computations
class TurningPoints:
    turning_point_left: float | None
    turning_point_right: float | None
    tunneling_regime: Regime
