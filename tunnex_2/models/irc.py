from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class IRCData:
    E_x: np.ndarray  # IRC value [IRC, amu^0.5 * bohr] - E
    E_y: Any  # Electronic energy [E; Hartree] value
    ZPVE_x: Any  # IRC value [IRC, amu^0.5 * bohr] - ZPVE
    ZPVE_y: Any  # Zero-point vibrational Energy [ZPVE; Hartree] value
