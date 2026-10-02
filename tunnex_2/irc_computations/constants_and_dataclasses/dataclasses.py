# --- Modules ---
from dataclasses import dataclass
from pathlib import Path


# --- Dataclasses ---
@dataclass # It stores the initial configurations for irc_computations
class MakeProjConfig:
    # Calculation settings
    prog_mode : str = 'gaussian'
    proj_freq: bool = True
    calc_all: bool = True
    hybrid_mode: bool = False
    eckart : bool = False
    qmt_module : bool = True
    prob_aver_mode_qmt : str = 'finite_sum'

    # Input files
    ts_input_file : str | Path | None = None
    react_input_file : str | Path | None = None
    prod_input_file : str | Path | None = None

    # Already computed files
    ts_computed_file : str | Path | None = None
    irc_computed_file : str | Path | None = None
    react_computed_file : str | Path | None = None
    prod_computed_file : str | Path | None = None

    # Tunnex input file
    tunnex_input : str | Path | None = None


@dataclass # It stores the IRC data
class IRCData:
    electronic_energies: list[float]
    zpve_energies_forward: list[float] | None
    zpve_energies_reverse: list[float] | None


@dataclass # It stores the electronic energies and ZPVEs for the transition state, reactant, and product
class EnergyPointsData:
    transition_state_el_energy: tuple[float, float]
    transition_state_zpve: tuple[float, float]
    reactant_el_energy: tuple[float, float]
    reactant_zpve: tuple[float, float]
    product_el_energy: tuple[float, float]
    product_zpve: tuple[float, float]


@dataclass # It stores the attempt frequencies, correlations, and numbers of vibrational levels
class AttemptFreqLevelNum:
    att_freq_react: float
    att_freq_react_corr: list[list[float, float]]
    att_freq_prod: float
    att_freq_prod_corr: list[list[float, float]]
    num_levels_react: int
    num_levels_prod: int


@dataclass
class TunnexInputSettings: # It stores the important import parameters for the TUNNEX 2.0 computations
    freq: float
    E0: float
    ZPVE0: float
    number_of_levels: int = 10
    T: float = 10
    potential_scaling_factor: float = 1.0
    T_arrhenius_min: float = 5
    T_arrhenius_max: float = 1000
    T_step: float = 5