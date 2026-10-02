# --- Modules ---
from pathlib import Path
from tunnex_2.irc_computations.module_irc_computations import pipeline_irc_computations # type: ignore
from tunnex_2.irc_computations.module_eckart_potential import pipeline_eckart_potential # type: ignore
from tunnex_2.irc_computations.constants_and_dataclasses.dataclasses import MakeProjConfig # type: ignore
from tunnex_2.qmt_computations.qmt_computations_module import qmt_computations_main # type: ignore


# --- Defining the main function to compute IRC and run TUNNEX 2.0 ---
def main(configs : MakeProjConfig) -> None:

    print() # Print an empty line before the program output

    prob_aver_mode = configs.prob_aver_mode_qmt

    if configs.tunnex_input is None:

        if configs.ts_input_file is None:
            raise ValueError(f"TS input file is required for the computations. Got {configs.ts_input_file}.")

        if configs.prog_mode not in ('orca', 'gaussian'):
            raise ValueError(f"Expected 'orca' or 'gaussian'. Got {configs.prog_mode}.")

        if configs.eckart:
            if configs.react_input_file is None:
                raise ValueError(f"Reactant input file is required for Eckart potential computations. Got {configs.react_input_file}.")

            if configs.prod_input_file is None:
                raise ValueError(f"Product input file is required for Eckart potential computations. Got {configs.prod_input_file}.")

            tunnex_forward, tunnex_backward = pipeline_eckart_potential(configs)

            if configs.qmt_module:
                qmt_computations_main(tunnex_forward, prob_aver_mode)
                qmt_computations_main(tunnex_backward, prob_aver_mode)


        else:
            tunnex_forward, tunnex_backward = pipeline_irc_computations(configs)

            if configs.qmt_module:
                qmt_computations_main(tunnex_forward, prob_aver_mode)
                qmt_computations_main(tunnex_backward, prob_aver_mode)

    else:
        qmt_computations_main(configs.tunnex_input, prob_aver_mode)
  
    return None


# How to install (requires-python = ">=3.10"):
# 1. In terminal type: conda create -n tunnex_2 python=3.13                                     # Creating a conda environment
# 2. In terminal type: conda activate tunnex_2                                                  # Activating a conda environment
# 3. Go to a project folder (where pyproject is located) and in terminal type: pip install -e . # Installing TUNNEX 2.0
# 4. In terminal type: 'tunnex_2 --help'                                                        # Checking the installed files
# 5. Use TUNNEX 2.0 as 'tunnex_2 your_filename.txt'.                                            # Done!


# An alternative way to run the code for debugging etc.
#if __name__ == "__main__":

    #folder = Path("0_test_gaussian")
    #file = folder / "ch3coh.gjf"
    #program = 'gaussian'

    #configs = MakeProjConfig(ts_input_file=file, prog_mode=program)

    #main(configs)