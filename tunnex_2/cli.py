# --- Modules ---
import argparse
from tunnex_2.irc_computations.constants_and_dataclasses.dataclasses import MakeProjConfig # type: ignore
from tunnex_2.main import main # type: ignore

# --- Dealing with key errors ---
def _parser_error(parser, message):
    parser.error(f"{message}\n"
    "Use --help for more information.")


# --- Parsing command-line arguments ---
def cli() -> None:
    parser = argparse.ArgumentParser(prog="tunnex_2")

    parser.add_argument("files", nargs="*", help="input files for tunnex_2 computations.")

    parser.add_argument("--orca", action="store_true", help="use ORCA to compute the IRC data (default is Gaussian).")
    parser.add_argument("--no_zpve", action="store_true", help="do not calculate ZPVE (only for Gaussian).")
    parser.add_argument("--calcfc", action="store_true", help="faster but less accurate method for projected frequencies (only for Gaussian).")
    parser.add_argument("--hybrid", action="store_true", help="scale IRC curve using stationary points.")
    parser.add_argument("--eckt", action="store_true", help="use Eckart potential to compute the IRC data.")
    parser.add_argument("--comp", action="store_true", help="use already computed files instead of running calculations.")
    parser.add_argument("--qmt_only", action="store_true", help="compute only tunneling half-lives using the ready input file.")
    parser.add_argument("--no_qmt", action="store_true", help="compute only IRC data without computing tunneling half-lives.")
    parser.add_argument("--inf_sum_qmt", action="store_true", help="use infinite vibrational-level sum for temperature averaging (experimental)")
    parser.add_argument("--int_qmt", action="store_true", help="use integration for temperature averaging (experimental)")

    args = parser.parse_args()

    prog_mode_key = 'orca' if args.orca else 'gaussian'

    if args.inf_sum_qmt and args.int_qmt:
        _parser_error(parser, "--inf_sum_qmt cannot be combined with --int_qmt.")

    if args.no_qmt and (args.inf_sum_qmt or args.int_qmt):
        _parser_error(parser, "--no_qmt cannot be combined with --inf_sum_qmt or --int_qmt.")

    if args.inf_sum_qmt:
        qmt_temp_aver_mode = 'infinite_sum'
    elif args.int_qmt:
        qmt_temp_aver_mode = 'integral'
    else:
        qmt_temp_aver_mode = 'finite_sum'

    configs_data = MakeProjConfig(
    prog_mode = prog_mode_key,
    proj_freq = not args.no_zpve,
    calc_all = not args.calcfc,
    hybrid_mode = args.hybrid,
    eckart = args.eckt,
    qmt_module = not args.no_qmt,
    prob_aver_mode_qmt = qmt_temp_aver_mode,
    )

    if args.qmt_only:
        qmt_only_forbidden_flags = (
        args.orca
        or args.no_zpve
        or args.calcfc
        or args.hybrid
        or args.eckt
        or args.comp
        or args.no_qmt
        )

        if qmt_only_forbidden_flags:
            _parser_error(parser, "--qmt_only cannot be combined with IRC calculation options or --no_qmt.")

        if len(args.files) != 1:
            _parser_error(parser, f"QMT-module requires exactly 1 input file "
            f"(standard tunnex-input file). Got {len(args.files)} files.")

        configs_data.tunnex_input = args.files[0]

    else:

        if args.orca and args.no_zpve:
            _parser_error(parser, "--no_zpve cannot be used with --orca.")

        if args.orca and args.calcfc:
            _parser_error(parser, "--calcfc cannot be used with --orca.")

        if args.eckt and args.hybrid:
            _parser_error(parser, "--hybrid cannot be used with --eckt.")

        if args.eckt and args.comp:
            if len(args.files) != 4:
                _parser_error(parser, f"--eckt and --comp require exactly 4 input files "
                f"(guessed and computed transition state, reactant, and product). Got {len(args.files)} files.")

            configs_data.ts_input_file, configs_data.ts_computed_file = args.files[0], args.files[1]
            # The guessed TS input file is kept for later use.
            configs_data.react_computed_file, configs_data.prod_computed_file = args.files[2], args.files[3]

        elif args.comp:
            if len(args.files) not in (2, 3, 5):
                _parser_error(parser, f"Default mode with --comp requires 2, 3, or 5 input files "
                f"(guessed and computed transition state, computed irc, both computed reactant and product). Got {len(args.files)} files.")

            configs_data.ts_input_file, configs_data.ts_computed_file = args.files[0], args.files[1]
            # The guessed TS input file is kept for later use.
            configs_data.irc_computed_file = args.files[2] if len(args.files) > 2 else None
            configs_data.react_computed_file = args.files[3] if len(args.files) > 3 else None
            configs_data.prod_computed_file = args.files[4] if len(args.files) > 4 else None

        elif args.eckt:
            if len(args.files) != 3:
                _parser_error(parser, f"--eckt requires exactly 3 input files "
                f"(guessed transition state, reactant, and product). Got {len(args.files)} files.")

            configs_data.ts_input_file, configs_data.react_input_file, configs_data.prod_input_file = args.files

        else:
            if len(args.files) != 1:
                _parser_error(parser, f"Default mode requires exactly 1 input file "
                f"(guessed transition state). Got {len(args.files)} files.")
            configs_data.ts_input_file = args.files[0]

    main(configs=configs_data)

    return None