from pathlib import Path
import sys

from tunnex_2.data_io.reader import read_input  # type: ignore
from tunnex_2.data_io.writer import write_output, build_report  # type: ignore
from tunnex_2.core.potential import Potential  # type: ignore
from tunnex_2.pipeline import run_pipeline  # type: ignore


def tunnex_2(input_path: str | Path) -> None:
    input_path = Path(input_path).resolve()

    output_path = input_path.with_name(
        input_path.stem + "_tunnex_2_out" + input_path.suffix
    )

    params, irc = read_input(input_path)

    potential = Potential(
        irc.E_x,
        irc.E_y,
        irc.ZPVE_x,
        irc.ZPVE_y,
        params.E0,
        params.ZPVE0,
        params.potential_scaling_factor,
    )

    interpolated_irc, level_results, res_temper_aver, arrhenius_rate_values = (
        run_pipeline(params, potential)
    )

    general_report = build_report(
        level_results,
        res_temper_aver,
        params.T,
        arrhenius_rate_values,
        interpolated_irc,
        potential.offset,
    )

    write_output(output_path, general_report)

    print(f"Results saved to: {output_path}")


def tunnex_2_cli():
    import sys

    if len(sys.argv) < 2:
        print("Usage: tunnex_2 <input_file>")
        return

    tunnex_2(sys.argv[1])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: tunnex_2 <input_file>")
        sys.exit(1)

    tunnex_2(sys.argv[1])

# How to install: do 'pip install -e' in the project folder. Then run the code in the programm line as 'tunnex_2 your_file_name.txt'
