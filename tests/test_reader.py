# --- Test: test_read_input. Unit tests for .\qmt_computations\read_input.py ---
# Run with: pytest test_read_input.py -v ---


# --- Modules ---
import pytest # type: ignore
from unittest.mock import patch
from contextlib import ExitStack

import numpy as np
from pathlib import Path


# --- Module to test ---
from tunnex_2.qmt_computations.read_input import ( # type: ignore
    _parse_line,
    _validate_positive,
    _read_irc_matrix,
    read_input)


# --- Helpers and shared data ---
FUNC_PATH = "tunnex_2.qmt_computations.read_input"

HEADER_LINE_1 = "freq=1500.0;T=298.15"
HEADER_LINE_2 = "E0=-1.0;corr-ZPVE0=0.01;potential_scaling_factor=1.0;N=2"
HEADER_LINE_3 = "T_arrhenius_min=200.0;T_arrhenius_max=400.0;T_step=10.0"

E_BLOCK = ["IRC; E", "-1.0;-10.0", "0.0;-9.5", "1.0;-10.0", "END; E"]
ZPVE_BLOCK = ["IRC; ZPVE", "-1.0;0.01", "0.0;0.02", "1.0;0.01", "END; ZPVE"]


def _make_input_text(header1=HEADER_LINE_1, header2=HEADER_LINE_2, header3=HEADER_LINE_3,
                      e_block=None, zpve_block=None):
    """ Should assemble a full, valid input-file text from header lines and IRC blocks """
    e_block = e_block if e_block is not None else E_BLOCK
    zpve_block = zpve_block if zpve_block is not None else ZPVE_BLOCK
    lines = [header1, header2, header3] + e_block + zpve_block
    return "\n".join(lines)


def _write_input_file(tmp_path, text, name="input.txt"):
    """ Should write the given text to a temporary input file and return its path """
    f = tmp_path / name
    f.write_text(text)
    return f


def _read_input_with_mocked_dataclasses(path):
    """ Should call read_input with TunnexInputSettings and IRCDataQMT patched to capture kwargs """
    with ExitStack() as stack:
        settings_mock = stack.enter_context(
            patch(FUNC_PATH + ".TunnexInputSettings",
                  side_effect=lambda **kw: ("TunnexInputSettings", kw)))
        irc_mock = stack.enter_context(
            patch(FUNC_PATH + ".IRCDataQMT",
                  side_effect=lambda **kw: ("IRCDataQMT", kw)))
        result = read_input(path)
    return settings_mock, irc_mock, result


# ===========================================================================
# --- _parse_line - normal behaviour ---
# ===========================================================================
class TestParseLineNormal:

    def test_parses_single_key_value_pair(self):
        """ Verify that a single key=value pair is parsed into a dict with a float value """
        result = _parse_line("freq=1500.0", 1)
        assert result == {"freq": 1500.0}, f"expected {{'freq': 1500.0}}, got {result}"

    def test_parses_multiple_key_value_pairs(self):
        """ Verify that multiple semicolon-separated key=value pairs are all parsed """
        result = _parse_line("freq=1500.0;T=298.15", 1)
        expected = {"freq": 1500.0, "T": 298.15}
        assert result == expected, f"expected {expected}, got {result}"

    def test_strips_whitespace_around_keys_and_values(self):
        """ Verify that whitespace around keys and values is stripped before parsing """
        result = _parse_line(" freq = 1500.0 ; T = 298.15 ", 1)
        expected = {"freq": 1500.0, "T": 298.15}
        assert result == expected, f"expected {expected}, got {result}"


# ===========================================================================
# --- _parse_line - error handling ---
# ===========================================================================
class TestParseLineErrors:

    def test_raises_on_missing_equals_sign(self):
        """ Verify that a line without an '=' sign raises a ValueError """
        with pytest.raises(ValueError):
            _parse_line("freq1500.0", 3)

    def test_raises_on_non_numeric_value(self):
        """ Verify that a non-numeric value raises a ValueError """
        with pytest.raises(ValueError):
            _parse_line("freq=abc", 2)

    def test_error_message_includes_line_number(self):
        """ Verify that the raised error message includes the given line number """
        with pytest.raises(ValueError, match=r"\[Line 7\]"):
            _parse_line("bad", 7)


# ===========================================================================
# --- _validate_positive - normal behaviour ---
# ===========================================================================
class TestValidatePositiveNormal:

    def test_accepts_positive_value(self):
        """ Verify that a strictly positive value passes validation without error """
        _validate_positive("freq", 1500.0)

    def test_accepts_zero_when_allow_zero_is_true(self):
        """ Verify that zero is accepted when allow_zero is True """
        _validate_positive("max_level", 0, allow_zero=True)

    def test_accepts_positive_value_when_allow_zero_is_true(self):
        """ Verify that a positive value still passes validation when allow_zero is True """
        _validate_positive("max_level", 3, allow_zero=True)


# ===========================================================================
# --- _validate_positive - error handling ---
# ===========================================================================
class TestValidatePositiveErrors:

    def test_raises_on_zero_by_default(self):
        """ Verify that zero raises a ValueError when allow_zero is False (default) """
        with pytest.raises(ValueError):
            _validate_positive("freq", 0)

    def test_raises_on_negative_value_by_default(self):
        """ Verify that a negative value raises a ValueError when allow_zero is False """
        with pytest.raises(ValueError):
            _validate_positive("freq", -1.0)

    def test_raises_on_negative_value_even_when_allow_zero_is_true(self):
        """ Verify that a negative value raises a ValueError even when allow_zero is True """
        with pytest.raises(ValueError):
            _validate_positive("max_level", -1, allow_zero=True)

    def test_error_message_includes_name_and_value(self):
        """ Verify that the raised error message includes the parameter name and offending value """
        with pytest.raises(ValueError, match=r"freq.*-5"):
            _validate_positive("freq", -5)


# ===========================================================================
# --- _read_irc_matrix - normal behaviour ---
# ===========================================================================
class TestReadIrcMatrixNormal:

    def test_extracts_x_and_y_values_between_tags(self):
        """ Verify that x and y values are correctly extracted from within the start/end tags """
        lines = E_BLOCK
        x_vals, y_vals = _read_irc_matrix(lines, "IRC; E", "END; E")
        assert np.allclose(x_vals, [-1.0, 0.0, 1.0]), f"expected {[-1.0, 0.0, 1.0]}, got {x_vals}"
        assert np.allclose(y_vals, [-10.0, -9.5, -10.0]), f"expected {[-10.0, -9.5, -10.0]}, got {y_vals}"

    def test_returns_numpy_arrays(self):
        """ Verify that both returned values are NumPy arrays """
        x_vals, y_vals = _read_irc_matrix(E_BLOCK, "IRC; E", "END; E")
        assert isinstance(x_vals, np.ndarray), f"expected {np.ndarray}, got {type(x_vals)}"
        assert isinstance(y_vals, np.ndarray), f"expected {np.ndarray}, got {type(y_vals)}"

    def test_skips_blank_lines_inside_block(self):
        """ Verify that blank lines inside the block are skipped without affecting the extraction """
        lines = ["IRC; E", "-1.0;-10.0", "", "0.0;-9.5", "END; E"]
        x_vals, y_vals = _read_irc_matrix(lines, "IRC; E", "END; E")
        assert np.allclose(x_vals, [-1.0, 0.0]), f"expected {[-1.0, 0.0]}, got {x_vals}"

    def test_ignores_lines_outside_block(self):
        """ Verify that lines outside the start/end tags are not included in the extraction """
        lines = ["junk;line"] + E_BLOCK + ["also;junk"]
        x_vals, y_vals = _read_irc_matrix(lines, "IRC; E", "END; E")
        assert len(x_vals) == 3, f"expected {3}, got {len(x_vals)}"


# ===========================================================================
# --- _read_irc_matrix - error handling ---
# ===========================================================================
class TestReadIrcMatrixErrors:

    def test_raises_on_missing_block(self):
        """ Verify that a ValueError is raised when the start tag is not found at all """
        with pytest.raises(ValueError):
            _read_irc_matrix(["some", "other", "lines"], "IRC; E", "END; E")

    def test_raises_on_empty_block(self):
        """ Verify that a ValueError is raised when the block between tags contains no data rows """
        lines = ["IRC; E", "END; E"]
        with pytest.raises(ValueError):
            _read_irc_matrix(lines, "IRC; E", "END; E")

    def test_raises_on_malformed_row(self):
        """ Verify that a row without a semicolon-separated x;y pair raises a ValueError """
        lines = ["IRC; E", "not_a_valid_row", "END; E"]
        with pytest.raises(ValueError):
            _read_irc_matrix(lines, "IRC; E", "END; E")

    def test_raises_on_non_increasing_x_values(self):
        """ Verify that non-strictly-increasing IRC x values raise a ValueError """
        lines = ["IRC; E", "1.0;-10.0", "0.5;-9.5", "END; E"]
        with pytest.raises(ValueError):
            _read_irc_matrix(lines, "IRC; E", "END; E")

    def test_raises_on_duplicate_x_values(self):
        """ Verify that a repeated (non-strictly-increasing) IRC x value raises a ValueError """
        lines = ["IRC; E", "0.0;-10.0", "0.0;-9.5", "END; E"]
        with pytest.raises(ValueError):
            _read_irc_matrix(lines, "IRC; E", "END; E")


# ===========================================================================
# --- read_input - file-level validation ---
# ===========================================================================
class TestReadInputFileValidation:

    def test_raises_file_not_found_error_for_missing_file(self, tmp_path):
        """ Verify that a FileNotFoundError is raised when the input file does not exist """
        missing = tmp_path / "does_not_exist.txt"
        with pytest.raises(FileNotFoundError):
            read_input(missing)

    def test_raises_on_too_few_header_lines(self, tmp_path):
        """ Verify that a ValueError is raised when the file has fewer than 3 header lines """
        f = _write_input_file(tmp_path, "freq=1500.0;T=298.15\nE0=-1.0")
        with pytest.raises(ValueError):
            read_input(f)

    def test_accepts_str_path_as_well_as_path_object(self, tmp_path):
        """ Verify that read_input accepts a plain string path, not just a Path object """
        f = _write_input_file(tmp_path, _make_input_text())
        _read_input_with_mocked_dataclasses(str(f))  # should not raise


# ===========================================================================
# --- read_input - header key validation ---
# ===========================================================================
class TestReadInputHeaderValidation:

    def test_raises_on_unexpected_key_in_line_one(self, tmp_path):
        """ Verify that an unexpected key in the first header line raises a ValueError """
        text = _make_input_text(header1="freq=1500.0;bogus=1.0")
        f = _write_input_file(tmp_path, text)
        with pytest.raises(ValueError):
            read_input(f)

    def test_raises_on_unexpected_key_in_line_two(self, tmp_path):
        """ Verify that an unexpected key in the second header line raises a ValueError """
        text = _make_input_text(header2="E0=-1.0;corr-ZPVE0=0.01;potential_scaling_factor=1.0;N=2;extra=9")
        f = _write_input_file(tmp_path, text)
        with pytest.raises(ValueError):
            read_input(f)

    def test_raises_on_unexpected_key_in_line_three(self, tmp_path):
        """ Verify that an unexpected key in the third header line raises a ValueError """
        text = _make_input_text(header3="T_arrhenius_min=200.0;T_arrhenius_max=400.0;T_step=10.0;extra=1")
        f = _write_input_file(tmp_path, text)
        with pytest.raises(ValueError):
            read_input(f)


# ===========================================================================
# --- read_input - parameter validation ---
# ===========================================================================
class TestReadInputParameterValidation:

    def test_raises_on_non_positive_freq(self, tmp_path):
        """ Verify that a non-positive freq value raises a ValueError """
        text = _make_input_text(header1="freq=0.0;T=298.15")
        f = _write_input_file(tmp_path, text)
        with pytest.raises(ValueError):
            read_input(f)

    def test_raises_on_non_positive_temperature(self, tmp_path):
        """ Verify that a non-positive temperature value raises a ValueError """
        text = _make_input_text(header1="freq=1500.0;T=-1.0")
        f = _write_input_file(tmp_path, text)
        with pytest.raises(ValueError):
            read_input(f)

    def test_raises_on_non_positive_scaling_factor(self, tmp_path):
        """ Verify that a non-positive potential_scaling_factor raises a ValueError """
        text = _make_input_text(header2="E0=-1.0;corr-ZPVE0=0.01;potential_scaling_factor=0.0;N=2")
        f = _write_input_file(tmp_path, text)
        with pytest.raises(ValueError):
            read_input(f)

    def test_accepts_zero_for_upper_level_to_compute(self, tmp_path):
        """ Verify that N=0 (the_upper_level_to_compute) is accepted since zero is allowed for it """
        text = _make_input_text(header2="E0=-1.0;corr-ZPVE0=0.01;potential_scaling_factor=1.0;N=0")
        f = _write_input_file(tmp_path, text)
        _read_input_with_mocked_dataclasses(f)  # should not raise

    def test_raises_on_negative_upper_level_to_compute(self, tmp_path):
        """ Verify that a negative N (the_upper_level_to_compute) raises a ValueError """
        text = _make_input_text(header2="E0=-1.0;corr-ZPVE0=0.01;potential_scaling_factor=1.0;N=-1")
        f = _write_input_file(tmp_path, text)
        with pytest.raises(ValueError):
            read_input(f)

    def test_raises_on_non_positive_T_arrhenius_min(self, tmp_path):
        """ Verify that a non-positive T_arrhenius_min raises a ValueError """
        text = _make_input_text(header3="T_arrhenius_min=0.0;T_arrhenius_max=400.0;T_step=10.0")
        f = _write_input_file(tmp_path, text)
        with pytest.raises(ValueError):
            read_input(f)

    def test_raises_on_non_positive_T_arrhenius_max(self, tmp_path):
        """ Verify that a non-positive T_arrhenius_max raises a ValueError """
        text = _make_input_text(header3="T_arrhenius_min=200.0;T_arrhenius_max=0.0;T_step=10.0")
        f = _write_input_file(tmp_path, text)
        with pytest.raises(ValueError):
            read_input(f)

    def test_raises_on_non_positive_T_step(self, tmp_path):
        """ Verify that a non-positive T_step raises a ValueError """
        text = _make_input_text(header3="T_arrhenius_min=200.0;T_arrhenius_max=400.0;T_step=0.0")
        f = _write_input_file(tmp_path, text)
        with pytest.raises(ValueError):
            read_input(f)

    def test_raises_when_T_arrhenius_min_greater_than_max(self, tmp_path):
        """ Verify that a ValueError is raised when T_arrhenius_min exceeds T_arrhenius_max """
        text = _make_input_text(header3="T_arrhenius_min=500.0;T_arrhenius_max=400.0;T_step=10.0")
        f = _write_input_file(tmp_path, text)
        with pytest.raises(ValueError):
            read_input(f)


# ===========================================================================
# --- read_input - IRC block consistency ---
# ===========================================================================
class TestReadInputIrcConsistency:

    def test_raises_when_initial_irc_values_differ(self, tmp_path):
        """ Verify that a ValueError is raised when the E and ZPVE blocks start at different IRC values """
        e_block = ["IRC; E", "-1.0;-10.0", "0.0;-9.5", "1.0;-10.0", "END; E"]
        zpve_block = ["IRC; ZPVE", "-2.0;0.01", "0.0;0.02", "1.0;0.01", "END; ZPVE"]
        text = _make_input_text(e_block=e_block, zpve_block=zpve_block)
        f = _write_input_file(tmp_path, text)
        with pytest.raises(ValueError):
            read_input(f)

    def test_raises_when_final_irc_values_differ(self, tmp_path):
        """ Verify that a ValueError is raised when the E and ZPVE blocks end at different IRC values """
        e_block = ["IRC; E", "-1.0;-10.0", "0.0;-9.5", "1.0;-10.0", "END; E"]
        zpve_block = ["IRC; ZPVE", "-1.0;0.01", "0.0;0.02", "2.0;0.01", "END; ZPVE"]
        text = _make_input_text(e_block=e_block, zpve_block=zpve_block)
        f = _write_input_file(tmp_path, text)
        with pytest.raises(ValueError):
            read_input(f)


# ===========================================================================
# --- read_input - successful parsing ---
# ===========================================================================
class TestReadInputSuccess:

    def test_returns_a_two_element_tuple(self, tmp_path):
        """ Verify that read_input returns exactly a (params, irc) tuple """
        f = _write_input_file(tmp_path, _make_input_text())
        _, _, result = _read_input_with_mocked_dataclasses(f)
        assert len(result) == 2, f"expected {2}, got {len(result)}"

    def test_constructs_settings_with_correct_scalar_parameters(self, tmp_path):
        """ Verify that TunnexInputSettings is constructed with the correctly parsed scalar parameters """
        f = _write_input_file(tmp_path, _make_input_text())
        settings_mock, _, _ = _read_input_with_mocked_dataclasses(f)
        kwargs = settings_mock.call_args.kwargs
        assert kwargs["freq"] == 1500.0, f"expected {1500.0}, got {kwargs['freq']}"
        assert kwargs["T"] == 298.15, f"expected {298.15}, got {kwargs['T']}"
        assert kwargs["E0"] == -1.0, f"expected {-1.0}, got {kwargs['E0']}"
        assert kwargs["ZPVE0"] == 0.01, f"expected {0.01}, got {kwargs['ZPVE0']}"
        assert kwargs["potential_scaling_factor"] == 1.0, \
            f"expected {1.0}, got {kwargs['potential_scaling_factor']}"

    def test_constructs_settings_with_upper_level_as_int(self, tmp_path):
        """ Verify that the_upper_level_to_compute is passed to TunnexInputSettings as an int """
        f = _write_input_file(tmp_path, _make_input_text())
        settings_mock, _, _ = _read_input_with_mocked_dataclasses(f)
        kwargs = settings_mock.call_args.kwargs
        assert kwargs["the_upper_level_to_compute"] == 2, \
            f"expected {2}, got {kwargs['the_upper_level_to_compute']}"
        assert isinstance(kwargs["the_upper_level_to_compute"], int), \
            f"expected {int}, got {type(kwargs['the_upper_level_to_compute'])}"

    def test_constructs_settings_with_correct_arrhenius_parameters(self, tmp_path):
        """ Verify that TunnexInputSettings is constructed with the correctly parsed Arrhenius temperature range """
        f = _write_input_file(tmp_path, _make_input_text())
        settings_mock, _, _ = _read_input_with_mocked_dataclasses(f)
        kwargs = settings_mock.call_args.kwargs
        assert kwargs["T_arrhenius_min"] == 200.0, f"expected {200.0}, got {kwargs['T_arrhenius_min']}"
        assert kwargs["T_arrhenius_max"] == 400.0, f"expected {400.0}, got {kwargs['T_arrhenius_max']}"
        assert kwargs["T_arrhenius_step"] == 10.0, f"expected {10.0}, got {kwargs['T_arrhenius_step']}"

    def test_constructs_irc_data_with_correct_arrays(self, tmp_path):
        """ Verify that IRCDataQMT is constructed with the correctly parsed E and ZPVE arrays """
        f = _write_input_file(tmp_path, _make_input_text())
        _, irc_mock, _ = _read_input_with_mocked_dataclasses(f)
        kwargs = irc_mock.call_args.kwargs
        assert np.allclose(kwargs["E_x"], [-1.0, 0.0, 1.0]), f"expected {[-1.0, 0.0, 1.0]}, got {kwargs['E_x']}"
        assert np.allclose(kwargs["E_y"], [-10.0, -9.5, -10.0]), \
            f"expected {[-10.0, -9.5, -10.0]}, got {kwargs['E_y']}"
        assert np.allclose(kwargs["ZPVE_x"], [-1.0, 0.0, 1.0]), \
            f"expected {[-1.0, 0.0, 1.0]}, got {kwargs['ZPVE_x']}"
        assert np.allclose(kwargs["ZPVE_y"], [0.01, 0.02, 0.01]), \
            f"expected {[0.01, 0.02, 0.01]}, got {kwargs['ZPVE_y']}"

    def test_returned_tuple_contains_the_constructed_objects(self, tmp_path):
        """ Verify that the returned tuple contains the objects built from TunnexInputSettings and IRCDataQMT """
        f = _write_input_file(tmp_path, _make_input_text())
        _, _, (params, irc) = _read_input_with_mocked_dataclasses(f)
        assert params[0] == "TunnexInputSettings", f"expected {'TunnexInputSettings'}, got {params[0]}"
        assert irc[0] == "IRCDataQMT", f"expected {'IRCDataQMT'}, got {irc[0]}"

    def test_allows_initial_and_final_irc_values_within_tolerance(self, tmp_path):
        """ Verify that tiny floating-point differences within tolerance at IRC endpoints are accepted """
        e_block = ["IRC; E", "-1.0;-10.0", "0.0;-9.5", "1.0;-10.0", "END; E"]
        zpve_block = ["IRC; ZPVE", "-1.0000001;0.01", "0.0;0.02", "1.0000001;0.01", "END; ZPVE"]
        text = _make_input_text(e_block=e_block, zpve_block=zpve_block)
        f = _write_input_file(tmp_path, text)
        _read_input_with_mocked_dataclasses(f)  # should not raise