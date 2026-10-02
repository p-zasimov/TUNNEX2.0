# --- Test: test_build_report. Unit tests for .\qmt_computations\build_report.py
# Run with: pytest test_build_report.py -v ---


# --- Modules ---
import pytest # type: ignore
from unittest.mock import MagicMock, patch
from contextlib import ExitStack

import numpy as np
from pathlib import Path

from tunnex_2.constants_and_settings.constants_and_settings import KJ_MOL_M1_TO_HARTREE # type: ignore


# --- Module to test ---
from tunnex_2.qmt_computations.build_report import ( # type: ignore
    _fmt,
    _section_level_results_1,
    _section_level_results_2,
    _section_temperature_aver_results,
    _section_arrhenius,
    _section_irc_interpol,
    _section_offset,
    build_report,
    write_output,
    CHAR_NUM_VAL,
    CHAR_NUM_T,
    CHAR_NUM_OFFSET,
    ROUND_NUM_VAL,
    ROUND_NUM_T)


# --- Helpers and shared data ---
FUNC_PATH = "tunnex_2.qmt_computations.build_report"


def _make_level(vib_index=0, vib_energy=0.01, turning_point_left=-0.5, turning_point_right=0.5,
                 wkb=1.2, transmission_probability=0.3, reaction_rate=1e10,
                 half_s=6.9e-11, half_h=1.9e-14, half_d=8.0e-16, half_y=2.2e-18):
    """ Should build a MagicMock standing in for a single LevelResult row """
    level = MagicMock()
    level.vib_index = vib_index
    level.vib_energy = vib_energy
    level.turning_point_left = turning_point_left
    level.turning_point_right = turning_point_right
    level.wkb = wkb
    level.transmission_probability = transmission_probability
    level.reaction_rate = reaction_rate
    level.half_s = half_s
    level.half_h = half_h
    level.half_d = half_d
    level.half_y = half_y
    return level


def _make_temperature_aver(transmission_probability=0.25, reaction_rate=2e9,
                            half_s=3.5e-10, half_h=9.7e-14, half_d=4.0e-15, half_y=1.1e-17):
    """ Should build a MagicMock standing in for a TemperatureAverResult object """
    data = MagicMock()
    data.transmission_probability = transmission_probability
    data.reaction_rate = reaction_rate
    data.half_s = half_s
    data.half_h = half_h
    data.half_d = half_d
    data.half_y = half_y
    return data


def _make_arrhenius_row(arrhenius_temperature=298.15, arrhenius_temperature_inv=1.0 / 298.15,
                         log_reaction_rate=23.5):
    """ Should build a MagicMock standing in for a single ArrheniusResult row """
    row = MagicMock()
    row.arrhenius_temperature = arrhenius_temperature
    row.arrhenius_temperature_inv = arrhenius_temperature_inv
    row.log_reaction_rate = log_reaction_rate
    return row


def _make_irc_row(irc_value=0.0, energy_value=0.0):
    """ Should build a single dict-like row for the interpolated IRC grid """
    return {"irc_value": irc_value, "energy_value": energy_value}


# ===========================================================================
# --- _fmt - normal behaviour ---
# ===========================================================================
class TestFmtNormal:

    def test_formats_plain_float_in_scientific_notation(self):
        """ Verify that a plain float value is formatted in scientific notation and centered """
        result = _fmt(1.2345, 28, 6)
        assert "e" in result, f"expected scientific notation in {result}"
        assert len(result) == 28, f"expected length {28}, got {len(result)}"

    def test_formats_with_is_int_flag_as_fixed_point(self):
        """ Verify that is_int=True formats the value with fixed-point notation instead of scientific """
        result = _fmt(3.0, 10, 6, is_int=True)
        assert result.strip() == "3.00", f"expected {'3.00'}, got {result.strip()}"

    def test_none_value_returns_no_turning_point_label(self):
        """ Verify that a None value is rendered as the 'No turning point' label """
        result = _fmt(None, 28, 6)
        assert "No turning point" in result, f"expected 'No turning point' in {result}"

    def test_infinite_value_returns_plus_inf_label(self):
        """ Verify that an infinite value is rendered as the '+inf' label """
        result = _fmt(np.inf, 28, 6)
        assert "+inf" in result, f"expected '+inf' in {result}"

    def test_result_width_matches_char_num(self):
        """ Verify that the returned string always has the requested character width """
        result = _fmt(0.001, 40, 4)
        assert len(result) == 40, f"expected length {40}, got {len(result)}"

    def test_rounding_digits_are_respected(self):
        """ Verify that the number of digits after the decimal point matches round_num """
        result = _fmt(1.23456789, 30, 3)
        mantissa = result.strip().split("e")[0]
        decimals = mantissa.split(".")[1]
        assert len(decimals) == 3, f"expected {3}, got {len(decimals)}"


# ===========================================================================
# --- _section_level_results_1 - normal behaviour ---
# ===========================================================================
class TestSectionLevelResults1Normal:

    def test_returns_a_string(self):
        """ Verify that the function returns a single string """
        result = _section_level_results_1([_make_level()])
        assert isinstance(result, str), f"expected {str}, got {type(result)}"

    def test_includes_the_section_title(self):
        """ Verify that the output includes the section title describing the table contents """
        result = _section_level_results_1([_make_level()])
        assert "vibrational level indexes" in result, f"expected title text in {result}"

    def test_includes_a_row_for_each_level(self):
        """ Verify that one output row is produced for each vibrational level in the input """
        levels = [_make_level(vib_index=i) for i in range(3)]
        result = _section_level_results_1(levels)
        for i in range(3):
            assert str(i) in result, f"expected index {i} to appear in {result}"

    def test_empty_data_still_returns_header_only(self):
        """ Verify that an empty list of levels still produces a header without raising """
        result = _section_level_results_1([])
        assert "N" in result, f"expected header text in {result}"

    def test_none_turning_points_render_as_label(self):
        """ Verify that a level with None turning points renders the 'No turning point' label in the row """
        level = _make_level(turning_point_left=None, turning_point_right=None)
        result = _section_level_results_1([level])
        assert "No turning point" in result, f"expected 'No turning point' in {result}"

    def test_vib_energy_is_converted_using_kj_mol_conversion_factor(self):
        """ Verify that vib_energy is divided by KJ_MOL_M1_TO_HARTREE before being formatted into the row """
        level = _make_level(vib_energy=KJ_MOL_M1_TO_HARTREE * 2.0)
        result = _section_level_results_1([level])
        assert "2.00000e+00" in result, f"expected converted energy in {result}"


# ===========================================================================
# --- _section_level_results_2 - normal behaviour ---
# ===========================================================================
class TestSectionLevelResults2Normal:

    def test_returns_a_string(self):
        """ Verify that the function returns a single string """
        result = _section_level_results_2([_make_level()])
        assert isinstance(result, str), f"expected {str}, got {type(result)}"

    def test_includes_the_section_title(self):
        """ Verify that the output includes text describing transmission probabilities and half-lives """
        result = _section_level_results_2([_make_level()])
        assert "transmission probabilities" in result, f"expected title text in {result}"

    def test_includes_a_row_for_each_level(self):
        """ Verify that one output row is produced for each vibrational level in the input """
        levels = [_make_level(vib_index=i) for i in range(4)]
        result = _section_level_results_2(levels)
        for i in range(4):
            assert str(i) in result, f"expected index {i} to appear in {result}"

    def test_infinite_half_life_renders_plus_inf_label(self):
        """ Verify that an infinite half_y value renders the '+inf' label in the row """
        level = _make_level(half_y=np.inf)
        result = _section_level_results_2([level])
        assert "+inf" in result, f"expected '+inf' in {result}"


# ===========================================================================
# --- _section_temperature_aver_results - normal behaviour ---
# ===========================================================================
class TestSectionTemperatureAverResultsNormal:

    def test_returns_a_string(self):
        """ Verify that the function returns a single string """
        result = _section_temperature_aver_results(_make_temperature_aver(), 298.15)
        assert isinstance(result, str), f"expected {str}, got {type(result)}"

    def test_includes_the_given_temperature_in_the_title(self):
        """ Verify that the given temperature value appears in the section title """
        result = _section_temperature_aver_results(_make_temperature_aver(), 310.0)
        assert "310.00" in result, f"expected {'310.00'} in {result}"

    def test_includes_averaged_data_row_label(self):
        """ Verify that the data row is labeled as 'averaged data' """
        result = _section_temperature_aver_results(_make_temperature_aver(), 298.15)
        assert "averaged data" in result, f"expected {'averaged data'} in {result}"

    def test_includes_transmission_probability_value(self):
        """ Verify that the transmission probability value from the data object appears in the output """
        data = _make_temperature_aver(transmission_probability=0.55)
        result = _section_temperature_aver_results(data, 298.15)
        assert "5.50000e-01" in result, f"expected formatted probability in {result}"


# ===========================================================================
# --- _section_arrhenius - normal behaviour ---
# ===========================================================================
class TestSectionArrheniusNormal:

    def test_returns_a_string(self):
        """ Verify that the function returns a single string """
        result = _section_arrhenius([_make_arrhenius_row()])
        assert isinstance(result, str), f"expected {str}, got {type(result)}"

    def test_includes_the_section_title(self):
        """ Verify that the output mentions Arrhenius plot data """
        result = _section_arrhenius([_make_arrhenius_row()])
        assert "Arrhenius plot data" in result, f"expected title text in {result}"

    def test_includes_a_row_for_each_arrhenius_point(self):
        """ Verify that one output row is produced for each row in the Arrhenius data """
        rows = [_make_arrhenius_row(arrhenius_temperature=float(t)) for t in (200, 250, 300)]
        result = _section_arrhenius(rows)
        for t in (200, 250, 300):
            assert f"{float(t):^28.6e}" in result, f"expected formatted temperature {t} in {result}"

    def test_empty_data_still_returns_header_only(self):
        """ Verify that an empty list of Arrhenius rows still produces a header without raising """
        result = _section_arrhenius([])
        assert "ln(k)" in result, f"expected header text in {result}"


# ===========================================================================
# --- _section_irc_interpol - normal behaviour ---
# ===========================================================================
class TestSectionIrcInterpolNormal:

    def test_returns_a_string(self):
        """ Verify that the function returns a single string """
        result = _section_irc_interpol([_make_irc_row()])
        assert isinstance(result, str), f"expected {str}, got {type(result)}"

    def test_includes_the_section_title(self):
        """ Verify that the output mentions the ZPVE-corrected IRC-surface """
        result = _section_irc_interpol([_make_irc_row()])
        assert "ZPVE-corrected IRC-surface" in result, f"expected title text in {result}"

    def test_energy_is_converted_using_kj_mol_conversion_factor(self):
        """ Verify that energy_value is divided by KJ_MOL_M1_TO_HARTREE before being formatted into the row """
        row = _make_irc_row(irc_value=1.0, energy_value=KJ_MOL_M1_TO_HARTREE * 3.0)
        result = _section_irc_interpol([row])
        assert "3.000000" in result, f"expected converted energy in {result}"

    def test_includes_a_row_for_each_grid_point(self):
        """ Verify that one output row is produced for each point in the interpolated IRC grid """
        rows = [_make_irc_row(irc_value=float(i)) for i in range(5)]
        result = _section_irc_interpol(rows)
        for i in range(5):
            assert f"{float(i):^28.6f}" in result, f"expected formatted irc value {i} in {result}"


# ===========================================================================
# --- _section_offset - normal behaviour ---
# ===========================================================================
class TestSectionOffsetNormal:

    def test_returns_a_string(self):
        """ Verify that the function returns a single string """
        result = _section_offset(0.0)
        assert isinstance(result, str), f"expected {str}, got {type(result)}"

    def test_includes_the_offset_definition_text(self):
        """ Verify that the output includes the textual definition of the offset """
        result = _section_offset(0.0)
        assert "Offset is defined as" in result, f"expected definition text in {result}"

    def test_offset_is_converted_using_kj_mol_conversion_factor(self):
        """ Verify that the offset value is divided by KJ_MOL_M1_TO_HARTREE before being formatted """
        result = _section_offset(KJ_MOL_M1_TO_HARTREE * 5.0)
        assert "5.000000" in result, f"expected converted offset in {result}"


# ===========================================================================
# --- build_report - integration of sections ---
# ===========================================================================
class TestBuildReportNormal:

    def _call(self):
        """ Should call build_report with mocked, self-consistent input data of all kinds """
        levels = [_make_level(vib_index=0), _make_level(vib_index=1)]
        aver = _make_temperature_aver()
        arrhenius = [_make_arrhenius_row()]
        irc = [_make_irc_row()]
        return build_report(levels, aver, 298.15, arrhenius, irc, 0.0)

    def test_returns_a_string(self):
        """ Verify that build_report returns a single combined string """
        result = self._call()
        assert isinstance(result, str), f"expected {str}, got {type(result)}"

    def test_calls_each_section_builder_exactly_once(self):
        """ Verify that build_report delegates to each section-building function exactly once """
        levels = [_make_level()]
        aver = _make_temperature_aver()
        arrhenius = [_make_arrhenius_row()]
        irc = [_make_irc_row()]
        with ExitStack() as stack:
            mocks = {
                name: stack.enter_context(patch(FUNC_PATH + "." + name, return_value=f"<{name}>"))
                for name in (
                    "_section_level_results_1",
                    "_section_level_results_2",
                    "_section_temperature_aver_results",
                    "_section_arrhenius",
                    "_section_irc_interpol",
                    "_section_offset")
            }
            build_report(levels, aver, 298.15, arrhenius, irc, 1.0)
        for name, mock in mocks.items():
            mock.assert_called_once()

    def test_passes_correct_arguments_to_temperature_aver_section(self):
        """ Verify that build_report forwards the averaged data and temperature to their section builder """
        levels = [_make_level()]
        aver = _make_temperature_aver()
        arrhenius = [_make_arrhenius_row()]
        irc = [_make_irc_row()]
        with patch(FUNC_PATH + "._section_temperature_aver_results", return_value="") as mock:
            build_report(levels, aver, 310.5, arrhenius, irc, 0.0)
        args = mock.call_args.args
        assert args[0] is aver, f"expected {aver}, got {args[0]}"
        assert args[1] == 310.5, f"expected {310.5}, got {args[1]}"

    def test_passes_correct_offset_to_offset_section(self):
        """ Verify that build_report forwards the given offset value to the offset section builder """
        levels = [_make_level()]
        aver = _make_temperature_aver()
        arrhenius = [_make_arrhenius_row()]
        irc = [_make_irc_row()]
        with patch(FUNC_PATH + "._section_offset", return_value="") as mock:
            build_report(levels, aver, 298.15, arrhenius, irc, 42.0)
        mock.assert_called_once_with(42.0)

    def test_output_contains_content_from_all_sections(self):
        """ Verify that the combined report string contains the output of every section """
        levels = [_make_level()]
        aver = _make_temperature_aver()
        arrhenius = [_make_arrhenius_row()]
        irc = [_make_irc_row()]
        with ExitStack() as stack:
            for name in (
                "_section_level_results_1",
                "_section_level_results_2",
                "_section_temperature_aver_results",
                "_section_arrhenius",
                "_section_irc_interpol",
                "_section_offset"):
                stack.enter_context(patch(FUNC_PATH + "." + name, return_value=f"<{name}>"))
            result = build_report(levels, aver, 298.15, arrhenius, irc, 0.0)
        for name in (
            "_section_level_results_1",
            "_section_level_results_2",
            "_section_temperature_aver_results",
            "_section_arrhenius",
            "_section_irc_interpol",
            "_section_offset"):
            assert f"<{name}>" in result, f"expected {f'<{name}>'} in {result}"


# ===========================================================================
# --- write_output - normal behaviour ---
# ===========================================================================
class TestWriteOutputNormal:

    def test_writes_report_text_to_the_given_path(self, tmp_path):
        """ Verify that write_output writes the exact report string to the given file path """
        path = tmp_path / "report.out"
        write_output(path, "some report content")
        assert path.read_text(encoding="utf-8") == "some report content", \
            f"expected {'some report content'}, got {path.read_text(encoding='utf-8')}"

    def test_creates_file_when_missing(self, tmp_path):
        """ Verify that write_output creates the output file if it does not already exist """
        path = tmp_path / "new_report.out"
        assert not path.exists()
        write_output(path, "report body")
        assert path.is_file(), f"expected {True}, got {path.is_file()}"

    def test_overwrites_existing_file_contents(self, tmp_path):
        """ Verify that write_output overwrites any pre-existing content at the given path """
        path = tmp_path / "existing.out"
        path.write_text("old content")
        write_output(path, "new content")
        assert path.read_text(encoding="utf-8") == "new content", \
            f"expected {'new content'}, got {path.read_text(encoding='utf-8')}"

    def test_uses_utf8_encoding(self, tmp_path):
        """ Verify that write_output writes the file using UTF-8 encoding (e.g. for non-ASCII minus signs) """
        path = tmp_path / "unicode_report.out"
        write_output(path, "offset = −1.234560 kJ mol-1")
        assert "−" in path.read_text(encoding="utf-8"), f"expected unicode minus sign preserved"
