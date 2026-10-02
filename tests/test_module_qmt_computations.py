"""
test_qmt_computations_module.py — unit tests for qmt_computations_main.

Run with:
    pytest test_qmt_computations_module.py -v
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# Adjust import path as needed:
from qmt_computations_module import qmt_computations_main


# ===========================================================================
# Helpers
# ===========================================================================

def _make_irc():
    m = MagicMock()
    m.E_x    = [0.0, 1.0, 2.0]
    m.E_y    = [-153.0, -152.9, -153.0]
    m.ZPVE_x = [0.0, 1.0, 2.0]
    m.ZPVE_y = [0.03, 0.04, 0.03]
    return m


def _make_params():
    m = MagicMock()
    m.E0                      = -153.15
    m.ZPVE0                   = 0.02
    m.potential_scaling_factor = 1.0
    m.T                       = 300.0
    return m


def _base_patches(tmp_path):
    """Patch dict for a complete successful run."""
    inp = tmp_path / "mol_forward.txt"
    inp.write_text("dummy", encoding="utf-8")
    return inp, {
        "qmt_computations_module.read_input":
            MagicMock(return_value=(_make_params(), _make_irc())),
        "qmt_computations_module.detect_outliers_local":
            MagicMock(),
        "qmt_computations_module.Potential":
            MagicMock(return_value=MagicMock(offset=0.01)),
        "qmt_computations_module.qmt_computations_pipeline":
            MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock())),
        "qmt_computations_module.build_report":
            MagicMock(return_value="report text"),
        "qmt_computations_module.write_output":
            MagicMock(),
    }


# ===========================================================================
# File validation
# ===========================================================================

class TestQmtComputationsMainValidation:

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="File not found"):
            qmt_computations_main(tmp_path / "nonexistent.txt")

    def test_string_path_accepted(self, tmp_path):
        inp, patches = _base_patches(tmp_path)
        with patch.multiple("qmt_computations_module", **{
            k.split(".")[-1]: v for k, v in patches.items()
        }):
            qmt_computations_main(str(inp))   # should not raise

    def test_path_object_accepted(self, tmp_path):
        inp, patches = _base_patches(tmp_path)
        with patch.multiple("qmt_computations_module", **{
            k.split(".")[-1]: v for k, v in patches.items()
        }):
            qmt_computations_main(Path(inp))


# ===========================================================================
# Output path construction
# ===========================================================================

class TestOutputPathConstruction:

    def test_output_filename_has_tunnex_2_suffix(self, tmp_path):
        inp, patches = _base_patches(tmp_path)
        captured = {}
        def capture_write(path, report):
            captured["path"] = path
        patches["qmt_computations_module.write_output"] = MagicMock(side_effect=capture_write)
        with patch.multiple("qmt_computations_module", **{
            k.split(".")[-1]: v for k, v in patches.items()
        }):
            qmt_computations_main(inp)
        assert captured["path"].name == "mol_forward_tunnex_2.out"

    def test_output_file_in_same_directory_as_input(self, tmp_path):
        inp, patches = _base_patches(tmp_path)
        captured = {}
        def capture_write(path, report):
            captured["path"] = path
        patches["qmt_computations_module.write_output"] = MagicMock(side_effect=capture_write)
        with patch.multiple("qmt_computations_module", **{
            k.split(".")[-1]: v for k, v in patches.items()
        }):
            qmt_computations_main(inp)
        assert captured["path"].parent == inp.parent


# ===========================================================================
# read_input
# ===========================================================================

class TestReadInput:

    def test_read_input_called_with_resolved_path(self, tmp_path):
        inp, patches = _base_patches(tmp_path)
        with patch.multiple("qmt_computations_module", **{
            k.split(".")[-1]: v for k, v in patches.items()
        }):
            qmt_computations_main(inp)
        read_mock = patches["qmt_computations_module.read_input"]
        read_mock.assert_called_once()
        called_path = read_mock.call_args[0][0]
        assert called_path == inp.resolve()


# ===========================================================================
# detect_outliers_local
# ===========================================================================

class TestDetectOutliers:

    def test_detect_outliers_called_twice(self, tmp_path):
        inp, patches = _base_patches(tmp_path)
        with patch.multiple("qmt_computations_module", **{
            k.split(".")[-1]: v for k, v in patches.items()
        }):
            qmt_computations_main(inp)
        dol = patches["qmt_computations_module.detect_outliers_local"]
        assert dol.call_count == 2

    def test_detect_outliers_called_for_el_energy_and_zpve(self, tmp_path):
        inp, patches = _base_patches(tmp_path)
        with patch.multiple("qmt_computations_module", **{
            k.split(".")[-1]: v for k, v in patches.items()
        }):
            qmt_computations_main(inp)
        dol = patches["qmt_computations_module.detect_outliers_local"]
        modes = [c[0][2] for c in dol.call_args_list]
        assert "electronic energy" in modes
        assert "ZPVE" in modes

    def test_detect_outliers_called_with_irc_data(self, tmp_path):
        inp, patches = _base_patches(tmp_path)
        irc = _make_irc()
        patches["qmt_computations_module.read_input"] = MagicMock(
            return_value=(_make_params(), irc)
        )
        with patch.multiple("qmt_computations_module", **{
            k.split(".")[-1]: v for k, v in patches.items()
        }):
            qmt_computations_main(inp)
        dol = patches["qmt_computations_module.detect_outliers_local"]
        first_call_x = dol.call_args_list[0][0][0]
        assert first_call_x is irc.E_x


# ===========================================================================
# Potential construction
# ===========================================================================

class TestPotentialConstruction:

    def test_potential_called_with_correct_args(self, tmp_path):
        inp, patches = _base_patches(tmp_path)
        params = _make_params()
        irc    = _make_irc()
        patches["qmt_computations_module.read_input"] = MagicMock(
            return_value=(params, irc)
        )
        with patch.multiple("qmt_computations_module", **{
            k.split(".")[-1]: v for k, v in patches.items()
        }):
            qmt_computations_main(inp)
        pot_cls = patches["qmt_computations_module.Potential"]
        pot_cls.assert_called_once_with(
            irc.E_x, irc.E_y,
            irc.ZPVE_x, irc.ZPVE_y,
            params.E0, params.ZPVE0,
            params.potential_scaling_factor,
        )


# ===========================================================================
# qmt_computations_pipeline
# ===========================================================================

class TestQmtComputationsPipeline:

    def test_pipeline_called_with_params_potential_and_mode(self, tmp_path):
        inp, patches = _base_patches(tmp_path)
        pot_instance = MagicMock(offset=0.01)
        patches["qmt_computations_module.Potential"] = MagicMock(
            return_value=pot_instance
        )
        with patch.multiple("qmt_computations_module", **{
            k.split(".")[-1]: v for k, v in patches.items()
        }):
            qmt_computations_main(inp, prob_aver_mode="infinite_sum")
        pipe = patches["qmt_computations_module.qmt_computations_pipeline"]
        pipe.assert_called_once()
        args = pipe.call_args[0]
        assert args[1] is pot_instance
        assert args[2] == "infinite_sum"

    def test_default_prob_aver_mode_is_finite_sum(self, tmp_path):
        inp, patches = _base_patches(tmp_path)
        with patch.multiple("qmt_computations_module", **{
            k.split(".")[-1]: v for k, v in patches.items()
        }):
            qmt_computations_main(inp)   # no prob_aver_mode argument
        pipe = patches["qmt_computations_module.qmt_computations_pipeline"]
        assert pipe.call_args[0][2] == "finite_sum"


# ===========================================================================
# build_report and write_output
# ===========================================================================

class TestBuildReportAndWriteOutput:

    def test_build_report_called(self, tmp_path):
        inp, patches = _base_patches(tmp_path)
        with patch.multiple("qmt_computations_module", **{
            k.split(".")[-1]: v for k, v in patches.items()
        }):
            qmt_computations_main(inp)
        patches["qmt_computations_module.build_report"].assert_called_once()

    def test_write_output_called_with_report(self, tmp_path):
        inp, patches = _base_patches(tmp_path)
        patches["qmt_computations_module.build_report"] = MagicMock(
            return_value="my_report"
        )
        with patch.multiple("qmt_computations_module", **{
            k.split(".")[-1]: v for k, v in patches.items()
        }):
            qmt_computations_main(inp)
        wo = patches["qmt_computations_module.write_output"]
        wo.assert_called_once()
        assert wo.call_args[0][1] == "my_report"

    def test_build_report_receives_offset(self, tmp_path):
        inp, patches = _base_patches(tmp_path)
        pot_instance = MagicMock(offset=0.099)
        patches["qmt_computations_module.Potential"] = MagicMock(
            return_value=pot_instance
        )
        with patch.multiple("qmt_computations_module", **{
            k.split(".")[-1]: v for k, v in patches.items()
        }):
            qmt_computations_main(inp)
        br = patches["qmt_computations_module.build_report"]
        # offset is the last positional argument to build_report
        assert br.call_args[0][-1] == 0.099

    def test_returns_none(self, tmp_path):
        inp, patches = _base_patches(tmp_path)
        with patch.multiple("qmt_computations_module", **{
            k.split(".")[-1]: v for k, v in patches.items()
        }):
            result = qmt_computations_main(inp)
        assert result is None
