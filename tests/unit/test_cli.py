"""Smoke tests for the superquantx command line interface."""

from click.testing import CliRunner

from superquantx.cli.main import cli


def _run(*args):
    result = CliRunner().invoke(cli, list(args))
    assert result.exit_code == 0, result.output
    return result.output


def test_version_verbose_lists_sdks():
    out = _run("-v", "version")
    assert "SuperQuantX version" in out
    assert "\\n" not in out


def test_help_uses_real_command_name():
    out = _run("--help")
    assert "superquantx list-backends" in out
    assert "sqx " not in out


def test_list_backends_checks_simulator_and_never_ticks_planned():
    out = _run("list-backends")
    assert "OK   simulator" in out
    planned = out.split("Planned, not implemented:")[1]
    assert "Azure Quantum" in planned and "Rigetti" in planned
    assert "OK   azure" not in out and "OK   rigetti" not in out
