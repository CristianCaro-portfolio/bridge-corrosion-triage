import os

from typer.testing import CliRunner

from bridge_triage.cli import app


def test_demo_runs_offline(root):
    cwd = os.getcwd()
    os.chdir(root)
    try:
        result = CliRunner().invoke(app, ["demo"])
    finally:
        os.chdir(cwd)
    assert result.exit_code == 0, result.output
    assert "8 findings stored" in result.output
