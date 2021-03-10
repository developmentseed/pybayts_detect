from click.testing import CliRunner

from pybayts.cli import main


def test_main():
    runner = CliRunner()
    result = runner.invoke(main, [])

    # this test is silly anyway, just make sure it runs for now
    assert True
    assert result.exit_code == 0
