"""
Module that contains the command line app.

Why does this file exist, and why not put this in __main__?

  You might be tempted to import things from __main__ later, but that will cause
  problems: the code will get executed twice:

  - When you run `python -m pybayts` python will execute
    ``__main__.py`` as a script. That means there won't be any
    ``pybayts.__main__`` in ``sys.modules``.
  - When you import __main__ it will get executed again (as a module) because
    there's no ``pybayts.__main__`` in ``sys.modules``.

  Also see (1) from http://click.pocoo.org/5/setuptools/#setuptools-integration
"""
import fire
from eval import run_bayts_and_evaluate


if __name__ == "__main__":
    fire.Fire(run_bayts_and_evaluate)
