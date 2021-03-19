"""
Module that contains the command line app.

Why does this file exist, and why not put this in __main__?

  You might be tempted to import things from __main__ later, but that will cause
  problems: the code will get executed twice:

  - When you run `python -mpybayts` python will execute
    ``__main__.py`` as a script. That means there won't be any
    ``pybayts.__main__`` in ``sys.modules``.
  - When you import __main__ it will get executed again (as a module) because
    there's no ``pybayts.__main__`` in ``sys.modules``.

  Also see (1) from http://click.pocoo.org/5/setuptools/#setuptools-integration
"""
import click
from pybayts.bayts import (
    merge_cpnf_tseries,
    deseason_ts,
    create_bayts_ts,
    iterative_bays_update,
)
from pybayts.data.io import read_and_stack_tifs


@click.command()
def main():
    folder_vv = "/home/rave/ms-sar/ms-sar-deforestation-internal/data/baytsdata/s1vv_tseries/"
    folder_ndvi = "/home/rave/ms-sar/ms-sar-deforestation-internal/data/baytsdata/lndvi_tseries/"
    pdf_type_l = ("gaussian", "gaussian")
    pdf_forest_l = (0, 0.1)  # mean and sd
    pdf_nonforest_l = (-0.5, 0.125)  # mean and sd
    bwf_l = (0.1, 0.9)
    pdf_type_s = ("gaussian", "gaussian")
    pdf_forest_s = (-1, 0.75)  # mean and sd
    pdf_nonforest_s = (-4, 1)  # mean and sd
    bwf_s = (0.1, 0.9)

    s1vv_ts = read_and_stack_tifs(folder_vv, ds="vv")
    s1vv_ts.name = "s1vv"

    lndvi_ts = read_and_stack_tifs(folder_ndvi, ds="lndvi")
    lndvi_ts.name = "lndvi"

    _ = deseason_ts(s1vv_ts)
    _ = deseason_ts(lndvi_ts)

    cpnf_ts = merge_cpnf_tseries(
        s1vv_ts,
        lndvi_ts,
        pdf_type_l,
        pdf_type_s,
        pdf_forest_l,
        pdf_nonforest_l,
        pdf_forest_s,
        pdf_nonforest_s,
        bwf_l,
        bwf_s,
    )

    bayts = create_bayts_ts(cpnf_ts)

    ibayts = iterative_bays_update(bayts, chi=0.5, cpnf_min=0.5)


if __name__ == "__main__":
    main()
