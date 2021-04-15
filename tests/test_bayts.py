import numpy as np


def test_calc_cpnf():
    from pybayts.bayts import calc_cpnf

    ndvi_time_series = [0.5, 0.6, 0.7]
    pdf_type = ("gaussian", "gaussian")
    pdf_forest = (0.85, 0.1)  # mean and sd
    pdf_nonforest = (0.3, 0.2)  # mean and sd
    bwf = (0, 1)
    # calculate conditional non-forest probabilities
    probabilities = calc_cpnf(
        ndvi_time_series, pdf_type, pdf_forest, pdf_nonforest, bwf
    )
    expected = np.array([0.99283853, 0.78698604, 0.17248069])
    assert np.allclose(probabilities, expected)


def test_read_and_stack_example_tif():
    from pybayts.data.io import read_and_stack_example_tifs

    folder_vv = "tests/baytsdata/s1vv_tseries/"
    folder_ndvi = "tests/baytsdata/lndvi_tseries/"
    s1vv_ts = read_and_stack_example_tifs(folder_vv, ds="vv")
    lndvi_ts = read_and_stack_example_tifs(folder_ndvi, ds="lndvi")
    assert len(s1vv_ts.shape) == 3
    assert len(lndvi_ts.shape) == 3


def test_stack_merge():
    from pybayts.bayts import merge_cpnf_tseries
    from pybayts.data.io import read_and_stack_example_tifs

    folder_vv = "tests/baytsdata/s1vv_tseries/"
    folder_ndvi = "tests/baytsdata/lndvi_tseries/"

    pdf_type_l = ("gaussian", "gaussian")
    pdf_forest_l = (0, 0.1)  # mean and sd
    pdf_nonforest_l = (-0.5, 0.125)  # mean and sd
    bwf_l = (0, 1)
    pdf_type_s = ("gaussian", "gaussian")
    pdf_forest_s = (-1, 0.75)  # mean and sd
    pdf_nonforest_s = (-4, 1)  # mean and sd
    bwf_s = (0, 1)

    s1vv_ts = read_and_stack_example_tifs(folder_vv, ds="vv")
    s1vv_ts.name = "s1vv"

    lndvi_ts = read_and_stack_example_tifs(folder_ndvi, ds="lndvi")
    lndvi_ts.name = "lndvi"

    ds = merge_cpnf_tseries(
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

    assert ds


def test_deseason():
    import numpy as np
    from pandas import read_csv

    from pybayts.bayts import deseason_ts
    from pybayts.data.io import read_and_stack_example_tifs

    s1vv_ts_pix = np.array(
        read_csv("tests/baytsdata/single_deseasoned_ts_s1vv.csv")["x"]
    )[-16:]

    folder_vv = "tests/baytsdata/s1vv_tseries/"

    s1vv_ts = read_and_stack_example_tifs(folder_vv, ds="vv")
    s1vv_ts.name = "s1vv"

    deseason_ts(s1vv_ts)

    nonnan_deseasoned = s1vv_ts[:, 0, 52][~np.isnan(s1vv_ts[:, 0, 52])].data
    # need to truncate one of these since one file wasn't used for truth result
    assert np.allclose(nonnan_deseasoned[1:], s1vv_ts_pix, rtol=1e-03)


def test_create_bayts():
    import numpy as np
    from pandas import read_csv

    from pybayts.bayts import create_bayts_ts
    from pybayts.bayts import deseason_ts
    from pybayts.bayts import merge_cpnf_tseries
    from pybayts.data.io import read_and_stack_example_tifs

    folder_vv = "tests/baytsdata/s1vv_tseries/"
    folder_ndvi = "tests/baytsdata/lndvi_tseries/"

    pdf_type_l = ("gaussian", "gaussian")
    pdf_forest_l = (0, 0.1)  # mean and sd
    pdf_nonforest_l = (-0.5, 0.125)  # mean and sd
    bwf_l = (0.1, 0.9)
    pdf_type_s = ("gaussian", "gaussian")
    pdf_forest_s = (-1, 0.75)  # mean and sd
    pdf_nonforest_s = (-4, 1)  # mean and sd
    bwf_s = (0.1, 0.9)

    s1vv_ts = read_and_stack_example_tifs(folder_vv, ds="vv")
    s1vv_ts.name = "s1vv"

    lndvi_ts = read_and_stack_example_tifs(folder_ndvi, ds="lndvi")
    lndvi_ts.name = "lndvi"

    deseason_ts(s1vv_ts)
    deseason_ts(lndvi_ts)

    ds = merge_cpnf_tseries(
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

    bayts = create_bayts_ts(ds)

    df = read_csv("tests/baytsdata/test_bayts.csv")
    bayts_r = df["PNF"].values[-19:]
    bayts_ts = bayts[:, 0, 52][~np.isnan(bayts[:, 0, 52].data)]

    assert np.allclose(bayts_r, bayts_ts[1:], rtol=1e-2)


def test_match_r_results():
    from datetime import datetime

    import rioxarray as rx
    import xarray as xr

    from pybayts.bayts import bayts_da_to_date_array
    from pybayts.bayts import create_bayts_ts
    from pybayts.bayts import deseason_ts
    from pybayts.bayts import loop_bayts_update
    from pybayts.bayts import merge_cpnf_tseries
    from pybayts.data.io import read_and_stack_example_tifs

    masked_bayts_r = rx.open_rasterio(
        "tests/baytsdata/bayts_spatial_result.tif",
        masked=True,
    )

    vv_folder = "tests/baytsdata/s1vv_tseries/"
    ndvi_folder = "tests/baytsdata/lndvi_tseries/"
    pdf_type_l = ("gaussian", "gaussian")
    chi = 0.9
    cpnf_min = 0.5
    pdf_forest_l = (0, 0.1)  # mean and sd
    pdf_nonforest_l = (-0.5, 0.125)  # mean and sd
    bwf_l = (0.1, 0.9)
    pdf_type_s = ("gaussian", "gaussian")
    pdf_forest_s = (-1, 0.75)  # mean and sd
    pdf_nonforest_s = (-4, 1)  # mean and sd
    bwf_s = (0.1, 0.9)

    s1vv_ts = read_and_stack_example_tifs(vv_folder, ds="vv")
    s1vv_ts.name = "s1vv"

    lndvi_ts = read_and_stack_example_tifs(ndvi_folder, ds="lndvi")
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

    initial_change = xr.where(bayts >= 0.5, True, False)

    monitor_start = datetime(2016, 1, 1)
    flagged_change = loop_bayts_update(
        bayts.data,
        initial_change.data,
        initial_change.date.values,
        chi,
        cpnf_min,
        monitor_start,
    )
    bayts.name = "bayts"
    baytsds = bayts.to_dataset()
    baytsds = baytsds.sel(date=slice(monitor_start, None))
    # Need a dataset for the date coordinates
    baytsds["flagged_change"] = (("date", "y", "x"), flagged_change)
    date_index_arr, actual_dates, decimal_yr_arr = bayts_da_to_date_array(baytsds)
    r_results_ds = masked_bayts_r.to_dataset(dim="band")
    r_results_ds["py_decimal_years"] = (("y", "x"), decimal_yr_arr)
    diff = abs(masked_bayts_r.sel(band=3) - r_results_ds["py_decimal_years"])
    assert (
        diff.max() < 0.0135
    )  # each pixel's detected date should be within ~ 5 days, or
