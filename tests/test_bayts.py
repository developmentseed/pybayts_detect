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


def test_calc_cpnf_real_data():
    from pandas import read_csv

    from pybayts.bayts import calc_cpnf
    from pybayts.bayts import deseason_ts
    from pybayts.data.io import read_and_stack_tifs

    folder_vv = "tests/baytsdata/s1vv_tseries/"
    pdf_type_s = ("gaussian", "gaussian")
    pdf_forest_s = (-1, 0.75)  # mean and sd
    pdf_nonforest_s = (-4, 1)  # mean and sd
    bwf_s = (0, 1)

    s1vv_ts = read_and_stack_tifs(folder_vv, ds="vv")
    s1vv_ts.name = "s1vv"
    deseason_ts(s1vv_ts)
    dseason_test = s1vv_ts[:, 0, 52][~np.isnan(s1vv_ts[:, 0, 52].data)]
    pnf_s1vv = calc_cpnf(dseason_test, pdf_type_s, pdf_forest_s, pdf_nonforest_s, bwf_s)
    df = read_csv("tests/baytsdata/pnf_test_s1vv.csv")
    s1vv_ts_pix = np.array(df["PNF"])
    s1vv_ts_pix = s1vv_ts_pix[~np.isnan(s1vv_ts_pix)][-16:]
    np.allclose(pnf_s1vv, s1vv_ts_pix, rtol=1e-03)


def test_read_and_stack_example_tif():
    from pybayts.data.io import read_and_stack_tifs

    folder_vv = "tests/baytsdata/s1vv_tseries/"
    folder_ndvi = "tests/baytsdata/lndvi_tseries/"
    s1vv_ts = read_and_stack_tifs(folder_vv, ds="vv")
    lndvi_ts = read_and_stack_tifs(folder_ndvi, ds="lndvi")
    assert len(s1vv_ts.shape) == 3
    assert len(lndvi_ts.shape) == 3


def test_stack_merge():
    from pybayts.bayts import merge_cpnf_tseries
    from pybayts.data.io import read_and_stack_tifs

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

    s1vv_ts = read_and_stack_tifs(folder_vv, ds="vv")
    s1vv_ts.name = "s1vv"

    lndvi_ts = read_and_stack_tifs(folder_ndvi, ds="lndvi")
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
    from pybayts.data.io import read_and_stack_tifs

    s1vv_ts_pix = np.array(
        read_csv("tests/baytsdata/single_deseasoned_ts_s1vv.csv")["x"]
    )[-16:]

    folder_vv = "tests/baytsdata/s1vv_tseries/"

    s1vv_ts = read_and_stack_tifs(folder_vv, ds="vv")
    s1vv_ts.name = "s1vv"

    deseason_ts(s1vv_ts)

    nonnan_deseasoned = s1vv_ts[:, 0, 52][~np.isnan(s1vv_ts[:, 0, 52])].data
    assert np.allclose(nonnan_deseasoned, s1vv_ts_pix, rtol=1e-03)


def test_create_bayts():
    import numpy as np
    from pandas import read_csv

    from pybayts.bayts import create_bayts_ts
    from pybayts.bayts import deseason_ts
    from pybayts.bayts import merge_cpnf_tseries
    from pybayts.data.io import read_and_stack_tifs

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

    s1vv_ts = read_and_stack_tifs(folder_vv, ds="vv")
    s1vv_ts.name = "s1vv"

    lndvi_ts = read_and_stack_tifs(folder_ndvi, ds="lndvi")
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

    assert np.allclose(bayts_r, bayts_ts, rtol=1e-2)