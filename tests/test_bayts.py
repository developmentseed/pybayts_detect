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


def test_create_bayts():
    from pybayts.bayts import create_bayts_ts
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

    cond_nf_prob_ts = create_bayts_ts(ds)

    assert len(cond_nf_prob_ts.shape) == 3

def test_deseason():
    from pybayts.bayts import create_bayts_ts
    from pybayts.bayts import deseason_ts
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

    deseason_ts(s1vv_ts)

    nonnan_deseasoned = s1vv_ts[:,0,52][~np.isnan(s1vv_ts[:,0,52])].data
    np.allclose(nonnan_deseasoned, s1vv_ts_pix, rtol=1e-03)
