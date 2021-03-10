import numpy as np

import pybayts.pdf as ppdf


def test_calc_pnf():
    ndvi_time_series = [0.5, 0.6, 0.7]
    pdf_type = ("gaussian", "gaussian")
    pdf_forest = (0.85, 0.1)  # mean and sd
    pdf_nonforest = (0.3, 0.2)  # mean and sd
    bwf = (0, 1)
    # calculate conditional non-forest probabilities
    probabilities = ppdf.calc_pnf(
        ndvi_time_series, pdf_type, pdf_forest, pdf_nonforest, bwf
    )
    expected = np.array([0.99283853, 0.78698604, 0.17248069])
    assert np.allclose(probabilities, expected)
