import numpy as np

import pybayts.pdf as ppdf


def test_calc_pnf():
    ndvi = [0.5, 0.6, 0.7]
    pdf_type = ["gaussian", "gaussian"]
    pdfF = [0.85, 0.1]  # mean and sd
    pdfNF = [0.3, 0.2]  # mean and sd
    pdf = tuple(pdf_type + pdfF + pdfNF)
    # calculate conditional non-forest probabilities
    probabilities = ppdf.calc_pnf(ndvi, pdf)
    expected = np.array([0.99283853, 0.78698604, 0.17248069])
    assert np.allclose(probabilities, expected)
