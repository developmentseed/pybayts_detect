"""Calculating conditional non-forest probabilities

Calculating conditional non-forest probability (PNF). Probabilities are calculated 
based on pdfs for Forest and Non-Forest (derived from F and NF distributions). 
Guassian and Weibull pdf types are supported.

Original author: Johannes Reiche (Wageningen University)
Implemented in python by: Development Seed
References: {http://www.mdpi.com/2072-4292/7/5/4973}{Reiche et al. (2015): A Bayesian Approach to Combine Landsat and ALOS PALSAR Time Series for Near Real-Time Deforestation Detection. Remote Sensing. 7(5), 4973-4996; doi:10.3390/rs70504973}
"""

from typing import Tuple
import numpy as np
import scipy.stats as stats


def calc_pnf(
    time_series: np.ndarray, pdf: Tuple, bwf: Tuple = (0, 1)
) -> np.ndarray:
    """Calculating conditional non-forest probability (PNF).

    Args:
        time_series (np.ndarray): np.ndarray containing time series at a single pixel.
        pdf (Tuple): Tuple describing forest and nonforest distributions.
            pdf[0] = pdf type for Forest, pdf[1] = pdf type for Nonforest, pdf[2] and pdf[3] = pdf
            parameters describing Forest, pdf[5] and pdf[6] = pdf parameter describing Nonforest.
            pdf types supported: Gaussian or Weibull.
        bwf (Tuple, optional): Block weighting function to truncate the NF probability.
            Defaults to (0,1) = no truncation.

    Returns:
        np.ndarray: Non-forest probabilities, same length as the time series.

    Example:
        ndvi = [0.5,0.6,0.7]
        pdf_type = ["gaussian","gaussian"]
        pdfF = (0.85, 0.1) # mean and sd
        pdfNF = (0.3, 0.2)  #mean and sd
        pdf = pdf_type + pdfF + pdfNF
        # calculate conditional non-forest probabilities
        calc_pnf(ndvi, pdf)
    """

    if len(time_series) > 0:
        # Gaussian pdf (mean and sd)
        if pdf[0] == "gaussian":
            pf = stats.norm.pdf(x=time_series, loc=pdf[2], scale=pdf[3])
        elif pdf[0] == "weibull":
            pf = stats.weibull.pdf(x=time_series, shape=pdf[2], scale=pdf[3])
        else:
            raise ValueError("Must supply 'gaussian' or 'weibull' for pdf[0].")

        # Weibull pdf (shape and scale), TODO figure out why loc (the mean) wasn't supplied for this distribution
        if pdf[1] == "gaussian":
            pnf = stats.norm.pdf(x=time_series, loc=pdf[4], scale=pdf[5])
        elif pdf[1] == "weibull":
            pnf = stats.weibull.pdf(x=time_series, scale=pdf[5], scale=pdf[5])
        else:
            raise ValueError("Must supply 'gaussian' or 'weibull' for pdf[1].")

        # calculate conditinal NF
        pnf[pnf < 1e-10000] = 0
        pnf[pnf > 0] = pnf[pnf > 0] / (pf[pnf > 0] + pnf[pnf > 0])
        ## apply block weighting function
        pnf[pnf < bwf[0]] = bwf[0]
        pnf[pnf > bwf[1]] = bwf[1]
        return pnf
