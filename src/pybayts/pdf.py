"""Calculating conditional non-forest probabilities

Calculating conditional non-forest probability (PNF). Probabilities are
calculated based on pdfs for Forest and Non-Forest (derived from F and
NF distributions). Gaussian and Weibull pdf types are supported.

Original author: Johannes Reiche (Wageningen University)
Implemented in python by: Development Seed
References: {http://www.mdpi.com/2072-4292/7/5/4973}{Reiche et al. (2015): A Bayesian Approach to Combine Landsat and ALOS PALSAR Time Series for Near Real-Time Deforestation Detection. Remote Sensing. 7(5), 4973-4996; doi:10.3390/rs70504973}
"""

from typing import Tuple

import scipy.stats as stats


def calc_pnf(
    time_series,
    pdf_type: Tuple,
    forest_dist: Tuple,
    nforest_dist: Tuple,
    bwf: Tuple = (0, 1),
):
    """Calculating conditional non-forest probability (PNF).

    Args:
        time_series (np.ndarray): np.ndarray containing a time series at a
            single pixel.
        pdf_type (Tuple): Tuple describing forest and nonforest distributions.
            pdf[0] = pdf type for Forest, pdf[1] = pdf type for Nonforest. pdf
            types supported: "gaussian" or "weibull".
        forest_dist (Tuple): forest_dist[0] and forest_dist[1] = pdf
            parameters describing Forest. For Gaussian, (mean, sd).
            For Weibull, (shape, sd).
        nforest_dist (Tuple): nforest_dist[0] and nforest_dist[1] = pdf
            parameters describing Non-Forest. For Gaussian, (mean, sd).
            For Weibull, (shape, sd).
        bwf (Tuple, optional): Block weighting function to truncate the NF
            probability. Defaults to (0,1) = no truncation.

    Returns:
        np.ndarray: Non-forest probabilities, same length as the time series.

    Example:
        ndvi = [0.5,0.6,0.7]
        pdf_type = ["gaussian", "gaussian"]
        pdfF = [0.85, 0.1]  # mean and sd
        pdfNF = [0.3, 0.2]  # mean and sd
        pdf = tuple(pdf_type + pdfF + pdfNF)
        # calculate conditional non-forest probabilities
        calc_pnf(ndvi, pdf)
    """

    if len(time_series) > 0:
        # Gaussian pdf (mean and sd)
        if pdf_type[0] == "gaussian":
            pf = stats.norm.pdf(x=time_series, loc=forest_dist[0], scale=forest_dist[1])
        elif pdf_type[0] == "weibull":
            pf = stats.weibull.pdf(
                x=time_series, shape=forest_dist[0], scale=forest_dist[1]
            )
        else:
            raise ValueError("Must supply 'gaussian' or 'weibull' for pdf[0].")

        # Weibull pdf (shape and scale), TODO figure out why loc (the mean) wasn't supplied for this distribution
        if pdf_type[1] == "gaussian":
            pnf = stats.norm.pdf(
                x=time_series, loc=nforest_dist[0], scale=nforest_dist[1]
            )
        elif pdf_type[1] == "weibull":
            pnf = stats.weibull.pdf(
                x=time_series, shape=nforest_dist[0], scale=nforest_dist[1]
            )
        else:
            raise ValueError("Must supply 'gaussian' or 'weibull' for pdf[1].")

        # calculate conditinal NF
        pnf[pnf < 1e-100] = 0
        pnf[pnf > 0] = pnf[pnf > 0] / (pf[pnf > 0] + pnf[pnf > 0])
        # apply block weighting function
        pnf[pnf < bwf[0]] = bwf[0]
        pnf[pnf > bwf[1]] = bwf[1]
        return pnf
