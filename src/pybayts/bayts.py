"""Main functions used for calculating bayts time series from multiple time series

Original author: Johannes Reiche (Wageningen University)
Implemented in python by: Development Seed
References: {http://www.mdpi.com/2072-4292/7/5/4973}{Reiche et al. (2015): A Bayesian Approach to Combine Landsat and ALOS PALSAR Time Series for Near Real-Time Deforestation Detection. Remote Sensing. 7(5), 4973-4996; doi:10.3390/rs70504973}
"""

from typing import List
from typing import Tuple

import scipy.stats as stats
import xarray as xr


def stack_merge_cpnf_tseries(s1vv_ts, lndvi_ts, pdf_type_l: Tuple, pdf_type_s: Tuple, 
        pdf_forest_l: Tuple,, pdf_nonforest_l: Tuple, pdf_forest_s: Tuple,, pdf_nonforest_s: Tuple, 
        bwf_l: Tuple= (0.1, 0.9), bwf_s: Tuple= (0.1, 0.9)):
    """Calculates CPNF, adds them to their respective observation DataArrays and outer joins two xarray 
        DataArrays into a Dataset.

    Args:
        s1vv_ts (xr.DataArray): The Sentinel-1 time series. Must have 
            dims ["date", "y", "x"] and name "s1vv".
        lndvi_ts (xr.DataArray): The NDVI time series. Must have 
            dims ["date", "y", "x"] and name "lndvi".
        pdf_type_l (Tuple): tuple describing forest and nonforest distributions for landsat ndvi.
            pdf[0] = pdf type for Forest, pdf[1] = pdf type for Nonforest. pdf
            types supported: "gaussian" or "weibull".
        pdf_type_s (Tuple): Tuple describing forest and nonforest distributions for sentinel-1 vv.
            pdf[0] = pdf type for Forest, pdf[1] = pdf type for Nonforest. pdf
            types supported: "gaussian" or "weibull".
        pdf_forest_l (Tuple): pdf parameters describing forest for landsat. For Gaussian, (mean, sd).
            For Weibull, (shape, sd).
        pdf_nonforest_l (Tuple): pdf parameters describing non-forest for landsat. For Gaussian, (mean, sd).
            For Weibull, (shape, sd).
        pdf_forest_s (Tuple): pdf parameters describing forest for sentinel-1. For Gaussian, (mean, sd).
            For Weibull, (shape, sd).
        pdf_nonforest_s (Tuple): pdf parameters describing forest for sentinel-1. For Gaussian, (mean, sd).
            For Weibull, (shape, sd).
        bwf_l (Tuple, optional): Block weighting function to truncate the NF
            probability for landsat. Defaults to (0.1, 0.9).
        bwf_s (Tuple, optional): Block weighting function to truncate the NF
            probability for sentinel-1. Defaults to (0.1, 0.9).

    Returns:
        xr.Dataset: A dataset with 4 variables, "s1vv", "lndvi", "pnf_s1vv",
           "pnf_lndvi" and dims ["date", "y", "x"].
    """
    lndvi_ts = lndvi_ts.to_dataset()
    s1vv_ts = s1vv_ts.to_dataset()
    pnf_l = calc_cpnf(lndvi_ts, pdf_type_l, pdf_forest_l, pdf_nonforest_l, bwf_l)
    pnf_s = calc_cpnf(s1vv_ts, pdf_type_s, pdf_forest_s, pdf_nonforest_s, bwf_s)
    lndvi_ts['pnf_lndvi'] = (["date", "y", "x"], pnf_l)
    s1vv_ts['pnf_s1vv'] = (["date", "y", "x"], pnf_s)
    outer_merge_ts = xr.merge([s1vv_ts, lndvi_ts], join="outer", compat="override")
    return outer_merge_ts

def create_bayts_ts(timeseries):
    """[summary]

    Args:
        timeseries (xr.DataSet): An xarray Dataset containing the two 
            time series and conditional non-forest probabilities. 
        pdf_lst (Tuple): The pdf parameters for calc_cpnf.
        bwf (tuple, optional): Block wieghting parameters for calc_cpnf. 
            Defaults to (0.1, 0.9).

    Returns:
        xr.DataSet: A Dataset containing the original 2 time series and 
            refined conditional non-forest probabilities, or "bayts time 
            series". 
    """
    # refined cpnf for dates with observation of s1vv and lndvi
    # lines 68-77 jreiche bayts
    refined_cpnf_two_obs = calc_posterior(timeseries['pnf_s1vv'], timeseries['pnf_lndvi'])
    timeseries_refined = xr.where(timeseries['pnf_s1vv'].notnull() & timeseries['pnf_lndvi'].notnull(), refined_cpnf_two_obs, timeseries)
    nan_s1vv = timeseries_refined['pnf_s1vv'].isnull()
    refined_s1vv = xr.where(nan_s1vv, timeseries_refined['pnf_lndvi'], timeseries_refined['pnf_s1vv'])
    timeseries_refined['pnf_s1vv'] = refined_s1vv
    return timeseries_refined


def calc_cpnf(
    time_series,
    pdf_type: Tuple,
    forest_dist: Tuple,
    nforest_dist: Tuple,
    bwf: Tuple,
):
    """Calculating conditional non-forest probability (CPNF).

    Calculating conditional non-forest probability (CPNF). Probabilities are
    calculated based on pdfs for Forest and Non-Forest (derived from F and
    NF distributions). Gaussian and Weibull pdf types are supported.

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
        bwf (Tuple): Block weighting function to truncate the NF
            probability.

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
            # the probability of observing the observation given it is non-forest
            pobs_f = stats.norm.pdf(x=time_series, loc=forest_dist[0], scale=forest_dist[1])
        elif pdf_type[0] == "weibull":
            pobs_f = stats.weibull.pdf(
                x=time_series, shape=forest_dist[0], scale=forest_dist[1]
            )
        else:
            raise ValueError("Must supply 'gaussian' or 'weibull' for pdf[0].")

        # Weibull pdf (shape and scale), TODO figure out why loc (the mean) wasn't supplied for this distribution
        if pdf_type[1] == "gaussian":
            pobs_nf = stats.norm.pdf(
                x=time_series, loc=nforest_dist[0], scale=nforest_dist[1]
            )
        elif pdf_type[1] == "weibull":
            pobs_nf = stats.weibull.pdf(
                x=time_series, shape=nforest_dist[0], scale=nforest_dist[1]
            )
        else:
            raise ValueError("Must supply 'gaussian' or 'weibull' for pdf[1].")

        # calculate conditinal NF
        pobs_nf[pobs_nf < 1e-100] = 0
        # pobs_nf is now the conditional NF probability, not p of the observation given NF
        pobs_nf[pobs_nf > 0] = pobs_nf[pobs_nf > 0] / (pobs_f[pobs_nf > 0] + pobs_nf[pobs_nf > 0])
        # apply block weighting function
        pobs_nf[pobs_nf < bwf[0]] = bwf[0]
        pobs_nf[pobs_nf > bwf[1]] = bwf[1]
        return pobs_nf

def calc_posterior(prior, likelihood):
    """Calculates posterior probability according to Bayes' theorem.

    Args:
        prior (float):  Probability a pixel is non-forest before the next data observation is accounted for. 
            Single value or array like.
        likelihood (float): Likelihood of observing a pixel observation given it is non-forest. 
            Single value or array like of same shape. 

    Returns:
        float: The posterior probability of a pixel being non forest given an observed data point.
    """
    return (prior * likelihood)/ ((prior * likelihood) + ((1 - prior) * (1 - likelihood)))


