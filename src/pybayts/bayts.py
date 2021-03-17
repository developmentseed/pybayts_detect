"""Main functions used for calculating bayts time series from multiple time series

Original author: Johannes Reiche (Wageningen University)
Implemented in python by: Development Seed
References: {http://www.mdpi.com/2072-4292/7/5/4973}{Reiche et al. (2015): A Bayesian Approach to Combine Landsat and ALOS PALSAR Time Series for Near Real-Time Deforestation Detection. Remote Sensing. 7(5), 4973-4996; doi:10.3390/rs70504973}
"""

from typing import Tuple

import scipy.stats as stats
import xarray as xr


def stack_merge_cpnf_tseries(
    s1vv_ts,
    lndvi_ts,
    pdf_type_l: Tuple,
    pdf_type_s: Tuple,
    pdf_forest_l: Tuple,
    pdf_nonforest_l: Tuple,
    pdf_forest_s: Tuple,
    pdf_nonforest_s: Tuple,
    bwf_l: Tuple = (0.1, 0.9),
    bwf_s: Tuple = (0.1, 0.9),
):
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
    cpnf_l = calc_cpnf(lndvi_ts, pdf_type_l, pdf_forest_l, pdf_nonforest_l, bwf_l)
    cpnf_s = calc_cpnf(s1vv_ts, pdf_type_s, pdf_forest_s, pdf_nonforest_s, bwf_s)
    lndvi_ts = lndvi_ts.to_dataset()
    s1vv_ts = s1vv_ts.to_dataset()
    lndvi_ts["cpnf_lndvi"] = (["date", "y", "x"], cpnf_l)
    s1vv_ts["cpnf_s1vv"] = (["date", "y", "x"], cpnf_s)
    outer_merge_ts = xr.merge([s1vv_ts, lndvi_ts], join="outer", compat="override")
    return outer_merge_ts

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
            pobs_f = stats.norm.pdf(
                x=time_series, loc=forest_dist[0], scale=forest_dist[1]
            )
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
        pobs_nf[pobs_nf > 0] = pobs_nf[pobs_nf > 0] / (
            pobs_f[pobs_nf > 0] + pobs_nf[pobs_nf > 0]
        )
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
    return (prior * likelihood) / (
        (prior * likelihood) + ((1 - prior) * (1 - likelihood))
    )

def create_bayts_ts(timeseries):
    """Calculates the initial conditional non-forest probability time series before iterative bayesian updating.

    Args:
        timeseries (xr.DataSet): An xarray Dataset containing the two
            time series and conditional non-forest probabilities for each timeseries.

    Returns:
        xr.Dataarray: A Datarray containing the refined conditional non-forest probabilities 
        (a single time series). This is the data fusion part of the bayts algorithm.
    """
    # refined cpnf for dates with observation of s1vv and lndvi
    # lines 68-77 jreiche bayts
    refined_cpnf_two_obs = calc_posterior(
        timeseries["cpnf_s1vv"], timeseries["cpnf_lndvi"]
    )
    # where we have ndvi observations, we want to use the refined cpnf since ndvi is more related to deforestation than backscatter
    timeseries["cpnf_s1vv_refined"] = xr.where(
        timeseries["cpnf_s1vv"].notnull() & timeseries["cpnf_lndvi"].notnull(),
        refined_cpnf_two_obs,
        timeseries["cpnf_s1vv"],
    )
    # where we don't have backscatter but we have ndvi, we want to use ndvi cpnf
    nan_s1vv = timeseries["cpnf_s1vv_refined"].isnull()
    bayts = xr.where(
        nan_s1vv, timeseries["cpnf_lndvi"], timeseries["cpnf_s1vv_refined"]
    )
    # any nans left in the output should be from image boundary issues or quality masking
    return bayts

def iterative_bays_update(bayts, chi: float = 0.5, cpnf_min: float = 0.5):
    """Iterates through each pixel time series to refine the conditional non-forest probability using bayesian updating.

    Args:
        bayts (xr.DataArray): "bayts" time series created with create_bayts from two time series 
            (vv backscatter and ndvi).
        chi (float, optional): Threshold of Pchange at which the change is confirmed. Defaults to 0.5.
        cpnf_min (float, optional): Threshold of conditional non-forest probability above which the first
            observation is flagged. Defaults to 0.5
    """

    # need to probably figure out a better way to do this than iterating over each pixel ts individually
    # in a single process. 1 ts per dask process? numba? cython?
    # https://numpy.org/doc/stable/reference/arrays.nditer.html#arrays-nditer
    for y in range(len(bayts_ts['y'])):
        for x in range(len(bayts_ts['x'])): 
            pixel_ts = bayts_ts[:,y,x]
            possible_nf_indices = np.argwhere(pixel_ts > cpnf_min)
            for ind in possible_nf_indices:
                possible_forest_mask = pixel_ts.where(pixel_ts  < cpnf_min)
#       for (r in 1:length(ind)){
#         for (t in ind[r]:en) {
#           #############################################################
#           # step 2.1: Update Flag and PChange for current time step (t)
#           # (case 1) No confirmed or flagged change: 
#           if (bayts$Flag[(t - 1)] == "0" || bayts$Flag[(t - 
#                                                         1)] == "oldFlag") {
#             i <- 0
#             prior <- as.double(bayts$PNF[t - 1])
#             likelihood <- as.double(bayts$PNF[t])
#             # can be replaced with calcPosterior?
#             postieror <- (prior * likelihood)/((prior * 
#                                                   likelihood) + ((1 - prior) * (1 - likelihood)))
#             bayts$Flag[t] <- "Flag"
#             bayts$PChange[t] <- postieror
#           }
#           # (case 2) Flagged change at preveous time step: update PChange
#           if (bayts$Flag[(t - 1)] == "Flag") {
#             prior <- as.double(bayts$PChange[t - 1])
#             likelihood <- as.double(bayts$PNF[t])
#             postieror <- (prior * likelihood)/((prior * likelihood) + 
#                                                  ((1 - prior) * (1 - likelihood)))
#             bayts$PChange[t] <- postieror
#             bayts$Flag[t] <- "Flag"
#             i <- i + 1
#           }
#           ###############################################
#           # step 2.2: Confirm and reject flagged changes
#           if (bayts$Flag[(t)] == "Flag") {
#             if ((i > 0)) {
#               if ((as.double(bayts$PChange[t])) < 0.5) {
#                 bayts$Flag[(t - i):t] <- 0
#                 bayts$Flag[(t - i)] <- "oldFlag"
#                 break 
#               }
#             }
#             # confirm change in case PChange >= chi
#             if ((as.double(bayts$PChange[t])) >= chi) {
#               if ((as.double(bayts$PNF[t])) >= 0.5) {
#                 bayts$Flag[min(which(bayts$Flag == "Flag")):t] <- "Change"
#                 return(bayts)
#               }
#             }
#           }
#         }
#       }
#     }
#   }
#   return(bayts)
# }
