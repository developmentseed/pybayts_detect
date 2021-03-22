"""Main functions used for calculating bayts time series from multiple time series

Original author: Johannes Reiche (Wageningen University)
Implemented in python by: Development Seed
References: {http://www.mdpi.com/2072-4292/7/5/4973}{Reiche et al. (2015): A Bayesian Approach to Combine Landsat and ALOS PALSAR Time Series for Near Real-Time Deforestation Detection. Remote Sensing. 7(5), 4973-4996; doi:10.3390/rs70504973}
"""

from typing import Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
import xarray as xr
from scipy.sparse import coo_matrix


def merge_cpnf_tseries(
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
    cpnf_l = calc_cpnf(
        lndvi_ts, pdf_type_l, pdf_forest_l, pdf_nonforest_l, bwf_l
    )
    cpnf_s = calc_cpnf(
        s1vv_ts, pdf_type_s, pdf_forest_s, pdf_nonforest_s, bwf_s
    )
    lndvi_ts = lndvi_ts.to_dataset()
    s1vv_ts = s1vv_ts.to_dataset()
    lndvi_ts["cpnf_lndvi"] = (["date", "y", "x"], cpnf_l)
    s1vv_ts["cpnf_s1vv"] = (["date", "y", "x"], cpnf_s)
    outer_merge_ts = xr.merge(
        [s1vv_ts, lndvi_ts], join="outer", compat="override"
    )
    return outer_merge_ts


def deseason_ts(
    timeseries,
    percentile: float = 0.95,
    min_v: float = None,
    max_v: float = None,
):
    """Deseasonalizes the timeseries in place. This doesn't do any seasonal curve fitting,
        but instead subtracts a percentile.

    Args:
        timeseries (xr.DataArray): With dims (date, y, x)
        percentile (float, optional): the percentile to use to subtract from each raster scene. Defaults to 0.95.
        min_v ([type], optional): Truncates timeseries while deseasonalizing. Defaults to None.
        max_v ([type], optional):  Truncates timeseries while deseasonalizing. Defaults to None.

    Returns:
        [type]: Deseasoned timeseries.
    """
    if max_v:
        timeseries = timeseries.where(timeseries > max_v, np.nan, timeseries)
    if min_v:
        timeseries = timeseries.where(timeseries < min_v, np.nan, timeseries)
    # percentiles for each raster scene
    percentiles = timeseries.quantile(
        percentile, dim=("x", "y"), interpolation="nearest"
    )

    # deseasonalize
    for i in range(len(timeseries)):
        timeseries[i] = timeseries[i] - percentiles[i]

    return timeseries


def calc_cpnf(
    timeseries,
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

    if len(timeseries) > 0:
        # Gaussian pdf (mean and sd)
        if pdf_type[0] == "gaussian":
            # the probability of observing the observation given it is non-forest
            pobs_f = stats.norm.pdf(
                x=timeseries, loc=forest_dist[0], scale=forest_dist[1]
            )
        elif pdf_type[0] == "weibull":
            pobs_f = stats.weibull.pdf(
                x=timeseries, shape=forest_dist[0], scale=forest_dist[1]
            )
        else:
            raise ValueError("Must supply 'gaussian' or 'weibull' for pdf[0].")

        # Weibull pdf (shape and scale), TODO figure out why loc (the mean) wasn't supplied for this distribution
        if pdf_type[1] == "gaussian":
            pobs_nf = stats.norm.pdf(
                x=timeseries, loc=nforest_dist[0], scale=nforest_dist[1]
            )
        elif pdf_type[1] == "weibull":
            pobs_nf = stats.weibull.pdf(
                x=timeseries, shape=nforest_dist[0], scale=nforest_dist[1]
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


def bayts_update(bayts, chi: float = 0.5, cpnf_min: float = 0.5):
    """Iterates through each pixel time series to refine the conditional non-forest probability using bayesian updating.

    Returns a boolean xarray Dataset with dimensions (date, y, x) and two additional
        variables besides the bayts timeseries:
        Non-Forest-Change: where "True" means a change was flagged confirmed as Non-Forest.
            False means a change was not initially flagged, or was flagged but then
            unflagged by the iterative bayesian updating.
        Initial-Flagged-Change: where "True" indicates that the observation initially satisfied
            the cpnf_min criteria to be flagged as possible change. "False" indicates the
            observation was never considered as a possible change.

    Args:
        bayts (xr.DataArray): "bayts" time series created with create_bayts from two time series
            (vv backscatter and ndvi).
        chi (float, optional): Threshold of Pchange at which the change is confirmed. Defaults to 0.5.
        cpnf_min (float, optional): Threshold of conditional non-forest probability above which the first
            observation is flagged. Also used to check and keep posterior probabilities flagged for updating. Defaults to 0.5
    """
    assert (
        chi >= cpnf_min
    )  # chi should be greater or equal to the initial criteria
    assert chi >= 0.5  # chi should be greater than .5
    bayts.name = "bayts"
    bayts = bayts.to_dataset()
    bayts["initial_flag"] = xr.where(bayts["bayts"] > cpnf_min, True, False)
    bayts["flagged_change"] = (
        ("date", "y", "x"),
        np.full(bayts["initial_flag"].shape, False),
    )  # this is updated in the loop
    bayts["updated_bayts"] = bayts["bayts"]
    # need to probably figure out a better way to do this than iterating over each pixel ts individually
    # in a single process. 1 ts per dask process? numba? cython?
    # https://numpy.org/doc/stable/reference/arrays.nditer.html#arrays-nditer
    for y in range(len(bayts["y"])):
        for x in range(len(bayts["x"])):
            notnan_mask = bayts["bayts"].isel(y=y, x=x).notnull()
            pixel_ts = bayts.isel(y=y, x=x)
            # we need the date coords as positional integers since xarray doesn't support using
            # location based indexing to return a view instead of a copy.
            # we need views to assign updated posterior values to specific dates later.
            pixel_ts = pixel_ts.assign_coords(
                date_i=("date", list(range(0, len(pixel_ts.date))))
            )
            pixel_ts = pixel_ts.where(notnan_mask, drop=True)
            # don't update if all values are nan
            if bool(pixel_ts["updated_bayts"].isnull().all()):
                pass
            else:
                pixel_ts = update_pixel(pixel_ts, chi, cpnf_min)
                # set valid pixels to their update values. we first index by index label then by index location
                bayts["updated_bayts"].loc[pixel_ts.date][:, y, x] = pixel_ts[
                    "updated_bayts"
                ]
                if bool(pixel_ts["flagged_change"].any()):
                    bayts["flagged_change"][pixel_ts.date_i, y, x] = pixel_ts[
                        "flagged_change"
                    ].astype(
                        bool
                    )  # drop=True above makes this float, probably an xarray bug?
                # otherwise, no detected change, each obs in this time series stays flagged as False
    return bayts


def update_pixel(pixel_ts, chi, cpnf_min):
    """Modifies a single pixel view of a spatial timeseries to update the probabilities.

    Args:
        pixel_ts (xr.Dataset): An xarray Dataset with a single (date) dimension and 4 variables:
            the original time series "bayts", the initially flagged nonforest observations "initial_flag",
            the updated flaged changes "flagged_change", and the updated bayts time series "updated_bayts".
    """
    possible_nf_indices = np.argwhere(pixel_ts["initial_flag"].data)
    # for each observation, we update it starting from the observation and it's next future neighbor
    for ind in possible_nf_indices:
        for t in range(int(ind) + 1, len(pixel_ts["date"])):
            prior = pixel_ts["updated_bayts"][t - 1]
            likelihood = pixel_ts["updated_bayts"][t]
            posterior = calc_posterior(prior, likelihood)
            pixel_ts["updated_bayts"][
                t
            ] = posterior  # in the next time step, if it is reached, the posterior will be the prior
            if posterior >= chi:
                # if the previously flagged observation gets posterior computed and it is above the
                # threshold, we flag it and stop searching this time series for a high confidence
                # deforestation event (as determined by chi) deforestation event.
                pixel_ts["flagged_change"][t] = True
                return pixel_ts
            elif posterior < cpnf_min or t == len(pixel_ts["date"]):
                # if the previously flagged observation gets posterior computed and it is below the
                # threshold or if all possible updates have been made, we unflag it and go on to the
                # next possible deforested detection in the time series. Or stop if we are out of
                # possible detections
                break
            else:
                # If the posterior is greater than the cpnf_min but less than chi,
                # we need to keep searching the time series.
                pass
    return pixel_ts  # this is returned if none of the initially flagged observations were confirmed with chi


def bayts_to_date_array(bayts_result):
    """Processes result from bayts_update, returning array of dates of
        flagged change for the aoi.

    Args:
        bayts_result (xarray.Dataset): The result from bayts_update, containing the variables:
            flagged_change: a boolean array where True represents deforestation detected at confidence above chi
            updated_bayts: the confidence scores/probabilities that an observation is deforested.
            bayts: the original, initial probabilities. Unused in this function and useful for debugging.

    Returns:
        (np.array, np.array): A tuple containing an integer 2D numpy array with indices. These indices
            correspond to the second numpy array, which lists the detected dates.
    """
    date_coords = np.argwhere(bayts_result["flagged_change"].data)
    coord_df = pd.DataFrame(date_coords, columns=["date", "y", "x"])
    date_c = coord_df.date.values
    y_c = coord_df.y.values
    x_c = coord_df.x.values
    date_data = bayts_result["flagged_change"]["date"][date_c].values

    date_indices = [i for i, d in enumerate(date_data)]

    date_index_array = coo_matrix(
        (date_indices, (y_c, x_c)),
        shape=bayts_result["flagged_change"].shape[-2:],
    ).toarray()
    return date_index_array, date_data
