"""Main functions used for calculating bayts time series from multiple time series

Original author: Johannes Reiche (Wageningen University)
Implemented in python by: Development Seed
References: {http://www.mdpi.com/2072-4292/7/5/4973}{Reiche et al. (2015): A Bayesian Approach to Combine Landsat and ALOS PALSAR Time Series for Near Real-Time Deforestation Detection. Remote Sensing. 7(5), 4973-4996; doi:10.3390/rs70504973}
"""

import time
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
import xarray as xr
from scipy.sparse import coo_matrix
from tqdm import tqdm


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
    cpnf_l = calc_cpnf(lndvi_ts, pdf_type_l, pdf_forest_l, pdf_nonforest_l, bwf_l)
    cpnf_s = calc_cpnf(s1vv_ts, pdf_type_s, pdf_forest_s, pdf_nonforest_s, bwf_s)
    lndvi_ts = lndvi_ts.to_dataset()
    s1vv_ts = s1vv_ts.to_dataset()
    lndvi_ts["cpnf_lndvi"] = (["date", "y", "x"], cpnf_l)
    s1vv_ts["cpnf_s1vv"] = (["date", "y", "x"], cpnf_s)
    outer_merge_ts = xr.merge([s1vv_ts, lndvi_ts], join="outer", compat="override")
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
    percentiles = timeseries.quantile(percentile, dim=("x", "y"), interpolation="nearest")

    for i in tqdm(range(len(timeseries))):
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
            pobs_f = stats.norm.pdf(x=timeseries, loc=forest_dist[0], scale=forest_dist[1])
        elif pdf_type[0] == "weibull":
            pobs_f = stats.weibull.pdf(x=timeseries, shape=forest_dist[0], scale=forest_dist[1])
        else:
            raise ValueError("Must supply 'gaussian' or 'weibull' for pdf[0].")

        # Weibull pdf (shape and scale), TODO figure out why loc (the mean) wasn't supplied for this distribution
        if pdf_type[1] == "gaussian":
            pobs_nf = stats.norm.pdf(x=timeseries, loc=nforest_dist[0], scale=nforest_dist[1])
        elif pdf_type[1] == "weibull":
            pobs_nf = stats.weibull.pdf(x=timeseries, shape=nforest_dist[0], scale=nforest_dist[1])
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
    return (prior * likelihood) / ((prior * likelihood) + ((1 - prior) * (1 - likelihood)))


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
    refined_cpnf_two_obs = calc_posterior(timeseries["cpnf_s1vv"], timeseries["cpnf_lndvi"])
    # where we have ndvi observations, we want to use the refined cpnf since ndvi is more related to deforestation than backscatter
    timeseries["cpnf_s1vv_refined"] = xr.where(
        timeseries["cpnf_s1vv"].notnull() & timeseries["cpnf_lndvi"].notnull(),
        refined_cpnf_two_obs,
        timeseries["cpnf_s1vv"],
    )
    # where we don't have backscatter but we have ndvi, we want to use ndvi cpnf
    nan_s1vv = timeseries["cpnf_s1vv_refined"].isnull()
    bayts = xr.where(nan_s1vv, timeseries["cpnf_lndvi"], timeseries["cpnf_s1vv_refined"])
    # any nans left in the output should be from image boundary issues or quality masking
    return bayts


def subset_by_midpoint(bayts):
    """Subset the time series to 100x100 width and height

    Args:
        bayts (numpy.ndarray): The probability time series.

    Returns:
        numpy.ndarray: The subsetted timeseries.
    """
    ymid, xmid = int(bayts.shape[1] / 2), int(bayts.shape[2] / 2)
    radius = 50
    baytssubset = bayts.isel(
        y=slice(ymid - radius, ymid + radius),
        x=slice(xmid - radius, xmid + radius),
    )
    return baytssubset


def bayts_update_ufunc(
    pixel_ts: np.array, initial_flag: np.array, chi: float, cpnf_min: float
) -> Tuple:
    """Iterates through each pixel time series to refine the conditional non-forest probability using bayesian updating.

    This is meant to be vectorized with xarray.apply_ufunc. For example:

    xr.apply_ufunc(bayts_update_ufunc, baytsds['updated_bayts'], baytsds['initial_flag'], input_core_dims=[["date"],
        ["date"]], kwargs={'chi': chi, "cpnf_min":cpnf_min}, dask = 'allowed', vectorize = True)

    Args:
        pixel_ts (np.array): "bayts" time series created with create_bayts. Can be either
            vv backscatter or ndvi, or any other single pixel time series with NaNs.
        initial_flag (np.array): time series of initial flags (most False) that note observations flagged as Non-Forest.
        This gets updated to find the earliest detected Non-Forest probability.
        chi (float, optional): Threshold of Pchange at which the change is confirmed. Defaults to 0.5.
        cpnf_min (float, optional): Threshold of conditional non-forest probability above which the first
            observation is flagged. Also used to check and keep posterior probabilities flagged for updating. Defaults to 0.5


    Returns: A numpy array with dimensions (date, y, x) and two additional
        variables besides the bayts timeseries.
    """
    # don't update if all values are nan
    pixel_ts = pixel_ts.copy()
    initial_flag = initial_flag.copy()
    # don't update if all values are nan
    if np.all(np.isnan(pixel_ts)):
        return initial_flag
    else:
        pixel_ts_nonan = pixel_ts[~np.isnan(pixel_ts)]
        initial_flag_nonan = initial_flag[~np.isnan(pixel_ts)]
        flag_status = update_pixel_ufunc(pixel_ts_nonan, initial_flag_nonan, chi, cpnf_min)
        is_confirmed_flagged_change_ts = np.char.array(flag_status) == np.char.array("Confirmed")
        flagged_change_full_size = np.zeros(pixel_ts.shape, dtype=bool)
        if np.any(is_confirmed_flagged_change_ts):
            flagged_change_full_size[~np.isnan(pixel_ts)] = is_confirmed_flagged_change_ts
        return flagged_change_full_size


def update_pixel_ufunc(pixel_ts, initial_flag, chi: float, cpnf_min: float):
    """Modifies a single pixel view of a spatial timeseries to update the probabilities.

    Args:
        pixel_ts (np.array): A numpy array with a single (date) dimension containing the probabilities that are to be updated.
        initial_flag (np.array): The initially flagged nonforest observations that are updated to be False if not above chi or True if above chi.
    """
    original_pixel_ts = pixel_ts.copy()
    flagged_change = initial_flag.copy()
    possible_nf_indices = np.argwhere(initial_flag)
    # for each observation, we update it starting from the observation and it's next future neighbor
    flag_status = np.char.array(flagged_change)  # change to a list get byte to str
    true_mask = np.full_like(flag_status, "True")
    noflag_mask = np.full_like(flag_status, "NoFl")
    flagged_mask = np.full_like(flag_status, "Flag")
    flag_status = np.where(flag_status == true_mask, flagged_mask, noflag_mask)
    flag_status = [status.decode("UTF-8") for status in flag_status]
    t_plus_one_obs_used_for_updating = 0
    # ind should never be 0 or t-1 doesn't work
    for ind in possible_nf_indices:
        assert ind != 0
        for t in range(int(ind), len(pixel_ts)):
            if flag_status[t - 1] == "NoFl":
                prior = pixel_ts[t - 1]
                t_plus_one_obs_used_for_updating = 0
            elif flag_status[t - 1] == "Flag":
                prior = pixel_ts[t - 1]
                t_plus_one_obs_used_for_updating += 1
            else:
                raise ValueError(
                    f"the flag_status at t-1 should be NoFl or Flag but was {flag_status[t- 1]}"
                )
            likelihood = pixel_ts[t]
            posterior = calc_posterior(prior, likelihood)
            # in the next time step, if it is reached, the posterior will be the prior
            pixel_ts[t] = posterior
            flag_status[t] = "Flag"
            if pixel_ts[0] < 0.2:
                b = 1
            # Confirm and reject flagged changes
            if t_plus_one_obs_used_for_updating > 0 and posterior < cpnf_min:
                # if the previously flagged observation gets posterior computed and it is below the
                # threshold or if all possible updates have been made, we unflag it and go on to the
                # next possible deforested detection in the time series. Or stop if we are out of
                # possible detections
                flag_status = np.char.array(flag_status)
                flag_status[t - t_plus_one_obs_used_for_updating : t + 1] = "NoFl"
                flag_status = flag_status.tolist()
                if pixel_ts[0] < 0.2:
                    b = 1
                break
            if posterior >= chi and original_pixel_ts[t] >= cpnf_min:
                # if the previously flagged observation gets posterior computed and it is above the
                # threshold, we flag it and stop searching this time series for a high confidence
                # deforestation event (as determined by chi) deforestation event. Later, we also set all other
                # observations to False in this time series if they are not "Confirmed".
                first_date_index_flagged = flag_status.index("Flag")
                flag_status[first_date_index_flagged] = "Confirmed"
                if pixel_ts[0] < 0.2:
                    b = 1
                assert flag_status.count("Confirmed") == 1
                return flag_status
            # If the posterior is greater than the cpnf_min but less than chi,
            # we need to keep searching the time series.
    assert flag_status.count("Confirmed") == 0
    return flag_status  # this is returned if none of the initially flagged observations were confirmed with chi


def loop_bayts_update(bayts, initial_change, date_index, chi, cpnf_min, monitor_start=None):
    """Loop through pixels to update each pixel time series probabilities. Used for debugging.

    Args:
        bayts (numpy.array): A numpy array with the probabilities for the fused timeseries and dimensions (date, y, x).
        initial_change ([type]): The initially flagged nonforest observations with dimensions (date, y, x).
            Updated to be False if not above chi or True if above chi.
        date_index (numpy.array): 1 dimensional date64 array of dates for the time series.
        monitor_start (datetime, optional): A datetime threshold used to truncate the timeseries to only monitor the latter part. Defaults to None.

    Returns:
        numpy.array: Boolean array with shape (date, y, x) showing true where deforestation is detected and False where not.
            NOTE: If monitor_start is used, this needs to be assigned to an xarray DataArray with the same dimensions later
            so that dates are properly assigned to the True booleans.
    """
    flagged_change_output = initial_change.copy()
    after_monitor_start = date_index > np.datetime64(monitor_start)
    for y in tqdm(range(bayts.shape[1])):
        for x in range(bayts.shape[2]):
            pixel_ts = bayts[:, y, x]
            nanmask = np.isnan(pixel_ts)
            initial_change_ts = initial_change[:, y, x]
            # don't update if all values are nan
            if nanmask.all():
                pass
            else:
                if monitor_start:
                    # used to truncate a monitoring period to focus on latter part of timeseries. needs to happen in loop since
                    # we need to include the observation that is before and closest to the monitor start date and this
                    # can occur at a variable date
                    dates_before_monitoring = date_index[~nanmask][
                        date_index[~nanmask] < np.datetime64(monitor_start)
                    ]
                    if len(dates_before_monitoring) == 0:
                        pixel_ts = np.concatenate([0.5], pixel_ts)
                        initial_change_ts = np.concatenate([False], initial_change_ts)
                        after_monitor_start_t_minus_1 = date_index >= monitor_start_t_minus_1
                        after_monitor_start_t_minus_1 = np.concatenate(
                            [False], after_monitor_start_t_minus_1
                        )
                        after_monitor_start_t_minus_1_indices = np.where(
                            after_monitor_start_t_minus_1
                        )
                    else:
                        monitor_start_t_minus_1 = dates_before_monitoring[-2]
                        # the length varies depending on the pixel because of irregular observations and nodata gaps from masking
                        after_monitor_start_t_minus_1 = date_index > monitor_start_t_minus_1
                        after_monitor_start_t_minus_1_indices = np.where(
                            after_monitor_start_t_minus_1
                        )
                        pixel_ts = pixel_ts[after_monitor_start_t_minus_1]
                        initial_change_ts = initial_change_ts[after_monitor_start_t_minus_1]
                        # we don't consider the observation before the monitoring start date, we only use it for updating
                        initial_change_ts[0] = False
                    is_confirmed_flagged_change_ts = bayts_update_ufunc(
                        pixel_ts, initial_change_ts, chi, cpnf_min
                    )
                    # we only want the output that occurs after the monitor start
                    confirmed_date = dates_to_decimal_years(
                        date_index[date_index > monitor_start_t_minus_1][
                            is_confirmed_flagged_change_ts
                        ]
                    )
                    flagged_change_output[
                        after_monitor_start_t_minus_1_indices[0][1:], y, x
                    ] = is_confirmed_flagged_change_ts[1:]
                    if pixel_ts[0] < 0.2:
                        b = 1
                else:
                    flagged_change_ts = bayts_update_ufunc(
                        pixel_ts, initial_change_ts, chi, cpnf_min
                    )
                    flagged_change_output[:, y, x] = flagged_change_ts[after_monitor_start]
    if monitor_start:
        return flagged_change_output[after_monitor_start]
    else:
        return flagged_change_output


def bayts_da_to_date_array(flagged_change):
    """Processes result from bayts_update, returning array of dates of
        flagged change for the aoi.

    Args:
        bayts_result (xarray.DataArray): The result from bayts_update, containing the variables:
            flagged_change: a boolean array where True represents deforestation detected at confidence above chi
            updated_bayts: the confidence scores/probabilities that an observation is deforested.
            bayts: the original, initial probabilities. Unused in this function and useful for debugging.

    Returns:
        (np.array, np.array, np.array): A tuple containing an integer 2D numpy array with indices. These indices
            correspond to the second numpy array, which lists the detected dates. The "0" index maps to the 0 position.
            The third array contains the dates in units of decimal years, for easier visualization and comparison with
            the R results.
    """
    flagged_change = flagged_change["flagged_change"]
    date_coords = np.argwhere(flagged_change.data)
    coord_df = pd.DataFrame(date_coords, columns=["date", "y", "x"])
    date_c = coord_df.date.values
    y_c = coord_df.y.values
    x_c = coord_df.x.values
    date_data = flagged_change["date"][date_c].values
    # sparse.coo_matrix does not populate with NaN where there ar enot dates, it populates with 0s.
    # so we need to use 1 indexing here

    datetimes = pd.to_datetime(np.datetime_as_string(date_data))

    decimal_yrs = [to_year_fraction(pddt) for pddt in datetimes]

    date_indices = np.array([i + 1 for i, d in enumerate(date_data)])

    date_index_arr = coo_matrix(
        (date_indices, (y_c, x_c)),
        shape=flagged_change.shape[-2:],
        dtype=int,
    ).toarray()
    # convert back to 0 indexed with nans where no deforestation detected
    date_index_arr = np.where(date_index_arr == 0, np.nan, date_index_arr - 1)
    date_indices = date_indices - 1

    # return for viz and R compare. dates not exact because of leap years and drift
    decimal_yrs_arr = coo_matrix(
        (decimal_yrs, (y_c, x_c)),
        shape=flagged_change.shape[-2:],
        dtype=float,
    ).toarray()
    return (
        date_index_arr,
        date_data,
        np.where(decimal_yrs_arr == 0, np.nan, decimal_yrs_arr),
    )


def to_year_fraction(date):
    """From https://stackoverflow.com/questions/6451655/how-to-convert-python-datetime-dates-to-decimal-float-years

    Args:
        date (datetime.datetime): A datetime.datetime object.
    """

    def sinceEpoch(date):  # returns seconds since epoch
        return time.mktime(date.timetuple())

    s = sinceEpoch

    year = date.year
    startOfThisYear = datetime(year=year, month=1, day=1)
    startOfNextYear = datetime(year=year + 1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed / yearDuration

    return date.year + fraction


def dates_to_decimal_years(npdates):
    """Convert np.datetime64 type array to list of decimal years.

    Args:
        npdates (numpy.ndarray): Array of numpy dates.

    Returns:
        List: List of decimal years.
    """
    return [to_year_fraction(pd.to_datetime(date)) for date in npdates]
