"""
Module to evaluate Bayts algorithm against reference data in our study AOIs.
Author: @developmentseed
"""

from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import rioxarray as rx
import xarray as xr
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from pybayts.bayts import bayts_da_to_date_array
from pybayts.bayts import create_bayts_ts
from pybayts.bayts import deseason_ts
from pybayts.bayts import loop_bayts_update
from pybayts.bayts import merge_cpnf_tseries
from pybayts.bayts import subset_by_midpoint
from pybayts.bayts import to_year_fraction
from pybayts.data.stack import create_two_timeseries

from pybayts.plot import plot_cm


def generate_cm(flat_groundtruth, flat_decimal_yr_arr, year):
    """Compute confusion matrix using co-registered reference and bayts inference data.
    Args:
        flat_groundtruth (array): flattened array of boolean values where [True = change, False = no change]
        for a given year

        flat_decimal_yr_arr (array): flattened array of boolean values where [True = change, False = no change]
        for a given year

        year (integer): the year associated with the flat_groundtruth and flat_decimal_yr_arr arrays.

    Returns:
        A confusion matrix for a single year.
    """

    labels = [False, True]
    cm = confusion_matrix(flat_groundtruth, flat_decimal_yr_arr, labels=labels)
    print(f"Confusion matrix for {year}:", cm)
    return cm


def generate_f1(flat_groundtruth, flat_decimal_yr_arr, year):
    """Compute F1 score using co-registered reference and bayts inference data.
    Args:
        flat_groundtruth (array): flattened array of boolean values where [True = change, False = no change]
        for a given year

        flat_decimal_yr_arr (array): flattened array of boolean values where [True = change, False = no change]
        for a given year

        year (integer): the year associated with the flat_groundtruth and flat_decimal_yr_arr arrays.

    Returns:
        The F1 score for a single year.
    """

    f1 = f1_score(flat_groundtruth, flat_decimal_yr_arr, average="macro")
    print(f"F1 score for {year}:", f1)
    return f1


def evaluate(groundtruth, decimal_yr_arr, aoi_name, figdir, sub_category_ref=None):
    """Evaluate co-registered reference and bayts inference data using confusion matrix and F1 score.
    Args:
        groundtruth (xarray): rioxarray of clipped reference image (may need to run reproject match against a sample time series mosaic for the AOI)

        decimal_yr_arr (array): array of boolean values where [True = change, False = no change]

        aio_name (str): the area of interest to evaluate, which can be one of ['Brazil', 'DRC', 'Indonesia'].

        sub_category_ref (str): ["terrabrasilia", "gfw"]

    Returns:
        Printed confusion matrices and F1 scores for each year in the study period.
    """

    decimal_yr_arr_years = np.unique(decimal_yr_arr.astype(np.uint16))

    if aoi_name == "Brazil":

        if sub_category_ref == "terrabrasilia":
            # terrabrasilia ground truth years
            class_year_dict = {
                0: 2008,
                1: 2009,
                2: 2010,
                3: 2011,
                4: 2012,
                5: 2013,
                6: 2014,
                7: 2015,
                8: 2016,
                9: 2017,
                10: 2018,
                11: 2019,
            }
        elif sub_category_ref == "gfw":
            # Global Forest Watch ground truth years
            class_year_dict = {
                0: 2000,
                1: 2001,
                2: 2002,
                3: 2003,
                4: 2004,
                5: 2005,
                6: 2006,
                7: 2007,
                8: 2008,
                9: 2009,
                10: 2010,
                11: 2011,
                12: 2012,
                13: 2013,
                14: 2014,
                15: 2015,
                16: 2016,
                17: 2017,
                18: 2018,
                19: 2019,
            }
        elif sub_category_ref is None:
            pass
        else:
            raise ValueError(
                f"sub_category_ref should be None, 'gfw', or 'terrabrasilia, not {sub_category_ref}'"
            )

    else:
        # aoi is either DRC or Indonesia
        # Global Forest Watch ground truth years
        class_year_dict = {
            0: 2000,
            1: 2001,
            2: 2002,
            3: 2003,
            4: 2004,
            5: 2005,
            6: 2006,
            7: 2007,
            8: 2008,
            9: 2009,
            10: 2010,
            11: 2011,
            12: 2012,
            13: 2013,
            14: 2014,
            15: 2015,
            16: 2016,
            17: 2017,
            18: 2018,
            19: 2019,
        }

    year_gt_list = []
    year_pr_list = []

    for cl in class_year_dict:
        year_gt = class_year_dict[cl]
        year_gt_list.append(year_gt)

    for yr in decimal_yr_arr_years.tolist():
        year_pr_list.append(yr)

    match_years = set(year_gt_list) & set(year_pr_list)

    print("year_gt_list, year_pr_list: ", year_gt_list, year_pr_list)
    print("match_years: ", match_years)

    f1s_dict = {}

    for year in match_years:

        cl = list(class_year_dict.keys())[list(class_year_dict.values()).index(year)]

        groundtruth_arr = groundtruth.copy()
        groundtruth_arr = groundtruth_arr == cl

        groundtruth_flat = groundtruth_arr.values.flatten()
        decimal_yr_arr_flat = decimal_yr_arr.astype(np.uint16).flatten() == year

        cm = generate_cm(groundtruth_flat, decimal_yr_arr_flat, year)
        f1 = generate_f1(groundtruth_flat, decimal_yr_arr_flat, year)
        print(f"For year {year}, the f1 score is {f1}")
        f1s_dict[year] = f1

        plot_cm(cm, aoi_name, year, figdir)

    return f1s_dict


def run_bayts_and_evaluate(
    landsat_csv: str,
    sentinel_csv: str,
    sentinel_aoi_csv_path: str,
    geojson_p: str,
    aoi_EPSG: str,
    sentinel_in_folder: str,
    l8_ndvi_outfolder_dir: str,
    l8_merged_ndvi_outfolder_dir: str,
    sentinel_merged_outfolder_dir: str,
    eval_path: str,
    pdf_type_l: Tuple,
    pdf_type_s: Tuple,
    pdf_forest_l: Tuple,
    pdf_nonforest_l: Tuple,
    pdf_forest_s: Tuple,
    pdf_nonforest_s: Tuple,
    bwf_l: Tuple,
    bwf_s: Tuple,
    chi: float = 0.9,
    cpnf_min: float = 0.5,
    aoi_name: str = "Brazil",
    sub_cat: str = None,
):
    """Runs bayts given all inputs and parameters and evaluates with specified groundtruth or comparison data.

    Also saves confusion matrix figs in a local directory.

    Args:
        landsat_csv (str): CSV with paths to scenes on Azure West Europe, including SAS token.
        sentinel_csv (str): CSV with GRD IDs from an ASF Vertex search, with the Full .SAFE ID.
        sentinel_aoi_csv_path (str): Path to save out a csv with actual paths to sentinel scenes
            that are stored on Azure filestorage, since the ASF .SAFE IDs do not directly correspond
            to the filenames that are downloaded.
        geojson_p (str): Path to geojson aoi.
        aoi_EPSG (str): The aoi projection (UTM). Provide as "EPSG:<the epsg code>"
        sentinel_in_folder (str): The folder containing the sentinel scenes on azure.
        l8_ndvi_outfolder_dir (str): Folder on azure file storage to save Landsat NDVI
        l8_merged_ndvi_outfolder_dir (str): Folder to save merged day scenes of NDVI.
        sentinel_merged_outfolder_dir (str): Folder to save merged day scenes of Sentinel-1 VV backscatter.
        eval_path (str): The path to the evaluation tif. TODO include funcs to create this from vectors in pybayts and document how this should be formatted.
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
        chi (float, optional): Threshold of Pchange at which the change is confirmed. Defaults to 0.9.
        cpnf_min (float, optional): Threshold of conditional non-forest probability above which the first
            observation is flagged. Also used to check and keep posterior probabilities flagged for updating. Defaults to 0.5.
        aio_name (str, optional): the area of interest to evaluate, which can be one of ['Brazil', 'DRC', 'Indonesia'].
        sub_category_ref (str, optional): ["terrabrasilia", "gfw"]

    Returns:
        [type]: [description]
    """

    ndvi_ts, s1_ts = create_two_timeseries(
        landsat_csv,
        sentinel_csv,
        sentinel_aoi_csv_path,
        geojson_p,
        aoi_EPSG,
        sentinel_in_folder,
        l8_ndvi_outfolder_dir,
        l8_merged_ndvi_outfolder_dir,
        sentinel_merged_outfolder_dir,
    )

    s1_ts.name = "s1"

    ndvi_ts.name = "ndvi"

    _ = deseason_ts(s1_ts.load())  # required to load because of percentile math
    _ = deseason_ts(ndvi_ts.load())

    cpnf_ts = merge_cpnf_tseries(
        s1_ts,
        ndvi_ts,
        pdf_type_l,
        pdf_type_s,
        pdf_forest_l,
        pdf_nonforest_l,
        pdf_forest_s,
        pdf_nonforest_s,
        bwf_l,
        bwf_s,
    )

    bayts = create_bayts_ts(cpnf_ts)
    bayts = subset_by_midpoint(bayts)

    initial_change = xr.where(bayts >= 0.5, True, False)
    # for R compare
    decimal_years = [to_year_fraction(pd.to_datetime(date)) for date in bayts.date.values]
    monitor_start = datetime(2016, 1, 1)
    flagged_change = loop_bayts_update(
        bayts.data,
        initial_change.data,
        initial_change.date.values,
        chi,
        cpnf_min,
        monitor_start,
    )
    bayts.name = "bayts"
    baytsds = bayts.to_dataset()
    baytsds = baytsds.sel(date=slice(monitor_start, None))
    # Need a dataset for the date coordinates
    baytsds["flagged_change"] = (("date", "y", "x"), flagged_change)

    date_index_arr, actual_dates, decimal_yr_arr = bayts_da_to_date_array(baytsds)

    print(f"decimal_yr_arr: {decimal_yr_arr}")

    tb = rx.open_rasterio(eval_path)

    groundtruth_arr_repr_match = tb.rio.reproject_match(subset_by_midpoint(ndvi_ts))
    groundtruth_arr_repr_match = groundtruth_arr_repr_match.squeeze()

    print(
        "checking for matching width and height between tseries and reference data: ",
        groundtruth_arr_repr_match.shape,
        decimal_yr_arr.shape,
    )

    f1scores = evaluate(groundtruth_arr_repr_match, decimal_yr_arr, aoi_name, "./figs", sub_cat)
    return f1scores, decimal_yr_arr, groundtruth_arr_repr_match
