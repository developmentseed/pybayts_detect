"""
Module to compute forest and non-forest distributions from the SAR and NDVI data in our study AOIs.
Author: @developmentseed
"""

import os, json
import rasterio
import numpy as np
import math
import rasterstats
from rasterstats import zonal_stats
import rioxarray as rx
import xarray as xr
import statistics

from pybayts.data.stack import group_merge_stack

def variance(arr):
    n = len(arr)
    mean = sum(arr) / n
    deviations = [(x - mean) ** 2 for x in arr]
    variance = sum(deviations) / n
    return variance

def stdev(arr):
    var = variance(arr)
    std_dev = math.sqrt(var)
    return std_dev

def mean_std_timeseries(directory, geojson):
    # define list to hold images in a time series
    masked_list = []
    for f in directory:
        # open image in time series
        op = rasterio.open(f)
        # read each image in time series
        arr = op.read()
        # apply nodata mask to all pixels outside of sampling polygon
        masked = rasterio.features.geometry_mask(geojson, arr.shape, op.transform, all_touched=True, invert=False)
        # convert shape from c,w,h to w,h,c
        masked = masked.transpose(1,2,0)
        # append masked image to time series list
        masked_list.append(masked)
    # concatenate the time series list into a multi-channel image
    masked_stack = np.dstack(masked_list)
    arr = masked_stack.copy()
    # flatten the masked time series array
    arr_flat = arr.flatten()
    # remove all nodata from the flattened array
    arr_mask_nodata = arr_flat[arr_flat != 0]
    # compute the mean and std
    mean = np.mean(arr_mask_nodata) 
    std = np.std(arr_mask_nodata) 
    truemean = sum(arr_mask_nodata) / float(len(arr_mask_nodata))
    truestd = stdev(arr_mask_nodata)
    assert mean == truemean
    assert std == truestd
    return mean, std



def compute_stats(merged_dir_sar, merged_dir_ndvi, sar_subset_forest, sar_subset_nonforest, ndvi_subset_forest, ndvi_subset_nonforest):
    """Compute forest and non-forest distributions for SAR and NDVI time series.
    Args:
        merged_dir_sar (str): path to single day SAR mosaics

        merged_dir_ndvi (str): path to single day SAR mosaics

        sar_subset_forest (str): path to geojson for sample forested area in SAR imagery
        
        sar_subset_nonforest (str): path to geojson for sample non-forested area in SAR imagery
        
        ndvi_subset_forest (str): path to geojson for sample forested area in NDVI imagery
        
        ndvi_subset_nonforest (str): path to geojson for sample non-forested area in NDVI imagery

    Returns:
        Average forest and non-forest mean and standard deviations from the SAR and NDVI time series.
    """   
      
    arg_tuples = zip([merged_dir_sar, merged_dir_sar, merged_dir_ndvi, merged_dir_ndvi], [sar_subset_forest,sar_subset_nonforest,
                                                                                        ndvi_subset_forest,ndvi_subset_nonforest])
    mean_std_tups = []
    for directory, geojson in arg_tuples:
        mean, std = mean_std_timeseries(directory, geojson)
        mean_std_tup = [mean, std]
        mean_std_tups.append(mean_std_tup)

    return mean_std_tups


