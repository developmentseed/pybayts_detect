"""
Module to compute forest and non-forest distributions from the SAR and NDVI data in our study AOIs.
Author: @developmentseed
"""

import os, json
import rasterio
import numpy as np
import rasterstats
from rasterstats import zonal_stats
import rioxarray as rx
import xarray as xr
import statistics

from pybayts.data.stack import group_merge_stack


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
    sar_stats_forest = [] 
    sar_stats_nonforest = []
    for sar in merged_dir_sar:
        sarr = rio.open_rasterio(sar)
        sarr.rio.reproject("EPSG:4326")
        sarr.rio.to_raster(os.path.join(merged_dir_sar, sar[:-4]+"_4326.tif"))
        forest_distr_sar = zonal_stats(sar_subset_forest, os.path.join(merged_dir_sar, sar[:-4]+"_4326.tif"),
            stats="min mean max std count")
        sar_stats_forest.append(forest_distr_sar)
        nonforest_distr_sar = zonal_stats(sar_subset_nonforest, os.path.join(merged_dir_sar, sar[:-4]+"_4326.tif"),
            stats="min mean max std count")   
        sar_stats_nonforest.append(nonforest_distr_sar)
        

    ndvi_stats_forest = [] 
    ndvi_stats_nonforest = []
    for ndvi in merged_dir_ndvi:
        ndvir = rio.open_rasterio(ndvi)
        ndvir.rio.reproject("EPSG:4326")
        ndvir.rio.to_raster(os.path.join(merged_dir_ndvi, ndvi[:-4]+"_4326.tif"))
        forest_distr_ndvi = zonal_stats(ndvi_subset_forest, os.path.join(merged_dir_ndvi, ndvi[:-4]+"_4326.tif"),
            stats="min mean max std count")
        ndvi_stats_forest.append(forest_distr_ndvi)
        nonforest_distr_ndvi = zonal_stats(ndvi_subset_nonforest, os.path.join(merged_dir_ndvi, ndvi[:-4]+"_4326.tif"),
            stats="min mean max std count")   
        ndvi_stats_nonforest.append(nonforest_distr_ndvi)
    
    sar_stats_forest_mean_list = []
    sar_stats_forest_std_list = []
    for stats_ in sar_stats_forest:
        stats_dict = stats_[0]
        print(stats_dict)
        stats_list = []
        for key,value in stats_dict.items() :
            stats_list.append(value)
        mean, std = stats_list[2], stats_list[3]
        sar_stats_forest_mean_list.append(mean)
        sar_stats_forest_std_list.append(std)
    sar_stats_forest = [statistics.mean(sar_stats_forest_mean), statistics.mean(sar_stats_forest_std)]
        
    sar_stats_nonforest_mean_list = []
    sar_stats_nonforest_std_list = []
    for stats_ in sar_stats_nonforest:
        stats_dict = stats_[0]
        print(stats_dict)
        stats_list = []
        for key,value in stats_dict.items() :
            stats_list.append(value)
        mean, std = stats_list[2], stats_list[3]
        sar_stats_nonforest_mean_list.append(mean)
        sar_stats_nonforest_std_list.append(std)
    sar_stats_nonforest = [statistics.mean(sar_stats_nonforest_mean), statistics.mean(sar_stats_nonforest_std)]

    ndvi_stats_forest_mean_list = []
    ndvi_stats_forest_std_list = []
    for stats_ in ndvi_stats_forest:
        stats_dict = stats_[0]
        print(stats_dict)
        stats_list = []
        for key,value in stats_dict.items() :
            stats_list.append(value)
        mean, std = stats_list[2], stats_list[3]
        ndvi_stats_forest_mean_list.append(mean)
        ndvi_stats_forest_std_list.append(std)
    ndvi_stats_forest = [statistics.mean(ndvi_stats_forest_mean), statistics.mean(ndvi_stats_forest_std)]
        
    ndvi_stats_nonforest_mean_list = []
    ndvi_stats_nonforest_std_list = []
    for stats_ in ndvi_stats_nonforest:
        stats_dict = stats_[0]
        print(stats_dict)
        stats_list = []
        for key,value in stats_dict.items() :
            stats_list.append(value)
        mean, std = stats_list[2], stats_list[3]
        ndvi_stats_nonforest_mean_list.append(mean)
        ndvi_stats_nonforest_std_list.append(std)
    ndvi_stats_nonforest = [statistics.mean(ndvi_stats_nonforest_mean), statistics.mean(ndvi_stats_nonforest_std)]

    return sar_stats_forest, sar_stats_nonforest, ndvi_stats_forest, ndvi_stats_nonforest
