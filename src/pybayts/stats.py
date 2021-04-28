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
    ndvi_stats_forest = [] 
    ndvi_stats_nonforest = []    
    
    def get_stats_dict(dir, geojson):
        dist_list = []
        for f in dir:
            arr = rio.open_rasterio(f)
            arr.rio.reproject("EPSG:4326")
            arr.rio.to_raster(os.path.join(dir, f[:-4]+"_4326.tif"))
            dist = zonal_stats(geojson, os.path.join(dir, f[:-4]+"_4326.tif"),
            stats="min mean max std count")
            dist_list.append(dist)
            return dist_list
        
    sar_dist_forest = get_stats_dict(merged_dir_sar, sar_subset_forest)
    sar_dist_nonforest = get_stats_dict(merged_dir_sar, sar_subset_nonforest)
    ndvi_stats_forest = get_stats_dict(merged_dir_ndvi, ndvi_subset_forest)
    ndvi_stats_nonforest = get_stats_dict(merged_dir_ndvi, ndvi_subset_nonforest)

    
    def get_averages(dist_list):
        mean_list = []
        std_list = []
        for stats_ in dist_list:
            stats_dict = stats_[0]
            print(stats_dict)
            stats_list = []
            for key,value in stats_dict.items() :
                stats_list.append(value)
            mean, std = stats_list[2], stats_list[3]
            mean_list.append(mean)
            std_list.append(std)
        return mean_list, std_list

    sar_stats_forest_mean_list, sar_stats_forest_std_list = get_averages(sar_dist_forest)
    sar_stats_forest = [statistics.mean(sar_stats_forest_mean_list), statistics.mean(sar_stats_forest_std_list)]
    
    sar_stats_nonforest_mean_list, sar_stats_nonforest_std_list = get_averages(sar_dist_nonforest)
    sar_stats_nonforest = [statistics.mean(sar_stats_nonforest_mean_list), statistics.mean(sar_stats_nonforest_std_list)]
    
    ndvi_stats_forest_mean_list, ndvi_stats_forest_std_list = get_averages(ndvi_stats_forest)
    ndvi_stats_forest = [statistics.mean(ndvi_stats_forest_mean_list), statistics.mean(ndvi_stats_forest_std_list)]
    
    ndvi_stats_nonforest_mean_list, ndvi_stats_nonforest_std_list = get_averages(ndvi_stats_nonforest)
    ndvi_stats_forest = [statistics.mean(ndvi_stats_nonforest_mean_list), statistics.mean(ndvi_stats_nonforest_std_list)]

    return sar_stats_forest, sar_stats_nonforest, ndvi_stats_forest, ndvi_stats_nonforest
