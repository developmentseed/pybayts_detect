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


def mean_std_timeseries(directory, geojson):
    masked_list = []
    for f in directory:
        op = rasterio.open(f)
        arr = op.read()
        masked = rasterio.features.geometry_mask(geojson, arr.shape, op.transform, all_touched=True, invert=False)
        masked = masked.transpose(1,2,0)
        masked_list.append(masked)
    masked_stack = np.dstack(masked_list)
    masked_stack = masked_stack.transpose(2,0,1)
    masked_stack_out = rasterio.open(f"{directory}/masked_stack.tiff", 'w', driver='Gtiff',
                              width=op.width, height=op.height,
                              count=range(len(masked_list)),
                              crs=op.crs,
                              transform=op.transform,
                              dtype=op.dtype)

    masked_stack_out.write(masked_stack)
    masked_stack_out.close()
    arr = rio.open_rasterio(f"{directory}/masked_stack.tiff")
    arr.rio.reproject("EPSG:4326")
    arr.rio.to_raster(os.path.join(dir_in, f[:-4]+"_4326.tif"))
    arr_mask_nodata = np.ma.masked_array(arr, arr == 0)
    mean = np.mean(arr_mask_nodata) 
    std = np.std(arr_mask_nodata)   
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


