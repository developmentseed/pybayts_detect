from typing import List
import os
import pandas as pd
import rioxarray as rio
import xarray as xr
from iteration_utilities import groupedby
import numpy as np
import rasterio
import json
import geopandas as gpd
from geocube.api.core import make_geocube


def get_scene_paths(csv_path: str):
    """Get scene paths to tif files.

    Args:
        tif_folder (str): path to the tifs
        ds (str): the dateset type. determines how to format date in filename.

    Returns:
        list of scene paths on an azure file storage container.
    """
    scenes = pd.read_csv(csv_path)
    if "GRD" in csv_path:
        scenes = scenes["Granule Name"].tolist()
    elif "ls8" in csv_path:
        scenes = scenes["azure_path"].tolist()
    return scenes


def scene_id_to_ndvi_arr(outdir: str, b4_path: str, b5_path: str) -> str:
    """Function to save NDVI for a Landsat 8 scene.

    NDVI = (Red - NIR) / (Red + NIR)

    Args:
        outdir (str): The path to the dir to save landsat scene folders with ndvi.
        landsat_scene_id (str): The Landsat scene id / folder name

    Returns:
        str: The path to the NDVI .tif
    """
    # confirm scene ids are the same
    assert "B4" in b4_path
    assert "B5" in b5_path
    scene_id_b4 = b4_path.split("/")[-2]
    assert scene_id_b4 == b5_path.split("/")[-2]
    red = rio.open_rasterio(b4_path, masked=True)  # red
    nir = rio.open_rasterio(b5_path, masked=True)  # near infrared

    calc_ndvi = (nir - red) / (nir + red)
    ndvi_path = os.path.join(outdir, scene_id_b4 + "_NDVI.tif")

    calc_ndvi.to_raster(ndvi_path)

    return ndvi_path


def groupby_date(scenes: List[str]) -> List[List]:
    """Takes a list of scenes, groups them by date.

    Each group will be merged into a single mosaic for that date if there are
    more than one scenes in the group.

    Args:
        scenes (List[str]): The output from scene_id_to_ndvi_arr or path to s1 backscatter VV .tif

    Returns:
        List[List]: A list of lists, where each inner list contains the paths occurring at the same date and aoi.
    """
    if "landsateuwest" in scenes[0]:
        group_dict = groupedby(scenes, key=lambda x: x[111:119])
    elif "IW_GRD" in scenes[0]:
        group_dict = groupedby(scenes, key=lambda x: x[17:25])
    return group_dict


def open_geojson(geojson_path):
    with open(geojson_path) as f:
        feature = json.load(f)
    return feature


def make_aoi_grid(geojson_path: str, epsg: str = "EPSG:32722"):
    aoi_gdf = gpd.read_file(geojson_path)
    # utm epsg for brazil since landsat and sentinel in utm
    aoi_grid_ds = make_geocube(aoi_gdf, output_crs=epsg, resolution=(30, 30))
    return aoi_grid_ds


def open_merge_per_date(paths: List[str]):
    """Opens each path in a list of paths and merges with rioxarray.

    Args:
        paths (List[str]): List of paths for the same date.

    Returns:
        xarray.DataArray: A DataArray with geospatial metadata in .rio.
    """
    arrs = []
    for p in paths:
        arr = rio.open_rasterio(p)
        arrs.append(arr)
    name = arr.name
    merged = rio.merge.merge_arrays(arrs)
    merged.name = "merged_" + name
    return merged


def merge_each_date(date_group_dict, aoi_grid, outfolder):
    """Takes a group dict from groupby_date and iterates through to merge and save rasters.

    Also aligns the output array to the aoi grid so the results can be stacked into a time series.
    Args:
        date_group_dict (dict): A dictionary with keys as str dates and values a tuple of paths.
        outfolder (str): Where to save the merged rasters. Full path to the folder.

    Returns:
        str: The path to the merged rasters, or just the single raster if there is only one for
         that particular date.
    """
    for date, group in date_group_dict.items():
        if len(group) > 1:
            merged_date_arr = open_merge_per_date(group)
        else:
            merged_date_arr = rio.open_rasterio(group[0])
            merged_date_arr.name = group[0]
        outpath = os.path.join(outfolder, merged_date_arr.name)
        merged_date_arr = reproject_match_to_aoi(merged_date_arr, aoi_grid)
        merged_date_arr.to_raster(outpath)
        return outpath


def reproject_match_to_aoi(da, match_da):
    """Reprojects Sentinel-1 or Landsat to 30,30 meter resolution and snaps to a common grid.

    Also makes all arrays same extent, with NaNs for no observations.

    Args:
        da (xr.DataArray): The xr.DataArray for a single date.
        match_da (xr.Dataset): The aoi grid. Can be created with
            make_geocube(aoi_gdf, output_crs="EPSG:32722", resolution=(30,30)).
            Make sure to use the correct EPSG for the AOI.

    Returns:
        xr.DataArray: The processed xr.DataArray, ready to be stacked into a time series.
    """

    return da.rio.reproject_match(match_da)
