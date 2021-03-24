from typing import List
import os
import pandas as pd
import rioxarray as rio
import xarray as xr
from iteration_utilities import groupedby
import numpy as np
import rasterio


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
        scenes["Date"] = scenes["Granule Name"].apply(lambda x: x[17:25])
    elif "ls8" in csv_path:
        scenes["Date"] = scenes["azure_path"].apply(lambda x: x[111:119])
    # dates_lst = scenes.Date.tolist()
    return scenes


def scene_id_to_ndvi_arr(root_dir: str, landsat_scene_id: str) -> str:
    """Function to save NDVI for a Landsat 8 scene.

    NDVI = (Red - NIR) / (Red + NIR)

    Args:
        root_dir (str): The path to the dir containing landsat scene folders.
        landsat_scene_id (str): The Landsat scene id / folder name

    Returns:
        str: The path to the NDVI .tif
    """

    landsat_dir = os.path.join(root_dir, landsat_scene_id)

    if not os.path.exists(landsat_dir):
        os.makedirs(landsat_dir)

    src_4 = rasterio.open(
        os.path.join(root_dir, landsat_scene_id + "_B4.TIF")
    )  # red
    src_5 = rasterio.open(
        os.path.join(root_dir, landsat_scene_id + "_B5.TIF")
    )  # near infrared

    meta = src_5.meta

    # arrays
    red = src_4.read()
    nir = src_5.read()

    calc_ndvi = np.where((nir + red) == 0.0, 0, (nir - red) / (nir + red))
    ndvi_path = os.path.join(landsat_dir, landsat_scene_id + "_NDVI.tif")

    with rasterio.open(ndvi_path, "w", **meta) as dst:
        dst.write(calc_ndvi.astype(rasterio.uint16))

    return ndvi_path


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


def merge_each_date(date_group_dict, outfolder):
    """Takes a group dict from groupby_date and iterates through to merge and save rasters.

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
    merged_date_arr.to_raster(outpath)
    return outpath
