from typing import List
import os
import pandas as pd
import rioxarray as rio
from rioxarray.merge import merge_arrays
import xarray as xr
from iteration_utilities import groupedby
import json
import geopandas as gpd
from geocube.api.core import make_geocube
from .qa import Collection2_QAValues, pixel_to_qa
import time
from pathlib import Path
from itertools import islice
import re
from datetime import datetime
from geocube.api.core import make_geocube
from pathlib import Path
from tqdm import tqdm


def create_two_timeseries(
    landsat_csv,
    sentinel_csv,
    sentinel_aoi_csv_path,
    geojson_p,
    aoi_EPSG,
    sentinel_in_folder,
    l8_ndvi_outfolder_dir,
    l8_merged_ndvi_outfolder_dir,
    sentinel_merged_outfolder_dir,
):
    for i in [
        l8_ndvi_outfolder_dir,
        l8_merged_ndvi_outfolder_dir,
        sentinel_merged_outfolder_dir,
    ]:
        if os.path.exists(i) == False:
            os.makedirs(i)

    sentinel_names = get_scene_paths(sentinel_csv)
    landsat_paths = get_scene_paths(landsat_csv)

    aoi_gdf = gpd.read_file(geojson_p)

    # utm epsg for brazil since landsat and sentinel in utm
    aoi_grid_ds = make_geocube(
        aoi_gdf, output_crs=aoi_EPSG, resolution=(30, 30)
    )

    liter = iter(landsat_paths)
    lp_tups = list(zip(liter, liter))
    ndvi_paths = [str(i) for i in Path(l8_ndvi_outfolder_dir).glob("*")]
    print("Computing NDVI and saving...")
    if len(ndvi_paths) == len(lp_tups):
        pass
    else:
        assert len(ndvi_paths) == 0
        start = time.time()
        for l8_scene_band_tup in lp_tups:
            ndvi_path, p_time = scene_id_to_ndvi_arr(
                l8_ndvi_outfolder_dir,
                l8_scene_band_tup[0],
                l8_scene_band_tup[1],
            )
            ndvi_paths.append(ndvi_path)
        print("Total", time.time() - start)

    ndvi_ts = group_merge_stack(
        ndvi_paths, aoi_grid_ds, l8_merged_ndvi_outfolder_dir, date_index=3
    )

    sfolders = list(Path(sentinel_in_folder).glob("*"))
    sentinel_paths = sentinel_paths_for_aoi_csv(
        sentinel_names,
        sfolders,
        sentinel_aoi_csv_path,
    )
    s1_ts = group_merge_stack(
        sentinel_paths,
        aoi_grid_ds,
        sentinel_merged_outfolder_dir,
        date_index=2,
    )

    return ndvi_ts, s1_ts


def get_scene_paths(csv_path: str):
    """Get scene paths to tif files.

    Args:
        tif_folder (str): path to the tifs
        ds (str): the dateset type. determines how to format date in filename.

    Returns:
        list of scene paths on an azure file storage container.
    """
    scenes = pd.read_csv(csv_path)
    if "GRD" in csv_path or "asf_search" in csv_path:
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
    start = time.time()
    # confirm scene ids are the same
    assert "B4" in b4_path
    assert "B5" in b5_path
    scene_id_b4 = b4_path.split("/")[-2]
    assert scene_id_b4 == b5_path.split("/")[-2]
    # attempting to take advantage of COG internal tiling
    # page 4 says its 256x256
    # https://prd-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/atoms/files/LSDS-1388-Landsat-Cloud-Optimized-GeoTIFF_DFCB-v2.0.pdf
    # https://corteva.github.io/rioxarray/stable/examples/read-locks.html
    with rio.open_rasterio(
        b4_path, masked=True, lock=False, chunks=(1, "auto", -1)
    ) as red, rio.open_rasterio(
        b5_path, masked=True, lock=False, chunks=(1, "auto", -1)
    ) as nir:
        calc_ndvi = (nir - red) / (nir + red)
        ndvi_path = os.path.join(outdir, scene_id_b4 + "_NDVI.tif")
        qa_path = (
            b4_path.split("SR_B4")[0]
            + "QA_PIXEL.TIF"
            + "?"
            + b4_path.split("?")[1]
        )
        clear_mask = get_clear_qa_mask(qa_path)
        calc_ndvi.where(clear_mask).rio.to_raster(ndvi_path)
        return ndvi_path, time.time() - start


def get_clear_qa_mask(qa_path):
    with rio.open_rasterio(qa_path, lock=False, chunks=(1, "auto", -1)) as qa:
        qa = qa.rio.set_nodata(1)
        qa = qa.sel(band=1)
        mask = pixel_to_qa(qa.values, QAValues=Collection2_QAValues)
        return mask == Collection2_QAValues.Clear.value["out"]


def groupby_date(scenes: List[str]) -> List[List]:
    """Takes a list of scenes, groups them by date.

    Each group will be merged into a single mosaic for that date if there are
    more than one scenes in the group. The file path must contain value specified in conditionals.

    Args:
        scenes (List[str]): The output from scene_id_to_ndvi_arr or path to s1 backscatter VV .tif

    Returns:
        List[List]: A list of lists, where each inner list contains the paths occurring at the same date and aoi.
    """
    if "landsateuwest" in scenes[0] or "l8processed" in scenes[0]:
        group_dict = groupedby(
            scenes, key=lambda x: x.split("/")[-1].split("_")[3]
        )
    elif "RTC30" in scenes[0]:
        group_dict = groupedby(
            scenes, key=lambda x: x.split("/")[-1].split("_")[2][0:8]
        )
    elif "NDVI" in scenes[0]:
        group_dict = groupedby(
            scenes, key=lambda x: x.split("/")[-1].split("_")[3]
        )
    else:
        raise ValueError(
            "The scene paths do not contain a Landsat or Sentinel-1 identifier sub string."
        )
    return group_dict


def open_geojson(geojson_path):
    with open(geojson_path) as f:
        feature = json.load(f)
    return feature


def make_aoi_grid(geojson_path: str, epsg: str = "EPSG:32722"):
    """Uses geocube package to turn a geojson into a raster grid.

    Timeseries rasters are later reprojected, snapped, and clipped to this grid.

    Args:
        geojson_path (str):
        epsg (str, optional): [description]. Defaults to "EPSG:32722".

    Returns:
        [type]: [description]
    """
    print(f"Making aoi grid with crs {epsg}")
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
    merged = merge_arrays(arrs)
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
    print("Merging same date scenes, if any...")
    for date, group in date_group_dict.items():
        fname = group[0].split("/")[-1]
        outpath = os.path.join(outfolder, fname)
        if os.path.exists(outpath):
            pass
        else:
            if len(group) > 1:
                print("Date group: ", group)
                merged_date_arr = open_merge_per_date(group)
            else:
                merged_date_arr = rio.open_rasterio(group[0])
            merged_date_arr = reproject_match_to_aoi(merged_date_arr, aoi_grid)
            # day is exact, other info plike path row in the name is not since it's been merged
            merged_date_arr.name = group[0].split("/")[-1]
            merged_date_arr.rio.to_raster(outpath)
    return list(Path(outfolder).glob("*"))


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


def get_SAFE_id(logfile: str) -> str:
    """Parses the ASF log file to get the SAFE formatted file name for matching with aoi granule names.

    Args:
        logfile (str): Path to log file.

    Returns:
        str: The file id (without .SAFE extension)
    """
    with open(logfile) as myfile:
        content = list(islice(myfile, 2))[1]
    return (
        re.search(
            r"SAFE\s+directory(?:(?!\.SAFE)(?:.|\n))*\.SAFE",
            content,
            re.DOTALL,
        )
        .group()
        .split(": ")[-1]
        .split(".SAFE")[0]
    )


def sentinel_paths_for_aoi_csv(
    sentinel_names_csv: List,
    sentinel_folders_azure: List,
    path_file_name: str,
) -> List[str]:
    """Gets the Sentinel paths on Azure given a granule list in csv format.

    The log files on azure need to be parsed to get matched to the granule ids in the granule list.
    The script can take a bit the first time it is run since parsing log file takes some seconds.

    Args:
        sentinel_names_csv (List): Granule ids in csv. This can be the csv used to order the data for an aoi with ASF vertex.
        sentinel_folders_azure (List): The list of all folders on Azure.
        path_file_name (str): Name of csv file to save out the paths to vv file son azure that correspond to a csv/aoi.

    Returns:
        List[str]: List of VV.tif file paths.
    """
    print("Searching for correct Sentinel-1 paths...")
    if os.path.exists(path_file_name):
        print(f"Using existing paths csv {path_file_name}")
        vvpdf = pd.read_csv(path_file_name)
        return vvpdf["0"].tolist()
    else:
        vvpaths = []
        for azure_folder_path in tqdm(sentinel_folders_azure):
            logfile_path = list(azure_folder_path.glob("*.log"))[0]
            safe_id_azure_log = get_SAFE_id(logfile_path)
            for safe_id_csv in sentinel_names_csv:
                if safe_id_azure_log == safe_id_csv:
                    vvpath = os.path.join(
                        azure_folder_path,
                        str(azure_folder_path).split("/")[-1] + "_VV.tif",
                    )
                    vvpaths.append(vvpath)
        pd.DataFrame(vvpaths).to_csv(path_file_name)
        return vvpaths


def group_merge_stack(paths, aoi_grid_ds, merged_outfolder_dir, date_index=3):
    date_groups = groupby_date(paths)

    merged_paths = merge_each_date(
        date_groups, aoi_grid_ds, merged_outfolder_dir
    )
    print("Final step, opening and stacking time series...")
    arrs = []
    for p in merged_paths:
        arr = rio.open_rasterio(str(p), lock=False, chunks=(1, "auto", -1))
        datestr = str(p).split("/")[-1].split("_")[date_index]
        dt = datetime(int(datestr[0:4]), int(datestr[4:6]), int(datestr[6:8]))
        arr["date"] = dt
        arrs.append(arr.sel({"band": 1}))
    ts = xr.concat(arrs, dim="date")
    return ts.sortby("date")
