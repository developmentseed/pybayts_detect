""" Funcs for reading data. """
from datetime import datetime
from pathlib import Path

import pandas as pd
import rioxarray as rio
import xarray as xr


def read_and_stack_tifs(tif_folder: str, ds: str):
    """Reads the multiple tif files comprising the sentinel-1 vh time series.

    These multiple tifs were converted from the .rda object store named s1vh_riau_raster.rda.
    This func assumes a file pattern like S1_20170703T225539.tif.

    Args:
        tif_folder (str): path to the tifs
        ds (str): the dateset type. determines how to format date in filename.

    Returns:
        xarray.DataArray: An xarray DataArray sorted by date. rioxarray spatial metadata is stored in .rio attribute.
    """
    folder = Path(tif_folder)
    files = folder.glob("*.tif")
    arrs = []
    names = []
    for i in files:
        arr = rio.open_rasterio(i, masked=True)
        if ds == "vh":
            date = i.name[3:11]
            dt = datetime(int(date[0:4]), int(date[4:6]), int(date[6:8]))
        elif ds == "vv":
            date = i.name[0:10]
            year, month, day = date.split("-")
            dt = datetime(int(year), int(month), int(day))
        elif ds == "lndvi":
            date = i.name[0:10]
            year, month, day = date.split("-")
            dt = datetime(int(year), int(month), int(day))
        names.append(dt)

        arr["date"] = dt
        arr["name"] = i.name

        arrs.append(arr.sel({"band": 1}))
    ts = xr.concat(arrs, dim="date")
    return ts.sortby("date")


def read_single_pixel_csv(csv_path: str, date_column: str):
    """Read test pixel time series csv for bolivia.

    Args:
        csv_path (str): Path to csv.
        date_column (str): Name of date column within the csv.

    Returns:
        pd.DataFrame: A regular time series dataframe with a date column, date index, and observations.
    """
    df = pd.read_csv(
        csv_path,
        index_col=date_column,
        parse_dates=True,
    )
    df["dates"] = df.index.format()
    df = df.asfreq("1D")
    return df
