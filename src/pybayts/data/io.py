""" Funcs for reading data. """
from datetime import datetime
from pathlib import Path

import rioxarray as rio
import xarray as xr


def read_and_stack_tifs(tif_folder: str):
    """Reads the multiple tif files comprising the sentinel-1 vh time series.

    These multiple tifs were converted from the .rda object store named s1vh_riau_raster.rda.
    This func assumes a file pattern like S1_20170703T225539.tif.

    Args:
        tif_folder (str): [description]

    Returns:
        xarray.DataArray: An xarray DataArray sorted by date. rioxarray spatial metadata is stored in .rio attribute.
    """
    folder = Path(tif_folder)
    files = folder.glob("*.tif")
    arrs = []
    names = []
    for i in files:
        arr = rio.open_rasterio(i)

        date = i.name[3:11]
        isodate = "-".join([date[0:4], date[4:6], date[6:8]])
        dt = datetime.fromisoformat(isodate)
        names.append(dt)

        arr["date"] = dt
        arr["name"] = i.name

        arrs.append(arr.sel({"band": 1}))

    s1vv_ts = xr.concat(arrs, dim="date")
    return s1vv_ts
