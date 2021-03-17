"""Functions for plotting pixel and raster time series."""
import matplotlib.pyplot as plt
import seaborn as sns


def plot_df_pixel_ts(df, obs_column: str = "lndvi_obs"):
    """Plots a regularly spaced time series with NaNs displayed as a scatterplot.

    Turn an irregularly spaced time series in day units into a regularly spaced one with:
    df.asfreq("1D")

    The df must have a "dates" column and the specified observations column.

    Args:
        df (pd.DataFrame): A dataframe with columns "dates" and obs_column
        obs_column (str, optional): The str for the observation column. Defaults to "lndvi_obs".
    """
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, ax=ax, x="dates", y=obs_column)
    xtix = ax.get_xticks()
    freq = int(5)
    ax.set_xticks(xtix[::freq])
    # nicer label format for dates
    fig.autofmt_xdate()


def plot_da_pixel_ts(da, obs_column: str):
    """Plots a regularly spaced time series with NaNs displayed as a scatterplot.

    Args:
        da ([type]): A DataArray with one dimension named "date". Can be taken from a 3D array.
            For example: bayts_ts[:,50,20] takes a single pixel's time series if bayts_ts is a
            3D DataArray with dims (date, y x).
        obs_column (str): The name of the observation column, can be chosen byt he user to reflect
            what the DataArray measures.
    """
    df = da.to_dataframe(name=obs_column)
    df["dates"] = df.index.format()
    df = df.asfreq("1D")
    plot_df_pixel_ts(df, obs_column)
