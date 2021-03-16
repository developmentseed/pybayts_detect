"""Functions for plotting pixel and raster time series."""
import matplotlib.pyplot as plt
import seaborn as sns


def plot_pixel_ts(df, obs_column: str = "lndvi_obs"):
    """Plots a regularly spaced time series with or without NaNs as a scatterplot.

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
