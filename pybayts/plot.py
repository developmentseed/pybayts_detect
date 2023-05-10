"""Functions for plotting pixel and raster time series."""
import os

import matplotlib.pyplot as plt
import numpy as np
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


def plot_cm(cm, aoi_name: str, year: int, figdir: str):
    """Plots a confusion matrix and saves it witht he aoi_name in figdir.

    Args:
        cm (numpy.ndarray): The confusion matrix.
        aoi_name (str): The aoi_name string
        year (int): The year that is being evaluated.
        figdir (str): The path to the directory to save the figures.
    """
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=["change", "no change"],
        yticklabels=["change", "no change"],
        title=f"Normalized Confusion Matrix for {aoi_name} in {year}",
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f"  # 'd' # if normalize else 'd'
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)
    classes = [False, True]
    ax.set_ylim(len(classes) - 0.5, -0.5)
    fig.savefig(os.path.join(figdir, f"cm_{aoi_name}_{year}.png"))
