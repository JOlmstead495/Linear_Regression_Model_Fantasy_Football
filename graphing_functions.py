import pandas as pd
import numpy as np
import maplotlib.pyplot as plt


def heatMap(df):
    """
    Function creates and shows correlation heatmap plot based on a dataframe.
    Args:
        df (pandas.DataFrame): DataFrame to find correlation.
    Returns:
        None
    """
    # Create Correlation df. Any correlation over 75%
    corr = df.corr()[abs(df.corr()) > .75]
    # Plot figsize
    fig, ax = plt.subplots(figsize=(20, 20))
    # Generate Color Map, red & blue
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    # Generate Heat Map, allow annotations and place floats in map
    sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
    # Apply xticks
    plt.xticks(range(len(corr.columns)), corr.columns,
               horizontalalignment='left')
    # Apply yticks
    plt.yticks(range(len(corr.columns)), corr.columns,
               horizontalalignment='right')
    # Show plot
    plt.grid(True)
    plt.show()
    return
