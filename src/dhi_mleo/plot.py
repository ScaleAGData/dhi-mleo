from typing import Any, List, Optional, Tuple

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from sklearn.model_selection import LeaveOneGroupOut


def corr_matrix(
    df: pd.DataFrame,
    title: str = "Correlation Matrix",
    figsize: Tuple[int, int] = (12, 8),
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Plot a Pearson correlation heatmap for all numeric columns in a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing numeric columns to compute correlations.
    title : str, optional
        Title of the plot (default is "Correlation Matrix").
    figsize : tuple of int, optional
        Figure size in inches (width, height), default is (12, 8).
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes are created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes object containing the heatmap.
    """
    corr = df.corr()
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        show = True
    else:
        show = False

    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        annot=False,
        ax=ax,
    )
    ax.set_title(title, size=12, weight="bold", pad=15)
    ax.tick_params(labelsize=9)
    if show:
        plt.show()
    return ax


def corr_target(
    df: pd.DataFrame,
    target: str,
    figsize: Tuple[int, int] = (4, 7),
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Plot the correlation of each variable with a target variable as a heatmap.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing numeric columns including the target.
    target : str
        Name of the target column to compute correlation against.
    figsize : tuple of int, optional
        Figure size in inches (width, height), default is (4, 7).
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes are created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes object containing the target correlation heatmap.
    """
    corr = df.corr()
    corr_target = corr[[target]].drop(index=target)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        show = True
    else:
        show = False

    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr_target,
        cmap=cmap,
        center=0,
        annot=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5, "aspect": 30},
        ax=ax,
    )
    ax.set_title(
        f"Correlation with {target}\n({len(corr_target)} variables)",
        size=12,
        weight="bold",
        pad=15,
    )
    ax.tick_params(labelsize=9)
    if show:
        plt.show()
    return ax


def corr_bar(
    df: pd.DataFrame,
    target: str,
    figsize: Tuple[int, int] = (8, 5),
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Plot a horizontal bar chart of Pearson correlation between the target and all other variables.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing numeric columns including the target.
    target : str
        Name of the target column to compute correlation against.
    figsize : tuple of int, optional
        Figure size in inches (width, height), default is (8, 5).
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes are created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes object containing the correlation bar chart.
    """
    corr = df.corr()[[target]].drop(index=target)
    corr_sorted = corr.sort_values(by=target, ascending=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        show = True
    else:
        show = False

    corr_sorted[target].plot.barh(ax=ax, width=0.7, edgecolor="black")
    ax.set_title(
        f"Correlation with {target}\n({len(corr_sorted)} variables)",
        size=12,
        weight="bold",
        pad=15,
    )
    ax.set_xlabel(f"Pearson Correlation with {target}", fontsize=12)
    ax.set_ylabel("Variable", fontsize=12)
    ax.axvline(0, color="grey", linewidth=1, linestyle="--")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    if show:
        plt.tight_layout()
        plt.show()
    return ax


def plot_loso_distributions(
    df: pd.DataFrame,
    target: str,
    group_col: str,
    cols: int = 4,
    figsize_per_row: Tuple[int, int] = (16, 3),
) -> None:
    """
    Plot the distribution of target values for train/test splits per group
    using Leave-One-Group-Out (LOGO) cross-validation.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing features, target, and grouping column.
    target : str, optional
        Target column name to plot distributions.
    group_col : str, optional
        Column name to group by for LOGO splitting.
    cols : int, optional
        Number of subplot columns (default is 4).
    figsize_per_row : tuple of int, optional
        Width and height per row of subplots in inches (default is (16, 3)).

    Returns
    -------
    None
        Displays a figure with KDE plots for each group's train/test distributions.

    """
    logo = LeaveOneGroupOut()
    grouped_targets = {}

    for train_idx, test_idx in logo.split(df, groups=df[group_col]):
        group_left_out = df.iloc[test_idx][group_col].iloc[0]
        y_train = df.iloc[train_idx][target].values
        y_test = df.iloc[test_idx][target].values
        grouped_targets[group_left_out] = {"train": y_train, "test": y_test}

    num_groups = len(grouped_targets)
    rows = int(np.ceil(num_groups / cols))
    fig, axs = plt.subplots(
        rows, cols, figsize=(figsize_per_row[0], figsize_per_row[1] * rows)
    )
    axs = axs.flatten()

    for idx, (group, data) in enumerate(grouped_targets.items()):
        sns.kdeplot(
            data["train"],
            fill=True,
            label="Train",
            ax=axs[idx],
            color="tab:blue",
            alpha=0.6,
        )
        sns.kdeplot(
            data["test"],
            fill=True,
            label="Test",
            ax=axs[idx],
            color="tab:orange",
            alpha=0.6,
        )
        axs[idx].set_title(f"{group_col} {group}")
        axs[idx].set_xlabel(target)
        axs[idx].set_ylabel("Density")
        if idx == 0:
            axs[idx].legend()

    for i in range(num_groups, len(axs)):
        fig.delaxes(axs[i])

    plt.suptitle(
        f"Distribution of {target} in Train/Test for Each {group_col} (LOGO split)",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()


def plot_predictions_by_station(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    model: Any,
    features: List[str],
    target_col: str,
    date_col: str,
    station_col: str,
) -> None:
    """
    Plot observed vs predicted target values for each station over time.

    Parameters
    ----------
    df_train : pandas.DataFrame
        Training dataset containing features, target, date, and station columns.
    df_test : pandas.DataFrame
        Testing dataset containing features, target, date, and station columns.
    model : object
        Trained model implementing a `.predict()` method.
    features : list of str
        Column names used as features for prediction.
    target_col : str, optional
        Target column name.
    date_col : str, optional
        Column name containing dates.
    station_col : str, optional
        Column name containing station identifiers.

    Returns
    -------
    None
        Displays scatter plots comparing observed and predicted values over time
        for each station.
    """
    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train["Pred_SM"] = model.predict(df_train[features])
    df_test["Pred_SM"] = model.predict(df_test[features])
    df_train["Split"] = "Train"
    df_test["Split"] = "Test"

    df_plot = pd.concat([df_train, df_test], ignore_index=True)

    legend_elements = [
        mlines.Line2D(
            [],
            [],
            color="tab:blue",
            marker="o",
            linestyle="None",
            markersize=6,
            markeredgecolor="k",
            label="Observed (Train)",
        ),
        mlines.Line2D(
            [],
            [],
            color="tab:green",
            marker="o",
            linestyle="None",
            markersize=6,
            markeredgecolor="k",
            label="Observed (Test)",
        ),
        mlines.Line2D(
            [],
            [],
            color="orange",
            marker="x",
            linestyle="None",
            markersize=6,
            label="Predicted (Train)",
        ),
        mlines.Line2D(
            [],
            [],
            color="red",
            marker="x",
            linestyle="None",
            markersize=6,
            label="Predicted (Test)",
        ),
    ]

    # per station
    for station in df_plot[station_col].unique():
        df_st = df_plot[df_plot[station_col] == station]

        fig, ax = plt.subplots(figsize=(10, 5))

        # true
        ax.scatter(
            df_st[df_st["Split"] == "Train"][date_col],
            df_st[df_st["Split"] == "Train"][target_col],
            c="tab:blue",
            label="_nolegend_",
            s=20,
            alpha=0.7,
            edgecolor="k",
        )
        ax.scatter(
            df_st[df_st["Split"] == "Test"][date_col],
            df_st[df_st["Split"] == "Test"][target_col],
            c="tab:green",
            label="_nolegend_",
            s=20,
            alpha=0.7,
            edgecolor="k",
        )
        # preds
        ax.scatter(
            df_st[df_st["Split"] == "Train"][date_col],
            df_st[df_st["Split"] == "Train"]["Pred_SM"],
            c="orange",
            label="_nolegend_",
            s=20,
            alpha=0.7,
            marker="x",
        )
        ax.scatter(
            df_st[df_st["Split"] == "Test"][date_col],
            df_st[df_st["Split"] == "Test"]["Pred_SM"],
            c="red",
            label="_nolegend_",
            s=20,
            alpha=0.7,
            marker="x",
        )

        ax.set_title(
            f"Soil Moisture Over Time - Station {station}", fontsize=14, weight="bold"
        )
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Soil Moisture", fontsize=12)
        ax.legend(handles=legend_elements, loc="lower left", fontsize=10, frameon=True)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        fig.autofmt_xdate(rotation=25)
        plt.tight_layout()
        plt.show()


def plot_scatter_fit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Scatter Plot",
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Plot scatter of true vs predicted values with 1-to-1 line and regression fit.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true target values.
    y_pred : np.ndarray
        Array of predicted target values.
    title : str, optional
        Title of the plot (default is "Scatter Plot").
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes are created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes object containing the scatter plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        show = True
    else:
        show = False

    ax.scatter(y_true, y_pred, alpha=0.6, s=14)
    ax.plot(
        [min(y_true), max(y_true)],
        [min(y_true), max(y_true)],
        color="grey",
        linestyle="--",
        label="1:1 line",
    )
    sns.regplot(
        x=y_true,
        y=y_pred,
        ax=ax,
        line_kws={
            "color": "#41ab5d",
            "alpha": 0.7,
            "lw": 2,
            "linestyle": "dashed",
            "label": "Linear fit",
        },
        scatter=False,
    )
    ax.set_title(title, size=12, weight="bold", pad=15)
    ax.legend(loc="upper left")

    if show:
        plt.show()
    return ax
