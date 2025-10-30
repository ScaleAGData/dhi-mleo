from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut


def regression_metrics(
    y_true: Union[list, np.ndarray],
    y_pred: Union[list, np.ndarray],
    label: str = "",
    return_results: bool = False,
) -> Union[None, Dict[str, float]]:
    """
    Compute and display common regression metrics between true and predicted values.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.
    label : str, optional
        Optional label to prefix metric print statements (default is '').
    return_results : bool, optional
        If True, return the metrics as a dictionary (default is False).

    Returns
    -------
    results : dict, optional
        Dictionary containing computed metrics:
        - 'Pearson_r' : float, Pearson correlation coefficient
        - 'R2' : float, coefficient of determination
        - 'RMSE' : float, root mean squared error
        - 'MAE' : float, mean absolute error
        Returned only if `return_results=True`.
    """

    r = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    print(f"{label} Pearson r: {r:.4f}")
    print(f"{label} R²: {r2:.4f}")
    print(f"{label} RMSE: {rmse:.4f}")
    print(f"{label} MAE: {mae:.4f}\n")

    if return_results:
        return {"Pearson_r": r, "R2": r2, "RMSE": rmse, "MAE": mae}


def evaluate_baseline(
    df_train: pd.DataFrame, df_test: pd.DataFrame, baseline_col: str, target_col: str
) -> Dict[str, Tuple[pd.Series, pd.Series]]:
    """
    Evaluate a baseline prediction column against the target on training, testing,
    and combined datasets using regression metrics.

    Parameters
    ----------
    df_train : pandas.DataFrame
        Training dataset containing both the baseline predictions and the target.
    df_test : pandas.DataFrame
        Testing dataset containing both the baseline predictions and the target.
    baseline_col : str
        Column name in the dataframes representing the baseline prediction.
    target_col : str
        Column name in the dataframes representing the ground truth target.

    Returns
    -------
    results : dict
        Dictionary containing tuples of (y_true, y_pred) for each dataset split:
        - 'train': training set
        - 'test': testing set
        - 'full': concatenated train + test set
    """

    y_train_true = df_train[target_col].values.ravel()
    y_train_pred = df_train[baseline_col].values.ravel()

    y_test_true = df_test[target_col].values.ravel()
    y_test_pred = df_test[baseline_col].values.ravel()

    y_full_true = pd.concat([df_train[target_col], df_test[target_col]]).values.ravel()
    y_full_pred = pd.concat(
        [df_train[baseline_col], df_test[baseline_col]]
    ).values.ravel()

    print("Baseline Results for Training Set:")
    regression_metrics(y_train_true, y_train_pred, label="Train")

    print("Baseline Results for Test Set:")
    regression_metrics(y_test_true, y_test_pred, label="Test")

    print("Baseline Results for Full Dataset:")
    regression_metrics(y_full_true, y_full_pred, label="Overall")

    return {
        "train": (y_train_true, y_train_pred),
        "test": (y_test_true, y_test_pred),
        "full": (y_full_true, y_full_pred),
    }


def station_metrics(group: pd.DataFrame, target_col: str, pred_col: str) -> pd.Series:
    """
    Compute regression metrics (Pearson r, R², RMSE, MAE) for a single station or group.

    Parameters
    ----------
    group : pandas.DataFrame
        DataFrame corresponding to a single station or group. Must contain
        columns for the true target and predicted values.
    target_col : str
        Name of the column containing ground truth target values.
    pred_col : str
        Name of the column containing predicted values.

    Returns
    -------
    pandas.Series
        Series with the following keys:
        - 'Pearson_r': Pearson correlation coefficient (float)
        - 'R2': Coefficient of determination (float)
        - 'RMSE': Root mean squared error (float)
        - 'MAE': Mean absolute error (float)
        If the group has fewer than 2 samples or contains only NaNs,
        all metrics are returned as NaN.
    """
    y_true = group[target_col].values
    y_pred = group[pred_col].values

    if len(y_true) < 2 or np.all(np.isnan(y_true)) or np.all(np.isnan(y_pred)):
        return pd.Series(
            {"Pearson_r": np.nan, "R2": np.nan, "RMSE": np.nan, "MAE": np.nan}
        )

    try:
        r_pearson, _ = pearsonr(y_true, y_pred)
    except Exception:
        r_pearson = np.nan

    return pd.Series(
        {
            "Pearson_r": r_pearson,
            "R2": r2_score(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAE": mean_absolute_error(y_true, y_pred),
        }
    )


def evaluate_stations(
    df: pd.DataFrame, station_col: str, target_col: str, pred_col: str
):
    """
    Evaluate baseline product performance per station (or any grouping column).

    Parameters
    ----------
    df : DataFrame
        Input dataframe.
    station_col : str
        Column name to group by (e.g. "Station", "SiteID").
    target_col : str
        Ground-truth variable.
    pred_col : str
        Predicted/baseline variable.

    Returns
    -------
    station_results : DataFrame
        Metrics per station.
    average_metrics : Series
        Mean of each metric across all stations.
    """
    station_results = df.groupby(station_col).apply(
        lambda g: station_metrics(g, target_col, pred_col)
    )

    average_metrics = station_results.mean(numeric_only=True)

    print(f"\nStation-level metrics (grouped by `{station_col}`):")
    print(station_results)

    print("\nAverage metrics across stations:")
    print(average_metrics)

    return station_results, average_metrics


def loso_cv_ml(
    df: pd.DataFrame,
    features: list,
    target: str,
    group_col: str,
    model: Any,
    return_predictions: bool = False,
):
    """
    Perform Leave-One-Group-Out (LOSO) cross-validation for a regression model.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing features, target, and grouping column.
    features : list
        List of feature column names to use for training.
    target : str
        Target column name.
    group_col : str
        Column used for grouping (e.g., "Station").
    model : scikit-learn regressor instance
        Model to train.
    return_predictions : bool
        If True, also returns a DataFrame with true and predicted values per station.

    Returns
    -------
    results_df : pd.DataFrame
        LOSO metrics per group.
    avg_metrics : pd.Series
        Mean metrics across all groups.
    predictions_df : pd.DataFrame (optional)
        True vs predicted values per group (if return_predictions=True).
    """
    logo = LeaveOneGroupOut()
    results = []
    pred_list = []

    for train_idx, test_idx in logo.split(df, groups=df[group_col]):
        df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]

        X_train = df_train[features]
        y_train = df_train[target].values.ravel()
        X_test = df_test[features]
        y_test = df_test[target].values.ravel()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r_pearson = np.corrcoef(y_test, y_pred)[0][1]
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        results.append(
            {
                group_col: df_test[group_col].iloc[0],
                "Pearson_r": r_pearson,
                "R2": r2,
                "RMSE": rmse,
                "MAE": mae,
            }
        )

        if return_predictions:
            pred_list.append(
                pd.DataFrame(
                    {group_col: df_test[group_col], "y_true": y_test, "y_pred": y_pred}
                )
            )

    results_df = pd.DataFrame(results)
    avg_metrics = results_df.drop(columns=group_col).mean()

    if return_predictions:
        predictions_df = pd.concat(pred_list, ignore_index=True)
        return results_df, avg_metrics, predictions_df

    return results_df, avg_metrics
