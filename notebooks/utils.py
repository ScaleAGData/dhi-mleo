import pandas as pd


def preprocess_sensor_data(
    df_A=None,
    df_B=None,
    sensor_option="AVG",
    target_column="INSITU_SM",
    date_col="Date",
    station_col="Station",
) -> pd.DataFrame:
    """
    Preprocess sensor dataframes (Sensor A, Sensor B, or both averaged).

    Parameters
    ----------
    df_A : pd.DataFrame or None
        Dataframe for Sensor A.
    df_B : pd.DataFrame or None
        Dataframe for Sensor B.
    sensor_option : str
        "A"   -> use only Sensor A
        "B"   -> use only Sensor B
        "AVG" -> average both sensors where available; otherwise take whichever is available
    target_column : str
        Column to average across sensors (e.g., "INSITU_SM").
    date_col : str
        Name of the datetime column.
    station_col : str
        Name of the station column.

    Returns
    -------
    pd.DataFrame
        Preprocessed dataframe according to the sensor option.
    """

    if sensor_option == "A":
        if df_A is None:
            raise ValueError("df_A must be provided when sensor_option='A'.")
        return df_A.copy()

    elif sensor_option == "B":
        if df_B is None:
            raise ValueError("df_B must be provided when sensor_option='B'.")
        return df_B.copy()

    elif sensor_option == "AVG":
        if df_A is None and df_B is None:
            raise ValueError("At least one of df_A or df_B must be provided for 'AVG'.")
        if df_A is None:
            return df_B.copy()
        if df_B is None:
            return df_A.copy()

        df_A = df_A.copy()
        df_B = df_B.copy()
        df_A[station_col] = df_A[station_col].astype(str)
        df_B[station_col] = df_B[station_col].astype(str)
        df_A[date_col] = pd.to_datetime(df_A[date_col])
        df_B[date_col] = pd.to_datetime(df_B[date_col])

        stations_A = set(df_A[station_col])
        stations_B = set(df_B[station_col])
        common_stations = stations_A & stations_B
        only_A = stations_A - stations_B
        only_B = stations_B - stations_A

        df_common_A = df_A[df_A[station_col].isin(common_stations)]
        df_common_B = df_B[df_B[station_col].isin(common_stations)]
        df_only_A = df_A[df_A[station_col].isin(only_A)]
        df_only_B = df_B[df_B[station_col].isin(only_B)]

        merged_common = pd.merge(
            df_common_A,
            df_common_B,
            on=[date_col, station_col],
            suffixes=("_A", "_B"),
            how="outer",
        )

        df_common_final = merged_common[
            [col for col in merged_common.columns if col.endswith("_A")]
        ].copy()
        df_common_final.columns = [c[:-2] for c in df_common_final.columns]
        df_common_final[date_col] = merged_common[date_col]
        df_common_final[station_col] = merged_common[station_col]

        df_common_final[target_column] = merged_common[
            [f"{target_column}_A", f"{target_column}_B"]
        ].mean(axis=1, skipna=True)

        data = pd.concat([df_common_final, df_only_A, df_only_B], ignore_index=True)
        data = data.sort_values(by=[date_col, station_col]).reset_index(drop=True)
        return data

    else:
        raise ValueError("Invalid sensor_option. Choose 'A', 'B', or 'AVG'.")
