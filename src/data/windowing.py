import pandas as pd
import numpy as np

from pathlib import Path

from typing import List, Tuple
from numpy.typing import ArrayLike
from pandas import DataFrame


def get_lag_columns(columns: List[str], lag_steps: int) -> List[str]:
    """
    Returns the new title columns, by adding the lag number
    Includes the initial column names in the end.
    """
    new_columns = []
    for i in range(lag_steps, 0, -1):
        new_columns += [col_title + f"-{i}" for col_title in columns]

    new_columns += columns
    return new_columns


def get_lead_columns(columns: List[str], lead_steps: int) -> List[str]:
    """
    Returns the new title columns, by adding the lead number
    Includes the initial column names in the beginning.
    """
    new_columns = []
    new_columns += columns
    for i in range(1, lead_steps + 1):
        new_columns += [col_title + f"+{i}" for col_title in columns]

    return new_columns


def windowed_df_lag(df: DataFrame, columns_to_lag: List[str], lag_steps: int):
    temp_df = df[columns_to_lag]

    for i in range(1, lag_steps + 1):
        tmp = df[columns_to_lag].shift(i)
        temp_df = pd.concat([tmp, temp_df], axis=1)

    temp_df.columns = get_lag_columns(columns_to_lag, lag_steps)

    return temp_df.iloc[lag_steps:, :]


def windowed_df_lead(
    df: DataFrame, columns_to_lead: List[str], lead_steps: int
) -> DataFrame:
    temp_df = df[columns_to_lead]

    for i in range(1, lead_steps + 1):
        tmp = df[columns_to_lead].shift(-i)
        temp_df = pd.concat([tmp, temp_df], axis=1)

    temp_df.columns = get_lead_columns(columns_to_lead, lead_steps)

    return temp_df.iloc[:-lead_steps, :]


def create_windowed_df_per_machine(
    df: DataFrame,
    columns_to_lag: List[str],
    columns_to_lead: List[str],
    lag_steps: int,
    lead_steps: int,
) -> Tuple[DataFrame, ArrayLike]:
    if lag_steps == 0:
        lag_df = df[columns_to_lag]
    else:
        lag_df = windowed_df_lag(df, columns_to_lag, lag_steps)

    if lead_steps == 0:
        lead_df = df[columns_to_lead]
    else:
        lead_df = windowed_df_lead(df, columns_to_lead, lead_steps)

    if lead_steps > 0:
        lag_df = lag_df.iloc[:-lead_steps, :]

    if lag_steps > 0:
        lead_df = lead_df.iloc[lag_steps:, :]

    final_df = pd.concat([lag_df, lead_df], axis=1)

    return final_df, final_df.index.to_numpy()


def create_windowed_df(
    df: DataFrame,
    id_col: str,
    columns_to_lag: List[str],
    columns_to_lead: List[str],
    lag_steps: int,
    lead_steps: int,
) -> Tuple[DataFrame, ArrayLike]:
    """
    Return telemetry data per id_col
    """

    unique_ids = df[id_col].unique()

    for i, id_ in enumerate(unique_ids):
        df_id = df.loc[df[id_col] == id_]

        df_id = df_id.reset_index(drop=True)

        telemetry_df, idx = create_windowed_df_per_machine(
            df_id, columns_to_lag, columns_to_lead, lag_steps, lead_steps
        )

        # Drop all columns used for telemetry lead and lag
        df_rest = df_id.drop(columns_to_lag + columns_to_lead, axis=1)

        df_rest = df_rest.iloc[idx, :]

        # Combine the reduced column with the lead-lagged telemtry dataframe
        tmp_df = pd.concat([df_rest, telemetry_df], axis=1)

        # Stack for each machine the recreated dataframe.
        # If it is the fist time the loop runs, create the new dataframe with the first recreated df
        if i == 0:
            final_df = tmp_df.copy()
        else:
            final_df = pd.concat([final_df, tmp_df], axis=0)

    final_df.reset_index(inplace=True, drop=True)
    return final_df


DATA_PATH = Path.cwd() / ".." / ".." / "data" / "processed" / "data_processed.csv"
TELEMETRY_COLUMNS = ["voltmean_3h", "rotatemean_3h"]
TARGET_COLUMNS = ["comp1_life"]

data = pd.read_csv(DATA_PATH.as_posix())

tst_df = create_windowed_df(data, "machineID", TELEMETRY_COLUMNS, TARGET_COLUMNS, 2, 1)
