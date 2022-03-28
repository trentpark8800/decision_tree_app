from typing import Dict, TypeVar

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder

DataFrame = TypeVar("pandas.core.DataFrame")
Series = TypeVar("pandas.core.Series")
Figure = TypeVar("matplotlib.pyplot.figure")


def read_xl_data_into_dataframe(data, sheet_name: str) -> DataFrame:
    """_summary_

    Args:
        sheet_name (str): _description_

    Returns:
        DataFrame: _description_
    """
    df: DataFrame = pd.read_excel(data, sheet_name=sheet_name, engine="openpyxl")
    return df


def create_column_encoders(df: DataFrame) -> Dict[str, LabelEncoder]:
    """_summary_

    Args:
        df (DataFrame): _description_

    Returns:
        Dict[LabelEncoder]: _description_
    """
    encoders: Dict[LabelEncoder] = {}

    for column in df.columns:
        if not is_numeric_dtype(df[column]):
            encoders[column] = LabelEncoder()

    return encoders


def encode_columns(df: DataFrame, encoders: Dict[str, LabelEncoder]) -> DataFrame:
    """_summary_

    Args:
        df (DataFrame): _description_

    Returns:
        DataFrame: _description_
    """
    encoded_df: DataFrame = df.copy()

    for column in df.columns:
        if not is_numeric_dtype(df[column]):
            encoded_df[column] = encoders[column].fit_transform(encoded_df[column].astype("str"))

    return encoded_df
