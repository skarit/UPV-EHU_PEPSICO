from typing import Literal, Optional

import pandas as pd
import logging

log = logging.getLogger(__name__)


def _get_weekly_data(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Aggregates a column ("Week") with the value of first monday for a given date

    Args:
        df (pd.DataFrame): The dataframe
        column (str): The time column to generate the week day

    Returns:
        pd.DataFrame: The dataframe with this new column. 
    """
    df['Week'] = df[column].apply(lambda x: x - pd.Timedelta(days=x.weekday()))
    
    return df

def _generate_dicotomic_features(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Generates a dicotomic variable for each of the values present in a provided column"""
    for column_name in df[column].unique():
        df[column_name] = df[column].apply(lambda x: 1 if x == column else 0)
    
    return df

def _aggregate_by_occurrence(df:pd.DataFrame, 
                            column: str, 
                            transform_type: Literal["max","size"], 
                            new_column_name: Optional[str]= None) -> pd.DataFrame:
    """Aggregates by a provided column with an specific transformation"""
    auxiliar = df.groupby(column)
    if new_column_name is None:
        df = auxiliar.max(transform_type).reset_index()
    else:
        df[new_column_name] = auxiliar.transform(transform_type)

    return df

def _generate_is_holiday(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a flag column to indicate is holiday"""
    df["is_holiday"] = 1

    return df

def _clean_output(df: pd.DataFrame) -> pd.DataFrame:
    """Drop not necessary columns and reset index"""
    # df = df.drop("DT",axis=1)
    # df = df.drop("HOL_NM",axis=1)
    df = df.reset_index(drop = True)
    df = df.rename(columns={"Week": "time_var"})
    
    return df

def feature_holidays(holidays_processed: pd.DataFrame) -> pd.DataFrame:

    holidays_processed = _get_weekly_data(holidays_processed, "DT")
    holidays_processed = _generate_dicotomic_features(holidays_processed, "HOL_NM")
    holidays_processed = _aggregate_by_occurrence(holidays_processed, "Week", "size", "number_of_holidays")
    holidays_processed = _aggregate_by_occurrence(holidays_processed, "Week", "max")
    holidays_processed = _generate_is_holiday(holidays_processed)
    holidays_processed = _clean_output(holidays_processed)

    return holidays_processed