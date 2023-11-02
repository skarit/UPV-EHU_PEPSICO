from typing import Dict
import pandas as pd
import logging

from .utils import * 

log = logging.getLogger(__name__) 

def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates in the dataframe"""
    df = df.drop_duplicates()
    return df

def _rename_holidays(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Removes special characters for a provided column 
    """
    type_of_holiday = df[column].unique()
    for element in type_of_holiday: 
        df.loc[df[column] == element,column] = element.replace(" de ","").replace(" ","").replace("ñ","n").replace("'","")
    
    return df

def clean_holidays(holidays: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """Clean holidays raw data

    Args:
        holidays (pd.DataFrame): The raw data dataframe
        parameters (Dict): The parameters

    Returns:
        pd.DataFrame: cleaned dataset
    """
    holidays_params = parameters['holidays']
    log.info(f'Received parameters: {holidays_params}')
    holidays = typing_time_dataset(holidays, "DT")
    holidays = _drop_duplicates(holidays)
    holidays = _rename_holidays(holidays, "HOL_NM")

    return holidays