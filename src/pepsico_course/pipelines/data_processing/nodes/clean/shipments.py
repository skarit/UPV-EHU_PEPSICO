from typing import Dict
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from scipy.stats import mstats
from .utils import create_model_id, normalize_columns
import logging

log = logging.getLogger(__name__)


def _make_continuous(shipments: pd.DataFrame, fill_method: str) -> pd.DataFrame:
    """
    Make the dataframe continuous, re-indexing based on max_date and min_date

    Args:
        shipments (pd.DataFrame): DataFrame containing the demand 
        fill_method (str): Method for filling the missing datapoints

    Returns:
        pd.DataFrame: DataFrame with the `time_var` continuous 
    """

    shipments['time_var'] = pd.to_datetime(shipments['time_var'])
    old_len = len(shipments)

    def _reindex(df: pd.DataFrame, method: str) -> pd.DataFrame:
        """Function that reindex the DataFrame

        Args:
            df (pd.DataFrame): Input DataFrame
            method (str): Method for filling

        Returns:
            pd.DataFrame: DataFrame re-indexed
        """

        min_date = df.index.min()
        max_date = df.index.max()

        df = df.reindex(  
                        pd.date_range(min_date,  # Generate new data range
                                      max_date, 
                                      freq='W-MON'
                                    ),
                        method=method
                    )
        return df

    shipments = (shipments
                 .sort_values('time_var')
                 .set_index('time_var')  # Setting the `time_var` needed for re-indexing
                 .groupby(['prod_code', 'customer', 'location', 'category']) # Calculate at `item level`
                 ['shipments']
                 .apply(lambda x: _reindex(x, fill_method))
                .reset_index()
                .rename(columns={'level_4': 'time_var'})
                )

    new_len = len(shipments)
    log.info(f'Added {new_len-old_len} new lines with the method {fill_method}')

    return shipments


def _remove_continuous_zeros(shipments: pd.DataFrame, n_zeros: int) -> pd.DataFrame:
    """
    Remove the TimeSeries with more than `n_zeroes` zeros

    Args:
        shipments (pd.DataFrame): DataFrame containing the demand 
        n_zeros (int): Number of zeros for which remove the TimeSeries

    Returns:
        pd.DataFrame: DataFrame which containing TS with number of zeros > `n_zeros`
    """

    old_n_dfu = len(shipments[['prod_code', 'customer', 'location']].drop_duplicates())

    shipments['n_zeros'] = (shipments
                            .groupby(['prod_code', 'customer', 'location'])
                            ['shipments']
                            .transform(lambda x: len(x[x==0]))
                            )
    
    shipments = shipments[shipments['n_zeros'] < n_zeros]
    shipments = shipments.drop('n_zeros', axis=1)

    new_n_dfu = len(shipments[['prod_code', 'customer', 'location']].drop_duplicates())
    log.info(f'Removed {old_n_dfu-new_n_dfu} DFUs with too many zeros')
    return shipments


def _winsorize(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Decompose tthe input DataFrame using STL and then winsorizing the residuals 

    https://en.wikipedia.org/wiki/Winsorizing

    Args:
        df (pd.DataFrame): DataFrame containing the demand
        threshold (float): Threshold representing the percentiles (in float)

    Returns:
        pd.DataFrame: DataFrame containing the demand winsorized
    """

    stl = STL(df, period=52, seasonal=53)  # Generate STL decomposition
    res = stl.fit()
    
    trend = res.trend 
    seasonal = res.seasonal
    residual = res.resid

    residual_winsorize = mstats.winsorize(
                        residual, limits=[threshold, threshold]
                    )
    
    # For those element which are "winsorized" we simple remove the residual 
    adjustment = (np.ma.getdata(residual_winsorize) - residual).round()
    residual_winsorize[adjustment != 0] = 0  # Removing of the residuals
    observed_winsorize = (trend + seasonal + residual_winsorize).round() # TS recomposition

    # Change negative values to zero
    observed_winsorize = map(lambda x: max(x, 0), observed_winsorize)
    
    return observed_winsorize


def _outlier_removal(shipments: pd.DataFrame, percentile: int) -> pd.DataFrame:
    """
    Remove outlier from the demand using STL decomposition + Winsorization

    Args:
        shipments (pd.DataFrame): DataFrame containing the demand
        percentile (int): Threshold representing the percentiles (in float)

    Returns:
        pd.DataFrame: DataFrame containing the demand winsorized
    """

    shipments['new_shipments'] = (shipments           
                             .sort_values(by='time_var')
                             .groupby(['prod_code', 'customer', 'location'])
                             ['shipments']
                             .transform(lambda x: _winsorize(x, percentile))
                    )
    
    changed = len(shipments[shipments['shipments'] != shipments['new_shipments']])
    log.info(f'Changed {changed}/{len(shipments)} points')
    shipments = shipments.drop('shipments', axis=1)
    shipments = shipments.rename({'new_shipments': 'shipments'})
    return shipments

def clean_shipments(shipments: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """
    Kedro-Node function for cleaning the shipments DataFrame

    Args:
        shipments (pd.DataFrame): DataFrame containing the demand
        parameters (Dict): Parameters used for the cleaning methods

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """

    shipments_params = parameters['shipments']
    log.info(f'Received parameters: {shipments_params}')

    shipments = _make_continuous(shipments, shipments_params['fill_method'])
    shipments = _remove_continuous_zeros(shipments, shipments_params['n_zeros'])
    shipments = _outlier_removal(shipments, shipments_params['percentile'])
    shipments = normalize_columns(shipments, "customer")
    shipments = normalize_columns(shipments, "location")
    shipments = create_model_id(shipments)

    return shipments