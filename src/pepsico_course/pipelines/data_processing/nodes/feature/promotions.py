import logging
from typing import Dict, List

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

def _generate_dicotomic_features(df:pd.DataFrame, column: str) -> pd.DataFrame:
    for disscount in df[column].unique():
        if disscount != np.nan:
            df[str(disscount)] = df[column].apply(lambda x: 1 if x == disscount else 0) 

    return df

def _generate_national_features(df: pd.DataFrame, 
                                column: str, 
                                new_column_name: str, 
                                conditions: List[str]) -> pd.DataFrame:
    df[new_column_name] = df[column].apply(lambda x: 1 if x in conditions else 0)
    
    return df

def _aggregate_promo(df: pd.DataFrame, 
                     column: str, 
                     new_column_name: str, 
                     prefix: str, 
                     aggregated_value: str
                     ) -> pd.DataFrame:
    """Aggregate the promo type"""
    df[new_column_name] = df[column]
    for promo in df[column].unique():
        if promo is not None and promo[0] == prefix:
            df.loc[df[column] == promo, new_column_name] = aggregated_value
    
    return df

def _clean_promo_output(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df

def feature_promotions(promotions_processed: pd.DataFrame) -> pd.DataFrame:
    promotions_processed = _generate_dicotomic_features(promotions_processed, "promo_type")
    promotions_processed = _generate_national_features(promotions_processed, 
                                                       "customer", 
                                                       "is_national", 
                                                       ["BM", "EROSKI", "MERCADONA"]
                                                       )
    
    promotions_processed = _generate_national_features(promotions_processed, 
                                                       "customer", 
                                                       "is_basque", 
                                                       ["BM", "EROSKI"]
                                                       )
    
    promotions_processed = _aggregate_promo(promotions_processed, 
                                            "promo_type", 
                                            "promo_type_aggregated", 
                                            "d", 
                                            "discount")
    promotions_processed = _clean_promo_output(promotions_processed)

    return promotions_processed