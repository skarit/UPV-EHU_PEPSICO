import logging
from typing import Dict

import numpy as np
import pandas as pd

from .utils import * 

log = logging.getLogger(__name__)


def _normalize_promo_type(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Normalize the promo type names"""

    df.loc[(df["promo_type"] == '-') | (df["promo_type"].isnull()), "promo_type"] = None
    df.loc[(df["promo_type"] == '10% descuento') | (df["promo_type"] == "10% desc"), "promo_type"] = "d10%"
    df.loc[df["promo_type"] == 'menos 50%', "promo_type"] = "d50%"
    df.loc[df["promo_type"] == '20%', "promo_type"] = "d20%"
    df.loc[(df["promo_type"] == 'tres por dos') | (df["promo_type"] == '3x2'), "promo_type"] = "p3x2"

    return df

def clean_promotions(promotions: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """Clean promotions dataframe"""
    promotions_params = parameters['promotions']
    log.info(f'Received parameters: {promotions_params}')
    promotions = typing_time_dataset(promotions, "time_var")
    promotions = normalize_columns(promotions, "customer")
    promotions = normalize_columns(promotions, "location")
    promotions = _normalize_promo_type(promotions, "promo_type")
    promotions = create_model_id(promotions)

    return promotions