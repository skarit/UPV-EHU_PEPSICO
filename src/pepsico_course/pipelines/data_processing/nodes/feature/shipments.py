import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)

def _encode_categorical_columns(shipments_processed: pd.DataFrame) -> pd.DataFrame:

    cat_cols = ['customer', 'location', 'category']
    new_names = list(map(lambda x: f'{x}_encoded', cat_cols))

    shipments_processed[new_names] = (shipments_processed[cat_cols]
                                      .astype('category')
                                      .apply(lambda x: x.cat.codes)
                                      )

    return shipments_processed


def _encode_time_variables(shipments_processed: pd.DataFrame) -> pd.DataFrame:


    # Generated dummy variables

    # Generate dummy variables for the Year
    shipments_processed = (shipments_processed
                           .sort_values(by='time_var')
                           .groupby(['prod_code', 'customer', 'location'])
                           .apply(lambda x: pd.concat([x,
                                                       pd.get_dummies(x['time_var'].dt.year,
                                                                      drop_first=True,
                                                                      prefix="year")
                                                        ],
                                                        axis=1
                                                        )
                                   )
                             .reset_index(drop='True')
                        )


    # Encode month of the year as cyclical feature
    shipments_processed['sin_month'] = shipments_processed.apply(lambda x: np.sin(x['time_var'].month / 12 * 2 * np.pi), axis=1)
    shipments_processed['cos_month'] = shipments_processed.apply(lambda x: np.cos(x['time_var'].month / 12 * 2 * np.pi), axis=1)

    # Encode week of the year as cyclical feature
    shipments_processed['sin_week'] = shipments_processed.apply(lambda x: np.sin(x['time_var'].isocalendar().week / 52 * 2 * np.pi), axis=1)
    shipments_processed['cos_week'] = shipments_processed.apply(lambda x: np.cos(x['time_var'].isocalendar().week / 52 * 2 * np.pi), axis=1)

    return shipments_processed


def feature_shipments(shipments_processed: pd.DataFrame) -> pd.DataFrame:

    shipments_processed = _encode_categorical_columns(shipments_processed)
    shipments_processed = _encode_time_variables(shipments_processed)

    return shipments_processed