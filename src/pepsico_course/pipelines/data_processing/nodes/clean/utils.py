import pandas as pd

def create_model_id(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate the column `model_id` by combining the columns:
    `prod_code`, `customer`, `location`

    Args:
        data (pd.DataFrame): DataFrame containing the demand

    Returns:
        pd.DataFrame: DataFrame with the new column `model_id` created
    """

    data['model_id'] = (data['prod_code'] + 
                             '#' 
                             + data['customer'] + 
                             '#' 
                             + data['location']
                            )
    return data

def typing_time_dataset(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Change types"""
    df[column] = pd.to_datetime(df[column]) 
    
    return df

def normalize_columns(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Normalize customers names"""
    for customer in df[column].unique():
        df.loc[df[column] == customer, column] = customer.upper()

    return df