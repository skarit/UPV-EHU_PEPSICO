import pandas as pd

def create_model_input(shipments_featured: pd.DataFrame, 
                       promotions_featured: pd.DataFrame, 
                       holidays_featured: pd.DataFrame) -> pd.DataFrame:

    facts_data = shipments_featured[['model_id', 'customer', 'location', 'category']]
    
    shipments_featured = shipments_featured.drop(['prod_code', 'customer', 'location', 'category'], axis=1)
    promotions_featured = promotions_featured.drop(['prod_code', 'customer', 'location'], axis=1)

    mrd = shipments_featured.merge(promotions_featured, on=['model_id', 'time_var'], how="left")
    mrd = mrd.merge(holidays_featured, on="time_var", how="left")

    mrd = mrd.fillna(0)

    return mrd, facts_data
