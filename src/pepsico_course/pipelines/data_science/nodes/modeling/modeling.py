from typing import Dict, Tuple
import pandas as pd
import numpy as np
import logging
import statsmodels.api as sm
import lightgbm as lgb
log = logging.getLogger(__name__) 


def _compute_accuracy(df, y='shipments', y_pred='y_hat'):    
    # Initialize the accuracy metric to 0
    metric = 0

    # Check if the sum of actual values is not equal to 0    
    if df[y].sum() != 0:
        # Calculate the accuracy using the formula:
        # 1 - (sum of absolute differences between actual and predicted) / sum of actual values
        metric = np.max([0, 1 - np.abs(df[y_pred] - df[y]).sum() / df[y].sum()])
    # If the sum of actual values is 0 and the sum of absolute differences is 0, set the accuracy to 1
    elif (df[y].sum() == 0) & (np.abs(df[y_pred] - df[y]).sum() == 0):
        metric = 1
    # If the sum of actual values is 0 and the sum of absolute differences is greater than 0, set the accuracy to 0
    elif (df[y].sum() == 0) & (np.abs(df[y_pred] - df[y]).sum() > 0):
        metric = 0

    # Return the calculated accuracy metric as a pandas Series
    return pd.Series({'accuracy': metric})


def _ts_fit_predict(df, model_id, time_var, target_var, y_hat, horizon, order, seasonal_order, trend):
    print(f"Processing {df[model_id].iloc[0]}")
    df = df.sort_values(time_var).reset_index()
    endog = df[target_var][:-horizon]
    # Construct the model
    # mod = sm.tsa.SARIMAX(endog, order=(1, 1, 1), trend='c')
    mod = sm.tsa.SARIMAX(endog, order=order, seasonal_order=seasonal_order, trend=trend)
    # Estimate the parameters
    res = mod.fit(disp=0)
    #print(res.summary())
    fcast = res.get_forecast(steps=horizon).summary_frame()    
    df[y_hat] , df['y_ci_lower'], df['y_ci_upper'] = np.nan, np.nan, np.nan    
    df.loc[-horizon:,y_hat], df.loc[-horizon:,'y_ci_lower'], df.loc[-horizon:,'y_ci_upper'] = fcast['mean'], fcast['mean_ci_lower'], fcast['mean_ci_upper']
    # Note: since we did not specify the alpha parameter, the
    # confidence level is at the default, 95%
    #print(fcast_res.summary_frame())
    return df


def time_series_approach(df, horizon, order, time_var, primary_key, y_hat, target_var, seasonal_order, trend):

    df['time_var'] = pd.to_datetime(df['time_var']) # Change to timestamp

    df_res_ts = df.groupby(primary_key).apply(lambda x: _ts_fit_predict(x,  primary_key, time_var, target_var, y_hat, horizon, order, seasonal_order, trend)).reset_index(drop=True)
    df_res_ts.dropna().groupby(primary_key).apply(_compute_accuracy, y=target_var, y_pred=y_hat)

    return df_res_ts


def ml_predict_item(df, fcst_start_date, primary_key, time_var, target_var, model, y_hat):
    print(f"Processing {df[primary_key].iloc[0]}")
    # test data
    df_test = df[df[time_var].values>=fcst_start_date].drop(columns=[time_var, target_var, primary_key]).astype(float)
    y_test = df[df[time_var].values>=fcst_start_date][target_var]
    dataset_test = lgb.Dataset(df_test, y_test)
    # prediction
    y_pred = model.predict(df_test, predict_disable_shape_check=True)
    df[y_hat] = np.nan
    df.loc[df[time_var].values>=fcst_start_date, y_hat] = y_pred
    return df


def _ml_fit_predict(df, time_var, primary_key, target_var, y_hat, fcst_start_date, params):    
    # training data
    df_train = df[df[time_var].values<fcst_start_date].drop(columns=[time_var, target_var, primary_key]).astype(float)
    y_train = df[df[time_var].values<fcst_start_date][target_var]
    dataset_train = lgb.Dataset(df_train, y_train)
    # fitting the model globally
    lgbm_model = lgb.train(params, train_set=dataset_train)
    # make predictions locally
    df = df.groupby(primary_key).apply(lambda x: ml_predict_item(df=df, fcst_start_date=fcst_start_date, primary_key=primary_key, time_var=time_var, target_var=target_var, model=lgbm_model, y_hat=y_hat)).reset_index(drop=True)    
    return df


def ml_approach(df, horizon, time_var, target_var, primary_key, y_hat, lgbm_params):

    fcst_start_date = pd.to_datetime(df[df[primary_key]=='6860_07#MERCADONA#VITORIA'][time_var].values[-horizon])
    df_res_ml = _ml_fit_predict(df=df, time_var=time_var, primary_key=primary_key, target_var=target_var, y_hat=y_hat,fcst_start_date=fcst_start_date, params=lgbm_params)
    df_res_ml = df_res_ml.dropna().groupby(primary_key).apply(_compute_accuracy, y=target_var, y_pred=y_hat).reset_index()
    return df_res_ml
