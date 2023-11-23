from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes.modeling.modeling import time_series_approach, ml_approach


def create_pipeline(**kwargs) -> Pipeline:


    modeling_pipeline = pipeline(
        [
            node(
                func=time_series_approach,
                # inputs=["model_input", "params:arima_model_options"],
                inputs=["model_input", "params:general_options.horizon",
                        "params:arima_model_options.order" , "params:general_options.time_var",
                        "params:general_options.primary_key", "params:general_options.y_hat",
                        "params:general_options.target_var", "params:arima_model_options.seasonal_order",
                        "params:arima_model_options.trend"],
                outputs="arima_results",
                name="arima_node",
            ),
            node(
                func=ml_approach,
                inputs=["model_input", "params:general_options.horizon","params:general_options.time_var",
                        "params:general_options.target_var","params:general_options.primary_key",
                        "params:general_options.y_hat", "params:lgb_model_options"],
                outputs="ml_results",
                name="ml_node",
            ),
        ]
    )

    return pipeline(
        pipe=modeling_pipeline,
        namespace="data_science",
        inputs=["model_input"],
        outputs=["arima_results", "ml_results"],
    )