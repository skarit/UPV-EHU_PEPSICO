from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes.clean.shipments import clean_shipments
from .nodes.clean.promotions import clean_promotions
from .nodes.clean.holidays import clean_holidays

from .nodes.feature.shipments import feature_shipments
from .nodes.feature.promotions import feature_promotions
from .nodes.feature.holidays import feature_holidays

def create_pipeline(**kwargs) -> Pipeline:


    shipments_pipeline = pipeline(
        [
            node(
                func=clean_shipments,
                inputs=["shipments", "params:processing_params"],
                outputs="shipments_processed",
                name="clean_shipments_node",
            ),
            node(
                func=feature_shipments,
                inputs=["shipments_processed"],
                outputs="shipments_featured",
                name="feature_shipments_node",
            ),
        ]
    )

    promotions_pipeline = pipeline(
        [
            node(
                func=clean_promotions,
                inputs=["promotions", "params:processing_params"],
                outputs="promotions_processed",
                name="clean_promotions_node",
            ),
            node(
                func=feature_promotions,
                inputs=["promotions_processed"],
                outputs="promotions_featured",
                name="feature_promotions_node",
            ),
        ]
    )

    holidays_pipeline = pipeline(
        [
            node(
                func=clean_holidays,
                inputs=["holidays", "params:processing_params"],
                outputs="holidays_processed",
                name="clean_holidays_node",
            ), 
            node(
                func=feature_holidays,
                inputs=["holidays_processed"],
                outputs="holidays_featured",
                name="feature_holidays_node",
            ),
        ]
    )

    return pipeline(
        pipe=shipments_pipeline + promotions_pipeline + holidays_pipeline,
        namespace="data_processing",
        inputs=["shipments", "promotions", "holidays"],
        outputs=["shipments_featured", "promotions_featured", "holidays_featured"],
    )