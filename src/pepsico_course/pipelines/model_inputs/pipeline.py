from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from .nodes.model_input import create_model_input

def create_pipeline(**kwargs) -> Pipeline:


    pipe = pipeline(
        [
            node(
                func=create_model_input,
                inputs=["shipments_featured", "promotions_featured", "holidays_featured"],
                outputs=["model_input", "facts_data"],
                name="create_model_input",
            ),
        ]
    )

    return pipeline(
        pipe=pipe,
        namespace="mrd_creation",
        inputs=["shipments_featured", "promotions_featured", "holidays_featured"],
        outputs=["model_input", "facts_data"],
    )