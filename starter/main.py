# Put the code for your API here.

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List

app = FastAPI(
    title="Census Income Inference API",
    description="A simple API to return a greeting on GET and perform model inference on POST.",
    version="1.0"
)

class InferenceInput(BaseModel):
    features: List[float] = Field(
        ...,
        example=[2.0, 1.5, 1.0, 0.5, 1.0],
        description="A list of numerical features for the model inference."
    )

@app.get("/", summary="Greeting Endpoint", tags=["Root"])
def read_root() -> dict:
    """
    Returns a greeting message.
    """
    return {"message": "Welcome to the Census Income Inference API!"}

@app.post("/inference", summary="Run Model Inference", tags=["Inference"])
def run_inference(input_data: InferenceInput) -> dict:
    """
    Performs a dummy model inference on the input features.
    
    This example uses a simple threshold on the sum of features.
    """
    # Dummy inference: if the sum of features is greater than a threshold, return 1; otherwise, return 0.
    threshold = 5.0
    prediction = 1 if sum(input_data.features) > threshold else 0
    return {"prediction": prediction}