"""
app.py - A minimal FastAPI application for Census Income Inference.
Adjust your Render start command to:
    uvicorn app:app --host 0.0.0.0 --port $PORT
if your file is named app.py
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List

app = FastAPI(
    title="Census Income Inference API",
    description="This API provides a greeting at the root (GET) and model inference at the /inference endpoint (POST).",
    version="1.0"
)


class InferenceInput(BaseModel):
    """
    Pydantic model for inference requests.
    """
    features: List[float] = Field(
        ...,
        example=[2.0, 1.5, 1.0, 0.5, 1.0],
        description="A list of numerical features used for model inference."
    )


@app.get("/", summary="Greeting Endpoint", tags=["Root"])
def read_root() -> dict:
    """
    GET endpoint that returns a greeting message.
    """
    return {"message": "Welcome to the Census Income Inference API!"}


@app.post("/inference", summary="Run Model Inference", tags=["Inference"])
def run_inference(input_data: InferenceInput) -> dict:
    """
    POST endpoint that simulates a model inference.

    For this example, the inference is dummy:
    If the sum of the features is greater than a threshold (5.0), it returns 1; otherwise, 0.
    """
    threshold = 5.0
    prediction = 1 if sum(input_data.features) > threshold else 0
    return {"prediction": prediction}