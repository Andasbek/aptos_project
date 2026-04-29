from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_name: str
    checkpoint_path: str
    checkpoint_exists: bool
    device: str
    classes: dict[int, str]


class PredictionResponse(BaseModel):
    predicted_class: int
    class_name: str
    confidence: float
    probabilities: dict[str, float]

