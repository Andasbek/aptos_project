from typing import Literal

from pydantic import BaseModel, Field


Language = Literal["en", "ru", "kk"]


class HealthResponse(BaseModel):
    status: str


class ModelOption(BaseModel):
    name: str
    checkpoint_path: str
    checkpoint_exists: bool
    is_default: bool


class ModelInfoResponse(BaseModel):
    default_model: str
    available_models: list[ModelOption]
    device: str
    classes: dict[int, str]
    llm_enabled: bool
    supported_languages: list[Language]


class PredictionResponse(BaseModel):
    predicted_class: int
    class_name: str
    confidence: float
    probabilities: dict[str, float]
    model_name: str


class ExplainRequest(BaseModel):
    prediction: PredictionResponse
    language: Language = "en"


class ExplainResponse(BaseModel):
    explanation: str
    language: Language


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=20)
    prediction: PredictionResponse | None = None
    language: Language = "en"


class ChatResponse(BaseModel):
    reply: str
    language: Language
