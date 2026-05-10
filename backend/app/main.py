from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .inference import predict_image
from .llm import (
    LLMConfigurationError,
    LLMRequestError,
    chat_about_prediction,
    generate_explanation,
    llm_available,
    supported_languages,
)
from .model_loader import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL_NAME,
    DEVICE,
    available_models,
)
from .schemas import (
    ChatRequest,
    ChatResponse,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    ModelInfoResponse,
    ModelOption,
    PredictionResponse,
)
from .utils import class_mapping, read_image


app = FastAPI(
    title="APTOS Diabetic Retinopathy API",
    description="Inference API for APTOS 2019 diabetic retinopathy classification.",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        default_model=DEFAULT_MODEL_NAME,
        available_models=[ModelOption(**option) for option in available_models()],
        device=str(DEVICE),
        classes=class_mapping(),
        llm_enabled=llm_available(),
        supported_languages=supported_languages(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL_NAME),
) -> PredictionResponse:
    if model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown model '{model}'. Available: {list(AVAILABLE_MODELS)}"
            ),
        )

    if file.content_type is not None and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    try:
        image = read_image(image_bytes)
        prediction = predict_image(image, model_name=model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {exc}",
        ) from exc

    return PredictionResponse(**prediction)


@app.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest) -> ExplainResponse:
    try:
        explanation = generate_explanation(
            request.prediction.model_dump(),
            language=request.language,
        )
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMRequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ExplainResponse(explanation=explanation, language=request.language)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    user_messages = [message.model_dump() for message in request.messages]
    prediction = (
        request.prediction.model_dump() if request.prediction is not None else None
    )

    try:
        reply = chat_about_prediction(
            user_messages,
            prediction,
            language=request.language,
        )
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMRequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ChatResponse(reply=reply, language=request.language)
