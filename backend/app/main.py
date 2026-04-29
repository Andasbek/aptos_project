from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .inference import predict_image
from .model_loader import CHECKPOINT_PATH, DEVICE, MODEL_NAME
from .schemas import HealthResponse, ModelInfoResponse, PredictionResponse
from .utils import class_mapping, read_image


app = FastAPI(
    title="APTOS Diabetic Retinopathy API",
    description="Inference API for APTOS 2019 diabetic retinopathy classification.",
    version="1.0.0",
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
        model_name=MODEL_NAME,
        checkpoint_path=str(CHECKPOINT_PATH),
        checkpoint_exists=CHECKPOINT_PATH.exists(),
        device=str(DEVICE),
        classes=class_mapping(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if file.content_type is not None and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    try:
        image = read_image(image_bytes)
        prediction = predict_image(image)
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

