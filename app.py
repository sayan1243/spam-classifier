from functools import lru_cache
import pickle
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from sms_preprocessing import ensure_nltk_resources, transform_text


PROJECT_DIR = Path(__file__).resolve().parent
STATIC_DIR = PROJECT_DIR / "static"

app = FastAPI(
    title="SMS Spam Classifier",
    description="Classify a text message as spam or not spam.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class PredictionRequest(BaseModel):
    message: str = Field(min_length=1, max_length=5000)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, message: str) -> str:
        message = message.strip()
        if not message:
            raise ValueError("Message cannot be blank.")
        return message


class PredictionResponse(BaseModel):
    prediction: Literal["spam", "ham"]
    is_spam: bool


@lru_cache(maxsize=1)
def load_artifacts():
    ensure_nltk_resources()
    with (PROJECT_DIR / "vectorizer.pkl").open("rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    with (PROJECT_DIR / "model.pkl").open("rb") as model_file:
        model = pickle.load(model_file)
    return vectorizer, model


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", include_in_schema=False)
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    vectorizer, model = load_artifacts()
    transformed_message = transform_text(request.message)
    features = vectorizer.transform([transformed_message])
    is_spam = int(model.predict(features)[0]) == 1
    return PredictionResponse(
        prediction="spam" if is_spam else "ham",
        is_spam=is_spam,
    )
