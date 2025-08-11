from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from io import BytesIO

import pandas as pd
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette import status
from PIL import Image

from utils.model_func import (
    class_id_to_label,
    load_pt_model_weather,
    load_pt_model_text,
    transform_image,
    preprocessing_text,
    load_rubert_tokenizer,
    device,  # используем то же устройство, что и в utils
)

logger = logging.getLogger("uvicorn.error")


# ---------------------- Pydantic-схемы ----------------------
class WeatherResponse(BaseModel):
    class_name: str
    class_prob: float


class TextInput(BaseModel):
    text: str


class TextResponse(BaseModel):
    label: str
    prob: float


# ---------------------- Жизненный цикл приложения ----------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    На старте загружаем модели и токенайзер, на остановке освобождаем.
    """
    app.state.text_model = load_pt_model_text()
    app.state.weather_model = load_pt_model_weather()
    app.state.tokenizer = load_rubert_tokenizer()
    logger.info("Models and tokenizer loaded (device=%s)", device)
    yield
    # Явный del необязателен, но пусть будет:
    del app.state.text_model, app.state.weather_model, app.state.tokenizer
    logger.info("Models and tokenizer unloaded")


app = FastAPI(title="My First FastAPI App", lifespan=lifespan)

# ---------------------- Маршруты ----------------------
@app.get("/")
def root():
    return {
        "message": "Привет! Это моё первое FastAPI-приложение. "
                   "Можешь протестировать классификацию погоды по фото и отзывов по тексту.",
        "endpoints": {
            "weather": "/clf_weather",
            "text": "/clf_text",
            "docs": "/docs",
        },
    }


@app.post("/clf_weather", response_model=WeatherResponse, status_code=status.HTTP_200_OK)
async def classify_image(request: Request, file: UploadFile = File(...)):
    """
    Классификация изображения (погода).
    Принимает JPEG/PNG/WEBP. Возвращает имя класса и вероятность.
    """
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(415, f"Unsupported image type: {file.content_type}")


    raw_bytes = await file.read()
    try:
        image = Image.open(BytesIO(raw_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Cannot read image: {e}") from e

    tensor = transform_image(image).to(device)  # [1,3,224,224]
    logger.info("Transformed image shape: %s", tuple(tensor.shape))

    model = request.app.state.weather_model
    with torch.no_grad():
        logits = model(tensor)                          # [1, num_classes]
        probs = torch.softmax(logits, dim=1)[0]         # [num_classes]
        top_idx = int(torch.argmax(probs).item())
        prob = float(probs[top_idx].item())

    return WeatherResponse(
        class_name=class_id_to_label(top_idx),
        class_prob=prob,
    )


@app.post("/clf_text", response_model=TextResponse, status_code=status.HTTP_200_OK)
async def clf_text(request: Request, data: TextInput):
    """
    Классификация отзыва (позитив/негатив).
    Возвращает метку и вероятность класса (p in [0,1]).
    """
    clean_text = preprocessing_text(pd.Series([data.text])).iloc[0]
    tokenizer = request.app.state.tokenizer

    encoded = tokenizer(
        clean_text,
        max_length=64,
        truncation=True,
        padding="max_length",
    )

    input_ids = torch.tensor([encoded["input_ids"]], device=device)
    attention_mask = torch.tensor([encoded["attention_mask"]], device=device)

    model = request.app.state.text_model
    with torch.no_grad():
        prob = float(model(input_ids, attention_mask).item())

    label = "Отзыв позитивен." if prob >= 0.5 else "Отзыв негативен."
    return TextResponse(label=label, prob=prob)


# ---------------------- Точка входа ----------------------
if __name__ == "__main__":
    # запуск: python main.py
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
