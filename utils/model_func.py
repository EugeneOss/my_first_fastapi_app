from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import json
import re
from typing import List

import pandas as pd
import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as transforms
from transformers import AutoModel, AutoTokenizer

# --- Базовые пути (не зависят от текущей рабочей директории) ---
ROOT = Path(__file__).resolve().parents[1]           # .../my_first_fastapi_app/
DATA_DIR = ROOT / "back" / "data"
MODELS_DIR = ROOT / "back" / "models"

# --- Устройство для инференса ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------- Классы/лейблы ----------------------
@lru_cache
def load_classes() -> List[str]:
    """Загружает список меток классов один раз и кэширует результат."""
    with (DATA_DIR / "labels_wheather.json").open(encoding="utf-8") as f:
        labels = json.load(f)
    if not isinstance(labels, list):
        raise ValueError("labels_wheather.json должен содержать список строк.")
    return labels


def class_id_to_label(i: int) -> str:
    """Преобразует индекс класса в название класса."""
    labels = load_classes()
    if i < 0 or i >= len(labels):
        raise IndexError(f"Индекс класса {i} вне диапазона [0, {len(labels)-1}].")
    return labels[i]


# ---------------------- Модель для изображений ----------------------
def load_pt_model_weather():
    """
    Загружает ResNet50, добавляет обуенные веса.
    Возвращает модель в режиме eval() и на нужном устройстве.
    """
    model = resnet50(num_classes=11)
    state = torch.load(MODELS_DIR / "model_resnet50_wheather.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


# ---------------------- Токенайзер и текстовая модель ----------------------
def load_rubert_tokenizer():
    """Загружает токенайзер для 'cointegrated/rubert-tiny2'."""
    return AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")


def load_pt_model_text():
    """
    Загружает RuBERT-tiny2 + верхушку LSTM-классификатора и твои веса.
    Модель возвращается в режиме eval() и на нужном устройстве.
    """
    rubert_model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")

    class BERT_LSTM_Classifier(nn.Module):
        def __init__(self, bert_model, hidden_dim=128, lstm_layers=1, dropout=0.3):
            super().__init__()
            self.bert = bert_model
            # В rubert-tiny2 hidden_size = 312
            self.lstm = nn.LSTM(
                input_size=312,
                hidden_size=hidden_dim,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=True,
            )
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_dim * 2, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, input_ids, attention_mask):
            with torch.no_grad():
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            lstm_input = outputs.last_hidden_state           # [B, T, 312]
            lstm_out, _ = self.lstm(lstm_input)              # [B, T, 2*hidden]
            out = lstm_out[:, -1, :]                         # берём последний токен
            out = self.dropout(out)
            out = self.classifier(out)                       # [B, 1]
            return self.sigmoid(out).squeeze(-1)             # [B]

    model = BERT_LSTM_Classifier(bert_model=rubert_model)
    state = torch.load(MODELS_DIR / "bert_lstm_model_polic.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


# ---------------------- Преобразования ----------------------
def transform_image(img):
    """
    Input: PIL.Image
    Returns: тензор [1, 3, 224, 224] (на CPU; перенос на device делать снаружи)
    """
    trnsfrms = transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return trnsfrms(img).unsqueeze(0)


def preprocessing_text(x: pd.Series) -> pd.Series:
    """
    Базовая очистка текста. При желании адаптируй под свою модель.
    """
    x = x.str.replace(r'<[^>]+>', ' ', regex=True)           # убрать HTML-теги
    x = x.str.replace(r'#\S+', '', regex=True)               # хэштеги
    x = x.str.replace(r'http\S+', '', regex=True)            # ссылки
    x = x.str.replace(r'@\s*[^ ]+|@', '', regex=True)        # упоминания
    x = x.str.replace(r'[^\w\s]', ' ', regex=True)           # пунктуация -> пробел
    x = x.str.replace(r'\s+', ' ', regex=True)               # сжать пробелы
    x = x.str.replace('br ', ' ', regex=False)               # спец. случай
    x = x.str.replace(r'[^А-Яа-яЁё\s]', ' ', regex=True)
    x = x.str.replace(r'\b[А-яА-ЯЁё]\b', '', regex=True)     # одиночные буквы
    x = x.str.lower()
    x = x.str.replace(r'\bне\s+(\w+)\b', r'не_\1', regex=True)
    x = x.str.replace(r'\s+', ' ', regex=True).str.strip()
    return x
