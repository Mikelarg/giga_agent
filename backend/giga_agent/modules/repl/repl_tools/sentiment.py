from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import numpy as np
from langchain.tools import ToolRuntime

from giga_agent.core.agent.runtime_resolver import RuntimeResolver
from giga_agent.core.logging import get_logger
from giga_agent.embeddings.base import BaseEmbeddingRuntime

_MODELS_DIR = Path(__file__).resolve().parent / "models"
_MODEL_PREFIX = "sentiment_"
_MODEL_SUFFIX = ".npz"

logger = get_logger(__name__)


def _validate_texts(texts: list[str]) -> None:
    if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
        raise ValueError("All texts must be strings.")


class NumpySentimentModel:
    """Компактный линейный sentiment-классификатор без sklearn runtime."""

    def __init__(
        self,
        weights: np.ndarray,
        bias: np.ndarray,
        classes: np.ndarray,
    ) -> None:
        if weights.ndim != 2:
            raise ValueError("Sentiment model weights must be a 2D array.")
        if bias.shape != (weights.shape[0],):
            raise ValueError("Sentiment model bias shape does not match weights.")
        if classes.shape != (weights.shape[0],):
            raise ValueError("Sentiment model classes shape does not match weights.")
        self.weights = weights
        self.bias = bias
        self.classes = classes

    @classmethod
    def load(cls, path: Path) -> "NumpySentimentModel":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                weights=np.asarray(data["weights"], dtype=np.float64),
                bias=np.asarray(data["bias"], dtype=np.float64),
                classes=np.asarray(data["classes"]),
            )

    def predict_labels(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.ndim != 2 or matrix.shape[1] != self.weights.shape[1]:
            raise ValueError(
                "Embedding dimensions do not match the sentiment model: "
                f"got {matrix.shape}, expected (*, {self.weights.shape[1]})."
            )
        logits = matrix @ self.weights.T + self.bias
        return self.classes[np.argmax(logits, axis=1)]


def _build_model_key(file_path: Path) -> str | None:
    stem = file_path.stem
    if not stem.startswith(_MODEL_PREFIX):
        return None
    key = stem[len(_MODEL_PREFIX) :].strip()
    return key or None


def _model_paths() -> dict[str, Path]:
    return {
        key: path
        for path in _MODELS_DIR.glob(f"{_MODEL_PREFIX}*{_MODEL_SUFFIX}")
        if (key := _build_model_key(path)) is not None
    }


_SENTIMENT_MODELS: dict[str, NumpySentimentModel] = {}
_MODEL_LOAD_LOCK = threading.Lock()


def _load_sentiment_model(model_id: str) -> NumpySentimentModel:
    """Load only the requested classifier, once per process, on first use."""
    with _MODEL_LOAD_LOCK:
        if model_id in _SENTIMENT_MODELS:
            return _SENTIMENT_MODELS[model_id]
        paths = _model_paths()
        model_path = paths.get(model_id)
        if model_path is None:
            available = ", ".join(sorted(paths)) or "нет"
            raise ValueError(
                "Для вашей embedding модели не настроена sentiment модель. "
                "Попросите администратора настроить sentiment модель под вашу embedding модель. "
                f"Текущая embedding модель: '{model_id}'. "
                f"Доступные sentiment модели: {available}."
            )
        started = time.monotonic()
        model = NumpySentimentModel.load(model_path)
        _SENTIMENT_MODELS[model_id] = model
        logger.info(
            "Loaded sentiment model",
            key=model_id,
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )
        return model


async def _resolve_user_embeddings(
    tool_runtime: ToolRuntime,
) -> tuple[BaseEmbeddingRuntime, str]:
    resolver = RuntimeResolver.from_config(tool_runtime.config)
    embedding_runtime = await resolver.get_embedding_runtime()
    return embedding_runtime, embedding_runtime.model_id


async def predict_sentiments(
    texts: list[str],
    tool_runtime: ToolRuntime | None = None,
) -> list[str]:
    """Определяет настроение текста в одну из этих меток:
    ["positive", "negative", "neutral"]
    Используй в том случае, если нужно определить настроение массива текстов
    Помни, что ты должен вызывать функцию только с именованными агрументами.

    Пример: predict_sentiments(texts=['текст'])

    Args:
        texts: Список текстов на анализ
        tool_runtime: Runtime текущего вызова инструмента

    """
    _validate_texts(texts)
    if tool_runtime is None:
        raise ValueError("tool_runtime is required for predict_sentiments.")

    embedding_runtime, embedding_model_id = await _resolve_user_embeddings(tool_runtime)
    clf = await asyncio.to_thread(_load_sentiment_model, embedding_model_id)

    embeddings = await embedding_runtime.get_embeddings()
    embs = await embeddings.aembed_documents(texts)
    matrix = np.vstack(embs).astype("float32")
    labels = clf.predict_labels(matrix)
    return [str(label) for label in labels]


async def get_embeddings(
    texts: list[str],
    tool_runtime: ToolRuntime | None = None,
) -> list[list[float]]:
    """Получает эмбединги для списка текстов
    (можно использовать для кластеризации вместе с umap и hdbscan)

    Args:
        texts: Список текстов
        tool_runtime: Runtime текущего вызова инструмента

    """
    _validate_texts(texts)
    if tool_runtime is None:
        raise ValueError("tool_runtime is required for get_embeddings.")

    embedding_runtime, _ = await _resolve_user_embeddings(tool_runtime)
    embeddings = await embedding_runtime.get_embeddings()
    embs = await embeddings.aembed_documents(texts)
    return [list(map(float, row)) for row in embs]
