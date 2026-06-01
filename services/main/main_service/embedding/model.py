from pathlib import Path
import logging
from time import perf_counter
from typing import Protocol

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ImageTextEmbedder(Protocol):
    def embed_image(self, image_path: Path) -> np.ndarray:
        ...

    def embed_images(self, image_paths: list[Path]) -> list[np.ndarray]:
        ...

    def embed_text(self, text: str) -> np.ndarray:
        ...


class TransformersSiglipEmbedder:
    def __init__(
        self,
        *,
        model_id: str,
        revision: str = "main",
        device: str | None = None,
        dtype: str = "float16",
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.dtype = dtype
        self._model = None
        self._processor = None
        self._torch = None

    def embed_image(self, image_path: Path) -> np.ndarray:
        return self.embed_images([image_path])[0]

    def embed_images(self, image_paths: list[Path]) -> list[np.ndarray]:
        started_at = perf_counter()
        self._load()
        assert self._torch is not None
        assert self._processor is not None
        assert self._model is not None
        if not image_paths:
            return []
        logger.info(
            "embedding_model_image_batch_encode_start images=%s model_id=%s",
            len(image_paths),
            self.model_id,
        )
        images = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                images.append(image.convert("RGB"))
        inputs = self._processor(images=images, return_tensors="pt")
        inputs = {key: value.to(self._model.device) for key, value in inputs.items()}
        with self._torch.no_grad():
            features = self._model.get_image_features(**inputs)
        vectors = _normalized_batch_numpy(features, self._torch)
        logger.info(
            (
                "embedding_model_image_batch_encode_complete images=%s model_id=%s "
                "dimension=%s seconds=%.3f"
            ),
            len(image_paths),
            self.model_id,
            vectors.shape[1],
            perf_counter() - started_at,
        )
        return [vector for vector in vectors]

    def embed_text(self, text: str) -> np.ndarray:
        started_at = perf_counter()
        self._load()
        assert self._torch is not None
        assert self._processor is not None
        assert self._model is not None
        logger.info(
            "embedding_model_text_encode_start model_id=%s text_length=%s",
            self.model_id,
            len(text),
        )
        inputs = self._processor(
            text=[text],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self._model.device) for key, value in inputs.items()}
        with self._torch.no_grad():
            features = self._model.get_text_features(**inputs)
        vector = _normalized_numpy(features, self._torch)
        logger.info(
            "embedding_model_text_encode_complete model_id=%s dimension=%s seconds=%.3f",
            self.model_id,
            vector.shape[0],
            perf_counter() - started_at,
        )
        return vector

    def _load(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        started_at = perf_counter()
        logger.info(
            "embedding_model_load_start model_id=%s revision=%s dtype=%s",
            self.model_id,
            self.revision,
            self.dtype,
        )
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Embedding requires optional dependencies: transformers and torch."
            ) from exc

        torch_dtype = _torch_dtype(torch, self.dtype)
        processor = AutoProcessor.from_pretrained(
            self.model_id,
            revision=self.revision,
        )
        model = AutoModel.from_pretrained(
            self.model_id,
            revision=self.revision,
            torch_dtype=torch_dtype,
        )
        device = _resolve_device(torch, self.device)
        model = model.to(device)
        model.eval()
        self._torch = torch
        self._processor = processor
        self._model = model
        first_param = next(model.parameters())
        logger.info(
            (
                "embedding_model_load_complete model_id=%s requested_device=%s "
                "actual_device=%s dtype=%s seconds=%.3f"
            ),
            self.model_id,
            self.device or "auto",
            device,
            first_param.dtype,
            perf_counter() - started_at,
        )


def _torch_dtype(torch_module: object, dtype: str) -> object:
    if dtype == "float16":
        return getattr(torch_module, "float16")
    if dtype == "bfloat16":
        return getattr(torch_module, "bfloat16")
    if dtype == "float32":
        return getattr(torch_module, "float32")
    raise ValueError(f"Unsupported torch dtype: {dtype}")


def _select_device(torch_module: object) -> str:
    if torch_module.cuda.is_available():
        return "cuda"

    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"

    return "cpu"


def _resolve_device(torch_module: object, requested_device: str | None) -> str:
    if requested_device is None or requested_device == "auto":
        return _select_device(torch_module)
    return requested_device


def _normalized_numpy(features: object, torch_module: object) -> np.ndarray:
    return _normalized_batch_numpy(features, torch_module)[0]


def _normalized_batch_numpy(features: object, torch_module: object) -> np.ndarray:
    selected = _select_feature_tensor(features)
    normalized = selected / selected.norm(dim=-1, keepdim=True)
    return normalized.detach().cpu().numpy().astype(np.float32)


def _select_feature_tensor(features: object) -> object:
    if hasattr(features, "pooler_output"):
        return features.pooler_output
    return features
