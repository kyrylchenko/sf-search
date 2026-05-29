from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image


class ImageTextEmbedder(Protocol):
    def embed_image(self, image_path: Path) -> np.ndarray:
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
        self._load()
        assert self._torch is not None
        assert self._processor is not None
        assert self._model is not None
        with Image.open(image_path) as image:
            inputs = self._processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {key: value.to(self._model.device) for key, value in inputs.items()}
        with self._torch.no_grad():
            features = self._model.get_image_features(**inputs)
        return _normalized_numpy(features, self._torch)

    def embed_text(self, text: str) -> np.ndarray:
        self._load()
        assert self._torch is not None
        assert self._processor is not None
        assert self._model is not None
        inputs = self._processor(text=[text], padding=True, return_tensors="pt")
        inputs = {key: value.to(self._model.device) for key, value in inputs.items()}
        with self._torch.no_grad():
            features = self._model.get_text_features(**inputs)
        return _normalized_numpy(features, self._torch)

    def _load(self) -> None:
        if self._model is not None and self._processor is not None:
            return
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
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        self._torch = torch
        self._processor = processor
        self._model = model


def _torch_dtype(torch_module: object, dtype: str) -> object:
    if dtype == "float16":
        return getattr(torch_module, "float16")
    if dtype == "bfloat16":
        return getattr(torch_module, "bfloat16")
    if dtype == "float32":
        return getattr(torch_module, "float32")
    raise ValueError(f"Unsupported torch dtype: {dtype}")


def _normalized_numpy(features: object, torch_module: object) -> np.ndarray:
    selected = _select_feature_tensor(features)
    normalized = selected / selected.norm(dim=-1, keepdim=True)
    return normalized.detach().cpu().numpy()[0].astype(np.float32)


def _select_feature_tensor(features: object) -> object:
    if hasattr(features, "pooler_output"):
        return features.pooler_output
    return features
