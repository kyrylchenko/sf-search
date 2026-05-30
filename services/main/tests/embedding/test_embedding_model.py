import numpy as np

from main_service.embedding import model as embedding_model
from main_service.embedding.model import TransformersSiglipEmbedder, _normalized_numpy


class FakeTensor:
    def __init__(self, values: list[list[float]]) -> None:
        self.values = np.asarray(values, dtype=np.float32)

    def norm(self, dim: int, keepdim: bool) -> "FakeTensor":
        return FakeTensor(np.linalg.norm(self.values, axis=dim, keepdims=keepdim).tolist())

    def __truediv__(self, other: "FakeTensor") -> "FakeTensor":
        return FakeTensor((self.values / other.values).tolist())

    def to(self, _device: str) -> "FakeTensor":
        return self

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class FakeOutputWithPooling:
    def __init__(self) -> None:
        self.pooler_output = FakeTensor([[3.0, 4.0]])


def test_normalized_numpy_uses_pooler_output_from_transformers_model_output() -> None:
    vector = _normalized_numpy(FakeOutputWithPooling(), torch_module=None)

    assert vector.dtype == np.float32
    assert np.allclose(vector, np.array([0.6, 0.8], dtype=np.float32))


class FakeBackend:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class FakeBackends:
    def __init__(self, mps_available: bool) -> None:
        self.mps = FakeBackend(mps_available)


class FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class FakeTorch:
    def __init__(self, *, cuda_available: bool, mps_available: bool) -> None:
        self.cuda = FakeCuda(cuda_available)
        self.backends = FakeBackends(mps_available)


def test_select_device_prefers_mps_before_cpu() -> None:
    torch_module = FakeTorch(cuda_available=False, mps_available=True)

    assert embedding_model._select_device(torch_module) == "mps"


class FakeProcessor:
    def __init__(self) -> None:
        self.text_kwargs: dict[str, object] | None = None

    def __call__(self, **kwargs: object) -> dict[str, FakeTensor]:
        self.text_kwargs = kwargs
        return {"input_ids": FakeTensor([[1.0, 2.0]])}


class FakeNoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


class FakeTorchForText:
    def no_grad(self) -> FakeNoGrad:
        return FakeNoGrad()


class FakeModel:
    device = "cpu"

    def get_text_features(self, **kwargs: object) -> FakeTensor:
        return FakeTensor([[3.0, 4.0]])


def test_siglip_text_embedding_uses_trained_max_length_padding() -> None:
    processor = FakeProcessor()
    embedder = TransformersSiglipEmbedder(model_id="test")
    embedder._processor = processor
    embedder._model = FakeModel()
    embedder._torch = FakeTorchForText()

    embedder.embed_text("orange car")

    assert processor.text_kwargs is not None
    assert processor.text_kwargs["text"] == ["orange car"]
    assert processor.text_kwargs["padding"] == "max_length"
    assert processor.text_kwargs["truncation"] is True
    assert processor.text_kwargs["return_tensors"] == "pt"
