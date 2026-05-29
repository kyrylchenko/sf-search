import numpy as np

from main_service.embedding.model import _normalized_numpy


class FakeTensor:
    def __init__(self, values: list[list[float]]) -> None:
        self.values = np.asarray(values, dtype=np.float32)

    def norm(self, dim: int, keepdim: bool) -> "FakeTensor":
        return FakeTensor(np.linalg.norm(self.values, axis=dim, keepdims=keepdim).tolist())

    def __truediv__(self, other: "FakeTensor") -> "FakeTensor":
        return FakeTensor((self.values / other.values).tolist())

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
