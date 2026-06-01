import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from main_service.config import CONFIG
from main_service.embedding.model import TransformersSiglipEmbedder
from main_service.logging_config import configure_cli_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test local embedding runtime.")
    parser.add_argument("--tiles-dir", default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--log-level", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = CONFIG
    configure_cli_logging(args.log_level or settings.log_level)
    tiles_dir = Path(args.tiles_dir or settings.pano_view_storage_dir)
    paths = sorted(tiles_dir.glob("**/*.jpg"))[: args.limit]
    if not paths:
        raise SystemExit(f"No .jpg tiles found under {tiles_dir}")
    embedder = TransformersSiglipEmbedder(
        model_id=args.model_id or settings.embedding_model_id,
        revision=args.model_revision or settings.embedding_model_revision,
        dtype=args.dtype or settings.embedding_dtype,
        device=_device_or_none(args.device or settings.embedding_device),
    )
    batch_size = max(1, args.batch_size or settings.embedding_batch_size)
    started_at = perf_counter()
    timings: list[float] = []
    for batch in _chunks(paths, batch_size):
        batch_started_at = perf_counter()
        vectors = embedder.embed_images(batch)
        timings.append(perf_counter() - batch_started_at)
        if len(vectors) != len(batch):
            raise RuntimeError("Embedder returned the wrong number of vectors.")
    total_seconds = perf_counter() - started_at
    assert embedder._model is not None
    assert embedder._torch is not None
    first_param = next(embedder._model.parameters())
    cuda = embedder._torch.cuda
    payload = {
        "batch_size": batch_size,
        "cuda_available": bool(cuda.is_available()),
        "cuda_device_name": cuda.get_device_name(0) if cuda.is_available() else None,
        "first_param_device": str(first_param.device),
        "first_param_dtype": str(first_param.dtype),
        "images": len(paths),
        "mean_batch_seconds": mean(timings),
        "model_device": str(embedder._model.device),
        "requested_device": args.device or settings.embedding_device,
        "requested_dtype": args.dtype or settings.embedding_dtype,
        "tiles_per_minute": len(paths) / total_seconds * 60,
        "total_seconds": total_seconds,
    }
    print(json.dumps(payload, sort_keys=True))


def _chunks[T](items: list[T], size: int) -> list[list[T]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _device_or_none(device: str | None) -> str | None:
    return None if device is None or device == "auto" else device


if __name__ == "__main__":
    main()
