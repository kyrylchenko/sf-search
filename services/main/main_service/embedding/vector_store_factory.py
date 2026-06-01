from pathlib import Path

from main_service.config import Settings
from main_service.db.services.panorama_view_embedding_service import EmbeddingModelSpec
from main_service.embedding.qdrant_store import QdrantVectorStore
from main_service.embedding.vector_store import LocalHnswVectorStore, VectorStore


def create_vector_store(
    *,
    settings: Settings,
    model_spec: EmbeddingModelSpec,
    vector_store_kind: str | None = None,
    vector_store_dir: Path | None = None,
    qdrant_url: str | None = None,
    qdrant_collection: str | None = None,
) -> VectorStore:
    kind = vector_store_kind or settings.embedding_vector_store_kind
    if kind == "qdrant":
        return QdrantVectorStore(
            url=qdrant_url or settings.qdrant_url,
            collection_name=qdrant_collection or settings.qdrant_collection,
            dimension=model_spec.embedding_dimension,
            vector_on_disk=settings.qdrant_vector_on_disk,
            hnsw_on_disk=settings.qdrant_hnsw_on_disk,
            on_disk_payload=settings.qdrant_on_disk_payload,
            upsert_wait=settings.qdrant_upsert_wait,
            timeout_seconds=settings.qdrant_timeout_seconds,
        )
    if kind == "local_hnsw":
        return LocalHnswVectorStore(
            root_dir=Path(vector_store_dir or settings.embedding_vector_store_dir),
            model_id=model_spec.model_id,
            dimension=model_spec.embedding_dimension,
        )
    raise ValueError(f"Unsupported vector store kind: {kind}")
