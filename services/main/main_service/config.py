from typing import Final, Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    area_to_process_geojson_filepath: str = Field(
        default="target.geojson",
        validation_alias=AliasChoices(
            "AREA_TO_PROCESS_GEOJSON_FILEPATH",
            "AREA_TO_PROCESS_GEOJSON_FILENAME",
            "area_to_process_geojson_filepath",
        ),
    )
    db_user: str = Field()
    db_password: str = Field()
    db_host: str = Field()
    db_port: int = Field()
    db_name: str = Field()
    db_driver: str = Field(default="postgresql+psycopg")
    map_tiles_zoom: int = Field(default=17)
    discovery_tile_order: Literal["sequential", "random"] = Field(default="random")
    discovery_tile_random_seed: int | None = Field(default=None)
    discovery_concurrency: int = Field(default=20)
    max_attempts: int = Field(default=5)
    pano_download_concurrency: int = Field(default=5)
    pano_download_storage_dir: str = Field(default=".local/panoramas")
    nats_url: str = Field(default="nats://localhost:4222")
    pano_download_stream: str = Field(default="PANO_DOWNLOADS")
    pano_download_subject: str = Field(default="pano.download.requested")
    pano_processing_stream: str = Field(default="PANO_PROCESSING")
    pano_processing_subject: str = Field(default="pano.processing.requested")
    pano_embedding_stream: str = Field(default="PANO_EMBEDDING")
    pano_embedding_subject: str = Field(default="pano.embedding.requested")
    pano_downloader_consumer: str = Field(default="pano-downloader")
    pano_processing_consumer: str = Field(default="pano-processor")
    pano_embedding_consumer: str = Field(default="pano-embedder")
    max_downloader_queue_depth: int = Field(default=1000)
    max_processing_queue_depth: int = Field(default=50)
    max_embedding_queue_depth: int = Field(default=100)
    pano_viewsets_dir: str = Field(default="../../docs/data/viewsets")
    pano_view_storage_dir: str = Field(default=".local/panorama-view-tmp")
    pano_processing_concurrency: int = Field(default=4)
    pano_view_max_render_concurrency: int = Field(default=4)
    pano_view_render_scale: int = Field(default=2)
    pano_view_output_format: str = Field(default="jpeg")
    pano_view_jpeg_quality: int = Field(default=95)
    pano_embedding_concurrency: int = Field(default=1)
    embedding_model_provider: str = Field(default="transformers")
    embedding_model_id: str = Field(default="google/siglip2-so400m-patch14-384")
    embedding_model_revision: str = Field(default="main")
    embedding_preprocess_version: str = Field(default="siglip2-384-rgb-v1")
    embedding_dimension: int = Field(default=1152)
    embedding_dtype: str = Field(default="float16")
    embedding_device: str = Field(default="auto")
    embedding_batch_size: int = Field(default=1)
    embedding_vector_store_dir: str = Field(default=".local/embedding-indexes")
    embedding_vector_store_kind: str = Field(default="qdrant")
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection: str = Field(default="panorama_view_embeddings_siglip2")
    qdrant_vector_on_disk: bool = Field(default=True)
    qdrant_hnsw_on_disk: bool = Field(default=False)
    qdrant_on_disk_payload: bool = Field(default=True)
    qdrant_upsert_wait: bool = Field(default=True)
    qdrant_timeout_seconds: float = Field(default=30.0)
    observability_enabled: bool = Field(default=False)
    otel_exporter_otlp_endpoint: str = Field(default="http://localhost:4317")
    otel_exporter_otlp_insecure: bool = Field(default=True)
    otel_exporter_otlp_timeout_seconds: float = Field(default=10.0)
    otel_metric_export_interval_millis: int = Field(default=10_000)
    otel_service_version: str = Field(default="local")
    deployment_environment: str = Field(default="local")
    monitoring_interval_seconds: float = Field(default=15.0)
    log_level: str = Field(default="INFO")
    service_idle_sleep_seconds: float = Field(default=5.0)


CONFIG: Final[Settings] = Settings()  # type: ignore
