from typing import Final

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
    discovery_concurrency: int = Field(default=20)
    max_attempts: int = Field(default=5)
    pano_download_concurrency: int = Field(default=5)
    pano_download_storage_dir: str = Field(default=".local/panoramas")
    nats_url: str = Field(default="nats://localhost:4222")
    pano_download_stream: str = Field(default="PANO_DOWNLOADS")
    pano_download_subject: str = Field(default="pano.download.requested")
    pano_processing_stream: str = Field(default="PANO_PROCESSING")
    pano_processing_subject: str = Field(default="pano.processing.requested")
    pano_downloader_consumer: str = Field(default="pano-downloader")
    max_processing_queue_depth: int = Field(default=50)


CONFIG: Final[Settings] = Settings()  # type: ignore
