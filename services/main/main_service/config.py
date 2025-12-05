from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Final


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    area_to_process_geojson_filename: str = Field(default="target.geojson")


CONFIG: Final[Settings] = Settings()
