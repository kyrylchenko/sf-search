from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Final


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    area_to_process_geojson_filename: str = Field(default="target.geojson")
    db_user: str = Field()
    db_password: str = Field()
    db_host: str = Field()
    db_port: int = Field()
    db_name: str = Field()


CONFIG: Final[Settings] = Settings()  # type: ignore
