from main_service.config import Settings


def test_settings_accepts_existing_geojson_filepath_env_name() -> None:
    settings = Settings(
        _env_file=None,
        AREA_TO_PROCESS_GEOJSON_FILEPATH="city.geojson",
        db_user="user",
        db_password="password",
        db_host="localhost",
        db_port=3306,
        db_name="sf_search",
    )

    assert settings.area_to_process_geojson_filepath == "city.geojson"
