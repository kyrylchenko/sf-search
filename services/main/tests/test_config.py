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


def test_settings_include_discovery_backpressure_default() -> None:
    settings = Settings(
        _env_file=None,
        db_user="user",
        db_password="password",
        db_host="localhost",
        db_port=3306,
        db_name="sf_search",
    )

    assert settings.max_downloader_queue_depth == 1000


def test_settings_include_processing_render_concurrency_cap() -> None:
    settings = Settings(
        _env_file=None,
        db_user="user",
        db_password="password",
        db_host="localhost",
        db_port=3306,
        db_name="sf_search",
    )

    assert settings.pano_processing_concurrency == 4
    assert settings.pano_view_max_render_concurrency == 4


def test_settings_include_observability_defaults() -> None:
    settings = Settings(
        _env_file=None,
        db_user="user",
        db_password="password",
        db_host="localhost",
        db_port=3306,
        db_name="sf_search",
    )

    assert settings.observability_enabled is False
    assert settings.otel_exporter_otlp_endpoint == "http://localhost:4317"
    assert settings.otel_exporter_otlp_insecure is True
    assert settings.otel_metric_export_interval_millis == 10000
    assert settings.monitoring_interval_seconds == 15.0
