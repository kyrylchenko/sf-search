from main_service.config import Settings


def test_downloader_defaults_are_configured_for_local_dev() -> None:
    settings = Settings(
        _env_file=None,
        db_user="user",
        db_password="password",
        db_host="localhost",
        db_port=5432,
        db_name="sf_search",
    )

    assert settings.pano_download_concurrency == 5
    assert settings.pano_download_storage_dir == ".local/panoramas"
    assert settings.nats_url == "nats://localhost:4222"
    assert settings.pano_processing_stream == "PANO_PROCESSING"
    assert settings.pano_processing_subject == "pano.processing.requested"
    assert settings.max_processing_queue_depth == 50
