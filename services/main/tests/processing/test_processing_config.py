from main_service.config import Settings


def test_processing_defaults_are_configured_for_local_dev() -> None:
    settings = Settings(
        _env_file=None,
        db_user="user",
        db_password="password",
        db_host="localhost",
        db_port=5432,
        db_name="sf_search",
    )

    assert settings.pano_processing_consumer == "pano-processor"
    assert settings.pano_viewsets_dir == "../../docs/data/viewsets"
    assert settings.pano_view_storage_dir == ".local/panorama-view-tmp"
    assert settings.pano_processing_concurrency == 4
    assert settings.pano_view_max_render_concurrency == 4
    assert settings.pano_view_render_scale == 2
    assert settings.pano_view_output_format == "jpeg"
    assert settings.pano_view_jpeg_quality == 95
