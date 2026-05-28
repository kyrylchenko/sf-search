from main_service.config import Settings
from main_service.db.initialize_engine import build_database_url


def test_build_database_url_defaults_to_postgres_psycopg() -> None:
    settings = Settings(
        _env_file=None,
        db_user="user",
        db_password="password",
        db_host="localhost",
        db_port=5432,
        db_name="sf_search",
    )

    url = build_database_url(settings)

    assert url.drivername == "postgresql+psycopg"
    assert url.username == "user"
    assert url.password == "password"
    assert url.host == "localhost"
    assert url.port == 5432
    assert url.database == "sf_search"


def test_build_database_url_allows_driver_override() -> None:
    settings = Settings(
        _env_file=None,
        db_user="user",
        db_password="password",
        db_host="db",
        db_port=5432,
        db_name="sf_search",
        db_driver="postgresql+psycopg",
    )

    assert build_database_url(settings).drivername == "postgresql+psycopg"
