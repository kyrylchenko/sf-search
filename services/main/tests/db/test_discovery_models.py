from sqlalchemy import create_engine, inspect

from main_service.db.models.base import Base
from main_service.db.models.map_tile import MapTile
from main_service.db.models.map_tile_panorama import MapTilePanorama
from main_service.db.models.panorama import Panorama


def test_discovery_tables_exist_in_metadata() -> None:
    assert "map_tile_table" in Base.metadata.tables
    assert "map_tile_panorama_table" in Base.metadata.tables


def test_map_tile_has_unique_tile_key_and_status_columns() -> None:
    columns = MapTile.__table__.c
    constraint_columns = {
        tuple(column.name for column in constraint.columns)
        for constraint in MapTile.__table__.constraints
        if constraint.name == "uq_map_tile_xyz"
    }

    assert ("x", "y", "z") in constraint_columns
    assert "discovery_status" in columns
    assert "attempt_count" in columns
    assert "last_error" in columns


def test_map_tile_panorama_has_unique_link_key() -> None:
    constraint_columns = {
        tuple(column.name for column in constraint.columns)
        for constraint in MapTilePanorama.__table__.constraints
        if constraint.name == "uq_map_tile_panorama"
    }

    assert ("map_tile_id", "panorama_id") in constraint_columns


def test_panorama_has_discovery_and_later_stage_status_columns() -> None:
    columns = Panorama.__table__.c

    assert "metadata_status" in columns
    assert "download_status" in columns
    assert "discovered_at_tile_count" in columns
    assert "attempt_count" in columns
    assert "last_error" in columns
    assert "image_path" in columns
    assert "metadata_json" in columns
    assert "downloaded_at" in columns


def test_metadata_can_create_sqlite_schema() -> None:
    engine = create_engine("sqlite:///:memory:")

    Base.metadata.create_all(engine)

    assert inspect(engine).has_table("map_tile_table") is True
    assert inspect(engine).has_table("map_tile_panorama_table") is True
