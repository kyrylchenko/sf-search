from pathlib import Path

from main_service.downloader.storage import (
    finalize_temp_file,
    pano_image_path,
    sha256_file,
    temp_pano_image_path,
)
from main_service.ingestion.types import PanoramaId


def test_pano_image_path_sanitizes_pano_id() -> None:
    path = pano_image_path(
        Path(".local/panoramas"),
        PanoramaId("../pano/a:b"),
    )

    assert path == Path(".local/panoramas/pano_a_b.jpg")


def test_temp_pano_image_path_uses_same_directory() -> None:
    final_path = Path(".local/panoramas/pano-a.jpg")

    assert temp_pano_image_path(final_path) == Path(".local/panoramas/pano-a.tmp.jpg")


def test_finalize_temp_file_creates_parent_and_replaces_final(tmp_path: Path) -> None:
    final_path = tmp_path / "nested" / "pano-a.jpg"
    temp_path = temp_pano_image_path(final_path)
    temp_path.parent.mkdir(parents=True)
    temp_path.write_bytes(b"new-image")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_bytes(b"old-image")

    finalize_temp_file(temp_path, final_path)

    assert final_path.read_bytes() == b"new-image"
    assert not temp_path.exists()


def test_sha256_file_hashes_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "pano-a.jpg"
    path.write_bytes(b"image-bytes")

    assert (
        sha256_file(path)
        == "2c8648d103e3dd7ad87660da0f126a1443b6d21ac1bd3ec000c5e24e2373a90c"
    )
