import hashlib
import os
import re
from pathlib import Path

from main_service.ingestion.types import PanoramaId

_SAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_-]+")


def pano_image_path(storage_dir: Path, pano_id: PanoramaId) -> Path:
    return storage_dir / f"{_safe_pano_filename(pano_id)}.jpg"


def temp_pano_image_path(final_path: Path) -> Path:
    return final_path.with_name(f"{final_path.stem}.tmp{final_path.suffix}")


def finalize_temp_file(temp_path: Path, final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temp_path, final_path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_pano_filename(pano_id: PanoramaId) -> str:
    sanitized = _SAFE_FILENAME_CHARS.sub("_", pano_id.value).strip("._-")
    return sanitized or "pano"
