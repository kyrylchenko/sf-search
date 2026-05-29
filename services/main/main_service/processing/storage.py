from pathlib import Path

from main_service.downloader.storage import safe_storage_segment
from main_service.ingestion.types import PanoramaId


def panorama_view_image_path(
    storage_dir: Path,
    *,
    pano_id: PanoramaId,
    viewset_name: str,
    view_id: str,
    view_spec_hash: str,
    render_scale: int,
    output_format: str,
) -> Path:
    extension = "jpg" if output_format.lower() in {"jpeg", "jpg"} else output_format.lower()
    return (
        storage_dir
        / safe_storage_segment(pano_id.value, fallback="pano")
        / safe_storage_segment(viewset_name, fallback="viewset")
        / (
            f"{safe_storage_segment(view_id, fallback='view')}-"
            f"{view_spec_hash[:12]}-s{render_scale}.{extension}"
        )
    )


def temp_view_image_path(final_path: Path) -> Path:
    return final_path.with_name(f"{final_path.stem}.tmp{final_path.suffix}")
