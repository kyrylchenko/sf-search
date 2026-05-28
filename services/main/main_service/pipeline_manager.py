from collections import deque
from typing import Deque, Final, Optional, Sequence

from main_service.db.models.panorama import Panorama
from main_service.db.services.panorama_service import PanoramaService
from main_service.processing.tiling import TileSpec


EMBEDDING_QUEUE_THRESHOLD = 100


class PipelineManager:
    def __init__(
        self,
        panos_to_process: Sequence[str],
        tile_specs: Sequence[TileSpec],
        pano_service: PanoramaService,
    ) -> None:
        self._panos_to_process: Deque[str] = deque(panos_to_process)
        self._tile_specs: Sequence[TileSpec] = tile_specs
        self._pano_service = pano_service

    def start(self) -> None:
        raise NotImplementedError(
            "Pipeline orchestration is not implemented yet. "
            "Write a spec and plan before adding crawler/downloader/embedder behavior."
        )

    def _refill_queue_if_needed(self, current_queue_size: int) -> None:
        if current_queue_size >= EMBEDDING_QUEUE_THRESHOLD:
            return
        if not self._panos_to_process:
            return

        pano_id_to_process: Final[str] = self._panos_to_process.pop()
        found_pano: Final[Optional[Panorama]] = (
            self._pano_service.find_panorama_by_orig_id(pano_id_to_process)
        )
        if found_pano is not None:
            return

        self._process_pano(pano_id_to_process)

    def _process_pano(self, pano_id: str) -> None:
        found_pano: Final[Optional[Panorama]] = (
            self._pano_service.find_panorama_by_orig_id(pano_id)
        )
        if found_pano is not None:
            return

        raise NotImplementedError(
            f"Panorama processing for {pano_id!r} is not implemented yet."
        )
