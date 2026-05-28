from typing import Deque, MutableSequence, Optional, Sequence, Tuple, Final
import copy
from collections import deque
from main_service.db.models.panorama import Panorama
from main_service.db.services import panorama_service
from main_service.db.services.panorama_service import PanoramaService
from main_service.processing.tiling import TileSpec

EMBEDDING_QUEUE_THRESHOLD = 100

class PipelineManager:
    def __init__(
        self, panos_to_process: Sequence[str],
        tile_specs: Sequence[TileSpec],
        pano_service: PanoramaService
    ) -> None:
        self._panos_to_process: Deque = deque(panos_to_process)
        self._tile_specs: Sequence[TileSpec] = tile_specs
        self._pano_service = pano_service
        pass

    def start(self):
        pass

    def _refill_queue_if_needed(self):
        size = 5
        if size >= EMBEDDING_QUEUE_THRESHOLD:
            return

        pano_id_to_process: Final[str] = self._panos_to_process.pop()
        found_pano: Final[Optional[Panorama]] = self._pano_service.find_panorama_by_orig_id(pano_id_to_process)
        if found_pano:
            return


        # check inner list for tiles, if not empty take treshold * count of spliting tiles of tiles
        # if empty refil it from tiles
        # start downloading ids
        # download id -> split it into tiles
        # put tiles on queue check if queue is still below, if so, continue
        pass

    def _process_pano(self, pano_id: str):
        found_pano: Final[Optional[Panorama]] = self._pano_service.find_panorama_by_orig_id(pano_id)
        if found_pano is not None:
            return
        
        pano_metadata = 


