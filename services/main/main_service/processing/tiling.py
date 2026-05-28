from typing import MutableSequence, NamedTuple, Sequence, Tuple
from mercantile import Tile
from numpy.typing import NDArray
import py360convert
from py360convert.utils import DType


class TileSpec(NamedTuple):
    yaw: float
    pitch: float
    roll: float
    fov: float
    width: int
    height: int


def create_tiles_for_pano(panorama: NDArray[DType], tile_specs: Sequence[TileSpec]):
    created_tiles: MutableSequence[Tuple[TileSpec, NDArray]] = []

    for spec in tile_specs:
        tile = py360convert.e2p(
            panorama,
            spec.fov,
            spec.yaw,
            spec.pitch,
            (spec.height, spec.width),
            spec.roll,
        )
        created_tiles.append((spec, tile))

    return created_tiles


tile_specs_default = [
    TileSpec(yaw=0, pitch=0, roll=0, fov=70, width=1000, height=1000),
    TileSpec(yaw=180, pitch=10, roll=0, fov=70, width=1000, height=1000),
    TileSpec(yaw=90, pitch=10, roll=0, fov=70, width=1000, height=1000),
    TileSpec(yaw=-90, pitch=10, roll=0, fov=70, width=1000, height=1000),
]
