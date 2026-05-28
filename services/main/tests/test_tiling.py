import numpy as np

from main_service.processing.tiling import TileSpec, create_tiles_for_pano


def test_create_tiles_for_pano_returns_one_tile_per_spec() -> None:
    panorama = np.zeros((128, 256, 3), dtype=np.uint8)
    specs = [
        TileSpec(yaw=0, pitch=0, roll=0, fov=70, width=64, height=32),
        TileSpec(yaw=90, pitch=10, roll=0, fov=60, width=48, height=48),
    ]

    tiles = create_tiles_for_pano(panorama, specs)

    assert [spec for spec, _ in tiles] == specs
    assert tiles[0][1].shape == (32, 64, 3)
    assert tiles[1][1].shape == (48, 48, 3)
