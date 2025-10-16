from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict
from PIL import Image
from tqdm import tqdm

from pano_utils import read_gpano_xmp, uncrop_to_full_equirect, tile_grid, yaw_pitch_for_bbox

# -------- settings --------
PANOS_DIR = Path("panos")                    # where your .jpg/.jpeg panoramas live
MANIFEST_PATH = Path("data/tiles_manifest.json")
TILE_SIZE = 1024
OVERLAP = 0
SAVE_PREVIEWS = False
PREVIEW_DIR = Path("data/tiles_preview")     # only used if SAVE_PREVIEWS=True

def process_all() -> None:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    pano_files = sorted(list(PANOS_DIR.glob("**/*.jp*g")))
    if not pano_files:
        print(f"No JPEGs found under {PANOS_DIR}.")
        return

    records: List[Dict] = []

    for p in tqdm(pano_files, desc="Tiling panoramas"):
        pano_id = p.stem
        with Image.open(p) as im:
            im = im.convert("RGB")
        gpano = read_gpano_xmp(p)
        full = uncrop_to_full_equirect(im, gpano)
        W, H = full.size
        if W < TILE_SIZE or H < TILE_SIZE:
            print(f"Skipping {p} (smaller than tile size).")
            continue

        boxes = tile_grid(W, H, TILE_SIZE, OVERLAP)

        for idx, bbox in enumerate(boxes):
            yaw, pitch = yaw_pitch_for_bbox(bbox, W, H)
            tile_id = f"{pano_id}_{idx}"
            rec = {
                "tile_id": tile_id,
                "pano_path": str(p),
                "pano_id": pano_id,
                "bbox": bbox,               # (x0,y0,x1,y1) on FULL canvas
                "yaw_deg": yaw,
                "pitch_deg": pitch,
                "full_w": W,
                "full_h": H,
                "tile_size": TILE_SIZE,
            }
            records.append(rec)

            if SAVE_PREVIEWS:
                tile = full.crop(bbox)
                PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
                tile.save(PREVIEW_DIR / f"{tile_id}.jpg", quality=92)

    with open(MANIFEST_PATH, "w") as f:
        json.dump(records, f)
    print(f"Wrote {len(records)} tiles to {MANIFEST_PATH}")

if __name__ == "__main__":
    process_all()
