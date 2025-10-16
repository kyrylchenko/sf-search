
from __future__ import annotations
import math
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import numpy as np
from PIL import Image
from defusedxml import ElementTree as ET

# -------- GPano / XMP helpers --------

def read_gpano_xmp(jpeg_path: Path) -> Optional[Dict[str, str]]:
    """
    Read GPano XMP tags from a JPEG (if present). Returns a dict of strings or None.
    We use Pillow's getxmp(); works on common Photo Sphere / Street View files.
    """
    try:
        with Image.open(jpeg_path) as im:
            xmp = im.getxmp()  # {'xmpmeta': '<x:xmpmeta ...>...</x:xmpmeta>'}
        if not xmp or "xmpmeta" not in xmp:
            return None
        root = ET.fromstring(xmp["xmpmeta"])
        ns = {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "GPano": "http://ns.google.com/photos/1.0/panorama/",
        }
        desc = root.find(".//rdf:Description", ns)
        if desc is None:
            return None
        keys = [
            "ProjectionType",
            "FullPanoWidthPixels", "FullPanoHeightPixels",
            "CroppedAreaImageWidthPixels", "CroppedAreaImageHeightPixels",
            "CroppedAreaLeftPixels", "CroppedAreaTopPixels",
            "PoseHeadingDegrees", "PosePitchDegrees", "PoseRollDegrees",
        ]
        out = {}
        for k in keys:
            v = desc.get(f"{{{ns['GPano']}}}{k}")
            if v is not None:
                out[k] = v
        return out or None
    except Exception:
        return None


def uncrop_to_full_equirect(img: Image.Image, gpano: Optional[Dict[str, str]]) -> Image.Image:
    """
    If XMP indicates the JPEG is a cropped region of a full 360x180 equirectangular,
    paste it into the full 2:1 canvas. Otherwise, return the image as-is.
    """
    if not gpano:
        return img

    proj = gpano.get("ProjectionType", "equirectangular")
    if proj != "equirectangular":
        # Unknown projection: return as-is.
        return img

    try:
        full_w = int(gpano["FullPanoWidthPixels"])
        full_h = int(gpano["FullPanoHeightPixels"])
        crop_w = int(gpano["CroppedAreaImageWidthPixels"])
        crop_h = int(gpano["CroppedAreaImageHeightPixels"])
        left   = int(gpano["CroppedAreaLeftPixels"])
        top    = int(gpano["CroppedAreaTopPixels"])
    except KeyError:
        return img  # incomplete XMP; best effort

    canvas = Image.new("RGB", (full_w, full_h), (0, 0, 0))
    if img.size != (crop_w, crop_h):
        img = img.resize((crop_w, crop_h), Image.BICUBIC)
    canvas.paste(img, (left, top))
    return canvas


# -------- tiling helpers --------

def tile_grid(width: int, height: int, tile: int, overlap: int = 0) -> List[Tuple[int,int,int,int]]:
    """
    Produce a set of covering tiles (x0,y0,x1,y1) across the full canvas without wrap.
    Ensures the rightmost/bottommost edge is covered even if (size - tile) % step != 0.
    """
    assert tile > 0 and tile <= min(width, height)
    step = max(1, tile - overlap)

    xs = list(range(0, width - tile + 1, step))
    ys = list(range(0, height - tile + 1, step))
    if xs[-1] != width - tile:
        xs.append(width - tile)
    if ys[-1] != height - tile:
        ys.append(height - tile)

    boxes = []
    for y0 in ys:
        for x0 in xs:
            boxes.append((x0, y0, x0 + tile, y0 + tile))
    return boxes


def yaw_pitch_for_bbox(bbox: Tuple[int,int,int,int], full_w: int, full_h: int) -> Tuple[float, float]:
    """
    Approximate yaw/pitch at the center of a tile on a 2:1 equirectangular canvas.
    yaw: -180..+180 (positive to the right)
    pitch: +90..-90 (up to down)
    """
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    yaw = (cx / full_w) * 360.0 - 180.0
    pitch = 90.0 - (cy / full_h) * 180.0
    return yaw, pitch
