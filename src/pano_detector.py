from __future__ import annotations

import os
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Dict

import numpy as np


def _require(module_name: str, install_hint: str):
    try:
        return __import__(module_name)
    except ImportError as e:
        raise RuntimeError(
            f"Missing dependency '{module_name}'. {install_hint}"
        ) from e


# Try to import optional heavy deps with clear hints
cv2 = _require(
    "cv2",
    "Install OpenCV: pip install opencv-python"
)
ultra = _require(
    "ultralytics",
    "Install YOLOv8: pip install ultralytics"
)

# Optional deps for hardware detection and progress
try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:
    # minimal fallback so script still runs without tqdm
    class _TqdmFallback:
        def __init__(self, total=None, desc=None):
            self.total = total
            self.desc = desc
        def update(self, n=1):
            pass
        def close(self):
            pass
    def tqdm(*args, **kwargs):  # type: ignore
        return _TqdmFallback(kwargs.get('total'), kwargs.get('desc'))


def _detect_device() -> str:
    # Prefer Apple Metal on M-series, then CUDA, else CPU
    if torch is not None:
        try:
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                return 'cuda:0'
        except Exception:
            pass
    return 'cpu'

DEFAULT_DEVICE = os.getenv('PANO_DEVICE', _detect_device())


@dataclass
class Det:
    x1: float
    y1: float
    x2: float
    y2: float
    cls: int
    score: float


def load_yolo(model_path: Optional[str] = None):
    # Default to YOLOv12 nano if none provided (requires ultralytics supporting v12)
    path = model_path or os.getenv("PANO_YOLO_WEIGHTS", "yolo12n.pt")
    model = ultra.YOLO(path)
    return model


def equirect_to_perspective(equi: np.ndarray, yaw_deg: float, pitch_deg: float, fov_deg: float, out_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a perspective projection from an equirectangular image.
    Returns: (patch_bgr, map_x, map_y) where map_x/map_y are float32 remap grids back to equi pixels.
    """
    h, w = equi.shape[:2]
    fov = math.radians(fov_deg)
    # output grid
    xs = np.linspace(-1.0, 1.0, out_size, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, out_size, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)  # shape (N, N)
    # pinhole: scale plane by tan(fov/2)
    s = math.tan(fov / 2.0)
    px = xv * s
    py = -yv * s  # flip y so image y-down maps to +down in view
    pz = np.ones_like(px)
    # normalize ray directions
    norm = np.sqrt(px * px + py * py + pz * pz)
    px /= norm
    py /= norm
    pz /= norm
    # apply rotations: yaw around y, pitch around x
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    cos_x, sin_x = math.cos(pitch), math.sin(pitch)
    # R = R_y(yaw) @ R_x(pitch)
    # First Rx
    rx_x = px
    rx_y = cos_x * py - sin_x * pz
    rx_z = sin_x * py + cos_x * pz
    # Then Ry
    rw_x = cos_y * rx_x + sin_y * rx_z
    rw_y = rx_y
    rw_z = -sin_y * rx_x + cos_y * rx_z
    # to lon/lat
    lon = np.arctan2(rw_x, rw_z)  # [-pi, pi]
    lat = np.arctan2(rw_y, np.sqrt(rw_x * rw_x + rw_z * rw_z))  # [-pi/2, pi/2]
    # map to equirect pixels
    map_x = (lon / (2 * math.pi) + 0.5) * w
    map_y = (0.5 - lat / math.pi) * h
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    # sample with cv2.remap (border wrap horizontally, clamp vertically)
    patch = cv2.remap(equi, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return patch, map_x, map_y


def perspective_box_to_equirect(box: Tuple[float, float, float, float], map_x: np.ndarray, map_y: np.ndarray, seam_w: int) -> List[Det]:
    """
    Map a box from perspective space back to equirectangular coordinates by sampling its border and
    returning one or two axis-aligned bboxes in equi space (split if crosses the seam).
    Returns a list of Det-like bbox tuples (x1,y1,x2,y2) without class/score.
    """
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = np.clip(x1, 0, map_x.shape[1] - 1)
    x2 = np.clip(x2, 0, map_x.shape[1] - 1)
    y1 = np.clip(y1, 0, map_x.shape[0] - 1)
    y2 = np.clip(y2, 0, map_x.shape[0] - 1)
    # sample perimeter
    samples: List[Tuple[float, float]] = []
    N = 40
    if x2 <= x1 or y2 <= y1:
        return []
    for t in np.linspace(0, 1, N):
        # top and bottom edges
        xs = int(round(x1 + t * (x2 - x1)))
        samples.append((map_x[y1, xs], map_y[y1, xs]))
        samples.append((map_x[y2, xs], map_y[y2, xs]))
    for t in np.linspace(0, 1, N):
        ys = int(round(y1 + t * (y2 - y1)))
        samples.append((map_x[ys, x1], map_y[ys, x1]))
        samples.append((map_x[ys, x2], map_y[ys, x2]))
    xs = np.array([p[0] for p in samples])
    ys = np.array([p[1] for p in samples])
    # seam handling: work in angle space [0, 1) of width
    ang = xs / seam_w  # [0,1)
    # try both no-shift and +0.5 shift to reduce wrap span; pick minimal span
    spans = []
    for shift in (0.0, 0.5):
        a = (ang + shift) % 1.0
        span = a.max() - a.min()
        spans.append((span, shift, a))
    _, best_shift, a_best = min(spans, key=lambda t: t[0])
    a_sorted = a_best
    x_adj = (a_sorted % 1.0) * seam_w
    x_min = float(np.min(x_adj))
    x_max = float(np.max(x_adj))
    y_min = float(np.min(ys))
    y_max = float(np.max(ys))
    # Clamp to image bounds
    x_min_c = max(0.0, min(x_min, seam_w - 1.0))
    y_min_c = max(0.0, y_min)
    x_max_c = max(0.0, min(x_max, seam_w - 1.0))
    y_max_c = max(0.0, y_max)
    return [(x_min_c, y_min_c, x_max_c, y_max_c)]


def nms(boxes: List[Det], iou_thresh: float) -> List[Det]:
    if not boxes:
        return []
    # sort by score desc
    boxes = sorted(boxes, key=lambda d: d.score, reverse=True)
    keep: List[Det] = []
    arr = np.array([[b.x1, b.y1, b.x2, b.y2, b.score, b.cls] for b in boxes], dtype=np.float32)
    x1, y1, x2, y2, scores, cls = [arr[:, i] for i in range(6)]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    while order.size > 0:
        i = order[0]
        # class-wise suppression
        same_cls_idx = np.where(cls[order] == cls[i])[0]
        candidates = order[same_cls_idx]
        xx1 = np.maximum(x1[i], x1[candidates])
        yy1 = np.maximum(y1[i], y1[candidates])
        xx2 = np.minimum(x2[i], x2[candidates])
        yy2 = np.minimum(y2[i], y2[candidates])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[candidates] - inter + 1e-6)
        keep.append(boxes[i])
        remain = candidates[iou < iou_thresh]
        # remove processed and suppressed
        to_remove = set(candidates.tolist()) - set(remain.tolist())
        mask = np.ones_like(order, dtype=bool)
        for idx in to_remove.union({i}):
            mask = mask & (order != idx)
        order = order[mask]
    return keep

def _iou(b1: Det, b2: Det) -> float:
    xa = max(b1.x1, b2.x1)
    ya = max(b1.y1, b2.y1)
    xb = min(b1.x2, b2.x2)
    yb = min(b1.y2, b2.y2)
    w = max(0.0, xb - xa)
    h = max(0.0, yb - ya)
    inter = w * h
    a1 = max(0.0, (b1.x2 - b1.x1)) * max(0.0, (b1.y2 - b1.y1))
    a2 = max(0.0, (b2.x2 - b2.x1)) * max(0.0, (b2.y2 - b2.y1))
    denom = a1 + a2 - inter + 1e-6
    return inter / denom if denom > 0 else 0.0


def wbf_merge(dets: List[Det], iou_thresh: float = 0.55) -> List[Det]:
    """Simple Weighted Boxes Fusion per class to merge many overlapping boxes into one."""
    if not dets:
        return []
    merged: List[Det] = []
    # group by class
    by_cls: Dict[int, List[Det]] = {}
    for d in dets:
        by_cls.setdefault(d.cls, []).append(d)
    for cls_id, group in by_cls.items():
        remaining = sorted(group, key=lambda d: d.score, reverse=True)
        used = [False] * len(remaining)
        for i, d in enumerate(remaining):
            if used[i]:
                continue
            # start cluster
            cluster = [d]
            used[i] = True
            for j in range(i + 1, len(remaining)):
                if used[j]:
                    continue
                if _iou(d, remaining[j]) >= iou_thresh:
                    cluster.append(remaining[j])
                    used[j] = True
            # weighted average by score
            if len(cluster) == 1:
                merged.append(cluster[0])
            else:
                wsum = sum(b.score for b in cluster)
                x1 = sum(b.x1 * b.score for b in cluster) / wsum
                y1 = sum(b.y1 * b.score for b in cluster) / wsum
                x2 = sum(b.x2 * b.score for b in cluster) / wsum
                y2 = sum(b.y2 * b.score for b in cluster) / wsum
                score = max(b.score for b in cluster)
                merged.append(Det(x1, y1, x2, y2, cls_id, score))
    return merged


def _build_class_colors(names: Optional[Dict[int, str]], used_classes: List[int]) -> Dict[int, Tuple[int, int, int]]:
    # fixed palette; BGR for OpenCV
    palette = [
        (0, 255, 0), (0, 200, 255), (255, 0, 0), (255, 0, 255), (0, 128, 255),
        (0, 165, 255), (0, 0, 255), (128, 0, 255), (128, 255, 0), (255, 255, 0),
        (255, 128, 0), (255, 0, 128), (128, 128, 0), (0, 255, 128), (128, 0, 128),
    ]
    colors: Dict[int, Tuple[int, int, int]] = {}
    for idx, c in enumerate(sorted(set(used_classes))):
        colors[c] = palette[idx % len(palette)]
    return colors


def _draw_legend(img: np.ndarray, class_colors: Dict[int, Tuple[int, int, int]], class_names: Optional[Dict[int, str]]):
    if not class_colors:
        return
    H, W = img.shape[:2]
    items = [(cid, class_names.get(cid, str(cid)) if class_names else str(cid)) for cid in sorted(class_colors)]
    # Compute size
    pad = 8
    swatch = 16
    line_h = max(20, swatch + 4)
    width = 0
    for _, name in items:
        width = max(width, 10 + swatch + 8 + cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0])
    height = pad * 2 + line_h * len(items)
    x0 = max(0, W - width - pad)
    y0 = pad
    # bg
    cv2.rectangle(img, (x0, y0), (x0 + width, y0 + height), (255, 255, 255), thickness=-1)
    # entries
    y = y0 + pad
    for cid, name in items:
        color = class_colors[cid]
        cv2.rectangle(img, (x0 + 10, y), (x0 + 10 + swatch, y + swatch), color, thickness=-1)
        cv2.rectangle(img, (x0 + 10, y), (x0 + 10 + swatch, y + swatch), (50, 50, 50), thickness=1)
        cv2.putText(img, name, (x0 + 10 + swatch + 8, y + swatch - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        y += line_h


def annotate_image(img: np.ndarray, dets: List[Det], class_names: Optional[Dict[int, str]] = None) -> np.ndarray:
    out = img.copy()
    used_classes = [d.cls for d in dets]
    class_colors = _build_class_colors(class_names, used_classes)
    for d in dets:
        p1 = (int(d.x1), int(d.y1))
        p2 = (int(d.x2), int(d.y2))
        color = class_colors.get(d.cls, (0, 255, 0))
        cv2.rectangle(out, p1, p2, color, 2)
        label_name = class_names.get(d.cls, str(d.cls)) if class_names else str(d.cls)
        label = f"{label_name}:{d.score:.2f}"
        cv2.putText(out, label, (p1[0], max(0, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    _draw_legend(out, class_colors, class_names)
    return out


def run_tiling_preview(
    input_path: Path,
    output_dir: Path,
    out_size: int = 960,
    fovs: Iterable[float] = (40.0, 70.0, 100.0),
    overlap: float = 0.5,
    pitch_min: float = -75.0,
    pitch_max: float = 75.0,
):
    """Draw the projected tile footprints back onto the equirect image and save a preview."""
    output_dir.mkdir(parents=True, exist_ok=True)
    equi = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if equi is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")
    H, W = equi.shape[:2]
    preview = equi.copy()
    # Palette per FOV index
    fovs_list = list(fovs)
    palette = [
        (0, 200, 255), (0, 255, 0), (255, 0, 255), (0, 128, 255), (255, 255, 0),
        (255, 128, 0), (128, 0, 255), (0, 255, 128), (255, 0, 0), (128, 128, 0),
    ]
    # progress
    total_tiles = 0
    fov_specs: List[Tuple[float, np.ndarray, np.ndarray]] = []
    for fov in fovs_list:
        step = fov * (1.0 - overlap)
        if step <= 0:
            step = fov
        yaw_vals = np.arange(0.0, 360.0, max(1.0, step))
        pitch_vals = np.arange(pitch_min, pitch_max + 1e-6, max(1.0, step))
        total_tiles += len(yaw_vals) * len(pitch_vals)
        fov_specs.append((fov, yaw_vals, pitch_vals))
    pbar = tqdm(total=total_tiles, desc=f"tiles:{input_path.name}")

    for fov_idx, (fov, yaw_vals, pitch_vals) in enumerate(fov_specs):
        color = palette[fov_idx % len(palette)]
        for pitch in pitch_vals:
            for yaw in yaw_vals:
                patch, map_x, map_y = equirect_to_perspective(equi, yaw, pitch, fov, out_size)
                # map the full patch extent back
                boxes = perspective_box_to_equirect((1, 1, out_size - 2, out_size - 2), map_x, map_y, W)
                for (x1, y1, x2, y2) in boxes:
                    p1 = (int(x1), int(y1))
                    p2 = (int(x2), int(y2))
                    cv2.rectangle(preview, p1, p2, color, 2)
                pbar.update(1)
    pbar.close()

    out_path = output_dir / (input_path.stem + "-tiles" + input_path.suffix)
    cv2.imwrite(str(out_path), preview)
    print(f"Tiling preview saved: {out_path} | size={W}x{H} | tiles={total_tiles}")


def run_detection(
    input_dir: Path,
    output_dir: Path,
    model_weights: Optional[str] = None,
    out_size: int = 960,
    fovs: Iterable[float] = (40.0, 70.0, 100.0),
    overlap: float = 0.5,
    pitch_min: float = -75.0,
    pitch_max: float = 75.0,
    iou_nms: float = 0.5,
    score_thresh: float = 0.25,
):
    model = load_yolo(model_weights)
    class_names = model.model.names if hasattr(model, 'model') else None

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect common image types (case-insensitive)
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    images = []
    for pat in patterns:
        images.extend(input_dir.glob(pat))
    images = sorted(images)
    for img_path in images:
        equi = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if equi is None:
            print(f"Skip unreadable image: {img_path}")
            continue
        H, W = equi.shape[:2]
        # progress setup: estimate total tiles across all FOVs
        total_tiles = 0
        fov_specs: List[Tuple[float, np.ndarray, np.ndarray]] = []
        for fov in fovs:
            step = fov * (1.0 - overlap)
            if step <= 0:
                step = fov
            yaw_vals = np.arange(0.0, 360.0, max(1.0, step))
            pitch_vals = np.arange(pitch_min, pitch_max + 1e-6, max(1.0, step))
            total_tiles += len(yaw_vals) * len(pitch_vals)
            fov_specs.append((fov, yaw_vals, pitch_vals))
        print(f"Processing {img_path.name} | size={W}x{H} | tiles≈{total_tiles} | device={DEFAULT_DEVICE}")
        pbar = tqdm(total=total_tiles, desc=img_path.name)
        all_dets: List[Det] = []
        for fov, yaw_vals, pitch_vals in fov_specs:
            for pitch in pitch_vals:
                for yaw in yaw_vals:
                    patch, map_x, map_y = equirect_to_perspective(equi, yaw, pitch, fov, out_size)
                    # run yolo (force device for Apple MPS/GPU usage when available)
                    res_list = model.predict(source=patch[:, :, ::-1], imgsz=out_size, device=DEFAULT_DEVICE, verbose=False)
                    for res in res_list:
                        if not hasattr(res, 'boxes'):
                            continue
                        boxes = res.boxes
                        if boxes is None:
                            continue
                        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, 'xyxy') else None
                        conf = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else None
                        cls = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, 'cls') else None
                        if xyxy is None or conf is None or cls is None:
                            continue
                        for (x1, y1, x2, y2), s, c in zip(xyxy, conf, cls):
                            if s < score_thresh:
                                continue
                            # map back
                            mapped_boxes = perspective_box_to_equirect((x1, y1, x2, y2), map_x, map_y, W)
                            for xb in mapped_boxes:
                                mx1, my1, mx2, my2 = xb
                                all_dets.append(Det(mx1, my1, mx2, my2, int(c), float(s)))
                    pbar.update(1)

        # Merge: WBF to combine duplicates, then NMS for final pruning (per image)
        merged_stage1 = wbf_merge(all_dets, iou_thresh=max(0.3, min(0.9, iou_nms)))
        merged = nms(merged_stage1, iou_thresh=iou_nms)
        # Draw and save
        annotated = annotate_image(equi, merged, class_names)
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), annotated)
        print(f"Saved: {out_path} ({len(merged)} dets)")
        pbar.close()


def main():
    # Resolve project-rooted paths
    # Prefer workspace panos directory next to this script (../panos)
    base = Path(__file__).resolve().parents[1]
    in_dir = Path(os.getenv("PANO_DIR", base / "panos"))
    out_dir = Path(os.getenv("PANO_OUT", base / "panos-results"))
    weights = os.getenv("PANO_YOLO_WEIGHTS", None)

    print(f"Input: {in_dir}")
    print(f"Output: {out_dir}")
    if weights:
        print(f"Weights: {weights}")

    mode = os.getenv("PANO_MODE", "detect").lower()
    out_size = int(os.getenv("PANO_OUT_SIZE", "960"))
    fovs = tuple(float(x) for x in os.getenv("PANO_FOVS", "40,70,100").split(","))
    overlap = float(os.getenv("PANO_OVERLAP", "0.5"))
    pitch_min = float(os.getenv("PANO_PITCH_MIN", "-75"))
    pitch_max = float(os.getenv("PANO_PITCH_MAX", "75"))

    if mode == "tiles":
        single = os.getenv("PANO_SINGLE")
        if single:
            input_path = Path(single)
        else:
            # pick first image from input dir
            patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
            images = []
            for pat in patterns:
                images.extend(in_dir.glob(pat))
            images = sorted(images)
            if not images:
                raise FileNotFoundError(f"No images found in {in_dir}")
            input_path = images[0]
        print(f"Tiling preview mode | input={input_path}")
        run_tiling_preview(
            input_path=input_path,
            output_dir=out_dir,
            out_size=out_size,
            fovs=fovs,
            overlap=overlap,
            pitch_min=pitch_min,
            pitch_max=pitch_max,
        )
    else:
        run_detection(
            input_dir=in_dir,
            output_dir=out_dir,
            model_weights=weights,
            out_size=out_size,
            fovs=fovs,
            overlap=overlap,
            pitch_min=pitch_min,
            pitch_max=pitch_max,
            iou_nms=float(os.getenv("PANO_IOU_NMS", "0.5")),
            score_thresh=float(os.getenv("PANO_SCORE_THRESH", "0.25")),
        )


if __name__ == "__main__":
    main()
