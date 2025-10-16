from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # disable PIL's decompression bomb check for large equirectangular panoramas
from tqdm import tqdm
import hnswlib

from pano_utils import read_gpano_xmp, uncrop_to_full_equirect

# -------- files & settings --------
MANIFEST_PATH = Path("data/tiles_manifest.json")
EMB_NPY = Path("data/image_embeds.npy")
META_PARQUET = Path("data/tiles_metadata.parquet")
INDEX_PATH = Path("data/hnsw_cosine.index")

MODEL_NAME = "ViT-L-14"          # switch to other OpenCLIP models if you like
PRETRAINED = "laion2b_s32b_b82k"  # strong, widely used
BATCH = 2                        # good start for 16GB M3 MBP
SEED = 42

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

def load_model():
    torch.manual_seed(SEED)
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    tok = open_clip.get_tokenizer(MODEL_NAME)
    model = model.to(DEVICE)
    model.eval()
    return model, preprocess, tok

def encode_batch(model, batch: torch.Tensor) -> torch.Tensor:
    # Keep float32 path on MPS for stability; CUDA can use autocast(fp16) but we keep code unified.
    feats = model.encode_image(batch)
    feats = feats.float()
    feats /= feats.norm(dim=-1, keepdim=True)
    return feats

def iter_tiles_grouped(manifest: List[Dict[str, Any]]):
    """
    Yield (pano_path, full_image, group_records_for_this_pano).
    Reconstruct the full canvas ONCE per panorama for efficiency.
    """
    current_pano = None
    current_img = None
    group: List[Dict[str, Any]] = []

    for rec in manifest:
        pano_path = rec["pano_path"]
        if pano_path != current_pano:
            # flush previous
            if current_pano is not None:
                yield current_pano, current_img, group
                group = []
            # load new pano + reconstruct full
            p = Path(pano_path)
            with Image.open(p) as im:
                im = im.convert("RGB")
            gpano = read_gpano_xmp(p)
            full = uncrop_to_full_equirect(im, gpano)
            current_pano = pano_path
            current_img = full
        group.append(rec)

    if current_pano is not None and group:
        yield current_pano, current_img, group

def embed_all():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"{MANIFEST_PATH} not found. Run tile_panos.py first.")

    manifest = json.loads(MANIFEST_PATH.read_text())
    model, preprocess, _ = load_model()

    feats = []
    rows = []

    imgs = []
    metas = []
    for pano_path, full_img, group in tqdm(iter_tiles_grouped(manifest), desc="Embedding tiles"):
        for rec in group:
            x0, y0, x1, y1 = rec["bbox"]
            tile = full_img.crop((x0, y0, x1, y1))
            imgs.append(preprocess(tile))
            metas.append(rec)

            if len(imgs) == BATCH:
                batch = torch.stack(imgs).to(DEVICE)
                # run inference without tracking gradients and detach before converting to numpy
                with torch.no_grad():
                    emb = encode_batch(model, batch)
                feats.append(emb.cpu().detach().numpy())
                rows.extend(metas)
                imgs, metas = [], []

    if imgs:
        batch = torch.stack(imgs).to(DEVICE)
        with torch.no_grad():
            emb = encode_batch(model, batch)
        feats.append(emb.cpu().detach().numpy())
        rows.extend(metas)

    X = np.concatenate(feats, axis=0)             # float32
    X = X.astype("float16")                       # compact on disk
    np.save(EMB_NPY, X)
    pd.DataFrame(rows).to_parquet(META_PARQUET, index=False)
    print("Saved embeddings:", X.shape, "→", EMB_NPY)

def build_hnsw():
    X = np.load(EMB_NPY, mmap_mode="r").astype("float32")  # hnswlib expects float32
    dim = X.shape[1]
    index = hnswlib.Index(space='cosine', dim=dim)
    index.init_index(max_elements=X.shape[0], ef_construction=200, M=32)
    # optional: normalize (cosine space doesn't require it, but keeps consistency with IP)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms

    index.add_items(Xn, np.arange(Xn.shape[0]))
    index.set_ef(200)   # search depth
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    index.save_index(str(INDEX_PATH))
    print("Saved HNSW index →", INDEX_PATH)

if __name__ == "__main__":
    embed_all()
    build_hnsw()

    # --- FAISS alternative (if you prefer):
    # import faiss
    # X = np.load(EMB_NPY, mmap_mode="r").astype("float32")
    # d = X.shape[1]
    # faiss.normalize_L2(X)
    # index = faiss.IndexHNSWFlat(d, 32)
    # index.hnsw.efConstruction = 200
    # index.add(X)
    # faiss.write_index(index, "data/faiss_hnsw.index")
