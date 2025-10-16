from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image, ImageDraw
import hnswlib

INDEX_PATH = Path("data/hnsw_cosine.index")
META_PARQUET = Path("data/tiles_metadata.parquet")

MODEL_NAME = "ViT-L-14"
PRETRAINED = "laion2b_s32b_b82k"

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    tok = open_clip.get_tokenizer(MODEL_NAME)
    model = model.to(DEVICE)
    model.eval()
    return model, preprocess, tok

@torch.no_grad()
def text_embed(model, tok, text: str) -> np.ndarray:
    tokens = tok([text]).to(DEVICE)
    feat = model.encode_text(tokens)
    feat = feat.float()
    feat /= feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype("float32")

def draw_bbox(pano_path: str, bbox, out_path: Path):
    with Image.open(pano_path) as im:
        im = im.convert("RGB")
        draw = ImageDraw.Draw(im)
        draw.rectangle(bbox, outline="red", width=5)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(out_path)

def main(query_text: str, k: int = 30):
    if not INDEX_PATH.exists() or not META_PARQUET.exists():
        raise FileNotFoundError("Index or metadata missing. Run embed_index.py first.")

    model, _, tok = load_model()
    q = text_embed(model, tok, query_text)

    # load hnsw index
    meta = pd.read_parquet(META_PARQUET)
    dim = q.shape[1]
    p = hnswlib.Index(space='cosine', dim=dim)
    p.load_index(str(INDEX_PATH))
    p.set_ef(200)
    labels, dists = p.knn_query(q, k=k)

    hits = meta.iloc[labels[0]].copy()
    hits["cosine_distance"] = dists[0]
    print(hits[["tile_id","pano_id","yaw_deg","pitch_deg","cosine_distance"]])
    out_dir = Path("data/query_previews")
    for _, r in hits.head(5).iterrows():
        out_path = out_dir / f"{r['tile_id']}.jpg"
        draw_bbox(r["pano_path"], r["bbox"], out_path)
    print("Saved top-5 previews →", out_dir)

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "yellow taxi"
    main(q, k=50)
