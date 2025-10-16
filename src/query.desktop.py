from __future__ import annotations
from pathlib import Path
import threading

import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image, ImageDraw, ImageTk
import hnswlib
import tkinter as tk
from tkinter import ttk, messagebox

# Files
INDEX_PATH = Path("data/hnsw_cosine.index")
META_PARQUET = Path("data/tiles_metadata.parquet")

# Model
MODEL_NAME = "ViT-L-14"
PRETRAINED = "laion2b_s32b_b82k"

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

# Large equirect panos can exceed PIL's default safety limit
Image.MAX_IMAGE_PIXELS = None


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

def _decode_thumbnail(path: str, max_wh: tuple[int, int]) -> tuple[Image.Image, tuple[int, int]]:
    """Open a large pano as a downscaled thumbnail quickly. Returns (thumb, (orig_w, orig_h))."""
    with Image.open(path) as im:
        orig_w, orig_h = im.size
        # Try to hint JPEG decoder to load at lower resolution
        try:
            im.draft("RGB", max_wh)
        except Exception:
            pass
        im = im.convert("RGB")
        # Final downscale to fit UI size
        thumb = im.copy()
        thumb.thumbnail(max_wh, Image.LANCZOS)
    return thumb, (orig_w, orig_h)


def draw_bbox_on_thumbnail(thumb: Image.Image, bbox, orig_size: tuple[int, int]) -> Image.Image:
    """Draw bbox (in orig pixel coords) onto a copy of the given thumbnail image, scaled to match."""
    tw, th = thumb.size
    ow, oh = orig_size
    # scale bbox to thumbnail space
    l, t, r, b = [float(x) for x in bbox]
    sx = tw / max(1, ow)
    sy = th / max(1, oh)
    l = int(max(0, min(tw - 1, round(l * sx))))
    r = int(max(0, min(tw - 1, round(r * sx))))
    t = int(max(0, min(th - 1, round(t * sy))))
    b = int(max(0, min(th - 1, round(b * sy))))

    out = thumb.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle([l, t, r, b], outline="red", width=3)
    return out


class SearchUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Tile Search")

        # runtime state
        self.model = None
        self.preprocess = None
        self.tok = None
        self.meta = None
        self.index = None
        self.dim = None
        self._thumb_cache = {}
        self.image_refs = []
        # incremental rendering state
        self.current_hits = None
        self.rendered = 0
        self.batch_size = 30
        self._load_more_btn = None

        # layout
        main = ttk.Frame(root, padding=12)
        main.grid(row=0, column=0, sticky="nsew")
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)

        ttk.Label(main, text="Query:").grid(row=0, column=0, sticky="w")
        self.q_var = tk.StringVar(value="yellow taxi")
        self.q_entry = ttk.Entry(main, textvariable=self.q_var, width=50)
        self.q_entry.grid(row=0, column=1, sticky="ew", padx=(6, 6))
        main.columnconfigure(1, weight=1)

        ttk.Label(main, text="Top K:").grid(row=0, column=2, sticky="e")
        self.k_var = tk.IntVar(value=30)
        self.k_spin = ttk.Spinbox(main, from_=1, to=200, textvariable=self.k_var, width=6)
        self.k_spin.grid(row=0, column=3, sticky="w")

        self.btn = ttk.Button(main, text="Find", command=self.on_find)
        self.btn.grid(row=0, column=4, padx=(8, 0))

        self.status_var = tk.StringVar(value="Loading model/index…")
        ttk.Label(main, textvariable=self.status_var).grid(row=1, column=0, columnspan=5, sticky="w", pady=(8, 0))

        # Scrollable results area
        results = ttk.Frame(main)
        results.grid(row=2, column=0, columnspan=5, sticky="nsew", pady=(8, 0))
        main.rowconfigure(2, weight=1)
        self._canvas = tk.Canvas(results, highlightthickness=0)
        vsb = ttk.Scrollbar(results, orient="vertical", command=self._canvas.yview)
        self.inner = ttk.Frame(self._canvas)
        self.inner.bind("<Configure>", lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")))
        self._canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self._canvas.configure(yscrollcommand=vsb.set)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        results.rowconfigure(0, weight=1)
        results.columnconfigure(0, weight=1)

        # load assets
        try:
            self.load_assets()
            self.status_var.set("Ready.")
        except Exception as e:
            messagebox.showerror("Startup Error", str(e))
            self.status_var.set("Failed to load assets.")

        self.q_entry.bind("<Return>", lambda e: self.on_find())

    def load_assets(self):
        if not INDEX_PATH.exists() or not META_PARQUET.exists():
            raise FileNotFoundError("Index or metadata missing. Run embed.py first.")

        self.model, self.preprocess, self.tok = load_model()
        self.meta = pd.read_parquet(META_PARQUET)
        # infer dimension via dummy embed
        q = text_embed(self.model, self.tok, "test")
        self.dim = q.shape[1]
        self.index = hnswlib.Index(space='cosine', dim=self.dim)
        self.index.load_index(str(INDEX_PATH))
        self.index.set_ef(200)

    def on_find(self):
        if self.model is None or self.index is None:
            return
        query = self.q_var.get().strip()
        if not query:
            return
        k = max(1, int(self.k_var.get()))
        self.btn.config(state=tk.DISABLED)
        self.status_var.set("Searching…")
        threading.Thread(target=self._search_worker, args=(query, k), daemon=True).start()

    def _search_worker(self, query: str, k: int):
        try:
            q = text_embed(self.model, self.tok, query)
            labels, dists = self.index.knn_query(q, k=k)
            labels = labels[0]
            dists = dists[0]
            hits = self.meta.iloc[labels].copy()
            # Distances from hnswlib in 'cosine' space are 1 - cosine_similarity (for unit-norm vectors)
            hits["cosine_distance"] = dists.astype(float)
            # Cosine similarity in [-1, 1]
            cos_sim = 1.0 - hits["cosine_distance"]
            hits["cos_sim"] = cos_sim
            # Map to [0,1] for easier reading (optional display only)
            hits["conf_01"] = (cos_sim + 1.0) / 2.0
            # Relative confidence over the returned top-k using a softmax on cosine similarity
            # Temperature controls sharpness; larger -> peakier distribution
            temp = 30.0
            sims = cos_sim.to_numpy()
            scores = sims * temp
            scores = scores - scores.max()  # numerical stability
            exps = np.exp(scores)
            rel = exps / (exps.sum() + 1e-12)
            hits["rel_conf"] = rel
            # Sort by cosine similarity (equivalently lowest distance)
            hits = hits.sort_values("cos_sim", ascending=False)
        except Exception as e:
            self.root.after(0, lambda: self._search_error(e))
            return
        self.root.after(0, lambda: self._render_results(hits))

    def _search_error(self, e: Exception):
        messagebox.showerror("Search Error", str(e))
        self.status_var.set("Ready.")
        self.btn.config(state=tk.NORMAL)

    def _render_results(self, hits: pd.DataFrame):
        self._clear_results()
        self.current_hits = hits.reset_index(drop=True)
        self.rendered = 0
        self._render_next_batch()
        self.btn.config(state=tk.NORMAL)

    def _clear_results(self):
        if self._load_more_btn is not None:
            try:
                self._load_more_btn.destroy()
            except Exception:
                pass
            self._load_more_btn = None
        for w in self.inner.winfo_children():
            w.destroy()
        self.image_refs.clear()

    def _render_next_batch(self):
        if self.current_hits is None:
            return
        total = len(self.current_hits)
        start = self.rendered
        end = min(start + self.batch_size, total)
        for i in range(start, end):
            r = self.current_hits.iloc[i]
            try:
                pano_path: str = r["pano_path"]
                if pano_path not in self._thumb_cache:
                    thumb, orig_size = _decode_thumbnail(pano_path, (640, 360))
                    self._thumb_cache[pano_path] = (thumb, orig_size)
                else:
                    thumb, orig_size = self._thumb_cache[pano_path]
                img_with_box = draw_bbox_on_thumbnail(thumb, r["bbox"], orig_size)
                imgtk = ImageTk.PhotoImage(img_with_box)
                self.image_refs.append(imgtk)
                panel = ttk.Label(self.inner, image=imgtk)
                panel.grid(row=i, column=0, sticky="w", pady=6)
                # open full pano on click
                panel.bind("<Button-1>", lambda e, path=pano_path: self.open_full_image(path))

                info = (
                    f"{i+1}. tile={r['tile_id']}  pano={r['pano_id']}  "
                    f"yaw={r.get('yaw_deg','—')}  pitch={r.get('pitch_deg','—')}  "
                    f"cos={r['cos_sim']:.4f}  conf01={r['conf_01']:.4f}  rel={r['rel_conf']*100:.1f}%  "
                    f"dist={r['cosine_distance']:.4f}"
                )
                ttk.Label(self.inner, text=info).grid(row=i, column=1, sticky="w", padx=8)
            except Exception as e:
                ttk.Label(self.inner, text=f"[render error: {e}]").grid(row=i, column=0, sticky="w")
        self.rendered = end

        # Load more control
        if self.rendered < total:
            if self._load_more_btn is not None:
                try:
                    self._load_more_btn.destroy()
                except Exception:
                    pass
            self._load_more_btn = ttk.Button(self.inner, text="Load more…", command=self._render_next_batch)
            self._load_more_btn.grid(row=self.rendered, column=0, columnspan=2, pady=8, sticky="w")
        else:
            if self._load_more_btn is not None:
                try:
                    self._load_more_btn.destroy()
                except Exception:
                    pass
                self._load_more_btn = None

        self.status_var.set(f"Showing {self.rendered}/{total} results.")

    def open_full_image(self, pano_path: str):
        try:
            top = tk.Toplevel(self.root)
            top.title(pano_path)
            # Scrollable canvas
            frame = ttk.Frame(top)
            frame.grid(row=0, column=0, sticky="nsew")
            top.rowconfigure(0, weight=1)
            top.columnconfigure(0, weight=1)
            canvas = tk.Canvas(frame)
            xsb = ttk.Scrollbar(frame, orient="horizontal", command=canvas.xview)
            ysb = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
            canvas.configure(xscrollcommand=xsb.set, yscrollcommand=ysb.set)
            canvas.grid(row=0, column=0, sticky="nsew")
            xsb.grid(row=1, column=0, sticky="ew")
            ysb.grid(row=0, column=1, sticky="ns")
            frame.rowconfigure(0, weight=1)
            frame.columnconfigure(0, weight=1)

            with Image.open(pano_path) as im:
                im = im.convert("RGB")
                imgtk = ImageTk.PhotoImage(im)
            # Keep reference on the Toplevel
            top._imgtk = imgtk  # type: ignore[attr-defined]
            canvas.create_image(0, 0, image=imgtk, anchor="nw")
            canvas.configure(scrollregion=(0, 0, imgtk.width(), imgtk.height()))
            top.geometry("1200x800")
        except Exception as e:
            messagebox.showerror("Open Image Error", str(e))


def main():
    root = tk.Tk()
    app = SearchUI(root)
    root.geometry("1000x700")
    root.mainloop()


if __name__ == "__main__":
    main()
