from __future__ import annotations
from __future__ import annotations
import asyncio
import queue
import threading
from dataclasses import dataclass
from typing import List, Set, Tuple

import tkinter as tk
from tkinter import ttk

from streetlevel import streetview
from tile import lonlat_to_tilexy

# Bounding box for SF (lat, lon)

UL: Tuple[float, float] = (37.808496271257575, -122.53331206814802)
BR: Tuple[float, float] = (37.69569610052092, -122.34569836515641)
ZOOM = 17

# Concurrency for coverage fetches
CONCURRENCY = 20

# Visuals
CELL_SIZE = 28  # pixels per tile cell (larger squares)
PADDING = 10

# Earth constants
EARTH_CIRCUMFERENCE_M = 40075016.68557849  # meters


@dataclass
class TileInfo:
    x: int
    y: int
    count: int = 0
    item_id: int | None = None  # canvas rectangle id
    text_id: int | None = None  # canvas text id


def tile_bounds_from_bbox(ul: Tuple[float, float], br: Tuple[float, float], zoom: int) -> Tuple[range, range]:
    ul_lat, ul_lon = ul
    br_lat, br_lon = br
    x1, y1 = lonlat_to_tilexy(ul_lon, ul_lat, zoom)
    x2, y2 = lonlat_to_tilexy(br_lon, br_lat, zoom)
    x_min, x_max = sorted((x1, x2))
    y_min, y_max = sorted((y1, y2))
    return range(x_min, x_max + 1), range(y_min, y_max + 1)


def color_for_count(count: int, max_count: int) -> str:
    """Map a count to a green gradient hex color. Handles max_count=0."""
    if max_count <= 0:
        return "#eeeeee"
    t = max(0.0, min(1.0, count / max_count))
    # interpolate from light green (202, 255, 191) to dark green (43, 147, 72)
    a = (202, 255, 191)
    b = (43, 147, 72)
    r = int(a[0] + (b[0] - a[0]) * t)
    g = int(a[1] + (b[1] - a[1]) * t)
    bl = int(a[2] + (b[2] - a[2]) * t)
    return f"#{r:02x}{g:02x}{bl:02x}"


class SFGridApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("SF Tile Coverage • z=17")

        # UI layout
        main = ttk.Frame(root, padding=12)
        main.grid(row=0, column=0, sticky="nsew")
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)

        # Stats row
        self.status = ttk.Label(main, text="Idle")
        self.status.grid(row=0, column=0, columnspan=2, sticky="w")
        self.avg_var = tk.StringVar(value="avg: 0.00")
        self.unique_var = tk.StringVar(value="unique: 0")
        self.done_var = tk.StringVar(value="0/0 tiles")
        ttk.Label(main, textvariable=self.avg_var).grid(row=1, column=0, sticky="w", pady=(4, 8))
        ttk.Label(main, textvariable=self.unique_var).grid(row=1, column=1, sticky="w", pady=(4, 8))
        ttk.Label(main, textvariable=self.done_var).grid(row=1, column=2, sticky="w", pady=(4, 8))

        # Canvas with scrollbars
        canvas_frame = ttk.Frame(main)
        canvas_frame.grid(row=2, column=0, columnspan=3, sticky="nsew")
        main.rowconfigure(2, weight=1)
        main.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(canvas_frame, width=1000, height=800, bg="#f6f6f6")
        hsb = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        vsb = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=hsb.set, yscrollcommand=vsb.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        # Controls
        self.start_btn = ttk.Button(main, text="Start", command=self.start_scan)
        self.start_btn.grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.tile_size_var = tk.StringVar(value="tile size: — m")
        ttk.Label(main, textvariable=self.tile_size_var).grid(row=3, column=2, sticky="e", pady=(8, 0))

        # Prepare tiles
        self.x_range, self.y_range = tile_bounds_from_bbox(UL, BR, ZOOM)
        self.tiles: dict[Tuple[int, int], TileInfo] = {}
        self.total_tiles = len(self.x_range) * len(self.y_range)
        self._draw_grid()

        # background worker and queue
        self._queue: queue.Queue[Tuple[Tuple[int, int], int, int, int]] = queue.Queue()
        self._worker_thread: threading.Thread | None = None
        self._updater_running = False

    def _draw_grid(self):
        # Compute geometry
        cols = len(self.x_range)
        rows = len(self.y_range)
        pad = PADDING
        cell_w = CELL_SIZE
        cell_h = CELL_SIZE
        total_w = pad * 2 + cols * cell_w
        total_h = pad * 2 + rows * cell_h
        self.cell_w = cell_w
        self.cell_h = cell_h
        self.pad = pad

        # Draw cells
        self.canvas.delete("all")
        for yi, y in enumerate(self.y_range):
            for xi, x in enumerate(self.x_range):
                x0 = pad + xi * cell_w
                y0 = pad + yi * cell_h
                x1 = x0 + cell_w - 1
                y1 = y0 + cell_h - 1
                rect = self.canvas.create_rectangle(x0, y0, x1, y1, fill="#eeeeee", outline="#cccccc")
                # center text label for count
                cx = (x0 + x1) // 2
                cy = (y0 + y1) // 2
                txt = self.canvas.create_text(cx, cy, text="0", fill="#333333", font=("Helvetica", 9))
                self.tiles[(x, y)] = TileInfo(x=x, y=y, count=0, item_id=rect, text_id=txt)

        # Configure scrollable region to actual content size
        self.canvas.configure(scrollregion=(0, 0, total_w, total_h))
        self.status.config(text=f"Tiles prepared: {self.total_tiles} (cols={cols}, rows={rows})")
        self.done_var.set(f"0/{self.total_tiles} tiles")

        # Compute tile size in meters at bbox center latitude
        center_lat = (UL[0] + BR[0]) / 2.0
        import math
        tile_m = math.cos(math.radians(center_lat)) * EARTH_CIRCUMFERENCE_M / (2 ** ZOOM)
        self.tile_size_var.set(f"tile size: {tile_m:.0f} m per side @ lat {center_lat:.4f}")

    def start_scan(self):
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self.start_btn.config(state=tk.DISABLED)
        self.status.config(text="Scanning…")
        self._updater_running = True
        self.root.after(100, self._drain_queue_periodic)
        self._worker_thread = threading.Thread(target=self._scan_worker, daemon=True)
        self._worker_thread.start()

    def _drain_queue_periodic(self):
        try:
            while True:
                tile_xy, tile_count, tiles_done, unique = self._queue.get_nowait()
                self._apply_tile_update(tile_xy, tile_count, tiles_done, unique)
        except queue.Empty:
            pass
        if self._updater_running:
            self.root.after(100, self._drain_queue_periodic)

    def _apply_tile_update(self, tile_xy: Tuple[int, int], tile_count: int, tiles_done: int, unique: int):
        # Update tile color based on its count relative to current max among processed tiles
        self.tiles[tile_xy].count = tile_count
        # compute dynamic max among processed tiles
        max_count = max((t.count for t in self.tiles.values()), default=0)
        color = color_for_count(tile_count, max_count)
        info = self.tiles[tile_xy]
        if info.item_id is not None:
            self.canvas.itemconfig(info.item_id, fill=color)
        # update text count inside square
        if info.text_id is not None:
            self.canvas.itemconfig(info.text_id, text=str(tile_count))
        # stats
        avg = sum(t.count for t in self.tiles.values()) / max(1, tiles_done)
        self.avg_var.set(f"avg: {avg:.2f}")
        self.unique_var.set(f"unique: {unique}")
        self.done_var.set(f"{tiles_done}/{self.total_tiles} tiles")

    def _scan_worker(self):
        asyncio.run(self._scan_async())
        # stop updater loop once done
        self._updater_running = False
        self.status.config(text="Done")
        self.start_btn.config(state=tk.NORMAL)

    async def _scan_async(self):
        tiles: List[Tuple[int, int]] = [(x, y) for y in self.y_range for x in self.x_range]
        sem = asyncio.Semaphore(CONCURRENCY)
        unique_ids: Set[str] = set()
        tiles_done = 0
        lock = asyncio.Lock()

        async def fetch_one(x: int, y: int) -> Tuple[Tuple[int, int], int, Set[str]]:
            async with sem:
                panos = await asyncio.to_thread(streetview.get_coverage_tile, x, y)
            ids = {getattr(p, 'id', None) for p in panos if getattr(p, 'id', None)}
            return (x, y), len(panos), ids

        async def handle_task(coro):
            nonlocal tiles_done, unique_ids
            xy, cnt, ids = await coro
            async with lock:
                tiles_done += 1
                unique_ids.update(ids)
                # push to UI queue
                self._queue.put((xy, cnt, tiles_done, len(unique_ids)))

        tasks = [handle_task(fetch_one(x, y)) for (x, y) in tiles]
        # Run with a limited number of concurrent tasks alive to manage memory
        # We'll gather in chunks to stream updates
        CHUNK = 100
        for i in range(0, len(tasks), CHUNK):
            await asyncio.gather(*tasks[i:i+CHUNK])


def main():
    root = tk.Tk()
    _ = SFGridApp(root)
    root.geometry("1100x900")
    root.mainloop()


if __name__ == "__main__":
    main()
