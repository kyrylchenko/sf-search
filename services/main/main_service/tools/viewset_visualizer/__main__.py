import argparse
from pathlib import Path

from main_service.tools.viewset_visualizer.server import run_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize pano viewset overlays.")
    parser.add_argument("--pano", required=True, type=Path, help="Path to pano image.")
    parser.add_argument(
        "--viewsets",
        required=True,
        type=Path,
        help="Folder containing .json viewset definitions.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument(
        "--edge-samples",
        default=49,
        type=int,
        help="Samples per frustum edge. Higher values draw smoother curves.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_server(
        pano_path=args.pano,
        viewsets_dir=args.viewsets,
        host=args.host,
        port=args.port,
        edge_samples=args.edge_samples,
    )


if __name__ == "__main__":
    main()
