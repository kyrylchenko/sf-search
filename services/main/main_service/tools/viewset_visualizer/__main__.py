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
    parser.add_argument(
        "--google-api-key-env",
        default="GOOGLE_MAPS_EMBED_API_KEY",
        help="Environment variable containing a Google Maps Embed API key.",
    )
    parser.add_argument(
        "--north-offset",
        type=float,
        default=None,
        help="North-based heading of pano-relative 0 degrees. Defaults to GPano XMP PoseHeadingDegrees when present.",
    )
    parser.add_argument(
        "--pano-id",
        default=None,
        help="Google pano ID for embed links. Defaults to the pano filename stem.",
    )
    parser.add_argument("--latitude", type=float, default=None)
    parser.add_argument("--longitude", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_server(
        pano_path=args.pano,
        viewsets_dir=args.viewsets,
        host=args.host,
        port=args.port,
        edge_samples=args.edge_samples,
        google_api_key_env=args.google_api_key_env,
        north_offset=args.north_offset,
        pano_id=args.pano_id,
        latitude=args.latitude,
        longitude=args.longitude,
    )


if __name__ == "__main__":
    main()
