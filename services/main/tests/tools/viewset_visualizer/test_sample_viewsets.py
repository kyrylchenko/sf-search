from pathlib import Path

from main_service.tools.viewset_visualizer.viewsets import load_viewsets


def test_committed_sample_viewsets_include_requested_presets() -> None:
    root = Path(__file__).parents[5]
    viewsets = load_viewsets(root / "docs/data/viewsets")

    by_name = {viewset.name: viewset for viewset in viewsets}

    assert "wide-triptych-front-band" in by_name
    assert "center-no-sky-road" in by_name
    assert "small-object-grid-72" in by_name
    assert len(by_name["wide-triptych-front-band"].views) == 3
    assert len(by_name["center-no-sky-road"].views) == 1
    assert len(by_name["small-object-grid-72"].views) == 72
