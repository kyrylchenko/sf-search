from pathlib import Path

from main_service.tools.viewset_visualizer.viewsets import load_viewsets


def test_committed_sample_viewsets_include_requested_presets() -> None:
    root = Path(__file__).parents[5]
    viewsets = load_viewsets(root / "docs/data/viewsets")

    by_name = {viewset.name: viewset for viewset in viewsets}

    assert "wide-triptych-front-band" in by_name
    assert "center-no-sky-road" in by_name
    assert "small-object-grid-72" in by_name
    assert "v1-wide-center" in by_name
    assert len(by_name["wide-triptych-front-band"].views) == 3
    assert len(by_name["center-no-sky-road"].views) == 1
    assert len(by_name["small-object-grid-72"].views) == 72
    assert len(by_name["v1-wide-center"].views) == 7
    assert by_name["center-no-sky-road"].views[0].fov == 91
    assert [view.fov for view in by_name["v1-wide-center"].views[:6]] == [100] * 6
    assert by_name["v1-wide-center"].views[6].fov == 77
    assert all(
        view.view_kind != "small_object"
        for view in by_name["v1-wide-center"].views
    )
