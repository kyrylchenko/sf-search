from main_service.embedding.query_ui import QueryResult, render_results_page


def test_render_results_page_displays_local_images_and_embedding_metadata() -> None:
    html = render_results_page(
        query="u haul truck",
        results=[
            QueryResult(
                score=0.87,
                image_path=".local/panorama-views/pano-a/candidate/center.jpg",
                pano_id="pano-a",
                viewset_name="candidate",
                view_id="center",
                relative_heading=15.0,
                pitch=10.0,
                fov=77.0,
                rendered_width=1024,
                rendered_height=1024,
                model_id="google/siglip2-so400m-patch14-384",
                vector_id="42",
            )
        ],
    )

    assert "u haul truck" in html
    assert ".local/panorama-views/pano-a/candidate/center.jpg" in html
    assert "pano-a" in html
    assert "candidate / center" in html
    assert "score 0.8700" in html
    assert "heading 15.0" in html
    assert "google/siglip2-so400m-patch14-384" in html
