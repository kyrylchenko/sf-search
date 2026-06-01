from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from main_service.db.models.base import Base
from main_service.db.models.map_tile import MapTile
from main_service.db.models.panorama import Panorama
from main_service.db.models.panorama_view import PanoramaView
from main_service.db.models.panorama_view_embedding import PanoramaViewEmbedding
from main_service.ingestion.types import DownloadStatus, ProcessingStatus
from main_service.monitoring.snapshot import QueueSnapshotSource, build_pipeline_snapshot


class FakeQueue:
    def __init__(self, pending: int | Exception) -> None:
        self.pending = pending

    def pending_count(self) -> int:
        if isinstance(self.pending, Exception):
            raise self.pending
        return self.pending


class FakeQdrant:
    def __init__(self, snapshot: dict[str, object] | Exception) -> None:
        self.snapshot = snapshot

    def collection_snapshot(self) -> dict[str, object]:
        if isinstance(self.snapshot, Exception):
            raise self.snapshot
        return self.snapshot


def make_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


def test_build_pipeline_snapshot_counts_database_statuses() -> None:
    engine = make_engine()
    with Session(engine) as session:
        pano = Panorama(
            orig_id="pano-a",
            image_hash="hash-a",
            latitude=37.77,
            longitude=-122.42,
            download_status=DownloadStatus.DOWNLOADED.value,
        )
        session.add_all(
            [
                MapTile(x=1, y=2, z=17, discovery_status=ProcessingStatus.COMPLETE.value),
                MapTile(x=1, y=3, z=17, discovery_status=ProcessingStatus.PENDING.value),
                pano,
            ]
        )
        session.flush()
        view = PanoramaView(
            panorama_id=pano.id,
            viewset_name="small",
            viewset_description="small views",
            view_id="small-001",
            view_kind="object",
            view_spec_json={"id": "small-001"},
            view_spec_hash="spec-hash",
            relative_heading=0,
            pitch=0,
            fov=40,
            output_width=512,
            output_height=512,
            render_scale=2,
            rendered_width=1024,
            rendered_height=1024,
            output_format="jpeg",
            image_quality=95,
            interpolation_mode="bicubic",
            renderer_version="py360convert.e2p",
            source_image_path=".local/panos/pano-a.jpg",
            source_image_hash="hash-a",
            image_path=".local/views/pano-a/small-001.jpg",
            image_hash="view-hash",
            image_bytes=123,
            processing_status=ProcessingStatus.COMPLETE.value,
        )
        session.add(view)
        session.flush()
        session.add(
            PanoramaViewEmbedding(
                panorama_view_id=view.id,
                model_provider="transformers",
                model_id="model-a",
                model_revision="main",
                preprocess_version="v1",
                source_image_path=view.image_path or "",
                source_image_hash=view.image_hash or "",
                source_image_bytes=view.image_bytes,
                embedding_status=ProcessingStatus.COMPLETE.value,
                embedding_dimension=4,
                embedding_dtype="float32",
            )
        )
        session.commit()

    snapshot = build_pipeline_snapshot(
        engine=engine,
        queues=QueueSnapshotSource(
            download=FakeQueue(7),
            processing=FakeQueue(5),
            embedding=FakeQueue(3),
        ),
    )

    assert snapshot.status_counts["map_tiles"]["complete"] == 1
    assert snapshot.status_counts["map_tiles"]["pending"] == 1
    assert snapshot.status_counts["panoramas"]["downloaded"] == 1
    assert snapshot.status_counts["panorama_views"]["complete"] == 1
    assert snapshot.status_counts["embeddings"]["complete"] == 1
    assert snapshot.queue_depths == {
        "download": 7,
        "processing": 5,
        "embedding": 3,
    }
    assert snapshot.coverage["panos_with_location"] == 1


def test_build_pipeline_snapshot_includes_embedding_progress() -> None:
    engine = make_engine()
    with Session(engine) as session:
        pano = Panorama(
            orig_id="pano-a",
            image_hash="hash-a",
            download_status=DownloadStatus.DOWNLOADED.value,
        )
        session.add(pano)
        session.flush()
        views = []
        for index in range(2):
            view = PanoramaView(
                panorama_id=pano.id,
                viewset_name="small",
                viewset_description="small views",
                view_id=f"small-{index:03d}",
                view_kind="object",
                view_spec_json={"id": f"small-{index:03d}"},
                view_spec_hash=f"spec-hash-{index}",
                relative_heading=0,
                pitch=0,
                fov=40,
                output_width=512,
                output_height=512,
                render_scale=2,
                rendered_width=1024,
                rendered_height=1024,
                output_format="jpeg",
                image_quality=95,
                interpolation_mode="bicubic",
                renderer_version="py360convert.e2p",
                source_image_path=".local/panos/pano-a.jpg",
                source_image_hash="hash-a",
                image_path=f".local/views/pano-a/small-{index:03d}.jpg",
                image_hash=f"view-hash-{index}",
                image_bytes=123,
                processing_status=ProcessingStatus.COMPLETE.value,
            )
            session.add(view)
            views.append(view)
        session.flush()
        for view in views:
            session.add(
                PanoramaViewEmbedding(
                    panorama_view_id=view.id,
                    model_provider="transformers",
                    model_id="model-a",
                    model_revision="main",
                    preprocess_version="v1",
                    source_image_path=view.image_path or "",
                    source_image_hash=view.image_hash or "",
                    source_image_bytes=view.image_bytes,
                    embedding_status=ProcessingStatus.COMPLETE.value,
                    embedding_dimension=4,
                    embedding_dtype="float32",
                )
            )
        session.commit()

    snapshot = build_pipeline_snapshot(
        engine=engine,
        queues=QueueSnapshotSource(
            download=FakeQueue(0),
            processing=FakeQueue(0),
            embedding=FakeQueue(0),
        ),
    )

    assert snapshot.embedding_progress["panos_fully_embedded"] == 1
    assert snapshot.embedding_progress["embeddings_complete"] == 2
    assert snapshot.embedding_progress["panos_with_multiple_views"] == 1


def test_build_pipeline_snapshot_keeps_working_when_queue_depth_fails() -> None:
    engine = make_engine()

    snapshot = build_pipeline_snapshot(
        engine=engine,
        queues=QueueSnapshotSource(
            download=FakeQueue(RuntimeError("nats unavailable")),
            processing=FakeQueue(5),
            embedding=FakeQueue(3),
        ),
    )

    assert snapshot.queue_depths == {
        "download": None,
        "processing": 5,
        "embedding": 3,
    }
    assert snapshot.queue_errors == {"download": "nats unavailable"}


def test_build_pipeline_snapshot_includes_qdrant_collection_status() -> None:
    engine = make_engine()

    snapshot = build_pipeline_snapshot(
        engine=engine,
        queues=QueueSnapshotSource(
            download=FakeQueue(0),
            processing=FakeQueue(0),
            embedding=FakeQueue(0),
        ),
        qdrant=FakeQdrant(
            {
                "collection": "panorama_view_embeddings_siglip2",
                "status": "green",
                "points_count": 42,
                "vectors_count": 42,
                "indexed_vectors_count": 40,
            }
        ),
    )

    assert snapshot.qdrant == {
        "collection": "panorama_view_embeddings_siglip2",
        "status": "green",
        "points_count": 42,
        "vectors_count": 42,
        "indexed_vectors_count": 40,
    }
    assert snapshot.qdrant_errors == {}


def test_build_pipeline_snapshot_keeps_working_when_qdrant_fails() -> None:
    engine = make_engine()

    snapshot = build_pipeline_snapshot(
        engine=engine,
        queues=QueueSnapshotSource(
            download=FakeQueue(0),
            processing=FakeQueue(0),
            embedding=FakeQueue(0),
        ),
        qdrant=FakeQdrant(RuntimeError("qdrant unavailable")),
    )

    assert snapshot.qdrant is None
    assert snapshot.qdrant_errors == {"collection": "qdrant unavailable"}
