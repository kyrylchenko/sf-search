from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from main_service.db.models.panorama import Panorama
from main_service.db.models.panorama_view import PanoramaView
from main_service.ingestion.types import DownloadStatus, PanoramaId, ProcessingStatus


@dataclass(frozen=True)
class PanoramaViewSpecRecord:
    source_image_path: str
    source_image_hash: str
    viewset_name: str
    viewset_description: str
    view_id: str
    view_kind: str
    view_spec_json: dict[str, object]
    view_spec_hash: str
    relative_heading: float
    pitch: float
    fov: float
    output_width: int
    output_height: int
    render_scale: int
    rendered_width: int
    rendered_height: int
    output_format: str
    image_quality: int | None
    interpolation_mode: str
    renderer_version: str


class PanoramaViewService:
    def __init__(self, engine: Engine):
        self.engine = engine

    def get_downloaded_panorama(self, pano_id: PanoramaId) -> Panorama | None:
        with Session(self.engine) as session:
            panorama = session.execute(
                select(Panorama).filter_by(orig_id=pano_id.value)
            ).scalar_one_or_none()
            if (
                panorama is None
                or panorama.download_status != DownloadStatus.DOWNLOADED.value
                or not panorama.image_path
            ):
                return None

            session.expunge(panorama)
            return panorama

    def claim_view_for_processing(
        self,
        pano_id: PanoramaId,
        spec: PanoramaViewSpecRecord,
    ) -> PanoramaView | None:
        with Session(self.engine) as session:
            panorama = session.execute(
                select(Panorama).filter_by(orig_id=pano_id.value)
            ).scalar_one_or_none()
            if panorama is None:
                return None

            view = session.execute(
                select(PanoramaView).filter_by(
                    panorama_id=panorama.id,
                    viewset_name=spec.viewset_name,
                    view_id=spec.view_id,
                    view_spec_hash=spec.view_spec_hash,
                    render_scale=spec.render_scale,
                    output_format=spec.output_format,
                    source_image_hash=spec.source_image_hash,
                )
            ).scalar_one_or_none()
            if (
                view is not None
                and view.processing_status == ProcessingStatus.COMPLETE.value
                and view.image_hash
            ):
                return None

            if view is None:
                view = PanoramaView(
                    panorama_id=panorama.id,
                    viewset_name=spec.viewset_name,
                    view_id=spec.view_id,
                    view_spec_hash=spec.view_spec_hash,
                    render_scale=spec.render_scale,
                    output_format=spec.output_format,
                    source_image_hash=spec.source_image_hash,
                )
                session.add(view)

            self._copy_spec(view, spec)
            view.processing_status = ProcessingStatus.PROCESSING.value
            view.embedding_status = view.embedding_status or ProcessingStatus.PENDING.value
            view.attempt_count = (view.attempt_count or 0) + 1
            view.last_error = None
            view.processed_at = None
            session.flush()
            session.refresh(view)
            session.expunge(view)
            session.commit()
            return view

    def mark_view_complete(
        self,
        view_id: int,
        *,
        image_path: str,
        image_hash: str,
        image_bytes: int,
    ) -> PanoramaView:
        with Session(self.engine) as session:
            view = self._get_view(session, view_id)
            view.processing_status = ProcessingStatus.COMPLETE.value
            view.image_path = image_path
            view.image_hash = image_hash
            view.image_bytes = image_bytes
            view.last_error = None
            view.processed_at = datetime.now(timezone.utc)
            session.flush()
            session.refresh(view)
            session.expunge(view)
            session.commit()
            return view

    def mark_view_failed(self, view_id: int, error: str) -> PanoramaView:
        with Session(self.engine) as session:
            view = self._get_view(session, view_id)
            view.processing_status = ProcessingStatus.FAILED.value
            view.last_error = error[:2000]
            view.processed_at = datetime.now(timezone.utc)
            session.flush()
            session.refresh(view)
            session.expunge(view)
            session.commit()
            return view

    def clear_view_temp_image_path(
        self,
        view_id: int,
        *,
        expected_path: str,
    ) -> PanoramaView:
        with Session(self.engine) as session:
            view = self._get_view(session, view_id)
            if view.image_path == expected_path:
                view.image_path = None
            session.flush()
            session.refresh(view)
            session.expunge(view)
            session.commit()
            return view

    def list_views_for_panorama(self, pano_id: PanoramaId) -> list[PanoramaView]:
        with Session(self.engine) as session:
            rows = (
                session.execute(
                    select(PanoramaView)
                    .join(Panorama, Panorama.id == PanoramaView.panorama_id)
                    .where(Panorama.orig_id == pano_id.value)
                    .order_by(PanoramaView.viewset_name, PanoramaView.view_id)
                )
                .scalars()
                .all()
            )
            for row in rows:
                session.expunge(row)
            return rows

    def _get_view(self, session: Session, view_id: int) -> PanoramaView:
        view = session.get(PanoramaView, view_id)
        if view is None:
            raise ValueError(f"Panorama view does not exist: {view_id}")
        return view

    def _copy_spec(self, view: PanoramaView, spec: PanoramaViewSpecRecord) -> None:
        view.source_image_path = spec.source_image_path
        view.source_image_hash = spec.source_image_hash
        view.viewset_name = spec.viewset_name
        view.viewset_description = spec.viewset_description
        view.view_id = spec.view_id
        view.view_kind = spec.view_kind
        view.view_spec_json = spec.view_spec_json
        view.view_spec_hash = spec.view_spec_hash
        view.relative_heading = spec.relative_heading
        view.pitch = spec.pitch
        view.fov = spec.fov
        view.output_width = spec.output_width
        view.output_height = spec.output_height
        view.render_scale = spec.render_scale
        view.rendered_width = spec.rendered_width
        view.rendered_height = spec.rendered_height
        view.output_format = spec.output_format
        view.image_quality = spec.image_quality
        view.interpolation_mode = spec.interpolation_mode
        view.renderer_version = spec.renderer_version
