import asyncio
import json

from main_service.embedding.nats_source import (
    PanoEmbeddingJob,
    embedding_job_from_dict,
    NatsPanoEmbeddingJobSource,
)
from main_service.ingestion.types import PanoramaId


class FakeNatsMessage:
    def __init__(self, payload: bytes) -> None:
        self.data = payload
        self.acked = False

    async def ack(self) -> None:
        self.acked = True


class FakeSubscription:
    def __init__(self, messages: list[FakeNatsMessage]) -> None:
        self.messages = messages
        self.fetch_calls: list[dict[str, object]] = []

    async def fetch(self, limit: int, timeout: float) -> list[FakeNatsMessage]:
        self.fetch_calls.append({"limit": limit, "timeout": timeout})
        return self.messages[:limit]


def test_embedding_job_from_dict_requires_view_id_and_image_path() -> None:
    job = embedding_job_from_dict(
        {
            "pano_id": "pano-a",
            "view_id": 123,
            "image_path": ".local/panorama-views/pano-a/candidate/center.jpg",
        }
    )

    assert job == PanoEmbeddingJob(
        pano_id=PanoramaId("pano-a"),
        view_id=123,
        image_path=".local/panorama-views/pano-a/candidate/center.jpg",
    )


def test_nats_embedding_source_fetches_and_acks_jobs() -> None:
    message = FakeNatsMessage(
        b'{"pano_id":"pano-a","view_id":123,"image_path":".local/view.jpg"}'
    )
    subscription = FakeSubscription([message])
    source = NatsPanoEmbeddingJobSource(nats_client=None, subscription=subscription)

    jobs = asyncio.run(source.fetch(limit=5))
    asyncio.run(jobs[0].ack())

    assert subscription.fetch_calls == [{"limit": 5, "timeout": 1.0}]
    assert jobs[0].job == PanoEmbeddingJob(
        pano_id=PanoramaId("pano-a"),
        view_id=123,
        image_path=".local/view.jpg",
    )
    assert message.acked


def test_nats_embedding_source_acks_invalid_messages_and_returns_valid_jobs() -> None:
    invalid_json = FakeNatsMessage(b"{not-json")
    invalid_schema = FakeNatsMessage(json.dumps({"pano_id": "pano-a"}).encode("utf-8"))
    valid = FakeNatsMessage(
        b'{"pano_id":"pano-a","view_id":123,"image_path":".local/view.jpg"}'
    )
    subscription = FakeSubscription([invalid_json, invalid_schema, valid])
    source = NatsPanoEmbeddingJobSource(nats_client=None, subscription=subscription)

    jobs = asyncio.run(source.fetch(limit=5))

    assert [job.job for job in jobs] == [
        PanoEmbeddingJob(
            pano_id=PanoramaId("pano-a"),
            view_id=123,
            image_path=".local/view.jpg",
        )
    ]
    assert invalid_json.acked
    assert invalid_schema.acked
    assert not valid.acked
