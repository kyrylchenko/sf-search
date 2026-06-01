import asyncio
import json

from main_service.downloader.nats_source import (
    NatsPanoDownloadJobSource,
    pano_download_job_from_dict,
)
from main_service.downloader.runner import PanoDownloadJob
from main_service.ingestion.types import PanoramaId


def test_pano_download_job_from_dict_parses_pano_id() -> None:
    assert pano_download_job_from_dict({"pano_id": "pano-a"}) == PanoDownloadJob(
        pano_id=PanoramaId("pano-a")
    )


class FakeNatsMessage:
    def __init__(self, payload: dict[str, object] | bytes) -> None:
        self.data = (
            payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        )
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


def test_nats_source_fetches_jobs_and_acks_underlying_messages() -> None:
    message = FakeNatsMessage({"pano_id": "pano-a"})
    subscription = FakeSubscription([message])
    source = NatsPanoDownloadJobSource(
        nats_client=None,
        subscription=subscription,
    )

    jobs = asyncio.run(source.fetch(limit=5))
    asyncio.run(jobs[0].ack())

    assert subscription.fetch_calls == [{"limit": 5, "timeout": 1.0}]
    assert jobs[0].job == PanoDownloadJob(pano_id=PanoramaId("pano-a"))
    assert message.acked


def test_nats_source_acks_invalid_messages_and_returns_valid_jobs() -> None:
    invalid_json = FakeNatsMessage(b"{not-json")
    invalid_schema = FakeNatsMessage({"image_path": ".local/panoramas/pano-a.jpg"})
    valid = FakeNatsMessage({"pano_id": "pano-a"})
    subscription = FakeSubscription([invalid_json, invalid_schema, valid])
    source = NatsPanoDownloadJobSource(
        nats_client=None,
        subscription=subscription,
    )

    jobs = asyncio.run(source.fetch(limit=5))

    assert [job.job for job in jobs] == [PanoDownloadJob(pano_id=PanoramaId("pano-a"))]
    assert invalid_json.acked
    assert invalid_schema.acked
    assert not valid.acked
