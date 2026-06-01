import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

import nats
from nats.errors import TimeoutError as NatsTimeoutError
from nats.js.api import ConsumerConfig, StreamConfig
from nats.js.errors import NotFoundError

from main_service.ingestion.types import PanoramaId

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PanoEmbeddingJob:
    pano_id: PanoramaId
    view_id: int
    image_path: str


class ReceivedPanoEmbeddingJob(Protocol):
    job: PanoEmbeddingJob

    async def ack(self) -> None:
        ...


def embedding_job_from_dict(payload: dict[str, object]) -> PanoEmbeddingJob:
    raw_pano_id = payload.get("pano_id")
    if not isinstance(raw_pano_id, str) or raw_pano_id == "":
        raise ValueError("Embedding message missing pano_id")
    raw_view_id = payload.get("view_id")
    if not isinstance(raw_view_id, int):
        raise ValueError("Embedding message missing view_id")
    raw_image_path = payload.get("image_path")
    if not isinstance(raw_image_path, str) or raw_image_path == "":
        raise ValueError("Embedding message missing image_path")
    return PanoEmbeddingJob(
        pano_id=PanoramaId(raw_pano_id),
        view_id=raw_view_id,
        image_path=raw_image_path,
    )


@dataclass
class NatsReceivedPanoEmbeddingJob:
    job: PanoEmbeddingJob
    _message: Any

    async def ack(self) -> None:
        await self._message.ack()


class NatsPanoEmbeddingJobSource:
    def __init__(self, nats_client: Any, subscription: Any) -> None:
        self._nats_client = nats_client
        self._subscription = subscription

    @classmethod
    async def connect(
        cls,
        *,
        servers: str | list[str],
        stream_name: str,
        subject: str,
        durable_consumer: str,
    ) -> "NatsPanoEmbeddingJobSource":
        server_list = [servers] if isinstance(servers, str) else servers
        logger.info(
            "embedding_nats_connect_start stream=%s subject=%s durable=%s",
            stream_name,
            subject,
            durable_consumer,
        )
        nats_client = await nats.connect(servers=server_list)
        jetstream = nats_client.jetstream()
        await _ensure_stream(jetstream, stream_name, subject)
        subscription = await jetstream.pull_subscribe(
            subject,
            durable=durable_consumer,
            stream=stream_name,
            config=ConsumerConfig(durable_name=durable_consumer),
        )
        logger.info(
            "embedding_nats_connect_complete stream=%s subject=%s durable=%s",
            stream_name,
            subject,
            durable_consumer,
        )
        return cls(nats_client=nats_client, subscription=subscription)

    async def fetch(self, limit: int) -> list[ReceivedPanoEmbeddingJob]:
        try:
            logger.info("embedding_nats_fetch_start limit=%s", limit)
            messages = await self._subscription.fetch(limit, timeout=1.0)
        except NatsTimeoutError:
            logger.info("embedding_nats_fetch_timeout limit=%s", limit)
            return []

        received_jobs: list[ReceivedPanoEmbeddingJob] = []
        invalid_messages = 0
        for message in messages:
            try:
                payload = json.loads(message.data.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Embedding message payload must be a JSON object")
                job = embedding_job_from_dict(payload)
            except Exception as exc:
                invalid_messages += 1
                await message.ack()
                logger.warning(
                    "embedding_nats_invalid_message_acked error=%s",
                    exc,
                )
                continue
            received_jobs.append(
                NatsReceivedPanoEmbeddingJob(
                    job=job,
                    _message=message,
                )
            )
        logger.info(
            "embedding_nats_fetch_complete jobs=%s invalid_messages=%s",
            len(received_jobs),
            invalid_messages,
        )
        return received_jobs

    async def close(self) -> None:
        if self._nats_client is not None:
            await self._nats_client.close()
            logger.info("embedding_nats_closed")


async def _ensure_stream(jetstream: Any, stream_name: str, subject: str) -> None:
    try:
        await jetstream.stream_info(stream_name)
    except NotFoundError:
        await jetstream.add_stream(
            config=StreamConfig(
                name=stream_name,
                subjects=[subject],
            )
        )
