import asyncio
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Protocol, TypeVar

import nats
from nats.js.api import StreamConfig
from nats.js.errors import NotFoundError

from main_service.ingestion.types import MapTileKey, PanoramaId

T = TypeVar("T")


def _run_async(awaitable: Awaitable[T]) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    raise RuntimeError(
        "NatsJetStreamPanoDownloadQueue sync methods cannot run inside "
        "an active asyncio event loop."
    )


@dataclass(frozen=True)
class PanoDownloadMessage:
    pano_id: PanoramaId
    source_tile: MapTileKey

    def to_dict(self) -> dict[str, object]:
        return {
            "pano_id": self.pano_id.value,
            "source": "coverage_discovery",
            "discovered_from_tile": {
                "x": self.source_tile.x,
                "y": self.source_tile.y,
                "z": self.source_tile.z,
            },
        }


class PanoDownloadQueue(Protocol):
    def pending_count(self) -> int:
        ...

    def enqueue(self, message: PanoDownloadMessage) -> None:
        ...


class InMemoryPanoDownloadQueue:
    def __init__(self) -> None:
        self.messages: list[PanoDownloadMessage] = []

    def pending_count(self) -> int:
        return len(self.messages)

    def enqueue(self, message: PanoDownloadMessage) -> None:
        self.messages.append(message)


class NatsJetStreamPanoDownloadQueue:
    def __init__(
        self,
        jetstream: Any,
        stream_name: str,
        subject: str,
        nats_client: Any | None = None,
    ) -> None:
        self._jetstream = jetstream
        self._stream_name = stream_name
        self._subject = subject
        self._nats_client = nats_client

    @classmethod
    def connect(
        cls,
        servers: str | list[str],
        stream_name: str,
        subject: str,
    ) -> "NatsJetStreamPanoDownloadQueue":
        return _run_async(cls._connect_async(servers, stream_name, subject))

    @classmethod
    async def _connect_async(
        cls,
        servers: str | list[str],
        stream_name: str,
        subject: str,
    ) -> "NatsJetStreamPanoDownloadQueue":
        server_list = [servers] if isinstance(servers, str) else servers
        nats_client = await nats.connect(servers=server_list)
        jetstream = nats_client.jetstream()
        queue = cls(
            jetstream=jetstream,
            stream_name=stream_name,
            subject=subject,
            nats_client=nats_client,
        )
        await queue.ensure_stream_async()
        return queue

    async def ensure_stream_async(self) -> None:
        try:
            await self._jetstream.stream_info(self._stream_name)
        except NotFoundError:
            await self._jetstream.add_stream(
                config=StreamConfig(
                    name=self._stream_name,
                    subjects=[self._subject],
                )
            )

    def pending_count(self) -> int:
        return _run_async(self._pending_count_async())

    async def _pending_count_async(self) -> int:
        stream_info = await self._jetstream.stream_info(self._stream_name)
        return int(stream_info.state.messages)

    def enqueue(self, message: PanoDownloadMessage) -> None:
        _run_async(self._enqueue_async(message))

    async def _enqueue_async(self, message: PanoDownloadMessage) -> None:
        payload = json.dumps(message.to_dict()).encode("utf-8")
        await self._jetstream.publish(self._subject, payload)

    def close(self) -> None:
        if self._nats_client is not None:
            _run_async(self._nats_client.close())
