import asyncio
import json
import threading
from dataclasses import dataclass
from typing import Any, Awaitable, Protocol, TypeVar

import nats
from nats.js.api import StreamConfig
from nats.js.errors import NotFoundError

from main_service.ingestion.types import MapTileKey, PanoramaId

T = TypeVar("T")


class AsyncRunner(Protocol):
    def run(self, awaitable: Awaitable[T]) -> T:
        ...


class PerCallAsyncRunner:
    def run(self, awaitable: Awaitable[T]) -> T:
        return _run_async(awaitable)


class BackgroundAsyncRunner:
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="nats-jetstream-loop",
            daemon=True,
        )
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, awaitable: Awaitable[T]) -> T:
        future = asyncio.run_coroutine_threadsafe(awaitable, self._loop)
        return future.result()

    def close(self) -> None:
        if self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self._loop.close()


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


@dataclass(frozen=True)
class PanoProcessingMessage:
    pano_id: PanoramaId
    image_path: str

    def to_dict(self) -> dict[str, object]:
        return {
            "pano_id": self.pano_id.value,
            "image_path": self.image_path,
            "source": "pano_downloader",
        }


class PanoDownloadQueue(Protocol):
    def pending_count(self) -> int:
        ...

    def enqueue(self, message: PanoDownloadMessage) -> None:
        ...


class PanoProcessingQueue(Protocol):
    def pending_count(self) -> int:
        ...

    def enqueue(self, message: PanoProcessingMessage) -> None:
        ...


class InMemoryPanoDownloadQueue:
    def __init__(self) -> None:
        self.messages: list[PanoDownloadMessage] = []

    def pending_count(self) -> int:
        return len(self.messages)

    def enqueue(self, message: PanoDownloadMessage) -> None:
        self.messages.append(message)


class InMemoryPanoProcessingQueue:
    def __init__(self) -> None:
        self.messages: list[PanoProcessingMessage] = []

    def pending_count(self) -> int:
        return len(self.messages)

    def enqueue(self, message: PanoProcessingMessage) -> None:
        self.messages.append(message)


class NatsJetStreamPanoDownloadQueue:
    def __init__(
        self,
        jetstream: Any,
        stream_name: str,
        subject: str,
        nats_client: Any | None = None,
        async_runner: AsyncRunner | None = None,
    ) -> None:
        self._jetstream = jetstream
        self._stream_name = stream_name
        self._subject = subject
        self._nats_client = nats_client
        self._async_runner = async_runner or PerCallAsyncRunner()

    @classmethod
    def connect(
        cls,
        servers: str | list[str],
        stream_name: str,
        subject: str,
    ) -> "NatsJetStreamPanoDownloadQueue":
        runner = BackgroundAsyncRunner()
        try:
            return runner.run(cls._connect_async(servers, stream_name, subject, runner))
        except Exception:
            runner.close()
            raise

    @classmethod
    async def _connect_async(
        cls,
        servers: str | list[str],
        stream_name: str,
        subject: str,
        async_runner: AsyncRunner,
    ) -> "NatsJetStreamPanoDownloadQueue":
        server_list = [servers] if isinstance(servers, str) else servers
        nats_client = await nats.connect(servers=server_list)
        jetstream = nats_client.jetstream()
        queue = cls(
            jetstream=jetstream,
            stream_name=stream_name,
            subject=subject,
            nats_client=nats_client,
            async_runner=async_runner,
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
        return self._async_runner.run(self._pending_count_async())

    async def _pending_count_async(self) -> int:
        stream_info = await self._jetstream.stream_info(self._stream_name)
        return int(stream_info.state.messages)

    def enqueue(self, message: PanoDownloadMessage) -> None:
        self._async_runner.run(self._enqueue_async(message))

    async def _enqueue_async(self, message: PanoDownloadMessage) -> None:
        payload = json.dumps(message.to_dict()).encode("utf-8")
        await self._jetstream.publish(self._subject, payload)

    def close(self) -> None:
        if self._nats_client is not None:
            self._async_runner.run(self._nats_client.close())
        if isinstance(self._async_runner, BackgroundAsyncRunner):
            self._async_runner.close()


class NatsJetStreamPanoProcessingQueue(NatsJetStreamPanoDownloadQueue):
    def enqueue(self, message: PanoProcessingMessage) -> None:
        self._async_runner.run(self._enqueue_processing_async(message))

    async def _enqueue_processing_async(self, message: PanoProcessingMessage) -> None:
        payload = json.dumps(message.to_dict()).encode("utf-8")
        await self._jetstream.publish(self._subject, payload)
