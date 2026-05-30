import asyncio
from dataclasses import dataclass

from main_service.service_loop import run_service_loop


@dataclass(frozen=True)
class FakeBatchResult:
    count: int


def test_service_loop_continues_after_idle_batch() -> None:
    results = [FakeBatchResult(0), FakeBatchResult(1)]
    calls: list[str] = []
    sleeps: list[float] = []

    async def run_batch() -> FakeBatchResult:
        calls.append("batch")
        return results.pop(0)

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    asyncio.run(
        run_service_loop(
            service_name="test-worker",
            run_batch=run_batch,
            should_idle=lambda result: result.count == 0,
            idle_sleep_seconds=2.5,
            once=False,
            max_batches=2,
            sleep=sleep,
        )
    )

    assert calls == ["batch", "batch"]
    assert sleeps == [2.5]


def test_service_loop_once_exits_after_first_batch_without_sleeping() -> None:
    calls: list[str] = []
    sleeps: list[float] = []

    async def run_batch() -> FakeBatchResult:
        calls.append("batch")
        return FakeBatchResult(0)

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    asyncio.run(
        run_service_loop(
            service_name="test-worker",
            run_batch=run_batch,
            should_idle=lambda result: result.count == 0,
            idle_sleep_seconds=2.5,
            once=True,
            sleep=sleep,
        )
    )

    assert calls == ["batch"]
    assert sleeps == []
