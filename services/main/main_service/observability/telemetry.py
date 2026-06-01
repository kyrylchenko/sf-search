import contextlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class PipelineTelemetry(Protocol):
    enabled: bool

    def record_event(self, event: str, payload: dict[str, object]) -> None:
        ...

    def record_duration(
        self,
        name: str,
        seconds: float,
        attributes: dict[str, object],
    ) -> None:
        ...

    def set_gauge(
        self,
        name: str,
        value: int | float,
        attributes: dict[str, object],
    ) -> None:
        ...

    def span(self, name: str, attributes: dict[str, object] | None = None) -> Iterator[None]:
        ...

    def flush(self) -> None:
        ...

    def shutdown(self) -> None:
        ...


@dataclass
class NoopTelemetry:
    enabled: bool = False

    def record_event(self, event: str, payload: dict[str, object]) -> None:
        return None

    def record_duration(
        self,
        name: str,
        seconds: float,
        attributes: dict[str, object],
    ) -> None:
        return None

    def set_gauge(
        self,
        name: str,
        value: int | float,
        attributes: dict[str, object],
    ) -> None:
        return None

    @contextlib.contextmanager
    def span(
        self,
        name: str,
        attributes: dict[str, object] | None = None,
    ) -> Iterator[None]:
        yield

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


def configure_observability(settings: object, service_name: str) -> PipelineTelemetry:
    if not bool(getattr(settings, "observability_enabled", False)):
        return NoopTelemetry()
    try:
        return _configure_open_telemetry(settings, service_name)
    except ImportError as exc:
        logger.warning(
            "observability_disabled_missing_dependency service=%s error=%s",
            service_name,
            exc,
        )
        return NoopTelemetry()
    except Exception:
        logger.exception("observability_setup_failed service=%s", service_name)
        return NoopTelemetry()


def _configure_open_telemetry(settings: object, service_name: str) -> PipelineTelemetry:
    from opentelemetry import metrics, trace
    from opentelemetry._logs import set_logger_provider
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.metrics import Observation
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    endpoint = str(getattr(settings, "otel_exporter_otlp_endpoint"))
    insecure = bool(getattr(settings, "otel_exporter_otlp_insecure"))
    timeout = float(getattr(settings, "otel_exporter_otlp_timeout_seconds"))
    metric_interval = int(getattr(settings, "otel_metric_export_interval_millis"))
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": str(getattr(settings, "otel_service_version", "local")),
            "deployment.environment": str(
                getattr(settings, "deployment_environment", "local")
            ),
        }
    )

    trace_provider = TracerProvider(resource=resource)
    trace_provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=endpoint,
                insecure=insecure,
                timeout=timeout,
            )
        )
    )
    trace.set_tracer_provider(trace_provider)

    metric_exporter = OTLPMetricExporter(
        endpoint=endpoint,
        insecure=insecure,
        timeout=timeout,
    )
    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[
            PeriodicExportingMetricReader(
                metric_exporter,
                export_interval_millis=metric_interval,
            )
        ],
    )
    metrics.set_meter_provider(meter_provider)

    log_provider = LoggerProvider(resource=resource)
    log_provider.add_log_record_processor(
        BatchLogRecordProcessor(
            OTLPLogExporter(
                endpoint=endpoint,
                insecure=insecure,
                timeout=timeout,
            )
        )
    )
    set_logger_provider(log_provider)
    logging.getLogger().addHandler(
        LoggingHandler(level=logging.NOTSET, logger_provider=log_provider)
    )

    meter = metrics.get_meter("sf_search.pipeline")
    return OpenTelemetryRecorder(
        service_name=service_name,
        tracer=trace.get_tracer("sf_search.pipeline"),
        event_counter=meter.create_counter(
            "sf_search_pipeline_events_total",
            description="Pipeline progress events by service and event.",
        ),
        duration_histogram=meter.create_histogram(
            "sf_search_pipeline_duration_seconds",
            unit="s",
            description="Pipeline operation durations.",
        ),
        gauge_callback_type=Observation,
        meter=meter,
        trace_provider=trace_provider,
        meter_provider=meter_provider,
        log_provider=log_provider,
    )


class OpenTelemetryRecorder:
    enabled = True

    def __init__(
        self,
        *,
        service_name: str,
        tracer: Any,
        event_counter: Any,
        duration_histogram: Any,
        gauge_callback_type: Any,
        meter: Any,
        trace_provider: Any,
        meter_provider: Any,
        log_provider: Any,
    ) -> None:
        self.service_name = service_name
        self._tracer = tracer
        self._event_counter = event_counter
        self._duration_histogram = duration_histogram
        self._observation_type = gauge_callback_type
        self._trace_provider = trace_provider
        self._meter_provider = meter_provider
        self._log_provider = log_provider
        self._gauges: dict[str, dict[tuple[tuple[str, object], ...], int | float]] = {}
        self._observable_gauges: set[str] = set()
        self._meter = meter

    def record_event(self, event: str, payload: dict[str, object]) -> None:
        self._event_counter.add(
            1,
            {
                "service": self.service_name,
                "event": event,
            },
        )

    def record_duration(
        self,
        name: str,
        seconds: float,
        attributes: dict[str, object],
    ) -> None:
        self._duration_histogram.record(
            seconds,
            _low_cardinality_attributes(
                {"service": self.service_name, "operation": name, **attributes}
            ),
        )

    def set_gauge(
        self,
        name: str,
        value: int | float,
        attributes: dict[str, object],
    ) -> None:
        if name not in self._observable_gauges:
            self._meter.create_observable_gauge(
                name,
                callbacks=[self._make_gauge_callback(name)],
            )
            self._observable_gauges.add(name)
        key = tuple(sorted(_low_cardinality_attributes(attributes).items()))
        self._gauges.setdefault(name, {})[key] = value

    def _make_gauge_callback(self, name: str) -> Any:
        def callback(_options: object) -> list[object]:
            return [
                self._observation_type(value, dict(attributes))
                for attributes, value in self._gauges.get(name, {}).items()
            ]

        return callback

    @contextlib.contextmanager
    def span(
        self,
        name: str,
        attributes: dict[str, object] | None = None,
    ) -> Iterator[None]:
        with self._tracer.start_as_current_span(
            name,
            attributes=_low_cardinality_attributes(attributes or {}),
        ):
            yield

    def flush(self) -> None:
        self._trace_provider.force_flush()
        self._meter_provider.force_flush()
        self._log_provider.force_flush()

    def shutdown(self) -> None:
        self.flush()
        self._trace_provider.shutdown()
        self._meter_provider.shutdown()
        self._log_provider.shutdown()


def _low_cardinality_attributes(
    attributes: dict[str, object],
) -> dict[str, str | int | float | bool]:
    clean: dict[str, str | int | float | bool] = {}
    for key, value in attributes.items():
        if isinstance(value, str | int | float | bool):
            clean[key] = value
    return clean
