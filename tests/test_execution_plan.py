"""Tests for the install-time plan / per-request execution state split.

These tests drive `PrometheusInstrumentatorMiddleware` directly on top of
recording ASGI apps. They lock in the order of ASGI events and metric
callbacks for normal, streaming, error, cancelled and empty-body requests,
and prove that the no-body path never buffers or copies response bodies.
"""

import asyncio
from typing import List, Tuple

import pytest
from prometheus_client import REGISTRY, CollectorRegistry

from prometheus_fastapi_instrumentator import metrics
from prometheus_fastapi_instrumentator.middleware import (
    PrometheusInstrumentatorMiddleware,
    RequestExecutionState,
)
from prometheus_fastapi_instrumentator.plan import InstrumentationPlan


def make_scope(path: str = "/items", method: str = "GET") -> dict:
    return {
        "type": "http",
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "root_path": "",
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }


async def empty_receive() -> dict:
    return {"type": "http.request", "body": b"", "more_body": False}


def response_start(status: int = 200) -> dict:
    return {
        "type": "http.response.start",
        "status": status,
        "headers": [(b"content-type", b"text/plain")],
    }


def body_message(body: bytes, more_body: bool = False) -> dict:
    return {"type": "http.response.body", "body": body, "more_body": more_body}


class RecordingSend:
    """Send callable that records every forwarded message."""

    def __init__(self) -> None:
        self.messages: List[dict] = []

    async def __call__(self, message: dict) -> None:
        self.messages.append(message)


def spy_on_states(monkeypatch) -> List[RequestExecutionState]:
    """Captures every RequestExecutionState created by the middleware."""
    import prometheus_fastapi_instrumentator.middleware as middleware_module

    created: List[RequestExecutionState] = []
    original = middleware_module.RequestExecutionState

    class SpyState(original):  # type: ignore[valid-type, misc]
        def __init__(self, plan, scope):
            super().__init__(plan, scope)
            created.append(self)

    monkeypatch.setattr(middleware_module, "RequestExecutionState", SpyState)
    return created


async def _wait_for(events: list, item: str) -> None:
    while item not in events:
        await asyncio.sleep(0)


# ------------------------------------------------------------------------------
# Event and sample order


async def test_event_and_metric_order_streaming() -> None:
    """Locks the order of ASGI events and metric callback invocations."""
    events: List[object] = []
    chunks = [b"0xxx", b"1xxx", b"2xxx"]

    async def app(scope, receive, send) -> None:
        events.append("app:start")
        await send(response_start())
        for chunk in chunks:
            await send(body_message(chunk, more_body=True))
        await send(body_message(b""))
        events.append("app:done")

    def first_instrumentation(info: metrics.Info) -> None:
        events.append(("metric:sync-1", info.modified_status, info.response.body))

    def second_instrumentation(info: metrics.Info) -> None:
        events.append(("metric:sync-2", info.modified_duration >= 0))

    async def async_instrumentation(info: metrics.Info) -> None:
        events.append(("metric:async", info.modified_handler))

    plan = InstrumentationPlan(
        instrumentations=[first_instrumentation, second_instrumentation],
        async_instrumentations=[async_instrumentation],
        body_handlers=[r".*"],
        should_group_untemplated=False,
    )
    middleware = PrometheusInstrumentatorMiddleware(app, plan=plan)

    send = RecordingSend()
    await middleware(make_scope(), empty_receive, send)

    # All app messages are forwarded in order.
    assert [m["type"] for m in send.messages] == ["http.response.start"] + [
        "http.response.body"
    ] * 4

    # Metric callbacks only run after the app has fully completed, so the
    # duration is never recorded before the response is sent completely.
    assert events[0] == "app:start"
    assert events[1] == "app:done"
    assert events[2] == ("metric:sync-1", "2xx", b"0xxx1xxx2xxx")
    assert events[3] == ("metric:sync-2", True)
    assert events[4] == ("metric:async", "/items")


async def test_metric_samples_recorded_in_order() -> None:
    """Locks the samples produced for a normal request."""
    registry = CollectorRegistry(auto_describe=True)

    async def app(scope, receive, send) -> None:
        await send(response_start(status=201))
        await send(body_message(b"hello"))

    plan = InstrumentationPlan(
        instrumentations=[
            metrics.requests(registry=registry),
            metrics.latency(registry=registry),
        ],
        should_group_untemplated=False,
    )
    middleware = PrometheusInstrumentatorMiddleware(app, plan=plan)

    await middleware(make_scope(), empty_receive, RecordingSend())

    assert (
        registry.get_sample_value(
            "http_requests_total",
            {"handler": "/items", "method": "GET", "status": "2xx"},
        )
        == 1.0
    )
    assert (
        registry.get_sample_value(
            "http_request_duration_seconds_count",
            {"handler": "/items", "method": "GET", "status": "2xx"},
        )
        == 1.0
    )


async def test_duration_not_recorded_before_app_returns() -> None:
    """Background work after the last body chunk delays the metric callback."""
    events: List[str] = []
    background_gate = asyncio.Event()

    async def app(scope, receive, send) -> None:
        await send(response_start())
        await send(body_message(b"data"))
        events.append("response-sent")
        # Simulates a Starlette background task: the ASGI app is still
        # running even though the response body is fully sent.
        await background_gate.wait()
        events.append("background-done")

    def instrumentation(info: metrics.Info) -> None:
        events.append("metric")

    middleware = PrometheusInstrumentatorMiddleware(
        app, plan=InstrumentationPlan(instrumentations=[instrumentation])
    )

    send = RecordingSend()
    task = asyncio.create_task(middleware(make_scope(), empty_receive, send))

    await asyncio.wait_for(_wait_for(events, "response-sent"), timeout=5)
    assert events == ["response-sent"]
    assert "metric" not in events

    background_gate.set()
    await task
    assert events == ["response-sent", "background-done", "metric"]


# ------------------------------------------------------------------------------
# Exceptions and cancellation


async def test_exception_propagates_and_metrics_use_default_500() -> None:
    calls: List[metrics.Info] = []

    async def app(scope, receive, send) -> None:
        raise RuntimeError("boom")

    middleware = PrometheusInstrumentatorMiddleware(
        app,
        plan=InstrumentationPlan(instrumentations=[calls.append]),
    )

    with pytest.raises(RuntimeError, match="boom"):
        await middleware(make_scope(), empty_receive, RecordingSend())

    assert len(calls) == 1
    assert calls[0].modified_status == "5xx"


async def test_exception_after_response_start_keeps_status() -> None:
    calls: List[metrics.Info] = []

    async def app(scope, receive, send) -> None:
        await send(response_start(status=200))
        await send(body_message(b"partial", more_body=True))
        raise ValueError("mid-stream failure")

    middleware = PrometheusInstrumentatorMiddleware(
        app,
        plan=InstrumentationPlan(instrumentations=[calls.append]),
    )

    with pytest.raises(ValueError, match="mid-stream failure"):
        await middleware(make_scope(), empty_receive, RecordingSend())

    assert len(calls) == 1
    assert calls[0].modified_status == "2xx"


async def test_metric_callback_exception_propagates() -> None:
    async def app(scope, receive, send) -> None:
        await send(response_start())
        await send(body_message(b"data"))

    def broken_instrumentation(info: metrics.Info) -> None:
        raise KeyError("metric failure")

    middleware = PrometheusInstrumentatorMiddleware(
        app,
        plan=InstrumentationPlan(instrumentations=[broken_instrumentation]),
    )

    with pytest.raises(KeyError, match="metric failure"):
        await middleware(make_scope(), empty_receive, RecordingSend())


async def test_client_cancel_records_once_and_releases_gauge() -> None:
    started = asyncio.Event()
    hang = asyncio.Event()
    calls: List[metrics.Info] = []

    async def app(scope, receive, send) -> None:
        await send(response_start())
        await send(body_message(b"chunk", more_body=True))
        started.set()
        await hang.wait()

    plan = InstrumentationPlan(
        instrumentations=[calls.append],
        should_instrument_requests_inprogress=True,
        inprogress_name="test_cancel_inprogress",
    )
    middleware = PrometheusInstrumentatorMiddleware(app, plan=plan)

    task = asyncio.create_task(middleware(make_scope(), empty_receive, RecordingSend()))
    await asyncio.wait_for(started.wait(), timeout=5)

    # While the response is not fully sent, no duration may be recorded.
    assert calls == []
    assert REGISTRY.get_sample_value("test_cancel_inprogress") == 1.0

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(calls) == 1
    assert REGISTRY.get_sample_value("test_cancel_inprogress") == 0.0


# ------------------------------------------------------------------------------
# Body collection


async def test_empty_body_with_body_handlers() -> None:
    calls: List[metrics.Info] = []

    async def app(scope, receive, send) -> None:
        await send(response_start(status=204))
        await send(body_message(b""))

    middleware = PrometheusInstrumentatorMiddleware(
        app,
        plan=InstrumentationPlan(instrumentations=[calls.append], body_handlers=[r".*"]),
    )

    await middleware(make_scope(), empty_receive, RecordingSend())

    assert len(calls) == 1
    assert calls[0].response.body == b""


async def test_no_body_needed_means_no_buffering_and_no_copy(monkeypatch) -> None:
    """Proves the no-body path forwards chunks without any extra copy."""
    chunks = [b"x" * 1_000_000 for _ in range(4)]
    created = spy_on_states(monkeypatch)

    async def app(scope, receive, send) -> None:
        await send(response_start())
        for chunk in chunks:
            await send(body_message(chunk, more_body=True))
        await send(body_message(b""))

    middleware = PrometheusInstrumentatorMiddleware(
        app, plan=InstrumentationPlan(instrumentations=[lambda info: None])
    )

    send = RecordingSend()
    await middleware(make_scope(), empty_receive, send)

    # Forwarded chunks are the identical objects: no repacking, no copying.
    forwarded = [m for m in send.messages if m["type"] == "http.response.body"]
    assert all(forwarded[i]["body"] is chunks[i] for i in range(len(chunks)))

    # The execution state never buffered anything.
    assert len(created) == 1
    state = created[0]
    assert state.collects_response_body is False
    assert state.body_chunks == []
    assert state.body_size == 0
    assert state.body == b""


async def test_body_collected_up_to_configured_limit(monkeypatch) -> None:
    chunks = [b"a" * 1_000_000, b"b" * 1_000_000, b"c" * 1_000_000]
    created = spy_on_states(monkeypatch)
    calls: List[metrics.Info] = []

    async def app(scope, receive, send) -> None:
        await send(response_start())
        for chunk in chunks:
            await send(body_message(chunk, more_body=True))
        await send(body_message(b""))

    middleware = PrometheusInstrumentatorMiddleware(
        app,
        plan=InstrumentationPlan(
            instrumentations=[calls.append],
            body_handlers=[r".*"],
            body_max_bytes=2_500_000,
        ),
    )

    send = RecordingSend()
    await middleware(make_scope(), empty_receive, send)

    # Client still receives the full, untouched stream.
    forwarded = [m for m in send.messages if m["type"] == "http.response.body"]
    assert all(forwarded[i]["body"] is chunks[i] for i in range(len(chunks)))

    # The collected body respects the limit and is marked as truncated.
    assert len(created) == 1
    state = created[0]
    assert state.body_truncated is True
    assert state.body_size == 2_500_000
    assert len(calls) == 1
    assert calls[0].response.body == b"a" * 1_000_000 + b"b" * 1_000_000 + b"c" * 500_000


async def test_excluded_handler_never_buffers(monkeypatch) -> None:
    created = spy_on_states(monkeypatch)
    calls: List[metrics.Info] = []

    async def app(scope, receive, send) -> None:
        await send(response_start())
        await send(body_message(b"payload"))

    middleware = PrometheusInstrumentatorMiddleware(
        app,
        plan=InstrumentationPlan(
            instrumentations=[calls.append],
            excluded_handlers=[r"^/excluded$"],
            body_handlers=[r".*"],
        ),
    )

    await middleware(make_scope(path="/excluded"), empty_receive, RecordingSend())

    assert calls == []
    assert len(created) == 1
    assert created[0].collects_response_body is False
    assert created[0].body_chunks == []


# ------------------------------------------------------------------------------
# Plan sharing


async def test_plan_shared_between_apps_and_requests() -> None:
    """One immutable plan serves two apps concurrently without mixing state."""
    calls: List[Tuple[str, str]] = []

    def instrumentation(info: metrics.Info) -> None:
        calls.append((info.method, info.modified_handler))

    plan = InstrumentationPlan(
        instrumentations=[instrumentation], should_group_untemplated=False
    )

    gate_a = asyncio.Event()
    gate_b = asyncio.Event()

    async def app_a(scope, receive, send) -> None:
        await send(response_start())
        gate_a.set()
        await gate_b.wait()
        await send(body_message(b"a"))

    async def app_b(scope, receive, send) -> None:
        await send(response_start())
        gate_b.set()
        await gate_a.wait()
        await send(body_message(b"b"))

    middleware_a = PrometheusInstrumentatorMiddleware(app_a, plan=plan)
    middleware_b = PrometheusInstrumentatorMiddleware(app_b, plan=plan)

    await asyncio.gather(
        middleware_a(make_scope(path="/a", method="GET"), empty_receive, RecordingSend()),
        middleware_b(
            make_scope(path="/b", method="POST"), empty_receive, RecordingSend()
        ),
    )

    assert sorted(calls) == [("GET", "/a"), ("POST", "/b")]

    # The plan itself never stored any per-request data.
    assert not hasattr(plan, "request")
    assert isinstance(plan.instrumentations, tuple)


async def test_middleware_kwargs_still_supported() -> None:
    """The middleware keeps accepting the plain keyword configuration."""
    calls: List[metrics.Info] = []

    async def app(scope, receive, send) -> None:
        await send(response_start())
        await send(body_message(b"hello"))

    middleware = PrometheusInstrumentatorMiddleware(
        app,
        instrumentations=[calls.append],
        body_handlers=[r".*"],
        should_group_status_codes=True,
    )

    await middleware(make_scope(), empty_receive, RecordingSend())

    assert len(calls) == 1
    assert calls[0].modified_status == "2xx"
    assert calls[0].response.body == b"hello"
    assert isinstance(middleware.plan, InstrumentationPlan)
