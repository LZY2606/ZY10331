"""Tests for the install-time plan / per-request execution state split.

Uses a recording ASGI app to lock down the order of ASGI events and metric
sample writes, and covers normal, streaming, exception, cancel, empty body
and shared-config scenarios.
"""

import asyncio
import dataclasses
from typing import List, Optional

import pytest
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import REGISTRY, CollectorRegistry
from starlette.background import BackgroundTask
from starlette.testclient import TestClient

from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_fastapi_instrumentator.middleware import (
    PrometheusInstrumentatorMiddleware,
)
from prometheus_fastapi_instrumentator.plan import (
    InstrumentationPlan,
    Phase,
    RequestExecution,
)

from helpers import utils


def _scope(path: str = "/", method: str = "GET") -> dict:
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


async def _receive() -> dict:
    return {"type": "http.request", "body": b"", "more_body": False}


class RecordingApp:
    """Raw ASGI app that records its own lifecycle events.

    `events` is shared with the test so ASGI events, app progress and
    instrumentation calls end up in one ordered list.
    """

    def __init__(
        self,
        events: list,
        chunks: tuple = (b"hello", b" ", b"world"),
        status: int = 200,
        exc: Optional[BaseException] = None,
        fail_before_start: bool = False,
        chunk_hook=None,
    ) -> None:
        self.events = events
        self.chunks = chunks
        self.status = status
        self.exc = exc
        self.fail_before_start = fail_before_start
        self.chunk_hook = chunk_hook

    async def __call__(self, scope, receive, send) -> None:
        self.events.append("app:start")
        if self.exc is not None and self.fail_before_start:
            self.events.append("app:raise")
            raise self.exc
        await send(
            {
                "type": "http.response.start",
                "status": self.status,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        self.events.append("app:response-start")
        for chunk in self.chunks:
            await send(
                {"type": "http.response.body", "body": chunk, "more_body": True}
            )
            self.events.append(f"app:body-chunk:{chunk!r}")
            if self.chunk_hook:
                self.chunk_hook()
            await asyncio.sleep(0)
        await send({"type": "http.response.body", "body": b"", "more_body": False})
        self.events.append("app:body-end")
        if self.exc is not None:
            self.events.append("app:raise")
            raise self.exc
        self.events.append("app:return")


def _make_send(events: list, fail_on_chunk: Optional[int] = None):
    """Returns a send callable that records events and can simulate cancels."""

    state = {"chunks": 0}
    sent_messages: List[dict] = []

    async def send(message: dict) -> None:
        sent_messages.append(message)
        if message["type"] == "http.response.body" and message.get("body"):
            state["chunks"] += 1
            if fail_on_chunk is not None and state["chunks"] == fail_on_chunk:
                events.append("send:client-cancel")
                raise asyncio.CancelledError()
        events.append(f"send:{message['type']}")

    send.sent_messages = sent_messages
    return send


def _recording_instrumentation(events: list, store: Optional[list] = None):
    def instrumentation(info: metrics.Info) -> None:
        events.append("instrumentation:sync")
        if store is not None:
            store.append(info)

    return instrumentation


# ------------------------------------------------------------------------------
# Plan immutability and per-request state


def test_plan_is_immutable_and_holds_no_request_state():
    middleware = PrometheusInstrumentatorMiddleware(RecordingApp([]))

    assert isinstance(middleware.plan, InstrumentationPlan)
    with pytest.raises(dataclasses.FrozenInstanceError):
        middleware.plan.should_group_status_codes = False

    plan_dict = vars(middleware.plan)
    assert "request" not in plan_dict
    assert not any(isinstance(value, RequestExecution) for value in plan_dict.values())


async def test_each_request_gets_independent_execution_state():
    events: list = []
    infos: list = []
    app = RecordingApp(events, chunks=(b"aaa", b"bb"))
    middleware = PrometheusInstrumentatorMiddleware(
        app,
        body_handlers=[r".*"],
        instrumentations=[_recording_instrumentation(events, infos)],
    )

    await middleware(_scope(), _receive, _make_send(events))
    await middleware(_scope(), _receive, _make_send(events))

    assert len(infos) == 2
    assert infos[0].response.body == b"aaabb"
    assert infos[1].response.body == b"aaabb"
    assert infos[0].response is not infos[1].response


async def test_concurrent_requests_share_plan_but_not_state():
    infos: list = []

    async def app(scope, receive, send) -> None:
        path = scope["path"].encode()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        for _ in range(3):
            await asyncio.sleep(0)
            await send(
                {"type": "http.response.body", "body": path, "more_body": True}
            )
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    middleware = PrometheusInstrumentatorMiddleware(
        app,
        body_handlers=[r".*"],
        instrumentations=[_recording_instrumentation([], infos)],
    )

    paths = [f"/item/{num}" for num in range(10)]
    await asyncio.gather(
        *[
            middleware(_scope(path=path), _receive, _make_send([]))
            for path in paths
        ]
    )

    bodies = sorted(info.response.body for info in infos)
    assert bodies == sorted(path.encode() * 3 for path in paths)


# ------------------------------------------------------------------------------
# Event and sample ordering


async def test_event_order_normal_response():
    events: list = []
    app = RecordingApp(events)
    middleware = PrometheusInstrumentatorMiddleware(
        app, instrumentations=[_recording_instrumentation(events)]
    )

    await middleware(_scope(), _receive, _make_send(events))

    assert events == [
        "app:start",
        "send:http.response.start",
        "app:response-start",
        "send:http.response.body",
        "app:body-chunk:b'hello'",
        "send:http.response.body",
        "app:body-chunk:b' '",
        "send:http.response.body",
        "app:body-chunk:b'world'",
        "send:http.response.body",
        "app:body-end",
        "app:return",
        "instrumentation:sync",
    ]


async def test_samples_written_only_after_response_complete():
    registry = CollectorRegistry()
    requests_metric = metrics.requests(registry=registry)
    # No app in scope, so the untemplated handler is grouped to `none`.
    labels = {"handler": "none", "method": "GET", "status": "2xx"}

    events: list = []

    def assert_no_sample_yet() -> None:
        assert registry.get_sample_value("http_requests_total", labels) is None

    app = RecordingApp(events, chunk_hook=assert_no_sample_yet)
    middleware = PrometheusInstrumentatorMiddleware(
        app, instrumentations=[requests_metric]
    )

    await middleware(_scope(), _receive, _make_send(events))

    assert registry.get_sample_value("http_requests_total", labels) == 1.0


async def test_async_instrumentation_runs_after_sync():
    events: list = []

    async def async_instrumentation(info: metrics.Info) -> None:
        events.append("instrumentation:async")

    app = RecordingApp(events, chunks=(b"x",))
    middleware = PrometheusInstrumentatorMiddleware(
        app,
        instrumentations=[_recording_instrumentation(events)],
        async_instrumentations=[async_instrumentation],
    )

    await middleware(_scope(), _receive, _make_send(events))

    assert events[-2:] == ["instrumentation:sync", "instrumentation:async"]


# ------------------------------------------------------------------------------
# Streaming, duration and background tasks


async def test_streaming_duration_recorded_only_at_finish():
    events: list = []
    infos: list = []
    chunks = tuple(f"chunk{num}".encode() for num in range(5))
    app = RecordingApp(events, chunks=chunks)
    middleware = PrometheusInstrumentatorMiddleware(
        app, instrumentations=[_recording_instrumentation(events, infos)]
    )

    await middleware(_scope(), _receive, _make_send(events))

    # Instrumentation must run after the final body chunk, never early.
    assert events.index("instrumentation:sync") > events.index("app:body-end")
    assert len(infos) == 1
    info = infos[0]
    assert info.modified_duration >= info.modified_duration_without_streaming
    assert info.modified_duration_without_streaming >= 0.0


def test_background_task_completes_before_duration_recorded():
    events: list = []
    app = FastAPI()

    @app.get("/")
    def root():
        return PlainTextResponse(
            "ok", background=BackgroundTask(events.append, "background-task")
        )

    def instrumentation(info: metrics.Info) -> None:
        events.append("instrumentation")

    Instrumentator().add(instrumentation).instrument(app)

    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert events == ["background-task", "instrumentation"]


# ------------------------------------------------------------------------------
# Exceptions and client cancels


async def test_exception_still_records_and_propagates():
    events: list = []
    infos: list = []
    app = RecordingApp(events, exc=RuntimeError("boom"), fail_before_start=True)
    middleware = PrometheusInstrumentatorMiddleware(
        app, instrumentations=[_recording_instrumentation(events, infos)]
    )

    with pytest.raises(RuntimeError):
        await middleware(_scope(), _receive, _make_send(events))

    assert events == ["app:start", "app:raise", "instrumentation:sync"]
    assert len(infos) == 1
    assert infos[0].modified_status == "5xx"


async def test_exception_after_response_start_keeps_sent_status():
    events: list = []
    infos: list = []
    app = RecordingApp(events, chunks=(), exc=RuntimeError("boom"))
    middleware = PrometheusInstrumentatorMiddleware(
        app, instrumentations=[_recording_instrumentation(events, infos)]
    )

    with pytest.raises(RuntimeError):
        await middleware(_scope(), _receive, _make_send(events))

    assert events[-1] == "instrumentation:sync"
    assert len(infos) == 1
    assert infos[0].modified_status == "2xx"


async def test_client_cancel_records_once_and_not_early():
    events: list = []
    infos: list = []
    app = RecordingApp(events, chunks=(b"aaa", b"bbb", b"ccc"))
    middleware = PrometheusInstrumentatorMiddleware(
        app, instrumentations=[_recording_instrumentation(events, infos)]
    )

    with pytest.raises(asyncio.CancelledError):
        await middleware(_scope(), _receive, _make_send(events, fail_on_chunk=2))

    # Cancelled while streaming the second chunk. Metrics are recorded exactly
    # once, after the cancel, never before it.
    assert events.count("instrumentation:sync") == 1
    assert events.index("instrumentation:sync") > events.index("send:client-cancel")
    assert len(infos) == 1


def test_metric_callback_failure_propagates():
    app = FastAPI()

    @app.get("/")
    def root():
        return "ok"

    def failing_instrumentation(info: metrics.Info) -> None:
        raise ValueError("metric callback failed")

    Instrumentator().add(failing_instrumentation).instrument(app)

    with pytest.raises(ValueError, match="metric callback failed"):
        TestClient(app).get("/")


# ------------------------------------------------------------------------------
# Body collection, limits and truncation


async def test_empty_body_with_body_handlers():
    events: list = []
    infos: list = []
    app = RecordingApp(events, chunks=())
    middleware = PrometheusInstrumentatorMiddleware(
        app,
        body_handlers=[r".*"],
        instrumentations=[_recording_instrumentation(events, infos)],
    )

    await middleware(_scope(), _receive, _make_send(events))

    assert len(infos) == 1
    assert infos[0].response.body == b""
    assert infos[0].response_body_truncated is False


async def test_no_body_metric_does_not_buffer_or_copy():
    """Proves the no-body path passes chunks through without copying.

    The chunks sent to the client must be the exact same objects the app
    produced. Any whole-body copy or concatenation by the middleware would
    break identity.
    """
    events: list = []
    infos: list = []
    chunks = tuple(bytes([num]) * 1_000_000 for num in range(5))
    app = RecordingApp(events, chunks=chunks)
    middleware = PrometheusInstrumentatorMiddleware(
        app, instrumentations=[_recording_instrumentation(events, infos)]
    )

    send = _make_send(events)
    await middleware(_scope(), _receive, send)

    sent_chunks = [
        message["body"]
        for message in send.sent_messages
        if message["type"] == "http.response.body" and message.get("body")
    ]
    assert len(sent_chunks) == len(chunks)
    for sent_chunk, original_chunk in zip(sent_chunks, chunks):
        assert sent_chunk is original_chunk

    assert len(infos) == 1
    assert infos[0].response.body == b""
    assert infos[0].response_body_truncated is False


async def test_body_max_size_truncates_and_marks():
    events: list = []
    infos: list = []
    app = RecordingApp(events, chunks=(b"abc", b"def", b"ghi"))
    middleware = PrometheusInstrumentatorMiddleware(
        app,
        body_handlers=[r".*"],
        body_max_size=4,
        instrumentations=[_recording_instrumentation(events, infos)],
    )

    send = _make_send(events)
    await middleware(_scope(), _receive, send)

    assert len(infos) == 1
    assert infos[0].response.body == b"abcd"
    assert infos[0].response_body_truncated is True

    # Client still receives the full, untruncated body.
    sent_body = b"".join(
        message.get("body", b"")
        for message in send.sent_messages
        if message["type"] == "http.response.body"
    )
    assert sent_body == b"abcdefghi"


async def test_body_max_size_exact_fit_is_not_truncated():
    infos: list = []
    app = RecordingApp([], chunks=(b"abc", b"def"))
    middleware = PrometheusInstrumentatorMiddleware(
        app,
        body_handlers=[r".*"],
        body_max_size=6,
        instrumentations=[_recording_instrumentation([], infos)],
    )

    await middleware(_scope(), _receive, _make_send([]))

    assert infos[0].response.body == b"abcdef"
    assert infos[0].response_body_truncated is False


def test_body_max_size_via_instrumentator_chain():
    infos: list = []
    app = FastAPI()

    @app.get("/", response_class=PlainTextResponse)
    def root():
        return "0123456789"

    def instrumentation(info: metrics.Info) -> None:
        infos.append(info)

    (
        Instrumentator(body_handlers=[r".*"], body_max_size=5)
        .add(instrumentation)
        .instrument(app)
    )

    response = TestClient(app).get("/")

    assert response.content == b"0123456789"
    assert len(infos) == 1
    assert infos[0].response.body == b"01234"
    assert infos[0].response_body_truncated is True


# ------------------------------------------------------------------------------
# Shared configuration across apps


def test_two_apps_share_single_instrumentator_config():
    utils.reset_collectors()

    app1 = FastAPI()
    app2 = FastAPI()

    @app1.get("/")
    def root1():
        return "app1"

    @app2.get("/")
    def root2():
        return "app2"

    instrumentator = Instrumentator().add(metrics.requests())
    instrumentator.instrument(app1)
    instrumentator.instrument(app2)

    assert TestClient(app1).get("/").status_code == 200
    assert TestClient(app2).get("/").status_code == 200

    assert (
        REGISTRY.get_sample_value(
            "http_requests_total",
            {"handler": "/", "method": "GET", "status": "2xx"},
        )
        == 2.0
    )


def test_two_apps_shared_plan_keeps_per_request_state_separate():
    utils.reset_collectors()
    infos: list = []

    app1 = FastAPI()
    app2 = FastAPI()

    @app1.get("/", response_class=PlainTextResponse)
    def root1():
        return "first"

    @app2.get("/", response_class=PlainTextResponse)
    def root2():
        return "second-app"

    def instrumentation(info: metrics.Info) -> None:
        infos.append(info)

    (
        Instrumentator(body_handlers=[r".*"])
        .add(instrumentation)
        .instrument(app1)
        .instrument(app2)
    )

    TestClient(app1).get("/")
    TestClient(app2).get("/")

    assert [info.response.body for info in infos] == [b"first", b"second-app"]
