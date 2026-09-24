"""Install-time instrumentation plan and per-request execution state.

This module splits the middleware concerns in two:

* `InstrumentationPlan` is built once when the middleware is installed. It is
  an immutable (frozen) value object that captures the metric callbacks, the
  request/response body requirements, the label options, the exclusion
  matcher and the in-progress gauge. A plan can be reused concurrently by any
  number of requests because it never stores per-request data.

* `RequestExecution` is created per request by the middleware. It is the
  only mutable object and advances through the phases `start`,
  `response-start`, `body-chunk`, `finish` and `error` as the ASGI
  lifecycle progresses.
"""

from __future__ import annotations

import asyncio
import enum
from dataclasses import dataclass
from http import HTTPStatus
from timeit import default_timer
from typing import (
    Awaitable,
    Callable,
    List,
    Optional,
    Pattern,
    Tuple,
    Union,
)

from prometheus_client import Gauge
from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import Message

from prometheus_fastapi_instrumentator import metrics, routing


class Phase(enum.Enum):
    """Lifecycle phases of a single request execution."""

    START = "start"
    RESPONSE_START = "response-start"
    BODY_CHUNK = "body-chunk"
    FINISH = "finish"
    ERROR = "error"


@dataclass(frozen=True)
class ExclusionMatcher:
    """Decides if a handler should be excluded from instrumentation."""

    should_ignore_untemplated: bool = False
    patterns: Tuple[Pattern[str], ...] = ()

    def is_excluded(self, handler: str, is_templated: bool) -> bool:
        if not is_templated and self.should_ignore_untemplated:
            return True

        if any(pattern.search(handler) for pattern in self.patterns):
            return True

        return False


@dataclass(frozen=True)
class BodyRequirements:
    """Which bodies the metric callbacks need for a given handler.

    `max_size` caps the buffered response body. `None` means unlimited.
    """

    request: bool = False
    response: bool = False
    max_size: Optional[int] = None


@dataclass(frozen=True)
class InstrumentationPlan:
    """Immutable install-time configuration shared by all requests."""

    instrumentations: Tuple[Callable[[metrics.Info], None], ...] = ()
    async_instrumentations: Tuple[Callable[[metrics.Info], Awaitable[None]], ...] = ()
    exclusion: ExclusionMatcher = ExclusionMatcher()
    body_handlers: Tuple[Pattern[str], ...] = ()
    body_max_size: Optional[int] = None
    should_group_status_codes: bool = True
    should_group_untemplated: bool = True
    should_include_root_path: bool = False
    should_round_latency_decimals: bool = False
    round_latency_decimals: int = 4
    inprogress: Optional[Gauge] = None
    inprogress_labels: bool = False

    def resolve_handler(self, request: Request) -> Tuple[str, bool]:
        """Extracts either template or (if no template) path.

        Returns:
            Tuple[str, bool]: Tuple with two elements. First element is either
                template or if no template the path. Second element tells you
                if the path is templated or not. When configured, the template
                includes the application's effective `root_path`.
        """
        route_name = routing.get_route_name(
            request, should_include_root_path=self.should_include_root_path
        )
        return route_name or request.url.path, True if route_name else False

    def body_requirements(self, handler: str) -> BodyRequirements:
        """Determines the bodies the plan needs for the given handler."""
        return BodyRequirements(
            request=False,
            response=any(pattern.search(handler) for pattern in self.body_handlers),
            max_size=self.body_max_size,
        )


class RequestExecution:
    """Mutable per-request state. Never shared between requests."""

    def __init__(self, plan: InstrumentationPlan, request: Request) -> None:
        self.plan = plan
        self.request = request
        self.phase = Phase.START

        self.start_time = default_timer()

        handler, is_templated = plan.resolve_handler(request)
        self.is_excluded = plan.exclusion.is_excluded(handler, is_templated)
        self.handler = (
            "none" if not is_templated and plan.should_group_untemplated else handler
        )

        self.body_requirements = plan.body_requirements(self.handler)
        # Buffer is only allocated if a metric actually needs the response
        # body. Otherwise chunks pass through without any copying.
        self.body: Optional[bytearray] = (
            bytearray() if self.body_requirements.response else None
        )
        self.body_truncated = False

        self.status_code: Union[int, HTTPStatus] = 500
        self.headers: List[Tuple[bytes, bytes]] = []
        self.response_start_time: Optional[float] = None

        self._inprogress_child: Optional[Gauge] = None
        if not self.is_excluded and plan.inprogress is not None:
            if plan.inprogress_labels:
                self._inprogress_child = plan.inprogress.labels(
                    request.method, self.handler
                )
            else:
                self._inprogress_child = plan.inprogress
            self._inprogress_child.inc()

    def observe(self, message: Message) -> None:
        """Advances the state machine based on an outgoing ASGI message."""
        if message["type"] == "http.response.start":
            self.phase = Phase.RESPONSE_START
            self.headers = message["headers"]
            self.status_code = message["status"]
            self.response_start_time = default_timer()
        elif message["type"] == "http.response.body" and message["body"]:
            self.phase = Phase.BODY_CHUNK
            self._collect_body_chunk(message["body"])

    def _collect_body_chunk(self, chunk: bytes) -> None:
        buffer = self.body
        if buffer is None:
            return

        max_size = self.body_requirements.max_size
        if max_size is None:
            buffer += chunk
        elif len(buffer) + len(chunk) <= max_size:
            buffer += chunk
        else:
            remaining = max_size - len(buffer)
            if remaining > 0:
                buffer += chunk[:remaining]
            self.body_truncated = True

    def error(self) -> None:
        """Marks the execution as failed. Recording still happens on finish."""
        self.phase = Phase.ERROR

    async def finish(self) -> None:
        """Records metrics by running the plan's instrumentation callbacks.

        Duration is only computed here, at the very end of the ASGI
        lifecycle, so streaming responses, background tasks and client
        cancels never lead to a duration being recorded early.
        """
        if self.phase is Phase.FINISH:
            return
        self.phase = Phase.FINISH

        status = (
            str(self.status_code.value)
            if isinstance(self.status_code, HTTPStatus)
            else str(self.status_code)
        )

        if self.is_excluded:
            return

        duration = max(default_timer() - self.start_time, 0.0)
        duration_without_streaming = 0.0

        if self.response_start_time:
            duration_without_streaming = max(
                self.response_start_time - self.start_time, 0.0
            )

        if self._inprogress_child is not None:
            self._inprogress_child.dec()

        if self.plan.should_round_latency_decimals:
            duration = round(duration, self.plan.round_latency_decimals)
            duration_without_streaming = round(
                duration_without_streaming, self.plan.round_latency_decimals
            )

        if self.plan.should_group_status_codes:
            status = status[0] + "xx"

        response = Response(
            content=bytes(self.body) if self.body is not None else b"",
            headers=Headers(raw=self.headers),
            status_code=self.status_code,
        )

        info = metrics.Info(
            request=self.request,
            response=response,
            method=self.request.method,
            modified_handler=self.handler,
            modified_status=status,
            modified_duration=duration,
            modified_duration_without_streaming=duration_without_streaming,
            response_body_truncated=self.body_truncated,
        )

        for instrumentation in self.plan.instrumentations:
            instrumentation(info)

        await asyncio.gather(
            *[
                instrumentation(info)
                for instrumentation in self.plan.async_instrumentations
            ]
        )
