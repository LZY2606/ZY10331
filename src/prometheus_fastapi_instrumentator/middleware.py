from __future__ import annotations

import asyncio
from http import HTTPStatus
from timeit import default_timer
from typing import (
    Awaitable,
    Callable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from prometheus_client import REGISTRY, CollectorRegistry, Gauge
from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from prometheus_fastapi_instrumentator import metrics
from prometheus_fastapi_instrumentator.plan import InstrumentationPlan


class RequestExecutionState:
    """Mutable state of a single request, advanced through the ASGI lifecycle.

    Created by the middleware for every incoming HTTP request. Progresses
    through the phases `start` -> `response-start` -> `body-chunk` ->
    `finish`, with `error` marking an exceptional termination. All data that
    only lives for the duration of one request is kept here so that the
    `InstrumentationPlan` can stay immutable and shareable.
    """

    def __init__(self, plan: InstrumentationPlan, scope: Scope) -> None:
        self.plan = plan
        self.request = Request(scope)
        self.start_time = default_timer()
        self.phase = "start"

        handler, is_templated = plan.get_handler(self.request)
        self.is_excluded = plan.is_handler_excluded(handler, is_templated)
        self.handler = (
            "none" if not is_templated and plan.should_group_untemplated else handler
        )

        # The body is only buffered if a metric actually needs it. Excluded
        # handlers never run instrumentations, so they never need the body.
        self.collects_response_body = (
            not self.is_excluded and plan.should_collect_response_body(self.handler)
        )

        self.status_code: Union[int, HTTPStatus] = 500
        self.headers: list = []
        self.response_start_time: Optional[float] = None
        self.body_chunks: List[bytes] = []
        self.body_size = 0
        self.body_truncated = False

        self._inprogress_labels: Optional[Tuple[str, str]] = None
        if not self.is_excluded and plan.inprogress is not None:
            if plan.inprogress_labels:
                self._inprogress_labels = (self.request.method, self.handler)
                plan.inprogress.labels(*self._inprogress_labels).inc()
            else:
                plan.inprogress.inc()

    def on_response_start(self, message: Message) -> None:
        self.phase = "response-start"
        self.headers = message["headers"]
        self.status_code = message["status"]
        self.response_start_time = default_timer()

    def on_body_chunk(self, chunk: bytes) -> None:
        if not chunk or not self.collects_response_body:
            return

        self.phase = "body-chunk"

        limit = self.plan.body_max_bytes
        if limit is not None:
            remaining = limit - self.body_size
            if remaining <= 0:
                self.body_truncated = True
                return
            if len(chunk) > remaining:
                chunk = chunk[:remaining]
                self.body_truncated = True

        self.body_chunks.append(chunk)
        self.body_size += len(chunk)

    def on_message(self, message: Message) -> None:
        if message["type"] == "http.response.start":
            self.on_response_start(message)
        elif message["type"] == "http.response.body":
            self.on_body_chunk(message["body"])

    def mark_error(self) -> None:
        self.phase = "error"

    @property
    def body(self) -> bytes:
        return b"".join(self.body_chunks)

    async def finish(self) -> None:
        """Records durations and runs the planned metric callbacks.

        Only called once the wrapped app has completed or raised, so the
        duration is never recorded before the response has been fully sent
        (including background tasks and streaming bodies).
        """
        self.phase = "finish"

        if self.is_excluded:
            return

        plan = self.plan

        status = (
            str(self.status_code.value)
            if isinstance(self.status_code, HTTPStatus)
            else str(self.status_code)
        )

        duration = max(default_timer() - self.start_time, 0.0)
        duration_without_streaming = 0.0

        if self.response_start_time:
            duration_without_streaming = max(
                self.response_start_time - self.start_time, 0.0
            )

        if plan.inprogress is not None:
            if self._inprogress_labels is not None:
                plan.inprogress.labels(*self._inprogress_labels).dec()
            else:
                plan.inprogress.dec()

        if plan.should_round_latency_decimals:
            duration = round(duration, plan.round_latency_decimals)
            duration_without_streaming = round(
                duration_without_streaming, plan.round_latency_decimals
            )

        if plan.should_group_status_codes:
            status = status[0] + "xx"

        response = Response(
            content=self.body,
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
        )

        for instrumentation in plan.instrumentations:
            instrumentation(info)

        await asyncio.gather(
            *[instrumentation(info) for instrumentation in plan.async_instrumentations]
        )


class PrometheusInstrumentatorMiddleware:
    """ASGI middleware that records Prometheus metrics for HTTP requests.

    The middleware itself only translates the constructor configuration into
    an immutable `InstrumentationPlan` (unless one is passed in directly) and
    creates a `RequestExecutionState` per request that is advanced through
    the ASGI lifecycle.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        should_group_status_codes: bool = True,
        should_ignore_untemplated: bool = False,
        should_group_untemplated: bool = True,
        should_include_root_path: bool = False,
        should_round_latency_decimals: bool = False,
        should_respect_env_var: bool = False,
        should_instrument_requests_inprogress: bool = False,
        should_exclude_streaming_duration: bool = False,
        excluded_handlers: Sequence[str] = (),
        body_handlers: Sequence[str] = (),
        round_latency_decimals: int = 4,
        env_var_name: str = "ENABLE_METRICS",
        inprogress_name: str = "http_requests_inprogress",
        inprogress_labels: bool = False,
        instrumentations: Sequence[Callable[[metrics.Info], None]] = (),
        async_instrumentations: Sequence[Callable[[metrics.Info], Awaitable[None]]] = (),
        metric_namespace: str = "",
        metric_subsystem: str = "",
        should_only_respect_2xx_for_highr: bool = False,
        latency_highr_buckets: Sequence[Union[float, str]] = (
            0.01,
            0.025,
            0.05,
            0.075,
            0.1,
            0.25,
            0.5,
            0.75,
            1,
            1.5,
            2,
            2.5,
            3,
            3.5,
            4,
            4.5,
            5,
            7.5,
            10,
            30,
            60,
        ),
        latency_lowr_buckets: Sequence[Union[float, str]] = (0.1, 0.5, 1),
        registry: CollectorRegistry = REGISTRY,
        custom_labels: dict = {},
        body_max_bytes: Optional[int] = None,
        plan: Optional[InstrumentationPlan] = None,
    ) -> None:
        self.app = app

        if plan is None:
            if not instrumentations:
                default_instrumentation = metrics.default(
                    metric_namespace=metric_namespace,
                    metric_subsystem=metric_subsystem,
                    should_only_respect_2xx_for_highr=should_only_respect_2xx_for_highr,
                    should_exclude_streaming_duration=should_exclude_streaming_duration,
                    latency_highr_buckets=latency_highr_buckets,
                    latency_lowr_buckets=latency_lowr_buckets,
                    registry=registry,
                    custom_labels=custom_labels,
                )
                if default_instrumentation:
                    instrumentations = [default_instrumentation]
                else:
                    instrumentations = []

            plan = InstrumentationPlan(
                instrumentations=instrumentations,
                async_instrumentations=async_instrumentations,
                should_group_status_codes=should_group_status_codes,
                should_ignore_untemplated=should_ignore_untemplated,
                should_group_untemplated=should_group_untemplated,
                should_include_root_path=should_include_root_path,
                should_round_latency_decimals=should_round_latency_decimals,
                round_latency_decimals=round_latency_decimals,
                excluded_handlers=excluded_handlers,
                body_handlers=body_handlers,
                body_max_bytes=body_max_bytes,
                should_instrument_requests_inprogress=should_instrument_requests_inprogress,
                inprogress_name=inprogress_name,
                inprogress_labels=inprogress_labels,
                registry=registry,
            )

        self.plan = plan

        # Attribute views kept for backwards compatibility.
        self.should_group_status_codes = plan.should_group_status_codes
        self.should_ignore_untemplated = plan.should_ignore_untemplated
        self.should_group_untemplated = plan.should_group_untemplated
        self.should_include_root_path = plan.should_include_root_path
        self.should_round_latency_decimals = plan.should_round_latency_decimals
        self.should_instrument_requests_inprogress = plan.inprogress is not None

        self.round_latency_decimals = plan.round_latency_decimals
        self.env_var_name = env_var_name
        self.inprogress_name = inprogress_name
        self.inprogress_labels = plan.inprogress_labels
        self.registry = plan.registry
        self.custom_labels = custom_labels
        self.should_respect_env_var = should_respect_env_var

        self.excluded_handlers = plan.excluded_handlers
        self.body_handlers = plan.body_handlers

        self.instrumentations = plan.instrumentations
        self.async_instrumentations = plan.async_instrumentations

        self.inprogress: Optional[Gauge] = plan.inprogress

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        state = RequestExecutionState(self.plan, scope)

        async def send_wrapper(message: Message) -> None:
            state.on_message(message)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except BaseException:
            state.mark_error()
            raise
        finally:
            await state.finish()

    def _get_handler(self, request: Request) -> Tuple[str, bool]:
        """Extracts either template or (if no template) path.

        Args:
            request (Request): Python Requests request object.

        Returns:
            Tuple[str, bool]: Tuple with two elements. First element is either
                template or if no template the path. Second element tells you
                if the path is templated or not. When configured, the template
                includes the application's effective `root_path`.
        """
        return self.plan.get_handler(request)

    def _is_handler_excluded(self, handler: str, is_templated: bool) -> bool:
        """Determines if the handler should be ignored.

        Args:
            handler (str): Handler that handles the request.
            is_templated (bool): Shows if the request is templated.

        Returns:
            bool: `True` if excluded, `False` if not.
        """
        return self.plan.is_handler_excluded(handler, is_templated)
