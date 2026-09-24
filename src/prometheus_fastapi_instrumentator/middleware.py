from __future__ import annotations

import re
from typing import Awaitable, Callable, Optional, Sequence, Tuple, Union

from prometheus_client import REGISTRY, CollectorRegistry, Gauge
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from prometheus_fastapi_instrumentator import metrics
from prometheus_fastapi_instrumentator.plan import (
    ExclusionMatcher,
    InstrumentationPlan,
    RequestExecution,
)


class PrometheusInstrumentatorMiddleware:
    """ASGI middleware that records Prometheus metrics for HTTP requests.

    Installation builds a single immutable `InstrumentationPlan` that is
    reused by all requests. Each request gets its own `RequestExecution`
    state machine that is advanced through the phases `start`,
    `response-start`, `body-chunk`, `finish` and `error`.
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
        body_max_size: Optional[int] = None,
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
    ) -> None:
        self.app = app

        self.should_group_status_codes = should_group_status_codes
        self.should_ignore_untemplated = should_ignore_untemplated
        self.should_group_untemplated = should_group_untemplated
        self.should_include_root_path = should_include_root_path
        self.should_round_latency_decimals = should_round_latency_decimals
        self.should_respect_env_var = should_respect_env_var
        self.should_instrument_requests_inprogress = should_instrument_requests_inprogress

        self.round_latency_decimals = round_latency_decimals
        self.env_var_name = env_var_name
        self.inprogress_name = inprogress_name
        self.inprogress_labels = inprogress_labels
        self.registry = registry
        self.custom_labels = custom_labels

        self.excluded_handlers = [re.compile(path) for path in excluded_handlers]
        self.body_handlers = [re.compile(path) for path in body_handlers]
        self.body_max_size = body_max_size

        if instrumentations:
            self.instrumentations = instrumentations
        else:
            default_instrumentation = metrics.default(
                metric_namespace=metric_namespace,
                metric_subsystem=metric_subsystem,
                should_only_respect_2xx_for_highr=should_only_respect_2xx_for_highr,
                should_exclude_streaming_duration=should_exclude_streaming_duration,
                latency_highr_buckets=latency_highr_buckets,
                latency_lowr_buckets=latency_lowr_buckets,
                registry=self.registry,
                custom_labels=custom_labels,
            )
            if default_instrumentation:
                self.instrumentations = [default_instrumentation]
            else:
                self.instrumentations = []

        self.async_instrumentations = async_instrumentations

        self.inprogress: Optional[Gauge] = None
        if self.should_instrument_requests_inprogress:
            labels = (
                (
                    "method",
                    "handler",
                )
                if self.inprogress_labels
                else ()
            )
            self.inprogress = Gauge(
                name=self.inprogress_name,
                documentation="Number of HTTP requests in progress.",
                labelnames=labels,
                multiprocess_mode="livesum",
            )

        # Immutable install-time plan. Shared by all requests of this
        # middleware instance, safe for concurrent reuse, and never mutated
        # by per-request execution state.
        self.plan = InstrumentationPlan(
            instrumentations=tuple(self.instrumentations),
            async_instrumentations=tuple(self.async_instrumentations),
            exclusion=ExclusionMatcher(
                should_ignore_untemplated=self.should_ignore_untemplated,
                patterns=tuple(self.excluded_handlers),
            ),
            body_handlers=tuple(self.body_handlers),
            body_max_size=self.body_max_size,
            should_group_status_codes=self.should_group_status_codes,
            should_group_untemplated=self.should_group_untemplated,
            should_include_root_path=self.should_include_root_path,
            should_round_latency_decimals=self.should_round_latency_decimals,
            round_latency_decimals=self.round_latency_decimals,
            inprogress=self.inprogress,
            inprogress_labels=self.inprogress_labels,
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        execution = RequestExecution(plan=self.plan, request=Request(scope))

        async def send_wrapper(message: Message) -> None:
            execution.observe(message)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            execution.error()
            raise exc
        finally:
            await execution.finish()

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
        return self.plan.resolve_handler(request)

    def _is_handler_excluded(self, handler: str, is_templated: bool) -> bool:
        """Determines if the handler should be ignored.

        Args:
            handler (str): Handler that handles the request.
            is_templated (bool): Shows if the request is templated.

        Returns:
            bool: `True` if excluded, `False` if not.
        """
        return self.plan.exclusion.is_excluded(handler, is_templated)
