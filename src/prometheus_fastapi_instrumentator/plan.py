from __future__ import annotations

import re
from typing import (
    Awaitable,
    Callable,
    Optional,
    Pattern,
    Sequence,
    Tuple,
    Union,
)

from prometheus_client import REGISTRY, CollectorRegistry, Gauge
from starlette.requests import Request

from prometheus_fastapi_instrumentator import metrics, routing


class InstrumentationPlan:
    """Immutable install-time description of how requests are instrumented.

    The plan is assembled once when the middleware stack is built. It fixes
    the metric callbacks, the request/response body requirements, the label
    options, the exclusion matcher and the registry. A plan can be shared by
    any number of concurrent requests and even by multiple middleware
    instances because it never stores per-request data. All mutable state
    lives in `RequestExecutionState` (see `middleware.py`).
    """

    __slots__ = (
        "instrumentations",
        "async_instrumentations",
        "should_group_status_codes",
        "should_ignore_untemplated",
        "should_group_untemplated",
        "should_include_root_path",
        "should_round_latency_decimals",
        "round_latency_decimals",
        "excluded_handlers",
        "body_handlers",
        "needs_request_body",
        "needs_response_body",
        "body_max_bytes",
        "registry",
        "inprogress",
        "inprogress_labels",
    )

    def __init__(
        self,
        *,
        instrumentations: Sequence[Callable[[metrics.Info], None]] = (),
        async_instrumentations: Sequence[Callable[[metrics.Info], Awaitable[None]]] = (),
        should_group_status_codes: bool = True,
        should_ignore_untemplated: bool = False,
        should_group_untemplated: bool = True,
        should_include_root_path: bool = False,
        should_round_latency_decimals: bool = False,
        round_latency_decimals: int = 4,
        excluded_handlers: Sequence[Union[str, Pattern[str]]] = (),
        body_handlers: Sequence[Union[str, Pattern[str]]] = (),
        needs_request_body: bool = False,
        body_max_bytes: Optional[int] = None,
        should_instrument_requests_inprogress: bool = False,
        inprogress_name: str = "http_requests_inprogress",
        inprogress_labels: bool = False,
        registry: CollectorRegistry = REGISTRY,
    ) -> None:
        self.instrumentations = tuple(instrumentations)
        self.async_instrumentations = tuple(async_instrumentations)

        self.should_group_status_codes = should_group_status_codes
        self.should_ignore_untemplated = should_ignore_untemplated
        self.should_group_untemplated = should_group_untemplated
        self.should_include_root_path = should_include_root_path
        self.should_round_latency_decimals = should_round_latency_decimals
        self.round_latency_decimals = round_latency_decimals

        self.excluded_handlers = tuple(
            re.compile(pattern) for pattern in excluded_handlers
        )
        self.body_handlers = tuple(re.compile(pattern) for pattern in body_handlers)

        # Body requirements. No bundled metric consumes the request body, so
        # it defaults to not needed. The response body is only needed if at
        # least one handler pattern was configured to pass it through.
        self.needs_request_body = needs_request_body
        self.needs_response_body = bool(self.body_handlers)
        self.body_max_bytes = body_max_bytes

        self.registry = registry
        self.inprogress_labels = inprogress_labels

        self.inprogress: Optional[Gauge] = None
        if should_instrument_requests_inprogress:
            labels = ("method", "handler") if inprogress_labels else ()
            self.inprogress = Gauge(
                name=inprogress_name,
                documentation="Number of HTTP requests in progress.",
                labelnames=labels,
                multiprocess_mode="livesum",
            )

    def get_handler(self, request: Request) -> Tuple[str, bool]:
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

    def is_handler_excluded(self, handler: str, is_templated: bool) -> bool:
        """Determines if the handler should be ignored."""
        if not is_templated and self.should_ignore_untemplated:
            return True

        if any(pattern.search(handler) for pattern in self.excluded_handlers):
            return True

        return False

    def should_collect_response_body(self, handler: str) -> bool:
        """Determines if the response body must be buffered for this handler.

        Returns `False` whenever no metric needs the body, in which case the
        middleware must not buffer anything.
        """
        return self.needs_response_body and any(
            pattern.search(handler) for pattern in self.body_handlers
        )
