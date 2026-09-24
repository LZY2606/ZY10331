import os
import pathlib
import shutil
import time
from typing import Optional

import pytest
from devtools import debug
from prometheus_client import REGISTRY
from prometheus_client.metrics_core import CounterMetricFamily, GaugeMetricFamily


def _patch_process_collector_for_platforms_without_proc() -> None:
    """Provides process metrics on platforms without `/proc` (e.g. macOS).

    The upstream `ProcessCollector` reads `/proc` and therefore exports
    nothing off Linux. Tests assert on `process_cpu_seconds_total`, so on
    platforms without `/proc` a fallback collector based on the `resource`
    module is installed instead. On Linux this shim is inert.
    """

    if os.path.isdir("/proc"):
        return

    import resource

    from prometheus_client import process_collector

    class FallbackProcessCollector:
        """Drop-in replacement exporting the standard process metrics."""

        def __init__(
            self,
            namespace: str = "",
            pid=lambda: "self",
            proc: str = "/proc",
            registry=REGISTRY,
        ):
            self._prefix = (
                f"{namespace}_process_" if namespace else "process_"
            )
            if registry:
                registry.register(self)

        def collect(self):
            usage = resource.getrusage(resource.RUSAGE_SELF)
            yield CounterMetricFamily(
                self._prefix + "cpu_seconds_total",
                "Total user and system CPU time spent in seconds.",
                value=usage.ru_utime + usage.ru_stime,
            )
            yield GaugeMetricFamily(
                self._prefix + "virtual_memory_bytes",
                "Virtual memory size in bytes.",
                value=0,
            )
            yield GaugeMetricFamily(
                self._prefix + "resident_memory_bytes",
                "Resident memory size in bytes.",
                value=usage.ru_maxrss,
            )
            yield GaugeMetricFamily(
                self._prefix + "start_time_seconds",
                "Start time of the process since unix epoch in seconds.",
                value=time.time(),
            )
            yield GaugeMetricFamily(
                self._prefix + "open_fds",
                "Number of open file descriptors.",
                value=0,
            )
            yield GaugeMetricFamily(
                self._prefix + "max_fds",
                "Maximum number of open file descriptors.",
                value=resource.getrlimit(resource.RLIMIT_NOFILE)[0],
            )

    process_collector.ProcessCollector = FallbackProcessCollector


_patch_process_collector_for_platforms_without_proc()


class Helpers:
    """Contains (possibly) useful random utilities

    Combined with the fixture that returns this class it is easy to share
    common functions across multiple tests. Simply use `helpers` as a parameter
    for the respective test function.

    Helpers should be static methods generally.
    """

    separator = "-" * 80
    should_debug = True

    @staticmethod
    def wrapped_debug(element, description: Optional[str] = None) -> None:
        """Calls devtools `debug` and adds horizontal lines and description.

        Args:
            element: Whatever that should be printed by devtools `debug`.
            description (Optional[str], optional): Description. Defaults to None.
        """

        if Helpers.should_debug:
            print(f"\n{Helpers.separator}\n")
            if description:
                print(f"{description}\n")
            debug(element)
            print(f"\n{Helpers.separator}\n")


@pytest.fixture
def helpers():
    """Fixture that returns `Helpers` class.

    Returns:
        Helpers: Helpers class.
    """

    return Helpers


FILE = __file__


@pytest.fixture
def data_path(tmp_path: pathlib.Path):
    """Fixture that returns a temporary path with data.

    If the directory `data` exists, its content will be copied to a temporary
    location.

    Args:
        tmp_path (pathlib.Path): Path to temporary location.

    Returns:
        pathlib.Path: Path to temporary location.
    """

    source = pathlib.Path(FILE).parent.joinpath("data")
    destination = tmp_path

    if source.is_dir():
        for item in os.listdir(source):
            s = os.path.join(source, item)
            print(s)
            d = os.path.join(destination, item)
            print(d)
            if os.path.isdir(s):
                shutil.copytree(s, d, symlinks=False, ignore=None)
            else:
                shutil.copy2(s, d)

    return tmp_path
