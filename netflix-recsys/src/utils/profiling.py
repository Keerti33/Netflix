"""Training profiling utilities.

Provides a decorator and context manager that wrap any ``fit()`` call and
log:

- Wall-clock training time
- Peak RAM usage (via ``tracemalloc``)
- Serialised model size on disk (optional)

Usage::

    from src.utils.profiling import profile_fit

    @profile_fit
    def fit(self, train_df, **kwargs):
        ...

Or as a context manager::

    from src.utils.profiling import FitProfiler

    with FitProfiler("MyModel") as prof:
        model.fit(train_df)
    print(prof.summary())
"""

from __future__ import annotations

import functools
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ProfileResult:
    """Container for profiling results from a single fit() call."""

    model_name: str
    wall_time_s: float = 0.0
    peak_ram_mb: float = 0.0
    model_size_mb: float = 0.0

    def summary(self) -> str:
        """Return a human-readable one-line summary."""
        parts = [
            f"[{self.model_name}] Profiling:",
            f"wall_time={self.wall_time_s:.2f}s",
            f"peak_RAM={self.peak_ram_mb:.2f}MB",
        ]
        if self.model_size_mb > 0:
            parts.append(f"model_disk={self.model_size_mb:.2f}MB")
        return "  ".join(parts)

    def as_dict(self) -> dict:
        """Return profiling data as a plain dict."""
        return {
            "model_name": self.model_name,
            "wall_time_s": self.wall_time_s,
            "peak_ram_mb": self.peak_ram_mb,
            "model_size_mb": self.model_size_mb,
        }


class FitProfiler:
    """Context manager that profiles a block of code.

    Measures wall-clock time and peak memory allocation using
    ``tracemalloc``.

    Args:
        model_name: Label for log output.
        verbose: If True, prints the summary on exit.

    Example::

        with FitProfiler("SVD") as prof:
            model.fit(train_df)
        print(prof.result.wall_time_s)
    """

    def __init__(self, model_name: str = "Model", verbose: bool = True) -> None:
        self.model_name = model_name
        self.verbose = verbose
        self.result: ProfileResult = ProfileResult(model_name=model_name)

    def __enter__(self) -> "FitProfiler":
        tracemalloc.start()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed = time.perf_counter() - self._t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.result.wall_time_s = elapsed
        self.result.peak_ram_mb = peak / (1024 ** 2)

        if self.verbose:
            print(self.result.summary())
        return False  # do not suppress exceptions

    def set_model_path(self, path: str | Path) -> None:
        """Record serialised model size from a saved file.

        Args:
            path: Path to the saved model file.
        """
        p = Path(path)
        if p.exists():
            self.result.model_size_mb = p.stat().st_size / (1024 ** 2)


def profile_fit(func):
    """Decorator that profiles a model's ``fit()`` method.

    Wraps the function call with ``FitProfiler`` and attaches the
    ``ProfileResult`` to the model instance as ``self.profile_result_``.

    The decorated function must be a bound method whose first positional
    argument is ``self`` with a ``model_name`` attribute.

    Example::

        class MyModel(BaseRecommender):
            model_name = "MyModel"

            @profile_fit
            def fit(self, train_df, **kwargs):
                ...

        m = MyModel()
        m.fit(train_df)
        print(m.profile_result_.wall_time_s)
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        name = getattr(self, "model_name", type(self).__name__)
        with FitProfiler(name, verbose=True) as prof:
            result = func(self, *args, **kwargs)
        self.profile_result_ = prof.result
        return result

    return wrapper
