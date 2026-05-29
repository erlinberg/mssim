"""Plotting helpers for mssim.

The package is split into:
- :mod:`mssim.plots.general` for observable-agnostic I/O helpers
- :mod:`mssim.plots.utilities` for shared plotting utilities
- :mod:`mssim.plots.magnetization` for the current observable-specific code
"""

from __future__ import annotations

from .general import load, parameter_recap
from .magnetization import (
    magnetization,
    plot_magnetization,
    plot_magnetization_diff,
)

__all__ = [
    "magnetization",
    "load",
    "parameter_recap",
    "plot_magnetization",
    "plot_magnetization_diff",
]