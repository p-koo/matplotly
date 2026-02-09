"""matplotly — Inline Jupyter matplotlib figure editor.

Usage:
    # Pass a figure directly
    fig, ax = plt.subplots()
    ax.plot(x, y)
    matplotly(fig)

    # Decorator
    @matplotly
    def my_plot():
        plt.plot(x, y)

    # Context manager
    with matplotly() as pb:
        fig, ax = plt.subplots()
        ax.plot(x, y)
"""
from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["matplotly"]

import functools
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def _suppress_auto_display(fig: Figure) -> None:
    """Hide the ipympl auto-displayed canvas and remove from figure manager."""
    # Remove from pyplot's figure manager so ipympl stops tracking it
    try:
        import matplotlib._pylab_helpers as _helpers
        _helpers.Gcf.figs.pop(fig.number, None)
    except Exception:
        pass
    # Hide the auto-displayed ipympl canvas widget (if ipympl is active,
    # fig.canvas is an ipywidget that was already displayed by plt.subplots)
    try:
        fig.canvas.layout.display = 'none'
        fig.canvas.layout.height = '0px'
    except (AttributeError, Exception):
        pass
    # Clear any already-rendered static output from the cell
    try:
        from IPython.display import clear_output
        clear_output(wait=False)
    except Exception:
        pass


class _PlotBuildContext:
    """Context manager that captures the current figure on exit."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
        from ._api import PlotBuildSession
        fig = plt.gcf()
        _suppress_auto_display(fig)
        session = PlotBuildSession(fig)
        session.display()
        return False


def _capture_cell_source() -> str | None:
    """Grab the source of the currently executing Jupyter cell."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            # In[] list holds all executed cell sources; last entry is current
            in_list = ip.user_ns.get("In")
            if in_list and len(in_list) > 0:
                return in_list[-1]
    except Exception:
        pass
    return None


def matplotly(target: Figure | Any | None = None):
    """Interactive matplotlib figure editor.

    Parameters
    ----------
    target : Figure, callable, or None
        - Figure: opens the editor on that figure immediately.
        - callable: decorator mode — calls the function, captures plt.gcf(),
          and opens the editor.
        - None: returns a context manager.
    """
    from ._api import PlotBuildSession

    cell_source = _capture_cell_source()

    # Case 1: Figure passed directly
    if isinstance(target, Figure):
        _suppress_auto_display(target)
        session = PlotBuildSession(target, cell_source=cell_source)
        session.display()
        return None  # suppress Jupyter auto-display of return value

    # Case 2: Callable — decorator mode
    if callable(target):
        @functools.wraps(target)
        def wrapper(*args, **kwargs):
            target(*args, **kwargs)
            fig = plt.gcf()
            _suppress_auto_display(fig)
            cs = _capture_cell_source()
            session = PlotBuildSession(fig, cell_source=cs)
            session.display()
        return wrapper

    # Case 3: None — context manager mode
    if target is None:
        return _PlotBuildContext()

    raise TypeError(f"matplotly() expects a Figure, callable, or None, got {type(target)}")
