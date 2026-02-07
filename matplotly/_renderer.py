"""Canvas management — renders figure as PNG in an Output widget."""
from __future__ import annotations

import io
import time

import ipywidgets as widgets
from matplotlib.figure import Figure


class CanvasManager:
    """Wraps a matplotlib Figure for display in Jupyter.

    Always renders as a PNG inside an ipywidgets.Output widget.
    This avoids duplicate-figure issues with the ipympl widget backend
    (which auto-displays the canvas when plt.subplots() is called).
    """

    _MIN_DRAW_INTERVAL_S = 0.08  # 80ms throttle

    def __init__(self, fig: Figure):
        self._fig = fig
        self._last_draw = 0.0
        self._output = widgets.Output()
        self._widget = self._output
        self._render()

    @property
    def widget(self) -> widgets.Widget:
        return self._widget

    def _render(self) -> None:
        """Render figure as PNG inside the Output widget."""
        self._output.clear_output(wait=True)
        with self._output:
            from IPython.display import display as ipy_display, Image
            # Ensure labels/ticks/titles fit within the figure
            try:
                self._fig.tight_layout()
            except Exception:
                pass
            # Collect extra artists (like outside legends) for tight bbox
            extra = []
            for ax in self._fig.get_axes():
                leg = ax.get_legend()
                if leg is not None:
                    extra.append(leg)
            buf = io.BytesIO()
            self._fig.savefig(
                buf, format='png', bbox_inches='tight',
                bbox_extra_artists=extra or None,
                pad_inches=0.15,
                facecolor=self._fig.get_facecolor(),
                edgecolor='none', dpi=100)
            buf.seek(0)
            ipy_display(Image(data=buf.read()))

    def redraw(self) -> None:
        """Request a canvas redraw, throttled to avoid excess repaints."""
        now = time.monotonic()
        if now - self._last_draw < self._MIN_DRAW_INTERVAL_S:
            return
        self._last_draw = now
        self._render()

    def force_redraw(self) -> None:
        """Redraw immediately, bypassing the throttle."""
        self._last_draw = time.monotonic()
        self._render()
