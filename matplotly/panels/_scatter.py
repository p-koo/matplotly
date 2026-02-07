"""Scatter plot controls — color, size, alpha, marker, edge."""
from __future__ import annotations

import ipywidgets as widgets
import numpy as np
from matplotlib.colors import to_hex

from .._commands import Command
from .._types import ArtistGroup
from ._base import ArtistPanel


class ScatterPanel(ArtistPanel):
    def build(self) -> widgets.Widget:
        coll = self._group.artists[0]
        controls: list[widgets.Widget] = [
            widgets.HTML(f"<b>{self._group.label}</b>"),
        ]

        # Color
        fc = coll.get_facecolor()
        try:
            current_color = to_hex(fc[0]) if len(fc) > 0 else "#1f77b4"
        except Exception:
            current_color = "#1f77b4"
        color = widgets.ColorPicker(value=current_color, description="Color:")

        def _color_cb(change):
            old = coll.get_facecolor().copy()
            def _apply():
                coll.set_facecolor(change["new"])
            def _revert():
                coll.set_facecolor(old)
            self._execute_and_redraw(
                Command(coll, "facecolor", old, change["new"],
                        apply_fn=_apply, revert_fn=_revert,
                        description=f"{self._group.label} color"))
        color.observe(_color_cb, names="value")

        # Size
        sizes = coll.get_sizes()
        avg_size = float(np.mean(sizes)) if len(sizes) > 0 else 20.0
        size = widgets.FloatSlider(value=avg_size, min=1, max=200, step=1,
                                    description="Size:")

        def _size_cb(change):
            old = coll.get_sizes().copy()
            def _apply():
                coll.set_sizes([change["new"]] * len(coll.get_offsets()))
            def _revert():
                coll.set_sizes(old)
            self._execute_and_redraw(
                Command(coll, "sizes", old, change["new"],
                        apply_fn=_apply, revert_fn=_revert,
                        description=f"{self._group.label} size"))
        size.observe(_size_cb, names="value")

        # Alpha
        alpha_val = coll.get_alpha()
        alpha_sl = widgets.FloatSlider(
            value=alpha_val if alpha_val is not None else 1.0,
            min=0, max=1, step=0.05, description="Alpha:")

        def _alpha_cb(change):
            self._execute_and_redraw(
                Command(coll, "alpha", coll.get_alpha(), change["new"],
                        description=f"{self._group.label} alpha"))
        alpha_sl.observe(_alpha_cb, names="value")

        # Marker
        from matplotlib.markers import MarkerStyle
        markers = [("circle", "o"), ("square", "s"), ("triangle", "^"),
                   ("diamond", "D"), ("plus", "+"), ("x", "x"), ("star", "*"),
                   ("point", ".")]

        # Get current marker paths to try to detect shape
        marker_dd = widgets.Dropdown(options=markers, value="o",
                                     description="Marker:",
                                     style={"description_width": "48px"},
                                     layout=widgets.Layout(width="150px"))

        def _marker_cb(change):
            old_paths = coll.get_paths()
            new_marker = MarkerStyle(change["new"])
            new_path = new_marker.get_path().transformed(new_marker.get_transform())
            def _apply():
                coll.set_paths([new_path])
            def _revert():
                coll.set_paths(old_paths)
            self._execute_and_redraw(
                Command(coll, "paths", old_paths, change["new"],
                        apply_fn=_apply, revert_fn=_revert,
                        description=f"{self._group.label} marker"))
        marker_dd.observe(_marker_cb, names="value")

        # Edge width
        lw = coll.get_linewidths()
        edge_w = widgets.FloatSlider(
            value=float(lw[0]) if len(lw) > 0 else 1.0,
            min=0, max=5, step=0.1, description="Edge width:")

        def _ew_cb(change):
            old = coll.get_linewidths().copy()
            def _apply():
                coll.set_linewidths([change["new"]])
            def _revert():
                coll.set_linewidths(old)
            self._execute_and_redraw(
                Command(coll, "linewidths", old, change["new"],
                        apply_fn=_apply, revert_fn=_revert,
                        description=f"{self._group.label} edge width"))
        edge_w.observe(_ew_cb, names="value")

        controls.extend([color, size, alpha_sl, marker_dd, edge_w])
        return widgets.VBox(controls)
