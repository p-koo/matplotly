"""Violin plot controls — face/edge color, alpha, linewidth."""
from __future__ import annotations

import ipywidgets as widgets
from matplotlib.colors import to_hex

from .._commands import Command
from .._types import ArtistGroup
from ._base import ArtistPanel


class ViolinPanel(ArtistPanel):
    def build(self) -> widgets.Widget:
        coll = self._group.artists[0]  # PolyCollection
        controls: list[widgets.Widget] = [
            widgets.HTML(f"<b>{self._group.label}</b>"),
        ]

        # Face color
        fc = coll.get_facecolor()
        try:
            current_fc = to_hex(fc[0]) if len(fc) > 0 else "#1f77b4"
        except Exception:
            current_fc = "#1f77b4"
        face_color = widgets.ColorPicker(value=current_fc, description="Face color:")

        def _fc_cb(change):
            old = coll.get_facecolor().copy()
            def _apply():
                coll.set_facecolor(change["new"])
            def _revert():
                coll.set_facecolor(old)
            self._execute_and_redraw(
                Command(coll, "facecolor", old, change["new"],
                        apply_fn=_apply, revert_fn=_revert,
                        description="Violin face color"))
        face_color.observe(_fc_cb, names="value")

        # Edge color
        ec = coll.get_edgecolor()
        try:
            current_ec = to_hex(ec[0]) if len(ec) > 0 else "#000000"
        except Exception:
            current_ec = "#000000"
        edge_color = widgets.ColorPicker(value=current_ec, description="Edge color:")

        def _ec_cb(change):
            old = coll.get_edgecolor().copy()
            def _apply():
                coll.set_edgecolor(change["new"])
            def _revert():
                coll.set_edgecolor(old)
            self._execute_and_redraw(
                Command(coll, "edgecolor", old, change["new"],
                        apply_fn=_apply, revert_fn=_revert,
                        description="Violin edge color"))
        edge_color.observe(_ec_cb, names="value")

        # Alpha
        alpha_val = coll.get_alpha()
        alpha_sl = widgets.FloatSlider(
            value=alpha_val if alpha_val is not None else 0.7,
            min=0, max=1, step=0.05, description="Alpha:")

        def _alpha_cb(change):
            self._execute_and_redraw(
                Command(coll, "alpha", coll.get_alpha(), change["new"],
                        description="Violin alpha"))
        alpha_sl.observe(_alpha_cb, names="value")

        # Linewidth
        lw = coll.get_linewidths()
        lw_val = float(lw[0]) if len(lw) > 0 else 1.0
        linewidth = widgets.FloatSlider(value=lw_val, min=0, max=5, step=0.1,
                                         description="Linewidth:")

        def _lw_cb(change):
            old = coll.get_linewidths().copy()
            def _apply():
                coll.set_linewidths([change["new"]])
            def _revert():
                coll.set_linewidths(old)
            self._execute_and_redraw(
                Command(coll, "linewidths", old, change["new"],
                        apply_fn=_apply, revert_fn=_revert,
                        description="Violin linewidth"))
        linewidth.observe(_lw_cb, names="value")

        controls.extend([face_color, edge_color, alpha_sl, linewidth])
        return widgets.VBox(controls)
