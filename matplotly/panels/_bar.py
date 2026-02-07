"""Bar chart controls — face/edge color, edge width, hatch, bar width."""
from __future__ import annotations

import ipywidgets as widgets
from matplotlib.colors import to_hex

from .._commands import BatchCommand, Command
from .._types import ArtistGroup
from ._base import ArtistPanel


class BarPanel(ArtistPanel):
    def build(self) -> widgets.Widget:
        patches = self._group.artists
        ref = patches[0]
        controls: list[widgets.Widget] = [
            widgets.HTML(f"<b>{self._group.label}</b>"),
        ]

        # Face color
        try:
            current_fc = to_hex(ref.get_facecolor())
        except Exception:
            current_fc = "#1f77b4"
        face_color = widgets.ColorPicker(value=current_fc, description="Face color:")

        def _fc_cb(change):
            cmds = [Command(p, "facecolor", to_hex(p.get_facecolor()), change["new"])
                    for p in patches]
            self._execute_and_redraw(BatchCommand(cmds, "Bar face color"))
        face_color.observe(_fc_cb, names="value")

        # Edge color
        try:
            current_ec = to_hex(ref.get_edgecolor())
        except Exception:
            current_ec = "#000000"
        edge_color = widgets.ColorPicker(value=current_ec, description="Edge color:")

        def _ec_cb(change):
            cmds = [Command(p, "edgecolor", to_hex(p.get_edgecolor()), change["new"])
                    for p in patches]
            self._execute_and_redraw(BatchCommand(cmds, "Bar edge color"))
        edge_color.observe(_ec_cb, names="value")

        # Edge width
        edge_width = widgets.FloatSlider(value=ref.get_linewidth(), min=0, max=5,
                                          step=0.1, description="Edge width:")

        def _ew_cb(change):
            cmds = [Command(p, "linewidth", p.get_linewidth(), change["new"])
                    for p in patches]
            self._execute_and_redraw(BatchCommand(cmds, "Bar edge width"))
        edge_width.observe(_ew_cb, names="value")

        # Hatch
        hatches = [("none", ""), ("/", "/"), ("\\", "\\"), ("|", "|"),
                   ("-", "-"), ("+", "+"), ("x", "x"), ("o", "o"),
                   (".", "."), ("*", "*")]
        hatch = widgets.Dropdown(options=hatches, value=ref.get_hatch() or "",
                                 description="Hatch:",
                                 style={"description_width": "48px"},
                                 layout=widgets.Layout(width="150px"))

        def _hatch_cb(change):
            cmds = [Command(p, "hatch", p.get_hatch() or "", change["new"])
                    for p in patches]
            self._execute_and_redraw(BatchCommand(cmds, "Bar hatch"))
        hatch.observe(_hatch_cb, names="value")

        # Alpha
        alpha_sl = widgets.FloatSlider(
            value=ref.get_alpha() if ref.get_alpha() is not None else 1.0,
            min=0, max=1, step=0.05, description="Alpha:")

        def _alpha_cb(change):
            cmds = [Command(p, "alpha", p.get_alpha(), change["new"])
                    for p in patches]
            self._execute_and_redraw(BatchCommand(cmds, "Bar alpha"))
        alpha_sl.observe(_alpha_cb, names="value")

        controls.extend([face_color, edge_color, edge_width, hatch, alpha_sl])
        return widgets.VBox(controls)
