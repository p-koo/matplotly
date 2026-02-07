"""Error bar controls — cap size, line width, color."""
from __future__ import annotations

import ipywidgets as widgets
from matplotlib.colors import to_hex

from .._commands import BatchCommand, Command
from .._types import ArtistGroup
from ._base import ArtistPanel


class ErrorbarPanel(ArtistPanel):
    def build(self) -> widgets.Widget:
        artists = self._group.artists
        controls: list[widgets.Widget] = [
            widgets.HTML(f"<b>{self._group.label}</b>"),
        ]

        # Find the data line (first Line2D that isn't a cap)
        from matplotlib.lines import Line2D
        data_line = None
        cap_lines = []
        bar_lines = []
        for a in artists:
            if isinstance(a, Line2D):
                if data_line is None:
                    data_line = a
                else:
                    cap_lines.append(a)
            else:
                bar_lines.append(a)

        # Color
        if data_line:
            try:
                current_color = to_hex(data_line.get_color())
            except Exception:
                current_color = "#1f77b4"
            color = widgets.ColorPicker(value=current_color, description="Color:")

            def _color_cb(change):
                cmds = []
                for a in artists:
                    if isinstance(a, Line2D):
                        cmds.append(Command(a, "color", to_hex(a.get_color()), change["new"]))
                    elif hasattr(a, "set_color"):
                        try:
                            old_c = to_hex(a.get_color()[0]) if hasattr(a.get_color(), '__len__') else to_hex(a.get_color())
                        except Exception:
                            old_c = current_color
                        cmds.append(Command(a, "color", old_c, change["new"]))
                self._execute_and_redraw(BatchCommand(cmds, "Errorbar color"))
            color.observe(_color_cb, names="value")
            controls.append(color)

            # Linewidth
            lw = widgets.FloatSlider(value=data_line.get_linewidth(), min=0.1, max=6,
                                      step=0.1, description="Line width:")

            def _lw_cb(change):
                cmds = [Command(a, "linewidth", a.get_linewidth(), change["new"])
                        for a in artists if isinstance(a, Line2D)]
                self._execute_and_redraw(BatchCommand(cmds, "Errorbar linewidth"))
            lw.observe(_lw_cb, names="value")
            controls.append(lw)

        # Cap size (marker size on cap lines)
        if cap_lines:
            ref_cap = cap_lines[0]
            cap_size = widgets.FloatSlider(
                value=ref_cap.get_markersize(), min=0, max=15, step=0.5,
                description="Cap size:")

            def _cap_cb(change):
                cmds = [Command(c, "markersize", c.get_markersize(), change["new"])
                        for c in cap_lines]
                self._execute_and_redraw(BatchCommand(cmds, "Cap size"))
            cap_size.observe(_cap_cb, names="value")
            controls.append(cap_size)

        if len(controls) == 1:
            controls.append(widgets.HTML("<i>No editable errorbar elements found.</i>"))

        return widgets.VBox(controls)
