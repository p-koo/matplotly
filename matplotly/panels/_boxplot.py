"""Box plot controls — box color, whisker style, median, flier markers."""
from __future__ import annotations

import ipywidgets as widgets
from matplotlib.colors import to_hex

from .._commands import BatchCommand, Command
from .._types import ArtistGroup
from ._base import ArtistPanel


class BoxplotPanel(ArtistPanel):
    def build(self) -> widgets.Widget:
        artists = self._group.artists
        controls: list[widgets.Widget] = [
            widgets.HTML(f"<b>{self._group.label}</b>"),
        ]

        # Separate artists by role
        boxes = [a for a in artists if hasattr(a, "_label") and "_box" in str(getattr(a, "_label", "")).lower()]
        medians = [a for a in artists if hasattr(a, "_label") and "_median" in str(getattr(a, "_label", "")).lower()]
        whiskers = [a for a in artists if hasattr(a, "_label") and "_whisker" in str(getattr(a, "_label", "")).lower()]

        # Box color (applies to box patches)
        if boxes:
            ref = boxes[0]
            try:
                if hasattr(ref, "get_facecolor"):
                    current = to_hex(ref.get_facecolor())
                else:
                    current = to_hex(ref.get_color())
            except Exception:
                current = "#1f77b4"
            box_color = widgets.ColorPicker(value=current, description="Box color:")

            def _bc_cb(change):
                cmds = []
                for b in boxes:
                    if hasattr(b, "set_facecolor"):
                        cmds.append(Command(b, "facecolor", current, change["new"]))
                    else:
                        cmds.append(Command(b, "color", current, change["new"]))
                self._execute_and_redraw(BatchCommand(cmds, "Box color"))
            box_color.observe(_bc_cb, names="value")
            controls.append(box_color)

        # Median color/width
        if medians:
            ref = medians[0]
            try:
                med_color_val = to_hex(ref.get_color())
            except Exception:
                med_color_val = "#ff7f0e"
            med_color = widgets.ColorPicker(value=med_color_val, description="Median clr:")

            def _mc_cb(change):
                cmds = [Command(m, "color", to_hex(m.get_color()), change["new"])
                        for m in medians]
                self._execute_and_redraw(BatchCommand(cmds, "Median color"))
            med_color.observe(_mc_cb, names="value")

            med_width = widgets.FloatSlider(
                value=ref.get_linewidth(), min=0.5, max=6, step=0.25,
                description="Median lw:")

            def _mw_cb(change):
                cmds = [Command(m, "linewidth", m.get_linewidth(), change["new"])
                        for m in medians]
                self._execute_and_redraw(BatchCommand(cmds, "Median width"))
            med_width.observe(_mw_cb, names="value")
            controls.extend([med_color, med_width])

        # Whisker style
        if whiskers:
            ref = whiskers[0]
            styles = [("solid", "-"), ("dashed", "--"), ("dotted", ":"), ("dashdot", "-.")]
            _ls_map = {"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-."}
            current_ls = ref.get_linestyle()
            current_ls = _ls_map.get(current_ls, current_ls)
            if current_ls not in [v for _, v in styles]:
                current_ls = "-"
            whisk_style = widgets.Dropdown(options=styles, value=current_ls,
                                           description="Whisker:",
                                           style={"description_width": "48px"},
                                           layout=widgets.Layout(width="150px"))

            def _ws_cb(change):
                cmds = [Command(w, "linestyle", w.get_linestyle(), change["new"])
                        for w in whiskers]
                self._execute_and_redraw(BatchCommand(cmds, "Whisker style"))
            whisk_style.observe(_ws_cb, names="value")

            whisk_width = widgets.FloatSlider(
                value=ref.get_linewidth(), min=0.1, max=5, step=0.1,
                description="Whisker lw:")

            def _ww_cb(change):
                cmds = [Command(w, "linewidth", w.get_linewidth(), change["new"])
                        for w in whiskers]
                self._execute_and_redraw(BatchCommand(cmds, "Whisker width"))
            whisk_width.observe(_ww_cb, names="value")
            controls.extend([whisk_style, whisk_width])

        if len(controls) == 1:
            controls.append(widgets.HTML("<i>No editable box plot elements found.</i>"))

        return widgets.VBox(controls)
