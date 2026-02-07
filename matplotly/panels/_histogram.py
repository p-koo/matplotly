"""Histogram controls — face/edge color, alpha, bin count."""
from __future__ import annotations

import ipywidgets as widgets
from matplotlib.colors import to_hex

from .._commands import BatchCommand, Command
from .._types import ArtistGroup
from ._base import ArtistPanel


class HistogramPanel(ArtistPanel):
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
            self._execute_and_redraw(BatchCommand(cmds, "Histogram face color"))
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
            self._execute_and_redraw(BatchCommand(cmds, "Histogram edge color"))
        edge_color.observe(_ec_cb, names="value")

        # Alpha
        alpha_sl = widgets.FloatSlider(
            value=ref.get_alpha() if ref.get_alpha() is not None else 1.0,
            min=0, max=1, step=0.05, description="Alpha:")

        def _alpha_cb(change):
            cmds = [Command(p, "alpha", p.get_alpha(), change["new"])
                    for p in patches]
            self._execute_and_redraw(BatchCommand(cmds, "Histogram alpha"))
        alpha_sl.observe(_alpha_cb, names="value")

        # Bin count — only if we have the original data
        bin_edges = self._group.metadata.get("bin_edges")
        if bin_edges is not None:
            current_bins = len(patches)
            bin_slider = widgets.IntSlider(value=current_bins, min=2, max=100,
                                            step=1, description="Bins:")

            def _bin_cb(change):
                # Re-binning requires the original data, which we may not have.
                # For now, just change visibility of patches as a visual cue.
                # Full re-bin would need stored raw data.
                pass

            bin_slider.observe(_bin_cb, names="value")
            controls.append(bin_slider)

        controls.extend([face_color, edge_color, alpha_sl])
        return widgets.VBox(controls)
