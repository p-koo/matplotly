"""Error bar controls — toggle-based design with per-section colors and alphas.

Four independent toggles, each with its own color and alpha:
  - Error Bars: show/hide error bar lines + caps
  - Markers: show/hide markers on the data line
  - Line: show/hide connecting line
  - Shaded Region: add ax.fill_between() overlay
"""
from __future__ import annotations

import numpy as np
import ipywidgets as widgets
from matplotlib.colors import to_hex

from .._commands import BatchCommand, Command
from .._types import ArtistGroup, PlotType
from ._base import ArtistPanel
from ._color_utils import (
    _DW, _NW, _SN,
    _make_color_dot, _refresh_legend, _slider_num,
    cmap_color_btn,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MARKERS = [
    ("none", ""), ("circle", "o"), ("square", "s"),
    ("triangle", "^"), ("diamond", "D"), ("plus", "+"),
    ("x", "x"), ("star", "*"),
]

_LINE_STYLES = [
    ("solid", "-"), ("dashed", "--"),
    ("dotted", ":"), ("dashdot", "-."),
]


class ErrorbarPanel(ArtistPanel):
    """Full-featured errorbar panel with independent toggles, per-section colors and alphas."""

    _plot_number: int = 0  # set by _api.py before build()
    _on_label_changed = None  # callback set by _api.py

    def build(self) -> widgets.Widget:
        meta = self._group.metadata
        container = meta.get("container")
        data_line = container[0] if container else None

        # --- Extract initial state from metadata ---
        self._x_data = meta.get("x_data")
        self._y_data = meta.get("y_data")
        self._yerr = meta.get("yerr")
        self._xerr = meta.get("xerr")
        self._has_yerr = meta.get("has_yerr", True)
        self._has_xerr = meta.get("has_xerr", False)

        base_color = meta.get("color", "#1f77b4")
        base_alpha = meta.get("alpha", 1.0)
        self._line_width = meta.get("line_width", 1.5)
        self._line_style = meta.get("line_style", "-")
        self._bar_lw = meta.get("bar_lw", 1.5)
        self._cap_size = meta.get("cap_size", 3.0)
        self._marker = meta.get("marker", "")
        self._marker_size = meta.get("marker_size", 6.0)

        # Per-section colors (all default to base introspected color)
        self._bar_color = base_color
        self._marker_color = base_color
        self._line_color = base_color
        self._shade_color = base_color

        # Per-section alphas
        self._bar_alpha = base_alpha
        self._marker_alpha = base_alpha
        self._line_alpha = base_alpha
        self._shade_alpha = 0.3

        # Determine initial toggle states from data line
        self._show_bars = True  # error bars on by default
        if data_line is not None:
            ls = data_line.get_linestyle()
            _ls_map = {"solid": "-", "dashed": "--",
                       "dotted": ":", "dashdot": "-."}
            ls_norm = _ls_map.get(ls, ls)
            self._show_line = ls_norm not in ("none", "None")
            # If line was initially hidden, reset style to solid
            # so toggling line on actually draws a visible line.
            if not self._show_line:
                self._line_style = "-"

            mk = data_line.get_marker()
            self._show_markers = bool(mk and mk not in ("", "None", "none"))
        else:
            self._show_line = True
            self._show_markers = False

        self._show_shaded = False

        # Label
        label = self._group.label
        if label.startswith("Errorbar: "):
            label = label[len("Errorbar: "):]
        self._label = label

        # --- Build UI ---
        return self._build_ui()

    def _build_ui(self) -> widgets.Widget:
        # --- Collapsible header ---
        num = self._plot_number or ""
        header_prefix = f"Errorbar {num}: " if num else ""
        self._header_prefix = header_prefix

        self._color_indicator = widgets.HTML(
            _make_color_dot(self._bar_color))

        toggle_btn = widgets.Button(
            description=f"  {header_prefix}{self._label}",
            icon="chevron-right",
            layout=widgets.Layout(width='100%', height='28px'))
        toggle_btn.style.button_color = '#f0f0f0'
        toggle_btn.style.font_weight = 'bold'
        self._toggle_btn = toggle_btn

        header_row = widgets.HBox(
            [self._color_indicator, toggle_btn],
            layout=widgets.Layout(align_items='center', gap='4px'))

        controls = self._build_controls()
        controls_box = widgets.VBox(
            controls,
            layout=widgets.Layout(display='none', padding='2px 0 4px 12px'))
        self._controls_box = controls_box
        self._is_expanded = False

        def _toggle(btn):
            self._is_expanded = not self._is_expanded
            if self._is_expanded:
                controls_box.layout.display = ''
                toggle_btn.icon = "chevron-down"
            else:
                controls_box.layout.display = 'none'
                toggle_btn.icon = "chevron-right"
        toggle_btn.on_click(_toggle)

        return widgets.VBox(
            [header_row, controls_box],
            layout=widgets.Layout(
                border='1px solid #ddd', border_radius='4px',
                margin='2px 0', padding='2px'))

    def _update_header_dot(self):
        """Update header color dot to reflect the primary visible color."""
        if self._show_line:
            c = self._line_color
        elif self._show_markers:
            c = self._marker_color
        elif self._show_bars:
            c = self._bar_color
        elif self._show_shaded:
            c = self._shade_color
        else:
            c = self._bar_color
        self._color_indicator.value = _make_color_dot(c)

    def _build_controls(self) -> list[widgets.Widget]:
        controls: list[widgets.Widget] = []

        # --- Name ---
        name_field = widgets.Text(
            value=self._label, description="Name:",
            style={"description_width": _DW},
            layout=widgets.Layout(width="95%"))

        def _on_name(change):
            old_label = self._label
            self._label = change["new"]
            pfx = self._header_prefix
            icon = "chevron-down" if self._is_expanded else "chevron-right"
            self._toggle_btn.icon = icon
            self._toggle_btn.description = f"  {pfx}{self._label}"
            container = self._group.metadata.get("container")
            if container and container[0] is not None:
                dl = container[0]
                self._stack.execute(
                    Command(dl, "label", old_label, self._label,
                            description="Errorbar label"))
            self._store_errorbar_info()
            _refresh_legend(self._group.axes)
            if self._on_label_changed is not None:
                self._on_label_changed()
            self._canvas.force_redraw()
        name_field.observe(_on_name, names="value")
        controls.append(name_field)

        # === Error Bars section (toggle) ===
        bars_cb = widgets.Checkbox(
            value=self._show_bars,
            description="Error Bars",
            indent=False,
            layout=widgets.Layout(margin='6px 0 0 0'))

        # Bar color
        def _on_bar_color(hex_val):
            self._bar_color = hex_val
            self._update_header_dot()
            self._redraw()
        bar_color_btn, bar_swatch_row = cmap_color_btn(
            self._bar_color, _on_bar_color)
        bar_color_row = widgets.HBox(
            [widgets.Label("Color:", layout=widgets.Layout(width='42px')),
             bar_color_btn],
            layout=widgets.Layout(align_items='center', gap='4px'))

        # Bar alpha
        bar_alpha_sl = widgets.FloatSlider(
            value=round(self._bar_alpha, 2),
            min=0, max=1, step=0.05, description="Alpha:", style=_SN)

        def _bar_alpha_cb(change):
            self._bar_alpha = change["new"]
            self._redraw()
        bar_alpha_sl.observe(_bar_alpha_cb, names="value")

        # Bar width
        bar_lw_sl = widgets.FloatSlider(
            value=round(self._bar_lw, 2),
            min=0.1, max=10, step=0.1, description="Bar w:", style=_SN)

        def _bar_lw_cb(change):
            self._bar_lw = change["new"]
            self._redraw()
        bar_lw_sl.observe(_bar_lw_cb, names="value")

        # Cap size
        cap_sl = widgets.FloatSlider(
            value=round(self._cap_size, 2),
            min=0, max=15, step=0.5, description="Cap sz:", style=_SN)

        def _cap_cb(change):
            self._cap_size = change["new"]
            self._redraw()
        cap_sl.observe(_cap_cb, names="value")

        bar_controls = widgets.VBox(
            [widgets.VBox([bar_color_row, bar_swatch_row]),
             _slider_num(bar_alpha_sl),
             _slider_num(bar_lw_sl), _slider_num(cap_sl)],
            layout=widgets.Layout(
                padding='0 0 0 12px',
                display='' if self._show_bars else 'none'))
        self._bar_controls = bar_controls

        # Store update_color for ColormapPanel (use bar color as primary)
        def _ext_update_color(hex_val):
            self._bar_color = hex_val
            bar_color_btn.style.button_color = hex_val
            self._update_header_dot()
        self._update_color = _ext_update_color

        def _bars_toggle(change):
            self._show_bars = change["new"]
            bar_controls.layout.display = '' if change["new"] else 'none'
            self._update_header_dot()
            self._redraw()
        bars_cb.observe(_bars_toggle, names="value")

        controls.append(bars_cb)
        controls.append(bar_controls)

        # === Markers section (toggle) ===
        markers_cb = widgets.Checkbox(
            value=self._show_markers,
            description="Markers",
            indent=False,
            layout=widgets.Layout(margin='2px 0 0 0'))

        # Marker color
        def _on_marker_color(hex_val):
            self._marker_color = hex_val
            self._update_header_dot()
            self._redraw()
        mkr_color_btn, mkr_swatch_row = cmap_color_btn(
            self._marker_color, _on_marker_color)
        mkr_color_row = widgets.HBox(
            [widgets.Label("Color:", layout=widgets.Layout(width='42px')),
             mkr_color_btn],
            layout=widgets.Layout(align_items='center', gap='4px'))

        # Marker alpha
        mkr_alpha_sl = widgets.FloatSlider(
            value=round(self._marker_alpha, 2),
            min=0, max=1, step=0.05, description="Alpha:", style=_SN)

        def _mkr_alpha_cb(change):
            self._marker_alpha = change["new"]
            self._redraw()
        mkr_alpha_sl.observe(_mkr_alpha_cb, names="value")

        # Marker dropdown
        cur_m = self._marker if self._marker in [
            v for _, v in _MARKERS] else ""
        marker_dd = widgets.Dropdown(
            options=_MARKERS, value=cur_m,
            description="Marker:",
            style={"description_width": _DW},
            layout=widgets.Layout(width="150px"))

        def _marker_cb(change):
            self._marker = change["new"]
            self._redraw()
        marker_dd.observe(_marker_cb, names="value")

        # Marker size
        ms_sl = widgets.FloatSlider(
            value=round(self._marker_size, 2),
            min=0, max=20, step=0.5, description="Mkr sz:", style=_SN)

        def _ms_cb(change):
            self._marker_size = change["new"]
            self._redraw()
        ms_sl.observe(_ms_cb, names="value")

        marker_controls = widgets.VBox(
            [widgets.VBox([mkr_color_row, mkr_swatch_row]),
             _slider_num(mkr_alpha_sl),
             marker_dd, _slider_num(ms_sl)],
            layout=widgets.Layout(
                padding='0 0 0 12px',
                display='' if self._show_markers else 'none'))
        self._marker_controls = marker_controls

        def _markers_toggle(change):
            self._show_markers = change["new"]
            # Default to circle when enabling markers with no shape set
            if change["new"] and not self._marker:
                self._marker = "o"
                marker_dd.value = "o"
            marker_controls.layout.display = '' if change["new"] else 'none'
            self._update_header_dot()
            self._redraw()
        markers_cb.observe(_markers_toggle, names="value")

        controls.append(markers_cb)
        controls.append(marker_controls)

        # === Line section (toggle) ===
        line_cb = widgets.Checkbox(
            value=self._show_line,
            description="Line",
            indent=False,
            layout=widgets.Layout(margin='2px 0 0 0'))

        # Line color
        def _on_line_color(hex_val):
            self._line_color = hex_val
            self._update_header_dot()
            self._redraw()
        line_color_btn, line_swatch_row = cmap_color_btn(
            self._line_color, _on_line_color)
        line_color_row = widgets.HBox(
            [widgets.Label("Color:", layout=widgets.Layout(width='42px')),
             line_color_btn],
            layout=widgets.Layout(align_items='center', gap='4px'))

        # Line alpha
        line_alpha_sl = widgets.FloatSlider(
            value=round(self._line_alpha, 2),
            min=0, max=1, step=0.05, description="Alpha:", style=_SN)

        def _line_alpha_cb(change):
            self._line_alpha = change["new"]
            self._redraw()
        line_alpha_sl.observe(_line_alpha_cb, names="value")

        # Line width
        lw_sl = widgets.FloatSlider(
            value=round(self._line_width, 2),
            min=0.1, max=10, step=0.1, description="Line w:", style=_SN)

        def _lw_cb(change):
            self._line_width = change["new"]
            self._redraw()
        lw_sl.observe(_lw_cb, names="value")

        # Line style
        ls_dd = widgets.Dropdown(
            options=_LINE_STYLES,
            value=self._line_style if self._line_style in [
                v for _, v in _LINE_STYLES] else "-",
            description="Line st:",
            style={"description_width": _DW},
            layout=widgets.Layout(width="150px"))

        def _ls_cb(change):
            self._line_style = change["new"]
            self._redraw()
        ls_dd.observe(_ls_cb, names="value")

        line_controls = widgets.VBox(
            [widgets.VBox([line_color_row, line_swatch_row]),
             _slider_num(line_alpha_sl),
             _slider_num(lw_sl), ls_dd],
            layout=widgets.Layout(
                padding='0 0 0 12px',
                display='' if self._show_line else 'none'))
        self._line_controls = line_controls

        def _line_toggle(change):
            self._show_line = change["new"]
            line_controls.layout.display = '' if change["new"] else 'none'
            self._update_header_dot()
            self._redraw()
        line_cb.observe(_line_toggle, names="value")

        controls.append(line_cb)
        controls.append(line_controls)

        # === Shaded Region section (toggle) ===
        shaded_cb = widgets.Checkbox(
            value=self._show_shaded,
            description="Shaded Region",
            indent=False,
            layout=widgets.Layout(margin='2px 0 0 0'))

        # Shade color
        def _on_shade_color(hex_val):
            self._shade_color = hex_val
            self._update_header_dot()
            self._redraw()
        shade_color_btn, shade_swatch_row = cmap_color_btn(
            self._shade_color, _on_shade_color)
        shade_color_row = widgets.HBox(
            [widgets.Label("Color:", layout=widgets.Layout(width='42px')),
             shade_color_btn],
            layout=widgets.Layout(align_items='center', gap='4px'))

        shade_alpha_sl = widgets.FloatSlider(
            value=round(self._shade_alpha, 2),
            min=0, max=1, step=0.05,
            description="Alpha:", style=_SN)

        def _shade_alpha_cb(change):
            self._shade_alpha = change["new"]
            self._redraw()
        shade_alpha_sl.observe(_shade_alpha_cb, names="value")

        shaded_controls = widgets.VBox(
            [widgets.VBox([shade_color_row, shade_swatch_row]),
             _slider_num(shade_alpha_sl)],
            layout=widgets.Layout(
                padding='0 0 0 12px',
                display='' if self._show_shaded else 'none'))
        self._shaded_controls = shaded_controls

        def _shaded_toggle(change):
            self._show_shaded = change["new"]
            shaded_controls.layout.display = '' if change["new"] else 'none'
            self._update_header_dot()
            self._redraw()
        shaded_cb.observe(_shaded_toggle, names="value")

        controls.append(shaded_cb)
        controls.append(shaded_controls)

        return controls

    # ------------------------------------------------------------------
    # Redraw logic
    # ------------------------------------------------------------------

    def _clear_errorbar_artists(self):
        """Remove all artists tagged with _matplotly_errorbar and the
        original ErrorbarContainer."""
        ax = self._group.axes

        # Remove tagged artists
        for line in list(ax.lines):
            if getattr(line, '_matplotly_errorbar', False):
                line.remove()
        for coll in list(ax.collections):
            if getattr(coll, '_matplotly_errorbar', False):
                coll.remove()

        # Remove original container
        container = self._group.metadata.get("container")
        if container is not None:
            try:
                if container[0] is not None and container[0] in ax.lines:
                    container[0].remove()
                for cap in container[1]:
                    if cap in ax.lines:
                        cap.remove()
                for barcol in container[2]:
                    if barcol in ax.collections:
                        barcol.remove()
                if container in ax.containers:
                    ax.containers.remove(container)
            except (ValueError, AttributeError):
                pass

    def _redraw(self):
        """Clear and redraw based on current toggle states."""
        ax = self._group.axes

        try:
            self._clear_errorbar_artists()

            x = self._x_data
            y = self._y_data
            if x is None or y is None:
                return

            # Compute fmt/ls/lw based on toggles
            # Persist default so _store_errorbar_info() sees it
            if self._show_markers and not self._marker:
                self._marker = "o"
            marker = self._marker if self._show_markers else "none"
            ls = self._line_style if self._show_line else "none"
            lw = self._line_width if self._show_line else 0

            # Error bars: hide by zeroing elinewidth and capsize
            elinewidth = self._bar_lw if self._show_bars else 0
            capsize = self._cap_size if self._show_bars else 0

            kw = {}
            if self._has_yerr and self._yerr is not None:
                kw["yerr"] = self._yerr
            if self._has_xerr and self._xerr is not None:
                kw["xerr"] = self._xerr

            # Use line color as the base color for ax.errorbar();
            # override per-section colors/alphas on the artists afterwards.
            base_color = self._line_color if self._show_line \
                else self._marker_color if self._show_markers \
                else self._bar_color

            container = ax.errorbar(
                x, y,
                fmt='',
                marker=marker,
                color=base_color,
                ecolor=self._bar_color,
                linewidth=lw,
                linestyle=ls,
                markersize=self._marker_size,
                capsize=capsize,
                elinewidth=elinewidth,
                label=self._label,
                **kw,
            )

            # Tag + apply per-section colors and alphas
            data_line = container[0]
            if data_line is not None:
                data_line._matplotly_errorbar = True
                if self._show_line:
                    data_line.set_color(self._line_color)
                    data_line.set_alpha(self._line_alpha)
                if self._show_markers:
                    data_line.set_markerfacecolor(self._marker_color)
                    data_line.set_markeredgecolor(self._marker_color)
                    # Marker alpha applied via RGBA on face/edge color
                    import matplotlib.colors as mcolors
                    mc_rgba = list(mcolors.to_rgba(self._marker_color))
                    mc_rgba[3] = self._marker_alpha
                    data_line.set_markerfacecolor(mc_rgba)
                    data_line.set_markeredgecolor(mc_rgba)
            for cap in container[1]:
                cap._matplotly_errorbar = True
                cap.set_color(self._bar_color)
                cap.set_alpha(self._bar_alpha)
            for barcol in container[2]:
                barcol._matplotly_errorbar = True
                barcol.set_color(self._bar_color)
                barcol.set_alpha(self._bar_alpha)

            # Store new container reference
            self._group.metadata["container"] = container

            # Shaded region
            if self._show_shaded and self._has_yerr and self._yerr is not None:
                yerr = self._yerr
                if np.ndim(yerr) == 1:
                    y_lo = y - yerr
                    y_hi = y + yerr
                else:
                    y_lo = y - yerr[0]
                    y_hi = y + yerr[1]

                fill = ax.fill_between(
                    x, y_lo, y_hi,
                    color=self._shade_color,
                    alpha=self._shade_alpha,
                )
                fill._matplotly_errorbar = True

            self._store_errorbar_info()
            _refresh_legend(ax)

        except Exception:
            import traceback, sys
            traceback.print_exc(file=sys.stderr)

        self._canvas.force_redraw()

    # ------------------------------------------------------------------
    # Info storage for code generation
    # ------------------------------------------------------------------

    def _store_errorbar_info(self):
        """Store errorbar info on axes for code generation."""
        ax = self._group.axes
        info = {
            "_group_id": id(self._group),
            "show_bars": self._show_bars,
            "show_markers": self._show_markers,
            "show_line": self._show_line,
            "show_shaded": self._show_shaded,
            "bar_color": self._bar_color,
            "marker_color": self._marker_color,
            "line_color": self._line_color,
            "shade_color": self._shade_color,
            "bar_alpha": self._bar_alpha,
            "marker_alpha": self._marker_alpha,
            "line_alpha": self._line_alpha,
            "shade_alpha": self._shade_alpha,
            "line_width": self._line_width,
            "line_style": self._line_style,
            "bar_lw": self._bar_lw,
            "cap_size": self._cap_size,
            "marker": self._marker,
            "marker_size": self._marker_size,
            "has_yerr": self._has_yerr,
            "has_xerr": self._has_xerr,
            "label": self._label,
        }

        if not hasattr(ax, '_matplotly_errorbar_info'):
            ax._matplotly_errorbar_info = []

        # Replace or append
        replaced = False
        for i, existing in enumerate(ax._matplotly_errorbar_info):
            if existing.get("_group_id") == id(self._group):
                ax._matplotly_errorbar_info[i] = info
                replaced = True
                break
        if not replaced:
            ax._matplotly_errorbar_info.append(info)
