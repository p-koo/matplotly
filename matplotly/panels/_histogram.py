"""Histogram controls — full-featured panel with re-binning, histtype, mode, CDF.

Structural parameters (bins, type, mode, cumulative, orientation) are shared
across all histograms on the same axes via HistogramSharedPanel.
Per-histogram panels control visual properties (name, color, edge, alpha).
"""
from __future__ import annotations

import numpy as np
import ipywidgets as widgets
from matplotlib.colors import to_hex

from .._commands import BatchCommand, Command
from .._types import ArtistGroup
from ._base import ArtistPanel
from ._color_utils import (
    _DW, _NW, _SN,
    _get_palette_colors, _make_color_dot, _refresh_legend, _slider_num,
)


class HistogramPanel(ArtistPanel):
    """Per-histogram panel: name, color, edge color, edge width, alpha."""

    _plot_number: int = 0
    _on_label_changed = None

    def build(self) -> widgets.Widget:
        patches = self._group.artists
        ref = patches[0]
        meta = self._group.metadata

        # Current state
        self._raw_data = meta.get("raw_data", np.array([0.0]))
        self._bins = len(patches)
        self._histtype = "bar"
        self._mode = "count"
        self._cumulative = False
        self._orientation = meta.get("orientation", "vertical")
        self._zorder = meta.get("zorder", ref.get_zorder())

        try:
            self._color = to_hex(ref.get_facecolor())
        except Exception:
            self._color = "#1f77b4"

        try:
            self._edgecolor = to_hex(ref.get_edgecolor())
        except Exception:
            self._edgecolor = self._color

        self._edge_width = ref.get_linewidth() if hasattr(ref, 'get_linewidth') else 1.0
        alpha = ref.get_alpha()
        self._alpha = alpha if alpha is not None else 0.7
        self._hatch = ref.get_hatch() or ""
        self._fill = ref.get_fill()
        self._linestyle = ref.get_linestyle() or "-"

        label = self._group.label
        if label.startswith("Histogram: "):
            label = label[len("Histogram: "):]
        self._label = label

        # --- Collapsible header ---
        num = self._plot_number or ""
        header_prefix = f"Hist {num}: " if num else ""
        self._header_prefix = header_prefix

        self._color_indicator = widgets.HTML(_make_color_dot(self._color))

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

        # --- Controls (collapsed by default) ---
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
            toggle_btn.description = f"  {self._header_prefix}{self._label}"
        toggle_btn.on_click(_toggle)

        return widgets.VBox(
            [header_row, controls_box],
            layout=widgets.Layout(
                border='1px solid #ddd', border_radius='4px',
                margin='2px 0', padding='2px'))

    def _build_controls(self):
        """Build per-histogram visual controls only."""
        controls = []

        # --- Name ---
        name_field = widgets.Text(
            value=self._label, description="Name:",
            style={"description_width": _DW},
            layout=widgets.Layout(width="95%"))

        def _on_name(change):
            self._label = change["new"]
            container = self._group.metadata.get("container")
            if container is not None:
                container.set_label(self._label)
            if self._group.artists:
                self._group.artists[0].set_label(self._label)
            pfx = self._header_prefix
            icon = "chevron-down" if self._is_expanded else "chevron-right"
            self._toggle_btn.icon = icon
            self._toggle_btn.description = f"  {pfx}{self._label}"
            _refresh_legend(self._group.axes)
            if self._on_label_changed is not None:
                self._on_label_changed()
            self._canvas.force_redraw()
        name_field.observe(_on_name, names="value")
        controls.append(name_field)

        # --- Face color ---
        controls.append(self._build_color_section(
            "Color:", self._color, is_face=True))

        # --- Edge color ---
        controls.append(self._build_color_section(
            "Edge:", self._edgecolor, is_face=False))

        # --- Edge width ---
        edge_w_sl = widgets.FloatSlider(
            value=round(self._edge_width, 2), min=0, max=5, step=0.1,
            description="Edge w:", style=_SN)

        def _ew_cb(change):
            self._edge_width = change["new"]
            for p in self._group.artists:
                p.set_linewidth(self._edge_width)
            _refresh_legend(self._group.axes)
            self._canvas.force_redraw()
        edge_w_sl.observe(_ew_cb, names="value")
        controls.append(_slider_num(edge_w_sl))

        # --- Alpha ---
        alpha_sl = widgets.FloatSlider(
            value=round(self._alpha, 2), min=0, max=1, step=0.05,
            description="Alpha:", style=_SN)

        def _alpha_cb(change):
            self._alpha = change["new"]
            for p in self._group.artists:
                p.set_alpha(self._alpha)
            _refresh_legend(self._group.axes)
            self._canvas.force_redraw()
        alpha_sl.observe(_alpha_cb, names="value")
        controls.append(_slider_num(alpha_sl))

        # --- Hatch ---
        hatches = [
            ("none", ""), ("/ / /", "/"), ("\\ \\ \\", "\\"),
            ("| | |", "|"), ("- - -", "-"), ("+ + +", "+"),
            ("x x x", "x"), ("o o o", "o"), ("O O O", "O"),
            (". . .", "."), ("* * *", "*"),
            ("// //", "//"), ("xx xx", "xx"),
        ]
        cur_hatch = self._hatch
        if cur_hatch not in [v for _, v in hatches]:
            cur_hatch = ""
        hatch_dd = widgets.Dropdown(
            options=hatches, value=cur_hatch, description="Hatch:",
            style=_SN, layout=widgets.Layout(width="180px"))

        def _hatch_cb(change):
            self._hatch = change["new"]
            for p in self._group.artists:
                p.set_hatch(self._hatch)
            _refresh_legend(self._group.axes)
            self._canvas.force_redraw()
        hatch_dd.observe(_hatch_cb, names="value")
        controls.append(hatch_dd)

        # --- Linestyle ---
        linestyles = [("solid", "-"), ("dashed", "--"),
                      ("dotted", ":"), ("dashdot", "-.")]
        _ls_map = {"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-."}
        cur_ls = _ls_map.get(self._linestyle, self._linestyle)
        if cur_ls not in [v for _, v in linestyles]:
            cur_ls = "-"
        ls_dd = widgets.Dropdown(
            options=linestyles, value=cur_ls, description="Style:",
            style=_SN, layout=widgets.Layout(width="180px"))

        def _ls_cb(change):
            self._linestyle = change["new"]
            for p in self._group.artists:
                p.set_linestyle(self._linestyle)
            _refresh_legend(self._group.axes)
            self._canvas.force_redraw()
        ls_dd.observe(_ls_cb, names="value")
        controls.append(ls_dd)

        # --- Fill toggle ---
        fill_cb = widgets.Checkbox(
            value=self._fill, description="Fill",
            style={"description_width": "auto"},
            indent=False)

        def _fill_cb(change):
            self._fill = change["new"]
            for p in self._group.artists:
                p.set_fill(self._fill)
            _refresh_legend(self._group.axes)
            self._canvas.force_redraw()
        fill_cb.observe(_fill_cb, names="value")
        controls.append(fill_cb)

        return controls

    # --- Shared-param redraw (called by HistogramSharedPanel) ----------------

    def redraw_with_shared(self, bins, histtype, mode, cumulative, orientation,
                           bin_edges=None):
        """Update structural params and redraw. Called by HistogramSharedPanel."""
        self._bins = bins
        self._histtype = histtype
        self._mode = mode
        self._cumulative = cumulative
        self._orientation = orientation
        self._redraw_histogram(skip_redraw=True, bin_edges=bin_edges)

    def _redraw_histogram(self, skip_redraw=False, bin_edges=None):
        """Remove old patches and re-create histogram with current parameters."""
        from matplotlib.container import BarContainer
        ax = self._group.axes
        raw_data = self._raw_data

        # Remove old patches via Artist.remove()
        owned = {id(p) for p in self._group.artists}
        for p in list(self._group.artists):
            p.remove()
        self._group.artists = []

        # Clean up BarContainers that held our patches
        ax.containers[:] = [
            c for c in ax.containers
            if not (isinstance(c, BarContainer)
                    and any(id(p) in owned for p in c))
        ]

        # Clear stale data limits so the new hist sets them fresh
        ax.stale = True
        ax.relim()

        # Build hist kwargs
        density = self._mode == "density"
        weights = None
        if self._mode == "probability":
            weights = np.ones_like(raw_data) / len(raw_data)
            density = False

        actual_bins = bin_edges if bin_edges is not None else self._bins
        n, bins, patches = ax.hist(
            raw_data,
            bins=actual_bins,
            histtype=self._histtype,
            density=density,
            weights=weights,
            cumulative=self._cumulative,
            orientation=self._orientation,
            color=self._color,
            edgecolor=self._edgecolor,
            linewidth=self._edge_width,
            alpha=self._alpha,
            label=self._label,
            zorder=self._zorder,
        )

        # Tag new container so code gen can identify histogram patches
        if isinstance(patches, BarContainer):
            patches._matplotly_is_histogram = True

        # Rescale axes to fit new data range
        ax.relim()
        ax.autoscale_view()

        # Update artist references
        if isinstance(patches, list):
            new_artists = list(patches)
        else:
            new_artists = [p for p in patches]
        self._group.artists = new_artists

        # Tag new patches for code gen and re-apply visual properties
        for p in new_artists:
            p._matplotly_is_hist_patch = True
            if self._hatch:
                p.set_hatch(self._hatch)
            if self._linestyle not in ("-", "solid"):
                p.set_linestyle(self._linestyle)
            if not self._fill:
                p.set_fill(False)

        # Update metadata
        if isinstance(n, np.ndarray):
            self._group.metadata["bin_edges"] = bins.tolist()
            self._group.metadata["bin_heights"] = n.tolist()
        self._group.metadata["raw_data"] = raw_data

        # Store hist info on axes for code generation
        # Preserve original_patch_count from initial setup
        _orig_count = None
        for _old in getattr(ax, '_matplotly_hist_info', []):
            if _old.get("_group_id") == id(self._group):
                _orig_count = _old.get("original_patch_count")
                break
        hist_info = {
            "artists": new_artists,
            "original_patch_count": _orig_count or len(new_artists),
            "raw_data": raw_data,
            "bins": self._bins,
            "histtype": self._histtype,
            "mode": self._mode,
            "density": density,
            "cumulative": self._cumulative,
            "orientation": self._orientation,
            "color": self._color,
            "edgecolor": self._edgecolor,
            "linewidth": self._edge_width,
            "alpha": self._alpha,
            "label": self._label,
            "zorder": self._zorder,
            "hatch": self._hatch,
            "fill": self._fill,
            "linestyle": self._linestyle,
        }
        if not hasattr(ax, '_matplotly_hist_info'):
            ax._matplotly_hist_info = []
        replaced = False
        for i, info in enumerate(ax._matplotly_hist_info):
            if info.get("_group_id") == id(self._group):
                hist_info["_group_id"] = id(self._group)
                ax._matplotly_hist_info[i] = hist_info
                replaced = True
                break
        if not replaced:
            hist_info["_group_id"] = id(self._group)
            ax._matplotly_hist_info.append(hist_info)

        _refresh_legend(ax)
        if not skip_redraw:
            self._canvas.force_redraw()

    def _store_hist_info(self):
        """Store histogram info on axes for code generation (initial setup)."""
        ax = self._group.axes
        # Tag individual patches so code gen can reliably skip them
        for p in self._group.artists:
            p._matplotly_is_hist_patch = True
        # Also tag the container
        container = self._group.metadata.get("container")
        if container is not None:
            container._matplotly_is_histogram = True
        hist_info = {
            "_group_id": id(self._group),
            "artists": self._group.artists,
            "original_patch_count": len(self._group.artists),
            "raw_data": self._raw_data,
            "bins": self._bins,
            "histtype": self._histtype,
            "mode": self._mode,
            "density": False,
            "cumulative": self._cumulative,
            "orientation": self._orientation,
            "color": self._color,
            "edgecolor": self._edgecolor,
            "linewidth": self._edge_width,
            "alpha": self._alpha,
            "label": self._label,
            "zorder": self._zorder,
            "hatch": self._hatch,
            "fill": self._fill,
            "linestyle": self._linestyle,
        }
        if not hasattr(ax, '_matplotly_hist_info'):
            ax._matplotly_hist_info = []
        ax._matplotly_hist_info.append(hist_info)

    def _update_hist_info(self):
        """Update the hist_info entry on axes (called after merge/visual change)."""
        ax = self._group.axes
        hist_info = {
            "_group_id": id(self._group),
            "artists": self._group.artists,
            "original_patch_count": None,
            "raw_data": self._raw_data,
            "bins": self._bins,
            "histtype": self._histtype,
            "mode": self._mode,
            "density": self._mode == "density",
            "cumulative": self._cumulative,
            "orientation": self._orientation,
            "color": self._color,
            "edgecolor": self._edgecolor,
            "linewidth": self._edge_width,
            "alpha": self._alpha,
            "label": self._label,
            "zorder": self._zorder,
            "hatch": self._hatch,
            "fill": self._fill,
            "linestyle": self._linestyle,
        }
        # Preserve original_patch_count from initial setup
        for existing in getattr(ax, '_matplotly_hist_info', []):
            if existing.get("_group_id") == id(self._group):
                hist_info["original_patch_count"] = existing.get(
                    "original_patch_count")
                break
        if not hasattr(ax, '_matplotly_hist_info'):
            ax._matplotly_hist_info = []
        replaced = False
        for i, info in enumerate(ax._matplotly_hist_info):
            if info.get("_group_id") == id(self._group):
                ax._matplotly_hist_info[i] = hist_info
                replaced = True
                break
        if not replaced:
            ax._matplotly_hist_info.append(hist_info)

    # --- Color section (reused for face + edge) ------------------------------

    def _build_color_section(self, label_text, current_color, is_face):
        """Color section with swatch palette, matching line/scatter pattern."""

        color_btn = widgets.Button(
            layout=widgets.Layout(width='28px', height='28px',
                                  padding='0', min_width='28px'),
            tooltip="Click to choose color")
        color_btn.style.button_color = current_color

        color_row = widgets.HBox(
            [widgets.Label(label_text, layout=widgets.Layout(width='42px')),
             color_btn],
            layout=widgets.Layout(align_items='center', gap='4px'))

        _cmap_name = ["tab10"]

        def _make_swatches(colors):
            btns = []
            for c in colors:
                b = widgets.Button(
                    layout=widgets.Layout(width="18px", height="16px",
                                          padding="0", margin="1px",
                                          min_width="18px"))
                b.style.button_color = c
                btns.append(b)
            return btns

        colors_10 = _get_palette_colors("tab10", 10)
        swatch_buttons = _make_swatches(colors_10)
        colors_20 = _get_palette_colors("tab10", 20)
        extra_buttons = _make_swatches(colors_20[10:])

        _icon_css = widgets.HTML(
            '<style>'
            '.pb-swatch-btn button {'
            '  padding:0 !important;'
            '  min-width:0 !important;'
            '  overflow:hidden !important;'
            '}'
            '.pb-swatch-btn .fa {'
            '  font-size:9px !important;'
            '  position:relative !important;'
            '  top:-7px !important;'
            '}'
            '</style>')

        expand_btn = widgets.Button(
            icon="plus", tooltip="Show more colors",
            layout=widgets.Layout(width="18px", height="16px",
                                  padding="0", min_width="18px",
                                  margin="1px"))
        expand_btn.style.button_color = "#e0e0e0"
        expand_btn.add_class("pb-swatch-btn")

        palette_btn = widgets.Button(
            icon="paint-brush", tooltip="Custom color...",
            layout=widgets.Layout(width="18px", height="16px",
                                  padding="0", min_width="18px",
                                  margin="1px"))
        palette_btn.style.button_color = "#e8e8e8"
        palette_btn.add_class("pb-swatch-btn")

        _picker_cls = f"pb-picker-hist-{id(self)}-{'face' if is_face else 'edge'}"
        picker = widgets.ColorPicker(
            value=current_color, concise=True,
            layout=widgets.Layout(width="1px", height="1px",
                                  overflow="hidden", padding="0",
                                  margin="0", border="0"))
        picker.add_class(_picker_cls)

        _js_out = widgets.Output(
            layout=widgets.Layout(height="0px", overflow="hidden"))

        def _on_palette_btn(b):
            with _js_out:
                _js_out.clear_output()
                from IPython.display import display as ipy_display, Javascript
                ipy_display(Javascript(
                    "setTimeout(function(){"
                    "var el=document.querySelector('.%s input[type=\"color\"]');"
                    "if(el)el.click();"
                    "},150);" % _picker_cls))
        palette_btn.on_click(_on_palette_btn)

        extra_row = widgets.HBox(
            extra_buttons,
            layout=widgets.Layout(display='none', padding='1px 0 0 0',
                                  align_items='center', gap='1px'))

        main_row = widgets.HBox(
            swatch_buttons + [expand_btn, palette_btn, picker, _icon_css,
                              _js_out],
            layout=widgets.Layout(align_items='center', gap='1px'))

        palette_panel = widgets.VBox(
            [main_row, extra_row],
            layout=widgets.Layout(display='none', padding='2px 0 0 0'))

        _updating = [False]

        def _sync_controls(hex_val):
            _updating[0] = True
            try:
                color_btn.style.button_color = hex_val
                picker.value = hex_val
                if is_face:
                    self._color_indicator.value = _make_color_dot(hex_val)
            finally:
                _updating[0] = False

        def _apply_color(hex_val):
            if is_face:
                self._color = hex_val
                for p in self._group.artists:
                    p.set_facecolor(hex_val)
            else:
                self._edgecolor = hex_val
                for p in self._group.artists:
                    p.set_edgecolor(hex_val)
            _refresh_legend(self._group.axes)
            self._canvas.force_redraw()

        def _wire_swatch(btn):
            def _on_swatch(b, _btn=btn):
                c = _btn.style.button_color
                _sync_controls(c)
                _apply_color(c)
            btn.on_click(_on_swatch)
        for b in swatch_buttons + extra_buttons:
            _wire_swatch(b)

        def _on_expand(b):
            cname = _cmap_name[0]
            if extra_row.layout.display == 'none':
                c20 = _get_palette_colors(cname, 20)
                for i, btn in enumerate(swatch_buttons):
                    btn.style.button_color = c20[i]
                for i, btn in enumerate(extra_buttons):
                    btn.style.button_color = c20[10 + i]
                extra_row.layout.display = ''
                expand_btn.icon = 'minus'
            else:
                c10 = _get_palette_colors(cname, 10)
                for i, btn in enumerate(swatch_buttons):
                    btn.style.button_color = c10[i]
                extra_row.layout.display = 'none'
                expand_btn.icon = 'plus'
        expand_btn.on_click(_on_expand)

        def _from_picker(change):
            if _updating[0]:
                return
            _sync_controls(change["new"])
            _apply_color(change["new"])
        picker.observe(_from_picker, names="value")

        def _toggle_palette(btn):
            if palette_panel.layout.display == 'none':
                palette_panel.layout.display = ''
            else:
                palette_panel.layout.display = 'none'
        color_btn.on_click(_toggle_palette)

        # External hooks for ColormapPanel
        if is_face:
            self._update_color = _sync_controls

            def _ext_update_palette(cmap_name):
                _cmap_name[0] = cmap_name
                is_expanded = extra_row.layout.display != 'none'
                if is_expanded:
                    c20 = _get_palette_colors(cmap_name, 20)
                    for i, btn in enumerate(swatch_buttons):
                        btn.style.button_color = c20[i]
                    for i, btn in enumerate(extra_buttons):
                        btn.style.button_color = c20[10 + i]
                else:
                    c10 = _get_palette_colors(cmap_name, 10)
                    for i, btn in enumerate(swatch_buttons):
                        btn.style.button_color = c10[i]
                    c20 = _get_palette_colors(cmap_name, 20)
                    for i, btn in enumerate(extra_buttons):
                        btn.style.button_color = c20[10 + i]
            self._update_palette = _ext_update_palette

        return widgets.VBox([color_row, palette_panel])


class HistogramSharedPanel:
    """Shared structural controls applied to all histograms on the same axes.

    Controls: bins, histtype, mode, cumulative, orientation, merge.
    When any value changes, all registered HistogramPanels are redrawn.
    """

    def __init__(self, panels: list[HistogramPanel], canvas):
        self._panels = panels
        self._canvas = canvas
        # Read initial values from the first panel
        ref = panels[0]
        self._bins = ref._bins
        self._histtype = ref._histtype
        self._mode = ref._mode
        self._cumulative = ref._cumulative
        self._orientation = ref._orientation
        self._merged = False
        self._rwidth = 0.8

    def build(self) -> widgets.Widget:
        controls = []

        # --- Bins slider ---
        bin_sl = widgets.IntSlider(
            value=self._bins, min=2, max=200, step=1,
            description="Bins:", style=_SN,
            continuous_update=False)

        def _bin_cb(change):
            self._bins = change["new"]
            self._apply_all()
        bin_sl.observe(_bin_cb, names="value")
        controls.append(_slider_num(bin_sl))

        # --- Histtype dropdown ---
        histtype_dd = widgets.Dropdown(
            options=[("bar", "bar"), ("step", "step"),
                     ("stepfilled", "stepfilled")],
            value=self._histtype, description="Type:",
            style=_SN, layout=widgets.Layout(width="180px"))

        def _ht_cb(change):
            self._histtype = change["new"]
            self._apply_all()
        histtype_dd.observe(_ht_cb, names="value")
        controls.append(histtype_dd)

        # --- Mode dropdown ---
        mode_dd = widgets.Dropdown(
            options=[("count", "count"), ("density", "density"),
                     ("probability", "probability")],
            value=self._mode, description="Mode:",
            style=_SN, layout=widgets.Layout(width="180px"))

        def _mode_cb(change):
            self._mode = change["new"]
            self._apply_all()
        mode_dd.observe(_mode_cb, names="value")
        controls.append(mode_dd)

        # --- Cumulative checkbox ---
        cum_cb = widgets.Checkbox(
            value=self._cumulative, description="Cumulative (CDF)",
            style={"description_width": "auto"},
            indent=False)

        def _cum_cb(change):
            self._cumulative = change["new"]
            self._apply_all()
        cum_cb.observe(_cum_cb, names="value")
        controls.append(cum_cb)

        # --- Orientation dropdown ---
        orient_dd = widgets.Dropdown(
            options=[("vertical", "vertical"), ("horizontal", "horizontal")],
            value=self._orientation, description="Orient:",
            style=_SN, layout=widgets.Layout(width="180px"))

        def _orient_cb(change):
            self._orientation = change["new"]
            self._apply_all()
        orient_dd.observe(_orient_cb, names="value")
        controls.append(orient_dd)

        # --- Merge (only shown when multiple histograms) ---
        if len(self._panels) > 1:
            merge_cb = widgets.Checkbox(
                value=False, description="Merge",
                style={"description_width": "auto"},
                indent=False)

            rwidth_sl = widgets.FloatSlider(
                value=self._rwidth, min=0.1, max=1.0, step=0.05,
                description="Bar width:", style=_SN,
                continuous_update=False)
            rwidth_row = _slider_num(rwidth_sl)
            rwidth_box = widgets.VBox(
                [rwidth_row],
                layout=widgets.Layout(display='none'))

            def _merge_cb(change):
                self._merged = change["new"]
                if self._merged:
                    rwidth_box.layout.display = ''
                    self._merge_histograms()
                else:
                    rwidth_box.layout.display = 'none'
                    self._unmerge_histograms()
            merge_cb.observe(_merge_cb, names="value")

            def _rw_cb(change):
                self._rwidth = change["new"]
                if self._merged:
                    self._merge_histograms()
            rwidth_sl.observe(_rw_cb, names="value")

            controls.append(merge_cb)
            controls.append(rwidth_box)

        return widgets.VBox(
            controls,
            layout=widgets.Layout(padding='4px 4px 4px 8px'))

    def _apply_all(self):
        """Redraw every histogram panel with the current shared parameters."""
        if self._merged:
            self._merge_histograms()
        else:
            # Compute unified bin edges from all panels' data so bins align
            all_raw = np.concatenate([p._raw_data for p in self._panels])
            unified_edges = np.histogram_bin_edges(all_raw, bins=self._bins)
            for panel in self._panels:
                panel.redraw_with_shared(
                    self._bins, self._histtype, self._mode,
                    self._cumulative, self._orientation,
                    bin_edges=unified_edges)
            self._canvas.force_redraw()

    def _clear_all_hist_patches(self):
        """Remove every histogram patch and its BarContainer from the axes.

        Uses ax.patches + ax.containers as source of truth rather than
        relying solely on panel._group.artists which can become stale.
        """
        from matplotlib.container import BarContainer
        ax = self._panels[0]._group.axes

        # Collect ALL patches that belong to ANY BarContainer on the axes
        bar_patch_ids = set()
        for c in ax.containers:
            if isinstance(c, BarContainer):
                for p in c:
                    bar_patch_ids.add(id(p))

        # Also include patches currently tracked by panels
        for panel in self._panels:
            for p in panel._group.artists:
                bar_patch_ids.add(id(p))

        # Remove all identified patches from axes
        for p in list(ax.patches):
            if id(p) in bar_patch_ids:
                p.remove()

        # Clear all BarContainers
        ax.containers[:] = [
            c for c in ax.containers
            if not isinstance(c, BarContainer)
        ]

        # Reset panel artist lists
        for panel in self._panels:
            panel._group.artists = []

        ax.stale = True
        ax.relim()

    def _merge_histograms(self):
        """Merge all histograms into a single ax.hist() call with grouped bars."""
        import sys
        ax = self._panels[0]._group.axes

        try:
            self._clear_all_hist_patches()

            # Collect data and per-dataset styles
            all_data = [p._raw_data for p in self._panels]
            colors = [p._color for p in self._panels]
            edgecolors = [p._edgecolor for p in self._panels]
            labels = [p._label for p in self._panels]

            # Build kwargs
            density = self._mode == "density"
            weights = None
            if self._mode == "probability":
                weights = [np.ones(len(d)) / len(d) for d in all_data]
                density = False

            # ax.hist() doesn't distribute edgecolor like color;
            # use single value, then apply per-container if they differ
            unique_ecs = list(set(edgecolors))
            ec_single = unique_ecs[0] if len(unique_ecs) == 1 else 'black'

            n, bins, patches_list = ax.hist(
                all_data,
                bins=self._bins,
                histtype=self._histtype,
                density=density,
                weights=weights,
                cumulative=self._cumulative,
                orientation=self._orientation,
                color=colors,
                edgecolor=ec_single,
                linewidth=self._panels[0]._edge_width,
                alpha=self._panels[0]._alpha,
                label=labels,
                rwidth=self._rwidth,
            )

            ax.relim()
            ax.autoscale_view()

            # Set merge flag on axes (backup for code gen)
            ax._matplotly_hist_merged = True
            ax._matplotly_hist_rwidth = self._rwidth

            # Mark new containers as histograms for code gen detection
            from matplotlib.container import BarContainer
            for c in ax.containers:
                if isinstance(c, BarContainer):
                    c._matplotly_is_histogram = True

            # Update each panel's artists with the corresponding patch group
            for i, panel in enumerate(self._panels):
                if isinstance(patches_list, list) and i < len(patches_list):
                    new_artists = list(patches_list[i])
                else:
                    new_artists = list(patches_list)
                panel._group.artists = new_artists

                # Re-apply per-dataset visual properties
                for p in new_artists:
                    p.set_edgecolor(panel._edgecolor)
                    p.set_alpha(panel._alpha)
                    p.set_linewidth(panel._edge_width)
                    if panel._hatch:
                        p.set_hatch(panel._hatch)
                    if panel._linestyle not in ("-", "solid"):
                        p.set_linestyle(panel._linestyle)
                    if not panel._fill:
                        p.set_fill(False)

                # Update hist info
                try:
                    panel._update_hist_info()
                except Exception as e:
                    print(f"[matplotly] _update_hist_info error: {e}",
                          file=sys.stderr)

            # Store merge state INSIDE the info dicts (primary source
            # for code gen — more reliable than ax-level flags)
            for info in getattr(ax, '_matplotly_hist_info', []):
                info['merged'] = True
                info['rwidth'] = self._rwidth

            try:
                _refresh_legend(ax)
            except Exception as e:
                print(f"[matplotly] _refresh_legend error: {e}",
                      file=sys.stderr)
        except Exception as exc:
            import traceback
            traceback.print_exc(file=sys.stderr)

        self._canvas.force_redraw()

    def _unmerge_histograms(self):
        """Unmerge back to individual ax.hist() calls."""
        import sys
        ax = self._panels[0]._group.axes

        try:
            self._clear_all_hist_patches()
            ax._matplotly_hist_merged = False

            # Redraw each individually — _redraw_histogram expects artists=[],
            # which is fine since _clear_all_hist_patches set them to [].
            for panel in self._panels:
                panel._bins = self._bins
                panel._histtype = self._histtype
                panel._mode = self._mode
                panel._cumulative = self._cumulative
                panel._orientation = self._orientation
                panel._redraw_histogram(skip_redraw=True)

            # Clear merge state from info dicts
            for info in getattr(ax, '_matplotly_hist_info', []):
                info['merged'] = False

            _refresh_legend(ax)
        except Exception as exc:
            import traceback
            traceback.print_exc(file=sys.stderr)

        self._canvas.force_redraw()
