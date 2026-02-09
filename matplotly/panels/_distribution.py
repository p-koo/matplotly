"""Distribution plot controls — box, violin, jitter, and all combinations.

DistributionPanel: per-group visual controls (box, violin, jitter sections).
DistributionSharedPanel: structural controls shared across all distribution
    groups on the same axes (display mode, width, gap, orientation, ticks).
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
    _get_palette_colors, _make_color_dot, _refresh_legend, _slider_num,
)


# ---------------------------------------------------------------------------
# Display mode definitions
# ---------------------------------------------------------------------------

DISPLAY_MODES = [
    ("box", "box"),
    ("violin", "violin"),
    ("jitter", "jitter"),
    ("box+violin", "box+violin"),
    ("box+jitter", "box+jitter"),
    ("violin+jitter", "violin+jitter"),
    ("box+violin+jitter", "box+violin+jitter"),
]


def _mode_has_box(mode: str) -> bool:
    return "box" in mode


def _mode_has_violin(mode: str) -> bool:
    return "violin" in mode


def _mode_has_jitter(mode: str) -> bool:
    return "jitter" in mode


# ---------------------------------------------------------------------------
# DistributionPanel — per-group visual controls
# ---------------------------------------------------------------------------

class DistributionPanel(ArtistPanel):
    """Per-group distribution panel with box, violin, and jitter controls."""

    _plot_number: int = 0
    _on_label_changed = None
    _shared_panel = None  # set by DistributionSharedPanel

    def build(self) -> widgets.Widget:
        meta = self._group.metadata

        # Extract data from metadata
        if self._group.plot_type == PlotType.BOXPLOT:
            self._raw_data = meta.get("raw_data", [])
            self._positions = meta.get("positions", [])
            self._original_positions = list(self._positions)
            self._box_stats = meta.get("box_stats", [])
            self._orientation = meta.get("orientation", "vertical")
        else:
            # VIOLIN
            self._raw_data = meta.get("raw_data", [])
            self._positions = meta.get("positions", [])
            self._original_positions = list(self._positions)
            self._box_stats = []  # will compute from raw_data if needed
            self._orientation = meta.get("orientation", "vertical")

        # Extract label
        label = self._group.label
        for prefix in ("Box plot ", "Violin "):
            if label.startswith(prefix):
                label = label[len(prefix):]
                break
        self._label = label

        # Detect initial colors from artists
        self._detect_initial_colors()

        # Box properties
        self._box_lw = 1.0
        self._box_hatch = ""
        self._notch = False
        # Use detected median color if available, otherwise match edge color
        self._median_color = (self._detected_median_color
                              if self._detected_median_color
                              else self._box_edgecolor)
        self._median_lw = 2.0
        self._whisker_style = "-"
        self._whisker_lw = 1.0
        self._cap_width = 5.0
        self._show_mean = False
        self._mean_style = "--"
        self._flier_marker = ""
        self._flier_size = 6.0
        self._flier_color = self._box_edgecolor

        # Violin properties
        self._violin_edgecolor = self._box_edgecolor
        self._violin_alpha = 0.3
        self._violin_inner = "box"

        # Jitter properties
        self._jitter_alpha = 0.5
        self._jitter_size = 3.0
        self._jitter_spread = 0.2
        self._jitter_marker = "o"

        # --- Collapsible header ---
        num = self._plot_number or ""
        header_prefix = f"Dist {num}: " if num else ""
        self._header_prefix = header_prefix

        self._color_indicator = widgets.HTML(_make_color_dot(self._box_color))

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

    def _detect_initial_colors(self):
        """Detect initial colors from the existing artists."""
        from matplotlib.lines import Line2D

        default = "#1f77b4"
        self._box_color = default
        self._box_edgecolor = "#000000"
        self._box_alpha = 1.0
        self._violin_color = default
        self._jitter_color = default
        self._detected_median_color = None  # set only if detected

        meta = self._group.metadata

        if self._group.plot_type == PlotType.BOXPLOT:
            box_patches = meta.get("box_patches", [])
            if box_patches:
                try:
                    self._box_color = to_hex(box_patches[0].get_facecolor())
                except Exception:
                    pass
                try:
                    self._box_edgecolor = to_hex(box_patches[0].get_edgecolor())
                except Exception:
                    pass

            # Detect median color from Line2D artists
            # In boxplots, median lines are horizontal (vertical mode) or
            # vertical (horizontal mode) at the median position.
            box_stats = meta.get("box_stats", [])
            if box_stats:
                med_val = box_stats[0].get("med") or box_stats[0].get("median")
                if med_val is not None:
                    orientation = meta.get("orientation", "vertical")
                    artists = self._group.artists
                    for a in artists:
                        if not isinstance(a, Line2D):
                            continue
                        try:
                            yd = a.get_ydata()
                            xd = a.get_xdata()
                            if orientation == "vertical":
                                # Median: horizontal line at y=med_val
                                if (len(yd) == 2 and abs(yd[0] - med_val) < 0.01
                                        and abs(yd[1] - med_val) < 0.01):
                                    self._detected_median_color = to_hex(a.get_color())
                                    break
                            else:
                                # Median: vertical line at x=med_val
                                if (len(xd) == 2 and abs(xd[0] - med_val) < 0.01
                                        and abs(xd[1] - med_val) < 0.01):
                                    self._detected_median_color = to_hex(a.get_color())
                                    break
                        except Exception:
                            pass

        elif self._group.plot_type == PlotType.VIOLIN:
            bodies = meta.get("bodies", [])
            if bodies:
                try:
                    fc = bodies[0].get_facecolor()
                    if len(fc) > 0:
                        self._violin_color = to_hex(fc[0])
                        self._box_color = self._violin_color
                except Exception:
                    pass
            # Detect stat line colors (median/whisker/bar lines)
            stat_lines = meta.get("stat_lines", [])
            for sl in stat_lines:
                try:
                    colors = sl.get_colors()
                    if len(colors) > 0:
                        detected = to_hex(colors[0])
                        self._box_edgecolor = detected
                        break
                except Exception:
                    pass

        self._violin_color = self._box_color
        self._jitter_color = self._box_color

    def _build_controls(self):
        """Build all control sections."""
        controls = []

        # --- Name ---
        name_field = widgets.Text(
            value=self._label, description="Name:",
            style={"description_width": _DW},
            layout=widgets.Layout(width="95%"))

        def _on_name(change):
            self._label = change["new"]
            pfx = self._header_prefix
            icon = "chevron-down" if self._is_expanded else "chevron-right"
            self._toggle_btn.icon = icon
            self._toggle_btn.description = f"  {pfx}{self._label}"
            if self._on_label_changed is not None:
                self._on_label_changed()
            self._trigger_redraw()
        name_field.observe(_on_name, names="value")
        controls.append(name_field)

        # --- Box section ---
        self._box_section = self._build_box_section()
        controls.append(self._box_section)

        # --- Violin section ---
        self._violin_section = self._build_violin_section()
        controls.append(self._violin_section)

        # --- Jitter section ---
        self._jitter_section = self._build_jitter_section()
        controls.append(self._jitter_section)

        return controls

    def _build_box_section(self):
        """Box visual controls."""
        parts = []

        header = widgets.HTML("<b style='font-size:11px;color:#555'>Box</b>")
        parts.append(header)

        # Color
        parts.append(self._build_color_section(
            "Color:", self._box_color, "box_color"))

        # Edge color
        parts.append(self._build_color_section(
            "Edge:", self._box_edgecolor, "box_edgecolor"))

        # Edge width
        ew_sl = widgets.FloatSlider(
            value=self._box_lw, min=0, max=5, step=0.1,
            description="Edge w:", style=_SN)
        def _ew_cb(change):
            self._box_lw = change["new"]
            self._trigger_redraw()
        ew_sl.observe(_ew_cb, names="value")
        sn = _slider_num(ew_sl, "54px")
        sn.layout.width = "95%"
        parts.append(sn)

        # Alpha
        alpha_sl = widgets.FloatSlider(
            value=self._box_alpha, min=0, max=1, step=0.05,
            description="Alpha:", style=_SN)
        def _alpha_cb(change):
            self._box_alpha = change["new"]
            self._trigger_redraw()
        alpha_sl.observe(_alpha_cb, names="value")
        sn = _slider_num(alpha_sl)
        sn.layout.width = "95%"
        parts.append(sn)

        # Hatch
        hatches = [
            ("none", ""), ("/ / /", "/"), ("\\ \\ \\", "\\"),
            ("| | |", "|"), ("- - -", "-"), ("+ + +", "+"),
            ("x x x", "x"), ("o o o", "o"),
        ]
        hatch_dd = widgets.Dropdown(
            options=hatches, value="", description="Hatch:",
            style=_SN, layout=widgets.Layout(width="180px"))
        def _hatch_cb(change):
            self._box_hatch = change["new"]
            self._trigger_redraw()
        hatch_dd.observe(_hatch_cb, names="value")
        parts.append(hatch_dd)

        return widgets.VBox(parts, layout=widgets.Layout(
            padding='4px 4px 4px 10px',
            border_left='3px solid #4a86c8'))

    def _build_violin_section(self):
        """Violin visual controls."""
        parts = []

        header = widgets.HTML(
            "<b style='font-size:11px;color:#555'>Violin</b>")
        parts.append(header)

        # Color
        parts.append(self._build_color_section(
            "Color:", self._violin_color, "violin_color"))

        # Edge color (full swatch palette, same as box edge)
        parts.append(self._build_color_section(
            "Edge:", self._violin_edgecolor, "violin_edgecolor"))

        # Alpha
        v_alpha = widgets.FloatSlider(
            value=self._violin_alpha, min=0, max=1, step=0.05,
            description="Alpha:", style=_SN)
        def _va_cb(change):
            self._violin_alpha = change["new"]
            self._trigger_redraw()
        v_alpha.observe(_va_cb, names="value")
        sn = _slider_num(v_alpha)
        sn.layout.width = "95%"
        parts.append(sn)

        # Inner style
        self._inner_dd = widgets.Dropdown(
            options=[("box", "box"), ("quartile", "quartile"),
                     ("stick", "stick"), ("none", "none")],
            value="box", description="Inner:",
            style=_SN, layout=widgets.Layout(width="180px"))
        def _inner_cb(change):
            self._violin_inner = change["new"]
            self._trigger_redraw()
        self._inner_dd.observe(_inner_cb, names="value")
        parts.append(self._inner_dd)

        return widgets.VBox(parts, layout=widgets.Layout(
            padding='4px 4px 4px 10px',
            border_left='3px solid #8e44ad'))

    def _build_jitter_section(self):
        """Jitter visual controls."""
        parts = []

        header = widgets.HTML(
            "<b style='font-size:11px;color:#555'>Jitter</b>")
        parts.append(header)

        # Color
        parts.append(self._build_color_section(
            "Color:", self._jitter_color, "jitter_color"))

        # Size
        j_size = widgets.FloatSlider(
            value=self._jitter_size, min=0.5, max=15, step=0.5,
            description="Size:", style=_SN)
        def _js_cb(change):
            self._jitter_size = change["new"]
            self._trigger_redraw()
        j_size.observe(_js_cb, names="value")
        sn = _slider_num(j_size)
        sn.layout.width = "95%"
        parts.append(sn)

        # Alpha
        j_alpha = widgets.FloatSlider(
            value=self._jitter_alpha, min=0, max=1, step=0.05,
            description="Alpha:", style=_SN)
        def _ja_cb(change):
            self._jitter_alpha = change["new"]
            self._trigger_redraw()
        j_alpha.observe(_ja_cb, names="value")
        sn = _slider_num(j_alpha)
        sn.layout.width = "95%"
        parts.append(sn)

        # Spread
        j_spread = widgets.FloatSlider(
            value=self._jitter_spread, min=0.0, max=0.5, step=0.02,
            description="Spread:", style=_SN)
        def _jspread_cb(change):
            self._jitter_spread = change["new"]
            self._trigger_redraw()
        j_spread.observe(_jspread_cb, names="value")
        sn = _slider_num(j_spread)
        sn.layout.width = "95%"
        parts.append(sn)

        # Marker
        markers = [("circle", "o"), ("square", "s"), ("triangle", "^"),
                   ("diamond", "D"), ("plus", "+"), ("x", "x"), ("star", "*")]
        j_marker = widgets.Dropdown(
            options=markers, value="o", description="Marker:",
            style=_SN, layout=widgets.Layout(width="150px"))
        def _jm_cb(change):
            self._jitter_marker = change["new"]
            self._trigger_redraw()
        j_marker.observe(_jm_cb, names="value")
        parts.append(j_marker)

        return widgets.VBox(parts, layout=widgets.Layout(
            padding='4px 4px 4px 10px',
            border_left='3px solid #27ae60'))

    def update_section_visibility(self, mode: str):
        """Show/hide box/violin/jitter sections based on display mode."""
        self._box_section.layout.display = (
            '' if _mode_has_box(mode) else 'none')
        self._violin_section.layout.display = (
            '' if _mode_has_violin(mode) else 'none')
        self._jitter_section.layout.display = (
            '' if _mode_has_jitter(mode) else 'none')

    def _trigger_redraw(self):
        """Trigger redraw through the shared panel."""
        self._store_dist_info()
        if self._shared_panel is not None:
            self._shared_panel._redraw()
        else:
            self._canvas.force_redraw()

    def _store_dist_info(self):
        """Store distribution info on axes for code generation."""
        ax = self._group.axes
        mode = "box"
        if self._shared_panel is not None:
            mode = self._shared_panel._mode

        width = 0.5
        if self._shared_panel is not None:
            width = self._shared_panel._width

        tick_ha = "center"
        tick_pad = 4.0
        tick_rotation = 0
        tick_labels = []
        tick_centers = list(self._positions)
        if self._shared_panel is not None:
            tick_ha = self._shared_panel._tick_ha
            tick_pad = self._shared_panel._tick_pad
            tick_rotation = self._shared_panel._tick_rotation
            tick_labels = list(self._shared_panel._tick_labels)
            tick_centers = list(self._shared_panel._tick_centers)

        dist_info = {
            "_group_id": id(self._group),
            "raw_data": self._raw_data,
            "positions": list(self._positions),
            "original_positions": list(self._original_positions),
            "label": self._label,
            "display_mode": mode,
            "orientation": self._orientation,
            "width": width,
            "tick_ha": tick_ha,
            "tick_pad": tick_pad,
            "tick_rotation": tick_rotation,
            "tick_labels": tick_labels,
            "tick_centers": tick_centers,
            # Box props
            "box_color": self._box_color,
            "box_edgecolor": self._box_edgecolor,
            "box_alpha": self._box_alpha,
            "box_lw": self._box_lw,
            "box_hatch": self._box_hatch,
            "notch": self._notch,
            "median_color": self._median_color,
            "median_lw": self._median_lw,
            "whisker_style": self._whisker_style,
            "whisker_lw": self._whisker_lw,
            "cap_width": self._cap_width,
            "show_mean": self._show_mean,
            "mean_style": self._mean_style,
            "flier_marker": self._flier_marker,
            "flier_size": self._flier_size,
            "flier_color": self._flier_color,
            # Violin props
            "violin_color": self._violin_color,
            "violin_edgecolor": self._violin_edgecolor,
            "violin_alpha": self._violin_alpha,
            "violin_inner": self._violin_inner,
            # Jitter props
            "jitter_color": self._jitter_color,
            "jitter_alpha": self._jitter_alpha,
            "jitter_size": self._jitter_size,
            "jitter_spread": self._jitter_spread,
            "jitter_marker": self._jitter_marker,
            # Original introspected box stats (avoids double reconstruction)
            "box_stats": list(self._box_stats) if self._box_stats else [],
        }

        if not hasattr(ax, '_matplotly_dist_info'):
            ax._matplotly_dist_info = []

        # Replace or append
        replaced = False
        for i, info in enumerate(ax._matplotly_dist_info):
            if info.get("_group_id") == id(self._group):
                ax._matplotly_dist_info[i] = dist_info
                replaced = True
                break
        if not replaced:
            ax._matplotly_dist_info.append(dist_info)

    # --- Color section helper ---

    def _build_color_section(self, label_text, current_color, attr_name):
        """Color section with swatch palette."""
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

        _picker_cls = f"pb-picker-dist-{id(self)}-{attr_name}"
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
                if attr_name == "box_color":
                    self._color_indicator.value = _make_color_dot(hex_val)
            finally:
                _updating[0] = False

        def _apply_color(hex_val):
            setattr(self, f"_{attr_name}", hex_val)
            self._trigger_redraw()

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

        # External hooks for ColormapPanel (only primary color)
        if attr_name == "box_color":
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


# ---------------------------------------------------------------------------
# DistributionSharedPanel — structural controls
# ---------------------------------------------------------------------------

class DistributionSharedPanel:
    """Shared structural controls for all distribution groups on the same axes.

    Controls: display mode, width, gap, orientation, tick labels, rotation.
    """

    def __init__(self, panels: list[DistributionPanel], canvas,
                 initial_mode: str = "box"):
        self._panels = panels
        self._canvas = canvas
        self._ax = panels[0]._group.axes
        self._mode = initial_mode
        self._width = 0.5
        self._gap = 0.0
        self._orientation = panels[0]._orientation

        # Wire panels to this shared panel
        for panel in panels:
            panel._shared_panel = self

        # Compute tick info
        n_groups = len(panels)
        ref = panels[0]
        self._n_ticks = len(ref._positions) if ref._positions else 0

        if n_groups > 1 and self._n_ticks > 0:
            all_pos = np.array([p._positions for p in panels
                                if len(p._positions) == self._n_ticks])
            if len(all_pos) > 0:
                self._tick_centers = np.mean(all_pos, axis=0).tolist()
            else:
                self._tick_centers = list(ref._positions)
        else:
            self._tick_centers = list(ref._positions)

        # Detect tick labels from axes
        if self._orientation == "vertical":
            labels = [t.get_text() for t in self._ax.get_xticklabels()]
        else:
            labels = [t.get_text() for t in self._ax.get_yticklabels()]
        if all(not t for t in labels):
            labels = [str(round(p, 2)) for p in self._tick_centers]
        self._tick_labels = labels[:self._n_ticks]
        self._tick_rotation = 0

        # Detect tick alignment + padding
        if self._orientation == "vertical":
            tick_objs = self._ax.get_xticklabels()
        else:
            tick_objs = self._ax.get_yticklabels()
        self._tick_ha = tick_objs[0].get_ha() if tick_objs else "center"
        self._tick_pad = 4.0

        # Detect initial width from positions
        if self._n_ticks >= 2 and n_groups == 1:
            spacing = abs(self._tick_centers[1] - self._tick_centers[0])
            self._width = min(spacing * 0.6, 0.8)

    def build(self) -> widgets.Widget:
        controls = []

        # --- Toggle section helper ---
        def _make_toggle_section(title, children, initially_open=True):
            box = widgets.VBox(children, layout=widgets.Layout(
                display='' if initially_open else 'none',
                padding='2px 0 2px 8px'))
            btn = widgets.Button(
                description=f"  {title}",
                icon="chevron-down" if initially_open else "chevron-right",
                layout=widgets.Layout(width='100%', height='24px'))
            btn.style.button_color = '#f5f5f5'
            def _toggle(b):
                if box.layout.display == 'none':
                    box.layout.display = ''
                    btn.icon = 'chevron-down'
                else:
                    box.layout.display = 'none'
                    btn.icon = 'chevron-right'
            btn.on_click(_toggle)
            return widgets.VBox([btn, box])

        # --- Swatch color section helper for shared controls ---
        def _build_shared_color(label_text, initial_color, on_change):
            """Build a color button + swatch palette that calls on_change(hex)."""
            color_btn = widgets.Button(
                layout=widgets.Layout(width='28px', height='28px',
                                      padding='0', min_width='28px'),
                tooltip="Click to choose color")
            color_btn.style.button_color = initial_color

            color_row = widgets.HBox(
                [widgets.Label(label_text,
                               layout=widgets.Layout(width='50px')),
                 color_btn],
                layout=widgets.Layout(align_items='center', gap='4px'))

            colors_10 = _get_palette_colors("tab10", 10)

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

            swatch_buttons = _make_swatches(colors_10)

            _pk_cls = f"pb-shared-{id(color_btn)}"
            picker = widgets.ColorPicker(
                value=initial_color, concise=True,
                layout=widgets.Layout(width="1px", height="1px",
                                      overflow="hidden", padding="0",
                                      margin="0", border="0"))
            picker.add_class(_pk_cls)

            palette_btn = widgets.Button(
                icon="paint-brush", tooltip="Custom color...",
                layout=widgets.Layout(width="18px", height="16px",
                                      padding="0", min_width="18px",
                                      margin="1px"))
            palette_btn.style.button_color = "#e8e8e8"
            palette_btn.add_class("pb-swatch-btn")

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
                        "},150);" % _pk_cls))
            palette_btn.on_click(_on_palette_btn)

            main_row = widgets.HBox(
                swatch_buttons + [palette_btn, picker, _js_out],
                layout=widgets.Layout(align_items='center', gap='1px'))

            palette_panel = widgets.VBox(
                [main_row],
                layout=widgets.Layout(display='none', padding='2px 0 0 0'))

            _updating = [False]

            def _sync(hex_val):
                _updating[0] = True
                try:
                    color_btn.style.button_color = hex_val
                    picker.value = hex_val
                finally:
                    _updating[0] = False

            def _apply(hex_val):
                _sync(hex_val)
                on_change(hex_val)

            for b in swatch_buttons:
                def _on_sw(btn, _b=b):
                    _apply(_b.style.button_color)
                b.on_click(_on_sw)

            def _from_picker(change):
                if _updating[0]:
                    return
                _apply(change["new"])
            picker.observe(_from_picker, names="value")

            def _toggle_palette(btn):
                if palette_panel.layout.display == 'none':
                    palette_panel.layout.display = ''
                else:
                    palette_panel.layout.display = 'none'
            color_btn.on_click(_toggle_palette)

            return widgets.VBox([color_row, palette_panel])

        # --- Display mode ---
        mode_dd = widgets.Dropdown(
            options=DISPLAY_MODES, value=self._mode,
            description="Plot Type:", style={"description_width": "64px"},
            layout=widgets.Layout(width="220px"))
        controls.append(mode_dd)

        # =============================================================
        # Structure section (width, gap, orientation)
        # =============================================================
        struct_parts = []

        width_sl = widgets.FloatSlider(
            value=round(self._width, 2), min=0.05, max=1.5, step=0.05,
            description="Width:", style=_SN,
            continuous_update=True)

        def _width_cb(change):
            self._width = change["new"]
            self._redraw()
        width_sl.observe(_width_cb, names="value")
        struct_parts.append(_slider_num(width_sl))

        if len(self._panels) > 1:
            gap_sl = widgets.FloatSlider(
                value=round(self._gap, 2), min=0.0, max=0.5, step=0.05,
                description="Gap:", style=_SN,
                continuous_update=True)

            def _gap_cb(change):
                self._gap = change["new"]
                self._redraw()
            gap_sl.observe(_gap_cb, names="value")
            struct_parts.append(_slider_num(gap_sl))

        orient_dd = widgets.Dropdown(
            options=[("vertical", "vertical"), ("horizontal", "horizontal")],
            value=self._orientation, description="Orient:",
            style=_SN, layout=widgets.Layout(width="180px"))

        def _orient_cb(change):
            self._orientation = change["new"]
            self._redraw()
        orient_dd.observe(_orient_cb, names="value")
        struct_parts.append(orient_dd)

        controls.append(_make_toggle_section("Structure", struct_parts))

        # =============================================================
        # Box Options section (shared across all groups)
        # =============================================================
        box_opt_parts = []

        # Notch + Show mean
        notch_cb = widgets.Checkbox(
            value=False, description="Notch",
            style={"description_width": "auto"}, indent=False)
        mean_cb = widgets.Checkbox(
            value=False, description="Show mean",
            style={"description_width": "auto"}, indent=False)

        def _notch_cb(change):
            for panel in self._panels:
                panel._notch = change["new"]
                panel._store_dist_info()
            self._redraw()
        notch_cb.observe(_notch_cb, names="value")

        def _mean_cb(change):
            for panel in self._panels:
                panel._show_mean = change["new"]
                panel._store_dist_info()
            self._redraw()
        mean_cb.observe(_mean_cb, names="value")

        box_opt_parts.append(widgets.HBox([notch_cb, mean_cb]))

        # No-fliers toggle
        no_flier_btn = widgets.ToggleButton(
            value=True, description="No fliers",
            icon="eye-slash",
            layout=widgets.Layout(width="120px", height="26px"),
            style={"description_width": "0px"})

        # Flier controls
        markers = [("circle", "o"), ("square", "s"), ("triangle", "^"),
                   ("diamond", "D"), ("plus", "+"), ("x", "x")]
        flier_marker_dd = widgets.Dropdown(
            options=markers, value="o", description="Flier:",
            style=_SN, layout=widgets.Layout(width="150px"))
        flier_size_sl = widgets.FloatSlider(
            value=6.0, min=0, max=15, step=0.5,
            description="Fl. sz:", style=_SN)
        def _fc_cb(hex_val):
            for panel in self._panels:
                panel._flier_color = hex_val
                panel._store_dist_info()
            self._redraw()
        flier_color_section = _build_shared_color(
            "Fl. clr:", self._panels[0]._flier_color if self._panels else "#000000", _fc_cb)

        def _fm_cb(change):
            for panel in self._panels:
                panel._flier_marker = change["new"]
                panel._store_dist_info()
            self._redraw()
        flier_marker_dd.observe(_fm_cb, names="value")

        def _fs_cb(change):
            for panel in self._panels:
                panel._flier_size = change["new"]
                panel._store_dist_info()
            self._redraw()
        flier_size_sl.observe(_fs_cb, names="value")

        flier_details = widgets.VBox(
            [flier_marker_dd, _slider_num(flier_size_sl), flier_color_section],
            layout=widgets.Layout(display='none'))

        def _no_flier_cb(change):
            if change["new"]:
                for panel in self._panels:
                    panel._flier_marker = ""
                    panel._store_dist_info()
                flier_details.layout.display = 'none'
                no_flier_btn.icon = "eye-slash"
            else:
                for panel in self._panels:
                    panel._flier_marker = flier_marker_dd.value
                    panel._store_dist_info()
                flier_details.layout.display = ''
                no_flier_btn.icon = "eye"
            self._redraw()
        no_flier_btn.observe(_no_flier_cb, names="value")

        box_opt_parts.append(no_flier_btn)
        box_opt_parts.append(flier_details)

        # Median color + lw
        def _mc_cb(hex_val):
            for panel in self._panels:
                panel._median_color = hex_val
                panel._store_dist_info()
            self._redraw()
        med_color_section = _build_shared_color(
            "Med clr:", self._panels[0]._median_color if self._panels else "#000000", _mc_cb)

        med_lw_sl = widgets.FloatSlider(
            value=2.0, min=0.5, max=6, step=0.25,
            description="Med lw:", style=_SN)

        def _mlw_cb(change):
            for panel in self._panels:
                panel._median_lw = change["new"]
                panel._store_dist_info()
            self._redraw()
        med_lw_sl.observe(_mlw_cb, names="value")

        box_opt_parts.append(med_color_section)
        box_opt_parts.append(_slider_num(med_lw_sl))

        # Whisker style
        ws_styles = [("solid", "-"), ("dashed", "--"),
                     ("dotted", ":"), ("dashdot", "-.")]
        whisk_dd = widgets.Dropdown(
            options=ws_styles, value="-", description="Whisk:",
            style=_SN, layout=widgets.Layout(width="150px"))

        def _ws_cb(change):
            for panel in self._panels:
                panel._whisker_style = change["new"]
                panel._store_dist_info()
            self._redraw()
        whisk_dd.observe(_ws_cb, names="value")
        box_opt_parts.append(whisk_dd)

        self._box_options_section = _make_toggle_section(
            "Box Options", box_opt_parts,
            initially_open=_mode_has_box(self._mode))
        # Hide entirely when mode has no box
        if not _mode_has_box(self._mode):
            self._box_options_section.layout.display = 'none'
        controls.append(self._box_options_section)

        # =============================================================
        # Ticks section
        # =============================================================
        tick_parts = []

        if self._n_ticks > 0:
            tick_widgets = []
            self._tick_fields = []
            for k in range(self._n_ticks):
                lbl = self._tick_labels[k] if k < len(self._tick_labels) else ""
                tw = widgets.Text(
                    value=lbl, layout=widgets.Layout(width="70px"))
                self._tick_fields.append(tw)

                def _tick_cb(change, idx=k):
                    if idx < len(self._tick_labels):
                        self._tick_labels[idx] = change["new"]
                    self._apply_tick_labels()
                tw.observe(_tick_cb, names="value")
                tick_widgets.append(tw)

            ticks_row = widgets.HBox(
                [widgets.Label("Ticks:", layout=widgets.Layout(width='42px'))]
                + tick_widgets,
                layout=widgets.Layout(flex_flow='row wrap', gap='2px'))
            tick_parts.append(ticks_row)

            rot_sl = widgets.IntSlider(
                value=self._tick_rotation, min=-90, max=90, step=5,
                description="Rot:", style=_SN,
                continuous_update=True)

            def _rot_cb(change):
                self._tick_rotation = change["new"]
                self._apply_tick_labels()
            rot_sl.observe(_rot_cb, names="value")
            tick_parts.append(_slider_num(rot_sl))

            align_dd = widgets.Dropdown(
                options=[("center", "center"), ("right", "right"),
                         ("left", "left")],
                value=self._tick_ha, description="Align:",
                style=_SN, layout=widgets.Layout(width="180px"))

            def _align_cb(change):
                self._tick_ha = change["new"]
                self._apply_tick_labels()
            align_dd.observe(_align_cb, names="value")
            tick_parts.append(align_dd)

            pad_sl = widgets.FloatSlider(
                value=self._tick_pad, min=0.0, max=20.0, step=0.5,
                description="Pad:", style=_SN,
                continuous_update=True)

            def _pad_cb(change):
                self._tick_pad = change["new"]
                self._apply_tick_labels()
            pad_sl.observe(_pad_cb, names="value")
            tick_parts.append(_slider_num(pad_sl))

        if tick_parts:
            controls.append(_make_toggle_section("Ticks", tick_parts))

        # --- Mode callback (wired after all sections exist) ---
        def _mode_cb(change):
            old_mode = self._mode
            self._mode = change["new"]
            has_both = (_mode_has_box(self._mode)
                        and _mode_has_violin(self._mode))
            had_both = (_mode_has_box(old_mode)
                        and _mode_has_violin(old_mode))
            if has_both and not had_both:
                for panel in self._panels:
                    panel._violin_inner = "none"
                    if hasattr(panel, '_inner_dd'):
                        panel._inner_dd.value = "none"
            for panel in self._panels:
                panel.update_section_visibility(self._mode)
            # Toggle box options visibility
            self._box_options_section.layout.display = (
                '' if _mode_has_box(self._mode) else 'none')
            self._redraw()
        mode_dd.observe(_mode_cb, names="value")

        # Set initial visibility + violin inner default
        if _mode_has_box(self._mode) and _mode_has_violin(self._mode):
            for panel in self._panels:
                panel._violin_inner = "none"
                if hasattr(panel, '_inner_dd'):
                    panel._inner_dd.value = "none"
        for panel in self._panels:
            panel.update_section_visibility(self._mode)

        return widgets.VBox(
            controls,
            layout=widgets.Layout(padding='4px 4px 4px 8px'))

    def _apply_tick_labels(self):
        """Update tick labels on axes."""
        ax = self._ax
        tc = np.array(self._tick_centers)
        rot_mode = ('anchor' if self._tick_ha != 'center'
                    else 'default')
        if self._orientation == "vertical":
            ax.set_xticks(tc)
            ax.set_xticklabels(self._tick_labels,
                               rotation=self._tick_rotation,
                               ha=self._tick_ha,
                               rotation_mode=rot_mode)
            ax.tick_params(axis='x', pad=self._tick_pad)
        else:
            ax.set_yticks(tc)
            ax.set_yticklabels(self._tick_labels,
                               rotation=self._tick_rotation,
                               ha=self._tick_ha,
                               rotation_mode=rot_mode)
            ax.tick_params(axis='y', pad=self._tick_pad)
        # Update dist info
        for info in getattr(ax, '_matplotly_dist_info', []):
            info['tick_labels'] = list(self._tick_labels)
            info['tick_centers'] = list(self._tick_centers)
            info['tick_rotation'] = self._tick_rotation
            info['tick_ha'] = self._tick_ha
            info['tick_pad'] = self._tick_pad
        self._canvas.force_redraw()

    def _redraw(self):
        """Clear all distribution artists and recreate based on current mode."""
        import sys
        ax = self._ax
        mode = self._mode
        n_groups = len(self._panels)
        n_ticks = self._n_ticks

        try:
            self._clear_distribution_artists()

            # Reset both axes so stale limits from a previous
            # orientation don't interfere with the new layout.
            ax.set_xlim(auto=True)
            ax.set_ylim(auto=True)

            # Compute positions for each group
            tick_centers = np.array(self._tick_centers, dtype=float)
            width = self._width
            gap = self._gap

            for j, panel in enumerate(self._panels):
                offset = (j - (n_groups - 1) / 2) * (width + gap)
                positions = tick_centers + offset
                panel._positions = positions.tolist()
                panel._orientation = self._orientation
                raw_data = panel._raw_data

                if not raw_data:
                    continue

                # Draw violin first (background)
                if _mode_has_violin(mode):
                    self._draw_violin(ax, panel, positions, raw_data, width)

                # Draw box on top
                if _mode_has_box(mode):
                    self._draw_box(ax, panel, positions, raw_data, width,
                                   narrow=_mode_has_violin(mode))

                # Draw jitter on top
                if _mode_has_jitter(mode):
                    self._draw_jitter(ax, panel, positions, raw_data)

            # Update tick labels + reset opposite axis
            from matplotlib.ticker import AutoLocator, ScalarFormatter
            rot_mode = ('anchor' if self._tick_ha != 'center'
                        else 'default')
            if self._orientation == "vertical":
                ax.set_xticks(tick_centers)
                ax.set_xticklabels(self._tick_labels,
                                   rotation=self._tick_rotation,
                                   ha=self._tick_ha,
                                   rotation_mode=rot_mode)
                ax.tick_params(axis='x', pad=self._tick_pad)
                # Reset y-axis to automatic ticking + clear rotation
                ax.yaxis.set_major_locator(AutoLocator())
                ax.yaxis.set_major_formatter(ScalarFormatter())
                ax.tick_params(axis='y', rotation=0, pad=4.0)
                for t in ax.get_yticklabels():
                    t.set_ha('right')
                    t.set_rotation_mode('default')
            else:
                ax.set_yticks(tick_centers)
                ax.set_yticklabels(self._tick_labels,
                                   rotation=self._tick_rotation,
                                   ha=self._tick_ha,
                                   rotation_mode=rot_mode)
                ax.tick_params(axis='y', pad=self._tick_pad)
                # Reset x-axis to automatic ticking + clear rotation
                ax.xaxis.set_major_locator(AutoLocator())
                ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.tick_params(axis='x', rotation=0, pad=4.0)
                for t in ax.get_xticklabels():
                    t.set_ha('center')
                    t.set_rotation_mode('default')

            ax.relim()
            ax.autoscale_view()

            # --- Explicitly set BOTH axes from scratch ---
            # Category axis: computed from tick_centers + element widths
            if n_ticks > 0:
                half_extent = max(width, width * 1.5) / 2 + 0.15
                if n_groups > 1:
                    half_extent += abs(
                        (n_groups - 1) / 2 * (width + gap))
                cat_lo = float(tick_centers[0]) - half_extent
                cat_hi = float(tick_centers[-1]) + half_extent
            else:
                cat_lo, cat_hi = -0.5, 0.5

            # Data axis: computed from raw data
            all_vals = []
            for panel in self._panels:
                for rd in panel._raw_data:
                    if len(rd) > 0:
                        all_vals.append(float(np.min(rd)))
                        all_vals.append(float(np.max(rd)))
            if all_vals:
                dmin, dmax = min(all_vals), max(all_vals)
                margin = (dmax - dmin) * 0.05
                if margin < 0.1:
                    margin = 1.0
                data_lo, data_hi = dmin - margin, dmax + margin
            else:
                data_lo, data_hi = 0.0, 1.0

            if self._orientation == "vertical":
                ax.set_xlim(cat_lo, cat_hi)
                ax.set_ylim(data_lo, data_hi)
            else:
                ax.set_ylim(cat_lo, cat_hi)
                ax.set_xlim(data_lo, data_hi)

            # Update all panels' dist info
            for panel in self._panels:
                panel._store_dist_info()

            # Refresh legend so name changes appear immediately
            _refresh_legend(ax)

        except Exception as exc:
            import traceback
            traceback.print_exc(file=sys.stderr)

        self._canvas.force_redraw()

    def _clear_distribution_artists(self):
        """Remove all distribution-related artists from axes."""
        from matplotlib.collections import LineCollection, PolyCollection
        ax = self._ax

        # Remove all _-prefixed lines (boxplot internal lines)
        for line in list(ax.lines):
            if line.get_label().startswith("_"):
                line.remove()

        # Remove distribution-tagged artists
        for coll in list(ax.collections):
            if getattr(coll, '_matplotly_dist', False):
                coll.remove()
            elif isinstance(coll, PolyCollection):
                # Check if this is a violin body
                if getattr(coll, '_matplotly_dist', False):
                    coll.remove()
            elif isinstance(coll, LineCollection):
                if getattr(coll, '_matplotly_dist', False):
                    coll.remove()

        # Remove distribution-tagged patches
        for patch in list(ax.patches):
            if getattr(patch, '_matplotly_dist', False):
                patch.remove()

        # Remove distribution-tagged scatter collections
        for coll in list(ax.collections):
            if getattr(coll, '_matplotly_dist_jitter', False):
                coll.remove()

        # Clear original artists from first introspection
        for panel in self._panels:
            for a in list(panel._group.artists):
                try:
                    a.remove()
                except (ValueError, AttributeError):
                    pass
            panel._group.artists = []

        ax.stale = True

    def _draw_box(self, ax, panel, positions, raw_data, width, narrow=False):
        """Draw box plot component."""
        box_width = width * 0.3 if narrow else width

        # Prefer original introspected box_stats so the rendering matches
        # the generated code (avoids losing fliers through reconstruction).
        orig = panel._box_stats if hasattr(panel, '_box_stats') else []
        bxp_stats = []

        if orig and len(orig) == len(raw_data):
            for k, s in enumerate(orig):
                if k >= len(positions):
                    break
                med = s.get('med') or s.get('median', 0)
                q1, q3 = s['q1'], s['q3']
                iqr = q3 - q1
                data = raw_data[k] if k < len(raw_data) else np.array([])
                mean_val = float(np.mean(data)) if len(data) > 0 else med
                notch_ci = 1.57 * iqr / np.sqrt(len(data)) if len(data) > 0 else 0
                bxp_stats.append({
                    'med': float(med),
                    'q1': float(q1),
                    'q3': float(q3),
                    'whislo': float(s['whislo']),
                    'whishi': float(s['whishi']),
                    'fliers': list(s.get('fliers', [])),
                    'mean': mean_val,
                    'cilo': float(med - notch_ci),
                    'cihi': float(med + notch_ci),
                })
        else:
            for k, data in enumerate(raw_data):
                if k >= len(positions):
                    break
                if len(data) == 0:
                    continue
                q1, med, q3 = np.percentile(data, [25, 50, 75])
                iqr = q3 - q1
                whislo = data[data >= q1 - 1.5 * iqr]
                whislo = float(np.min(whislo)) if len(whislo) > 0 else float(q1)
                whishi = data[data <= q3 + 1.5 * iqr]
                whishi = float(np.max(whishi)) if len(whishi) > 0 else float(q3)
                fliers = data[(data < whislo) | (data > whishi)]

                # Notch confidence interval (1.57 * IQR / sqrt(n))
                n = len(data)
                notch_ci = 1.57 * iqr / np.sqrt(n) if n > 0 else 0

                bxp_stats.append({
                    'med': float(med),
                    'q1': float(q1),
                    'q3': float(q3),
                    'whislo': whislo,
                    'whishi': whishi,
                    'fliers': fliers.tolist() if len(fliers) > 0 else [],
                    'mean': float(np.mean(data)),
                    'cilo': float(med - notch_ci),
                    'cihi': float(med + notch_ci),
                })

        if not bxp_stats:
            return

        flier_props = {
            'marker': panel._flier_marker or 'o',
            'markersize': panel._flier_size,
            'markerfacecolor': panel._flier_color,
            'markeredgecolor': panel._flier_color,
        }
        if not panel._flier_marker:
            flier_props = {'marker': ''}

        median_props = {
            'color': panel._median_color,
            'linewidth': panel._median_lw,
        }

        whisker_props = {
            'linestyle': panel._whisker_style,
            'linewidth': panel._whisker_lw,
        }

        box_props = {
            'facecolor': panel._box_color,
            'edgecolor': panel._box_edgecolor,
            'linewidth': panel._box_lw,
            'alpha': panel._box_alpha,
        }
        if panel._box_hatch:
            box_props['hatch'] = panel._box_hatch

        mean_props = {}
        if panel._show_mean:
            mean_props = {
                'linestyle': panel._mean_style,
                'color': panel._median_color,
            }

        try:
            bp = ax.bxp(
                bxp_stats,
                positions=list(positions[:len(bxp_stats)]),
                widths=box_width,
                vert=(self._orientation == "vertical"),
                patch_artist=True,
                shownotches=panel._notch,
                showmeans=panel._show_mean,
                meanline=panel._show_mean,
                flierprops=flier_props,
                medianprops=median_props,
                whiskerprops=whisker_props,
                boxprops=box_props,
                capprops={'linewidth': panel._whisker_lw},
                meanprops=mean_props,
            )

            # Tag all created artists
            for key in ('boxes', 'medians', 'whiskers', 'caps', 'fliers',
                        'means'):
                for artist in bp.get(key, []):
                    artist._matplotly_dist = True

            # Set legend label on first box so legend toggle can find it
            if bp.get('boxes'):
                bp['boxes'][0].set_label(panel._label)
        except Exception:
            import traceback, sys
            traceback.print_exc(file=sys.stderr)

    def _draw_violin(self, ax, panel, positions, raw_data, width):
        """Draw violin plot component."""
        # Filter out empty arrays
        valid_data = []
        valid_pos = []
        for k, data in enumerate(raw_data):
            if k >= len(positions):
                break
            if len(data) >= 2:
                valid_data.append(data)
                valid_pos.append(positions[k])

        if not valid_data:
            return

        try:
            vp = ax.violinplot(
                valid_data,
                positions=valid_pos,
                widths=width * 1.5,
                showmeans=False,
                showmedians=(panel._violin_inner != "none"),
                showextrema=(panel._violin_inner != "none"),
                vert=(self._orientation == "vertical"),
            )

            # Style violin bodies
            for body in vp.get('bodies', []):
                body.set_facecolor(panel._violin_color)
                body.set_edgecolor(panel._violin_edgecolor)
                body.set_alpha(panel._violin_alpha)
                body._matplotly_dist = True

            # Set legend label on first body if box is not shown
            if vp.get('bodies') and not _mode_has_box(self._mode):
                vp['bodies'][0].set_label(panel._label)

            # Style stat lines
            for key in ('cmeans', 'cmedians', 'cmins', 'cmaxes', 'cbars'):
                coll = vp.get(key)
                if coll is not None:
                    coll._matplotly_dist = True
                    coll.set_color(panel._violin_edgecolor)
                    coll.set_alpha(0.8)
        except Exception:
            import traceback, sys
            traceback.print_exc(file=sys.stderr)

    def _draw_jitter(self, ax, panel, positions, raw_data):
        """Draw jitter (strip) plot component."""
        rng = np.random.RandomState(42)
        for k, data in enumerate(raw_data):
            if k >= len(positions):
                break
            if len(data) == 0:
                continue
            pos = positions[k]
            jitter_offsets = rng.uniform(
                -panel._jitter_spread, panel._jitter_spread, len(data))

            if self._orientation == "vertical":
                x = pos + jitter_offsets
                y = data
            else:
                x = data
                y = pos + jitter_offsets

            # Label first scatter per panel for legend (jitter-only mode)
            jitter_only = (not _mode_has_box(self._mode)
                           and not _mode_has_violin(self._mode))
            lbl = panel._label if (k == 0 and jitter_only) else "_nolegend_"

            sc = ax.scatter(
                x, y,
                s=panel._jitter_size ** 2,
                c=panel._jitter_color,
                alpha=panel._jitter_alpha,
                marker=panel._jitter_marker,
                zorder=3,
                label=lbl,
            )
            sc._matplotly_dist_jitter = True
            sc._matplotly_dist = True
