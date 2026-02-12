"""Bar chart controls — per-group visual styling + shared structural controls.

BarPanel: per-group controls (name, face color, edge color, edge width,
          alpha, hatch, linestyle) with collapsible header and swatch palette.
BarSharedPanel: structural controls shared across all bar groups on the same
                axes (bar width, gap, orientation, tick labels, tick rotation).
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
    cmap_color_btn,
)


class BarPanel(ArtistPanel):
    """Per-group bar panel: name, color, edge color, edge width, alpha, hatch."""

    _plot_number: int = 0
    _on_label_changed = None

    def build(self) -> widgets.Widget:
        patches = self._group.artists
        ref = patches[0]
        meta = self._group.metadata

        # Extract geometry from metadata
        self._positions = meta.get("positions", [])
        self._values = meta.get("values", [])
        self._bottoms = meta.get("bottoms", [])
        self._bar_width = meta.get("bar_width", 0.8)
        self._orientation = meta.get("orientation", "vertical")
        self._zorder = meta.get("zorder", ref.get_zorder())

        try:
            self._color = to_hex(ref.get_facecolor())
        except Exception:
            self._color = "#1f77b4"

        try:
            self._edgecolor = to_hex(ref.get_edgecolor())
        except Exception:
            self._edgecolor = "#000000"

        self._edge_width = ref.get_linewidth() if hasattr(ref, 'get_linewidth') else 1.0
        alpha = ref.get_alpha()
        self._alpha = alpha if alpha is not None else 1.0
        self._hatch = ref.get_hatch() or ""
        self._linestyle = ref.get_linestyle() or "-"

        # Error bar state
        self._original_errbar_artists = meta.get("errbar_artists", [])
        self._errbar_values = meta.get("errbar_values", None)
        has_orig_eb = bool(self._original_errbar_artists)
        self._show_errorbars = has_orig_eb
        self._errbar_color = meta.get("errbar_color", self._color)
        self._errbar_alpha = 1.0
        self._errbar_linewidth = meta.get("errbar_linewidth", 1.5)
        self._errbar_capsize = meta.get("errbar_capsize", 3.0)
        self._errbar_linestyle = "-"
        self._errbar_color_auto = not has_orig_eb

        label = self._group.label
        if label.startswith("Bar: "):
            label = label[len("Bar: "):]
        self._label = label

        # --- Collapsible header ---
        num = self._plot_number or ""
        header_prefix = f"Bar {num}: " if num else ""
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
        """Build per-bar visual controls."""
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
            self._update_bar_info()
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
            self._update_bar_info()
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
            self._update_bar_info()
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
            self._update_bar_info()
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
            self._update_bar_info()
            _refresh_legend(self._group.axes)
            self._canvas.force_redraw()
        ls_dd.observe(_ls_cb, names="value")
        controls.append(ls_dd)

        # --- Error bars toggle + controls ---
        eb_toggle = widgets.Checkbox(
            value=self._show_errorbars, description="Error Bars",
            style={"description_width": "auto"},
            indent=False)

        def _on_eb_color(hex_val):
            self._errbar_color = hex_val
            self._errbar_color_auto = False
            self._draw_bar_errorbars()
            self._canvas.force_redraw()
        eb_color_btn, eb_swatch_row = cmap_color_btn(
            self._errbar_color, _on_eb_color)
        eb_color_row = widgets.HBox(
            [widgets.Label("Color:", layout=widgets.Layout(width='42px')),
             eb_color_btn],
            layout=widgets.Layout(align_items='center', gap='4px'))

        eb_alpha_sl = widgets.FloatSlider(
            value=self._errbar_alpha, min=0, max=1, step=0.05,
            description="Alpha:", style=_SN)
        def _eb_alpha_cb(change):
            self._errbar_alpha = change["new"]
            self._draw_bar_errorbars()
            self._canvas.force_redraw()
        eb_alpha_sl.observe(_eb_alpha_cb, names="value")

        eb_lw_sl = widgets.FloatSlider(
            value=self._errbar_linewidth, min=0.1, max=10, step=0.1,
            description="Line w:", style=_SN)
        def _eb_lw_cb(change):
            self._errbar_linewidth = change["new"]
            self._draw_bar_errorbars()
            self._canvas.force_redraw()
        eb_lw_sl.observe(_eb_lw_cb, names="value")

        eb_cap_sl = widgets.FloatSlider(
            value=self._errbar_capsize, min=0, max=15, step=0.5,
            description="Cap size:", style=_SN)
        def _eb_cap_cb(change):
            self._errbar_capsize = change["new"]
            self._draw_bar_errorbars()
            self._canvas.force_redraw()
        eb_cap_sl.observe(_eb_cap_cb, names="value")

        eb_ls_dd = widgets.Dropdown(
            options=[("solid", "-"), ("dashed", "--"),
                     ("dotted", ":"), ("dashdot", "-.")],
            value=self._errbar_linestyle, description="Style:",
            style=_SN, layout=widgets.Layout(width="180px"))
        def _eb_ls_cb(change):
            self._errbar_linestyle = change["new"]
            self._draw_bar_errorbars()
            self._canvas.force_redraw()
        eb_ls_dd.observe(_eb_ls_cb, names="value")

        eb_controls = widgets.VBox(
            [eb_color_row, eb_swatch_row,
             _slider_num(eb_alpha_sl),
             _slider_num(eb_lw_sl),
             _slider_num(eb_cap_sl),
             eb_ls_dd],
            layout=widgets.Layout(
                display='' if self._show_errorbars else 'none',
                padding='2px 0 2px 12px'))
        self._eb_color_btn = eb_color_btn

        def _eb_toggle_cb(change):
            self._show_errorbars = change["new"]
            if self._show_errorbars:
                eb_controls.layout.display = ''
                self._draw_bar_errorbars()
            else:
                eb_controls.layout.display = 'none'
                self._clear_bar_errorbars()
                self._update_bar_info()
                self._canvas.force_redraw()
        eb_toggle.observe(_eb_toggle_cb, names="value")
        controls.append(eb_toggle)
        controls.append(eb_controls)

        return controls

    # --- Error bar drawing ---------------------------------------------------

    def _clear_bar_errorbars(self):
        """Remove all error bar artists for this panel (matplotly + original)."""
        ax = self._group.axes
        gid = id(self._group)
        # Remove matplotly-tagged error bar artists
        for l in list(ax.lines):
            if (getattr(l, '_matplotly_bar_errorbar', False)
                    and getattr(l, '_matplotly_bar_eb_group', None) == gid):
                l.remove()
        for c in list(ax.collections):
            if (getattr(c, '_matplotly_bar_errorbar', False)
                    and getattr(c, '_matplotly_bar_eb_group', None) == gid):
                c.remove()
        from matplotlib.container import ErrorbarContainer
        ax.containers[:] = [
            c for c in ax.containers
            if not (isinstance(c, ErrorbarContainer)
                    and getattr(c, '_matplotly_bar_errorbar', False)
                    and getattr(c, '_matplotly_bar_eb_group', None) == gid)
        ]
        # Remove original error bar artists (from ax.bar(yerr=...))
        for art in self._original_errbar_artists:
            try:
                art.remove()
            except (ValueError, AttributeError):
                pass
        self._original_errbar_artists = []
        # Remove original ErrorbarContainer
        orig_eb = self._group.metadata.get("errbar_container")
        if orig_eb is not None:
            ax.containers[:] = [
                c for c in ax.containers if c is not orig_eb]
            self._group.metadata["errbar_container"] = None

    def _draw_bar_errorbars(self):
        """Draw error bars on the bar chart using the bar values as errors."""
        self._clear_bar_errorbars()
        if not self._show_errorbars:
            return

        ax = self._group.axes
        positions = np.asarray(self._positions)
        values = np.asarray(self._values)
        if len(positions) == 0 or len(values) == 0:
            return

        # Use stored error values if provided, otherwise use the bar values
        if self._errbar_values is not None:
            errors = np.asarray(self._errbar_values)
        else:
            errors = np.abs(values) * 0.1  # default: 10% of bar value

        is_horiz = self._orientation == "horizontal"
        if is_horiz:
            eb = ax.errorbar(
                values, positions, xerr=errors, fmt='none',
                ecolor=self._errbar_color,
                elinewidth=self._errbar_linewidth,
                capsize=self._errbar_capsize,
                alpha=self._errbar_alpha,
                zorder=self._zorder + 1,
            )
        else:
            eb = ax.errorbar(
                positions, values, yerr=errors, fmt='none',
                ecolor=self._errbar_color,
                elinewidth=self._errbar_linewidth,
                capsize=self._errbar_capsize,
                alpha=self._errbar_alpha,
                zorder=self._zorder + 1,
            )

        # Apply linestyle to bar segments
        if self._errbar_linestyle != "-":
            for bar_coll in eb[2]:
                bar_coll.set_linestyle(self._errbar_linestyle)

        # Tag all artists in the ErrorbarContainer
        gid = id(self._group)
        for artist in eb.get_children():
            artist._matplotly_bar_errorbar = True
            artist._matplotly_bar_eb_group = gid
            artist.set_label("_nolegend_")
        eb._matplotly_bar_errorbar = True
        eb._matplotly_bar_eb_group = gid

        self._update_bar_info()
        self._canvas.force_redraw()

    # --- Color section (reused for face + edge) ---

    def _build_color_section(self, label_text, current_color, is_face):
        """Color section with swatch palette, matching histogram pattern."""

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

        _picker_cls = f"pb-picker-bar-{id(self)}-{'face' if is_face else 'edge'}"
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
                # Sync error bar color if auto-tracking
                if self._errbar_color_auto:
                    self._errbar_color = hex_val
                    if hasattr(self, '_eb_color_btn'):
                        self._eb_color_btn.style.button_color = hex_val
                    if self._show_errorbars:
                        self._draw_bar_errorbars()
            else:
                self._edgecolor = hex_val
                for p in self._group.artists:
                    p.set_edgecolor(hex_val)
            self._update_bar_info()
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

    # --- Bar info storage (for code gen) ---

    def _store_bar_info(self):
        """Store bar info on axes for code generation (initial setup)."""
        ax = self._group.axes

        # Detect tick labels from axes
        orient = self._orientation
        if orient == "vertical":
            tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        else:
            tick_labels = [t.get_text() for t in ax.get_yticklabels()]

        # Compute tick centers as mean of all groups' positions on these axes
        if all(not t for t in tick_labels):
            tick_labels = [str(round(p, 2)) for p in self._positions]

        bar_info = {
            "_group_id": id(self._group),
            "values": list(self._values),
            "positions": list(self._positions),
            "bottoms": list(self._bottoms),
            "bar_width": self._bar_width,
            "bar_gap": 0.0,  # computed later by BarSharedPanel
            "orientation": self._orientation,
            "color": self._color,
            "edgecolor": self._edgecolor,
            "linewidth": self._edge_width,
            "alpha": self._alpha,
            "label": self._label,
            "hatch": self._hatch,
            "linestyle": self._linestyle,
            "zorder": self._zorder,
            "tick_labels": tick_labels,
            "tick_centers": list(self._positions),
            "tick_rotation": 0,
            "tick_ha": "center",
            "tick_pad": 4.0,
            "show_errorbars": self._show_errorbars,
            "errbar_color": self._errbar_color,
            "errbar_alpha": self._errbar_alpha,
            "errbar_linewidth": self._errbar_linewidth,
            "errbar_capsize": self._errbar_capsize,
            "errbar_linestyle": self._errbar_linestyle,
            "errbar_values": (self._errbar_values.tolist()
                              if self._errbar_values is not None
                              else None),
        }
        if not hasattr(ax, '_matplotly_bar_info'):
            ax._matplotly_bar_info = []
        ax._matplotly_bar_info.append(bar_info)

    def _update_bar_info(self):
        """Update the bar_info entry on axes after a visual change."""
        ax = self._group.axes
        if not hasattr(ax, '_matplotly_bar_info'):
            return
        for i, info in enumerate(ax._matplotly_bar_info):
            if info.get("_group_id") == id(self._group):
                info["values"] = list(self._values)
                info["positions"] = list(self._positions)
                info["bottoms"] = list(self._bottoms)
                info["bar_width"] = self._bar_width
                info["orientation"] = self._orientation
                info["color"] = self._color
                info["edgecolor"] = self._edgecolor
                info["linewidth"] = self._edge_width
                info["alpha"] = self._alpha
                info["label"] = self._label
                info["hatch"] = self._hatch
                info["linestyle"] = self._linestyle
                info["zorder"] = self._zorder
                info["show_errorbars"] = self._show_errorbars
                info["errbar_color"] = self._errbar_color
                info["errbar_alpha"] = self._errbar_alpha
                info["errbar_linewidth"] = self._errbar_linewidth
                info["errbar_capsize"] = self._errbar_capsize
                info["errbar_linestyle"] = self._errbar_linestyle
                info["errbar_values"] = (self._errbar_values.tolist()
                                         if self._errbar_values is not None
                                         else None)
                break


class BarSharedPanel:
    """Shared structural controls applied to all bar groups on the same axes.

    Controls: bar width, gap (grouped only), orientation, tick labels, tick rotation.
    """

    def __init__(self, panels: list[BarPanel], canvas):
        self._panels = panels
        self._canvas = canvas
        self._ax = panels[0]._group.axes
        n_groups = len(panels)

        # Read initial values
        ref = panels[0]
        self._bar_width = ref._bar_width
        self._orientation = ref._orientation
        self._n_ticks = len(ref._positions) if ref._positions else 0

        # Compute tick centers (mean of all groups' positions per tick)
        if n_groups > 1 and self._n_ticks > 0:
            all_pos = np.array([p._positions for p in panels])
            self._tick_centers = np.mean(all_pos, axis=0).tolist()
            # Compute initial gap
            if len(panels[0]._positions) > 0 and len(panels[1]._positions) > 0:
                self._bar_gap = abs(
                    panels[1]._positions[0] - panels[0]._positions[0]
                ) - self._bar_width
                if self._bar_gap < 0:
                    self._bar_gap = 0.0
            else:
                self._bar_gap = 0.0
        else:
            self._tick_centers = list(ref._positions)
            self._bar_gap = 0.0

        # Detect tick labels
        if self._orientation == "vertical":
            labels = [t.get_text() for t in self._ax.get_xticklabels()]
        else:
            labels = [t.get_text() for t in self._ax.get_yticklabels()]
        if all(not t for t in labels):
            labels = [str(round(p, 2)) for p in self._tick_centers]
        self._tick_labels = labels[:self._n_ticks]

        # Detect tick rotation
        if self._orientation == "vertical":
            tick_objs = self._ax.get_xticklabels()
        else:
            tick_objs = self._ax.get_yticklabels()
        self._tick_rotation = int(tick_objs[0].get_rotation()) if tick_objs else 0
        self._tick_ha = tick_objs[0].get_ha() if tick_objs else "center"
        self._tick_pad = 4.0  # default tick padding in points

    def build(self) -> widgets.Widget:
        controls = []
        n_groups = len(self._panels)

        # --- Bar width ---
        # Max width = 1/n_groups so bars from adjacent ticks don't overlap
        max_w = round(1.0 / n_groups, 2)
        width_sl = widgets.FloatSlider(
            value=min(round(self._bar_width, 2), max_w),
            min=0.05, max=max_w, step=0.05,
            description="Width:", style=_SN,
            continuous_update=True)

        def _width_cb(change):
            self._bar_width = change["new"]
            self._redraw_bars()
        width_sl.observe(_width_cb, names="value")
        controls.append(_slider_num(width_sl))

        # --- Bar gap (only for grouped bars) ---
        if n_groups > 1:
            gap_sl = widgets.FloatSlider(
                value=round(self._bar_gap, 2), min=0.0, max=0.5, step=0.05,
                description="Gap:", style=_SN,
                continuous_update=True)

            def _gap_cb(change):
                self._bar_gap = change["new"]
                self._redraw_bars()
            gap_sl.observe(_gap_cb, names="value")
            controls.append(_slider_num(gap_sl))

        # --- Orientation ---
        orient_dd = widgets.Dropdown(
            options=[("vertical", "vertical"), ("horizontal", "horizontal")],
            value=self._orientation, description="Orient:",
            style=_SN, layout=widgets.Layout(width="180px"))

        def _orient_cb(change):
            self._orientation = change["new"]
            self._redraw_bars()
        orient_dd.observe(_orient_cb, names="value")
        controls.append(orient_dd)

        # --- Tick labels ---
        if self._n_ticks > 0:
            tick_widgets = []
            self._tick_fields = []
            for k in range(self._n_ticks):
                lbl = self._tick_labels[k] if k < len(self._tick_labels) else ""
                tw = widgets.Text(
                    value=lbl,
                    layout=widgets.Layout(width="70px"))
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
            controls.append(ticks_row)

            # --- Tick rotation ---
            rot_sl = widgets.IntSlider(
                value=self._tick_rotation, min=-90, max=90, step=5,
                description="Rot:", style=_SN,
                continuous_update=True)

            def _rot_cb(change):
                self._tick_rotation = change["new"]
                self._apply_tick_labels()
            rot_sl.observe(_rot_cb, names="value")
            controls.append(_slider_num(rot_sl))

            # --- Tick alignment ---
            align_dd = widgets.Dropdown(
                options=[("center", "center"), ("right", "right"),
                         ("left", "left")],
                value=self._tick_ha, description="Align:",
                style=_SN, layout=widgets.Layout(width="180px"))

            def _align_cb(change):
                self._tick_ha = change["new"]
                self._apply_tick_labels()
            align_dd.observe(_align_cb, names="value")
            controls.append(align_dd)

            # --- Tick pad (distance from axis, in points) ---
            pad_sl = widgets.FloatSlider(
                value=self._tick_pad, min=0.0, max=20.0, step=0.5,
                description="Pad:", style=_SN,
                continuous_update=True)

            def _pad_cb(change):
                self._tick_pad = change["new"]
                self._apply_tick_labels()
            pad_sl.observe(_pad_cb, names="value")
            controls.append(_slider_num(pad_sl))

        return widgets.VBox(
            controls,
            layout=widgets.Layout(padding='4px 4px 4px 8px'))

    def _apply_tick_labels(self):
        """Update tick labels, rotation, alignment, and pad on the axes."""
        ax = self._ax
        tc = np.array(self._tick_centers)
        # Use rotation_mode='anchor' when ha is not center for clean rotated labels
        rot_mode = 'anchor' if self._tick_ha != 'center' else 'default'
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
        # Update bar info with new tick labels/centers
        for info in getattr(ax, '_matplotly_bar_info', []):
            info['tick_labels'] = list(self._tick_labels)
            info['tick_centers'] = list(self._tick_centers)
            info['tick_rotation'] = self._tick_rotation
            info['tick_ha'] = self._tick_ha
            info['tick_pad'] = self._tick_pad
        self._canvas.force_redraw()

    def _clear_all_bar_patches(self):
        """Remove all bar-chart BarContainers and patches (skip histograms)."""
        from matplotlib.container import BarContainer
        from .._introspect import FigureIntrospector as _FI
        ax = self._ax

        # Clear error bars for all panels
        for panel in self._panels:
            panel._clear_bar_errorbars()

        for c in list(ax.containers):
            if isinstance(c, BarContainer) and not _FI._is_histogram_container(c):
                for p in c:
                    p.remove()
        ax.containers[:] = [
            c for c in ax.containers
            if not isinstance(c, BarContainer)
            or _FI._is_histogram_container(c)
        ]

    def _redraw_bars(self):
        """Recompute positions and recreate all bar groups."""
        from matplotlib.container import BarContainer
        ax = self._ax
        n_groups = len(self._panels)
        n_ticks = self._n_ticks
        bw = self._bar_width
        bg = self._bar_gap

        # Remove existing bar patches
        self._clear_all_bar_patches()

        # Compute new positions
        tick_centers = np.arange(n_ticks, dtype=float)
        self._tick_centers = tick_centers.tolist()

        for j, panel in enumerate(self._panels):
            offset = (j - (n_groups - 1) / 2) * (bw + bg)
            positions = tick_centers + offset

            # Update panel state
            panel._positions = positions.tolist()
            panel._bar_width = bw
            panel._orientation = self._orientation

            # Build kwargs
            kwargs = {
                "width" if self._orientation == "vertical" else "height": bw,
                "color": panel._color,
                "edgecolor": panel._edgecolor,
                "linewidth": panel._edge_width,
                "alpha": panel._alpha,
                "label": panel._label,
                "zorder": panel._zorder,
            }
            if panel._hatch:
                kwargs["hatch"] = panel._hatch
            if panel._linestyle not in ("-", "solid"):
                kwargs["linestyle"] = panel._linestyle

            values = panel._values
            if self._orientation == "vertical":
                container = ax.bar(positions, values, **kwargs)
            else:
                container = ax.barh(positions, values, **kwargs)

            # Update artist references
            new_artists = list(container.patches)
            panel._group.artists = new_artists
            panel._group.metadata["container"] = container
            panel._group.metadata["positions"] = panel._positions
            panel._group.metadata["bar_width"] = bw
            panel._group.metadata["orientation"] = self._orientation

            # Update bar info for code gen
            panel._update_bar_info()

        # Update tick labels/positions and reset the opposite axis
        from matplotlib.ticker import AutoLocator, ScalarFormatter
        rot_mode = 'anchor' if self._tick_ha != 'center' else 'default'
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
                t.set_ha('center')
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

        # Update bar info with tick centers
        for info in getattr(ax, '_matplotly_bar_info', []):
            info['tick_labels'] = list(self._tick_labels)
            info['tick_centers'] = self._tick_centers
            info['bar_width'] = bw
            info['bar_gap'] = bg
            info['orientation'] = self._orientation
            info['tick_rotation'] = self._tick_rotation
            info['tick_ha'] = self._tick_ha
            info['tick_pad'] = self._tick_pad

        # Redraw error bars for panels that have them enabled
        for panel in self._panels:
            if panel._show_errorbars:
                panel._draw_bar_errorbars()

        ax.relim()
        ax.autoscale_view()
        _refresh_legend(ax)
        self._canvas.force_redraw()
