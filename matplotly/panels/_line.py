"""Line plot controls — collapsible per-curve sections with compact color UI."""
from __future__ import annotations

import ipywidgets as widgets
import matplotlib
from matplotlib.colors import ListedColormap, to_hex, to_rgb

from .._commands import BatchCommand, Command
from .._types import ArtistGroup, PlotType
from ._base import ArtistPanel

_COLORMAPS = [
    # Qualitative
    "tab10", "tab20", "Set1", "Set2", "Set3", "Paired",
    "Dark2", "Accent", "Pastel1", "Pastel2",
    # Diverging
    "bwr", "coolwarm", "RdBu", "RdYlBu", "RdYlGn",
    "PiYG", "PRGn", "seismic",
    # Sequential
    "viridis", "plasma", "inferno", "magma", "cividis",
    "Blues", "Reds", "Greens", "Oranges", "Purples",
    "YlOrRd", "YlGnBu", "BuGn",
    # Cyclic
    "twilight", "hsv",
]

_DW = "48px"    # uniform description_width
_NW = "50px"    # uniform number-edit width
_SN = {"description_width": _DW}


def _slider_num(slider):
    """Slider (no readout) + linked number edit box (2 dp)."""
    slider.readout = False
    slider.style = {"description_width": _DW}
    num = widgets.BoundedFloatText(
        value=round(slider.value, 2), step=slider.step,
        min=slider.min, max=slider.max,
        layout=widgets.Layout(width=_NW))
    widgets.link((slider, "value"), (num, "value"))
    return widgets.HBox([slider, num])


def _cmap_color(cmap, i, n):
    """Sample color i of n from a colormap.

    For small qualitative colormaps (tab10, Set1, etc. with N <= 20), returns
    sequential discrete colors (1st, 2nd, 3rd...).  For everything else
    (continuous or large-N listed colormaps like viridis with N=256),
    interpolates evenly across the full [0,1] range.
    """
    if isinstance(cmap, ListedColormap) and cmap.N <= 20:
        return cmap(i % cmap.N)
    return cmap(i / max(n - 1, 1))


def _get_palette_colors(cmap_name, n=10):
    """Get n hex colors from a colormap."""
    try:
        cmap = matplotlib.colormaps.get_cmap(cmap_name)
    except Exception:
        cmap = matplotlib.colormaps.get_cmap("tab10")
    return [to_hex(_cmap_color(cmap, i, n)) for i in range(n)]


def _make_color_dot(hex_color):
    return (f'<div style="width:14px;height:14px;background:{hex_color};'
            f'border:1px solid #666;border-radius:2px;flex-shrink:0"></div>')


def _refresh_legend(ax):
    """Recreate the legend so it reflects current artist properties."""
    if ax is None:
        return
    leg = ax.get_legend()
    if leg is None:
        return
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    props = {}
    try:
        props['frameon'] = leg.get_frame().get_visible()
    except Exception:
        pass
    try:
        props['fontsize'] = leg._fontsize
    except Exception:
        pass
    try:
        props['ncol'] = getattr(leg, '_ncols', 1)
    except Exception:
        pass
    try:
        if hasattr(leg, '_bbox_to_anchor') and leg._bbox_to_anchor is not None:
            inv = ax.transAxes.inverted()
            bx, by = inv.transform(
                (leg._bbox_to_anchor.x0, leg._bbox_to_anchor.y0))
            props['bbox_to_anchor'] = (round(bx, 3), round(by, 3))
        props['loc'] = leg._loc
    except Exception:
        pass
    ax.legend(handles, labels, **props)


class LinePanel(ArtistPanel):
    _plot_number: int = 0  # set by _api.py before build()

    def build(self) -> widgets.Widget:
        line = self._group.artists[0]
        current_label = line.get_label() or self._group.label

        try:
            current_color = to_hex(line.get_color())
        except Exception:
            current_color = "#1f77b4"

        # --- Collapsible header ---
        num = self._plot_number or ""
        header_prefix = f"Line {num}: " if num else ""
        self._header_prefix = header_prefix

        self._color_indicator = widgets.HTML(_make_color_dot(current_color))

        toggle_btn = widgets.Button(
            description=f"  {header_prefix}{current_label}",
            icon="chevron-right",
            layout=widgets.Layout(width='100%', height='28px'))
        toggle_btn.style.button_color = '#f0f0f0'
        toggle_btn.style.font_weight = 'bold'
        self._toggle_btn = toggle_btn

        header_row = widgets.HBox(
            [self._color_indicator, toggle_btn],
            layout=widgets.Layout(align_items='center', gap='4px'))

        # --- Controls (collapsed by default) ---
        controls = self._build_controls(line, current_label, current_color)
        controls_box = widgets.VBox(
            controls,
            layout=widgets.Layout(display='none', padding='2px 0 4px 12px'))
        self._controls_box = controls_box
        self._is_expanded = False

        def _toggle(btn, l=line):
            self._is_expanded = not self._is_expanded
            lbl = l.get_label() or self._group.label
            if self._is_expanded:
                controls_box.layout.display = ''
                toggle_btn.icon = "chevron-down"
                toggle_btn.description = f"  {self._header_prefix}{lbl}"
            else:
                controls_box.layout.display = 'none'
                toggle_btn.icon = "chevron-right"
                toggle_btn.description = f"  {self._header_prefix}{lbl}"
        toggle_btn.on_click(_toggle)

        return widgets.VBox(
            [header_row, controls_box],
            layout=widgets.Layout(
                border='1px solid #ddd', border_radius='4px',
                margin='2px 0', padding='2px'))

    def _build_controls(self, line, current_label, current_color):
        """Build all controls for this line."""
        controls = []

        # --- Name ---
        name_field = widgets.Text(
            value=current_label, description="Name:",
            style={"description_width": _DW},
            layout=widgets.Layout(width="95%"))

        def _on_name(change, l=line):
            old_label = l.get_label()
            new_label = change["new"]
            self._stack.execute(
                Command(l, "label", old_label, new_label,
                        description=f"{self._group.label} label"))
            pfx = self._header_prefix
            icon = "chevron-down" if self._is_expanded else "chevron-right"
            self._toggle_btn.icon = icon
            self._toggle_btn.description = f"  {pfx}{new_label}"
            _refresh_legend(l.axes)
            self._canvas.force_redraw()
        name_field.observe(_on_name, names="value")
        controls.append(name_field)

        # --- Color ---
        controls.append(self._build_color_section(line, current_color))

        # --- Width ---
        width = widgets.FloatSlider(
            value=round(line.get_linewidth(), 2),
            min=0.1, max=10, step=0.1, description="Width:", style=_SN)

        def _width_cb(change, l=line):
            self._stack.execute(
                Command(l, "linewidth", l.get_linewidth(), change["new"],
                        description=f"{self._group.label} width"))
            _refresh_legend(l.axes)
            self._canvas.force_redraw()
        width.observe(_width_cb, names="value")
        controls.append(_slider_num(width))

        # --- Style ---
        styles = [("solid", "-"), ("dashed", "--"),
                  ("dotted", ":"), ("dashdot", "-.")]
        _ls_map = {"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-."}
        current_ls = _ls_map.get(line.get_linestyle(), line.get_linestyle())
        if current_ls not in [v for _, v in styles]:
            current_ls = "-"
        style = widgets.Dropdown(
            options=styles, value=current_ls, description="Style:",
            style=_SN, layout=widgets.Layout(width="150px"))

        def _style_cb(change, l=line):
            self._stack.execute(
                Command(l, "linestyle", l.get_linestyle(), change["new"],
                        description=f"{self._group.label} style"))
            _refresh_legend(l.axes)
            self._canvas.force_redraw()
        style.observe(_style_cb, names="value")
        controls.append(style)

        # --- Alpha ---
        alpha_val = line.get_alpha()
        alpha_sl = widgets.FloatSlider(
            value=round(alpha_val if alpha_val is not None else 1.0, 2),
            min=0, max=1, step=0.05, description="Alpha:", style=_SN)

        def _alpha_cb(change, l=line):
            self._stack.execute(
                Command(l, "alpha", l.get_alpha(), change["new"],
                        description=f"{self._group.label} alpha"))
            _refresh_legend(l.axes)
            self._canvas.force_redraw()
        alpha_sl.observe(_alpha_cb, names="value")
        controls.append(_slider_num(alpha_sl))

        # --- Marker ---
        markers = [("none", ""), ("circle", "o"), ("square", "s"),
                   ("triangle", "^"), ("diamond", "D"), ("plus", "+"),
                   ("x", "x"), ("star", "*")]
        cur_m = line.get_marker()
        if cur_m not in [v for _, v in markers]:
            cur_m = ""
        marker = widgets.Dropdown(
            options=markers, value=cur_m, description="Marker:",
            style=_SN, layout=widgets.Layout(width="150px"))

        def _marker_cb(change, l=line):
            self._stack.execute(
                Command(l, "marker", l.get_marker(), change["new"],
                        description=f"{self._group.label} marker"))
            _refresh_legend(l.axes)
            self._canvas.force_redraw()
        marker.observe(_marker_cb, names="value")
        controls.append(marker)

        # --- Marker size ---
        marker_size = widgets.FloatSlider(
            value=round(line.get_markersize(), 2), min=0, max=20,
            step=0.5, description="Mkr sz:", style=_SN)

        def _ms_cb(change, l=line):
            self._stack.execute(
                Command(l, "markersize", l.get_markersize(), change["new"],
                        description=f"{self._group.label} marker size"))
            _refresh_legend(l.axes)
            self._canvas.force_redraw()
        marker_size.observe(_ms_cb, names="value")
        controls.append(_slider_num(marker_size))

        return controls

    def _build_color_section(self, line, current_color):
        """Color: click swatch → palette swatches + expand + colorwheel."""

        # --- Main color button (shows current color) ---
        color_btn = widgets.Button(
            layout=widgets.Layout(width='28px', height='28px',
                                  padding='0', min_width='28px'),
            tooltip="Click to choose color")
        color_btn.style.button_color = current_color

        color_row = widgets.HBox(
            [widgets.Label("Color:", layout=widgets.Layout(width='42px')),
             color_btn],
            layout=widgets.Layout(align_items='center', gap='4px'))

        # --- Palette: 10 swatches (compact row) + 10 more (expanded row) ---
        _cmap_name = ["tab10"]  # mutable, updated by ColormapPanel

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
        colors_20 = _get_palette_colors("tab10", 20)
        swatch_buttons = _make_swatches(colors_10)
        extra_buttons = _make_swatches(colors_20[10:])

        # CSS for smaller icons inside tool buttons
        _icon_css = widgets.HTML(
            '<style>'
            '.pb-sm-icon .fa { font-size: 8px !important; }'
            '</style>')

        # Expand / collapse button
        expand_btn = widgets.Button(
            icon="plus", tooltip="Show more colors",
            layout=widgets.Layout(width="24px", height="18px",
                                  padding="0", min_width="24px",
                                  margin="0 1px"))
        expand_btn.style.button_color = "#e0e0e0"
        expand_btn.add_class("pb-sm-icon")

        # Palette icon button — one click opens OS color dialog via JS
        palette_btn = widgets.Button(
            icon="tint", tooltip="Custom color...",
            layout=widgets.Layout(width="24px", height="18px",
                                  padding="0", min_width="24px",
                                  margin="0 1px"))
        palette_btn.style.button_color = "#e8e8e8"
        palette_btn.add_class("pb-sm-icon")

        # Hidden color picker (programmatically opened by JS click)
        _picker_cls = f"pb-picker-{id(line)}"
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

        # --- Sync logic ---
        _updating = [False]

        def _sync_controls(hex_val):
            """Update all visual controls without triggering callbacks."""
            _updating[0] = True
            try:
                color_btn.style.button_color = hex_val
                picker.value = hex_val
                self._color_indicator.value = _make_color_dot(hex_val)
            finally:
                _updating[0] = False

        def _apply(hex_val, l=line):
            self._stack.execute(
                Command(l, "color", to_hex(l.get_color()), hex_val,
                        description=f"{self._group.label} color"))
            _refresh_legend(l.axes)
            self._canvas.force_redraw()

        # Wire swatch clicks (both rows)
        def _wire_swatch(btn):
            def _on_swatch(b, _btn=btn):
                c = _btn.style.button_color
                _sync_controls(c)
                _apply(c)
            btn.on_click(_on_swatch)
        for b in swatch_buttons + extra_buttons:
            _wire_swatch(b)

        # Expand / collapse extra row
        def _on_expand(b):
            if extra_row.layout.display == 'none':
                extra_row.layout.display = ''
                expand_btn.icon = 'minus'
                expand_btn.tooltip = 'Show fewer colors'
            else:
                extra_row.layout.display = 'none'
                expand_btn.icon = 'plus'
                expand_btn.tooltip = 'Show more colors'
        expand_btn.on_click(_on_expand)

        # Picker changes
        def _from_picker(change):
            if _updating[0]:
                return
            _sync_controls(change["new"])
            _apply(change["new"])
        picker.observe(_from_picker, names="value")

        # Toggle palette panel
        def _toggle_palette(btn):
            if palette_panel.layout.display == 'none':
                palette_panel.layout.display = ''
            else:
                palette_panel.layout.display = 'none'
        color_btn.on_click(_toggle_palette)

        # External update hooks (used by ColormapPanel)
        self._update_color = _sync_controls

        def _ext_update_palette(cmap_name):
            _cmap_name[0] = cmap_name
            c10 = _get_palette_colors(cmap_name, 10)
            c20 = _get_palette_colors(cmap_name, 20)
            for i, btn in enumerate(swatch_buttons):
                btn.style.button_color = c10[i]
            for i, btn in enumerate(extra_buttons):
                btn.style.button_color = c20[10 + i]
        self._update_palette = _ext_update_palette

        return widgets.VBox([color_row, palette_panel])


class ColormapPanel:
    """Global colormap selector — compact header with expandable picker.

    Shows current colormap name + swatches. Click to expand full list.
    Selecting a colormap auto-applies it and collapses the list.
    """

    def __init__(self, groups: list[ArtistGroup], stack, canvas,
                 line_panels: list[LinePanel] | None = None):
        self._line_groups = [g for g in groups if g.plot_type == PlotType.LINE]
        self._stack = stack
        self._canvas = canvas
        self._line_panels = line_panels or []
        self._selected = "tab10"

    def apply(self, cmap_name: str) -> None:
        """Apply a colormap by name (public API for Preferred Defaults)."""
        self._do_apply(cmap_name)

    def build(self) -> widgets.Widget:
        if not self._line_groups:
            return widgets.HTML("<i>No lines to apply colormap to.</i>")

        # --- Compact header: name + large swatches on one row, Change below ---
        self._name_html = widgets.HTML(
            f"<b style='font-size:12px'>{self._selected}</b>")
        self._swatch_display = widgets.HTML(
            self._row_swatch(self._selected, n=10, size=20))

        header_row = widgets.HBox(
            [self._name_html, self._swatch_display],
            layout=widgets.Layout(align_items='center', gap='6px'))

        self._change_btn = widgets.Button(
            description="Change Colormap", icon="chevron-right",
            layout=widgets.Layout(width="100%", height="26px"))
        self._change_btn.style.button_color = '#f5f5f5'

        header = widgets.VBox([header_row, self._change_btn])

        # --- Expandable list of all colormaps ---
        self._row_btns: list[widgets.Button] = []
        self._swatch_btns: list[widgets.Button] = []
        rows = []
        for name in _COLORMAPS:
            # Small clickable name button
            btn = widgets.Button(
                description=name,
                layout=widgets.Layout(width='70px', height='22px',
                                      padding='0 2px', min_width='70px'),
                tooltip=name)
            btn.style.button_color = 'transparent'
            btn.style.font_weight = 'normal'
            btn.style.font_size = '10px'
            self._row_btns.append(btn)

            # Swatch gradient bar (HTML) + transparent overlay button
            swatch_html = self._row_swatch(name, n=20, size=20, stretch=True)
            swatch_w = widgets.HTML(
                swatch_html,
                layout=widgets.Layout(flex='1 1 auto', min_width='0'))
            # Invisible button overlaying the swatch for click detection
            swatch_click = widgets.Button(
                description='',
                layout=widgets.Layout(width='100%', height='20px',
                                      padding='0', margin='-20px 0 0 0'),
                tooltip=f"Apply {name}")
            swatch_click.style.button_color = 'transparent'
            self._swatch_btns.append(swatch_click)

            swatch_stack = widgets.VBox(
                [swatch_w, swatch_click],
                layout=widgets.Layout(flex='1 1 auto', min_width='0'))

            row = widgets.HBox(
                [btn, swatch_stack],
                layout=widgets.Layout(align_items='center', gap='6px',
                                      padding='4px 2px'))

            def _on_click(b, cmap_name=name):
                self._do_apply(cmap_name)
                self._cmap_list.layout.display = 'none'
                self._change_btn.icon = 'chevron-right'
            btn.on_click(_on_click)
            swatch_click.on_click(_on_click)
            rows.append(row)

        # Highlight initial selection
        for rb in self._row_btns:
            if rb.description == self._selected:
                rb.style.button_color = '#d4e6f1'
                rb.style.font_weight = 'bold'

        self._cmap_list = widgets.VBox(
            rows,
            layout=widgets.Layout(
                border='1px solid #ddd', padding='4px',
                display='none'))  # collapsed by default

        def _toggle_list(btn):
            if self._cmap_list.layout.display == 'none':
                self._cmap_list.layout.display = ''
                self._change_btn.icon = 'chevron-down'
            else:
                self._cmap_list.layout.display = 'none'
                self._change_btn.icon = 'chevron-right'
        self._change_btn.on_click(_toggle_list)

        return widgets.VBox([header, self._cmap_list])

    def _do_apply(self, cmap_name: str) -> None:
        """Apply colormap to all lines and update all UI elements."""
        self._selected = cmap_name
        cmap = matplotlib.colormaps.get_cmap(cmap_name)
        n = len(self._line_groups)
        cmds = []
        for i, group in enumerate(self._line_groups):
            line = group.artists[0]
            new_color = to_hex(_cmap_color(cmap, i, n))
            try:
                old_color = to_hex(line.get_color())
            except Exception:
                old_color = "#000000"
            cmds.append(Command(line, "color", old_color, new_color,
                                description=f"{group.label} color"))
        if cmds:
            self._stack.execute(BatchCommand(cmds, "Apply colormap"))
            # Rebuild legends to reflect new colors
            for ax_set in {g.artists[0].axes for g in self._line_groups}:
                _refresh_legend(ax_set)
            self._canvas.redraw()
            for i, panel in enumerate(self._line_panels):
                if hasattr(panel, "_update_color") and i < n:
                    color = to_hex(_cmap_color(cmap, i, n))
                    panel._update_color(color)
                if hasattr(panel, "_update_palette"):
                    panel._update_palette(cmap_name)

        # Update header display
        if hasattr(self, '_name_html'):
            self._name_html.value = (
                f"<b style='font-size:12px'>{cmap_name}</b>")
            self._swatch_display.value = self._row_swatch(cmap_name, n=10, size=20)
        # Highlight selection in list
        if hasattr(self, '_row_btns'):
            for rb in self._row_btns:
                if rb.description == cmap_name:
                    rb.style.button_color = '#d4e6f1'
                    rb.style.font_weight = 'bold'
                else:
                    rb.style.button_color = 'transparent'
                    rb.style.font_weight = 'normal'

    @staticmethod
    def _row_swatch(cmap_name: str, n: int = 10, size: int = 18,
                    stretch: bool = False) -> str:
        """Generate inline HTML swatches for a colormap.

        If *stretch* is True, renders as a CSS-gradient bar that fills its
        container width.  Otherwise, renders n fixed-width squares of *size*
        pixels each (used for the compact header).
        """
        try:
            cmap = matplotlib.colormaps.get_cmap(cmap_name)
        except Exception:
            return ""
        if stretch:
            # Build a banded CSS gradient that fills full width
            stops = []
            for i in range(n):
                c = to_hex(_cmap_color(cmap, i, n))
                pct0 = round(i / n * 100, 2)
                pct1 = round((i + 1) / n * 100, 2)
                stops.append(f"{c} {pct0}%, {c} {pct1}%")
            grad = ", ".join(stops)
            return (f'<div style="width:100%;height:{size}px;'
                    f'border-radius:3px;background:linear-gradient('
                    f'to right, {grad})"></div>')
        # Fixed-width squares
        spans = []
        for i in range(n):
            c = to_hex(_cmap_color(cmap, i, n))
            spans.append(
                f'<span style="display:inline-block;width:{size}px;'
                f'height:{size}px;background:{c};margin:0;border-right:'
                f'1px solid rgba(0,0,0,0.05)"></span>')
        return ''.join(spans)
