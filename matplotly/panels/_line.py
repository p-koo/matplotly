"""Line plot controls — collapsible per-curve sections with compact color UI."""
from __future__ import annotations

import ipywidgets as widgets
import matplotlib
from matplotlib.colors import to_hex

from .._commands import BatchCommand, Command
from .._types import ArtistGroup, PlotType
from ._base import ArtistPanel
from ._color_utils import (
    _COLORMAPS, _DW, _NW, _SN,
    _cmap_color, _get_palette_colors, _make_color_dot,
    _refresh_legend, _slider_num,
)


class LinePanel(ArtistPanel):
    _plot_number: int = 0  # set by _api.py before build()
    _on_label_changed = None  # callback set by _api.py for legend label sync

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
            if self._on_label_changed is not None:
                self._on_label_changed()
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
        swatch_buttons = _make_swatches(colors_10)
        colors_20 = _get_palette_colors("tab10", 20)
        extra_buttons = _make_swatches(colors_20[10:])

        # CSS: match swatch size, nudge FA icon up to center it
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

        # Expand / collapse button (same size as swatch: 18x16)
        expand_btn = widgets.Button(
            icon="plus", tooltip="Show more colors",
            layout=widgets.Layout(width="18px", height="16px",
                                  padding="0", min_width="18px",
                                  margin="1px"))
        expand_btn.style.button_color = "#e0e0e0"
        expand_btn.add_class("pb-swatch-btn")

        # Palette icon button (same size as swatch: 18x16)
        palette_btn = widgets.Button(
            icon="paint-brush", tooltip="Custom color...",
            layout=widgets.Layout(width="18px", height="16px",
                                  padding="0", min_width="18px",
                                  margin="1px"))
        palette_btn.style.button_color = "#e8e8e8"
        palette_btn.add_class("pb-swatch-btn")

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
            cname = _cmap_name[0]
            if extra_row.layout.display == 'none':
                # Expanding: switch row 1 to first 10 of 20-point sampling
                c20 = _get_palette_colors(cname, 20)
                for i, btn in enumerate(swatch_buttons):
                    btn.style.button_color = c20[i]
                for i, btn in enumerate(extra_buttons):
                    btn.style.button_color = c20[10 + i]
                extra_row.layout.display = ''
                expand_btn.icon = 'minus'
                expand_btn.tooltip = 'Show fewer colors'
            else:
                # Collapsing: switch row 1 back to 10-point full-range sampling
                c10 = _get_palette_colors(cname, 10)
                for i, btn in enumerate(swatch_buttons):
                    btn.style.button_color = c10[i]
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
                # Pre-compute row 2 for when it gets expanded
                c20 = _get_palette_colors(cmap_name, 20)
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
                 line_panels: list | None = None):
        self._color_groups = [g for g in groups
                              if g.plot_type in (PlotType.LINE, PlotType.SCATTER,
                                                 PlotType.HISTOGRAM,
                                                 PlotType.BAR,
                                                 PlotType.GROUPED_BAR,
                                                 PlotType.BOXPLOT,
                                                 PlotType.VIOLIN,
                                                 PlotType.ERRORBAR)]
        self._stack = stack
        self._canvas = canvas
        self._line_panels = line_panels or []
        self._selected = "tab10"

    def apply(self, cmap_name: str) -> None:
        """Apply a colormap by name (public API for Preferred Defaults)."""
        self._do_apply(cmap_name)

    def build(self) -> widgets.Widget:
        if not self._color_groups:
            return widgets.HTML("<i>No series to apply colormap to.</i>")

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
            swatch_html = self._row_swatch(name, n=10, size=20, stretch=True)
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
        """Apply colormap to all lines/scatter and update all UI elements."""
        self._selected = cmap_name
        cmap = matplotlib.colormaps.get_cmap(cmap_name)
        n = len(self._color_groups)
        cmds = []
        for i, group in enumerate(self._color_groups):
            new_color = to_hex(_cmap_color(cmap, i, n))
            if group.plot_type == PlotType.LINE:
                artist = group.artists[0] if group.artists else None
                if artist is None:
                    continue
                try:
                    old_color = to_hex(artist.get_color())
                except Exception:
                    old_color = "#000000"
                cmds.append(Command(artist, "color", old_color, new_color,
                                    description=f"{group.label} color"))
            elif group.plot_type == PlotType.SCATTER:
                artist = group.artists[0] if group.artists else None
                if artist is None:
                    continue
                old_fc = artist.get_facecolor().copy()
                _new = new_color  # capture for closures
                _old = old_fc
                _art = artist
                # Check if edge color should follow face color
                _panel = (self._line_panels[i]
                          if i < len(self._line_panels) else None)
                _sync_edge = (_panel is not None
                              and hasattr(_panel, '_edge_manual')
                              and not _panel._edge_manual)
                old_ec = artist.get_edgecolor().copy() if _sync_edge else None
                cmds.append(Command(
                    artist, "facecolor", old_fc, new_color,
                    apply_fn=lambda _a=_art, _n=_new, _se=_sync_edge: (
                        _a.set_facecolor(_n),
                        _a.set_edgecolor(_n) if _se else None),
                    revert_fn=lambda _a=_art, _o=_old, _se=_sync_edge, _oe=old_ec: (
                        _a.set_facecolor(_o),
                        _a.set_edgecolor(_oe) if _se else None),
                    description=f"{group.label} color"))
            elif group.plot_type == PlotType.HISTOGRAM:
                # Apply color to all patches in the histogram
                if not group.artists:
                    continue
                artist = group.artists[0]
                _patches = group.artists
                _new = new_color
                _old_colors = [to_hex(p.get_facecolor()) for p in _patches]
                def _apply_hist(_ps=_patches, _c=_new):
                    for p in _ps:
                        p.set_facecolor(_c)
                def _revert_hist(_ps=_patches, _oc=_old_colors):
                    for p, c in zip(_ps, _oc):
                        p.set_facecolor(c)
                cmds.append(Command(
                    artist, "facecolor", _old_colors, new_color,
                    apply_fn=_apply_hist, revert_fn=_revert_hist,
                    description=f"{group.label} color"))
                # Also update the panel's internal color state
                _panel = (self._line_panels[i]
                          if i < len(self._line_panels) else None)
                if _panel is not None and hasattr(_panel, '_color'):
                    _panel._color = new_color
            elif group.plot_type in (PlotType.BAR, PlotType.GROUPED_BAR):
                # Apply color to all patches in the bar group
                if not group.artists:
                    continue
                artist = group.artists[0]
                _patches = group.artists
                _new = new_color
                _old_colors = [to_hex(p.get_facecolor()) for p in _patches]
                def _apply_bar(_ps=_patches, _c=_new):
                    for p in _ps:
                        p.set_facecolor(_c)
                def _revert_bar(_ps=_patches, _oc=_old_colors):
                    for p, c in zip(_ps, _oc):
                        p.set_facecolor(c)
                cmds.append(Command(
                    artist, "facecolor", _old_colors, new_color,
                    apply_fn=_apply_bar, revert_fn=_revert_bar,
                    description=f"{group.label} color"))
                _panel = (self._line_panels[i]
                          if i < len(self._line_panels) else None)
                if _panel is not None and hasattr(_panel, '_color'):
                    _panel._color = new_color
            elif group.plot_type in (PlotType.BOXPLOT, PlotType.VIOLIN):
                # Apply color to distribution panel (no artist needed —
                # originals are cleared by _redraw, colours live on panel)
                _panel = (self._line_panels[i]
                          if i < len(self._line_panels) else None)
                if _panel is not None:
                    _old_bc = getattr(_panel, '_box_color', '#1f77b4')
                    _new_c = new_color
                    _p = _panel
                    def _apply_dist(_p=_p, _c=_new_c):
                        _p._box_color = _c
                        _p._violin_color = _c
                        _p._jitter_color = _c
                        if hasattr(_p, '_shared_panel') and _p._shared_panel:
                            _p._shared_panel._redraw()
                    def _revert_dist(_p=_p, _c=_old_bc):
                        _p._box_color = _c
                        _p._violin_color = _c
                        _p._jitter_color = _c
                        if hasattr(_p, '_shared_panel') and _p._shared_panel:
                            _p._shared_panel._redraw()
                    # Use group.axes as artist placeholder (original
                    # artists are removed by _redraw).
                    cmds.append(Command(
                        group.axes, "color", _old_bc, new_color,
                        apply_fn=_apply_dist, revert_fn=_revert_dist,
                        description=f"{group.label} color"))
            elif group.plot_type == PlotType.ERRORBAR:
                _panel = (self._line_panels[i]
                          if i < len(self._line_panels) else None)
                if _panel is not None:
                    _old_ec = getattr(_panel, '_bar_color', '#1f77b4')
                    _new_c = new_color
                    _p = _panel
                    def _apply_eb(_p=_p, _c=_new_c):
                        _p._bar_color = _c
                        _p._marker_color = _c
                        _p._line_color = _c
                        _p._shade_color = _c
                        _p._redraw()
                    def _revert_eb(_p=_p, _c=_old_ec):
                        _p._bar_color = _c
                        _p._marker_color = _c
                        _p._line_color = _c
                        _p._shade_color = _c
                        _p._redraw()
                    cmds.append(Command(
                        group.axes, "color", _old_ec, new_color,
                        apply_fn=_apply_eb, revert_fn=_revert_eb,
                        description=f"{group.label} color"))
        if cmds:
            self._stack.execute(BatchCommand(cmds, "Apply colormap"))
            for ax_set in {g.axes for g in self._color_groups}:
                _refresh_legend(ax_set)
            self._canvas.redraw()
            for i, panel in enumerate(self._line_panels):
                if hasattr(panel, "_update_color") and i < n:
                    color = to_hex(_cmap_color(cmap, i, n))
                    panel._update_color(color)
                    # Sync edge color UI if not manually overridden
                    if (hasattr(panel, '_edge_manual')
                            and not panel._edge_manual
                            and hasattr(panel, '_sync_edge_ui')):
                        panel._sync_edge_ui(color)
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
