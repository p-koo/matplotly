"""Scatter plot controls — collapsible per-collection sections with compact color UI."""
from __future__ import annotations

import ipywidgets as widgets
import numpy as np
from matplotlib.colors import to_hex
from matplotlib.markers import MarkerStyle

from .._commands import Command
from .._types import ArtistGroup
from ._base import ArtistPanel
from ._color_utils import (
    _DW, _NW, _SN,
    _get_palette_colors, _make_color_dot, _refresh_legend, _slider_num,
)


class ScatterPanel(ArtistPanel):
    _plot_number: int = 0  # set by _api.py before build()

    def build(self) -> widgets.Widget:
        coll = self._group.artists[0]
        current_label = coll.get_label() or self._group.label

        fc = coll.get_facecolor()
        try:
            current_color = to_hex(fc[0]) if len(fc) > 0 else "#1f77b4"
        except Exception:
            current_color = "#1f77b4"

        # --- Collapsible header ---
        num = self._plot_number or ""
        header_prefix = f"Scatter {num}: " if num else ""
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
        controls = self._build_controls(coll, current_label, current_color)
        controls_box = widgets.VBox(
            controls,
            layout=widgets.Layout(display='none', padding='2px 0 4px 12px'))
        self._controls_box = controls_box
        self._is_expanded = False

        def _toggle(btn, c=coll):
            self._is_expanded = not self._is_expanded
            lbl = c.get_label() or self._group.label
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

    _on_label_changed = None  # callback set by _api.py for legend label sync

    def _build_controls(self, coll, current_label, current_color):
        """Build all controls for this scatter collection."""
        controls = []
        self._edge_manual = False  # track if user manually changed edge color

        # Set initial edge color to match face color
        coll.set_edgecolor(current_color)

        # --- Name ---
        name_field = widgets.Text(
            value=current_label, description="Name:",
            style={"description_width": _DW},
            layout=widgets.Layout(width="95%"))

        def _on_name(change, c=coll):
            old_label = c.get_label()
            new_label = change["new"]
            self._stack.execute(
                Command(c, "label", old_label, new_label,
                        description=f"{self._group.label} label"))
            pfx = self._header_prefix
            icon = "chevron-down" if self._is_expanded else "chevron-right"
            self._toggle_btn.icon = icon
            self._toggle_btn.description = f"  {pfx}{new_label}"
            _refresh_legend(c.axes)
            if self._on_label_changed is not None:
                self._on_label_changed()
            self._canvas.force_redraw()
        name_field.observe(_on_name, names="value")
        controls.append(name_field)

        # --- Color (facecolor) ---
        controls.append(self._build_color_section(
            coll, current_color, "Color:", "facecolor",
            getter=lambda c: to_hex(c.get_facecolor()[0]) if len(c.get_facecolor()) > 0 else "#1f77b4",
            setter_name="set_facecolor",
            is_primary=True))

        # --- Edge color ---
        controls.append(self._build_color_section(
            coll, current_color, "Edge:", "edgecolor",
            getter=lambda c: to_hex(c.get_edgecolor()[0]) if len(c.get_edgecolor()) > 0 else "#000000",
            setter_name="set_edgecolor",
            is_primary=False,
            is_edge=True))

        # --- Size ---
        sizes = coll.get_sizes()
        avg_size = float(np.mean(sizes)) if len(sizes) > 0 else 20.0
        size_sl = widgets.FloatSlider(
            value=round(avg_size, 1), min=1, max=200, step=1,
            description="Size:", style=_SN)

        def _size_cb(change, c=coll):
            old = c.get_sizes().copy()
            new_val = change["new"]
            n_pts = len(c.get_offsets())
            def _apply():
                c.set_sizes([new_val] * n_pts)
            def _revert():
                c.set_sizes(old)
            self._stack.execute(
                Command(c, "sizes", old, new_val,
                        apply_fn=_apply, revert_fn=_revert,
                        description=f"{self._group.label} size"))
            _refresh_legend(c.axes)
            self._canvas.force_redraw()
        size_sl.observe(_size_cb, names="value")
        controls.append(_slider_num(size_sl))

        # --- Alpha ---
        alpha_val = coll.get_alpha()
        alpha_sl = widgets.FloatSlider(
            value=round(alpha_val if alpha_val is not None else 1.0, 2),
            min=0, max=1, step=0.05, description="Alpha:", style=_SN)

        def _alpha_cb(change, c=coll):
            self._stack.execute(
                Command(c, "alpha", c.get_alpha(), change["new"],
                        description=f"{self._group.label} alpha"))
            _refresh_legend(c.axes)
            self._canvas.force_redraw()
        alpha_sl.observe(_alpha_cb, names="value")
        controls.append(_slider_num(alpha_sl))

        # --- Marker ---
        markers = [("circle", "o"), ("square", "s"), ("triangle", "^"),
                   ("diamond", "D"), ("plus", "+"), ("x", "x"),
                   ("star", "*"), ("point", ".")]
        marker_dd = widgets.Dropdown(
            options=markers, value="o", description="Marker:",
            style=_SN, layout=widgets.Layout(width="150px"))

        def _marker_cb(change, c=coll):
            old_paths = c.get_paths()
            new_marker = MarkerStyle(change["new"])
            new_path = new_marker.get_path().transformed(
                new_marker.get_transform())
            def _apply():
                c.set_paths([new_path])
            def _revert():
                c.set_paths(old_paths)
            self._stack.execute(
                Command(c, "paths", old_paths, change["new"],
                        apply_fn=_apply, revert_fn=_revert,
                        description=f"{self._group.label} marker"))
            _refresh_legend(c.axes)
            self._canvas.force_redraw()
        marker_dd.observe(_marker_cb, names="value")
        controls.append(marker_dd)

        # --- Edge width ---
        lw = coll.get_linewidths()
        edge_w_sl = widgets.FloatSlider(
            value=round(float(lw[0]) if len(lw) > 0 else 1.0, 2),
            min=0, max=5, step=0.1, description="Edge w:", style=_SN)

        def _ew_cb(change, c=coll):
            old = c.get_linewidths().copy()
            new_val = change["new"]
            def _apply():
                c.set_linewidths([new_val])
            def _revert():
                c.set_linewidths(old)
            self._stack.execute(
                Command(c, "linewidths", old, new_val,
                        apply_fn=_apply, revert_fn=_revert,
                        description=f"{self._group.label} edge width"))
            _refresh_legend(c.axes)
            self._canvas.force_redraw()
        edge_w_sl.observe(_ew_cb, names="value")
        controls.append(_slider_num(edge_w_sl))

        return controls

    def _build_color_section(self, coll, current_color, label, prop_name,
                             getter, setter_name, is_primary,
                             is_edge=False):
        """Color section: click swatch -> palette swatches + expand + colorwheel.

        Mirrors LinePanel._build_color_section. If is_primary, exposes
        self._update_color and self._update_palette for ColormapPanel.
        """
        color_btn = widgets.Button(
            layout=widgets.Layout(width='28px', height='28px',
                                  padding='0', min_width='28px'),
            tooltip="Click to choose color")
        color_btn.style.button_color = current_color

        color_row = widgets.HBox(
            [widgets.Label(label, layout=widgets.Layout(width='42px')),
             color_btn],
            layout=widgets.Layout(align_items='center', gap='4px'))

        # --- Palette: 10 swatches (compact row) + 10 more (expanded row) ---
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

        _picker_cls = f"pb-picker-{id(coll)}-{prop_name}"
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
                if is_primary:
                    self._color_indicator.value = _make_color_dot(hex_val)
            finally:
                _updating[0] = False

        def _apply(hex_val, c=coll):
            old = getattr(c, f"get_{prop_name}")().copy()
            setter = getattr(c, setter_name)
            def _do_apply():
                setter(hex_val)
            def _do_revert():
                setter(old)
            self._stack.execute(
                Command(c, prop_name, old, hex_val,
                        apply_fn=_do_apply, revert_fn=_do_revert,
                        description=f"{self._group.label} {prop_name}"))
            _refresh_legend(c.axes)
            if is_primary and hasattr(self, '_marginals'):
                self._marginals.sync_colors()
            # Sync edge color to match face color (unless manually overridden)
            if is_primary and not self._edge_manual:
                c.set_edgecolor(hex_val)
                if hasattr(self, '_sync_edge_ui'):
                    self._sync_edge_ui(hex_val)
            if is_edge:
                self._edge_manual = True
            self._canvas.force_redraw()

        # Wire swatch clicks
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
                c20 = _get_palette_colors(cname, 20)
                for i, btn in enumerate(swatch_buttons):
                    btn.style.button_color = c20[i]
                for i, btn in enumerate(extra_buttons):
                    btn.style.button_color = c20[10 + i]
                extra_row.layout.display = ''
                expand_btn.icon = 'minus'
                expand_btn.tooltip = 'Show fewer colors'
            else:
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

        # Edge color UI sync — store so primary color can update it
        if is_edge:
            self._sync_edge_ui = _sync_controls

        # External update hooks (used by ColormapPanel) — primary color only
        if is_primary:
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
