"""Heatmap controls — per-heatmap + shared colorbar panel."""
from __future__ import annotations

import ipywidgets as widgets
import matplotlib
import numpy as np
from matplotlib.colors import (
    CenteredNorm, LogNorm, Normalize, SymLogNorm, to_hex,
)

from .._commands import Command
from .._types import ArtistGroup
from ._base import ArtistPanel
from ._color_utils import _DW, _NW, _SN, _cmap_color, _slider_num, cmap_color_btn


_CMAPS = [
    "viridis", "plasma", "inferno", "magma", "cividis",
    "coolwarm", "RdBu", "RdYlBu", "YlOrRd", "Blues", "Reds",
    "PuOr", "turbo", "Greys",
    "viridis_r", "plasma_r", "inferno_r", "magma_r", "cividis_r",
    "coolwarm_r", "RdBu_r", "RdYlBu_r", "YlOrRd_r", "Blues_r",
    "Reds_r", "PuOr_r", "turbo_r", "Greys_r",
]

_NORMS = ["linear", "log", "symlog", "centered"]

_INTERPS = ["nearest", "bilinear", "bicubic", "antialiased", "none"]

_ASPECTS = ["equal", "auto"]

_ANNOT_FMTS = [".2f", ".1f", ".0f", "d", ".2g"]

# CSS for swatch button icons (paint-brush, etc.)
_ICON_CSS = (
    '<style>'
    '.pb-swatch-btn button {'
    '  padding:0 !important; min-width:0 !important;'
    '  overflow:hidden !important;'
    '}'
    '.pb-swatch-btn .fa {'
    '  font-size:9px !important; position:relative !important;'
    '  top:-7px !important;'
    '}'
    '</style>')


def _row_swatch(cmap_name: str, n: int = 10, size: int = 18,
                stretch: bool = False) -> str:
    """Generate inline HTML swatches for a colormap."""
    try:
        cmap = matplotlib.colormaps.get_cmap(cmap_name)
    except Exception:
        return ""
    if stretch:
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
    spans = []
    for i in range(n):
        c = to_hex(_cmap_color(cmap, i, n))
        spans.append(
            f'<span style="display:inline-block;width:{size}px;'
            f'height:{size}px;background:{c};margin:0;border-right:'
            f'1px solid rgba(0,0,0,0.05)"></span>')
    return ''.join(spans)


class HeatmapPanel(ArtistPanel):
    """Per-heatmap controls: colormap, vmin/vmax, norm, interpolation, etc."""

    _plot_number: int = 0

    def build(self) -> widgets.Widget:
        meta = self._group.metadata
        mappable = meta['mappable']
        ax = self._group.axes
        data = meta.get('data')
        htype = meta.get('heatmap_type', 'imshow')

        # State
        self._mappable = mappable
        self._ax = ax
        self._data = data
        self._htype = htype
        self._cmap = meta.get('cmap', 'viridis')
        self._vmin = meta.get('vmin')
        self._vmax = meta.get('vmax')
        if data is not None:
            flat = np.asarray(data).ravel()
            finite = flat[np.isfinite(flat)]
            if len(finite) > 0:
                if self._vmin is None:
                    self._vmin = float(finite.min())
                if self._vmax is None:
                    self._vmax = float(finite.max())
        if self._vmin is None:
            self._vmin = 0.0
        if self._vmax is None:
            self._vmax = 1.0
        self._norm_type = meta.get('norm_type', 'Normalize')
        self._interpolation = meta.get('interpolation', 'nearest')
        self._aspect = meta.get('aspect', 'equal')
        self._alpha = meta.get('alpha', 1.0)
        self._annot_enabled = len(meta.get('annotations', [])) > 0
        self._annot_fmt = '.2f'
        self._annot_fontsize = 8.0
        self._annot_color = 'auto'
        self._annot_color_hex = '#ffffff'
        self._grid_enabled = False
        self._grid_lw = 1.0
        self._grid_color = '#ffffff'
        # Tick state
        self._xtick_show = True
        self._ytick_show = True
        self._xtick_labels_custom = ''
        self._ytick_labels_custom = ''

        # Header
        num = self._plot_number or ""
        header_prefix = f"Heatmap {num}: " if num else ""
        self._header_prefix = header_prefix
        label = self._group.label

        toggle_btn = widgets.Button(
            description=f"  {header_prefix}{label}",
            icon="chevron-right",
            layout=widgets.Layout(width='100%', height='28px'))
        toggle_btn.style.button_color = '#f0f0f0'
        toggle_btn.style.font_weight = 'bold'
        self._toggle_btn = toggle_btn

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

        icon_css = widgets.HTML(
            _ICON_CSS,
            layout=widgets.Layout(height='0px', overflow='hidden'))

        self._widget = widgets.VBox(
            [icon_css, toggle_btn, controls_box],
            layout=widgets.Layout(
                border='1px solid #ddd', border_radius='4px',
                margin='2px 0', padding='2px'))
        return self._widget

    def _build_controls(self):
        controls = []

        # --- Colormap (visual picker) ---
        controls.append(self._build_cmap_picker())

        # --- vmin / vmax ---
        vmin_w = widgets.FloatText(
            value=round(self._vmin, 4), description="vmin:",
            style=_SN, layout=widgets.Layout(width="160px"))
        vmax_w = widgets.FloatText(
            value=round(self._vmax, 4), description="vmax:",
            style=_SN, layout=widgets.Layout(width="160px"))

        def _vmin_cb(change):
            self._vmin = change["new"]
            self._apply_norm()
        def _vmax_cb(change):
            self._vmax = change["new"]
            self._apply_norm()
        vmin_w.observe(_vmin_cb, names="value")
        vmax_w.observe(_vmax_cb, names="value")
        controls.append(widgets.HBox([vmin_w, vmax_w]))

        # --- Normalization ---
        norm_map = {
            'Normalize': 'linear', 'LogNorm': 'log',
            'SymLogNorm': 'symlog', 'CenteredNorm': 'centered',
        }
        cur_norm = norm_map.get(self._norm_type, 'linear')
        norm_dd = widgets.Dropdown(
            options=_NORMS, value=cur_norm,
            description="Norm:", style=_SN,
            layout=widgets.Layout(width="150px"))

        def _norm_cb(change):
            self._norm_type = change["new"]
            self._apply_norm()
        norm_dd.observe(_norm_cb, names="value")
        controls.append(norm_dd)

        # --- Interpolation (imshow only) ---
        if self._htype == 'imshow':
            interp_dd = widgets.Dropdown(
                options=_INTERPS,
                value=self._interpolation if self._interpolation in _INTERPS else 'nearest',
                description="Interp:", style=_SN,
                layout=widgets.Layout(width="170px"))

            def _interp_cb(change):
                self._interpolation = change["new"]
                self._mappable.set_interpolation(change["new"])
                self._update_heatmap_info()
                self._canvas.force_redraw()
            interp_dd.observe(_interp_cb, names="value")
            controls.append(interp_dd)

        # --- Aspect (imshow only) ---
        if self._htype == 'imshow':
            aspect_dd = widgets.Dropdown(
                options=_ASPECTS,
                value=str(self._aspect) if str(self._aspect) in _ASPECTS else 'equal',
                description="Aspect:", style=_SN,
                layout=widgets.Layout(width="150px"))

            def _aspect_cb(change):
                self._aspect = change["new"]
                self._ax.set_aspect(change["new"])
                self._update_heatmap_info()
                self._canvas.force_redraw()
            aspect_dd.observe(_aspect_cb, names="value")
            controls.append(aspect_dd)

        # --- Alpha ---
        alpha_sl = widgets.FloatSlider(
            value=self._alpha, min=0.0, max=1.0, step=0.05,
            description="Alpha:", style=_SN)

        def _alpha_cb(change):
            self._alpha = change["new"]
            self._mappable.set_alpha(change["new"])
            self._update_heatmap_info()
            self._canvas.force_redraw()
        alpha_sl.observe(_alpha_cb, names="value")
        controls.append(_slider_num(alpha_sl))

        # --- Annotations ---
        controls.append(self._build_annotation_section())

        # --- Grid ---
        controls.append(self._build_grid_section())

        # --- Tick Labels ---
        controls.append(self._build_tick_section())

        return controls

    # ------------------------------------------------------------------
    # Colormap picker (visual, like ColormapPanel in _line.py)
    # ------------------------------------------------------------------

    def _build_cmap_picker(self):
        """Visual colormap picker with gradient previews."""
        self._cmap_name_html = widgets.HTML(
            f"<b style='font-size:12px'>{self._cmap}</b>")
        self._cmap_swatch_display = widgets.HTML(
            _row_swatch(self._cmap, n=10, size=20))

        header_row = widgets.HBox(
            [self._cmap_name_html, self._cmap_swatch_display],
            layout=widgets.Layout(align_items='center', gap='6px'))

        change_btn = widgets.Button(
            description="Change Colormap", icon="chevron-right",
            layout=widgets.Layout(width="100%", height="26px"))
        change_btn.style.button_color = '#f5f5f5'

        # Build rows for all cmaps
        row_btns = []
        rows = []
        for name in _CMAPS:
            btn = widgets.Button(
                description=name,
                layout=widgets.Layout(width='80px', height='22px',
                                      padding='0 2px', min_width='80px'),
                tooltip=name)
            btn.style.button_color = 'transparent'
            btn.style.font_weight = 'normal'
            btn.style.font_size = '10px'
            row_btns.append(btn)

            swatch_html = _row_swatch(name, n=10, size=20, stretch=True)
            swatch_w = widgets.HTML(
                swatch_html,
                layout=widgets.Layout(flex='1 1 auto', min_width='0'))
            swatch_click = widgets.Button(
                description='',
                layout=widgets.Layout(width='100%', height='20px',
                                      padding='0', margin='-20px 0 0 0'),
                tooltip=f"Apply {name}")
            swatch_click.style.button_color = 'transparent'

            swatch_stack = widgets.VBox(
                [swatch_w, swatch_click],
                layout=widgets.Layout(flex='1 1 auto', min_width='0'))

            row = widgets.HBox(
                [btn, swatch_stack],
                layout=widgets.Layout(align_items='center', gap='6px',
                                      padding='4px 2px'))

            def _on_click(b, cmap_name=name):
                self._cmap = cmap_name
                self._apply_cmap()
                self._cmap_name_html.value = (
                    f"<b style='font-size:12px'>{cmap_name}</b>")
                self._cmap_swatch_display.value = _row_swatch(
                    cmap_name, n=10, size=20)
                for rb in row_btns:
                    if rb.description == cmap_name:
                        rb.style.button_color = '#d4e6f1'
                        rb.style.font_weight = 'bold'
                    else:
                        rb.style.button_color = 'transparent'
                        rb.style.font_weight = 'normal'
                cmap_list.layout.display = 'none'
                change_btn.icon = 'chevron-right'

            btn.on_click(_on_click)
            swatch_click.on_click(_on_click)
            rows.append(row)

        # Highlight initial selection
        for rb in row_btns:
            if rb.description == self._cmap:
                rb.style.button_color = '#d4e6f1'
                rb.style.font_weight = 'bold'

        cmap_list = widgets.VBox(
            rows,
            layout=widgets.Layout(
                border='1px solid #ddd', padding='4px',
                display='none'))

        def _toggle_list(btn):
            if cmap_list.layout.display == 'none':
                cmap_list.layout.display = ''
                change_btn.icon = 'chevron-down'
            else:
                cmap_list.layout.display = 'none'
                change_btn.icon = 'chevron-right'
        change_btn.on_click(_toggle_list)

        return widgets.VBox([header_row, change_btn, cmap_list])

    # ------------------------------------------------------------------
    # Annotations section with swatch-based color picker
    # ------------------------------------------------------------------

    def _build_annotation_section(self):
        """Annotation controls with swatch-based color picker."""
        annot_tog = widgets.ToggleButton(
            value=self._annot_enabled, description="Annotations",
            icon="font", layout=widgets.Layout(width="130px", height="28px"))

        annot_fmt_dd = widgets.Dropdown(
            options=_ANNOT_FMTS, value=self._annot_fmt,
            description="Fmt:", style=_SN,
            layout=widgets.Layout(width="130px"))

        annot_fs_sl = widgets.FloatSlider(
            value=self._annot_fontsize, min=4, max=20, step=0.5,
            description="Font sz:", style=_SN)

        # Auto-contrast toggle + swatch color picker
        auto_cb = widgets.Checkbox(
            value=(self._annot_color == 'auto'),
            description="Auto contrast", indent=False,
            layout=widgets.Layout(width="auto"))

        def _on_annot_color(hex_val):
            self._annot_color_hex = hex_val
            if not auto_cb.value:
                self._annot_color = hex_val
                if self._annot_enabled:
                    self._apply_annotations()

        color_btn, swatch_row = cmap_color_btn(
            self._annot_color_hex, _on_annot_color)

        color_section = widgets.VBox(
            [widgets.HBox(
                [widgets.Label("Color:",
                               layout=widgets.Layout(width='42px')),
                 color_btn],
                layout=widgets.Layout(align_items='center', gap='4px')),
             swatch_row],
            layout=widgets.Layout(
                display='none' if auto_cb.value else ''))

        def _auto_cb(change):
            if change["new"]:
                self._annot_color = 'auto'
                color_section.layout.display = 'none'
            else:
                self._annot_color = self._annot_color_hex
                color_section.layout.display = ''
            if self._annot_enabled:
                self._apply_annotations()
        auto_cb.observe(_auto_cb, names="value")

        annot_controls = widgets.VBox(
            [annot_fmt_dd,
             _slider_num(annot_fs_sl),
             auto_cb,
             color_section],
            layout=widgets.Layout(
                display='' if self._annot_enabled else 'none',
                padding='2px 0 0 12px'))

        def _annot_tog_cb(change):
            self._annot_enabled = change["new"]
            annot_controls.layout.display = '' if change["new"] else 'none'
            self._apply_annotations()
        annot_tog.observe(_annot_tog_cb, names="value")

        def _annot_fmt_cb(change):
            self._annot_fmt = change["new"]
            if self._annot_enabled:
                self._apply_annotations()
        annot_fmt_dd.observe(_annot_fmt_cb, names="value")

        def _annot_fs_cb(change):
            self._annot_fontsize = change["new"]
            if self._annot_enabled:
                self._apply_annotations()
        annot_fs_sl.observe(_annot_fs_cb, names="value")

        return widgets.VBox([annot_tog, annot_controls])

    # ------------------------------------------------------------------
    # Grid section with swatch-based color picker
    # ------------------------------------------------------------------

    def _build_grid_section(self):
        """Grid controls with swatch-based color picker."""
        grid_tog = widgets.ToggleButton(
            value=self._grid_enabled, description="Grid Lines",
            icon="th", layout=widgets.Layout(width="130px", height="28px"))

        grid_lw_sl = widgets.FloatSlider(
            value=self._grid_lw, min=0.5, max=5.0, step=0.5,
            description="Width:", style=_SN)

        def _on_grid_color(hex_val):
            self._grid_color = hex_val
            if self._grid_enabled:
                self._apply_grid()

        grid_color_btn, grid_swatch_row = cmap_color_btn(
            self._grid_color, _on_grid_color)

        grid_color_section = widgets.VBox(
            [widgets.HBox(
                [widgets.Label("Color:",
                               layout=widgets.Layout(width='42px')),
                 grid_color_btn],
                layout=widgets.Layout(align_items='center', gap='4px')),
             grid_swatch_row])

        grid_controls = widgets.VBox(
            [_slider_num(grid_lw_sl), grid_color_section],
            layout=widgets.Layout(
                display='' if self._grid_enabled else 'none',
                padding='2px 0 0 12px'))

        def _grid_tog_cb(change):
            self._grid_enabled = change["new"]
            grid_controls.layout.display = '' if change["new"] else 'none'
            self._apply_grid()
        grid_tog.observe(_grid_tog_cb, names="value")

        def _grid_lw_cb(change):
            self._grid_lw = change["new"]
            if self._grid_enabled:
                self._apply_grid()
        grid_lw_sl.observe(_grid_lw_cb, names="value")

        return widgets.VBox([grid_tog, grid_controls])

    # ------------------------------------------------------------------
    # Tick label section
    # ------------------------------------------------------------------

    def _build_tick_section(self):
        """Tick visibility and custom label controls."""
        header = widgets.HTML(
            "<b style='font-size:11px'>Tick Labels</b>",
            layout=widgets.Layout(margin='4px 0 2px 0'))

        xtick_show = widgets.Checkbox(
            value=self._xtick_show, description="Show X ticks",
            indent=False, layout=widgets.Layout(width="auto"))
        xtick_labels = widgets.Text(
            value=self._xtick_labels_custom,
            description="X labels:",
            style=_SN,
            layout=widgets.Layout(width="95%"),
            placeholder="comma-separated (e.g., A,B,C)")

        ytick_show = widgets.Checkbox(
            value=self._ytick_show, description="Show Y ticks",
            indent=False, layout=widgets.Layout(width="auto"))
        ytick_labels = widgets.Text(
            value=self._ytick_labels_custom,
            description="Y labels:",
            style=_SN,
            layout=widgets.Layout(width="95%"),
            placeholder="comma-separated (e.g., A,B,C)")

        def _xtick_show_cb(change):
            self._xtick_show = change["new"]
            self._apply_ticks()
        xtick_show.observe(_xtick_show_cb, names="value")

        def _ytick_show_cb(change):
            self._ytick_show = change["new"]
            self._apply_ticks()
        ytick_show.observe(_ytick_show_cb, names="value")

        def _xtick_labels_cb(change):
            self._xtick_labels_custom = change["new"]
            self._apply_ticks()
        xtick_labels.observe(_xtick_labels_cb, names="value")

        def _ytick_labels_cb(change):
            self._ytick_labels_custom = change["new"]
            self._apply_ticks()
        ytick_labels.observe(_ytick_labels_cb, names="value")

        return widgets.VBox([
            header,
            xtick_show, xtick_labels,
            ytick_show, ytick_labels,
        ])

    # ------------------------------------------------------------------
    # Store / update info on axes for code gen
    # ------------------------------------------------------------------

    def _store_heatmap_info(self):
        """Store heatmap info on axes for code generation."""
        info = self._build_info()
        if not hasattr(self._ax, '_matplotly_heatmap_info'):
            self._ax._matplotly_heatmap_info = []
        existing = self._ax._matplotly_heatmap_info
        for i, old in enumerate(existing):
            if old.get('mappable') is self._mappable:
                existing[i] = info
                return
        existing.append(info)

    def _update_heatmap_info(self):
        self._store_heatmap_info()

    def _build_info(self):
        data = self._data
        return {
            'mappable': self._mappable,
            'heatmap_type': self._htype,
            'data': data,
            'cmap': self._cmap,
            'vmin': self._vmin,
            'vmax': self._vmax,
            'norm_type': self._norm_type,
            'interpolation': self._interpolation if self._htype == 'imshow' else None,
            'aspect': self._aspect if self._htype == 'imshow' else None,
            'alpha': self._alpha,
            'annot_enabled': self._annot_enabled,
            'annot_fmt': self._annot_fmt,
            'annot_fontsize': self._annot_fontsize,
            'annot_color': self._annot_color,
            'grid_enabled': self._grid_enabled,
            'grid_lw': self._grid_lw,
            'grid_color': self._grid_color,
            'xtick_show': self._xtick_show,
            'ytick_show': self._ytick_show,
            'xtick_labels': self._xtick_labels_custom,
            'ytick_labels': self._ytick_labels_custom,
        }

    # ------------------------------------------------------------------
    # Apply methods
    # ------------------------------------------------------------------

    def _apply_cmap(self):
        self._mappable.set_cmap(self._cmap)
        self._update_heatmap_info()
        cb = getattr(self._mappable, 'colorbar', None)
        if cb is not None:
            cb.update_normal(self._mappable)
        self._canvas.force_redraw()

    def _apply_norm(self):
        norm_name = self._norm_type
        vmin, vmax = self._vmin, self._vmax
        if norm_name == 'log':
            vmin = max(vmin, 1e-10)
            norm = LogNorm(vmin=vmin, vmax=vmax)
        elif norm_name == 'symlog':
            norm = SymLogNorm(linthresh=1.0, vmin=vmin, vmax=vmax)
        elif norm_name == 'centered':
            norm = CenteredNorm(vcenter=0)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
        self._mappable.set_norm(norm)
        self._mappable.set_clim(vmin, vmax)
        self._update_heatmap_info()
        cb = getattr(self._mappable, 'colorbar', None)
        if cb is not None:
            cb.update_normal(self._mappable)
        self._canvas.force_redraw()

    def _apply_annotations(self):
        for txt in list(self._ax.texts):
            txt.remove()

        if self._annot_enabled and self._data is not None:
            data = np.asarray(self._data)
            if data.ndim < 2:
                side = int(np.sqrt(data.size))
                if side * side == data.size:
                    data = data.reshape(side, side)
                else:
                    self._update_heatmap_info()
                    self._canvas.force_redraw()
                    return
            nrows, ncols = data.shape
            vmin, vmax = self._vmin, self._vmax
            vmid = (vmin + vmax) / 2.0 if vmax > vmin else vmin

            for i in range(nrows):
                for j in range(ncols):
                    val = data[i, j]
                    if not np.isfinite(val):
                        continue
                    if self._annot_color == 'auto':
                        color = 'white' if val > vmid else 'black'
                    else:
                        color = self._annot_color
                    try:
                        txt = format(val, self._annot_fmt)
                    except (ValueError, TypeError):
                        txt = str(val)
                    self._ax.text(
                        j, i, txt,
                        ha='center', va='center',
                        fontsize=self._annot_fontsize,
                        color=color)

        self._update_heatmap_info()
        self._canvas.force_redraw()

    def _apply_grid(self):
        if self._data is None:
            self._canvas.force_redraw()
            return
        data = np.asarray(self._data)
        if data.ndim < 2:
            side = int(np.sqrt(data.size))
            if side * side == data.size:
                data = data.reshape(side, side)
            else:
                self._canvas.force_redraw()
                return
        nrows, ncols = data.shape

        if self._grid_enabled:
            self._ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
            self._ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
            self._ax.grid(
                which='minor', color=self._grid_color,
                linewidth=self._grid_lw, linestyle='-')
            self._ax.tick_params(which='minor', length=0)
        else:
            self._ax.grid(False, which='minor')

        self._update_heatmap_info()
        self._canvas.force_redraw()

    def _apply_ticks(self):
        """Apply tick visibility and custom labels."""
        if not self._xtick_show:
            self._ax.tick_params(axis='x', bottom=False, top=False,
                                labelbottom=False, labeltop=False)
        else:
            self._ax.tick_params(axis='x', bottom=True,
                                labelbottom=True)
            if self._xtick_labels_custom.strip():
                labels = [l.strip() for l in
                          self._xtick_labels_custom.split(',')]
                positions = list(range(len(labels)))
                self._ax.set_xticks(positions)
                self._ax.set_xticklabels(labels)

        if not self._ytick_show:
            self._ax.tick_params(axis='y', left=False, right=False,
                                labelleft=False, labelright=False)
        else:
            self._ax.tick_params(axis='y', left=True,
                                labelleft=True)
            if self._ytick_labels_custom.strip():
                labels = [l.strip() for l in
                          self._ytick_labels_custom.split(',')]
                positions = list(range(len(labels)))
                self._ax.set_yticks(positions)
                self._ax.set_yticklabels(labels)

        self._update_heatmap_info()
        self._canvas.force_redraw()

    # Callback for label changes (wired by _api.py)
    _on_label_changed = None


class HeatmapSharedPanel:
    """Shared colorbar controls for all heatmaps on one axes."""

    def __init__(self, heatmap_panels: list[HeatmapPanel], canvas):
        self._panels = heatmap_panels
        self._canvas = canvas
        self._mappable = heatmap_panels[0]._mappable
        self._ax = heatmap_panels[0]._ax
        self._fig = self._ax.figure

        meta = heatmap_panels[0]._group.metadata
        self._cbar_show = meta.get('has_colorbar', False)
        self._cbar_location = meta.get('colorbar_location', 'right')
        self._cbar_shrink = 1.0
        self._cbar_pad = 0.05
        self._cbar_label = meta.get('colorbar_label', '')
        self._cbar_label_fontsize = 12.0
        self._cbar_tick_fontsize = 10.0
        self._cbar_ref = getattr(self._mappable, 'colorbar', None)

    def build(self) -> widgets.Widget:
        controls = []

        header = widgets.HTML(
            "<b style='font-size:12px'>Colorbar</b>",
            layout=widgets.Layout(margin='4px 0'))
        controls.append(header)

        # Show toggle
        show_tog = widgets.ToggleButton(
            value=self._cbar_show, description="Show Colorbar",
            icon="bar-chart", layout=widgets.Layout(width="150px", height="28px"))

        # Location
        loc_dd = widgets.Dropdown(
            options=["right", "left", "top", "bottom"],
            value=self._cbar_location,
            description="Location:", style=_SN,
            layout=widgets.Layout(width="160px"))

        # Size (shrink)
        shrink_sl = widgets.FloatSlider(
            value=self._cbar_shrink, min=0.1, max=1.0, step=0.05,
            description="Size:", style=_SN)

        # Pad (gap between heatmap and colorbar)
        pad_sl = widgets.FloatSlider(
            value=self._cbar_pad, min=0.0, max=0.3, step=0.01,
            description="Pad:", style=_SN)

        label_w = widgets.Text(
            value=self._cbar_label, description="Label:",
            style=_SN, layout=widgets.Layout(width="95%"))

        label_fs_sl = widgets.FloatSlider(
            value=self._cbar_label_fontsize, min=6, max=20, step=0.5,
            description="Label sz:", style=_SN)

        tick_fs_sl = widgets.FloatSlider(
            value=self._cbar_tick_fontsize, min=6, max=20, step=0.5,
            description="Tick sz:", style=_SN)

        cbar_controls = widgets.VBox(
            [loc_dd, _slider_num(shrink_sl), _slider_num(pad_sl),
             label_w, _slider_num(label_fs_sl), _slider_num(tick_fs_sl)],
            layout=widgets.Layout(
                display='' if self._cbar_show else 'none',
                padding='2px 0 0 12px'))

        def _show_cb(change):
            self._cbar_show = change["new"]
            cbar_controls.layout.display = '' if change["new"] else 'none'
            self._redraw_colorbar()
        show_tog.observe(_show_cb, names="value")

        def _loc_cb(change):
            self._cbar_location = change["new"]
            if self._cbar_show:
                self._redraw_colorbar()
        loc_dd.observe(_loc_cb, names="value")

        def _shrink_cb(change):
            self._cbar_shrink = change["new"]
            if self._cbar_show:
                self._redraw_colorbar()
        shrink_sl.observe(_shrink_cb, names="value")

        def _pad_cb(change):
            self._cbar_pad = change["new"]
            if self._cbar_show:
                self._redraw_colorbar()
        pad_sl.observe(_pad_cb, names="value")

        def _label_cb(change):
            self._cbar_label = change["new"]
            if self._cbar_show and self._cbar_ref is not None:
                self._cbar_ref.set_label(
                    change["new"], fontsize=self._cbar_label_fontsize)
                self._store_colorbar_info()
                self._canvas.force_redraw()
        label_w.observe(_label_cb, names="value")

        def _label_fs_cb(change):
            self._cbar_label_fontsize = change["new"]
            if self._cbar_show and self._cbar_ref is not None:
                self._cbar_ref.set_label(
                    self._cbar_label, fontsize=change["new"])
                self._store_colorbar_info()
                self._canvas.force_redraw()
        label_fs_sl.observe(_label_fs_cb, names="value")

        def _tick_fs_cb(change):
            self._cbar_tick_fontsize = change["new"]
            if self._cbar_show and self._cbar_ref is not None:
                self._cbar_ref.ax.tick_params(labelsize=change["new"])
                self._store_colorbar_info()
                self._canvas.force_redraw()
        tick_fs_sl.observe(_tick_fs_cb, names="value")

        controls.append(show_tog)
        controls.append(cbar_controls)

        self._store_colorbar_info()

        return widgets.VBox(
            controls,
            layout=widgets.Layout(
                border='1px solid #cce', border_radius='4px',
                margin='2px 0', padding='4px'))

    def _redraw_colorbar(self):
        """Remove and recreate colorbar with current settings."""
        if self._cbar_ref is not None:
            try:
                self._cbar_ref.remove()
            except Exception:
                pass
            self._cbar_ref = None

        if self._cbar_show:
            self._cbar_ref = self._fig.colorbar(
                self._mappable, ax=self._ax,
                location=self._cbar_location,
                shrink=self._cbar_shrink,
                pad=self._cbar_pad)
            if self._cbar_label:
                self._cbar_ref.set_label(
                    self._cbar_label, fontsize=self._cbar_label_fontsize)
            self._cbar_ref.ax.tick_params(
                labelsize=self._cbar_tick_fontsize)

        self._store_colorbar_info()
        self._canvas.force_redraw()

    def _store_colorbar_info(self):
        """Store colorbar info on axes for code gen."""
        info = {
            'show': self._cbar_show,
            'location': self._cbar_location,
            'shrink': self._cbar_shrink,
            'pad': self._cbar_pad,
            'label': self._cbar_label,
            'label_fontsize': self._cbar_label_fontsize,
            'tick_fontsize': self._cbar_tick_fontsize,
        }
        self._ax._matplotly_colorbar_info = info
