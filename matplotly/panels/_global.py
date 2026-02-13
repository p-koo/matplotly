"""Global figure controls — fonts, axes/spines/ticks/grid, legend, layout."""
from __future__ import annotations

import ipywidgets as widgets
from matplotlib.colors import to_hex
from matplotlib.figure import Figure

from .._commands import BatchCommand, Command, CommandStack
from .._renderer import CanvasManager
from ._color_utils import _DW, _NW, _slider_num, _get_palette_colors, cmap_color_btn


_DD_SHORT = widgets.Layout(width="150px")   # short dropdowns (direction, style)
_DD_MED = widgets.Layout(width="210px")     # medium dropdowns (font, position, bg)
_GDW = "70px"                               # wider description_width for global panel
_STY = {"description_width": _GDW}


class GlobalPanel:
    """Controls that apply to the whole figure rather than individual artists."""

    def __init__(self, fig: Figure, stack: CommandStack, canvas: CanvasManager,
                 colormap_widget: widgets.Widget | None = None,
                 cmap_panel=None, multi_subplot: bool = False):
        self._fig = fig
        self._stack = stack
        self._canvas = canvas
        self._colormap_widget = colormap_widget
        self._cmap_panel = cmap_panel
        self._multi_subplot = multi_subplot
        self._per_subplot_panels: dict = {}  # set by _api.py for defaults propagation

    def _main_axes(self):
        """Return only non-marginal axes."""
        return [a for a in self._fig.get_axes()
                if not getattr(a, '_matplotly_marginal', False)]

    def build(self) -> widgets.Widget:
        if self._multi_subplot:
            return self._build_multi_subplot()
        return self._build_single_subplot()

    def _build_single_subplot(self) -> widgets.Widget:
        """Full single-subplot build — unchanged from original behavior."""
        self._legend_widget = self._build_legend()
        sections = [
            ("Figure Size", self._build_layout()),
            ("Axes & Grid", self._build_axes_grid()),
            ("Labels", self._build_typography()),
        ]
        if self._colormap_widget is not None:
            sections.insert(0, ("Colormap", self._colormap_widget))

        # "Preferred Defaults" button — one-click clean publication style
        defaults_btn = widgets.Button(
            description=" Preferred Defaults",
            icon="magic",
            layout=widgets.Layout(width="100%", margin="4px 0"))
        defaults_btn.style.button_color = '#e8f4fd'
        defaults_btn.style.font_weight = 'bold'
        defaults_btn.on_click(self._apply_preferred_defaults)

        accordion = widgets.Accordion(
            children=[s[1] for s in sections],
        )
        for i, (title, _) in enumerate(sections):
            accordion.set_title(i, title)
        accordion.selected_index = None
        _icon_css = widgets.HTML(
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
        return widgets.VBox([_icon_css, defaults_btn, accordion])

    def _build_multi_subplot(self) -> widgets.Widget:
        """Slim global build for multi-subplot figures.
        Labels/axes/legend are per-subplot; global keeps fonts, ticks, spacing."""
        self._legend_widget = widgets.VBox([])  # empty placeholder
        sections = [
            ("Figure Size", self._build_layout()),
            ("Labels", self._build_fonts_only()),
        ]
        if self._colormap_widget is not None:
            sections.insert(0, ("Colormap", self._colormap_widget))

        defaults_btn = widgets.Button(
            description=" Preferred Defaults",
            icon="magic",
            layout=widgets.Layout(width="100%", margin="4px 0"))
        defaults_btn.style.button_color = '#e8f4fd'
        defaults_btn.style.font_weight = 'bold'
        defaults_btn.on_click(self._apply_preferred_defaults)

        accordion = widgets.Accordion(
            children=[s[1] for s in sections],
        )
        for i, (title, _) in enumerate(sections):
            accordion.set_title(i, title)
        accordion.selected_index = None
        _icon_css = widgets.HTML(
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
        return widgets.VBox([_icon_css, defaults_btn, accordion])

    def _apply_preferred_defaults(self, _btn):
        """Apply clean publication defaults in one shot: Arial 10pt uniform,
        no top/right spines, legend frame off, tab10 colormap.
        For scatter plots: equal axes, equal figure size, equal ticks."""
        orig_redraw = self._canvas.redraw
        orig_force = self._canvas.force_redraw
        self._canvas.redraw = lambda: None  # suppress
        self._canvas.force_redraw = lambda: None  # suppress
        try:
            self._font_dd.value = "Arial"
            self._title_size_sl.value = 10.0
            self._label_size_sl.value = 10.0
            self._tick_size_sl.value = 10.0

            if self._cmap_panel is not None:
                self._cmap_panel.apply("tab10")

            # Scale figure to 75% of current size
            w, h = self._fig.get_size_inches()
            self._width_sl.value = round(w * 0.75, 2)
            self._height_sl.value = round(h * 0.75, 2)

            if self._multi_subplot:
                # Multi-subplot: per-subplot panels handle spines/legend
                for psp in self._per_subplot_panels.values():
                    psp.apply_defaults()
            else:
                self._spine_top_cb.value = False
                self._spine_right_cb.value = False
                self._spine_bottom_cb.value = True
                self._spine_left_cb.value = True
                self._legend_toggle.value = True
                self._frame_toggle.value = False
                self._legend_fontsize_sl.value = 10.0

                # Scatter-specific: equal axes, square figure, synced ticks
                from matplotlib.collections import PathCollection
                has_scatter = any(
                    isinstance(c, PathCollection)
                    for ax in self._main_axes() for c in ax.collections)
                if has_scatter:
                    self._equal_cb.value = True  # triggers _apply_equal
                    w, h = self._fig.get_size_inches()
                    side = round(max(w, h), 1)
                    self._width_sl.value = side
                    self._height_sl.value = side
                    xs = self._x_step.value
                    ys = self._y_step.value
                    step = max(xs, ys) if max(xs, ys) > 0 else 0
                    if step > 0:
                        self._x_step.value = step
                        self._y_step.value = step
        finally:
            self._canvas.redraw = orig_redraw
            self._canvas.force_redraw = orig_force
            self._canvas.force_redraw()

    # ------------------------------------------------------------------
    # Reusable color picker with colormap swatches + colorwheel
    # ------------------------------------------------------------------

    def _cmap_color_btn(self, initial_color, on_change_fn):
        """Compact color button; click toggles colormap swatch row + palette."""
        return cmap_color_btn(initial_color, on_change_fn, self._cmap_panel)

    # ------------------------------------------------------------------
    # Fonts-only — for multi-subplot mode (global font properties)
    # ------------------------------------------------------------------

    def _build_fonts_only(self) -> widgets.Widget:
        """Global font properties only — text content/color/bold/italic moved to PerSubplotPanel."""
        ax0 = self._main_axes()[0] if self._main_axes() else None

        cur_family = "sans-serif"
        if ax0:
            fam = ax0.title.get_fontfamily()
            if isinstance(fam, list) and fam:
                cur_family = fam[0]
            elif isinstance(fam, str):
                cur_family = fam

        cur_title_sz = round(ax0.title.get_fontsize(), 1) if ax0 else 12.0
        cur_label_sz = round(ax0.xaxis.label.get_fontsize(), 1) if ax0 else 10.0
        cur_tick_sz = 10.0
        if ax0:
            xticks = ax0.get_xticklabels()
            if xticks:
                cur_tick_sz = round(xticks[0].get_fontsize(), 1)

        families = ["Arial", "sans-serif", "serif", "monospace", "DejaVu Sans",
                     "Helvetica", "Times New Roman", "Courier New"]
        if cur_family not in families:
            families.insert(0, cur_family)

        self._font_dd = widgets.Dropdown(options=families, value=cur_family,
                                          description="Font:",
                                          style=_STY, layout=_DD_MED)
        self._title_size_sl = widgets.FloatSlider(
            value=cur_title_sz, min=4, max=36, step=0.5,
            description="Title sz:", style=_STY)

        import matplotlib as _mpl
        _default_pad = _mpl.rcParams.get('axes.titlepad', 6.0)
        self._title_pad_sl = widgets.FloatSlider(
            value=_default_pad, min=0, max=40, step=1,
            description="Title pad:", style=_STY)

        def _on_title_pad(change):
            for ax in self._main_axes():
                ax._set_title_offset_trans(change["new"])
            self._canvas.redraw()
        self._title_pad_sl.observe(_on_title_pad, names="value")

        self._label_size_sl = widgets.FloatSlider(
            value=cur_label_sz, min=4, max=36, step=0.5,
            description="Label sz:", style=_STY)

        _xl_pad = ax0.xaxis.labelpad if ax0 else 4.0
        _yl_pad = ax0.yaxis.labelpad if ax0 else 4.0

        self._xlabel_pad_sl = widgets.FloatSlider(
            value=_xl_pad, min=0, max=40, step=1,
            description="X lab pad:", style=_STY)
        self._ylabel_pad_sl = widgets.FloatSlider(
            value=_yl_pad, min=0, max=40, step=1,
            description="Y lab pad:", style=_STY)

        def _on_xlabel_pad(change):
            for ax in self._main_axes():
                ax.xaxis.labelpad = change["new"]
            self._canvas.redraw()

        def _on_ylabel_pad(change):
            for ax in self._main_axes():
                ax.yaxis.labelpad = change["new"]
            self._canvas.redraw()

        self._xlabel_pad_sl.observe(_on_xlabel_pad, names="value")
        self._ylabel_pad_sl.observe(_on_ylabel_pad, names="value")

        self._tick_size_sl = widgets.FloatSlider(
            value=cur_tick_sz, min=4, max=36, step=0.5,
            description="Tick sz:", style=_STY)

        def _on_font(change):
            cmds = []
            for ax in self._main_axes():
                for text in [ax.title, ax.xaxis.label, ax.yaxis.label]:
                    cmds.append(Command(text, "fontfamily", text.get_fontfamily(), change["new"]))
                for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                    cmds.append(Command(lbl, "fontfamily", lbl.get_fontfamily(), change["new"]))
                leg = ax.get_legend()
                if leg is not None:
                    for text in leg.get_texts():
                        cmds.append(Command(text, "fontfamily", text.get_fontfamily(), change["new"]))
                    ltitle = leg.get_title()
                    if ltitle and ltitle.get_text():
                        cmds.append(Command(ltitle, "fontfamily", ltitle.get_fontfamily(), change["new"]))
            if cmds:
                self._stack.execute(BatchCommand(cmds, "Change font family"))
                self._canvas.force_redraw()

        def _make_size_cb(getter, desc):
            def _cb(change):
                cmds = []
                for ax in self._main_axes():
                    for t in getter(ax):
                        cmds.append(Command(t, "fontsize", t.get_fontsize(), change["new"]))
                if cmds:
                    self._stack.execute(BatchCommand(cmds, desc))
                    self._canvas.redraw()
            return _cb

        self._font_dd.observe(_on_font, names="value")
        self._title_size_sl.observe(
            _make_size_cb(lambda ax: [ax.title], "Title size"), names="value")
        self._label_size_sl.observe(
            _make_size_cb(lambda ax: [ax.xaxis.label, ax.yaxis.label], "Label size"),
            names="value")
        self._tick_size_sl.observe(
            _make_size_cb(lambda ax: ax.get_xticklabels() + ax.get_yticklabels(),
                          "Tick size"), names="value")

        # Tick label colors
        cur_xtick_c = '#000000'
        cur_ytick_c = '#000000'
        if ax0:
            _xt = ax0.get_xticklabels()
            if _xt:
                try:
                    cur_xtick_c = to_hex(_xt[0].get_color())
                except Exception:
                    pass
            _yt = ax0.get_yticklabels()
            if _yt:
                try:
                    cur_ytick_c = to_hex(_yt[0].get_color())
                except Exception:
                    pass

        xtick_cp = widgets.ColorPicker(
            value=cur_xtick_c, concise=True,
            layout=widgets.Layout(width='28px', height='24px'))
        ytick_cp = widgets.ColorPicker(
            value=cur_ytick_c, concise=True,
            layout=widgets.Layout(width='28px', height='24px'))

        def _on_xtick_c(change):
            for ax in self._main_axes():
                ax.tick_params(axis='x', labelcolor=change["new"])
            self._canvas.redraw()

        def _on_ytick_c(change):
            for ax in self._main_axes():
                ax.tick_params(axis='y', labelcolor=change["new"])
            self._canvas.redraw()

        xtick_cp.observe(_on_xtick_c, names="value")
        ytick_cp.observe(_on_ytick_c, names="value")

        tick_color_row = widgets.HBox(
            [widgets.Label("Tick color:", layout=widgets.Layout(width='52px')),
             widgets.Label("X", layout=widgets.Layout(width='12px', margin='0 0 0 6px')),
             xtick_cp,
             widgets.Label("Y", layout=widgets.Layout(width='12px', margin='0 0 0 6px')),
             ytick_cp],
            layout=widgets.Layout(align_items='center', gap='2px'))

        return widgets.VBox([
            self._font_dd,
            _slider_num(self._title_size_sl, _GDW),
            _slider_num(self._title_pad_sl, _GDW),
            _slider_num(self._label_size_sl, _GDW),
            _slider_num(self._xlabel_pad_sl, _GDW),
            _slider_num(self._ylabel_pad_sl, _GDW),
            _slider_num(self._tick_size_sl, _GDW),
            tick_color_row,
        ])

    # ------------------------------------------------------------------
    # Tick & Spine params — for multi-subplot mode (global tick/spine)
    # ------------------------------------------------------------------

    def _build_tick_params(self) -> widgets.Widget:
        """Global tick direction/length/width and spine width — for multi-subplot."""
        ax0 = self._main_axes()[0] if self._main_axes() else None

        cur_tick_dir = "out"
        cur_tick_len = 4.0
        cur_tick_width = 1.0
        if ax0:
            xtick_objs = ax0.xaxis.get_major_ticks()
            if xtick_objs:
                tick = xtick_objs[0]
                tick_line = tick.tick1line
                marker = tick_line.get_marker()
                if marker == 2:
                    cur_tick_dir = "in"
                elif marker == 3:
                    cur_tick_dir = "out"
                if tick.tick2line.get_visible() and tick.tick1line.get_visible():
                    m1 = tick.tick1line.get_marker()
                    m2 = tick.tick2line.get_marker()
                    if m1 != m2:
                        cur_tick_dir = "inout"
                cur_tick_len = round(tick_line.get_markersize(), 1)
                cur_tick_width = round(tick_line.get_markeredgewidth(), 1)

        cur_spine_width = round(ax0.spines["bottom"].get_linewidth(), 1) if ax0 else 1.0

        self._tick_dir_dd = widgets.Dropdown(
            options=["out", "in", "inout"], value=cur_tick_dir,
            description="Dir:", style=_STY, layout=_DD_SHORT)
        self._tick_len_sl = widgets.FloatSlider(
            value=cur_tick_len, min=0, max=15, step=0.5,
            description="Length:", style=_STY)
        self._tick_width_sl = widgets.FloatSlider(
            value=cur_tick_width, min=0.1, max=4, step=0.1,
            description="Width:", style=_STY)
        self._spine_width_sl = widgets.FloatSlider(
            value=cur_spine_width, min=0.1, max=5.0, step=0.1,
            description="Spine W:", style=_STY)

        def _tick_cb(change):
            for ax in self._main_axes():
                ax.tick_params(direction=self._tick_dir_dd.value,
                               length=self._tick_len_sl.value,
                               width=self._tick_width_sl.value)
            self._canvas.redraw()
        self._tick_dir_dd.observe(_tick_cb, names="value")
        self._tick_len_sl.observe(_tick_cb, names="value")
        self._tick_width_sl.observe(_tick_cb, names="value")

        def _spine_width_cb(change):
            cmds = []
            for ax in self._main_axes():
                for name in ("top", "right", "bottom", "left"):
                    sp = ax.spines[name]
                    cmds.append(Command(sp, "linewidth", sp.get_linewidth(), change["new"]))
            self._stack.execute(BatchCommand(cmds, "Spine width"))
            self._canvas.redraw()
        self._spine_width_sl.observe(_spine_width_cb, names="value")

        return widgets.VBox([
            widgets.HTML("<b>Ticks</b>"),
            self._tick_dir_dd,
            _slider_num(self._tick_len_sl, _GDW),
            _slider_num(self._tick_width_sl, _GDW),
            widgets.HTML("<b>Spine Width</b>"),
            _slider_num(self._spine_width_sl, _GDW),
        ])

    # ------------------------------------------------------------------
    # Typography — reads actual figure state
    # ------------------------------------------------------------------

    def _build_typography(self) -> widgets.Widget:
        ax0 = self._main_axes()[0] if self._main_axes() else None

        # --- Text content (editable, clear to remove) ---
        cur_title = ax0.get_title() if ax0 else ""
        cur_xlabel = ax0.get_xlabel() if ax0 else ""
        cur_ylabel = ax0.get_ylabel() if ax0 else ""

        title_text = widgets.Text(
            value=cur_title, description="Title:", style=_STY,
            layout=widgets.Layout(width="95%"),
            placeholder="(empty to remove)")
        xlabel_text = widgets.Text(
            value=cur_xlabel, description="X label:", style=_STY,
            layout=widgets.Layout(width="95%"),
            placeholder="(empty to remove)")
        ylabel_text = widgets.Text(
            value=cur_ylabel, description="Y label:", style=_STY,
            layout=widgets.Layout(width="95%"),
            placeholder="(empty to remove)")

        def _on_title(change):
            if not self._main_axes():
                return
            ax = self._main_axes()[0]
            self._stack.execute(
                Command(ax.title, "text", ax.get_title(), change["new"],
                        description="Change title"))
            self._canvas.redraw()

        def _on_xlabel(change):
            if not self._main_axes():
                return
            ax = self._main_axes()[0]
            self._stack.execute(
                Command(ax.xaxis.label, "text", ax.get_xlabel(), change["new"],
                        description="Change X label"))
            self._canvas.redraw()

        def _on_ylabel(change):
            if not self._main_axes():
                return
            ax = self._main_axes()[0]
            self._stack.execute(
                Command(ax.yaxis.label, "text", ax.get_ylabel(), change["new"],
                        description="Change Y label"))
            self._canvas.redraw()

        title_text.observe(_on_title, names="value")
        xlabel_text.observe(_on_xlabel, names="value")
        ylabel_text.observe(_on_ylabel, names="value")

        # --- Text formatting (color / bold / italic) per label ---
        def _make_fmt(getter, prefix):
            t0 = getter(ax0) if ax0 else None
            try:
                cur_c = to_hex(t0.get_color()) if t0 else '#000000'
            except Exception:
                cur_c = '#000000'
            fw = t0.get_fontweight() if t0 else 'normal'
            cur_b = fw == 'bold' or (isinstance(fw, (int, float)) and fw >= 600)
            cur_i = (t0.get_fontstyle() == 'italic') if t0 else False

            def _on_c(hex_val):
                cmds = []
                for ax in self._main_axes():
                    t = getter(ax)
                    if t:
                        cmds.append(Command(t, "color",
                                            t.get_color(), hex_val))
                if cmds:
                    self._stack.execute(BatchCommand(cmds, f"{prefix} color"))
                    self._canvas.redraw()

            color_btn, swatch_row = self._cmap_color_btn(cur_c, _on_c)

            bb = widgets.ToggleButton(
                value=cur_b, description='B', tooltip='Bold',
                layout=widgets.Layout(width='32px', height='24px',
                                      padding='0', min_width='0'))
            bb.style.font_weight = 'bold'
            ib = widgets.ToggleButton(
                value=cur_i, description='I', tooltip='Italic',
                layout=widgets.Layout(width='32px', height='24px',
                                      padding='0', min_width='0'))
            ib.style.font_style = 'italic'

            def _on_b(change):
                w = 'bold' if change["new"] else 'normal'
                cmds = []
                for ax in self._main_axes():
                    t = getter(ax)
                    if t:
                        cmds.append(Command(t, "fontweight",
                                            t.get_fontweight(), w))
                if cmds:
                    self._stack.execute(BatchCommand(cmds, f"{prefix} weight"))
                    self._canvas.redraw()

            def _on_i(change):
                s = 'italic' if change["new"] else 'normal'
                cmds = []
                for ax in self._main_axes():
                    t = getter(ax)
                    if t:
                        cmds.append(Command(t, "fontstyle",
                                            t.get_fontstyle(), s))
                if cmds:
                    self._stack.execute(BatchCommand(cmds, f"{prefix} style"))
                    self._canvas.redraw()

            bb.observe(_on_b, names="value")
            ib.observe(_on_i, names="value")

            top_row = widgets.HBox(
                [color_btn, bb, ib],
                layout=widgets.Layout(align_items='center', gap='2px',
                                      padding='0 0 0 52px'))
            swatch_row.layout.padding = '0 0 0 52px'
            return widgets.VBox([top_row, swatch_row])

        title_fmt = _make_fmt(lambda ax: ax.title, "Title")
        xlabel_fmt = _make_fmt(lambda ax: ax.xaxis.label, "X label")
        ylabel_fmt = _make_fmt(lambda ax: ax.yaxis.label, "Y label")

        # --- Font family & sizes ---
        cur_family = "sans-serif"
        if ax0:
            fam = ax0.title.get_fontfamily()
            if isinstance(fam, list) and fam:
                cur_family = fam[0]
            elif isinstance(fam, str):
                cur_family = fam

        cur_title_sz = round(ax0.title.get_fontsize(), 1) if ax0 else 12.0
        cur_label_sz = round(ax0.xaxis.label.get_fontsize(), 1) if ax0 else 10.0
        cur_tick_sz = 10.0
        if ax0:
            xticks = ax0.get_xticklabels()
            if xticks:
                cur_tick_sz = round(xticks[0].get_fontsize(), 1)

        families = ["Arial", "sans-serif", "serif", "monospace", "DejaVu Sans",
                     "Helvetica", "Times New Roman", "Courier New"]
        if cur_family not in families:
            families.insert(0, cur_family)

        self._font_dd = widgets.Dropdown(options=families, value=cur_family,
                                          description="Font:",
                                          style=_STY, layout=_DD_MED)
        self._title_size_sl = widgets.FloatSlider(
            value=cur_title_sz, min=4, max=36, step=0.5,
            description="Title sz:", style=_STY)

        # Title pad — distance from axes top to title, in points
        import matplotlib as _mpl
        _default_pad = _mpl.rcParams.get('axes.titlepad', 6.0)
        self._title_pad_sl = widgets.FloatSlider(
            value=_default_pad, min=0, max=40, step=1,
            description="Title pad:", style=_STY)

        def _on_title_pad(change):
            for ax in self._main_axes():
                ax._set_title_offset_trans(change["new"])
            self._canvas.redraw()
        self._title_pad_sl.observe(_on_title_pad, names="value")

        self._label_size_sl = widgets.FloatSlider(
            value=cur_label_sz, min=4, max=36, step=0.5,
            description="Label sz:", style=_STY)

        # Label pad sliders
        _xl_pad = ax0.xaxis.labelpad if ax0 else 4.0
        _yl_pad = ax0.yaxis.labelpad if ax0 else 4.0

        self._xlabel_pad_sl = widgets.FloatSlider(
            value=_xl_pad, min=0, max=40, step=1,
            description="X lab pad:", style=_STY)
        self._ylabel_pad_sl = widgets.FloatSlider(
            value=_yl_pad, min=0, max=40, step=1,
            description="Y lab pad:", style=_STY)

        def _on_xlabel_pad(change):
            for ax in self._main_axes():
                ax.xaxis.labelpad = change["new"]
            self._canvas.redraw()

        def _on_ylabel_pad(change):
            for ax in self._main_axes():
                ax.yaxis.labelpad = change["new"]
            self._canvas.redraw()

        self._xlabel_pad_sl.observe(_on_xlabel_pad, names="value")
        self._ylabel_pad_sl.observe(_on_ylabel_pad, names="value")

        self._tick_size_sl = widgets.FloatSlider(
            value=cur_tick_sz, min=4, max=36, step=0.5,
            description="Tick sz:", style=_STY)

        def _on_font(change):
            cmds = []
            for ax in self._main_axes():
                for text in [ax.title, ax.xaxis.label, ax.yaxis.label]:
                    cmds.append(Command(text, "fontfamily", text.get_fontfamily(), change["new"]))
                for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                    cmds.append(Command(lbl, "fontfamily", lbl.get_fontfamily(), change["new"]))
                leg = ax.get_legend()
                if leg is not None:
                    for text in leg.get_texts():
                        cmds.append(Command(text, "fontfamily", text.get_fontfamily(), change["new"]))
                    ltitle = leg.get_title()
                    if ltitle and ltitle.get_text():
                        cmds.append(Command(ltitle, "fontfamily", ltitle.get_fontfamily(), change["new"]))
            if cmds:
                self._stack.execute(BatchCommand(cmds, "Change font family"))
                self._canvas.force_redraw()

        def _make_size_cb(getter, desc):
            def _cb(change):
                cmds = []
                for ax in self._main_axes():
                    for t in getter(ax):
                        cmds.append(Command(t, "fontsize", t.get_fontsize(), change["new"]))
                if cmds:
                    self._stack.execute(BatchCommand(cmds, desc))
                    self._canvas.redraw()
            return _cb

        self._font_dd.observe(_on_font, names="value")
        self._title_size_sl.observe(
            _make_size_cb(lambda ax: [ax.title], "Title size"), names="value")
        self._label_size_sl.observe(
            _make_size_cb(lambda ax: [ax.xaxis.label, ax.yaxis.label], "Label size"),
            names="value")
        self._tick_size_sl.observe(
            _make_size_cb(lambda ax: ax.get_xticklabels() + ax.get_yticklabels(),
                          "Tick size"), names="value")

        # --- Tick label colors ---
        cur_xtick_c = '#000000'
        cur_ytick_c = '#000000'
        if ax0:
            _xt = ax0.get_xticklabels()
            if _xt:
                try:
                    cur_xtick_c = to_hex(_xt[0].get_color())
                except Exception:
                    pass
            _yt = ax0.get_yticklabels()
            if _yt:
                try:
                    cur_ytick_c = to_hex(_yt[0].get_color())
                except Exception:
                    pass

        xtick_cp = widgets.ColorPicker(
            value=cur_xtick_c, concise=True,
            layout=widgets.Layout(width='28px', height='24px'))
        ytick_cp = widgets.ColorPicker(
            value=cur_ytick_c, concise=True,
            layout=widgets.Layout(width='28px', height='24px'))

        def _on_xtick_c(change):
            for ax in self._main_axes():
                ax.tick_params(axis='x', labelcolor=change["new"])
            self._canvas.redraw()

        def _on_ytick_c(change):
            for ax in self._main_axes():
                ax.tick_params(axis='y', labelcolor=change["new"])
            self._canvas.redraw()

        xtick_cp.observe(_on_xtick_c, names="value")
        ytick_cp.observe(_on_ytick_c, names="value")

        tick_color_row = widgets.HBox(
            [widgets.Label("Tick color:", layout=widgets.Layout(width='52px')),
             widgets.Label("X", layout=widgets.Layout(width='12px', margin='0 0 0 6px')),
             xtick_cp,
             widgets.Label("Y", layout=widgets.Layout(width='12px', margin='0 0 0 6px')),
             ytick_cp],
            layout=widgets.Layout(align_items='center', gap='2px'))

        return widgets.VBox([
            title_text, title_fmt,
            xlabel_text, xlabel_fmt,
            ylabel_text, ylabel_fmt,
            self._font_dd,
            _slider_num(self._title_size_sl, _GDW),
            _slider_num(self._title_pad_sl, _GDW),
            _slider_num(self._label_size_sl, _GDW),
            _slider_num(self._xlabel_pad_sl, _GDW),
            _slider_num(self._ylabel_pad_sl, _GDW),
            _slider_num(self._tick_size_sl, _GDW),
            tick_color_row,
        ])

    # ------------------------------------------------------------------
    # Axes & Grid — reads actual figure state
    # ------------------------------------------------------------------

    def _build_axes_grid(self) -> widgets.Widget:
        ax0 = self._main_axes()[0] if self._main_axes() else None

        # Read spine visibility from actual figure
        cur_spine_top = ax0.spines["top"].get_visible() if ax0 else True
        cur_spine_right = ax0.spines["right"].get_visible() if ax0 else True
        cur_spine_bottom = ax0.spines["bottom"].get_visible() if ax0 else True
        cur_spine_left = ax0.spines["left"].get_visible() if ax0 else True
        cur_spine_width = round(ax0.spines["bottom"].get_linewidth(), 1) if ax0 else 1.0

        # Read tick params
        cur_tick_dir = "out"
        cur_tick_len = 4.0
        cur_tick_width = 1.0
        if ax0:
            xtick_objs = ax0.xaxis.get_major_ticks()
            if xtick_objs:
                tick = xtick_objs[0]
                tick_line = tick.tick1line
                marker = tick_line.get_marker()
                if marker == 2:  # TICKUP = in
                    cur_tick_dir = "in"
                elif marker == 3:  # TICKDOWN = out
                    cur_tick_dir = "out"
                if tick.tick2line.get_visible() and tick.tick1line.get_visible():
                    m1 = tick.tick1line.get_marker()
                    m2 = tick.tick2line.get_marker()
                    if m1 != m2:
                        cur_tick_dir = "inout"
                cur_tick_len = round(tick_line.get_markersize(), 1)
                cur_tick_width = round(tick_line.get_markeredgewidth(), 1)

        # Read grid state
        cur_grid_on = False
        cur_grid_alpha = 0.2
        cur_grid_width = 0.8
        cur_grid_style = "-"
        if ax0:
            gridlines = ax0.xaxis.get_gridlines()
            if gridlines:
                cur_grid_on = gridlines[0].get_visible()
                cur_grid_alpha = round(gridlines[0].get_alpha() or 0.5, 2)
                cur_grid_width = round(gridlines[0].get_linewidth(), 1)
                cur_grid_style = gridlines[0].get_linestyle()
                _ls_map = {"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-."}
                cur_grid_style = _ls_map.get(cur_grid_style, cur_grid_style)

        self._spine_top_cb = widgets.Checkbox(
            value=cur_spine_top, description="Top", indent=False,
            layout=widgets.Layout(width="auto"))
        self._spine_right_cb = widgets.Checkbox(
            value=cur_spine_right, description="Right", indent=False,
            layout=widgets.Layout(width="auto"))
        self._spine_bottom_cb = widgets.Checkbox(
            value=cur_spine_bottom, description="Bottom", indent=False,
            layout=widgets.Layout(width="auto"))
        self._spine_left_cb = widgets.Checkbox(
            value=cur_spine_left, description="Left", indent=False,
            layout=widgets.Layout(width="auto"))
        self._spine_width_sl = widgets.FloatSlider(
            value=cur_spine_width, min=0.1, max=5.0, step=0.1,
            description="Width:", style=_STY)

        # --- Axis limits ---
        cur_xlim = ax0.get_xlim() if ax0 else (0.0, 1.0)
        cur_ylim = ax0.get_ylim() if ax0 else (0.0, 1.0)

        self._equal_cb = widgets.Checkbox(
            value=False, description="Equal axes", indent=False,
            layout=widgets.Layout(width="auto"))
        equal_cb = self._equal_cb

        self._xlim_min = widgets.FloatText(
            value=round(cur_xlim[0], 4), description="X min:",
            style={"description_width": "46px"},
            layout=widgets.Layout(width="140px"))
        self._xlim_max = widgets.FloatText(
            value=round(cur_xlim[1], 4), description="X max:",
            style={"description_width": "46px"},
            layout=widgets.Layout(width="140px"))
        self._ylim_min = widgets.FloatText(
            value=round(cur_ylim[0], 4), description="Y min:",
            style={"description_width": "46px"},
            layout=widgets.Layout(width="140px"))
        self._ylim_max = widgets.FloatText(
            value=round(cur_ylim[1], 4), description="Y max:",
            style={"description_width": "46px"},
            layout=widgets.Layout(width="140px"))
        xlim_min = self._xlim_min
        xlim_max = self._xlim_max
        ylim_min = self._ylim_min
        ylim_max = self._ylim_max

        _lim_updating = [False]

        def _apply_equal():
            """Compute union of X/Y ranges and apply to both axes + sync ticks."""
            from matplotlib.ticker import AutoLocator, MultipleLocator
            lo = min(xlim_min.value, ylim_min.value)
            hi = max(xlim_max.value, ylim_max.value)
            _lim_updating[0] = True
            try:
                xlim_min.value = round(lo, 4)
                xlim_max.value = round(hi, 4)
                ylim_min.value = round(lo, 4)
                ylim_max.value = round(hi, 4)
            finally:
                _lim_updating[0] = False
            for ax in self._main_axes():
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
            # Sync tick steps
            xs = self._x_step.value
            ys = self._y_step.value
            step = max(xs, ys)
            _step_updating[0] = True
            try:
                self._x_step.value = step
                self._y_step.value = step
            finally:
                _step_updating[0] = False
            for ax in self._main_axes():
                ax.xaxis.set_major_locator(
                    MultipleLocator(step) if step > 0 else AutoLocator())
                ax.yaxis.set_major_locator(
                    MultipleLocator(step) if step > 0 else AutoLocator())

        def _on_equal(change):
            if change["new"]:
                _apply_equal()
                self._canvas.force_redraw()

        equal_cb.observe(_on_equal, names="value")

        def _make_lim_cb(source_w, mirror_w):
            """Create a limit callback that mirrors to the paired axis when equal."""
            def _cb(change):
                if _lim_updating[0]:
                    return
                if equal_cb.value:
                    _lim_updating[0] = True
                    try:
                        mirror_w.value = change["new"]
                    finally:
                        _lim_updating[0] = False
                for ax in self._main_axes():
                    ax.set_xlim(xlim_min.value, xlim_max.value)
                    ax.set_ylim(ylim_min.value, ylim_max.value)
                self._canvas.force_redraw()
            return _cb

        xlim_min.observe(_make_lim_cb(xlim_min, ylim_min), names="value")
        xlim_max.observe(_make_lim_cb(xlim_max, ylim_max), names="value")
        ylim_min.observe(_make_lim_cb(ylim_min, xlim_min), names="value")
        ylim_max.observe(_make_lim_cb(ylim_max, xlim_max), names="value")

        # --- Axis scale ---
        cur_xscale = ax0.get_xscale() if ax0 else "linear"
        cur_yscale = ax0.get_yscale() if ax0 else "linear"
        _scale_opts = ["linear", "log", "symlog"]
        self._xscale_dd = widgets.Dropdown(
            options=_scale_opts, value=cur_xscale if cur_xscale in _scale_opts else "linear",
            description="X scale:", style=_STY, layout=_DD_SHORT)
        self._yscale_dd = widgets.Dropdown(
            options=_scale_opts, value=cur_yscale if cur_yscale in _scale_opts else "linear",
            description="Y scale:", style=_STY, layout=_DD_SHORT)

        def _scale_cb(change):
            for ax in self._main_axes():
                ax.set_xscale(self._xscale_dd.value)
                ax.set_yscale(self._yscale_dd.value)
            self._canvas.redraw()
        self._xscale_dd.observe(_scale_cb, names="value")
        self._yscale_dd.observe(_scale_cb, names="value")

        self._tick_dir_dd = widgets.Dropdown(
            options=["out", "in", "inout"], value=cur_tick_dir,
            description="Dir:", style=_STY, layout=_DD_SHORT)
        self._tick_len_sl = widgets.FloatSlider(
            value=cur_tick_len, min=0, max=15, step=0.5,
            description="Length:", style=_STY)
        self._tick_width_sl = widgets.FloatSlider(
            value=cur_tick_width, min=0.1, max=4, step=0.1,
            description="Width:", style=_STY)

        self._grid_toggle = widgets.Checkbox(
            value=cur_grid_on, description="Show grid", indent=False)
        self._grid_alpha_sl = widgets.FloatSlider(
            value=cur_grid_alpha, min=0.05, max=1, step=0.05,
            description="Alpha:", style=_STY)
        self._grid_width_sl = widgets.FloatSlider(
            value=cur_grid_width, min=0.1, max=4, step=0.1,
            description="Width:", style=_STY)

        valid_styles = ["-", "--", ":", "-."]
        if cur_grid_style not in valid_styles:
            cur_grid_style = "-"
        self._grid_style_dd = widgets.Dropdown(
            options=[("-", "-"), ("--", "--"), (":", ":"), ("-.", "-.")],
            value=cur_grid_style, description="Style:",
            style=_STY, layout=_DD_SHORT)

        def _spine_cb(spine_name):
            def _cb(change):
                cmds = []
                for ax in self._main_axes():
                    sp = ax.spines[spine_name]
                    cmds.append(Command(sp, "visible", sp.get_visible(), change["new"]))
                self._stack.execute(BatchCommand(cmds, f"Toggle {spine_name} spine"))
                self._canvas.redraw()
            return _cb

        self._spine_top_cb.observe(_spine_cb("top"), names="value")
        self._spine_right_cb.observe(_spine_cb("right"), names="value")
        self._spine_bottom_cb.observe(_spine_cb("bottom"), names="value")
        self._spine_left_cb.observe(_spine_cb("left"), names="value")

        def _spine_width_cb(change):
            cmds = []
            for ax in self._main_axes():
                for name in ("top", "right", "bottom", "left"):
                    sp = ax.spines[name]
                    cmds.append(Command(sp, "linewidth", sp.get_linewidth(), change["new"]))
            self._stack.execute(BatchCommand(cmds, "Spine width"))
            self._canvas.redraw()
        self._spine_width_sl.observe(_spine_width_cb, names="value")

        def _tick_cb(change):
            for ax in self._main_axes():
                ax.tick_params(direction=self._tick_dir_dd.value,
                               length=self._tick_len_sl.value,
                               width=self._tick_width_sl.value)
            self._canvas.redraw()
        self._tick_dir_dd.observe(_tick_cb, names="value")
        self._tick_len_sl.observe(_tick_cb, names="value")
        self._tick_width_sl.observe(_tick_cb, names="value")

        # --- Tick spacing (step) ---
        cur_x_step = 0.0
        cur_y_step = 0.0
        if ax0:
            xt = ax0.get_xticks()
            if len(xt) >= 2:
                cur_x_step = round(abs(xt[1] - xt[0]), 4)
            yt = ax0.get_yticks()
            if len(yt) >= 2:
                cur_y_step = round(abs(yt[1] - yt[0]), 4)

        self._x_step = widgets.FloatText(
            value=cur_x_step, description="X step:",
            style={"description_width": "46px"},
            layout=widgets.Layout(width="120px"))
        self._y_step = widgets.FloatText(
            value=cur_y_step, description="Y step:",
            style={"description_width": "46px"},
            layout=widgets.Layout(width="120px"))
        x_step = self._x_step
        y_step = self._y_step

        _step_updating = [False]

        def _tick_step_cb(change):
            if _step_updating[0]:
                return
            from matplotlib.ticker import AutoLocator, MultipleLocator
            # Sync steps when equal axes is on
            if equal_cb.value and change is not None:
                _step_updating[0] = True
                try:
                    v = change["new"]
                    x_step.value = v
                    y_step.value = v
                finally:
                    _step_updating[0] = False
            for ax in self._main_axes():
                xs = x_step.value
                ys = y_step.value
                if xs > 0:
                    ax.xaxis.set_major_locator(MultipleLocator(xs))
                else:
                    ax.xaxis.set_major_locator(AutoLocator())
                if ys > 0:
                    ax.yaxis.set_major_locator(MultipleLocator(ys))
                else:
                    ax.yaxis.set_major_locator(AutoLocator())
            self._canvas.force_redraw()

        x_step.observe(_tick_step_cb, names="value")
        y_step.observe(_tick_step_cb, names="value")

        def _grid_cb(change):
            for ax in self._main_axes():
                if self._grid_toggle.value:
                    ax.grid(True, alpha=self._grid_alpha_sl.value,
                            linewidth=self._grid_width_sl.value,
                            linestyle=self._grid_style_dd.value)
                else:
                    ax.grid(False)
            self._canvas.redraw()
        self._grid_toggle.observe(_grid_cb, names="value")
        self._grid_alpha_sl.observe(_grid_cb, names="value")
        self._grid_width_sl.observe(_grid_cb, names="value")
        self._grid_style_dd.observe(_grid_cb, names="value")

        return widgets.VBox([
            widgets.HTML("<b>Limits</b>"),
            equal_cb,
            widgets.HBox([xlim_min, xlim_max]),
            widgets.HBox([ylim_min, ylim_max]),
            widgets.HTML("<b>Spines</b>"),
            widgets.HBox([self._spine_top_cb, self._spine_right_cb,
                          self._spine_bottom_cb, self._spine_left_cb]),
            _slider_num(self._spine_width_sl, _GDW),
            widgets.HTML("<b>Scale</b>"),
            self._xscale_dd,
            self._yscale_dd,
            widgets.HTML("<b>Ticks</b>"),
            self._tick_dir_dd,
            _slider_num(self._tick_len_sl, _GDW),
            _slider_num(self._tick_width_sl, _GDW),
            widgets.HBox([x_step, y_step]),
            widgets.HTML("<small style='color:#888'>step 0 = auto spacing</small>"),
            widgets.HTML("<b>Grid</b>"),
            self._grid_toggle,
            _slider_num(self._grid_alpha_sl, _GDW),
            _slider_num(self._grid_width_sl, _GDW),
            self._grid_style_dd,
        ])

    # ------------------------------------------------------------------
    # Legend — reads actual figure state
    # ------------------------------------------------------------------

    def _build_legend(self) -> widgets.Widget:
        ax0 = self._main_axes()[0] if self._main_axes() else None
        leg = ax0.get_legend() if ax0 else None
        has_legend = leg is not None

        # Read current legend state from the actual figure
        _loc_names = {
            0: "best", 1: "upper right", 2: "upper left",
            3: "lower left", 4: "lower right", 5: "right",
            6: "center left", 7: "center right",
            8: "lower center", 9: "upper center", 10: "center",
        }
        cur_loc = "best"
        cur_frame = False
        cur_fontsize = 10.0
        cur_ncol = 1
        cur_markerfirst = True
        cur_handletextpad = 0.8
        cur_handleheight = 0.7

        if leg is not None:
            try:
                loc_code = leg._loc
                cur_loc = _loc_names.get(loc_code, "best")
            except Exception:
                pass
            try:
                # Default to frame off — apply immediately
                leg.get_frame().set_visible(False)
            except Exception:
                pass
            try:
                cur_fontsize = round(leg._fontsize, 1)
            except Exception:
                pass
            try:
                cur_ncol = leg._ncols
            except Exception:
                pass
            try:
                cur_markerfirst = leg._markerfirst
            except Exception:
                pass
            try:
                cur_handletextpad = round(leg.handletextpad, 2)
            except Exception:
                pass
            try:
                cur_handleheight = round(leg.handleheight, 2)
            except Exception:
                pass

        # Position info: each preset has a matplotlib loc and a base (x, y)
        _pos_info = {
            "upper right": {"loc": "upper right", "xy": (1.0, 1.0)},
            "upper left": {"loc": "upper left", "xy": (0.0, 1.0)},
            "lower left": {"loc": "lower left", "xy": (0.0, 0.0)},
            "lower right": {"loc": "lower right", "xy": (1.0, 0.0)},
            "right": {"loc": "center left", "xy": (1.0, 0.5)},
            "center left": {"loc": "center left", "xy": (0.0, 0.5)},
            "center right": {"loc": "center right", "xy": (1.0, 0.5)},
            "lower center": {"loc": "lower center", "xy": (0.5, 0.0)},
            "upper center": {"loc": "upper center", "xy": (0.5, 1.0)},
            "center": {"loc": "center", "xy": (0.5, 0.5)},
            "outside_right": {"loc": "center left", "xy": (1.02, 0.5)},
            "outside_upper_right": {"loc": "upper left", "xy": (1.02, 1.0)},
            "outside_lower_right": {"loc": "lower left", "xy": (1.02, 0.0)},
            "outside_upper_center": {"loc": "lower center", "xy": (0.5, 1.08)},
            "outside_lower_center": {"loc": "upper center", "xy": (0.5, -0.12)},
        }

        positions = [
            ("Upper right", "upper right"),
            ("Upper left", "upper left"),
            ("Lower left", "lower left"),
            ("Lower right", "lower right"),
            ("Right", "right"),
            ("Center left", "center left"),
            ("Center right", "center right"),
            ("Lower center", "lower center"),
            ("Upper center", "upper center"),
            ("Center", "center"),
            ("Outside right", "outside_right"),
            ("Outside upper right", "outside_upper_right"),
            ("Outside lower right", "outside_lower_right"),
            ("Outside top center", "outside_upper_center"),
            ("Outside bottom center", "outside_lower_center"),
        ]
        valid_values = [v for _, v in positions]
        # Resolve "best" (dynamic) to a concrete position and apply immediately
        if cur_loc == "best" or cur_loc not in valid_values:
            cur_loc = "upper right"
            if leg is not None and ax0 is not None:
                handles, labels = ax0.get_legend_handles_labels()
                if handles:
                    info = _pos_info[cur_loc]
                    ax0.legend(handles, labels,
                               loc=info["loc"],
                               bbox_to_anchor=info["xy"],
                               frameon=cur_frame,
                               fontsize=cur_fontsize,
                               ncol=cur_ncol,
                               markerfirst=cur_markerfirst,
                               handletextpad=cur_handletextpad,
                               handleheight=cur_handleheight)
                    leg = ax0.get_legend()

        # Compute initial X/Y from current legend position
        init_xy = _pos_info.get(cur_loc, {}).get("xy", (1.0, 1.0))
        cur_x, cur_y = init_xy
        if leg is not None and hasattr(leg, "_bbox_to_anchor") and \
                leg._bbox_to_anchor is not None and ax0 is not None:
            try:
                inv = ax0.transAxes.inverted()
                ax_x, ax_y = inv.transform(
                    (leg._bbox_to_anchor.x0, leg._bbox_to_anchor.y0))
                if (ax_x > 1.01 or ax_x < -0.01
                        or ax_y > 1.01 or ax_y < -0.01):
                    cur_x, cur_y = round(ax_x, 2), round(ax_y, 2)
            except Exception:
                pass

        # --- Show legend toggle ---
        self._legend_toggle = widgets.Checkbox(
            value=has_legend, description="Show legend", indent=False)

        self._legend_pos_dd = widgets.Dropdown(options=positions, value=cur_loc,
                                  description="Pos:", style=_STY,
                                  layout=_DD_MED)
        pos_dd = self._legend_pos_dd
        self._legend_x_sl = widgets.FloatSlider(
            value=round(cur_x, 2), min=-0.5, max=1.5,
            step=0.02, description="X pos:", style=_STY)
        self._legend_y_sl = widgets.FloatSlider(
            value=round(cur_y, 2), min=-0.5, max=1.5,
            step=0.02, description="Y pos:", style=_STY)
        x_sl = self._legend_x_sl
        y_sl = self._legend_y_sl
        self._frame_toggle = widgets.Checkbox(value=cur_frame, description="Frame",
                                              indent=False)
        self._legend_fontsize_sl = widgets.FloatSlider(
            value=cur_fontsize, min=4, max=24,
            step=0.5, description="Font sz:", style=_STY)
        self._legend_ncol = widgets.IntSlider(value=cur_ncol, min=1, max=6,
                                  step=1, description="Cols:", style=_STY)
        ncol = self._legend_ncol
        self._markerfirst_cb = widgets.Checkbox(
            value=cur_markerfirst, description="Marker first", indent=False)
        self._handletextpad_sl = widgets.FloatSlider(
            value=cur_handletextpad, min=0, max=3, step=0.1,
            description="Spacing:", style=_STY)
        self._handleheight_sl = widgets.FloatSlider(
            value=cur_handleheight, min=0.1, max=3, step=0.1,
            description="Height:", style=_STY)

        # Controls box — hidden when legend is off
        controls_box = widgets.VBox([
            pos_dd,
            _slider_num(x_sl, _GDW),
            _slider_num(y_sl, _GDW),
            self._frame_toggle,
            self._markerfirst_cb,
            _slider_num(self._legend_fontsize_sl, _GDW),
            _slider_num(self._handletextpad_sl, _GDW),
            _slider_num(self._handleheight_sl, _GDW),
            _slider_num(ncol, _GDW),
        ], layout=widgets.Layout(
            display="" if has_legend else "none"))

        _updating = [False]

        # --- Per-entry legend text colors ---
        _all_handles, _all_labels = (
            ax0.get_legend_handles_labels() if ax0 else ([], []))
        for _ax in self._main_axes():
            _h, _l = _ax.get_legend_handles_labels()
            _ax._matplotly_leg_text_colors = ['#000000'] * len(_l)
            _aleg = _ax.get_legend()
            if _aleg:
                for _ti, _tt in enumerate(_aleg.get_texts()):
                    try:
                        _ax._matplotly_leg_text_colors[_ti] = to_hex(
                            _tt.get_color())
                    except Exception:
                        pass

        _leg_color_rows = []
        self._leg_label_widgets = []  # store for dynamic updates
        for _idx, _lbl_text in enumerate(_all_labels):
            _cur_lc = '#000000'
            if leg and _idx < len(leg.get_texts()):
                try:
                    _cur_lc = to_hex(leg.get_texts()[_idx].get_color())
                except Exception:
                    pass

            def _make_lc_cb(i):
                def _on_lc(hex_val):
                    for _ax in self._main_axes():
                        if hasattr(_ax, '_matplotly_leg_text_colors'):
                            while len(_ax._matplotly_leg_text_colors) <= i:
                                _ax._matplotly_leg_text_colors.append(
                                    '#000000')
                            _ax._matplotly_leg_text_colors[i] = hex_val
                        _aleg = _ax.get_legend()
                        if _aleg and i < len(_aleg.get_texts()):
                            _aleg.get_texts()[i].set_color(hex_val)
                    self._canvas.force_redraw()
                return _on_lc

            _lc_btn, _lc_swatch = self._cmap_color_btn(
                _cur_lc, _make_lc_cb(_idx))
            _lbl_w = widgets.Label(_lbl_text)
            self._leg_label_widgets.append(_lbl_w)
            _leg_color_rows.append(widgets.VBox([
                widgets.HBox(
                    [_lc_btn, _lbl_w],
                    layout=widgets.Layout(align_items='center', gap='4px')),
                _lc_swatch,
            ]))

        leg_colors_box = widgets.VBox(
            ([widgets.HTML("<small><b>Text colors</b></small>")]
             + _leg_color_rows) if _leg_color_rows else [],
            layout=widgets.Layout(
                display="" if (has_legend and _leg_color_rows) else "none"))

        def _on_pos_change(change):
            """When position preset changes, update X/Y sliders to base."""
            if _updating[0]:
                return
            _updating[0] = True
            try:
                info = _pos_info.get(change["new"], {})
                bx, by = info.get("xy", (0.5, 0.5))
                x_sl.value = round(bx, 2)
                y_sl.value = round(by, 2)
            finally:
                _updating[0] = False
            _apply_legend()

        def _create_legend_on_axes():
            """Create or update legend on all axes that have labeled handles."""
            info = _pos_info.get(pos_dd.value, {})
            loc = info.get("loc", "upper right")
            for ax in self._main_axes():
                handles, labels = ax.get_legend_handles_labels()
                if not handles:
                    continue
                ax.legend(handles, labels,
                          loc=loc,
                          bbox_to_anchor=(x_sl.value, y_sl.value),
                          frameon=self._frame_toggle.value,
                          fontsize=self._legend_fontsize_sl.value,
                          ncol=ncol.value,
                          markerfirst=self._markerfirst_cb.value,
                          handletextpad=self._handletextpad_sl.value,
                          handleheight=self._handleheight_sl.value)
                # Apply stored text colors
                if hasattr(ax, '_matplotly_leg_text_colors'):
                    _aleg = ax.get_legend()
                    if _aleg:
                        for _ci, _cc in enumerate(
                                ax._matplotly_leg_text_colors):
                            if _ci < len(_aleg.get_texts()):
                                _aleg.get_texts()[_ci].set_color(_cc)

        def _remove_legend_from_axes():
            """Remove legend from all axes."""
            for ax in self._main_axes():
                leg = ax.get_legend()
                if leg is not None:
                    leg.remove()

        def _on_legend_toggle(change):
            if _updating[0]:
                return
            if change["new"]:
                controls_box.layout.display = ""
                leg_colors_box.layout.display = ""
                _create_legend_on_axes()
            else:
                controls_box.layout.display = "none"
                leg_colors_box.layout.display = "none"
                _remove_legend_from_axes()
            self._canvas.force_redraw()

        self._legend_toggle.observe(_on_legend_toggle, names="value")

        def _apply_legend(*_):
            if _updating[0]:
                return
            if not self._legend_toggle.value:
                return
            _create_legend_on_axes()
            self._canvas.force_redraw()

        pos_dd.observe(_on_pos_change, names="value")
        for w in [x_sl, y_sl, self._frame_toggle, self._markerfirst_cb,
                  self._legend_fontsize_sl, self._handletextpad_sl,
                  self._handleheight_sl, ncol]:
            w.observe(_apply_legend, names="value")

        return widgets.VBox([self._legend_toggle, controls_box, leg_colors_box])

    def _refresh_legend_labels(self):
        """Update text color row labels to match current artist labels."""
        ax0 = self._main_axes()[0] if self._main_axes() else None
        if ax0 is None:
            return
        _, labels = ax0.get_legend_handles_labels()
        for i, lbl_w in enumerate(self._leg_label_widgets):
            if i < len(labels):
                lbl_w.value = labels[i]

    # ------------------------------------------------------------------
    # Layout — reads from figure (unchanged)
    # ------------------------------------------------------------------

    def _build_layout(self) -> widgets.Widget:
        w, h = self._fig.get_size_inches()
        _base = [round(w, 2), round(h, 2)]  # mutable base for scale

        scale_sl = widgets.IntSlider(
            value=100, min=25, max=200, step=5,
            description="Scale:", style=_STY,
            continuous_update=False)

        self._width_sl = widgets.FloatSlider(
            value=_base[0], min=0.2, max=20, step=0.1,
            description="W (in):", style=_STY)
        self._height_sl = widgets.FloatSlider(
            value=_base[1], min=0.2, max=20, step=0.1,
            description="H (in):", style=_STY)
        width_sl = self._width_sl
        height_sl = self._height_sl

        _updating = [False]

        def _scale_cb(change):
            if _updating[0]:
                return
            _updating[0] = True
            try:
                pct = change["new"] / 100.0
                new_w = round(_base[0] * pct, 1)
                new_h = round(_base[1] * pct, 1)
                width_sl.value = max(width_sl.min, min(width_sl.max, new_w))
                height_sl.value = max(height_sl.min, min(height_sl.max, new_h))
                self._fig.set_size_inches(new_w, new_h)
                self._canvas.redraw()
            finally:
                _updating[0] = False
        scale_sl.observe(_scale_cb, names="value")

        def _layout_cb(change):
            if _updating[0]:
                return
            _updating[0] = True
            try:
                _base[0] = width_sl.value
                _base[1] = height_sl.value
                scale_sl.value = 100
                self._fig.set_size_inches(width_sl.value, height_sl.value)
                self._canvas.redraw()
            finally:
                _updating[0] = False
        width_sl.observe(_layout_cb, names="value")
        height_sl.observe(_layout_cb, names="value")

        bg_presets = [
            ("White", "#ffffff"),
            ("Light gray", "#f0f0f0"),
            ("Warm white", "#faf8f5"),
            ("Dark", "#2d2d2d"),
            ("Black", "#000000"),
            ("Transparent", "none"),
            ("Custom...", "__custom__"),
        ]
        bg_dd = widgets.Dropdown(options=bg_presets, value="#ffffff",
                                 description="BG:", style=_STY,
                                 layout=_DD_MED)
        bg_custom = widgets.ColorPicker(value="#ffffff", description="Custom:",
                                        style=_STY,
                                        layout=widgets.Layout(display="none"))

        def _bg_dd_cb(change):
            if change["new"] == "__custom__":
                bg_custom.layout.display = ""
                return
            bg_custom.layout.display = "none"
            self._fig.set_facecolor(change["new"])
            for ax in self._main_axes():
                ax.set_facecolor(change["new"])
            self._canvas.redraw()

        def _bg_custom_cb(change):
            self._fig.set_facecolor(change["new"])
            for ax in self._main_axes():
                ax.set_facecolor(change["new"])
            self._canvas.redraw()

        bg_dd.observe(_bg_dd_cb, names="value")
        bg_custom.observe(_bg_custom_cb, names="value")

        layout_items = [
            _slider_num(scale_sl, _GDW),
            _slider_num(width_sl, _GDW),
            _slider_num(height_sl, _GDW),
            bg_dd,
            bg_custom,
        ]

        # Subplot spacing controls (multi-subplot only)
        if self._multi_subplot:
            try:
                sp = self._fig.subplotpars
                init_hspace = round(sp.hspace, 2) if sp.hspace else 0.2
                init_wspace = round(sp.wspace, 2) if sp.wspace else 0.2
            except Exception:
                init_hspace = 0.2
                init_wspace = 0.2

            hspace_sl = widgets.FloatSlider(
                value=init_hspace, min=0, max=1.0, step=0.02,
                description="H space:", style=_STY)
            wspace_sl = widgets.FloatSlider(
                value=init_wspace, min=0, max=1.0, step=0.02,
                description="W space:", style=_STY)

            def _spacing_cb(change):
                spacing = {'hspace': hspace_sl.value, 'wspace': wspace_sl.value}
                self._fig._matplotly_spacing = spacing
                self._fig.subplots_adjust(**spacing)
                self._canvas.force_redraw()

            hspace_sl.observe(_spacing_cb, names="value")
            wspace_sl.observe(_spacing_cb, names="value")

            layout_items.append(widgets.HTML("<b>Subplot Spacing</b>"))
            layout_items.append(_slider_num(hspace_sl, _GDW))
            layout_items.append(_slider_num(wspace_sl, _GDW))

        return widgets.VBox(layout_items)
