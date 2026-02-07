"""Global figure controls — fonts, axes/spines/ticks/grid, legend, layout."""
from __future__ import annotations

import ipywidgets as widgets
from matplotlib.figure import Figure

from .._commands import BatchCommand, Command, CommandStack
from .._renderer import CanvasManager


_DW = "48px"    # uniform description_width for all controls
_NW = "50px"    # uniform number-edit width
_DD_SHORT = widgets.Layout(width="150px")   # short dropdowns (direction, style)
_DD_MED = widgets.Layout(width="210px")     # medium dropdowns (font, position, bg)
_STY = {"description_width": _DW}


def _slider_num(slider: widgets.FloatSlider | widgets.IntSlider) -> widgets.HBox:
    """Slider (no readout) + linked number edit box (2 decimal places)."""
    slider.readout = False
    slider.style = {"description_width": _DW}
    if isinstance(slider, widgets.IntSlider):
        num = widgets.IntText(value=slider.value,
                              layout=widgets.Layout(width=_NW))
    else:
        num = widgets.BoundedFloatText(
            value=round(slider.value, 2), step=slider.step,
            min=slider.min, max=slider.max,
            layout=widgets.Layout(width=_NW))
    widgets.link((slider, "value"), (num, "value"))
    return widgets.HBox([slider, num])


class GlobalPanel:
    """Controls that apply to the whole figure rather than individual artists."""

    def __init__(self, fig: Figure, stack: CommandStack, canvas: CanvasManager,
                 colormap_widget: widgets.Widget | None = None,
                 cmap_panel=None):
        self._fig = fig
        self._stack = stack
        self._canvas = canvas
        self._colormap_widget = colormap_widget
        self._cmap_panel = cmap_panel

    def build(self) -> widgets.Widget:
        self._legend_widget = self._build_legend()
        sections = [
            ("Figure Size", self._build_layout()),
            ("Axes & Grid", self._build_axes_grid()),
            ("Fonts", self._build_typography()),
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
        return widgets.VBox([defaults_btn, accordion])

    def _apply_preferred_defaults(self, _btn):
        """Apply clean publication defaults in one shot: Arial 10pt uniform,
        no top/right spines, legend frame off, tab10 colormap."""
        # Temporarily disable redraws — each widget change triggers its
        # observer which calls canvas.redraw(). The throttle would cause
        # only the first change to render.  We suppress all intermediate
        # redraws and do one force_redraw at the end.
        orig_redraw = self._canvas.redraw
        self._canvas.redraw = lambda: None  # suppress
        try:
            self._font_dd.value = "Arial"
            self._title_size_sl.value = 10.0
            self._label_size_sl.value = 10.0
            self._tick_size_sl.value = 10.0
            self._spine_top_cb.value = False
            self._spine_right_cb.value = False
            self._spine_bottom_cb.value = True
            self._spine_left_cb.value = True
            self._legend_toggle.value = True
            self._frame_toggle.value = False
            self._legend_fontsize_sl.value = 10.0
            if self._cmap_panel is not None:
                self._cmap_panel.apply("tab10")
        finally:
            self._canvas.redraw = orig_redraw
        # One render with all changes applied
        self._canvas.force_redraw()

    # ------------------------------------------------------------------
    # Typography — reads actual figure state
    # ------------------------------------------------------------------

    def _build_typography(self) -> widgets.Widget:
        ax0 = self._fig.get_axes()[0] if self._fig.get_axes() else None

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
            if not self._fig.get_axes():
                return
            ax = self._fig.get_axes()[0]
            self._stack.execute(
                Command(ax.title, "text", ax.get_title(), change["new"],
                        description="Change title"))
            self._canvas.redraw()

        def _on_xlabel(change):
            if not self._fig.get_axes():
                return
            ax = self._fig.get_axes()[0]
            self._stack.execute(
                Command(ax.xaxis.label, "text", ax.get_xlabel(), change["new"],
                        description="Change X label"))
            self._canvas.redraw()

        def _on_ylabel(change):
            if not self._fig.get_axes():
                return
            ax = self._fig.get_axes()[0]
            self._stack.execute(
                Command(ax.yaxis.label, "text", ax.get_ylabel(), change["new"],
                        description="Change Y label"))
            self._canvas.redraw()

        title_text.observe(_on_title, names="value")
        xlabel_text.observe(_on_xlabel, names="value")
        ylabel_text.observe(_on_ylabel, names="value")

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
        self._label_size_sl = widgets.FloatSlider(
            value=cur_label_sz, min=4, max=36, step=0.5,
            description="Label sz:", style=_STY)
        self._tick_size_sl = widgets.FloatSlider(
            value=cur_tick_sz, min=4, max=36, step=0.5,
            description="Tick sz:", style=_STY)

        def _on_font(change):
            cmds = []
            for ax in self._fig.get_axes():
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
                for ax in self._fig.get_axes():
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

        return widgets.VBox([
            title_text,
            xlabel_text,
            ylabel_text,
            self._font_dd,
            _slider_num(self._title_size_sl),
            _slider_num(self._label_size_sl),
            _slider_num(self._tick_size_sl),
        ])

    # ------------------------------------------------------------------
    # Axes & Grid — reads actual figure state
    # ------------------------------------------------------------------

    def _build_axes_grid(self) -> widgets.Widget:
        ax0 = self._fig.get_axes()[0] if self._fig.get_axes() else None

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
        cur_grid_alpha = 0.5
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
                for ax in self._fig.get_axes():
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
            for ax in self._fig.get_axes():
                for name in ("top", "right", "bottom", "left"):
                    sp = ax.spines[name]
                    cmds.append(Command(sp, "linewidth", sp.get_linewidth(), change["new"]))
            self._stack.execute(BatchCommand(cmds, "Spine width"))
            self._canvas.redraw()
        self._spine_width_sl.observe(_spine_width_cb, names="value")

        def _tick_cb(change):
            for ax in self._fig.get_axes():
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

        x_step = widgets.FloatText(
            value=cur_x_step, description="X step:",
            style={"description_width": "40px"},
            layout=widgets.Layout(width="110px"))
        y_step = widgets.FloatText(
            value=cur_y_step, description="Y step:",
            style={"description_width": "40px"},
            layout=widgets.Layout(width="110px"))

        def _tick_step_cb(change):
            from matplotlib.ticker import AutoLocator, MultipleLocator
            for ax in self._fig.get_axes():
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
            self._canvas.redraw()

        x_step.observe(_tick_step_cb, names="value")
        y_step.observe(_tick_step_cb, names="value")

        def _grid_cb(change):
            for ax in self._fig.get_axes():
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
            widgets.HTML("<b>Spines</b>"),
            widgets.HBox([self._spine_top_cb, self._spine_right_cb,
                          self._spine_bottom_cb, self._spine_left_cb]),
            _slider_num(self._spine_width_sl),
            widgets.HTML("<b>Ticks</b>"),
            self._tick_dir_dd,
            _slider_num(self._tick_len_sl),
            _slider_num(self._tick_width_sl),
            widgets.HBox([x_step, y_step]),
            widgets.HTML("<small style='color:#888'>step 0 = auto spacing</small>"),
            widgets.HTML("<b>Grid</b>"),
            self._grid_toggle,
            _slider_num(self._grid_alpha_sl),
            _slider_num(self._grid_width_sl),
            self._grid_style_dd,
        ])

    # ------------------------------------------------------------------
    # Legend — reads actual figure state
    # ------------------------------------------------------------------

    def _build_legend(self) -> widgets.Widget:
        ax0 = self._fig.get_axes()[0] if self._fig.get_axes() else None
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
        cur_frame = True
        cur_fontsize = 10.0
        cur_ncol = 1

        if leg is not None:
            try:
                loc_code = leg._loc
                cur_loc = _loc_names.get(loc_code, "best")
            except Exception:
                pass
            try:
                cur_frame = leg.get_frame().get_visible()
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

        # Position info: each preset has a matplotlib loc and a base (x, y)
        _pos_info = {
            "best": {"loc": "upper right", "xy": (1.0, 1.0)},
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
            ("Best", "best"),
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
        if cur_loc not in valid_values:
            cur_loc = "best"

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

        pos_dd = widgets.Dropdown(options=positions, value=cur_loc,
                                  description="Pos:", style=_STY,
                                  layout=_DD_MED)
        x_sl = widgets.FloatSlider(value=round(cur_x, 2), min=-0.5, max=1.5,
                                    step=0.02, description="X pos:", style=_STY)
        y_sl = widgets.FloatSlider(value=round(cur_y, 2), min=-0.5, max=1.5,
                                    step=0.02, description="Y pos:", style=_STY)
        self._frame_toggle = widgets.Checkbox(value=cur_frame, description="Frame",
                                              indent=False)
        self._legend_fontsize_sl = widgets.FloatSlider(
            value=cur_fontsize, min=4, max=24,
            step=0.5, description="Font sz:", style=_STY)
        ncol = widgets.IntSlider(value=cur_ncol, min=1, max=6, step=1,
                                  description="Cols:", style=_STY)

        # Controls box — hidden when legend is off
        controls_box = widgets.VBox([
            pos_dd,
            _slider_num(x_sl),
            _slider_num(y_sl),
            self._frame_toggle,
            _slider_num(self._legend_fontsize_sl),
            _slider_num(ncol),
        ], layout=widgets.Layout(
            display="" if has_legend else "none"))

        _updating = [False]

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
            for ax in self._fig.get_axes():
                handles, labels = ax.get_legend_handles_labels()
                if not handles:
                    continue
                ax.legend(handles, labels,
                          loc=loc,
                          bbox_to_anchor=(x_sl.value, y_sl.value),
                          frameon=self._frame_toggle.value,
                          fontsize=self._legend_fontsize_sl.value,
                          ncol=ncol.value)

        def _remove_legend_from_axes():
            """Remove legend from all axes."""
            for ax in self._fig.get_axes():
                leg = ax.get_legend()
                if leg is not None:
                    leg.remove()

        def _on_legend_toggle(change):
            if _updating[0]:
                return
            if change["new"]:
                controls_box.layout.display = ""
                _create_legend_on_axes()
            else:
                controls_box.layout.display = "none"
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
        for w in [x_sl, y_sl, self._frame_toggle, self._legend_fontsize_sl,
                  ncol]:
            w.observe(_apply_legend, names="value")

        return widgets.VBox([self._legend_toggle, controls_box])

    # ------------------------------------------------------------------
    # Layout — reads from figure (unchanged)
    # ------------------------------------------------------------------

    def _build_layout(self) -> widgets.Widget:
        w, h = self._fig.get_size_inches()
        orig_w, orig_h = round(w, 2), round(h, 2)

        scale_sl = widgets.IntSlider(
            value=100, min=25, max=200, step=5,
            description="Scale:", style=_STY)

        width_sl = widgets.FloatSlider(value=orig_w, min=2, max=20, step=0.25,
                                        description="W (in):", style=_STY)
        height_sl = widgets.FloatSlider(value=orig_h, min=1, max=16, step=0.25,
                                         description="H (in):", style=_STY)
        dpi_sl = widgets.IntSlider(value=int(self._fig.dpi), min=50, max=300, step=10,
                                    description="DPI:", style=_STY)

        _updating = [False]

        def _scale_cb(change):
            if _updating[0]:
                return
            _updating[0] = True
            try:
                pct = change["new"] / 100.0
                new_w = round(orig_w * pct, 2)
                new_h = round(orig_h * pct, 2)
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
            self._fig.set_size_inches(width_sl.value, height_sl.value)
            self._fig.set_dpi(dpi_sl.value)
            self._canvas.redraw()
        width_sl.observe(_layout_cb, names="value")
        height_sl.observe(_layout_cb, names="value")
        dpi_sl.observe(_layout_cb, names="value")

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
            for ax in self._fig.get_axes():
                ax.set_facecolor(change["new"])
            self._canvas.redraw()

        def _bg_custom_cb(change):
            self._fig.set_facecolor(change["new"])
            for ax in self._fig.get_axes():
                ax.set_facecolor(change["new"])
            self._canvas.redraw()

        bg_dd.observe(_bg_dd_cb, names="value")
        bg_custom.observe(_bg_custom_cb, names="value")

        return widgets.VBox([
            _slider_num(scale_sl),
            _slider_num(width_sl),
            _slider_num(height_sl),
            _slider_num(dpi_sl),
            bg_dd,
            bg_custom,
        ])
