"""Per-subplot controls — labels, ticks/spines, axes/grid, legend — for multi-subplot figures."""
from __future__ import annotations

import ipywidgets as widgets
from matplotlib.colors import to_hex
from matplotlib.figure import Figure

from .._commands import BatchCommand, Command, CommandStack
from .._renderer import CanvasManager
from ._color_utils import _DW, _NW, _slider_num, _get_palette_colors, cmap_color_btn


_DD_SHORT = widgets.Layout(width="150px")
_DD_MED = widgets.Layout(width="210px")
_GDW = "70px"
_STY = {"description_width": _GDW}


def _make_toggle_section(title, children, initially_open=True):
    """Collapsible section with chevron button."""
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


class PerSubplotPanel:
    """Controls for a single Axes: labels, ticks & spines, axes/grid, legend."""

    def __init__(self, ax, fig: Figure, stack: CommandStack,
                 canvas: CanvasManager, cmap_panel=None):
        self._ax = ax
        self._fig = fig
        self._stack = stack
        self._canvas = canvas
        self._cmap_panel = cmap_panel

    def build(self) -> widgets.Widget:
        """Returns VBox with Labels, Ticks & Spines, Axes, Legend toggle sections."""
        labels_sec = self._build_labels()
        ticks_sec = self._build_ticks_spines()
        axes_sec = self._build_axes()
        legend_sec = self._build_legend()

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

        return widgets.VBox([
            _icon_css,
            _make_toggle_section("Labels", [labels_sec], initially_open=False),
            _make_toggle_section("Ticks & Spines", [ticks_sec], initially_open=False),
            _make_toggle_section("Axes", [axes_sec], initially_open=False),
            _make_toggle_section("Legend", [legend_sec], initially_open=False),
        ])

    def apply_defaults(self):
        """Apply preferred defaults for this subplot."""
        self._spine_top_cb.value = False
        self._spine_right_cb.value = False
        self._spine_bottom_cb.value = True
        self._spine_left_cb.value = True
        self._legend_toggle.value = True
        self._frame_toggle.value = False

    # ------------------------------------------------------------------
    # Labels section
    # ------------------------------------------------------------------

    def _build_labels(self) -> widgets.Widget:
        ax = self._ax

        cur_title = ax.get_title()
        cur_xlabel = ax.get_xlabel()
        cur_ylabel = ax.get_ylabel()

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
            self._stack.execute(
                Command(ax.title, "text", ax.get_title(), change["new"],
                        description="Change title"))
            self._canvas.redraw()

        def _on_xlabel(change):
            self._stack.execute(
                Command(ax.xaxis.label, "text", ax.get_xlabel(), change["new"],
                        description="Change X label"))
            self._canvas.redraw()

        def _on_ylabel(change):
            self._stack.execute(
                Command(ax.yaxis.label, "text", ax.get_ylabel(), change["new"],
                        description="Change Y label"))
            self._canvas.redraw()

        title_text.observe(_on_title, names="value")
        xlabel_text.observe(_on_xlabel, names="value")
        ylabel_text.observe(_on_ylabel, names="value")

        # Per-label formatting: color + bold + italic
        def _make_fmt(getter, prefix):
            t0 = getter(ax)
            try:
                cur_c = to_hex(t0.get_color()) if t0 else '#000000'
            except Exception:
                cur_c = '#000000'
            fw = t0.get_fontweight() if t0 else 'normal'
            cur_b = fw == 'bold' or (isinstance(fw, (int, float)) and fw >= 600)
            cur_i = (t0.get_fontstyle() == 'italic') if t0 else False

            def _on_c(hex_val):
                t = getter(ax)
                if t:
                    self._stack.execute(
                        Command(t, "color", t.get_color(), hex_val,
                                description=f"{prefix} color"))
                    self._canvas.redraw()

            color_btn, swatch_row = cmap_color_btn(cur_c, _on_c, self._cmap_panel)

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
                t = getter(ax)
                if t:
                    self._stack.execute(
                        Command(t, "fontweight", t.get_fontweight(), w,
                                description=f"{prefix} weight"))
                    self._canvas.redraw()

            def _on_i(change):
                s = 'italic' if change["new"] else 'normal'
                t = getter(ax)
                if t:
                    self._stack.execute(
                        Command(t, "fontstyle", t.get_fontstyle(), s,
                                description=f"{prefix} style"))
                    self._canvas.redraw()

            bb.observe(_on_b, names="value")
            ib.observe(_on_i, names="value")

            top_row = widgets.HBox(
                [color_btn, bb, ib],
                layout=widgets.Layout(align_items='center', gap='2px',
                                      padding='0 0 0 52px'))
            swatch_row.layout.padding = '0 0 0 52px'
            return widgets.VBox([top_row, swatch_row])

        title_fmt = _make_fmt(lambda a: a.title, "Title")
        xlabel_fmt = _make_fmt(lambda a: a.xaxis.label, "X label")
        ylabel_fmt = _make_fmt(lambda a: a.yaxis.label, "Y label")

        return widgets.VBox([
            title_text, title_fmt,
            xlabel_text, xlabel_fmt,
            ylabel_text, ylabel_fmt,
        ])

    # ------------------------------------------------------------------
    # Ticks & Spines section (per-subplot)
    # ------------------------------------------------------------------

    def _build_ticks_spines(self) -> widgets.Widget:
        ax = self._ax

        # --- Tick label visibility ---
        cur_x_labels = True
        cur_y_labels = True
        xt_labels = ax.get_xticklabels()
        yt_labels = ax.get_yticklabels()
        if xt_labels:
            cur_x_labels = xt_labels[0].get_visible()
        if yt_labels:
            cur_y_labels = yt_labels[0].get_visible()

        x_labels_cb = widgets.Checkbox(
            value=cur_x_labels, description="X tick labels", indent=False,
            layout=widgets.Layout(width="auto"))
        y_labels_cb = widgets.Checkbox(
            value=cur_y_labels, description="Y tick labels", indent=False,
            layout=widgets.Layout(width="auto"))

        def _on_x_labels(change):
            for lbl in ax.get_xticklabels():
                lbl.set_visible(change["new"])
            self._canvas.redraw()

        def _on_y_labels(change):
            for lbl in ax.get_yticklabels():
                lbl.set_visible(change["new"])
            self._canvas.redraw()

        x_labels_cb.observe(_on_x_labels, names="value")
        y_labels_cb.observe(_on_y_labels, names="value")

        # --- Tick direction / length / width ---
        cur_tick_dir = "out"
        cur_tick_len = 4.0
        cur_tick_width = 1.0
        xtick_objs = ax.xaxis.get_major_ticks()
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

        tick_dir_dd = widgets.Dropdown(
            options=["out", "in", "inout"], value=cur_tick_dir,
            description="Dir:", style=_STY, layout=_DD_SHORT)
        tick_len_sl = widgets.FloatSlider(
            value=cur_tick_len, min=0, max=15, step=0.5,
            description="Length:", style=_STY)
        tick_width_sl = widgets.FloatSlider(
            value=cur_tick_width, min=0.1, max=4, step=0.1,
            description="Width:", style=_STY)

        def _tick_cb(change):
            ax.tick_params(direction=tick_dir_dd.value,
                           length=tick_len_sl.value,
                           width=tick_width_sl.value)
            self._canvas.redraw()
        tick_dir_dd.observe(_tick_cb, names="value")
        tick_len_sl.observe(_tick_cb, names="value")
        tick_width_sl.observe(_tick_cb, names="value")

        # --- Spine visibility ---
        cur_spine_top = ax.spines["top"].get_visible()
        cur_spine_right = ax.spines["right"].get_visible()
        cur_spine_bottom = ax.spines["bottom"].get_visible()
        cur_spine_left = ax.spines["left"].get_visible()
        cur_spine_width = round(ax.spines["bottom"].get_linewidth(), 1)

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
        spine_width_sl = widgets.FloatSlider(
            value=cur_spine_width, min=0.1, max=5.0, step=0.1,
            description="Width:", style=_STY)

        def _spine_cb(spine_name):
            def _cb(change):
                sp = ax.spines[spine_name]
                self._stack.execute(
                    Command(sp, "visible", sp.get_visible(), change["new"],
                            description=f"Toggle {spine_name} spine"))
                self._canvas.redraw()
            return _cb

        self._spine_top_cb.observe(_spine_cb("top"), names="value")
        self._spine_right_cb.observe(_spine_cb("right"), names="value")
        self._spine_bottom_cb.observe(_spine_cb("bottom"), names="value")
        self._spine_left_cb.observe(_spine_cb("left"), names="value")

        def _spine_width_cb(change):
            cmds = []
            for name in ("top", "right", "bottom", "left"):
                sp = ax.spines[name]
                cmds.append(Command(sp, "linewidth", sp.get_linewidth(),
                                    change["new"]))
            self._stack.execute(BatchCommand(cmds, "Spine width"))
            self._canvas.redraw()
        spine_width_sl.observe(_spine_width_cb, names="value")

        return widgets.VBox([
            widgets.HTML("<b>Tick Labels</b>"),
            widgets.HBox([x_labels_cb, y_labels_cb]),
            widgets.HTML("<b>Tick Style</b>"),
            tick_dir_dd,
            _slider_num(tick_len_sl, _GDW),
            _slider_num(tick_width_sl, _GDW),
            widgets.HTML("<b>Spines</b>"),
            widgets.HBox([self._spine_top_cb, self._spine_right_cb,
                          self._spine_bottom_cb, self._spine_left_cb]),
            _slider_num(spine_width_sl, _GDW),
        ])

    # ------------------------------------------------------------------
    # Axes section (limits, scale, tick spacing, grid)
    # ------------------------------------------------------------------

    def _build_axes(self) -> widgets.Widget:
        ax = self._ax

        # Equal axes
        equal_cb = widgets.Checkbox(
            value=False, description="Equal axes", indent=False,
            layout=widgets.Layout(width="auto"))

        # Axis limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()

        xlim_min = widgets.FloatText(
            value=round(cur_xlim[0], 4), description="X min:",
            style={"description_width": "46px"},
            layout=widgets.Layout(width="140px"))
        xlim_max = widgets.FloatText(
            value=round(cur_xlim[1], 4), description="X max:",
            style={"description_width": "46px"},
            layout=widgets.Layout(width="140px"))
        ylim_min = widgets.FloatText(
            value=round(cur_ylim[0], 4), description="Y min:",
            style={"description_width": "46px"},
            layout=widgets.Layout(width="140px"))
        ylim_max = widgets.FloatText(
            value=round(cur_ylim[1], 4), description="Y max:",
            style={"description_width": "46px"},
            layout=widgets.Layout(width="140px"))

        _lim_updating = [False]

        # Tick spacing
        cur_x_step = 0.0
        cur_y_step = 0.0
        xt = ax.get_xticks()
        if len(xt) >= 2:
            cur_x_step = round(abs(xt[1] - xt[0]), 4)
        yt = ax.get_yticks()
        if len(yt) >= 2:
            cur_y_step = round(abs(yt[1] - yt[0]), 4)

        x_step = widgets.FloatText(
            value=cur_x_step, description="X step:",
            style={"description_width": "46px"},
            layout=widgets.Layout(width="120px"))
        y_step = widgets.FloatText(
            value=cur_y_step, description="Y step:",
            style={"description_width": "46px"},
            layout=widgets.Layout(width="120px"))

        _step_updating = [False]

        def _apply_equal():
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
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            xs = x_step.value
            ys = y_step.value
            step = max(xs, ys)
            _step_updating[0] = True
            try:
                x_step.value = step
                y_step.value = step
            finally:
                _step_updating[0] = False
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
            def _cb(change):
                if _lim_updating[0]:
                    return
                if equal_cb.value:
                    _lim_updating[0] = True
                    try:
                        mirror_w.value = change["new"]
                    finally:
                        _lim_updating[0] = False
                ax.set_xlim(xlim_min.value, xlim_max.value)
                ax.set_ylim(ylim_min.value, ylim_max.value)
                self._canvas.force_redraw()
            return _cb

        xlim_min.observe(_make_lim_cb(xlim_min, ylim_min), names="value")
        xlim_max.observe(_make_lim_cb(xlim_max, ylim_max), names="value")
        ylim_min.observe(_make_lim_cb(ylim_min, xlim_min), names="value")
        ylim_max.observe(_make_lim_cb(ylim_max, xlim_max), names="value")

        # Axis scale
        cur_xscale = ax.get_xscale()
        cur_yscale = ax.get_yscale()
        _scale_opts = ["linear", "log", "symlog"]
        xscale_dd = widgets.Dropdown(
            options=_scale_opts,
            value=cur_xscale if cur_xscale in _scale_opts else "linear",
            description="X scale:", style=_STY, layout=_DD_SHORT)
        yscale_dd = widgets.Dropdown(
            options=_scale_opts,
            value=cur_yscale if cur_yscale in _scale_opts else "linear",
            description="Y scale:", style=_STY, layout=_DD_SHORT)

        def _scale_cb(change):
            ax.set_xscale(xscale_dd.value)
            ax.set_yscale(yscale_dd.value)
            self._canvas.redraw()
        xscale_dd.observe(_scale_cb, names="value")
        yscale_dd.observe(_scale_cb, names="value")

        # Tick spacing callbacks
        def _tick_step_cb(change):
            if _step_updating[0]:
                return
            from matplotlib.ticker import AutoLocator, MultipleLocator
            if equal_cb.value and change is not None:
                _step_updating[0] = True
                try:
                    v = change["new"]
                    x_step.value = v
                    y_step.value = v
                finally:
                    _step_updating[0] = False
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

        # Grid
        cur_grid_on = False
        cur_grid_alpha = 0.2
        cur_grid_width = 0.8
        cur_grid_style = "-"
        gridlines = ax.xaxis.get_gridlines()
        if gridlines:
            cur_grid_on = gridlines[0].get_visible()
            cur_grid_alpha = round(gridlines[0].get_alpha() or 0.5, 2)
            cur_grid_width = round(gridlines[0].get_linewidth(), 1)
            cur_grid_style = gridlines[0].get_linestyle()
            _ls_map = {"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-."}
            cur_grid_style = _ls_map.get(cur_grid_style, cur_grid_style)

        grid_toggle = widgets.Checkbox(
            value=cur_grid_on, description="Show grid", indent=False)
        grid_alpha_sl = widgets.FloatSlider(
            value=cur_grid_alpha, min=0.05, max=1, step=0.05,
            description="Alpha:", style=_STY)
        grid_width_sl = widgets.FloatSlider(
            value=cur_grid_width, min=0.1, max=4, step=0.1,
            description="Width:", style=_STY)

        valid_styles = ["-", "--", ":", "-."]
        if cur_grid_style not in valid_styles:
            cur_grid_style = "-"
        grid_style_dd = widgets.Dropdown(
            options=[("-", "-"), ("--", "--"), (":", ":"), ("-.", "-.")],
            value=cur_grid_style, description="Style:",
            style=_STY, layout=_DD_SHORT)

        def _grid_cb(change):
            if grid_toggle.value:
                ax.grid(True, alpha=grid_alpha_sl.value,
                        linewidth=grid_width_sl.value,
                        linestyle=grid_style_dd.value)
            else:
                ax.grid(False)
            self._canvas.redraw()
        grid_toggle.observe(_grid_cb, names="value")
        grid_alpha_sl.observe(_grid_cb, names="value")
        grid_width_sl.observe(_grid_cb, names="value")
        grid_style_dd.observe(_grid_cb, names="value")

        return widgets.VBox([
            equal_cb,
            widgets.HBox([xlim_min, xlim_max]),
            widgets.HBox([ylim_min, ylim_max]),
            widgets.HTML("<b>Scale</b>"),
            xscale_dd, yscale_dd,
            widgets.HTML("<b>Tick Spacing</b>"),
            widgets.HBox([x_step, y_step]),
            widgets.HTML("<small style='color:#888'>step 0 = auto spacing</small>"),
            widgets.HTML("<b>Grid</b>"),
            grid_toggle,
            _slider_num(grid_alpha_sl, _GDW),
            _slider_num(grid_width_sl, _GDW),
            grid_style_dd,
        ])

    # ------------------------------------------------------------------
    # Legend section
    # ------------------------------------------------------------------

    def _build_legend(self) -> widgets.Widget:
        ax = self._ax
        leg = ax.get_legend()
        has_legend = leg is not None

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
        if cur_loc == "best" or cur_loc not in valid_values:
            cur_loc = "upper right"
            if leg is not None:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    info = _pos_info[cur_loc]
                    ax.legend(handles, labels,
                              loc=info["loc"],
                              bbox_to_anchor=info["xy"],
                              frameon=cur_frame,
                              fontsize=cur_fontsize,
                              ncol=cur_ncol,
                              markerfirst=cur_markerfirst,
                              handletextpad=cur_handletextpad,
                              handleheight=cur_handleheight)
                    leg = ax.get_legend()

        init_xy = _pos_info.get(cur_loc, {}).get("xy", (1.0, 1.0))
        cur_x, cur_y = init_xy
        if leg is not None and hasattr(leg, "_bbox_to_anchor") and \
                leg._bbox_to_anchor is not None:
            try:
                inv = ax.transAxes.inverted()
                ax_x, ax_y = inv.transform(
                    (leg._bbox_to_anchor.x0, leg._bbox_to_anchor.y0))
                if (ax_x > 1.01 or ax_x < -0.01
                        or ax_y > 1.01 or ax_y < -0.01):
                    cur_x, cur_y = round(ax_x, 2), round(ax_y, 2)
            except Exception:
                pass

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
        legend_fontsize_sl = widgets.FloatSlider(
            value=cur_fontsize, min=4, max=24,
            step=0.5, description="Font sz:", style=_STY)
        ncol = widgets.IntSlider(value=cur_ncol, min=1, max=6, step=1,
                                  description="Cols:", style=_STY)
        markerfirst_cb = widgets.Checkbox(
            value=cur_markerfirst, description="Marker first", indent=False)
        handletextpad_sl = widgets.FloatSlider(
            value=cur_handletextpad, min=0, max=3, step=0.1,
            description="Spacing:", style=_STY)
        handleheight_sl = widgets.FloatSlider(
            value=cur_handleheight, min=0.1, max=3, step=0.1,
            description="Height:", style=_STY)

        controls_box = widgets.VBox([
            pos_dd,
            _slider_num(x_sl, _GDW),
            _slider_num(y_sl, _GDW),
            self._frame_toggle,
            markerfirst_cb,
            _slider_num(legend_fontsize_sl, _GDW),
            _slider_num(handletextpad_sl, _GDW),
            _slider_num(handleheight_sl, _GDW),
            _slider_num(ncol, _GDW),
        ], layout=widgets.Layout(
            display="" if has_legend else "none"))

        _updating = [False]

        # Per-entry legend text colors
        _all_handles, _all_labels = ax.get_legend_handles_labels()
        ax._matplotly_leg_text_colors = ['#000000'] * len(_all_labels)
        if leg:
            for _ti, _tt in enumerate(leg.get_texts()):
                try:
                    if _ti < len(ax._matplotly_leg_text_colors):
                        ax._matplotly_leg_text_colors[_ti] = to_hex(_tt.get_color())
                except Exception:
                    pass

        _leg_color_rows = []
        self._leg_label_widgets = []
        for _idx, _lbl_text in enumerate(_all_labels):
            _cur_lc = '#000000'
            if leg and _idx < len(leg.get_texts()):
                try:
                    _cur_lc = to_hex(leg.get_texts()[_idx].get_color())
                except Exception:
                    pass

            def _make_lc_cb(i):
                def _on_lc(hex_val):
                    if hasattr(ax, '_matplotly_leg_text_colors'):
                        while len(ax._matplotly_leg_text_colors) <= i:
                            ax._matplotly_leg_text_colors.append('#000000')
                        ax._matplotly_leg_text_colors[i] = hex_val
                    _aleg = ax.get_legend()
                    if _aleg and i < len(_aleg.get_texts()):
                        _aleg.get_texts()[i].set_color(hex_val)
                    self._canvas.force_redraw()
                return _on_lc

            _lc_btn, _lc_swatch = cmap_color_btn(
                _cur_lc, _make_lc_cb(_idx), self._cmap_panel)
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

        def _create_legend_on_ax():
            info = _pos_info.get(pos_dd.value, {})
            loc = info.get("loc", "upper right")
            handles, labels = ax.get_legend_handles_labels()
            if not handles:
                return
            ax.legend(handles, labels,
                      loc=loc,
                      bbox_to_anchor=(x_sl.value, y_sl.value),
                      frameon=self._frame_toggle.value,
                      fontsize=legend_fontsize_sl.value,
                      ncol=ncol.value,
                      markerfirst=markerfirst_cb.value,
                      handletextpad=handletextpad_sl.value,
                      handleheight=handleheight_sl.value)
            if hasattr(ax, '_matplotly_leg_text_colors'):
                _aleg = ax.get_legend()
                if _aleg:
                    for _ci, _cc in enumerate(ax._matplotly_leg_text_colors):
                        if _ci < len(_aleg.get_texts()):
                            _aleg.get_texts()[_ci].set_color(_cc)

        def _remove_legend_from_ax():
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

        def _on_legend_toggle(change):
            if _updating[0]:
                return
            if change["new"]:
                controls_box.layout.display = ""
                leg_colors_box.layout.display = ""
                _create_legend_on_ax()
            else:
                controls_box.layout.display = "none"
                leg_colors_box.layout.display = "none"
                _remove_legend_from_ax()
            self._canvas.force_redraw()

        self._legend_toggle.observe(_on_legend_toggle, names="value")

        def _apply_legend(*_):
            if _updating[0]:
                return
            if not self._legend_toggle.value:
                return
            _create_legend_on_ax()
            self._canvas.force_redraw()

        pos_dd.observe(_on_pos_change, names="value")
        for w in [x_sl, y_sl, self._frame_toggle, markerfirst_cb,
                  legend_fontsize_sl, handletextpad_sl,
                  handleheight_sl, ncol]:
            w.observe(_apply_legend, names="value")

        return widgets.VBox([self._legend_toggle, controls_box, leg_colors_box])

    def _refresh_legend_labels(self):
        """Update text color row labels to match current artist labels."""
        _, labels = self._ax.get_legend_handles_labels()
        for i, lbl_w in enumerate(self._leg_label_widgets):
            if i < len(labels):
                lbl_w.value = labels[i]
