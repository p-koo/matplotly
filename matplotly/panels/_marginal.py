"""Marginal histogram controls — figure-level panel for all scatter collections.

Each marginal (X and Y) has its own independent toggle and full parameter set:
bins, height, alpha, pad, tick controls (side, fontsize, step, range),
label (text, fontsize, bold, italic, color), title (text, fontsize, bold,
italic, color), and individual spine checkboxes.
"""
from __future__ import annotations

import ipywidgets as widgets
import numpy as np
from matplotlib.colors import to_hex
from matplotlib.ticker import AutoLocator, MultipleLocator

from .._commands import Command
from .._renderer import CanvasManager
from ._color_utils import _SN, _slider_num


class MarginalHistogramManager:
    """Manages marginal histogram axes for all scatter collections on an axes."""

    def __init__(self, fig, main_ax, scatter_colls, stack, canvas: CanvasManager):
        self._fig = fig
        self._ax = main_ax
        self._colls = list(scatter_colls)
        self._stack = stack
        self._canvas = canvas
        self._ax_x = None
        self._ax_y = None

        # Shared settings
        self._mode = 'overlay'
        self._separation = 0.1

        # Read scatter plot font sizes for defaults
        _tick_fs = 10.0
        _label_fs = 10.0
        _title_fs = 10.0
        try:
            xticks = main_ax.get_xticklabels()
            if xticks:
                _tick_fs = round(xticks[0].get_fontsize(), 1)
            _label_fs = round(main_ax.xaxis.label.get_fontsize(), 1)
            _title_fs = round(main_ax.title.get_fontsize(), 1)
        except Exception:
            pass

        # Per-histogram settings (fully independent)
        self._x = self._default_settings('x', _tick_fs, _label_fs, _title_fs)
        self._y = self._default_settings('y', _tick_fs, _label_fs, _title_fs)

        # Auto-tick state per histogram
        self._x_auto_ticks = True
        self._y_auto_ticks = True
        # UI update hooks (set by _build_section)
        self._x_tick_ui = None
        self._y_tick_ui = None

        # Register on figure so renderer can reposition after tight_layout
        if not hasattr(fig, '_matplotly_marginal_managers'):
            fig._matplotly_marginal_managers = []
        fig._matplotly_marginal_managers.append(self)

    @staticmethod
    def _default_settings(which, tick_fs=10.0, label_fs=10.0, title_fs=10.0):
        return {
            'enabled': False,
            'position': 'top' if which == 'x' else 'right',
            'bins': 20,
            'height': 1.0,
            'alpha': 0.5,
            'pad': 0,
            # Tick controls
            'tick_side': 'left' if which == 'x' else 'bottom',
            'tick_fontsize': tick_fs,
            'tick_step': 0,      # 0 = auto
            'range_min': 0,      # 0,0 = auto
            'range_max': 0,
            # Label
            'label': '',
            'label_fontsize': label_fs,
            'label_bold': False,
            'label_italic': False,
            'label_color': '#000000',
            # Title
            'title': '',
            'title_fontsize': title_fs,
            'title_bold': False,
            'title_italic': False,
            'title_color': '#000000',
            # Spines
            'spines': {'top': False, 'right': False,
                       'bottom': True, 'left': True},
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_color(coll):
        try:
            return to_hex(coll.get_facecolor()[0])
        except Exception:
            return '#1f77b4'

    def _get_coll_index(self, coll):
        for i, c in enumerate(self._ax.collections):
            if c is coll:
                return i
        return 0

    def _parent_ax_index(self):
        for i, a in enumerate(self._fig.get_axes()):
            if getattr(a, '_matplotly_marginal', False):
                continue
            if a is self._ax:
                return i
        return 0

    def _compute_default_spines(self, which):
        """Compute sensible spine visibility from position and tick side."""
        s = self._x if which == 'x' else self._y
        pos = s['position']
        ts = s['tick_side']
        result = {n: False for n in ('top', 'right', 'bottom', 'left')}

        if which == 'x':
            # Adjacent spine (between marginal and main plot)
            result['bottom' if pos == 'top' else 'top'] = True
            # Tick-side spine
            if ts in ('left', 'right'):
                result[ts] = True
        else:
            result['left' if pos == 'right' else 'right'] = True
            if ts in ('top', 'bottom'):
                result[ts] = True
        return result

    def _clear_data(self, ax):
        """Remove plotted data without resetting shared axis properties.

        Unlike cla(), this preserves the shared axis locator/formatter
        so the main scatter plot ticks are not affected.
        """
        while ax.patches:
            ax.patches[0].remove()
        if hasattr(ax, 'containers'):
            ax.containers.clear()
        ax.relim()

    def _read_auto_ticks(self, which):
        """Read the auto-computed tick step and range from a marginal axes."""
        ax = self._ax_x if which == 'x' else self._ax_y
        if ax is None:
            return None
        # Count axis is Y for x-marginal, X for y-marginal
        if which == 'x':
            ticks = ax.get_yticks()
            lim = ax.get_ylim()
        else:
            ticks = ax.get_xticks()
            lim = ax.get_xlim()
        step = 0.0
        if len(ticks) >= 2:
            step = round(abs(ticks[1] - ticks[0]), 4)
        rmin = round(min(lim), 4)
        rmax = round(max(lim), 4)
        return {'step': step, 'min': rmin, 'max': rmax}

    def _sync_auto_ui(self, which):
        """If auto mode is on, read auto values and update text boxes."""
        is_auto = self._x_auto_ticks if which == 'x' else self._y_auto_ticks
        ui = self._x_tick_ui if which == 'x' else self._y_tick_ui
        if not is_auto or ui is None:
            return
        vals = self._read_auto_ticks(which)
        if vals is None:
            return
        guard, step_w, rmin_w, rmax_w = ui
        guard[0] = True
        try:
            step_w.value = vals['step']
            rmin_w.value = vals['min']
            rmax_w.value = vals['max']
        finally:
            guard[0] = False

    # ------------------------------------------------------------------
    # Positioning
    # ------------------------------------------------------------------

    def _compute_x_rect(self):
        pos = self._ax.get_position()
        fig_h = self._fig.get_size_inches()[1]
        h_frac = self._x['height'] / fig_h
        pad_y = self._x['pad'] / fig_h
        if self._x['position'] == 'top':
            return [pos.x0, pos.y1 + pad_y, pos.width, h_frac]
        else:
            return [pos.x0, pos.y0 - h_frac - pad_y, pos.width, h_frac]

    def _compute_y_rect(self):
        pos = self._ax.get_position()
        fig_w = self._fig.get_size_inches()[0]
        w_frac = self._y['height'] / fig_w
        pad_x = self._y['pad'] / fig_w
        if self._y['position'] == 'right':
            return [pos.x1 + pad_x, pos.y0, w_frac, pos.height]
        else:
            return [pos.x0 - w_frac - pad_x, pos.y0, w_frac, pos.height]

    # ------------------------------------------------------------------
    # Enable / disable
    # ------------------------------------------------------------------

    def _enable_x(self):
        if self._ax_x is not None:
            return
        rect = self._compute_x_rect()
        self._ax_x = self._fig.add_axes(rect, sharex=self._ax)
        self._ax_x._matplotly_marginal = True
        self._draw_x()

    def _disable_x(self):
        if self._ax_x is None:
            return
        self._fig.delaxes(self._ax_x)
        self._ax_x = None

    def _enable_y(self):
        if self._ax_y is not None:
            return
        rect = self._compute_y_rect()
        self._ax_y = self._fig.add_axes(rect, sharey=self._ax)
        self._ax_y._matplotly_marginal = True
        self._draw_y()

    def _disable_y(self):
        if self._ax_y is None:
            return
        self._fig.delaxes(self._ax_y)
        self._ax_y = None

    # ------------------------------------------------------------------
    # Global bins
    # ------------------------------------------------------------------

    def _global_bin_edges(self, data_col, n_bins):
        """Compute shared bin edges across all scatter collections."""
        arrays = []
        for coll in self._colls:
            offsets = coll.get_offsets()
            if len(offsets) > 0:
                arrays.append(offsets[:, data_col])
        if not arrays:
            return np.linspace(0, 1, n_bins + 1)
        combined = np.concatenate(arrays)
        return np.histogram_bin_edges(combined, bins=n_bins)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_x(self):
        if self._ax_x is None:
            return
        ax = self._ax_x
        self._clear_data(ax)
        ax._matplotly_marginal = True

        # Save main axes limits — hist()/bar() can affect shared xlim
        main_xlim = self._ax.get_xlim()

        s = self._x
        bin_edges = self._global_bin_edges(0, s['bins'])
        alpha = s['alpha']

        if self._mode == 'overlay':
            for coll in self._colls:
                offsets = coll.get_offsets()
                if len(offsets) == 0:
                    continue
                ax.hist(offsets[:, 0], bins=bin_edges,
                        color=self._get_color(coll),
                        alpha=alpha, edgecolor='none')
        else:  # dodge
            n = len(self._colls)
            bin_w = bin_edges[1] - bin_edges[0]
            sub_w = bin_w / n
            bar_w = sub_w * (1 - self._separation)
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            for i, coll in enumerate(self._colls):
                offsets = coll.get_offsets()
                if len(offsets) == 0:
                    continue
                counts, _ = np.histogram(offsets[:, 0], bins=bin_edges)
                offset = sub_w * (i - (n - 1) / 2)
                ax.bar(centers + offset, counts, width=bar_w,
                       color=self._get_color(coll), alpha=alpha,
                       edgecolor='none')

        # Restore main axes limits
        self._ax.set_xlim(main_xlim)

        # Invert for bottom position (bars grow downward)
        if s['position'] == 'bottom':
            if not ax.yaxis_inverted():
                ax.invert_yaxis()
        else:
            if ax.yaxis_inverted():
                ax.invert_yaxis()

        self._apply_config(ax, 'x')
        self._store_info(ax, 'x')
        self._sync_auto_ui('x')

    def _draw_y(self):
        if self._ax_y is None:
            return
        ax = self._ax_y
        self._clear_data(ax)
        ax._matplotly_marginal = True

        # Save main axes limits — hist()/bar() can affect shared ylim
        main_ylim = self._ax.get_ylim()

        s = self._y
        bin_edges = self._global_bin_edges(1, s['bins'])
        alpha = s['alpha']

        if self._mode == 'overlay':
            for coll in self._colls:
                offsets = coll.get_offsets()
                if len(offsets) == 0:
                    continue
                ax.hist(offsets[:, 1], bins=bin_edges,
                        color=self._get_color(coll),
                        alpha=alpha, orientation='horizontal',
                        edgecolor='none')
        else:  # dodge
            n = len(self._colls)
            bin_w = bin_edges[1] - bin_edges[0]
            sub_w = bin_w / n
            bar_h = sub_w * (1 - self._separation)
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            for i, coll in enumerate(self._colls):
                offsets = coll.get_offsets()
                if len(offsets) == 0:
                    continue
                counts, _ = np.histogram(offsets[:, 1], bins=bin_edges)
                offset = sub_w * (i - (n - 1) / 2)
                ax.barh(centers + offset, counts, height=bar_h,
                        color=self._get_color(coll), alpha=alpha,
                        edgecolor='none')

        # Restore main axes limits
        self._ax.set_ylim(main_ylim)

        # Invert for left position (bars grow leftward)
        if s['position'] == 'left':
            if not ax.xaxis_inverted():
                ax.invert_xaxis()
        else:
            if ax.xaxis_inverted():
                ax.invert_xaxis()

        self._apply_config(ax, 'y')
        self._store_info(ax, 'y')
        self._sync_auto_ui('y')

    # ------------------------------------------------------------------
    # Spine / tick / label / title configuration
    # ------------------------------------------------------------------

    def _apply_config(self, ax, which):
        """Apply spines, ticks, labels, range, step, and title."""
        s = self._x if which == 'x' else self._y

        # Spines
        for name in ('top', 'right', 'bottom', 'left'):
            ax.spines[name].set_visible(s['spines'][name])

        if which == 'x':
            # Shared X axis — hide ALL data-axis ticks and labels
            ax.tick_params(axis='x', bottom=False, top=False,
                           labelbottom=False, labeltop=False)
            # Count-axis (Y) ticks
            ts = s['tick_side']
            tfs = s['tick_fontsize']
            if ts == 'left':
                ax.tick_params(axis='y', left=True, labelleft=True,
                               right=False, labelright=False, labelsize=tfs)
            elif ts == 'right':
                ax.tick_params(axis='y', left=False, labelleft=False,
                               right=True, labelright=True, labelsize=tfs)
            else:  # none
                ax.tick_params(axis='y', left=False, labelleft=False,
                               right=False, labelright=False)
            # Tick step
            if s['tick_step'] > 0:
                ax.yaxis.set_major_locator(MultipleLocator(s['tick_step']))
            else:
                ax.yaxis.set_major_locator(AutoLocator())
            # Range
            rmax = s['range_max']
            if rmax > 0:
                rmin = s['range_min']
                if s['position'] == 'bottom':
                    ax.set_ylim(rmax, rmin)
                else:
                    ax.set_ylim(rmin, rmax)
            # Label
            lbl = s['label']
            if lbl:
                if ts == 'right':
                    ax.yaxis.set_label_position('right')
                else:
                    ax.yaxis.set_label_position('left')
                weight = 'bold' if s['label_bold'] else 'normal'
                style = 'italic' if s['label_italic'] else 'normal'
                ax.set_ylabel(lbl, fontsize=s['label_fontsize'],
                              fontweight=weight, fontstyle=style,
                              color=s['label_color'])
            else:
                ax.set_ylabel('')
        else:
            # Shared Y axis — hide ALL data-axis ticks and labels
            ax.tick_params(axis='y', left=False, right=False,
                           labelleft=False, labelright=False)
            # Count-axis (X) ticks
            ts = s['tick_side']
            tfs = s['tick_fontsize']
            if ts == 'bottom':
                ax.tick_params(axis='x', bottom=True, labelbottom=True,
                               top=False, labeltop=False, labelsize=tfs)
            elif ts == 'top':
                ax.tick_params(axis='x', bottom=False, labelbottom=False,
                               top=True, labeltop=True, labelsize=tfs)
            else:  # none
                ax.tick_params(axis='x', bottom=False, labelbottom=False,
                               top=False, labeltop=False)
            # Tick step
            if s['tick_step'] > 0:
                ax.xaxis.set_major_locator(MultipleLocator(s['tick_step']))
            else:
                ax.xaxis.set_major_locator(AutoLocator())
            # Range
            rmax = s['range_max']
            if rmax > 0:
                rmin = s['range_min']
                if s['position'] == 'left':
                    ax.set_xlim(rmax, rmin)
                else:
                    ax.set_xlim(rmin, rmax)
            # Label
            lbl = s['label']
            if lbl:
                if ts == 'top':
                    ax.xaxis.set_label_position('top')
                else:
                    ax.xaxis.set_label_position('bottom')
                weight = 'bold' if s['label_bold'] else 'normal'
                style = 'italic' if s['label_italic'] else 'normal'
                ax.set_xlabel(lbl, fontsize=s['label_fontsize'],
                              fontweight=weight, fontstyle=style,
                              color=s['label_color'])
            else:
                ax.set_xlabel('')

        # Title
        title = s['title']
        if title:
            weight = 'bold' if s['title_bold'] else 'normal'
            style = 'italic' if s['title_italic'] else 'normal'
            ax.set_title(title, fontsize=s['title_fontsize'],
                         fontweight=weight, fontstyle=style,
                         color=s['title_color'])
        else:
            ax.set_title('')

    # ------------------------------------------------------------------
    # Code-gen metadata
    # ------------------------------------------------------------------

    def _store_info(self, ax, which):
        s = self._x if which == 'x' else self._y
        coll_info = []
        for coll in self._colls:
            coll_info.append({
                'coll_index': self._get_coll_index(coll),
                'color': self._get_color(coll),
            })
        inverted = ((s['position'] == 'bottom') if which == 'x'
                    else (s['position'] == 'left'))
        ax._matplotly_marginal_info = {
            'axis': which,
            'parent_ax_index': self._parent_ax_index(),
            'mode': self._mode,
            'position': s['position'],
            'height': s['height'],
            'pad': s['pad'],
            'bins': s['bins'],
            'alpha': s['alpha'],
            'separation': self._separation,
            'inverted': inverted,
            'tick_side': s['tick_side'],
            'tick_fontsize': s['tick_fontsize'],
            'tick_step': s['tick_step'],
            'range_min': s['range_min'],
            'range_max': s['range_max'],
            'label': s['label'],
            'label_fontsize': s['label_fontsize'],
            'label_bold': s['label_bold'],
            'label_italic': s['label_italic'],
            'label_color': s['label_color'],
            'title': s['title'],
            'title_fontsize': s['title_fontsize'],
            'title_bold': s['title_bold'],
            'title_italic': s['title_italic'],
            'title_color': s['title_color'],
            'collections': coll_info,
        }

    # ------------------------------------------------------------------
    # Rebuild / sync
    # ------------------------------------------------------------------

    def _rebuild(self):
        """Shrink main axes to make room, then reposition marginals."""
        pos = self._ax.get_position()
        fig_w, fig_h = self._fig.get_size_inches()
        x0, y0, w, h = pos.x0, pos.y0, pos.width, pos.height

        if self._ax_x is not None:
            h_frac = self._x['height'] / fig_h
            pad_frac = self._x['pad'] / fig_h
            need = h_frac + pad_frac
            if self._x['position'] == 'top':
                h -= need
            else:  # bottom
                y0 += need
                h -= need

        if self._ax_y is not None:
            w_frac = self._y['height'] / fig_w
            pad_frac = self._y['pad'] / fig_w
            need = w_frac + pad_frac
            if self._y['position'] == 'right':
                w -= need
            else:  # left
                x0 += need
                w -= need

        if self._ax_x is not None or self._ax_y is not None:
            self._ax.set_position([x0, y0, w, h])

        self._rebuild_x()
        self._rebuild_y()

    def _rebuild_x(self):
        if self._ax_x is not None:
            self._ax_x.set_position(self._compute_x_rect())
            self._draw_x()

    def _rebuild_y(self):
        if self._ax_y is not None:
            self._ax_y.set_position(self._compute_y_rect())
            self._draw_y()

    def sync_colors(self):
        """Redraw histograms to pick up scatter color changes."""
        if self._ax_x is not None:
            self._draw_x()
        if self._ax_y is not None:
            self._draw_y()

    # ------------------------------------------------------------------
    # Widget
    # ------------------------------------------------------------------

    def build_widget(self) -> widgets.Widget:
        # --- Shared controls ---
        mode_dd = widgets.Dropdown(
            options=[('Overlay', 'overlay'), ('Side-by-side', 'dodge')],
            value=self._mode, description='Mode:', style=_SN,
            layout=widgets.Layout(width='180px'))
        sep_sl = widgets.FloatSlider(
            value=self._separation, min=0.0, max=0.8,
            step=0.05, description='Gap:', style=_SN)
        sep_row = _slider_num(sep_sl)
        sep_row.layout.display = '' if self._mode == 'dodge' else 'none'

        def _on_mode(change):
            self._mode = change['new']
            sep_row.layout.display = '' if change['new'] == 'dodge' else 'none'
            self._rebuild()
            if self._canvas:
                self._canvas.force_redraw()

        def _on_sep(change):
            self._separation = change['new']
            self._rebuild()
            if self._canvas:
                self._canvas.force_redraw()

        mode_dd.observe(_on_mode, names='value')
        sep_sl.observe(_on_sep, names='value')

        shared_box = widgets.VBox([mode_dd, sep_row])

        # --- Per-histogram sections ---
        x_section = self._build_section('x')
        y_section = self._build_section('y')

        return widgets.VBox([shared_box, x_section, y_section])

    def _build_section(self, which):
        """Build a collapsible toggle section for one marginal histogram."""
        s = self._x if which == 'x' else self._y
        title = 'X Histogram' if which == 'x' else 'Y Histogram'

        rebuild_fn = self._rebuild_x if which == 'x' else self._rebuild_y
        enable_fn = self._enable_x if which == 'x' else self._enable_y
        disable_fn = self._disable_x if which == 'x' else self._disable_y

        # --- Enable + position header ---
        enable_cb = widgets.Checkbox(
            value=s['enabled'], description=title, indent=False,
            layout=widgets.Layout(width='130px'))
        if which == 'x':
            pos_dd = widgets.Dropdown(
                options=['top', 'bottom'], value=s['position'],
                layout=widgets.Layout(width='90px'))
        else:
            pos_dd = widgets.Dropdown(
                options=['right', 'left'], value=s['position'],
                layout=widgets.Layout(width='90px'))

        header = widgets.HBox(
            [enable_cb, pos_dd],
            layout=widgets.Layout(align_items='center'))

        # --- Data controls ---
        bins_sl = widgets.IntSlider(
            value=s['bins'], min=5, max=100, step=1,
            description='Bins:', style=_SN)
        height_sl = widgets.FloatSlider(
            value=s['height'], min=0.2, max=5.0, step=0.05,
            description='Height:', style=_SN)
        alpha_sl = widgets.FloatSlider(
            value=s['alpha'], min=0.0, max=1.0, step=0.05,
            description='Alpha:', style=_SN)
        pad_sl = widgets.FloatSlider(
            value=s['pad'], min=0.0, max=2.0, step=0.01,
            description='Pad:', style=_SN)

        # --- Tick controls ---
        if which == 'x':
            tick_dd = widgets.Dropdown(
                options=['left', 'right', 'none'], value=s['tick_side'],
                description='Ticks:', style=_SN,
                layout=widgets.Layout(width='150px'))
        else:
            tick_dd = widgets.Dropdown(
                options=['bottom', 'top', 'none'], value=s['tick_side'],
                description='Ticks:', style=_SN,
                layout=widgets.Layout(width='150px'))

        tick_fontsize_sl = widgets.FloatSlider(
            value=s['tick_fontsize'], min=4, max=20, step=0.5,
            description='Tick sz:', style=_SN)

        tick_step_w = widgets.FloatText(
            value=s['tick_step'], description='Step:',
            style={"description_width": "36px"},
            layout=widgets.Layout(width='110px'))

        # --- Range controls ---
        range_min_w = widgets.FloatText(
            value=s['range_min'], description='Min:',
            style={"description_width": "30px"},
            layout=widgets.Layout(width='100px'))
        range_max_w = widgets.FloatText(
            value=s['range_max'], description='Max:',
            style={"description_width": "30px"},
            layout=widgets.Layout(width='100px'))

        # --- Auto button ---
        _auto_ref = self._x_auto_ticks if which == 'x' else self._y_auto_ticks
        auto_btn = widgets.Button(
            description='Auto', icon='refresh',
            layout=widgets.Layout(width='70px', height='24px',
                                  padding='0', min_width='0'))
        auto_btn.style.button_color = '#d4edda' if _auto_ref else '#f0f0f0'
        _tick_guard = [False]

        # Register UI hooks so _sync_auto_ui can update text boxes
        if which == 'x':
            self._x_tick_ui = (_tick_guard, tick_step_w,
                               range_min_w, range_max_w)
        else:
            self._y_tick_ui = (_tick_guard, tick_step_w,
                               range_min_w, range_max_w)

        tick_step_row = widgets.HBox(
            [tick_step_w, auto_btn],
            layout=widgets.Layout(align_items='center', gap='4px'))
        range_row = widgets.HBox(
            [widgets.Label('Range:', layout=widgets.Layout(width='42px')),
             range_min_w, range_max_w],
            layout=widgets.Layout(align_items='center', gap='4px'))

        # --- Label controls ---
        label_w = widgets.Text(
            value=s['label'], description='Label:', style=_SN,
            placeholder='e.g. Count',
            layout=widgets.Layout(width='95%'))
        label_fontsize_sl = widgets.FloatSlider(
            value=s['label_fontsize'], min=4, max=20, step=0.5,
            description='Size:', style=_SN)
        label_bold = widgets.ToggleButton(
            value=s['label_bold'], description='B', tooltip='Bold',
            layout=widgets.Layout(width='32px', height='24px',
                                  padding='0', min_width='0'))
        label_bold.style.font_weight = 'bold'
        label_italic = widgets.ToggleButton(
            value=s['label_italic'], description='I', tooltip='Italic',
            layout=widgets.Layout(width='32px', height='24px',
                                  padding='0', min_width='0'))
        label_italic.style.font_style = 'italic'
        label_color = widgets.ColorPicker(
            value=s['label_color'], concise=True,
            layout=widgets.Layout(width='28px', height='24px'))
        label_fmt_row = widgets.HBox(
            [label_color, label_bold, label_italic],
            layout=widgets.Layout(align_items='center', gap='2px',
                                  padding='0 0 0 52px'))

        # --- Title controls ---
        title_w = widgets.Text(
            value=s['title'], description='Title:', style=_SN,
            placeholder='(optional)',
            layout=widgets.Layout(width='95%'))
        title_fontsize_sl = widgets.FloatSlider(
            value=s['title_fontsize'], min=4, max=20, step=0.5,
            description='Size:', style=_SN)
        title_bold = widgets.ToggleButton(
            value=s['title_bold'], description='B', tooltip='Bold',
            layout=widgets.Layout(width='32px', height='24px',
                                  padding='0', min_width='0'))
        title_bold.style.font_weight = 'bold'
        title_italic = widgets.ToggleButton(
            value=s['title_italic'], description='I', tooltip='Italic',
            layout=widgets.Layout(width='32px', height='24px',
                                  padding='0', min_width='0'))
        title_italic.style.font_style = 'italic'
        title_color = widgets.ColorPicker(
            value=s['title_color'], concise=True,
            layout=widgets.Layout(width='28px', height='24px'))
        title_fmt_row = widgets.HBox(
            [title_color, title_bold, title_italic],
            layout=widgets.Layout(align_items='center', gap='2px',
                                  padding='0 0 0 52px'))

        # --- Spine checkboxes ---
        sp = s['spines']
        sp_left = widgets.Checkbox(
            value=sp['left'], description='L', indent=False,
            layout=widgets.Layout(width='auto'))
        sp_bottom = widgets.Checkbox(
            value=sp['bottom'], description='B', indent=False,
            layout=widgets.Layout(width='auto'))
        sp_right = widgets.Checkbox(
            value=sp['right'], description='R', indent=False,
            layout=widgets.Layout(width='auto'))
        sp_top = widgets.Checkbox(
            value=sp['top'], description='T', indent=False,
            layout=widgets.Layout(width='auto'))
        spine_row = widgets.HBox(
            [widgets.Label('Spines:', layout=widgets.Layout(width='50px')),
             sp_left, sp_bottom, sp_right, sp_top],
            layout=widgets.Layout(align_items='center'))

        controls_box = widgets.VBox([
            _slider_num(bins_sl), _slider_num(height_sl),
            _slider_num(alpha_sl), _slider_num(pad_sl),
            widgets.HTML("<small><b>Ticks</b></small>"),
            tick_dd, _slider_num(tick_fontsize_sl),
            tick_step_row, range_row,
            widgets.HTML("<small><b>Label</b></small>"),
            label_w, _slider_num(label_fontsize_sl), label_fmt_row,
            widgets.HTML("<small><b>Title</b></small>"),
            title_w, _slider_num(title_fontsize_sl), title_fmt_row,
            spine_row,
        ], layout=widgets.Layout(
            display='' if s['enabled'] else 'none',
            padding='2px 0 4px 8px'))

        # --- Guard flag for programmatic spine updates ---
        _spine_updating = [False]

        def _update_spine_cbs(sp_dict):
            """Sync spine checkbox values without triggering rebuild."""
            _spine_updating[0] = True
            try:
                sp_top.value = sp_dict['top']
                sp_right.value = sp_dict['right']
                sp_bottom.value = sp_dict['bottom']
                sp_left.value = sp_dict['left']
            finally:
                _spine_updating[0] = False

        # --- Callbacks ---
        def _on_enable(change):
            old = s['enabled']
            new = change['new']
            s['enabled'] = new
            controls_box.layout.display = '' if new else 'none'
            def _apply():
                s['enabled'] = new
                (enable_fn if new else disable_fn)()
            def _revert():
                s['enabled'] = old
                (enable_fn if old else disable_fn)()
                enable_cb.value = old
                controls_box.layout.display = '' if old else 'none'
            if self._stack:
                self._stack.execute(
                    Command(self._ax, f'_marginal_{which}', old, new,
                            apply_fn=_apply, revert_fn=_revert,
                            description=f'marginal {which.upper()} histogram'))
            if self._canvas:
                self._canvas.force_redraw()

        def _on_pos(change):
            s['position'] = change['new']
            # Reset spines to sensible defaults for new position
            new_spines = self._compute_default_spines(which)
            s['spines'] = new_spines
            _update_spine_cbs(new_spines)
            ax_ref = self._ax_x if which == 'x' else self._ax_y
            if ax_ref is not None:
                disable_fn()
                enable_fn()
            if self._canvas:
                self._canvas.force_redraw()

        def _on_tick_side(change):
            s['tick_side'] = change['new']
            # Auto-update spines to follow tick side
            new_spines = self._compute_default_spines(which)
            s['spines'] = new_spines
            _update_spine_cbs(new_spines)
            rebuild_fn()
            if self._canvas:
                self._canvas.force_redraw()

        def _on_spine(name):
            def _cb(change):
                if _spine_updating[0]:
                    return
                s['spines'][name] = change['new']
                rebuild_fn()
                if self._canvas:
                    self._canvas.force_redraw()
            return _cb

        def _on_setting(key):
            def _cb(change):
                s[key] = change['new']
                rebuild_fn()
                if self._canvas:
                    self._canvas.force_redraw()
            return _cb

        enable_cb.observe(_on_enable, names='value')
        pos_dd.observe(_on_pos, names='value')
        tick_dd.observe(_on_tick_side, names='value')

        sp_top.observe(_on_spine('top'), names='value')
        sp_right.observe(_on_spine('right'), names='value')
        sp_bottom.observe(_on_spine('bottom'), names='value')
        sp_left.observe(_on_spine('left'), names='value')

        for sl, key in [(bins_sl, 'bins'), (height_sl, 'height'),
                        (alpha_sl, 'alpha'), (pad_sl, 'pad'),
                        (tick_fontsize_sl, 'tick_fontsize')]:
            sl.observe(_on_setting(key), names='value')

        label_w.observe(_on_setting('label'), names='value')
        label_fontsize_sl.observe(_on_setting('label_fontsize'), names='value')
        label_bold.observe(_on_setting('label_bold'), names='value')
        label_italic.observe(_on_setting('label_italic'), names='value')
        label_color.observe(_on_setting('label_color'), names='value')

        title_w.observe(_on_setting('title'), names='value')
        title_fontsize_sl.observe(_on_setting('title_fontsize'), names='value')
        title_bold.observe(_on_setting('title_bold'), names='value')
        title_italic.observe(_on_setting('title_italic'), names='value')
        title_color.observe(_on_setting('title_color'), names='value')

        def _on_tick_manual(key):
            """Tick step / range changed by user — turn off auto."""
            def _cb(change):
                if _tick_guard[0]:
                    return
                s[key] = change['new']
                if which == 'x':
                    self._x_auto_ticks = False
                else:
                    self._y_auto_ticks = False
                auto_btn.style.button_color = '#f0f0f0'
                rebuild_fn()
                if self._canvas:
                    self._canvas.force_redraw()
            return _cb

        def _on_auto(btn):
            """Restore auto mode: clear overrides, redraw, fill in values."""
            if which == 'x':
                self._x_auto_ticks = True
            else:
                self._y_auto_ticks = True
            auto_btn.style.button_color = '#d4edda'
            s['tick_step'] = 0
            s['range_min'] = 0
            s['range_max'] = 0
            rebuild_fn()
            if self._canvas:
                self._canvas.force_redraw()
            # _sync_auto_ui is called inside draw, so values are filled

        auto_btn.on_click(_on_auto)
        tick_step_w.observe(_on_tick_manual('tick_step'), names='value')
        range_min_w.observe(_on_tick_manual('range_min'), names='value')
        range_max_w.observe(_on_tick_manual('range_max'), names='value')

        return widgets.VBox(
            [header, controls_box],
            layout=widgets.Layout(
                border='1px solid #e0e0e0', border_radius='4px',
                margin='4px 0', padding='4px'))
