"""PlotBuildSession — main orchestrator that wires everything together."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import ipywidgets as widgets
from IPython.display import display
from matplotlib.figure import Figure

from ._code_gen import generate_code
from ._commands import CommandStack
from ._introspect import FigureIntrospector
from ._renderer import CanvasManager
from ._profiles import create_profiles_panel
from ._types import ArtistGroup, PlotType
from .panels import create_panel
from .panels._global import GlobalPanel
from .panels._histogram import HistogramSharedPanel
from .panels._line import ColormapPanel
from .panels._marginal import MarginalHistogramManager
from .panels._subplot import PerSubplotPanel


def _extract_hist_calls(source: str) -> list[dict]:
    """Extract .hist() call info from cell source using AST.

    Returns list of dicts with keys:
        start_line  – 1-indexed first line of the call
        end_line    – 1-indexed last line of the call
        data_var    – source text of the first positional argument
        label       – string label if a label=... kwarg is present
    """
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    results: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == 'hist'):
            continue
        if not node.args:
            continue

        data_var = ast.get_source_segment(source, node.args[0])
        if data_var is None:
            continue

        # Try to extract label keyword
        label = None
        for kw in node.keywords:
            if kw.arg == 'label' and isinstance(kw.value, ast.Constant):
                label = kw.value.value
                break

        results.append({
            'start_line': node.lineno,
            'end_line': node.end_lineno or node.lineno,
            'data_var': data_var,
            'label': label,
        })
    return results


def _match_hist_data_vars(
    calls: list[dict], infos: list[dict],
) -> list[str]:
    """Match extracted hist calls to hist_info entries, return data var names.

    Tries to match by label first; falls back to positional order.
    """
    info_labels = [hi.get('label', '') for hi in infos]
    matched: list[str | None] = [None] * len(infos)

    # Pass 1: match by label
    used_calls: set[int] = set()
    for i, il in enumerate(info_labels):
        for j, c in enumerate(calls):
            if j not in used_calls and c.get('label') == il and il:
                matched[i] = c['data_var']
                used_calls.add(j)
                break

    # Pass 2: fill remaining by positional order
    unused = [c for j, c in enumerate(calls) if j not in used_calls]
    ui = 0
    for i in range(len(matched)):
        if matched[i] is None and ui < len(unused):
            matched[i] = unused[ui]['data_var']
            ui += 1

    return [m if m is not None else f'<data_{i}>' for i, m in enumerate(matched)]


def _extract_plot_calls(source: str, method: str) -> list[dict]:
    """Extract .boxplot() / .violinplot() call info from cell source via AST.

    Returns list of dicts with keys:
        start_line  – 1-indexed first line of the call
        end_line    – 1-indexed last line of the call
        data_var    – source text of the first positional argument
        result_var  – assignment target name (e.g. 'bp' from 'bp = ax.boxplot(...)')
        label       – string label if a label=... kwarg is present
    """
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    results: list[dict] = []
    # Build a map from Call node id to the enclosing Assign target
    assign_map: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                assign_map[id(node.value)] = target.id

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == method):
            continue
        if not node.args:
            continue

        data_var = ast.get_source_segment(source, node.args[0])
        if data_var is None:
            continue

        # Try to extract label keyword
        label = None
        for kw in node.keywords:
            if kw.arg == 'label' and isinstance(kw.value, ast.Constant):
                label = kw.value.value
                break

        result_var = assign_map.get(id(node))

        results.append({
            'start_line': node.lineno,
            'end_line': node.end_lineno or node.lineno,
            'data_var': data_var,
            'result_var': result_var,
            'label': label,
        })
    return results


def _find_dependent_lines(source: str, var_name: str) -> set[int]:
    """Find all lines that reference *var_name* (e.g. bp['boxes']).

    Returns a set of 1-indexed line numbers.
    """
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    dep_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == var_name:
            # Skip the assignment line itself (handled by caller)
            dep_lines.add(node.lineno)
            # Also grab end_lineno for multiline statements
            parent = _find_stmt_parent(tree, node)
            if parent is not None:
                for ln in range(parent.lineno,
                                (parent.end_lineno or parent.lineno) + 1):
                    dep_lines.add(ln)
    return dep_lines


def _find_stmt_parent(tree, target_node):
    """Find the statement-level parent of a node in the AST."""
    import ast

    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            if child is target_node:
                # Return the statement-level node
                if isinstance(node, ast.Module):
                    return target_node
                return node
            # Check deeper: the target may be nested
            if _node_contains(child, target_node):
                if isinstance(child, ast.stmt):
                    return child
    return None


def _node_contains(parent, target):
    """Check if target is anywhere inside parent's subtree."""
    import ast
    for node in ast.walk(parent):
        if node is target:
            return True
    return False


def _unique_path(path: Path) -> Path:
    """If path exists, append (1), (2), ... until it doesn't."""
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    i = 1
    while True:
        candidate = parent / f"{stem}({i}){suffix}"
        if not candidate.exists():
            return candidate
        i += 1


class PlotBuildSession:
    """Orchestrates the interactive editing UI for a matplotlib Figure."""

    SIDEBAR_WIDTH = "380px"

    def __init__(self, fig: Figure, cell_source: str | None = None):
        self._fig = fig
        self._cell_source = cell_source
        self._canvas = CanvasManager(fig)
        self._stack = CommandStack(on_change=self._on_stack_change)

        # Introspect
        introspector = FigureIntrospector(fig)
        self._groups = introspector.introspect()

        # Toolbar buttons
        _tb = widgets.Layout(height="30px")
        self._undo_btn = widgets.Button(description="Undo", disabled=True,
                                         icon="undo",
                                         layout=widgets.Layout(width="80px", height="30px"))
        self._undo_btn.style.button_color = "#f5f5f5"
        self._redo_btn = widgets.Button(description="Redo", disabled=True,
                                         icon="repeat",
                                         layout=widgets.Layout(width="80px", height="30px"))
        self._redo_btn.style.button_color = "#f5f5f5"
        self._save_btn = widgets.Button(description="Save",
                                         icon="download",
                                         layout=widgets.Layout(width="80px", height="30px"))
        self._save_btn.style.button_color = "#e8eaf6"
        self._apply_btn = widgets.Button(description="Apply",
                                          icon="check",
                                          layout=widgets.Layout(width="80px", height="30px"),
                                          tooltip="Apply changes, generate code, and close")
        self._apply_btn.style.button_color = "#c8e6c9"
        self._close_btn = widgets.Button(description="Close",
                                          icon="times",
                                          layout=widgets.Layout(width="80px", height="30px"),
                                          tooltip="Revert all changes and close")
        self._close_btn.style.button_color = "#ffcdd2"

        # Save dialog (hidden until Save is clicked)
        self._save_dialog = self._build_save_dialog()

        self._undo_btn.on_click(self._on_undo)
        self._redo_btn.on_click(self._on_redo)
        self._save_btn.on_click(self._on_save_click)
        self._apply_btn.on_click(self._close_apply)
        self._close_btn.on_click(self._close_discard_direct)

        # Code output area that persists after the editor closes
        self._applied_code = widgets.Textarea(
            value="",
            layout=widgets.Layout(width="100%", height="300px"),
        )
        self._applied_status = widgets.HTML("")
        # Hidden Output widget for executing JavaScript (cell insertion)
        self._js_output = widgets.Output(
            layout=widgets.Layout(height="0px", overflow="hidden"))
        # Copy + Clear buttons (grouped together)
        self._copy_btn = widgets.Button(
            description="Copy", icon="clipboard",
            button_style="info", layout=widgets.Layout(width="75px"))
        self._copy_btn.on_click(self._copy_code_to_clipboard)
        self._clear_btn = widgets.Button(
            description="Clear", icon="times",
            button_style="", layout=widgets.Layout(width="75px"))
        self._clear_btn.on_click(lambda _: self._applied_box.close())
        self._applied_box = widgets.VBox(
            [widgets.HBox([self._applied_status,
                           widgets.HBox([self._copy_btn, self._clear_btn],
                                        layout=widgets.Layout(gap="4px"))],
                          layout=widgets.Layout(justify_content="space-between",
                                                align_items="center")),
             self._applied_code, self._js_output],
            layout=widgets.Layout(display="none"),
        )

        self._container = self._assemble()
        # Outer wrapper: editor + persistent code output
        self._outer = widgets.VBox([self._container, self._applied_box])

    def display(self) -> None:
        display(self._outer)

    # ------------------------------------------------------------------
    # Save dialog
    # ------------------------------------------------------------------

    def _build_save_dialog(self) -> widgets.VBox:
        self._save_name = widgets.Text(value="figure", description="Filename:",
                                        placeholder="figure",
                                        style={"description_width": "70px"})
        self._save_fmt = widgets.Dropdown(
            options=["pdf", "png", "svg", "jpg"],
            value="pdf",
            description="Format:",
            style={"description_width": "70px"},
        )
        self._save_dpi = widgets.IntText(value=150, description="DPI:",
                                          style={"description_width": "70px"},
                                          layout=widgets.Layout(width="160px"))
        self._save_dir = widgets.Text(value=os.getcwd(), description="Directory:",
                                       style={"description_width": "70px"},
                                       layout=widgets.Layout(width="100%"))
        self._save_confirm = widgets.Button(description="Save File",
                                             button_style="success",
                                             layout=widgets.Layout(width="100px"))
        self._save_cancel = widgets.Button(description="Cancel",
                                            layout=widgets.Layout(width="80px"))
        self._save_status = widgets.HTML(value="")

        self._save_confirm.on_click(self._do_save)
        self._save_cancel.on_click(lambda _: self._hide_save_dialog())

        dialog = widgets.VBox([
            widgets.HTML("<b>Save Figure</b>"),
            self._save_name,
            self._save_fmt,
            self._save_dpi,
            self._save_dir,
            widgets.HBox([self._save_confirm, self._save_cancel]),
            self._save_status,
        ], layout=widgets.Layout(
            border="1px solid #ccc",
            padding="8px",
            margin="4px 0",
            display="none",  # hidden initially
        ))
        return dialog

    def _show_save_dialog(self) -> None:
        self._save_dialog.layout.display = ""
        self._save_status.value = ""

    def _hide_save_dialog(self) -> None:
        self._save_dialog.layout.display = "none"

    def _do_save(self, _btn: Any) -> None:
        name = self._save_name.value.strip() or "figure"
        fmt = self._save_fmt.value
        dpi = self._save_dpi.value
        directory = Path(self._save_dir.value.strip() or os.getcwd())

        if not directory.is_dir():
            self._save_status.value = (
                "<span style='color:red'>Directory does not exist.</span>")
            return

        path = _unique_path(directory / f"{name}.{fmt}")
        try:
            # Collect extra artists (outside legends) for tight bbox
            extra = []
            for ax in self._fig.get_axes():
                leg = ax.get_legend()
                if leg is not None:
                    extra.append(leg)
            try:
                self._fig.tight_layout()
            except Exception:
                pass
            # Reposition marginal histograms after layout change
            for mgr in getattr(self._fig, '_matplotly_marginal_managers', []):
                mgr._rebuild()
            self._fig.savefig(str(path), dpi=dpi, bbox_inches="tight",
                              bbox_extra_artists=extra or None,
                              pad_inches=0.2,
                              facecolor=self._fig.get_facecolor())
            self._save_status.value = (
                f"<span style='color:green'>Saved to <code>{path}</code></span>")
        except Exception as e:
            self._save_status.value = (
                f"<span style='color:red'>Error: {e}</span>")

    # ------------------------------------------------------------------
    # Real data injection for distribution panels
    # ------------------------------------------------------------------

    def _inject_real_dist_data(self, dist_panels):
        """Replace reconstructed ~100-point data with real user data.

        Parses the cell source to find boxplot()/violinplot() data variable
        names, resolves them from the IPython user namespace, and swaps in the
        actual arrays so jitter/violin rendering uses full-fidelity data.
        """
        import numpy as np

        try:
            from IPython import get_ipython
            ip = get_ipython()
            if ip is None:
                return
            user_ns = ip.user_ns
        except Exception:
            return

        bp_calls = _extract_plot_calls(self._cell_source, 'boxplot')
        vp_calls = _extract_plot_calls(self._cell_source, 'violinplot')
        calls = bp_calls + vp_calls
        if not calls:
            return

        # Resolve each call's data variable from user namespace
        resolved = []
        for call in calls:
            try:
                val = eval(call['data_var'], user_ns)  # noqa: S307
            except Exception:
                resolved.append(None)
                continue
            # Normalise to list-of-arrays
            if isinstance(val, np.ndarray):
                if val.ndim == 2:
                    val = [val[:, i] for i in range(val.shape[1])]
                elif val.ndim == 1:
                    val = [val]
            if isinstance(val, (list, tuple)):
                try:
                    arrays = [np.asarray(d, dtype=float) for d in val]
                    resolved.append(arrays)
                except Exception:
                    resolved.append(None)
            else:
                resolved.append(None)

        # Match resolved data to panels
        if len(resolved) == len(dist_panels):
            # One call per panel — direct mapping
            for data, panel in zip(resolved, dist_panels):
                if data is not None and len(data) == len(panel._raw_data):
                    panel._raw_data = data
        elif len(resolved) == 1 and resolved[0] is not None:
            # Single call, possibly split across multiple panels
            data = resolved[0]
            total_boxes = sum(len(p._raw_data) for p in dist_panels)
            if len(data) == total_boxes:
                idx = 0
                for panel in dist_panels:
                    n = len(panel._raw_data)
                    panel._raw_data = data[idx:idx + n]
                    idx += n

    # ------------------------------------------------------------------
    # Close actions
    # ------------------------------------------------------------------

    def _copy_code_to_clipboard(self, _btn: Any) -> None:
        """Copy the generated code to clipboard via JavaScript."""
        import json
        code = self._applied_code.value
        js = "(function(){navigator.clipboard.writeText(%s)})();" % json.dumps(code)
        with self._js_output:
            self._js_output.clear_output()
            from IPython.display import display as ipy_display, Javascript
            ipy_display(Javascript(js))

    def _close_apply(self, _btn: Any) -> None:
        """Close editor, replace original cell with generated code."""
        import json
        import re

        # --- Merged-histogram preprocessing: extract data variable names
        # from the original source and store on axes before code gen. -------
        hist_comment_lines: set[int] = set()  # 1-indexed line numbers
        if self._cell_source:
            all_axes = self._fig.get_axes()
            main_axes = [a for a in all_axes
                         if not getattr(a, '_matplotly_marginal', False)
                         and not hasattr(a, '_colorbar')]
            for ax in main_axes:
                if not getattr(ax, '_matplotly_hist_merged', False):
                    continue
                calls = _extract_hist_calls(self._cell_source)
                if calls:
                    infos = getattr(ax, '_matplotly_hist_info', [])
                    data_vars = _match_hist_data_vars(calls, infos)
                    ax._matplotly_hist_data_vars = data_vars
                    for c in calls:
                        for ln in range(c['start_line'],
                                        c['end_line'] + 1):
                            hist_comment_lines.add(ln)

            # --- Distribution preprocessing: extract boxplot/violinplot
            # data variable names from the original source. -------
            for ax in main_axes:
                dist_infos = getattr(ax, '_matplotly_dist_info', [])
                if not dist_infos:
                    continue
                bp_calls = _extract_plot_calls(self._cell_source, 'boxplot')
                vp_calls = _extract_plot_calls(
                    self._cell_source, 'violinplot')
                calls = bp_calls + vp_calls
                if calls:
                    data_vars = [c['data_var'] for c in calls]
                    ax._matplotly_dist_data_vars = data_vars
                    # Comment out call lines
                    for c in calls:
                        for ln in range(c['start_line'],
                                        c['end_line'] + 1):
                            hist_comment_lines.add(ln)
                    # Comment out result variable references
                    for c in calls:
                        if c.get('result_var'):
                            dep_lines = _find_dependent_lines(
                                self._cell_source, c['result_var'])
                            # Remove the assignment line itself (already
                            # covered by call lines)
                            dep_lines -= set(range(c['start_line'],
                                                   c['end_line'] + 1))
                            hist_comment_lines.update(dep_lines)

        modifications = generate_code(self._fig, self._stack)

        # Build combined code: original (minus matplotly call,
        # with merged-hist calls commented out) + modifications
        if self._cell_source:
            src_lines = self._cell_source.splitlines()
            cleaned = []
            for i, line in enumerate(src_lines, 1):  # 1-indexed
                stripped = line.strip()
                if re.match(r'^matplotly\s*\(', stripped):
                    continue
                if i in hist_comment_lines:
                    cleaned.append('# ' + line)
                else:
                    cleaned.append(line)
            original = "\n".join(cleaned).rstrip()
            combined = original + "\n\n" + modifications
        else:
            combined = modifications

        # Show fallback textarea (dismissible) + try replacing original cell
        self._applied_code.value = combined
        self._applied_status.value = (
            "<b>Generated Code</b> "
            "<small style='color:#666'>"
            "(replaced in cell above — also copied to clipboard)</small>")
        self._applied_box.layout.display = ""
        self._container.close()

        # A snippet from the original cell source to identify which cell to
        # replace.  We use the first 80 characters to keep the JS short.
        cell_match = ""
        if self._cell_source:
            cell_match = self._cell_source[:80]

        # JavaScript: replace the originating cell's source, then clipboard
        js_code = """
        (function() {
            var code = %s;
            var match = %s;
            var replaced = false;

            // Classic Jupyter Notebook (< 7)
            if (typeof Jupyter !== 'undefined' && Jupyter.notebook) {
                try {
                    var cells = Jupyter.notebook.get_cells();
                    for (var i = 0; i < cells.length; i++) {
                        if (cells[i].cell_type === 'code' &&
                            cells[i].get_text().indexOf(match) !== -1) {
                            cells[i].set_text(code);
                            replaced = true;
                            break;
                        }
                    }
                } catch(e) {}
            }

            // JupyterLab / Notebook 7+
            if (!replaced) {
                try {
                    var nb = window.jupyterapp &&
                             window.jupyterapp.shell.currentWidget;
                    if (nb && nb.content && nb.content.model) {
                        var sm = nb.content.model.sharedModel;
                        if (sm && sm.cells) {
                            for (var i = 0; i < sm.cells.length; i++) {
                                var c = sm.cells[i];
                                if (c.cell_type === 'code' &&
                                    c.getSource().indexOf(match) !== -1) {
                                    c.setSource(code);
                                    replaced = true;
                                    break;
                                }
                            }
                        }
                    }
                } catch(e) {}
            }

            // Always copy to clipboard as fallback
            if (navigator.clipboard) {
                navigator.clipboard.writeText(code).catch(function(){});
            }
        })();
        """ % (json.dumps(combined), json.dumps(cell_match))

        with self._js_output:
            self._js_output.clear_output()
            from IPython.display import display as ipy_display, Javascript
            ipy_display(Javascript(js_code))

    def _close_discard_direct(self, _btn: Any) -> None:
        """Undo all changes (revert to original figure) and close."""
        while self._stack.can_undo:
            self._stack.undo()
        self._canvas.redraw()
        self._outer.close()

    # ------------------------------------------------------------------
    # UI assembly
    # ------------------------------------------------------------------

    def _assemble(self) -> widgets.VBox:
        toolbar = widgets.HBox(
            [self._undo_btn, self._redo_btn,
             self._save_btn, self._apply_btn, self._close_btn],
            layout=widgets.Layout(padding="4px", gap="4px"),
        )

        sidebar_sections = self._build_sidebar_sections()
        sidebar = widgets.VBox(
            sidebar_sections,
            layout=widgets.Layout(
                width=self.SIDEBAR_WIDTH,
                min_width=self.SIDEBAR_WIDTH,
                overflow_y="auto",
                overflow_x="auto",
                max_height="700px",
                padding="4px",
            ),
        )

        canvas_box = widgets.VBox(
            [self._canvas.widget, self._save_dialog],
            layout=widgets.Layout(flex="1 1 auto"),
        )

        main = widgets.HBox(
            [sidebar, canvas_box],
            layout=widgets.Layout(width="100%"),
        )

        return widgets.VBox([toolbar, main])

    def _build_sidebar_sections(self) -> list[widgets.Widget]:
        sections: list[widgets.Widget] = []

        # Create all artist panels first, collecting typed panel lists
        main_axes = [a for a in self._fig.get_axes()
                     if not getattr(a, '_matplotly_marginal', False)
                     and not hasattr(a, '_colorbar')]
        multi_subplot = len(main_axes) > 1
        line_panels: list = []
        scatter_panels: list = []
        histogram_panels: list = []
        bar_panels: list = []
        dist_panels: list = []
        heatmap_panels: list = []
        errorbar_panels: list = []
        line_counter = 0
        scatter_counter = 0
        histogram_counter = 0
        bar_counter = 0
        dist_counter = 0
        heatmap_counter = 0
        errorbar_counter = 0

        subplot_panels: dict[tuple[int, int], list[widgets.Widget]] = {}
        for group in self._groups:
            panel = create_panel(group, self._stack, self._canvas)
            if panel is None:
                continue
            if group.plot_type == PlotType.LINE:
                line_counter += 1
                panel._plot_number = line_counter
                _ = panel.widget
                line_panels.append(panel)
            elif group.plot_type == PlotType.SCATTER:
                scatter_counter += 1
                panel._plot_number = scatter_counter
                _ = panel.widget
                scatter_panels.append(panel)
            elif group.plot_type == PlotType.HISTOGRAM:
                histogram_counter += 1
                panel._plot_number = histogram_counter
                _ = panel.widget
                panel._store_hist_info()
                histogram_panels.append(panel)
            elif group.plot_type in (PlotType.BAR, PlotType.GROUPED_BAR):
                bar_counter += 1
                panel._plot_number = bar_counter
                _ = panel.widget
                panel._store_bar_info()
                bar_panels.append(panel)
            elif group.plot_type in (PlotType.BOXPLOT, PlotType.VIOLIN):
                dist_counter += 1
                panel._plot_number = dist_counter
                _ = panel.widget
                panel._store_dist_info()
                dist_panels.append(panel)
            elif group.plot_type == PlotType.HEATMAP:
                heatmap_counter += 1
                panel._plot_number = heatmap_counter
                _ = panel.widget
                panel._store_heatmap_info()
                heatmap_panels.append(panel)
            elif group.plot_type == PlotType.ERRORBAR:
                errorbar_counter += 1
                panel._plot_number = errorbar_counter
                _ = panel.widget
                panel._store_errorbar_info()
                errorbar_panels.append(panel)
            key = group.subplot_index
            subplot_panels.setdefault(key, []).append(panel.widget)

        # Inject real data from user namespace into distribution panels
        if dist_panels and self._cell_source:
            self._inject_real_dist_data(dist_panels)

        # Build shared histogram controls — one per subplot
        if histogram_panels:
            hist_by_subplot: dict[tuple[int, int], list] = {}
            for p in histogram_panels:
                hist_by_subplot.setdefault(
                    p._group.subplot_index, []).append(p)
            for sp_key, sp_panels in hist_by_subplot.items():
                shared_hist = HistogramSharedPanel(sp_panels, self._canvas)
                shared_hist_widget = shared_hist.build()
                existing = subplot_panels.get(sp_key, [])
                first_hw = sp_panels[0].widget
                idx = 0
                for j, w in enumerate(existing):
                    if w is first_hw:
                        idx = j
                        break
                existing.insert(idx, shared_hist_widget)
                subplot_panels[sp_key] = existing

        # Build shared bar controls — one per subplot
        if bar_panels:
            from .panels._bar import BarSharedPanel
            bar_by_subplot: dict[tuple[int, int], list] = {}
            for p in bar_panels:
                bar_by_subplot.setdefault(
                    p._group.subplot_index, []).append(p)
            for sp_key, sp_panels in bar_by_subplot.items():
                shared_bar = BarSharedPanel(sp_panels, self._canvas)
                shared_bar_widget = shared_bar.build()
                existing = subplot_panels.get(sp_key, [])
                first_bw = sp_panels[0].widget
                idx = 0
                for j, w in enumerate(existing):
                    if w is first_bw:
                        idx = j
                        break
                existing.insert(idx, shared_bar_widget)
                subplot_panels[sp_key] = existing

        # Build shared distribution controls — one per subplot
        if dist_panels:
            from .panels._distribution import DistributionSharedPanel
            dist_by_subplot: dict[tuple[int, int], list] = {}
            for p in dist_panels:
                dist_by_subplot.setdefault(
                    p._group.subplot_index, []).append(p)
            for sp_key, sp_panels in dist_by_subplot.items():
                types = {p._group.plot_type for p in sp_panels}
                if PlotType.BOXPLOT in types and PlotType.VIOLIN in types:
                    initial_mode = "box+violin"
                elif PlotType.VIOLIN in types:
                    initial_mode = "violin"
                else:
                    initial_mode = "box"
                shared_dist = DistributionSharedPanel(
                    sp_panels, self._canvas, initial_mode=initial_mode)
                shared_dist_widget = shared_dist.build()
                existing = subplot_panels.get(sp_key, [])
                first_dw = sp_panels[0].widget
                idx = 0
                for j, w in enumerate(existing):
                    if w is first_dw:
                        idx = j
                        break
                existing.insert(idx, shared_dist_widget)
                subplot_panels[sp_key] = existing

        # Build shared heatmap colorbar controls — one per subplot
        if heatmap_panels:
            from .panels._heatmap import HeatmapSharedPanel
            heatmap_by_subplot: dict[tuple[int, int], list] = {}
            for p in heatmap_panels:
                heatmap_by_subplot.setdefault(
                    p._group.subplot_index, []).append(p)
            for sp_key, sp_panels in heatmap_by_subplot.items():
                shared_heatmap = HeatmapSharedPanel(sp_panels, self._canvas)
                shared_heatmap_widget = shared_heatmap.build()
                existing = subplot_panels.get(sp_key, [])
                first_hw = sp_panels[0].widget
                idx = 0
                for j, w in enumerate(existing):
                    if w is first_hw:
                        idx = j
                        break
                existing.insert(idx, shared_heatmap_widget)
                subplot_panels[sp_key] = existing

        # Build colormap panel (or None) to embed inside Global
        cmap_widget = None
        cmap_panel_ref = None
        all_color_panels = (line_panels + scatter_panels + histogram_panels
                            + bar_panels + dist_panels + errorbar_panels)
        if all_color_panels:
            cmap_panel_ref = ColormapPanel(self._groups, self._stack, self._canvas,
                                           line_panels=all_color_panels)
            cmap_widget = cmap_panel_ref.build()

        # Global panel (includes colormap if lines exist)
        global_panel = GlobalPanel(self._fig, self._stack, self._canvas,
                                    colormap_widget=cmap_widget,
                                    cmap_panel=cmap_panel_ref,
                                    multi_subplot=multi_subplot)
        global_acc = widgets.Accordion(children=[global_panel.build()])
        global_acc.set_title(0, "Global")
        global_acc.selected_index = None
        sections.append(global_acc)

        def _type_display(name: str) -> str:
            return "Plot" if name == "Line" else name

        if multi_subplot:
            # Build subplot_index → Axes mapping
            nrows, ncols = FigureIntrospector._grid_shape(main_axes)
            subplot_axes = {}
            for idx, ax in enumerate(main_axes):
                r, c = divmod(idx, max(ncols, 1))
                subplot_axes[(r, c)] = ax

            per_subplot_refs: dict[tuple[int, int], PerSubplotPanel] = {}

            for (r, c), panels in sorted(subplot_panels.items()):
                ax = subplot_axes.get((r, c))
                if ax is None:
                    continue

                # Create PerSubplotPanel for this axes
                psp = PerSubplotPanel(ax, self._fig, self._stack, self._canvas,
                                       cmap_panel=cmap_panel_ref)
                psp_widget = psp.build()
                per_subplot_refs[(r, c)] = psp

                # Wire label-change callbacks for this subplot's artist panels
                for panel in (line_panels + scatter_panels + histogram_panels
                              + bar_panels + dist_panels
                              + errorbar_panels):
                    if panel._group.subplot_index == (r, c):
                        panel._on_label_changed = psp._refresh_legend_labels

                # Assemble: PerSubplotPanel first, then artist panels
                all_widgets = [psp_widget] + panels
                types_in_subplot = set()
                for g in self._groups:
                    if g.subplot_index == (r, c):
                        types_in_subplot.add(_type_display(
                            g.plot_type.name.replace("_", " ").title()))
                type_str = ", ".join(sorted(types_in_subplot))
                title = f"Subplot ({r+1},{c+1}): {type_str}"
                acc = widgets.Accordion(children=[widgets.VBox(all_widgets)])
                acc.set_title(0, title)
                acc.selected_index = None
                sections.append(acc)

            # Store per-subplot refs on GlobalPanel for Preferred Defaults
            global_panel._per_subplot_panels = per_subplot_refs
        else:
            # Wire label-change callbacks to global legend
            for panel in (line_panels + scatter_panels + histogram_panels
                          + bar_panels + dist_panels
                          + errorbar_panels):
                panel._on_label_changed = global_panel._refresh_legend_labels

            all_panels = []
            for panels in subplot_panels.values():
                all_panels.extend(panels)
            if all_panels:
                types_present = set()
                for g in self._groups:
                    types_present.add(_type_display(
                        g.plot_type.name.replace("_", " ").title()))
                type_str = ", ".join(sorted(types_present))
                acc = widgets.Accordion(children=[widgets.VBox(all_panels)])
                acc.set_title(0, type_str)
                acc.selected_index = None
                sections.append(acc)

        # Marginal histograms (only when scatter collections exist)
        scatter_colls_by_ax: dict[int, tuple] = {}
        for group in self._groups:
            if group.plot_type == PlotType.SCATTER:
                ax = group.axes
                ax_id = id(ax)
                if ax_id not in scatter_colls_by_ax:
                    scatter_colls_by_ax[ax_id] = (ax, [])
                scatter_colls_by_ax[ax_id][1].append(group.artists[0])

        for ax_id, (ax, colls) in scatter_colls_by_ax.items():
            mgr = MarginalHistogramManager(
                self._fig, ax, colls, self._stack, self._canvas)
            marginal_w = mgr.build_widget()
            marginal_acc = widgets.Accordion(children=[marginal_w])
            marginal_acc.set_title(0, "Marginal Histograms")
            marginal_acc.selected_index = None
            sections.append(marginal_acc)
            # Wire color sync from scatter panels to marginal manager
            for panel in scatter_panels:
                coll = panel._group.artists[0]
                if id(coll.axes) == ax_id:
                    panel._marginals = mgr

        # Legend (separate section — only for single-subplot)
        if not multi_subplot:
            legend_acc = widgets.Accordion(children=[global_panel._legend_widget])
            legend_acc.set_title(0, "Legend")
            legend_acc.selected_index = None
            sections.append(legend_acc)

        # Profiles
        all_artist_panels = (line_panels + scatter_panels + bar_panels
                             + histogram_panels + dist_panels
                             + heatmap_panels + errorbar_panels)
        profiles_panel = create_profiles_panel(
            global_panel, self._canvas, artist_panels=all_artist_panels)
        profiles_acc = widgets.Accordion(children=[profiles_panel])
        profiles_acc.set_title(0, "Profiles")
        profiles_acc.selected_index = None
        sections.append(profiles_acc)

        return sections

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_stack_change(self) -> None:
        self._undo_btn.disabled = not self._stack.can_undo
        self._redo_btn.disabled = not self._stack.can_redo

    def _on_undo(self, _btn: Any) -> None:
        self._stack.undo()
        self._canvas.redraw()

    def _on_redo(self, _btn: Any) -> None:
        self._stack.redo()
        self._canvas.redraw()

    def _on_save_click(self, _btn: Any) -> None:
        # Toggle save dialog visibility
        if self._save_dialog.layout.display == "none":
            self._show_save_dialog()
        else:
            self._hide_save_dialog()

