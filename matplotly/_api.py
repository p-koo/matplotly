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
from .panels._line import ColormapPanel


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
        self._export_btn = widgets.Button(description="Export",
                                           icon="code",
                                           layout=widgets.Layout(width="90px", height="30px"))
        self._export_btn.style.button_color = "#e3f2fd"
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

        # Code output area (shown below canvas when Export is clicked)
        self._code_output = widgets.Textarea(
            value="", placeholder="Click 'Export Code' to generate...",
            layout=widgets.Layout(width="100%", height="200px"),
        )
        self._code_box = widgets.VBox(
            [widgets.HTML("<b>Generated Code</b>"), self._code_output],
            layout=widgets.Layout(display="none"),  # hidden until Export
        )

        # Save dialog (hidden until Save is clicked)
        self._save_dialog = self._build_save_dialog()

        self._undo_btn.on_click(self._on_undo)
        self._redo_btn.on_click(self._on_redo)
        self._export_btn.on_click(self._on_export)
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

        modifications = generate_code(self._fig, self._stack)

        # Build combined code: original (minus matplotly call) + modifications
        if self._cell_source:
            lines = self._cell_source.splitlines()
            cleaned = []
            for line in lines:
                stripped = line.strip()
                if re.match(r'^matplotly\s*\(', stripped):
                    continue
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
            [self._undo_btn, self._redo_btn, self._export_btn,
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
            [self._canvas.widget, self._save_dialog, self._code_box],
            layout=widgets.Layout(flex="1 1 auto"),
        )

        main = widgets.HBox(
            [sidebar, canvas_box],
            layout=widgets.Layout(width="100%"),
        )

        return widgets.VBox([toolbar, main])

    def _build_sidebar_sections(self) -> list[widgets.Widget]:
        sections: list[widgets.Widget] = []

        # Create all artist panels first, collecting line panels
        axes_list = self._fig.get_axes()
        multi_subplot = len(axes_list) > 1
        line_panels: list = []
        line_counter = 0

        subplot_panels: dict[tuple[int, int], list[widgets.Widget]] = {}
        for group in self._groups:
            panel = create_panel(group, self._stack, self._canvas)
            if panel is None:
                continue
            if group.plot_type == PlotType.LINE:
                line_counter += 1
                panel._plot_number = line_counter
                # Trigger build so _update_color/_update_palette are set
                _ = panel.widget
                line_panels.append(panel)
            key = group.subplot_index
            subplot_panels.setdefault(key, []).append(panel.widget)

        # Build colormap panel (or None) to embed inside Global
        cmap_widget = None
        cmap_panel_ref = None
        if line_panels:
            cmap_panel_ref = ColormapPanel(self._groups, self._stack, self._canvas,
                                           line_panels=line_panels)
            cmap_widget = cmap_panel_ref.build()

        # Global panel (includes colormap if lines exist)
        global_panel = GlobalPanel(self._fig, self._stack, self._canvas,
                                    colormap_widget=cmap_widget,
                                    cmap_panel=cmap_panel_ref)
        global_acc = widgets.Accordion(children=[global_panel.build()])
        global_acc.set_title(0, "Global")
        global_acc.selected_index = None
        sections.append(global_acc)

        def _type_display(name: str) -> str:
            return "Plot" if name == "Line" else name

        if multi_subplot:
            for (r, c), panels in sorted(subplot_panels.items()):
                types_in_subplot = set()
                for g in self._groups:
                    if g.subplot_index == (r, c):
                        types_in_subplot.add(_type_display(
                            g.plot_type.name.replace("_", " ").title()))
                type_str = ", ".join(sorted(types_in_subplot))
                title = f"Subplot ({r+1},{c+1}): {type_str}"
                acc = widgets.Accordion(children=[widgets.VBox(panels)])
                acc.set_title(0, title)
                acc.selected_index = None
                sections.append(acc)
        else:
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

        # Legend (separate section below plot panels)
        legend_acc = widgets.Accordion(children=[global_panel._legend_widget])
        legend_acc.set_title(0, "Legend")
        legend_acc.selected_index = None
        sections.append(legend_acc)

        # Profiles
        profiles_panel = create_profiles_panel(global_panel, self._canvas)
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
        # Auto-update code if the code box is visible
        if self._code_box.layout.display != "none":
            self._code_output.value = generate_code(self._fig, self._stack)

    def _on_undo(self, _btn: Any) -> None:
        self._stack.undo()
        self._canvas.redraw()

    def _on_redo(self, _btn: Any) -> None:
        self._stack.redo()
        self._canvas.redraw()

    def _on_export(self, _btn: Any) -> None:
        code = generate_code(self._fig, self._stack)
        self._code_output.value = code
        # Toggle visibility
        if self._code_box.layout.display == "none":
            self._code_box.layout.display = ""
        else:
            self._code_box.layout.display = "none"

    def _on_save_click(self, _btn: Any) -> None:
        # Toggle save dialog visibility
        if self._save_dialog.layout.display == "none":
            self._show_save_dialog()
        else:
            self._hide_save_dialog()

