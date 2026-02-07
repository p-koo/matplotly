"""PDF/image upload → pipeline subprocess → apply extracted style to figure."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import ipywidgets as widgets
from matplotlib.colors import to_hex
from matplotlib.figure import Figure

from ._commands import BatchCommand, Command, CommandStack
from ._renderer import CanvasManager

PIPELINE_DIR = Path(__file__).resolve().parent.parent.parent / "FigureStyleExtractor" / "Pipeline"
PIPELINE_MAIN = PIPELINE_DIR / "pipeline_main.py"


def _run_pipeline(file_path: str) -> dict:
    """Run the extraction pipeline and return the FSP dict."""
    input_json = json.dumps({"input_path": file_path})
    result = subprocess.run(
        ["python3", str(PIPELINE_MAIN)],
        input=input_json,
        capture_output=True,
        text=True,
        cwd=str(PIPELINE_DIR),
        timeout=120,
    )

    fsp_dict = None
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("type") == "result" and "fsp" in msg:
            fsp_dict = msg["fsp"]

    if fsp_dict is None:
        stderr_tail = result.stderr[-500:] if result.stderr else "(no stderr)"
        raise RuntimeError(
            f"Pipeline did not produce a result.\n"
            f"Exit code: {result.returncode}\nstderr: {stderr_tail}"
        )
    return fsp_dict


def _apply_fsp_to_figure(fig: Figure, fsp: dict, stack: CommandStack) -> None:
    """Apply FSP extracted style to the figure as a single BatchCommand."""
    cmds: list[Command] = []

    typography = fsp.get("typography", {})
    colors = fsp.get("colors", {})
    axis_style = fsp.get("axis_style", {})

    # Typography
    font_family = typography.get("font_family")
    if font_family:
        for ax in fig.get_axes():
            for text in [ax.title, ax.xaxis.label, ax.yaxis.label]:
                cmds.append(Command(text, "fontfamily",
                                    text.get_fontfamily(), font_family))
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                cmds.append(Command(lbl, "fontfamily",
                                    lbl.get_fontfamily(), font_family))

    title_size = typography.get("title_size")
    if title_size:
        for ax in fig.get_axes():
            cmds.append(Command(ax.title, "fontsize",
                                ax.title.get_fontsize(), title_size))

    label_size = typography.get("label_size")
    if label_size:
        for ax in fig.get_axes():
            for lbl in [ax.xaxis.label, ax.yaxis.label]:
                cmds.append(Command(lbl, "fontsize",
                                    lbl.get_fontsize(), label_size))

    # Colors — apply to lines and patches in order
    color_list = colors.get("palette", [])
    if color_list:
        for ax in fig.get_axes():
            for i, line in enumerate(ax.lines):
                if i < len(color_list):
                    try:
                        old = to_hex(line.get_color())
                    except Exception:
                        old = "#000000"
                    cmds.append(Command(line, "color", old, color_list[i]))

    # Axis style
    grid_info = axis_style.get("grid", {})
    spine_info = axis_style.get("spines", {})

    if spine_info:
        for ax in fig.get_axes():
            for name in ("top", "right", "bottom", "left"):
                vis = spine_info.get(name, True)
                sp = ax.spines[name]
                cmds.append(Command(sp, "visible", sp.get_visible(), vis))

    bg = colors.get("background")
    if bg:
        fig.set_facecolor(bg)
        for ax in fig.get_axes():
            ax.set_facecolor(bg)

    if cmds:
        stack.execute(BatchCommand(cmds, "Import style from file"))


def create_import_panel(fig: Figure, stack: CommandStack,
                        canvas: CanvasManager) -> widgets.Widget:
    """Build the style import accordion section with file upload."""
    upload = widgets.FileUpload(
        accept=".pdf,.png,.jpg,.jpeg",
        multiple=False,
        description="Upload figure",
    )
    status = widgets.HTML("<i>Upload a PDF or image to extract and apply its style.</i>")

    def _on_upload(change):
        uploaded = change["new"]
        if not uploaded:
            return
        # ipywidgets FileUpload returns a tuple of dicts
        file_info = uploaded[0] if isinstance(uploaded, (list, tuple)) else uploaded
        content = file_info.get("content", b"") if isinstance(file_info, dict) else b""
        name = file_info.get("name", "upload.pdf") if isinstance(file_info, dict) else "upload.pdf"

        suffix = Path(name).suffix or ".pdf"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(content)
        tmp.close()

        status.value = "<i>Extracting style...</i>"
        try:
            fsp = _run_pipeline(tmp.name)
            _apply_fsp_to_figure(fig, fsp, stack)
            canvas.redraw()
            status.value = f"<b>Style applied from {name}</b>"
        except Exception as e:
            status.value = f"<span style='color:red'>Error: {e}</span>"
        finally:
            os.unlink(tmp.name)

    upload.observe(_on_upload, names="value")
    return widgets.VBox([upload, status])
