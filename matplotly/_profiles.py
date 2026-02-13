"""Save / load reusable plot style profiles."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import ipywidgets as widgets
from matplotlib.figure import Figure

PROFILES_DIR = Path.home() / ".matplotly" / "profiles"


def _ensure_dir() -> None:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)


def _list_profiles() -> list[str]:
    """Return sorted list of saved profile names (without .json)."""
    _ensure_dir()
    return sorted(p.stem for p in PROFILES_DIR.glob("*.json"))


def _read_profile(name: str) -> dict:
    path = PROFILES_DIR / f"{name}.json"
    return json.loads(path.read_text())


def _save_profile(name: str, data: dict) -> Path:
    _ensure_dir()
    path = PROFILES_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=2))
    return path


def _delete_profile(name: str) -> None:
    path = PROFILES_DIR / f"{name}.json"
    if path.exists():
        path.unlink()


def snapshot_from_global(global_panel) -> dict:
    """Read current style values from GlobalPanel widgets."""
    gp = global_panel
    data: dict[str, Any] = {}
    # Font
    data["font_family"] = gp._font_dd.value
    data["title_size"] = round(gp._title_size_sl.value, 1)
    data["label_size"] = round(gp._label_size_sl.value, 1)
    data["tick_size"] = round(gp._tick_size_sl.value, 1)
    # Title / label padding
    if hasattr(gp, '_title_pad_sl'):
        data["title_pad"] = round(gp._title_pad_sl.value, 1)
    if hasattr(gp, '_xlabel_pad_sl'):
        data["xlabel_pad"] = round(gp._xlabel_pad_sl.value, 1)
    if hasattr(gp, '_ylabel_pad_sl'):
        data["ylabel_pad"] = round(gp._ylabel_pad_sl.value, 1)
    # Spines (may not exist in multi-subplot mode)
    if hasattr(gp, '_spine_top_cb'):
        data["spine_top"] = gp._spine_top_cb.value
        data["spine_right"] = gp._spine_right_cb.value
        data["spine_bottom"] = gp._spine_bottom_cb.value
        data["spine_left"] = gp._spine_left_cb.value
    if hasattr(gp, '_spine_width_sl'):
        data["spine_width"] = round(gp._spine_width_sl.value, 1)
    # Ticks (may not exist in multi-subplot mode)
    if hasattr(gp, '_tick_dir_dd'):
        data["tick_direction"] = gp._tick_dir_dd.value
        data["tick_length"] = round(gp._tick_len_sl.value, 1)
        data["tick_width"] = round(gp._tick_width_sl.value, 1)
    if hasattr(gp, '_x_step'):
        data["x_tick_step"] = round(gp._x_step.value, 4)
    if hasattr(gp, '_y_step'):
        data["y_tick_step"] = round(gp._y_step.value, 4)
    # Axis scale
    if hasattr(gp, '_xscale_dd'):
        data["x_scale"] = gp._xscale_dd.value
    if hasattr(gp, '_yscale_dd'):
        data["y_scale"] = gp._yscale_dd.value
    # Grid (may not exist in multi-subplot mode)
    if hasattr(gp, '_grid_toggle'):
        data["grid_on"] = gp._grid_toggle.value
        data["grid_alpha"] = round(gp._grid_alpha_sl.value, 2)
        data["grid_width"] = round(gp._grid_width_sl.value, 1)
        data["grid_style"] = gp._grid_style_dd.value
    # Legend (may not exist in multi-subplot mode)
    if hasattr(gp, '_legend_toggle'):
        data["legend_show"] = gp._legend_toggle.value
        data["legend_frame"] = gp._frame_toggle.value
        data["legend_fontsize"] = round(gp._legend_fontsize_sl.value, 1)
    if hasattr(gp, '_legend_pos_dd'):
        data["legend_position"] = gp._legend_pos_dd.value
    if hasattr(gp, '_legend_ncol'):
        data["legend_columns"] = gp._legend_ncol.value
    # Colormap
    if gp._cmap_panel is not None:
        data["colormap"] = gp._cmap_panel._selected
    return data


def apply_profile(data: dict, global_panel, canvas) -> None:
    """Apply a profile dict by setting GlobalPanel widget values.

    Suppresses intermediate redraws and does one force_redraw at the end.
    """
    gp = global_panel
    orig_redraw = canvas.redraw
    canvas.redraw = lambda: None
    try:
        # Helper: get value if key exists and value is not None
        def _v(key):
            val = data.get(key)
            return val  # returns None if missing or explicitly null

        # Font
        if _v("font_family") is not None:
            # Add to dropdown if not present
            opts = list(gp._font_dd.options)
            if data["font_family"] not in opts:
                opts.insert(0, data["font_family"])
                gp._font_dd.options = opts
            gp._font_dd.value = data["font_family"]
        if _v("title_size") is not None:
            gp._title_size_sl.value = data["title_size"]
        if _v("label_size") is not None:
            gp._label_size_sl.value = data["label_size"]
        if _v("tick_size") is not None:
            gp._tick_size_sl.value = data["tick_size"]
        # Title / label padding
        if _v("title_pad") is not None and hasattr(gp, '_title_pad_sl'):
            gp._title_pad_sl.value = data["title_pad"]
        if _v("xlabel_pad") is not None and hasattr(gp, '_xlabel_pad_sl'):
            gp._xlabel_pad_sl.value = data["xlabel_pad"]
        if _v("ylabel_pad") is not None and hasattr(gp, '_ylabel_pad_sl'):
            gp._ylabel_pad_sl.value = data["ylabel_pad"]
        # Title / label bold (applied directly to figure)
        if _v("title_bold") is not None:
            fig = gp._fig
            w = "bold" if data["title_bold"] else "normal"
            for ax in fig.get_axes():
                ax.title.set_fontweight(w)
        if _v("label_bold") is not None:
            fig = gp._fig
            w = "bold" if data["label_bold"] else "normal"
            for ax in fig.get_axes():
                ax.xaxis.label.set_fontweight(w)
                ax.yaxis.label.set_fontweight(w)
        # Spines (may not exist in multi-subplot mode)
        if _v("spine_top") is not None and hasattr(gp, '_spine_top_cb'):
            gp._spine_top_cb.value = data["spine_top"]
        if _v("spine_right") is not None and hasattr(gp, '_spine_right_cb'):
            gp._spine_right_cb.value = data["spine_right"]
        if _v("spine_bottom") is not None and hasattr(gp, '_spine_bottom_cb'):
            gp._spine_bottom_cb.value = data["spine_bottom"]
        if _v("spine_left") is not None and hasattr(gp, '_spine_left_cb'):
            gp._spine_left_cb.value = data["spine_left"]
        if _v("spine_width") is not None and hasattr(gp, '_spine_width_sl'):
            gp._spine_width_sl.value = data["spine_width"]
        # Ticks (may not exist in multi-subplot mode)
        if _v("tick_direction") is not None and hasattr(gp, '_tick_dir_dd'):
            gp._tick_dir_dd.value = data["tick_direction"]
        if _v("tick_length") is not None and hasattr(gp, '_tick_len_sl'):
            gp._tick_len_sl.value = data["tick_length"]
        if _v("tick_width") is not None and hasattr(gp, '_tick_width_sl'):
            gp._tick_width_sl.value = data["tick_width"]
        # Tick spacing (0 = auto)
        if _v("x_tick_step") is not None and hasattr(gp, '_x_step'):
            gp._x_step.value = data["x_tick_step"]
        if _v("y_tick_step") is not None and hasattr(gp, '_y_step'):
            gp._y_step.value = data["y_tick_step"]
        # Axis scale
        if _v("x_scale") is not None and hasattr(gp, '_xscale_dd'):
            if data["x_scale"] in ("linear", "log", "symlog"):
                gp._xscale_dd.value = data["x_scale"]
        if _v("y_scale") is not None and hasattr(gp, '_yscale_dd'):
            if data["y_scale"] in ("linear", "log", "symlog"):
                gp._yscale_dd.value = data["y_scale"]
        # Grid (may not exist in multi-subplot mode)
        if _v("grid_on") is not None and hasattr(gp, '_grid_toggle'):
            gp._grid_toggle.value = data["grid_on"]
        if _v("grid_alpha") is not None and hasattr(gp, '_grid_alpha_sl'):
            gp._grid_alpha_sl.value = data["grid_alpha"]
        if _v("grid_width") is not None and hasattr(gp, '_grid_width_sl'):
            gp._grid_width_sl.value = data["grid_width"]
        if _v("grid_style") is not None and hasattr(gp, '_grid_style_dd'):
            gp._grid_style_dd.value = data["grid_style"]
        # Legend (may not exist in multi-subplot mode)
        if _v("legend_show") is not None and hasattr(gp, '_legend_toggle'):
            gp._legend_toggle.value = data["legend_show"]
        if _v("legend_frame") is not None and hasattr(gp, '_frame_toggle'):
            gp._frame_toggle.value = data["legend_frame"]
        if _v("legend_fontsize") is not None and hasattr(gp, '_legend_fontsize_sl'):
            gp._legend_fontsize_sl.value = data["legend_fontsize"]
        # Legend position and columns
        if _v("legend_position") is not None and hasattr(gp, '_legend_pos_dd'):
            pos = data["legend_position"]
            valid = [v for _, v in gp._legend_pos_dd.options]
            if pos in valid:
                gp._legend_pos_dd.value = pos
        if _v("legend_columns") is not None and hasattr(gp, '_legend_ncol'):
            ncol = data["legend_columns"]
            if isinstance(ncol, int) and 1 <= ncol <= 6:
                gp._legend_ncol.value = ncol
        # Colormap
        if _v("colormap") is not None and gp._cmap_panel is not None:
            gp._cmap_panel.apply(data["colormap"])
        # Background color
        if _v("background_color") is not None:
            bg = data["background_color"]
            fig = gp._fig
            fig.set_facecolor(bg)
            for ax in fig.get_axes():
                ax.set_facecolor(bg)
    finally:
        canvas.redraw = orig_redraw
    canvas.force_redraw()


def create_profiles_panel(global_panel, canvas,
                          artist_panels=None) -> widgets.Widget:
    """Build the Profiles accordion section.

    Contains AI Style Import and Saved Profiles (load/save/delete).
    """
    from ._ai_extract import create_ai_import_section

    return create_ai_import_section(
        global_panel, canvas, artist_panels=artist_panels)
