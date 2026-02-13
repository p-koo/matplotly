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
        # Font
        if "font_family" in data:
            # Add to dropdown if not present
            opts = list(gp._font_dd.options)
            if data["font_family"] not in opts:
                opts.insert(0, data["font_family"])
                gp._font_dd.options = opts
            gp._font_dd.value = data["font_family"]
        if "title_size" in data:
            gp._title_size_sl.value = data["title_size"]
        if "label_size" in data:
            gp._label_size_sl.value = data["label_size"]
        if "tick_size" in data:
            gp._tick_size_sl.value = data["tick_size"]
        # Spines (may not exist in multi-subplot mode)
        if "spine_top" in data and hasattr(gp, '_spine_top_cb'):
            gp._spine_top_cb.value = data["spine_top"]
        if "spine_right" in data and hasattr(gp, '_spine_right_cb'):
            gp._spine_right_cb.value = data["spine_right"]
        if "spine_bottom" in data and hasattr(gp, '_spine_bottom_cb'):
            gp._spine_bottom_cb.value = data["spine_bottom"]
        if "spine_left" in data and hasattr(gp, '_spine_left_cb'):
            gp._spine_left_cb.value = data["spine_left"]
        if "spine_width" in data and hasattr(gp, '_spine_width_sl'):
            gp._spine_width_sl.value = data["spine_width"]
        # Ticks (may not exist in multi-subplot mode)
        if "tick_direction" in data and hasattr(gp, '_tick_dir_dd'):
            gp._tick_dir_dd.value = data["tick_direction"]
        if "tick_length" in data and hasattr(gp, '_tick_len_sl'):
            gp._tick_len_sl.value = data["tick_length"]
        if "tick_width" in data and hasattr(gp, '_tick_width_sl'):
            gp._tick_width_sl.value = data["tick_width"]
        # Grid (may not exist in multi-subplot mode)
        if "grid_on" in data and hasattr(gp, '_grid_toggle'):
            gp._grid_toggle.value = data["grid_on"]
        if "grid_alpha" in data and hasattr(gp, '_grid_alpha_sl'):
            gp._grid_alpha_sl.value = data["grid_alpha"]
        if "grid_width" in data and hasattr(gp, '_grid_width_sl'):
            gp._grid_width_sl.value = data["grid_width"]
        if "grid_style" in data and hasattr(gp, '_grid_style_dd'):
            gp._grid_style_dd.value = data["grid_style"]
        # Legend (may not exist in multi-subplot mode)
        if "legend_show" in data and hasattr(gp, '_legend_toggle'):
            gp._legend_toggle.value = data["legend_show"]
        if "legend_frame" in data and hasattr(gp, '_frame_toggle'):
            gp._frame_toggle.value = data["legend_frame"]
        if "legend_fontsize" in data and hasattr(gp, '_legend_fontsize_sl'):
            gp._legend_fontsize_sl.value = data["legend_fontsize"]
        # Colormap
        if "colormap" in data and gp._cmap_panel is not None:
            gp._cmap_panel.apply(data["colormap"])
        # Background color
        if "background_color" in data:
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
