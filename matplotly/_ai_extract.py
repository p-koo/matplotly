"""AI-powered style extraction from plot images.

Supports Anthropic (Claude) and OpenAI (GPT-4o) as vision LLM providers.
API keys are persisted in ~/.matplotly/config.json.

Two-pass extraction: first pass extracts style, second pass compares the
modified figure against the reference and corrects discrepancies.
"""
from __future__ import annotations

import base64
import json
import io
from pathlib import Path
from typing import Any

CONFIG_DIR = Path.home() / ".matplotly"
CONFIG_FILE = CONFIG_DIR / "config.json"

# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from disk or return empty dict."""
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_config(data: dict) -> None:
    """Write config JSON to disk."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(data, indent=2))


def get_api_key(provider: str) -> str | None:
    """Get stored API key for *provider* ('anthropic' or 'openai')."""
    cfg = _load_config()
    return cfg.get("api_keys", {}).get(provider)


def set_api_key(provider: str, key: str) -> None:
    """Persist API key for *provider*."""
    cfg = _load_config()
    cfg.setdefault("api_keys", {})[provider] = key
    _save_config(cfg)


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

_NATIVE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

_CONVERT_TYPES = {".pdf", ".tiff", ".tif"}


def _encode_image(file_bytes: bytes, suffix: str) -> tuple[str, str]:
    """Return (base64_string, media_type) for the image.

    Native formats (.png, .jpg, .gif, .webp) are encoded directly.
    PDF is converted to PNG via PyMuPDF (preferred) or pdf2image.
    TIFF is converted to PNG via Pillow.
    """
    suffix = suffix.lower()

    if suffix in _NATIVE_TYPES:
        b64 = base64.standard_b64encode(file_bytes).decode()
        return b64, _NATIVE_TYPES[suffix]

    # PDF: try PyMuPDF first (no Pillow needed), then pdf2image
    if suffix == ".pdf":
        return _encode_pdf(file_bytes)

    # TIFF: convert via Pillow
    if suffix in (".tiff", ".tif"):
        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "Pillow is required to process TIFF images. "
                "Install with: pip install Pillow"
            )
        img = Image.open(io.BytesIO(file_bytes))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        b64 = base64.standard_b64encode(buf.getvalue()).decode()
        return b64, "image/png"

    raise ValueError(f"Unsupported image format: {suffix}")


def _encode_pdf(file_bytes: bytes) -> tuple[str, str]:
    """Convert first page of a PDF to base64 PNG.

    Tries PyMuPDF first (self-contained, no system deps), then
    falls back to pdf2image (requires poppler).
    """
    # Try PyMuPDF (fitz) — single pip install, no system deps
    try:
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page = doc[0]
        pix = page.get_pixmap(dpi=150)
        png_bytes = pix.tobytes("png")
        doc.close()
        b64 = base64.standard_b64encode(png_bytes).decode()
        return b64, "image/png"
    except ImportError:
        pass

    # Try pdf2image (requires poppler system library)
    try:
        from pdf2image import convert_from_bytes
        from PIL import Image
        images = convert_from_bytes(file_bytes, first_page=1, last_page=1)
        buf = io.BytesIO()
        images[0].convert("RGB").save(buf, format="PNG")
        b64 = base64.standard_b64encode(buf.getvalue()).decode()
        return b64, "image/png"
    except ImportError:
        pass

    raise ImportError(
        "PDF support requires PyMuPDF or pdf2image. "
        "Install with: pip install PyMuPDF"
    )



# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
You are a scientific-figure style analyst. Examine the uploaded plot image \
carefully and extract its visual style as JSON. Reply with ONLY a JSON object \
— no markdown fences, no explanation.

IMPORTANT — look closely at these visual details:

FIGURE SIZE: Estimate the figure dimensions in inches.
  - Single-column journal figures are typically 3.5" wide.
  - Double-column figures are typically 7" wide.
  - Presentation slides are typically 10" wide.
  - Estimate height from the aspect ratio you observe.

FONTS: Estimate point sizes by comparing text to the plot area.
  - Title text is typically 12-18 pt. Axis labels are typically 10-14 pt.
  - Tick labels are typically 8-12 pt. Legend text is typically 8-11 pt.
  - Identify the font family: serif fonts (Times, Computer Modern) have small
    strokes at letter ends; sans-serif fonts (Arial, Helvetica) do not.

FONT STYLE: Check if title and axis labels are bold or italic.
  - If the title text appears thicker/heavier than normal, set title_bold=true.
  - If axis label text appears thicker/heavier, set label_bold=true.

LABEL PADDING: Estimate spacing between labels and the axes.
  - title_pad is the gap between the title and the top of the plot (typically 6).
  - xlabel_pad / ylabel_pad is the gap between axis labels and tick labels
    (typically 4).

TICKS: Look very carefully at tick marks on the axes.
  - "out" = ticks point outward away from the plot area (most common in
    scientific figures)
  - "in" = ticks point inward into the plot area
  - "inout" = ticks cross the axis line in both directions
  - If you cannot see tick marks at all, use "out" as default.

TICK SPACING: THIS IS CRITICAL. Read the numeric labels on each axis carefully
  to determine the exact step between consecutive major ticks.
  - Example: y-axis labels 0, 0.5, 1.0, 1.5 → y_tick_step = 0.5
  - Example: x-axis labels 0, 2, 4, 6, 8, 10 → x_tick_step = 2.0
  - Example: x-axis labels 0, 1, 2, 3 → x_tick_step = 1.0
  - Count the number of ticks to verify: n_ticks = (max - min) / step + 1.
  - Use 0 only if you truly cannot read the tick labels.

AXIS SCALE: Check if axes use linear or logarithmic scale.
  - If tick labels increase exponentially (1, 10, 100, 1000) → "log"
  - Most plots use "linear" (default).

SPINES: Check each of the four borders of the plot area.
  - Many scientific plots hide the top and right spines (spine_top=false,
    spine_right=false). Look carefully — if there is no line along the top
    or right edge, set to false.

GRID: Look for light horizontal/vertical lines behind the data.
  - If there are no grid lines, set grid_on=false.
  - If grid is present, estimate its transparency (grid_alpha) and whether
    lines are solid "-", dashed "--", dotted ":", or dash-dot "-.".

LEGEND: Look carefully for a legend box in the plot.
  - If there is no legend at all, set legend_show=false.
  - If there is a legend, check if it has a visible border (legend_frame).
  - Estimate the legend font size relative to the axis labels.
  - Identify the legend position: where is it in the plot?
    Valid positions: "upper right", "upper left", "lower left", "lower right",
    "right", "center left", "center right", "lower center", "upper center",
    "center", "outside_right", "outside_upper_right".
  - Count the number of columns in the legend (usually 1).

COLORS: Use exact hex color values, not names. Sample the actual pixel color
  from each data series. Common scientific palettes:
  - tab10: #1f77b4, #ff7f0e, #2ca02c, #d62728, #9467bd, ...
  - Set1: #e41a1c, #377eb8, #4daf4a, #984ea3, #ff7f00, ...

BACKGROUND: Check if the plot background is white (#ffffff), light gray
  (#f0f0f0), or another color.

Schema:
{
  "plot_type": "<line|scatter|bar|histogram|boxplot|violin|heatmap|errorbar|mixed>",
  "num_series": <int>,
  "fig_width": <float inches>,
  "fig_height": <float inches>,
  "global": {
    "font_family": "<serif or sans-serif font name, e.g. Arial, Times New Roman, Helvetica>",
    "title_size": <float pt>,
    "title_bold": <bool>,
    "title_pad": <float, typically 6>,
    "label_size": <float pt>,
    "label_bold": <bool>,
    "xlabel_pad": <float, typically 4>,
    "ylabel_pad": <float, typically 4>,
    "tick_size": <float pt>,
    "spine_top": <bool>,
    "spine_right": <bool>,
    "spine_bottom": <bool>,
    "spine_left": <bool>,
    "spine_width": <float, typically 0.5-2.0>,
    "tick_direction": "<in|out|inout>",
    "tick_length": <float, typically 3-8>,
    "tick_width": <float, typically 0.5-2.0>,
    "x_tick_step": <float — step between x-axis ticks, 0 for auto>,
    "y_tick_step": <float — step between y-axis ticks, 0 for auto>,
    "x_scale": "<linear|log|symlog>",
    "y_scale": "<linear|log|symlog>",
    "grid_on": <bool>,
    "grid_alpha": <float 0-1>,
    "grid_width": <float>,
    "grid_style": "<-|--|:-|-.>",
    "legend_show": <bool>,
    "legend_position": "<position string from list above>",
    "legend_frame": <bool - does the legend have a visible border box?>,
    "legend_fontsize": <float pt>,
    "legend_columns": <int, usually 1>,
    "background_color": "<hex>",
    "colormap": "<valid matplotlib colormap name like tab10, Set1, viridis, etc.>"
  },
  "series": [
    {
      "type": "<line|scatter|bar|histogram|boxplot|violin|heatmap|errorbar>",
      "color": "<hex>",
      "alpha": <float 0-1>,
      "label": "<str or null>",
      "line_width": <float or null>,
      "line_style": "<-|--|:-|-. or null>",
      "marker": "<matplotlib marker: o s ^ D v + x * or null>",
      "marker_size": <float or null>,
      "edge_color": "<hex or null>",
      "edge_width": <float or null>,
      "hatch": "</ \\\\ | - + x or null>",
      "fill": <bool or null>
    }
  ]
}

Rules:
- Examine the image carefully for colors, font styles, and axis details.
- For tick_direction, look at whether ticks extend outward or inward from the
  axis line. Default to "out" if unclear.
- For x_tick_step and y_tick_step, read the actual tick labels on each axis.
  This is one of the most important fields — get the exact spacing right.
- For colormap, only use valid matplotlib names (tab10, Set1, viridis, plasma,
  etc.). Never output "custom".
- Use null for properties that don't apply to the series type.
- Output valid JSON only — no extra text.
"""

ASSESSMENT_PROMPT = """\
You are a scientific-figure style assessment agent. You are given:
1. A REFERENCE plot image.
2. A JSON object of extracted style parameters from that image.

Your job: look at the reference image and verify EACH parameter in the JSON. \
If a value is wrong, output the corrected value. If a value is correct, do NOT \
include it. Output a JSON object with the same schema containing ONLY the \
fields that need correction.

CRITICAL — check these carefully against the image:

FIGURE SIZE:
  - Does the aspect ratio match fig_width/fig_height?
  - A 3.5×4.5 figure is portrait; a 7×4 figure is wide landscape.

TICK SPACING (most commonly wrong):
  - Read the tick labels on each axis. Count the numeric labels.
  - For x-axis: what is the step between consecutive labels?
    Example: labels "0, 2, 4, 6, 8" → x_tick_step should be 2.0, NOT 1.0
    Example: labels "0.0, 0.5, 1.0, 1.5" → x_tick_step should be 0.5
  - For y-axis: same logic. Read the actual numbers shown.
  - This is the #1 most commonly wrong parameter. Double-check it.

FONT SIZES:
  - Does title_size look right relative to the plot?
  - Does label_size (axis labels) look right?
  - Does tick_size match the tick label text?
  - Does legend_fontsize match the legend text?

TICK MARKS:
  - tick_direction: do ticks point out, in, or inout?
  - tick_length and tick_width: do they match what you see?

SPINES:
  - Is each spine (top, right, bottom, left) visible or hidden?

GRID:
  - Is grid_on correct? Are there grid lines?

LEGEND:
  - legend_show: is there a legend?
  - legend_position: where is it in the plot?
  - legend_frame: does it have a border box?
  - legend_columns: how many columns?

COLORS:
  - Do the series colors match the actual data series in the image?
  - Check alpha values.

AXIS SCALE:
  - Are x_scale/y_scale correct? (linear vs log)

BOLD/ITALIC:
  - Is the title bold? Are axis labels bold?

If everything is correct, output: {}

Reply with ONLY a JSON object — no markdown fences, no explanation.
"""

# ---------------------------------------------------------------------------
# API call functions
# ---------------------------------------------------------------------------

def _parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    text = text.strip()
    # Strip ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


def _auto_install(package: str) -> None:
    """Install a package into the running kernel's environment."""
    import subprocess
    import sys
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", package])


def _call_anthropic(b64: str, media_type: str, api_key: str,
                    prompt: str = EXTRACTION_PROMPT) -> dict:
    """Call Claude API with vision. Uses claude-sonnet-4-20250514."""
    try:
        import anthropic
    except ImportError:
        _auto_install("anthropic")
        import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64,
                    },
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }],
    )
    return _parse_json_response(message.content[0].text)


def _call_openai(b64: str, media_type: str, api_key: str,
                 prompt: str = EXTRACTION_PROMPT) -> dict:
    """Call GPT-4o with vision."""
    try:
        import openai
    except ImportError:
        _auto_install("openai")
        import openai

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{b64}",
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return _parse_json_response(response.choices[0].message.content)


def extract_style(image_bytes: bytes, suffix: str,
                  provider: str, api_key: str) -> dict:
    """Route to correct provider, return parsed style dict."""
    if not api_key:
        raise ValueError("API key is required.")

    b64, media_type = _encode_image(image_bytes, suffix)

    if provider == "anthropic":
        return _call_anthropic(b64, media_type, api_key)
    elif provider == "openai":
        return _call_openai(b64, media_type, api_key)
    else:
        raise ValueError(f"Unknown provider: {provider!r}")


def assess_style(ref_b64: str, ref_media: str,
                 extracted: dict, provider: str, api_key: str) -> dict:
    """Assessment pass: verify extracted JSON against the reference image.

    Sends the reference image + the extracted JSON to the LLM and asks it
    to check each parameter. Returns only the fields that need correction.
    """
    json_str = json.dumps(extracted, indent=2)
    prompt = ASSESSMENT_PROMPT + "\n\nExtracted parameters:\n" + json_str

    if provider == "anthropic":
        return _call_anthropic(ref_b64, ref_media, api_key, prompt=prompt)
    elif provider == "openai":
        return _call_openai(ref_b64, ref_media, api_key, prompt=prompt)
    else:
        raise ValueError(f"Unknown provider: {provider!r}")


# ---------------------------------------------------------------------------
# Profile download / upload helpers
# ---------------------------------------------------------------------------

def _trigger_download(data: dict, filename: str, output_widget) -> None:
    """Trigger browser file download of JSON profile via JS."""
    from IPython.display import display as ipy_display, Javascript

    json_str = json.dumps(data, indent=2)
    b64 = base64.standard_b64encode(json_str.encode()).decode()
    js = (
        "(function(){"
        "var a=document.createElement('a');"
        f"a.href='data:application/json;base64,{b64}';"
        f"a.download={json.dumps(filename)};"
        "document.body.appendChild(a);a.click();document.body.removeChild(a);"
        "})();"
    )
    with output_widget:
        output_widget.clear_output()
        ipy_display(Javascript(js))


def _load_profile_json(file_bytes) -> dict:
    """Parse uploaded .json profile file back to AI result dict."""
    if isinstance(file_bytes, memoryview):
        file_bytes = bytes(file_bytes)
    return json.loads(file_bytes)


# ---------------------------------------------------------------------------
# Style application
# ---------------------------------------------------------------------------

def apply_ai_style(ai_result: dict, global_panel, canvas,
                   artist_panels: list | None = None) -> dict:
    """Apply extracted style to figure.

    1. Apply figsize via layout sliders (so widgets stay in sync)
    2. Map ai_result["global"] -> profile keys -> apply_profile()
    3. Map ai_result["series"] -> per-artist matplotlib artists directly
    Returns the global profile dict (for optional saving).
    """
    from ._profiles import apply_profile
    import matplotlib

    gp = global_panel

    # Apply figure size via sliders so UI stays in sync
    fig_w = ai_result.get("fig_width")
    fig_h = ai_result.get("fig_height")
    if fig_w is not None and hasattr(gp, '_width_sl'):
        try:
            gp._width_sl.value = float(fig_w)
        except (ValueError, TypeError):
            pass
    if fig_h is not None and hasattr(gp, '_height_sl'):
        try:
            gp._height_sl.value = float(fig_h)
        except (ValueError, TypeError):
            pass

    global_data = ai_result.get("global", {})

    # Drop invalid colormap names before apply_profile tries to use them
    cmap = global_data.get("colormap")
    if cmap is not None and cmap not in matplotlib.colormaps:
        del global_data["colormap"]

    apply_profile(global_data, global_panel, canvas)

    series = ai_result.get("series", [])
    if artist_panels and series:
        _apply_series_styles(series, artist_panels, canvas)

    return global_data


def _apply_series_styles(series: list[dict], panels: list, canvas) -> None:
    """Match series[i] -> panels[i] by index order."""
    orig_redraw = canvas.redraw
    canvas.redraw = lambda: None
    try:
        for i, s in enumerate(series):
            if i >= len(panels):
                break
            panel = panels[i]
            _apply_one_series(s, panel)
    finally:
        canvas.redraw = orig_redraw
    canvas.force_redraw()


def _apply_one_series(s: dict, panel) -> None:
    """Apply a single series style dict to an artist panel.

    Applies directly to the matplotlib artists in panel._group.artists
    so that changes are persistent. Also updates the panel's color sync
    callback if available.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.collections import PathCollection  # scatter

    color = s.get("color")
    alpha = s.get("alpha")
    line_width = s.get("line_width")
    line_style = s.get("line_style")
    marker = s.get("marker")
    marker_size = s.get("marker_size")
    edge_color = s.get("edge_color")
    edge_width = s.get("edge_width")
    hatch = s.get("hatch")

    # Get the actual matplotlib artists
    if not hasattr(panel, '_group'):
        return
    artists = panel._group.artists

    for artist in artists:
        # --- Color ---
        if color is not None:
            try:
                if isinstance(artist, Line2D):
                    artist.set_color(color)
                elif isinstance(artist, PathCollection):
                    artist.set_facecolor(color)
                elif isinstance(artist, Patch):
                    artist.set_facecolor(color)
            except Exception:
                pass

        # --- Alpha ---
        if alpha is not None:
            try:
                artist.set_alpha(alpha)
            except Exception:
                pass

        # --- Line width ---
        if line_width is not None:
            try:
                if isinstance(artist, Line2D):
                    artist.set_linewidth(line_width)
                elif isinstance(artist, PathCollection):
                    artist.set_linewidths([line_width])
            except Exception:
                pass

        # --- Line style ---
        if line_style is not None:
            try:
                if isinstance(artist, Line2D):
                    artist.set_linestyle(line_style)
            except Exception:
                pass

        # --- Marker ---
        if marker is not None:
            try:
                if isinstance(artist, Line2D):
                    artist.set_marker(marker)
            except Exception:
                pass

        # --- Marker size ---
        if marker_size is not None:
            try:
                if isinstance(artist, Line2D):
                    artist.set_markersize(marker_size)
                elif isinstance(artist, PathCollection):
                    artist.set_sizes([marker_size ** 2])
            except Exception:
                pass

        # --- Edge color ---
        if edge_color is not None:
            try:
                if isinstance(artist, Patch):
                    artist.set_edgecolor(edge_color)
                elif isinstance(artist, PathCollection):
                    artist.set_edgecolor(edge_color)
            except Exception:
                pass

        # --- Edge width ---
        if edge_width is not None:
            try:
                if isinstance(artist, Patch):
                    artist.set_linewidth(edge_width)
                elif isinstance(artist, PathCollection):
                    artist.set_linewidths([edge_width])
            except Exception:
                pass

        # --- Hatch ---
        if hatch is not None:
            try:
                if isinstance(artist, Patch):
                    artist.set_hatch(hatch)
            except Exception:
                pass

    # Update panel's color indicator/sync if available
    if color is not None and hasattr(panel, '_update_color'):
        try:
            panel._update_color(color)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# UI panel builder
# ---------------------------------------------------------------------------

def create_ai_import_section(global_panel, canvas,
                             artist_panels: list | None = None):
    """Build the AI Style Import section for the Profiles panel."""
    import ipywidgets as widgets

    # --- Provider dropdown ---
    provider_dd = widgets.Dropdown(
        options=[("OpenAI (GPT-4o)", "openai"),
                 ("Anthropic (Claude)", "anthropic")],
        value="openai",
        description="Provider:",
        style={"description_width": "60px"},
        layout=widgets.Layout(width="220px"),
    )

    # --- API key field with eye toggle ---
    key_field = widgets.Password(
        value="",
        placeholder="paste API key",
        description="API Key:",
        style={"description_width": "60px"},
        layout=widgets.Layout(width="230px"),
    )
    key_field_visible = widgets.Text(
        value="",
        placeholder="paste API key",
        description="API Key:",
        style={"description_width": "60px"},
        layout=widgets.Layout(width="230px", display="none"),
    )
    _key_visible = {"on": False}

    def _sync_key_fields(change):
        source = change["owner"]
        if source is key_field:
            key_field_visible.value = change["new"]
        else:
            key_field.value = change["new"]
    key_field.observe(_sync_key_fields, names="value")
    key_field_visible.observe(_sync_key_fields, names="value")

    eye_btn = widgets.Button(
        description="", icon="eye",
        layout=widgets.Layout(width="30px", height="30px"),
        tooltip="Show/hide API key",
    )

    def _toggle_eye(_btn):
        _key_visible["on"] = not _key_visible["on"]
        if _key_visible["on"]:
            key_field.layout.display = "none"
            key_field_visible.layout.display = ""
            eye_btn.icon = "eye-slash"
        else:
            key_field_visible.layout.display = "none"
            key_field.layout.display = ""
            eye_btn.icon = "eye"
    eye_btn.on_click(_toggle_eye)

    key_box = widgets.HBox(
        [key_field, key_field_visible, eye_btn],
        layout=widgets.Layout(gap="2px", align_items="center"),
    )

    save_key_btn = widgets.Button(
        description="Save Key", icon="key",
        button_style="",
        layout=widgets.Layout(width="90px"),
        tooltip="Save API key to ~/.matplotly/config.json",
    )

    # Pre-fill key if saved
    def _prefill_key(*_args):
        prov = provider_dd.value
        saved = get_api_key(prov)
        if saved:
            key_field.value = saved
    _prefill_key()
    provider_dd.observe(lambda c: _prefill_key(), names="value")

    def _on_save_key(_btn):
        prov = provider_dd.value
        k = key_field.value.strip()
        if k:
            set_api_key(prov, k)
            status.value = "<span style='color:green'>Key saved.</span>"
        else:
            status.value = "<span style='color:red'>Enter a key first.</span>"
    save_key_btn.on_click(_on_save_key)

    # --- Image upload ---
    image_upload = widgets.FileUpload(
        accept=".pdf,.png,.jpg,.jpeg,.tiff,.tif,.gif,.webp",
        multiple=False,
        description="Reference image",
        layout=widgets.Layout(width="280px"),
    )

    # --- Extract button ---
    extract_btn = widgets.Button(
        description="Extract Style", icon="magic",
        button_style="primary",
        layout=widgets.Layout(width="140px"),
    )

    # --- Status ---
    status = widgets.HTML("")

    # --- Post-extraction controls (hidden initially) ---
    js_output = widgets.Output(
        layout=widgets.Layout(height="0px", overflow="hidden"))

    download_btn = widgets.Button(
        description="Download Profile", icon="download",
        button_style="info",
        layout=widgets.Layout(width="150px"),
    )

    # Load Profile (upload .json file)
    profile_upload = widgets.FileUpload(
        accept=".json",
        multiple=False,
        description="Load Profile",
        layout=widgets.Layout(width="220px"),
    )

    post_extract_box = widgets.VBox([
        widgets.HBox([download_btn],
                     layout=widgets.Layout(gap="4px")),
        widgets.HBox([profile_upload],
                     layout=widgets.Layout(gap="4px")),
    ], layout=widgets.Layout(display="none"))

    # Shared state
    _state: dict[str, Any] = {"last_result": None, "ref_b64": None,
                               "ref_media": None}

    # --- Callbacks ---
    def _on_extract(_btn):
        uploaded = image_upload.value
        if not uploaded:
            status.value = "<span style='color:red'>Upload an image first.</span>"
            return

        file_info = uploaded[0] if isinstance(uploaded, (list, tuple)) else uploaded
        content = file_info.get("content", b"") if isinstance(file_info, dict) else b""
        name = file_info.get("name", "image.png") if isinstance(file_info, dict) else "image.png"
        suffix = Path(name).suffix or ".png"

        api_key = key_field.value.strip()
        if not api_key:
            status.value = "<span style='color:red'>Enter an API key.</span>"
            return

        prov = provider_dd.value
        status.value = "<i>Pass 1: Extracting style...</i>"
        extract_btn.disabled = True

        try:
            # Store encoded reference for verification pass
            ref_b64, ref_media = _encode_image(content, suffix)
            _state["ref_b64"] = ref_b64
            _state["ref_media"] = ref_media

            # Pass 1: extract style
            result = extract_style(content, suffix, prov, api_key)
            _state["last_result"] = result
            apply_ai_style(result, global_panel, canvas, artist_panels)

            n_series = len(result.get("series", []))
            status.value = "<i>Pass 2: Assessing parameters...</i>"

            # Pass 2: assessment — verify extracted JSON against reference
            try:
                corrections = assess_style(
                    ref_b64, ref_media, result, prov, api_key)
                if corrections:
                    # Merge corrections into result and re-apply
                    _merge_corrections(result, corrections)
                    apply_ai_style(result, global_panel, canvas, artist_panels)
                    status.value = (
                        f"<span style='color:green'>"
                        f"Style applied with corrections! "
                        f"Detected <b>{result.get('plot_type', '?')}</b> "
                        f"with {n_series} series.</span>"
                    )
                else:
                    status.value = (
                        f"<span style='color:green'>"
                        f"Style applied! Detected "
                        f"<b>{result.get('plot_type', '?')}</b> "
                        f"with {n_series} series.</span>"
                    )
            except Exception:
                # Verification failed — first pass result still applied
                status.value = (
                    f"<span style='color:green'>"
                    f"Style applied! Detected "
                    f"<b>{result.get('plot_type', '?')}</b> "
                    f"with {n_series} series.</span>"
                )

            post_extract_box.layout.display = ""
        except ImportError as e:
            status.value = f"<span style='color:red'>{e}</span>"
        except Exception as e:
            status.value = f"<span style='color:red'>Error: {e}</span>"
        finally:
            extract_btn.disabled = False

    extract_btn.on_click(_on_extract)

    def _on_download(_btn):
        result = _state.get("last_result")
        if result is None:
            status.value = "<span style='color:#888'>No extraction result to download.</span>"
            return
        _trigger_download(result, "ai_style_profile.json", js_output)
        status.value = "<span style='color:green'>Downloading profile...</span>"

    download_btn.on_click(_on_download)

    def _on_profile_upload(change):
        uploaded = change["new"]
        if not uploaded:
            return
        file_info = uploaded[0] if isinstance(uploaded, (list, tuple)) else uploaded
        content = file_info.get("content", b"") if isinstance(file_info, dict) else b""
        name = (file_info.get("name", "profile.json")
                if isinstance(file_info, dict) else "profile.json")
        try:
            result = _load_profile_json(content)
            _state["last_result"] = result
            apply_ai_style(result, global_panel, canvas, artist_panels)
            n_series = len(result.get("series", []))
            status.value = (
                f"<span style='color:green'>"
                f"Profile loaded from <b>{name}</b>! "
                f"{n_series} series styles applied.</span>"
            )
        except Exception as e:
            status.value = f"<span style='color:red'>Error loading profile: {e}</span>"

    profile_upload.observe(_on_profile_upload, names="value")

    # --- Assemble ---
    section = widgets.VBox([
        provider_dd,
        widgets.HBox([key_box, save_key_btn],
                     layout=widgets.Layout(gap="4px", align_items="center")),
        image_upload,
        extract_btn,
        status,
        post_extract_box,
        js_output,
    ])

    return section


def _merge_corrections(result: dict, corrections: dict) -> None:
    """Merge verification corrections into the extraction result in-place."""
    # Merge top-level keys (fig_width, fig_height, etc.)
    for key in ("fig_width", "fig_height", "plot_type", "num_series"):
        if key in corrections:
            result[key] = corrections[key]

    # Merge global corrections
    if "global" in corrections:
        result.setdefault("global", {}).update(corrections["global"])

    # Merge series corrections by index
    if "series" in corrections:
        existing = result.get("series", [])
        for i, s_corr in enumerate(corrections["series"]):
            if i < len(existing):
                existing[i].update(s_corr)
