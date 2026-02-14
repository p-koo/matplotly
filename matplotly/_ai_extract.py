"""AI-powered style extraction from plot images.

Supports Anthropic (Claude) and OpenAI (GPT-4o) as vision LLM providers.
API keys are read from environment variables (ANTHROPIC_API_KEY,
OPENAI_API_KEY) — no secrets are written to disk.

Two-pass extraction: first pass extracts style, second pass compares the
modified figure against the reference and corrects discrepancies.
"""
from __future__ import annotations

import base64
import json
import io
import os
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# API key helpers
# ---------------------------------------------------------------------------

_ENV_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def get_api_key(provider: str) -> str | None:
    """Get API key for *provider* from the environment."""
    env_var = _ENV_KEYS.get(provider, "")
    return os.environ.get(env_var) or None


def set_api_key(provider: str, key: str) -> None:
    """Set API key in the current process environment (not persisted)."""
    env_var = _ENV_KEYS.get(provider)
    if env_var:
        os.environ[env_var] = key


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

FONTS: This is CRITICAL. Use the RELATIVE SIZE METHOD to estimate font sizes:
  1. First, estimate tick_size by looking at the tick labels (small numbers on
     the axes). In most scientific figures, tick labels are 8-10 pt.
  2. Then estimate other sizes RELATIVE to tick labels:
     - label_size (axis labels like "X axis", "Y axis") is usually 1-2 pt
       larger than tick labels. If tick_size=9, label_size is typically 10-11.
     - title_size is usually 1-3 pt larger than label_size. If label_size=10,
       title_size is typically 11-13.
     - legend_fontsize is usually equal to or 1 pt smaller than tick_size.
       If tick_size=9, legend_fontsize is typically 8-9.
  3. Common size combinations in scientific figures:
     - Small/compact: tick=8, label=9, title=10, legend=8
     - Standard: tick=9, label=10, title=12, legend=9
     - Large/presentation: tick=12, label=14, title=16, legend=11
  - Identify the font family: serif fonts (Times, Computer Modern) have small
    strokes at letter ends; sans-serif fonts (Arial, Helvetica) do not.

FONT WEIGHT (bold): Default to false for ALL font weight fields.
  - title_bold: false (most scientific titles are NOT bold)
  - label_bold: false (most axis labels are NOT bold)
  - Only set to true if the text is UNMISTAKABLY heavier/thicker than
    surrounding text. When in doubt, always use false.

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

SPINES: THIS IS CRITICAL. Check ALL FOUR borders of the plot area individually.
  You MUST set each spine independently — do not skip any.
  - spine_top: Is there a line along the TOP edge of the plot area? If NO
    line is visible at the top, set spine_top=false.
  - spine_right: Is there a line along the RIGHT edge of the plot area? If NO
    line is visible on the right, set spine_right=false.
  - spine_bottom: Is there a line along the BOTTOM edge? (Usually true.)
  - spine_left: Is there a line along the LEFT edge? (Usually true.)
  - MOST scientific figures hide top and right spines. Default to
    spine_top=false, spine_right=false unless you clearly see those lines.
  - spine_width: thickness of visible spine lines (typically 0.5-1.5).

GRID: Look for light horizontal/vertical lines behind the data.
  - If there are no grid lines, set grid_on=false.
  - If grid is present, estimate its transparency (grid_alpha) and whether
    lines are solid "-", dashed "--", dotted ":", or dash-dot "-.".

LEGEND: Look carefully for a legend box in the plot.
  - If there is no legend at all, set legend_show=false.
  - legend_frame: DEFAULT TO false. Most scientific figures do NOT have a
    visible border/frame around the legend. Only set to true if you can
    clearly see a rectangular border line drawn around the legend entries.
    A subtle background shade does NOT count as a frame — the frame refers
    specifically to a visible black or dark outline/border box.
  - Estimate the legend font size relative to the tick labels (usually
    equal to or 1 pt smaller than tick_size).
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

Your job: go through EVERY parameter below one by one and verify it against \
the reference image. Output a JSON object with ONLY fields that need \
correction. Use the same JSON structure as the input (with "global" and \
optionally "series" keys). If everything looks correct, output: {}

RULES:
- Do NOT set title_bold or label_bold to true unless text is UNMISTAKABLY \
  heavier than normal weight. Default assumption is false.
- Do NOT change colors unless they are clearly a different hue.

=== COMPLETE CHECKLIST — verify EVERY item ===

1. FIGURE SIZE:
   [ ] fig_width — does this match the apparent figure width?
   [ ] fig_height — does this match the apparent aspect ratio?

2. FONT SIZES (use RELATIVE SIZE METHOD):
   [ ] tick_size — look at tick labels (small numbers on axes). Usually 8-10pt.
   [ ] label_size — axis labels. Usually 1-2pt larger than tick_size.
   [ ] title_size — plot title. Usually 1-3pt larger than label_size.
   [ ] legend_fontsize — legend text. Usually equal to or 1pt smaller than tick_size.
   Common combos: tick=8/label=9/title=10/legend=8 (compact),
   tick=9/label=10/title=12/legend=9 (standard),
   tick=12/label=14/title=16/legend=11 (large).
   Correct any size that is off by 1+ points.

3. FONT PROPERTIES:
   [ ] font_family — serif (strokes at letter ends) vs sans-serif (clean edges)?
   [ ] title_bold — false unless UNMISTAKABLY bold.
   [ ] label_bold — false unless UNMISTAKABLY bold.

4. LABEL PADDING:
   [ ] title_pad — gap between title and plot top (typically 6).
   [ ] xlabel_pad — gap between x-axis label and tick labels (typically 4).
   [ ] ylabel_pad — gap between y-axis label and tick labels (typically 4).

5. SPINES (check ALL four borders of the plot area):
   [ ] spine_top — is there a line along the TOP edge? true/false.
   [ ] spine_right — is there a line along the RIGHT edge? true/false.
   [ ] spine_bottom — is there a line along the BOTTOM edge? true/false.
   [ ] spine_left — is there a line along the LEFT edge? true/false.
   [ ] spine_width — how thick are the visible spine lines?

6. TICKS:
   [ ] tick_direction — "out" (away from plot), "in" (into plot), or "inout"?
   [ ] tick_length — how long are the tick marks? (typically 3-8)
   [ ] tick_width — how thick are the tick marks? (typically 0.5-2.0)

7. TICK SPACING (read the actual axis numbers):
   [ ] x_tick_step — step between consecutive x-axis labels.
       Example: labels "0, 2, 4, 6" → x_tick_step = 2.0
   [ ] y_tick_step — step between consecutive y-axis labels.
       Example: labels "0, 0.5, 1.0, 1.5" → y_tick_step = 0.5

8. AXIS SCALE:
   [ ] x_scale — "linear" or "log" (exponential labels like 1, 10, 100)?
   [ ] y_scale — "linear" or "log"?

9. GRID:
   [ ] grid_on — are there gridlines behind the data? true/false.
   [ ] grid_alpha — if grid is on, how transparent? (0.1-1.0)
   [ ] grid_width — how thick are the grid lines?
   [ ] grid_style — solid "-", dashed "--", dotted ":", dash-dot "-."?

10. LEGEND:
    [ ] legend_show — is there a legend visible? true/false.
    [ ] legend_frame — DEFAULT false. Only true if there is a clearly visible
        rectangular border/outline around the legend. Subtle background does
        NOT count as a frame.
    [ ] legend_fontsize — size of legend text.
    [ ] legend_position — location in the plot.
    [ ] legend_columns — number of columns (usually 1).

11. BACKGROUND:
    [ ] background_color — white "#ffffff", gray, or other?

12. COLORMAP:
    [ ] colormap — does the palette match? (tab10, Set1, viridis, etc.)

13. SERIES (for each data series):
    [ ] color — correct hex color?
    [ ] alpha — correct opacity?
    [ ] line_width, line_style, marker, marker_size — correct for line/scatter?
    [ ] edge_color, edge_width, hatch — correct for bar/patch?

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


_ANTHROPIC_MODELS = [
    ("Claude Sonnet 4", "claude-sonnet-4-20250514"),
    ("Claude Haiku 3.5", "claude-haiku-4-5-20251001"),
]

_OPENAI_MODELS = [
    ("GPT-5.2", "gpt-5.2"),
    ("GPT-5.1", "gpt-5.1"),
    ("GPT-5", "gpt-5"),
    ("GPT-5 mini", "gpt-5-mini"),
    ("GPT-5 nano", "gpt-5-nano"),
    ("GPT-4o", "gpt-4o"),
    ("GPT-4o mini", "gpt-4o-mini"),
    ("GPT-4.1", "gpt-4.1"),
    ("GPT-4.1 mini", "gpt-4.1-mini"),
    ("o4 mini", "o4-mini"),
    ("o3", "o3"),
    ("o3 mini", "o3-mini"),
]


def _call_anthropic(b64: str, media_type: str, api_key: str,
                    prompt: str = EXTRACTION_PROMPT,
                    model: str = "claude-sonnet-4-20250514") -> dict:
    """Call Claude API with vision."""
    try:
        import anthropic
    except ImportError:
        _auto_install("anthropic")
        import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
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
                 prompt: str = EXTRACTION_PROMPT,
                 model: str = "gpt-5.2") -> dict:
    """Call OpenAI API with vision."""
    try:
        import openai
    except ImportError:
        _auto_install("openai")
        import openai

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=2048,
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


def _read_current_font_sizes(global_panel) -> dict | None:
    """Read current font sizes from the figure for LLM calibration."""
    gp = global_panel
    if gp is None:
        return None
    try:
        sizes = {}
        if hasattr(gp, '_tick_size_sl'):
            sizes["tick_size"] = round(gp._tick_size_sl.value, 1)
        if hasattr(gp, '_label_size_sl'):
            sizes["label_size"] = round(gp._label_size_sl.value, 1)
        if hasattr(gp, '_title_size_sl'):
            sizes["title_size"] = round(gp._title_size_sl.value, 1)
        if hasattr(gp, '_legend_fontsize_sl'):
            sizes["legend_fontsize"] = round(gp._legend_fontsize_sl.value, 1)
        return sizes if sizes else None
    except Exception:
        return None


def extract_style(image_bytes: bytes, suffix: str,
                  provider: str, api_key: str,
                  model: str | None = None,
                  current_font_sizes: dict | None = None) -> dict:
    """Route to correct provider, return parsed style dict."""
    if not api_key:
        raise ValueError("API key is required.")

    b64, media_type = _encode_image(image_bytes, suffix)

    # Build prompt with optional font size calibration context
    prompt = EXTRACTION_PROMPT
    if current_font_sizes:
        hint = (
            "\n\nCALIBRATION CONTEXT — The current figure uses these font sizes "
            "(in points). Use them as a reference anchor to judge whether the "
            "reference image uses similar, larger, or smaller fonts. Adjust "
            "each size accordingly:\n"
        )
        for k, v in current_font_sizes.items():
            hint += f"  - Current {k}: {v} pt\n"
        hint += (
            "If the reference image text looks SIMILAR in size to these values, "
            "use values close to them. If it looks noticeably SMALLER, reduce by "
            "1-3 pt. If noticeably LARGER, increase by 1-3 pt. "
            "Do NOT deviate more than necessary.\n"
        )
        prompt = prompt + hint

    if provider == "anthropic":
        kw = {"model": model} if model else {}
        result = _call_anthropic(b64, media_type, api_key, prompt=prompt, **kw)
    elif provider == "openai":
        kw = {"model": model} if model else {}
        result = _call_openai(b64, media_type, api_key, prompt=prompt, **kw)
    else:
        raise ValueError(f"Unknown provider: {provider!r}")

    # Ensure every global parameter is present (fill omitted fields)
    _ensure_global_fields(result)
    return result


# Required global fields with sensible defaults if omitted by the LLM.
# This ensures every controllable parameter is always present.
_GLOBAL_DEFAULTS = {
    "font_family": "Arial",
    "title_size": 12.0,
    "title_bold": False,
    "title_pad": 6.0,
    "label_size": 10.0,
    "label_bold": False,
    "xlabel_pad": 4.0,
    "ylabel_pad": 4.0,
    "tick_size": 10.0,
    "spine_top": False,
    "spine_right": False,
    "spine_bottom": True,
    "spine_left": True,
    "spine_width": 1.0,
    "tick_direction": "out",
    "tick_length": 5.0,
    "tick_width": 1.0,
    "x_tick_step": 0,
    "y_tick_step": 0,
    "x_scale": "linear",
    "y_scale": "linear",
    "grid_on": False,
    "grid_alpha": 0.3,
    "grid_width": 0.5,
    "grid_style": "--",
    "legend_show": True,
    "legend_frame": False,
    "legend_fontsize": 10.0,
    "legend_position": "upper right",
    "legend_columns": 1,
    "background_color": "#ffffff",
}


def _ensure_global_fields(result: dict) -> None:
    """Fill in any missing global fields with defaults so every parameter
    is always present in the result. Modifies *result* in place."""
    g = result.setdefault("global", {})
    for key, default in _GLOBAL_DEFAULTS.items():
        if key not in g:
            g[key] = default


def assess_style(ref_b64: str, ref_media: str,
                 extracted: dict, provider: str, api_key: str,
                 model: str | None = None) -> dict:
    """Assessment pass: verify extracted JSON against the reference image.

    Sends the reference image + the extracted JSON to the LLM and asks it
    to check each parameter. Returns only the fields that need correction.
    """
    json_str = json.dumps(extracted, indent=2)
    prompt = ASSESSMENT_PROMPT + "\n\nExtracted parameters:\n" + json_str

    if provider == "anthropic":
        kw = {"model": model} if model else {}
        return _call_anthropic(ref_b64, ref_media, api_key, prompt=prompt, **kw)
    elif provider == "openai":
        kw = {"model": model} if model else {}
        return _call_openai(ref_b64, ref_media, api_key, prompt=prompt, **kw)
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
        options=[("OpenAI", "openai"),
                 ("Anthropic", "anthropic")],
        value="openai",
        description="Provider:",
        style={"description_width": "60px"},
        layout=widgets.Layout(width="180px"),
    )

    # --- Model dropdown ---
    model_dd = widgets.Dropdown(
        options=_OPENAI_MODELS,
        value=_OPENAI_MODELS[0][1],
        description="Model:",
        style={"description_width": "50px"},
        layout=widgets.Layout(width="180px"),
    )

    def _on_provider_change(change):
        prov = change["new"]
        if prov == "openai":
            model_dd.options = _OPENAI_MODELS
            model_dd.value = _OPENAI_MODELS[0][1]
        else:
            model_dd.options = _ANTHROPIC_MODELS
            model_dd.value = _ANTHROPIC_MODELS[0][1]
    provider_dd.observe(_on_provider_change, names="value")

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

    # Pre-fill key from environment variable
    def _prefill_key(*_args):
        prov = provider_dd.value
        saved = get_api_key(prov)
        if saved:
            key_field.value = saved
    _prefill_key()
    provider_dd.observe(lambda c: _prefill_key(), names="value")

    # --- Image upload ---
    image_upload = widgets.FileUpload(
        accept=".pdf,.png,.jpg,.jpeg,.tiff,.tif,.gif,.webp",
        multiple=False,
        description="Reference image",
        layout=widgets.Layout(width="280px"),
    )
    # Reset description after upload to remove "(N)" file count
    def _reset_image_desc(change):
        image_upload.description = "Reference image"
    image_upload.observe(_reset_image_desc, names="value")

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
    # Reset description after upload to remove "(N)" file count
    def _reset_profile_desc(change):
        profile_upload.description = "Load Profile"
    profile_upload.observe(_reset_profile_desc, names="value")

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
        # Keep key available for the rest of this session (env var only)
        set_api_key(prov, api_key)
        status.value = "<i>Pass 1: Extracting style...</i>"
        extract_btn.disabled = True

        try:
            # Store encoded reference for verification pass
            ref_b64, ref_media = _encode_image(content, suffix)
            _state["ref_b64"] = ref_b64
            _state["ref_media"] = ref_media

            # Pass 1: extract style (with current font sizes as calibration)
            mdl = model_dd.value
            cur_fonts = _read_current_font_sizes(global_panel)
            result = extract_style(content, suffix, prov, api_key,
                                   model=mdl, current_font_sizes=cur_fonts)
            _state["last_result"] = result
            apply_ai_style(result, global_panel, canvas, artist_panels)

            n_series = len(result.get("series", []))
            status.value = "<i>Pass 2: Assessing parameters...</i>"

            # Pass 2: assessment — verify extracted JSON against reference
            corrected = False
            try:
                corrections = assess_style(
                    ref_b64, ref_media, result, prov, api_key, model=mdl)
                if corrections:
                    _merge_corrections(result, corrections)
                    apply_ai_style(result, global_panel, canvas, artist_panels)
                    corrected = True
            except Exception:
                pass  # Verification failed — first pass result still applied

            # Build status with collapsible details
            _state["last_result"] = result  # update after corrections
            label = "with corrections" if corrected else ""
            g = result.get("global", {})
            details_json = json.dumps(g, indent=2)
            status.value = (
                f"<span style='color:green'>"
                f"Style applied{' ' + label if label else ''}! "
                f"Detected <b>{result.get('plot_type', '?')}</b> "
                f"with {n_series} series.</span>"
                f"<details><summary style='cursor:pointer;color:#666;"
                f"font-size:11px'>Show extracted parameters</summary>"
                f"<pre style='font-size:10px;max-height:300px;"
                f"overflow:auto;background:#f5f5f5;padding:4px;"
                f"margin-top:4px'>{details_json}</pre></details>"
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
        widgets.HBox([provider_dd, model_dd],
                     layout=widgets.Layout(gap="4px")),
        key_box,
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
