"""AI-powered style extraction from plot images.

Supports Anthropic (Claude) and OpenAI (GPT-4o) as vision LLM providers.
API keys are persisted in ~/.matplotly/config.json.
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
    PDF and TIFF are converted to PNG via Pillow (first page for PDF).
    Falls back gracefully if Pillow is not installed.
    """
    suffix = suffix.lower()

    if suffix in _NATIVE_TYPES:
        b64 = base64.standard_b64encode(file_bytes).decode()
        return b64, _NATIVE_TYPES[suffix]

    if suffix in _CONVERT_TYPES:
        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "Pillow is required to process PDF/TIFF images. "
                "Install with: pip install Pillow"
            )

        if suffix == ".pdf":
            try:
                from pdf2image import convert_from_bytes
                images = convert_from_bytes(file_bytes, first_page=1, last_page=1)
                img = images[0]
            except ImportError:
                # Fallback: try fitz (PyMuPDF)
                try:
                    import fitz
                    doc = fitz.open(stream=file_bytes, filetype="pdf")
                    page = doc[0]
                    pix = page.get_pixmap(dpi=150)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    doc.close()
                except ImportError:
                    raise ImportError(
                        "PDF support requires pdf2image or PyMuPDF. "
                        "Install with: pip install pdf2image  or  pip install PyMuPDF"
                    )
        else:
            img = Image.open(io.BytesIO(file_bytes))

        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        b64 = base64.standard_b64encode(buf.getvalue()).decode()
        return b64, "image/png"

    raise ValueError(f"Unsupported image format: {suffix}")


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
You are a scientific-figure style analyst. Examine the uploaded plot image and \
extract its visual style as JSON. Reply with ONLY a JSON object — no markdown \
fences, no commentary.

Use this exact schema:

{
  "plot_type": "<line|scatter|bar|histogram|boxplot|violin|heatmap|errorbar|mixed>",
  "num_series": <int>,
  "global": {
    "font_family": "<str>",
    "title_size": <float>, "label_size": <float>, "tick_size": <float>,
    "spine_top": <bool>, "spine_right": <bool>,
    "spine_bottom": <bool>, "spine_left": <bool>,
    "spine_width": <float>,
    "tick_direction": "<in|out|inout>",
    "tick_length": <float>, "tick_width": <float>,
    "grid_on": <bool>, "grid_alpha": <float>, "grid_width": <float>,
    "grid_style": "<-|--|:-|-.>",
    "legend_show": <bool>, "legend_frame": <bool>, "legend_fontsize": <float>,
    "background_color": "<hex>",
    "colormap": "<matplotlib colormap name or 'custom'>"
  },
  "series": [
    {
      "type": "<line|scatter|bar|histogram|boxplot|violin|heatmap|errorbar>",
      "color": "<hex>",
      "alpha": <float 0-1>,
      "label": "<str or null>",
      "line_width": <float or null>,
      "line_style": "<-|--|:-|-. or null>",
      "marker": "<matplotlib marker char or null>",
      "marker_size": <float or null>,
      "edge_color": "<hex or null>",
      "edge_width": <float or null>,
      "hatch": "<str or null>",
      "fill": <bool or null>
    }
  ]
}

Rules:
- Detect the plot type and number of data series.
- For each series, extract color (as hex), opacity, line/marker/bar properties.
- For global settings, estimate font sizes, spine visibility, grid, legend, and background.
- Use null for properties that don't apply to the series type.
- If you cannot determine a value, use a sensible default.
- Output valid JSON only.
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


def _call_anthropic(b64: str, media_type: str, api_key: str) -> dict:
    """Call Claude API with vision. Uses claude-sonnet-4-20250514."""
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
                    "text": EXTRACTION_PROMPT,
                },
            ],
        }],
    )
    return _parse_json_response(message.content[0].text)


def _call_openai(b64: str, media_type: str, api_key: str) -> dict:
    """Call GPT-4o with vision."""
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
                {
                    "type": "text",
                    "text": EXTRACTION_PROMPT,
                },
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


def _load_profile_json(file_bytes: bytes) -> dict:
    """Parse uploaded .json profile file back to AI result dict."""
    return json.loads(file_bytes)


# ---------------------------------------------------------------------------
# Style application
# ---------------------------------------------------------------------------

def apply_ai_style(ai_result: dict, global_panel, canvas,
                   artist_panels: list | None = None) -> dict:
    """Apply extracted style to figure.

    1. Map ai_result["global"] -> profile keys -> apply_profile()
    2. Map ai_result["series"] -> per-artist panel properties
    Returns the global profile dict (for optional saving).
    """
    from ._profiles import apply_profile

    global_data = ai_result.get("global", {})
    apply_profile(global_data, global_panel, canvas)

    series = ai_result.get("series", [])
    if artist_panels and series:
        _apply_series_styles(series, artist_panels, canvas)

    return global_data


def _apply_series_styles(series: list[dict], panels: list, canvas) -> None:
    """Match series[i] -> panels[i] by index order.

    For each panel, apply color, alpha, line_width, line_style,
    marker, marker_size, edge_color, edge_width, hatch via
    duck-typing (check for widget attributes before setting).
    """
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
    """Apply a single series style dict to an artist panel."""
    color = s.get("color")
    alpha = s.get("alpha")
    line_width = s.get("line_width")
    line_style = s.get("line_style")
    marker = s.get("marker")
    marker_size = s.get("marker_size")
    edge_color = s.get("edge_color")
    edge_width = s.get("edge_width")
    hatch = s.get("hatch")

    # Color — panels store color in _color attribute
    if color is not None and hasattr(panel, '_color'):
        panel._color = color
        # Trigger the color picker if it exists
        if hasattr(panel, '_apply_color'):
            panel._apply_color(color)

    # Alpha
    if alpha is not None and hasattr(panel, '_alpha'):
        panel._alpha = alpha

    # Line width — errorbar panels use _line_width
    if line_width is not None:
        if hasattr(panel, '_line_width'):
            panel._line_width = line_width

    # Line style
    if line_style is not None and hasattr(panel, '_line_style'):
        panel._line_style = line_style

    # Marker
    if marker is not None and hasattr(panel, '_marker'):
        panel._marker = marker

    # Marker size
    if marker_size is not None and hasattr(panel, '_marker_size'):
        panel._marker_size = marker_size

    # Edge color
    if edge_color is not None and hasattr(panel, '_edge_color'):
        panel._edge_color = edge_color

    # Edge width
    if edge_width is not None and hasattr(panel, '_edge_width'):
        panel._edge_width = edge_width

    # Hatch (bar/histogram panels)
    if hatch is not None and hasattr(panel, '_hatch'):
        panel._hatch = hatch


# ---------------------------------------------------------------------------
# UI panel builder
# ---------------------------------------------------------------------------

def create_ai_import_section(global_panel, canvas,
                             artist_panels: list | None = None):
    """Build the AI Style Import sub-section for the Profiles panel.

    Returns (widget, status_html, js_output) tuple.
    """
    import ipywidgets as widgets

    # --- Provider dropdown ---
    provider_dd = widgets.Dropdown(
        options=[("Anthropic (Claude)", "anthropic"),
                 ("OpenAI (GPT-4o)", "openai")],
        value="anthropic",
        description="Provider:",
        style={"description_width": "60px"},
        layout=widgets.Layout(width="220px"),
    )

    # --- API key field ---
    key_field = widgets.Password(
        value="",
        placeholder="paste API key",
        description="API Key:",
        style={"description_width": "60px"},
        layout=widgets.Layout(width="260px"),
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

    save_name = widgets.Text(
        value="", placeholder="profile name",
        layout=widgets.Layout(width="140px"),
    )
    save_profile_btn = widgets.Button(
        description="Save as Profile", icon="save",
        button_style="success",
        layout=widgets.Layout(width="130px"),
    )

    post_extract_box = widgets.VBox([
        widgets.HBox([download_btn],
                     layout=widgets.Layout(gap="4px")),
        widgets.HBox([save_name, save_profile_btn],
                     layout=widgets.Layout(gap="4px")),
    ], layout=widgets.Layout(display="none"))

    # --- Upload Profile (no API call) ---
    profile_upload = widgets.FileUpload(
        accept=".json",
        multiple=False,
        description="Upload Profile",
        layout=widgets.Layout(width="280px"),
    )

    # Shared state for the last extraction result
    _state: dict[str, Any] = {"last_result": None}

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
        status.value = "<i>Extracting style... (this may take a few seconds)</i>"
        extract_btn.disabled = True

        try:
            result = extract_style(content, suffix, prov, api_key)
            _state["last_result"] = result
            apply_ai_style(result, global_panel, canvas, artist_panels)
            n_series = len(result.get("series", []))
            status.value = (
                f"<span style='color:green'>"
                f"Style applied! Detected <b>{result.get('plot_type', '?')}</b> "
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

    def _on_save_profile(_btn):
        from ._profiles import _save_profile, _list_profiles

        result = _state.get("last_result")
        if result is None:
            status.value = "<span style='color:#888'>No extraction result.</span>"
            return
        name = save_name.value.strip()
        if not name:
            status.value = "<span style='color:red'>Enter a profile name.</span>"
            return
        safe = "".join(c for c in name if c.isalnum() or c in " _-").strip()
        if not safe:
            status.value = "<span style='color:red'>Invalid name.</span>"
            return
        # Save the global portion as a standard profile
        global_data = result.get("global", {})
        _save_profile(safe, global_data)
        save_name.value = ""
        status.value = (
            f"<span style='color:green'>Saved profile <b>{safe}</b></span>")

    save_profile_btn.on_click(_on_save_profile)

    def _on_profile_upload(change):
        uploaded = change["new"]
        if not uploaded:
            return
        file_info = uploaded[0] if isinstance(uploaded, (list, tuple)) else uploaded
        content = file_info.get("content", b"") if isinstance(file_info, dict) else b""
        name = file_info.get("name", "profile.json") if isinstance(file_info, dict) else "profile.json"

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
            post_extract_box.layout.display = ""
        except Exception as e:
            status.value = f"<span style='color:red'>Error loading profile: {e}</span>"

    profile_upload.observe(_on_profile_upload, names="value")

    # --- Assemble ---
    section = widgets.VBox([
        widgets.HTML("<hr style='margin:8px 0'>"
                     "<b style='font-size:12px'>AI Style Import</b>"),
        provider_dd,
        widgets.HBox([key_field, save_key_btn],
                     layout=widgets.Layout(gap="4px", align_items="center")),
        image_upload,
        extract_btn,
        status,
        post_extract_box,
        widgets.HTML("<hr style='margin:8px 0'>"
                     "<b style='font-size:11px'>Upload Profile (no API call)</b>"),
        profile_upload,
        js_output,
    ])

    return section
