"""Tests for AI-powered style extraction (mocked API calls).

Run:  python -m pytest tests/test_ai_extract.py -v
"""
from __future__ import annotations

import base64
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure matplotly is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from matplotly._ai_extract import (
    _encode_image,
    _load_config,
    _parse_json_response,
    _save_config,
    apply_ai_style,
    extract_style,
    get_api_key,
    set_api_key,
    CONFIG_FILE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_config(tmp_path, monkeypatch):
    """Redirect config storage to a temp directory."""
    cfg_file = tmp_path / "config.json"
    monkeypatch.setattr("matplotly._ai_extract.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("matplotly._ai_extract.CONFIG_FILE", cfg_file)
    return cfg_file


@pytest.fixture
def sample_ai_result():
    """A realistic AI extraction result."""
    return {
        "plot_type": "line",
        "num_series": 2,
        "global": {
            "font_family": "Arial",
            "title_size": 14.0,
            "label_size": 12.0,
            "tick_size": 10.0,
            "spine_top": False,
            "spine_right": False,
            "spine_bottom": True,
            "spine_left": True,
            "spine_width": 1.0,
            "tick_direction": "out",
            "tick_length": 5.0,
            "tick_width": 1.0,
            "grid_on": False,
            "grid_alpha": 0.3,
            "grid_width": 0.5,
            "grid_style": "--",
            "legend_show": True,
            "legend_frame": False,
            "legend_fontsize": 10.0,
            "background_color": "#ffffff",
            "colormap": "tab10",
        },
        "series": [
            {
                "type": "line",
                "color": "#1f77b4",
                "alpha": 1.0,
                "label": "Series A",
                "line_width": 2.0,
                "line_style": "-",
                "marker": "o",
                "marker_size": 6.0,
                "edge_color": None,
                "edge_width": None,
                "hatch": None,
                "fill": None,
            },
            {
                "type": "line",
                "color": "#ff7f0e",
                "alpha": 0.8,
                "label": "Series B",
                "line_width": 1.5,
                "line_style": "--",
                "marker": "s",
                "marker_size": 5.0,
                "edge_color": None,
                "edge_width": None,
                "hatch": None,
                "fill": None,
            },
        ],
    }


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_save_load_roundtrip(self, tmp_config):
        assert _load_config() == {}
        _save_config({"foo": "bar"})
        assert _load_config() == {"foo": "bar"}

    def test_api_key_save_load(self, tmp_config):
        assert get_api_key("anthropic") is None
        set_api_key("anthropic", "sk-test-123")
        assert get_api_key("anthropic") == "sk-test-123"
        assert get_api_key("openai") is None

    def test_api_key_overwrite(self, tmp_config):
        set_api_key("openai", "sk-old")
        set_api_key("openai", "sk-new")
        assert get_api_key("openai") == "sk-new"


# ---------------------------------------------------------------------------
# Image encoding tests
# ---------------------------------------------------------------------------

class TestEncodeImage:
    def test_encode_png(self):
        # Minimal 1x1 PNG
        png_bytes = (
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
            b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00'
            b'\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00'
            b'\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
        )
        b64, media_type = _encode_image(png_bytes, ".png")
        assert media_type == "image/png"
        # Verify round-trip
        decoded = base64.standard_b64decode(b64)
        assert decoded == png_bytes

    def test_encode_jpeg(self):
        # Minimal JPEG-like bytes (just checks encoding, not validity)
        jpeg_bytes = b'\xff\xd8\xff\xe0' + b'\x00' * 20
        b64, media_type = _encode_image(jpeg_bytes, ".jpg")
        assert media_type == "image/jpeg"
        decoded = base64.standard_b64decode(b64)
        assert decoded == jpeg_bytes

    def test_encode_jpeg_extension(self):
        jpeg_bytes = b'\xff\xd8\xff\xe0' + b'\x00' * 10
        b64, media_type = _encode_image(jpeg_bytes, ".jpeg")
        assert media_type == "image/jpeg"

    def test_encode_pdf(self):
        """PDF bytes → PNG via PyMuPDF."""
        import fitz
        # Create a minimal 1-page PDF with PyMuPDF
        doc = fitz.open()
        page = doc.new_page(width=72, height=72)
        page.draw_rect(fitz.Rect(10, 10, 62, 62), color=(1, 0, 0))
        pdf_bytes = doc.tobytes()
        doc.close()

        b64, media_type = _encode_image(pdf_bytes, ".pdf")
        assert media_type == "image/png"
        decoded = base64.standard_b64decode(b64)
        # Should start with PNG signature
        assert decoded[:4] == b'\x89PNG'

    def test_unsupported_format(self):
        with pytest.raises(ValueError, match="Unsupported"):
            _encode_image(b"data", ".bmp")


# ---------------------------------------------------------------------------
# JSON parsing tests
# ---------------------------------------------------------------------------

class TestParseResponse:
    def test_plain_json(self, sample_ai_result):
        text = json.dumps(sample_ai_result)
        parsed = _parse_json_response(text)
        assert parsed["plot_type"] == "line"
        assert parsed["num_series"] == 2

    def test_markdown_fence_stripping(self, sample_ai_result):
        text = "```json\n" + json.dumps(sample_ai_result) + "\n```"
        parsed = _parse_json_response(text)
        assert parsed["plot_type"] == "line"

    def test_bare_fence_stripping(self, sample_ai_result):
        text = "```\n" + json.dumps(sample_ai_result) + "\n```"
        parsed = _parse_json_response(text)
        assert parsed["plot_type"] == "line"

    def test_whitespace_handling(self, sample_ai_result):
        text = "  \n```json\n" + json.dumps(sample_ai_result) + "\n```\n  "
        parsed = _parse_json_response(text)
        assert parsed["plot_type"] == "line"


# ---------------------------------------------------------------------------
# Style application tests
# ---------------------------------------------------------------------------

class TestApplyGlobalStyle:
    def test_apply_global_style(self, sample_ai_result):
        """Verify that apply_ai_style calls apply_profile with global data."""
        mock_global = MagicMock()
        mock_canvas = MagicMock()

        with patch("matplotly._profiles.apply_profile") as mock_apply:
            result = apply_ai_style(sample_ai_result, mock_global, mock_canvas)

        mock_apply.assert_called_once_with(
            sample_ai_result["global"], mock_global, mock_canvas)
        assert result == sample_ai_result["global"]


class TestApplySeriesStyles:
    def test_apply_color_and_alpha(self):
        """Series color and alpha are applied to panel attributes."""
        panel = MagicMock()
        panel._color = "#000000"
        panel._alpha = 1.0
        panel._line_width = 1.0
        panel._marker = ""
        panel._marker_size = 4.0

        series = [{
            "color": "#ff0000",
            "alpha": 0.5,
            "line_width": 3.0,
            "marker": "^",
            "marker_size": 8.0,
        }]

        ai_result = {"global": {}, "series": series}
        mock_global = MagicMock()
        mock_canvas = MagicMock()

        with patch("matplotly._profiles.apply_profile"):
            apply_ai_style(ai_result, mock_global, mock_canvas,
                           artist_panels=[panel])

        assert panel._color == "#ff0000"
        assert panel._alpha == 0.5
        assert panel._line_width == 3.0
        assert panel._marker == "^"
        assert panel._marker_size == 8.0

    def test_apply_hatch_and_edge(self):
        """Hatch and edge properties applied when panel has them."""
        panel = MagicMock()
        panel._color = "#000"
        panel._alpha = 1.0
        panel._hatch = ""
        panel._edge_color = "#000"
        panel._edge_width = 1.0

        series = [{
            "color": "#aabbcc",
            "alpha": 0.7,
            "hatch": "//",
            "edge_color": "#112233",
            "edge_width": 2.0,
        }]

        ai_result = {"global": {}, "series": series}
        mock_global = MagicMock()
        mock_canvas = MagicMock()

        with patch("matplotly._profiles.apply_profile"):
            apply_ai_style(ai_result, mock_global, mock_canvas,
                           artist_panels=[panel])

        assert panel._hatch == "//"
        assert panel._edge_color == "#112233"
        assert panel._edge_width == 2.0

    def test_skip_extra_series(self):
        """More series than panels → extra series are silently skipped."""
        panel = MagicMock()
        panel._color = "#000"
        panel._alpha = 1.0

        series = [
            {"color": "#111", "alpha": 0.9},
            {"color": "#222", "alpha": 0.8},  # No matching panel
        ]

        ai_result = {"global": {}, "series": series}
        mock_global = MagicMock()
        mock_canvas = MagicMock()

        with patch("matplotly._profiles.apply_profile"):
            apply_ai_style(ai_result, mock_global, mock_canvas,
                           artist_panels=[panel])

        assert panel._color == "#111"

    def test_null_values_skipped(self):
        """Null values in series don't overwrite panel attrs."""
        panel = MagicMock()
        panel._color = "#original"
        panel._alpha = 0.9
        panel._line_width = 2.0

        series = [{"color": "#new", "alpha": None, "line_width": None}]

        ai_result = {"global": {}, "series": series}
        mock_global = MagicMock()
        mock_canvas = MagicMock()

        with patch("matplotly._profiles.apply_profile"):
            apply_ai_style(ai_result, mock_global, mock_canvas,
                           artist_panels=[panel])

        assert panel._color == "#new"
        assert panel._alpha == 0.9  # unchanged
        assert panel._line_width == 2.0  # unchanged


# ---------------------------------------------------------------------------
# extract_style() tests (mocked API)
# ---------------------------------------------------------------------------

class TestExtractStyle:
    def test_missing_api_key_error(self):
        with pytest.raises(ValueError, match="API key is required"):
            extract_style(b"fake", ".png", "anthropic", "")

    def test_anthropic_routing(self, sample_ai_result):
        png_bytes = b'\x89PNG' + b'\x00' * 50
        with patch("matplotly._ai_extract._call_anthropic",
                    return_value=sample_ai_result) as mock_call:
            result = extract_style(png_bytes, ".png", "anthropic", "sk-test")

        assert result["plot_type"] == "line"
        mock_call.assert_called_once()

    def test_openai_routing(self, sample_ai_result):
        png_bytes = b'\x89PNG' + b'\x00' * 50
        with patch("matplotly._ai_extract._call_openai",
                    return_value=sample_ai_result) as mock_call:
            result = extract_style(png_bytes, ".png", "openai", "sk-test")

        assert result["plot_type"] == "line"
        mock_call.assert_called_once()

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            extract_style(b"fake", ".png", "google", "key")
