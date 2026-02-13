# matplotly

[![PyPI version](https://img.shields.io/pypi/v/matplotly.svg)](https://pypi.org/project/matplotly/)
[![Python versions](https://img.shields.io/pypi/pyversions/matplotly.svg)](https://pypi.org/project/matplotly/)
[![License](https://img.shields.io/pypi/l/matplotly.svg)](https://github.com/p-koo/matplotly/blob/main/LICENSE)
[![Docs](https://readthedocs.org/projects/matplotly/badge/?version=latest)](https://matplotly.readthedocs.io)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/p-koo/matplotly/blob/main/examples/matplotly_demo.ipynb)

**Jupyter-native interactive matplotlib figure editor.**

Edit any matplotlib figure interactively — styles, colors, labels, legends — then export
reproducible Python code. No separate GUI; everything runs inline in your notebook.

## Features

- **10+ plot types**: line, scatter, bar, histogram, box, violin, errorbar, heatmap, fill, marginal
- **AI style extraction**: upload a reference plot image and automatically match its style (fonts, colors, spines, ticks, legend, etc.) using Claude or GPT vision models
- **Code generation**: export a standalone Python script that recreates your styled figure
- **Undo / redo**: full command history with keyboard shortcuts
- **Style profiles**: save and load reusable figure styles
- **Subplot support**: edit multi-panel figures with per-axes controls

## Installation

```bash
pip install matplotly[ai]
```

This includes everything: the interactive editor plus AI-powered style extraction via Claude or GPT vision models. Requires an Anthropic or OpenAI API key for the AI features (configured in the UI).

For a lightweight install **without** AI dependencies (`anthropic`, `openai`, `Pillow`, `PyMuPDF`):

```bash
pip install matplotly
```

The base install gives you the full interactive editor — manual controls for styles, colors, labels, legends, code export, undo/redo, and style profiles. The only difference is you won't have the AI style extraction tab.

Both versions require a Jupyter environment with the `ipympl` backend:

```python
%matplotlib widget
```

## Quick Start

```python
%matplotlib widget
import matplotlib.pyplot as plt
from matplotly import matplotly

# 1. Pass a figure directly
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
matplotly(fig)

# 2. Decorator mode
@matplotly
def my_plot():
    plt.plot([1, 2, 3], [1, 4, 9])

my_plot()

# 3. Context manager
with matplotly() as pb:
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
```

## Supported Plot Types

| Type | Description |
|------|-------------|
| Line | `ax.plot()` |
| Scatter | `ax.scatter()` |
| Bar | `ax.bar()` / `ax.barh()` |
| Histogram | `ax.hist()` |
| Box | `ax.boxplot()` |
| Violin | `ax.violinplot()` |
| Errorbar | `ax.errorbar()` |
| Heatmap | `ax.imshow()` / `ax.pcolormesh()` |
| Fill | `ax.fill_between()` / `ax.fill_betweenx()` |
| Marginal | Joint + marginal distribution plots |

## AI Style Extraction

Have a plot style you want to match? Upload a reference image and let a vision model do the work.

**How it works:** You upload a reference plot image (PNG, JPEG, PDF, TIFF, or WebP) and select a vision model. The system runs a two-pass agentic pipeline:

1. **Pass 1 — Extraction:** The vision model analyzes the reference image and extracts a structured JSON of style parameters: font sizes and families, spine visibility, tick direction/length, grid settings, legend placement, background color, and per-series properties (colors, line widths, markers, hatches, etc.).
2. **Pass 2 — Verification:** The same model re-examines the reference image against the extracted parameters, checking each one for accuracy. Only fields that need correction are updated.

The corrected styles are then applied to your figure automatically — updating both the matplotlib artists and the UI controls so everything stays in sync.

**Supported providers:**
- **Anthropic:** Claude Sonnet 4, Claude Haiku 3.5
- **OpenAI:** GPT-5, GPT-4o, and variants

**Saving and reusing styles:** After extraction, you can download the result as a JSON profile. Load it later to apply the same style to any figure — no API call needed.

**API keys:** Enter your Anthropic or OpenAI key in the UI. Keys are saved locally to `~/.matplotly/config.json` so you only need to enter them once.

## Documentation

Full documentation is available at [matplotly.readthedocs.io](https://matplotly.readthedocs.io).

## Contributing

Contributions are welcome! Please open an issue or pull request on
[GitHub](https://github.com/p-koo/matplotly).

## License

MIT — see [LICENSE](LICENSE) for details.
