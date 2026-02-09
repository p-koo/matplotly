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
- **Code generation**: export a standalone Python script that recreates your styled figure
- **Undo / redo**: full command history with keyboard shortcuts
- **Style profiles**: save and load reusable figure styles
- **Subplot support**: edit multi-panel figures with per-axes controls

## Installation

```bash
pip install matplotly
```

Requires a Jupyter environment with the `ipympl` backend enabled:

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

## Documentation

Full documentation is available at [matplotly.readthedocs.io](https://matplotly.readthedocs.io).

## Contributing

Contributions are welcome! Please open an issue or pull request on
[GitHub](https://github.com/p-koo/matplotly).

## License

MIT — see [LICENSE](LICENSE) for details.
