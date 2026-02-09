"""Sphinx configuration for matplotly docs."""

project = "matplotly"
copyright = "2025, Peter Koo"
author = "Peter Koo"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# MyST settings
myst_enable_extensions = ["colon_fence"]

# nbsphinx settings
nbsphinx_execute = "never"
