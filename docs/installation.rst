Installation
============

Install from PyPI
-----------------

For basic editing features:

.. code-block:: bash

   pip install matplotly

To include AI style extraction (recommended):

.. code-block:: bash

   pip install "matplotly[ai]"

Base vs full install
~~~~~~~~~~~~~~~~~~~~

The base install (``pip install matplotly``) includes:

- ``ipywidgets`` >= 8.0.0
- ``ipympl`` >= 0.9.0
- ``matplotlib`` >= 3.8.0
- ``numpy`` >= 1.24.0

The ``[ai]`` extra adds the dependencies needed for AI style extraction:

- ``anthropic`` >= 0.39.0 — Claude API client
- ``openai`` >= 1.0.0 — OpenAI API client
- ``Pillow`` >= 9.0 — image processing
- ``PyMuPDF`` >= 1.23.0 — PDF-to-PNG conversion

Backend setup
-------------

matplotly requires the ``ipympl`` interactive backend. Enable it at the top of
your notebook:

.. code-block:: python

   %matplotlib widget

This works in JupyterLab, classic Notebook, and VS Code notebooks.

Google Colab
~~~~~~~~~~~~

In Google Colab, enable the custom widget manager before importing matplotly.
Install in one cell and restart the runtime, then run:

.. code-block:: python

   from google.colab import output
   output.enable_custom_widget_manager()

   %matplotlib widget

   from matplotly import matplotly

Development install
-------------------

.. code-block:: bash

   git clone https://github.com/p-koo/matplotly.git
   cd matplotly
   pip install -e ".[dev]"
