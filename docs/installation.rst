Installation
============

Install from PyPI
-----------------

.. code-block:: bash

   pip install matplotly

Backend setup
-------------

matplotly requires the ``ipympl`` interactive backend. Enable it at the top of
your notebook:

.. code-block:: python

   %matplotlib widget

This works in JupyterLab, classic Notebook, and VS Code notebooks.

Development install
-------------------

.. code-block:: bash

   git clone https://github.com/p-koo/matplotly.git
   cd matplotly
   pip install -e ".[dev]"
