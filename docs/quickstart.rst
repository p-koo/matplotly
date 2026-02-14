Quick Start
===========

matplotly provides three ways to open the interactive editor.

1. Pass a figure directly
-------------------------

.. code-block:: python

   %matplotlib widget
   import matplotlib.pyplot as plt
   from matplotly import matplotly

   fig, ax = plt.subplots()
   ax.plot([1, 2, 3], [1, 4, 9])
   matplotly(fig)

2. Decorator mode
-----------------

.. code-block:: python

   @matplotly
   def my_plot():
       plt.plot([1, 2, 3], [1, 4, 9])

   my_plot()

The decorator calls your function, captures ``plt.gcf()``, and opens the editor.

3. Context manager
------------------

.. code-block:: python

   with matplotly() as pb:
       fig, ax = plt.subplots()
       ax.plot([1, 2, 3], [1, 4, 9])

On exit, the context manager captures the current figure and launches the editor.

What happens next
-----------------

After calling ``matplotly()``, the editor appears inline in your notebook. You'll
see:

- A **toolbar** at the top with Undo, Redo, Save, Apply, and Close buttons.
- A **sidebar** on the left with collapsible panels for global settings (fonts,
  axes, grid, legend) and per-series controls (colors, line styles, markers).
- A **canvas** on the right showing a live preview of your figure that updates as
  you change settings.

Edit any property in the sidebar and the canvas refreshes immediately. When you're
done, click **Apply** to export the styling as reproducible Python code that
replaces the current cell. See :doc:`editor` for a full walkthrough.
