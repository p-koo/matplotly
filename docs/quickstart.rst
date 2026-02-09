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
