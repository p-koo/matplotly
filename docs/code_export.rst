Code Export
===========

When you click **Apply** in the editor toolbar, matplotly generates Python code
that reproduces all your styling changes and writes it back into your notebook
cell.

What gets exported
------------------

The generated code covers every property you changed in the editor:

- **Per-series styling** — colors, linewidth, linestyle, markers, marker size,
  alpha, edge colors, edge width, hatch patterns, fill settings.
- **Axes settings** — title, x/y labels, axis limits, axis scales, spine
  visibility and width, tick direction / length / width / spacing.
- **Legend** — visibility, position, frame, font size, columns.
- **Grid** — on/off, alpha, width, style.
- **Figure size** — width and height.
- **Distribution plots** — box, violin, and jitter configurations are
  reconstructed using ``ax.bxp()`` with compact statistics.
- **Heatmaps** — colormap and colorbar settings.

How the code is structured
--------------------------

The generated code appends styling calls after your original plot code. A
typical output looks like:

.. code-block:: python

   # -- original plot code --
   fig, ax = plt.subplots()
   ax.plot(x, y, label="data")

   # -- matplotly styling --
   ax = fig.get_axes()[0]
   _lines = [l for l in ax.lines if not l.get_label().startswith('_')]
   _lines[0].set_color('#1f77b4')
   _lines[0].set_linewidth(2.0)

   ax.set_title('My Plot', fontsize=14, fontweight='bold')
   ax.set_xlabel('X', fontsize=12)
   ax.spines['top'].set_visible(False)
   ax.spines['right'].set_visible(False)

For multi-axes figures the code addresses each axes by index
(``fig.get_axes()[0]``, ``fig.get_axes()[1]``, etc.).

Where the code goes
-------------------

**Cell replacement**
   By default, Apply replaces the current Jupyter cell contents with the
   combined original code plus the generated styling code. This works in
   both JupyterLab and classic Notebook.

**Clipboard**
   The code is also copied to the clipboard via a JavaScript fallback, so you
   can paste it elsewhere if needed.

After Apply, the editor closes and the cell contains a standalone, reproducible
script — no matplotly import required to run it.
