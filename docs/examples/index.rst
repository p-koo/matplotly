Examples
========

matplotly supports a wide range of matplotlib plot types. Each demo notebook
in the repository shows how to use the editor with a specific plot type.
See :doc:`/plot_types` for a detailed reference of available controls.

Supported plot types
--------------------

**Line plots** — ``ax.plot()``
   Color, linewidth, linestyle, marker style, marker size, alpha. Each line
   gets its own collapsible panel in the sidebar.

**Scatter plots** — ``ax.scatter()``
   Face color, edge color, marker style, marker size, alpha. Edge color can
   auto-sync with face color or be set independently.

**Bar charts** — ``ax.bar()`` / ``ax.barh()``
   Per-bar color, edge color, edge width, alpha, hatch pattern. Shared controls
   for bar width, gap, orientation, and tick labels.

**Histograms** — ``ax.hist()``
   Per-histogram color, edge color, alpha, hatch. Shared controls for bin count,
   histogram type (bar/step/stepfilled), mode (count/frequency/density),
   cumulative toggle, and orientation.

**Box plots** — ``ax.boxplot()``
   Box color, linewidth, hatch, notch toggle. Separate controls for median line,
   whisker/cap styling, and flier (outlier) markers. Display mode switching
   between box, violin, jitter, and combinations.

**Violin plots** — ``ax.violinplot()``
   Same distribution controls as box plots. Switch between display modes
   (box, violin, jitter) and combinations.

**Error bars** — ``ax.errorbar()``
   Line color, linewidth, linestyle, marker styling. Separate controls for
   cap size and cap thickness.

**Heatmaps** — ``ax.imshow()`` / ``ax.pcolormesh()``
   Colormap selection and colorbar controls.

**Fill regions** — ``ax.fill_between()`` / ``ax.fill_betweenx()``
   Fill color, alpha, hatch pattern.

**Marginal plots** — joint + marginal distributions
   Scatter plot with marginal histograms on top and right axes. Histogram
   styling syncs with the scatter color by default.

See the ``demo_*.ipynb`` notebooks in the repository root for interactive
walkthroughs of each type.
