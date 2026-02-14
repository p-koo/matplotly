Plot Types Reference
====================

matplotly detects the plot type of each series in your figure and shows the
appropriate controls. This page lists every supported type and its available
settings.

Line plots — ``ax.plot()``
--------------------------

Each line gets a collapsible panel with:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Control
     - Description
   * - Name
     - Editable label (syncs with legend)
   * - Color
     - Color picker with palette dropdown
   * - Linewidth
     - Slider (0.1 – 10)
   * - Linestyle
     - Dropdown: solid, dashed, dotted, dash-dot
   * - Alpha
     - Opacity slider (0 – 1)
   * - Marker
     - Marker style dropdown (circle, square, triangle, etc.)
   * - Marker size
     - Slider

Scatter plots — ``ax.scatter()``
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Control
     - Description
   * - Name
     - Editable label
   * - Face color
     - Color picker with palette dropdown
   * - Edge color
     - Separate picker (auto-syncs with face color by default)
   * - Marker
     - Style dropdown
   * - Marker size
     - Slider
   * - Alpha
     - Opacity slider
   * - Edge width
     - Slider

Bar charts — ``ax.bar()`` / ``ax.barh()``
------------------------------------------

Per-bar-group controls:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Control
     - Description
   * - Name
     - Editable label
   * - Color
     - Fill color picker
   * - Edge color
     - Edge color picker
   * - Edge width
     - Slider
   * - Alpha
     - Opacity slider
   * - Hatch
     - Pattern dropdown (/, \\, \|, -, +, x, or none)

Shared controls (apply to all bars in the subplot):

- **Bar width** — slider
- **Gap** — spacing between groups
- **Orientation** — vertical or horizontal
- **Tick labels** — editable labels for each bar position

Histograms — ``ax.hist()``
---------------------------

Per-histogram controls: name, color, edge color, edge width, alpha, hatch.

Shared controls:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Control
     - Description
   * - Bins
     - Number of bins slider
   * - Histogram type
     - bar, step, or stepfilled
   * - Mode
     - count, frequency, or density
   * - Cumulative
     - Toggle
   * - Orientation
     - Vertical or horizontal
   * - Tick labels
     - Editable tick labels

Box plots — ``ax.boxplot()``
-----------------------------

Per-group controls:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Control
     - Description
   * - Name
     - Editable label
   * - Box color
     - Fill color
   * - Box linewidth
     - Slider
   * - Hatch
     - Pattern dropdown
   * - Notch
     - Toggle
   * - Median color
     - Color picker
   * - Median linewidth
     - Slider
   * - Whisker color
     - Color picker
   * - Whisker linewidth
     - Slider
   * - Whisker style
     - Linestyle dropdown
   * - Cap color
     - Color picker
   * - Cap linewidth
     - Slider
   * - Flier color
     - Outlier marker color
   * - Flier marker
     - Marker style dropdown
   * - Flier size
     - Slider
   * - Jitter alpha
     - Opacity for jitter points

Shared distribution controls (box, violin, and jitter):

- **Display mode** — box, violin, jitter, or combinations (box+violin,
  box+jitter, violin+jitter, box+violin+jitter)
- **Box width** / **Violin width** — sliders
- **Gap** — spacing between groups
- **Orientation** — vertical or horizontal
- **Tick labels** — editable

Violin plots — ``ax.violinplot()``
-----------------------------------

Violin plots share the same distribution panel as box plots. All the controls
listed under "Shared distribution controls" above apply, plus the per-group
controls for box styling, median, whiskers, caps, and fliers.

Error bars — ``ax.errorbar()``
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Control
     - Description
   * - Name
     - Editable label
   * - Line color
     - Color picker
   * - Linewidth
     - Slider
   * - Linestyle
     - Dropdown
   * - Marker
     - Style dropdown
   * - Marker size
     - Slider
   * - Alpha
     - Opacity slider
   * - Cap size
     - Slider
   * - Cap thickness
     - Slider

Heatmaps — ``ax.imshow()`` / ``ax.pcolormesh()``
--------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Control
     - Description
   * - Colormap
     - Dropdown of matplotlib colormaps
   * - Colorbar
     - Colorbar display settings

Fill regions — ``ax.fill_between()`` / ``ax.fill_betweenx()``
--------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Control
     - Description
   * - Color
     - Fill color picker
   * - Alpha
     - Opacity slider
   * - Hatch
     - Pattern dropdown

Marginal plots
--------------

When a figure has a scatter plot with marginal histograms (e.g. joint plot),
matplotly detects the marginal axes and adds separate histogram controls for
the top and right margins. Marginal histogram color syncs with the scatter
color by default.

Multi-subplot figures
---------------------

For figures with multiple subplots, matplotly shows:

- A simplified **Global** panel (figure size and fonts only).
- A **per-subplot** panel for each axes, containing axes labels, limits,
  scales, spines, ticks, grid, and legend controls.
- **Per-series** panels grouped under their respective subplot.
