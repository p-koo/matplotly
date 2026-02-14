Editor Guide
============

When you call ``matplotly(fig)``, an interactive editor appears inline in your
notebook. This page explains each part of the UI.

Layout
------

The editor has three areas:

.. code-block:: text

   ┌───────────────────────────────────────────────┐
   │ Toolbar:  [Undo] [Redo] [Save] [Apply] [Close]│
   ├──────────────────┬────────────────────────────┤
   │  Sidebar (380px) │  Canvas                    │
   │                  │  (live PNG preview)         │
   │  ▸ Global        │                            │
   │  ▸ Series 1      │                            │
   │  ▸ Series 2      │                            │
   │  ▸ Legend         │                            │
   │  ▸ Profiles       │                            │
   └──────────────────┴────────────────────────────┘

Toolbar
-------

The toolbar sits above the sidebar and canvas:

**Undo / Redo**
   Every property change is recorded on a command stack (up to 100 entries).
   Click Undo to revert the last change and Redo to reapply it.

**Save**
   Opens a dialog where you can export the current figure as an image file.
   Choose the format (PNG, PDF, SVG, etc.), DPI, and download location.

**Apply**
   Generates reproducible Python code for all your edits and replaces the
   current notebook cell with the code. The code is also copied to your
   clipboard. See :doc:`code_export` for details.

**Close (X)**
   Closes the editor. If you haven't applied, all changes are reverted and
   the original figure is restored.

Sidebar panels
--------------

The sidebar contains collapsible sections. Click a section header to
expand or collapse it.

Global panel
~~~~~~~~~~~~

Controls that affect the entire figure:

- **Figure size** — width and height sliders.
- **Labels** — font family dropdown, title / label / tick font sizes, bold
  toggles, and padding sliders for the title, x-label, and y-label.
- **Axes & Grid** — spine visibility (top, right, bottom, left), spine width,
  tick direction (in / out / inout), tick length and width, tick spacing for
  each axis, axis scale (linear / log), and grid settings (on/off, alpha,
  width, style).
- **Legend** — show/hide toggle, frame toggle, font size, position dropdown,
  and number of columns.
- **Colormap** — palette selector that recolors all series at once (visible
  when the figure contains colored artists).
- **Preferred Defaults** — one-click button that applies a publication-ready
  style: Arial font at 10 pt, no top/right spines, legend off, tab10
  colormap, and 75% figure scale.

For multi-subplot figures the global panel shows only figure size and font
settings. Axes, legend, and grid controls move into per-subplot panels.

Per-series panels
~~~~~~~~~~~~~~~~~

Each detected series (line, scatter collection, bar group, etc.) gets its own
collapsible panel with a colored dot indicator and the series label. Controls
vary by plot type — see :doc:`plot_types` for the full list.

Shared controls
~~~~~~~~~~~~~~~

Some plot types have shared controls that apply across all series of that type
within a subplot:

- **Histograms** — bin count, histogram type, mode, cumulative, orientation.
- **Bar charts** — bar width, gap, orientation, tick labels.
- **Distribution plots** — display mode (box / violin / jitter and combinations),
  box and violin widths, gap, orientation, tick labels.

Legend panel
~~~~~~~~~~~~

Below the series panels, the legend section lets you reorder legend entries
and toggle individual items on/off.

Profiles panel
~~~~~~~~~~~~~~

At the bottom of the sidebar, the Profiles section lets you save and load
style profiles. See :doc:`profiles`.

Canvas
------

The right side of the editor shows a PNG rendering of the figure. It updates
automatically whenever you change a setting in the sidebar. Rendering is
throttled (minimum 80 ms between redraws) to keep the UI responsive.

Editing workflow
----------------

1. Launch the editor with ``matplotly(fig)``.
2. Expand the Global panel to adjust fonts, axes, and legend.
3. Expand individual series panels to tweak colors, line styles, markers, etc.
4. Use Undo/Redo freely — every change is tracked.
5. Optionally save a style profile for reuse (see :doc:`profiles`).
6. Click **Apply** to export the code and close the editor, or **Close** to
   discard all changes.
