Style Profiles
==============

Style profiles let you save a figure's styling and reapply it to other figures
with one click.

Where profiles are stored
-------------------------

Profiles are saved as JSON files in ``~/.matplotly/profiles/``. Each profile
is a separate file named after the profile (e.g. ``journal_style.json``).

Saving a profile
----------------

1. Style your figure in the editor (manually or via :doc:`AI extraction <ai>`).
2. Scroll to the **Profiles** section at the bottom of the sidebar.
3. Enter a name and click **Save**.
4. Optionally click **Download** to get a copy of the JSON file.

Loading a profile
-----------------

1. Open any figure in the editor.
2. In the **Profiles** section, select a saved profile from the dropdown.
3. The profile's settings are applied to the figure immediately.

What a profile contains
-----------------------

A profile captures the global settings from the editor's Global panel:

- Font family, title / label / tick font sizes, bold flags
- Title, x-label, and y-label padding
- Spine visibility (top, right, bottom, left) and width
- Tick direction, length, width, and spacing
- Axis scales (linear / log)
- Grid settings (on/off, alpha, width, style)
- Legend settings (show, frame, font size, position, columns)
- Colormap selection

Per-series properties (individual colors, markers, etc.) are **not** included
in profiles — they depend on the specific plot being edited.

Sharing profiles
----------------

To share a profile with a collaborator:

1. Copy the JSON file from ``~/.matplotly/profiles/`` or use the Download
   button in the editor.
2. The recipient places the file in their own ``~/.matplotly/profiles/``
   directory.

The profile appears in the dropdown the next time the editor is opened.

Profile JSON format
-------------------

A profile is a flat JSON object. Here is an abbreviated example:

.. code-block:: json

   {
     "font_family": "Arial",
     "title_size": 14,
     "label_size": 12,
     "tick_size": 10,
     "title_bold": false,
     "label_bold": false,
     "spine_top": false,
     "spine_right": false,
     "spine_bottom": true,
     "spine_left": true,
     "spine_width": 1.0,
     "tick_direction": "out",
     "tick_length": 5,
     "tick_width": 1.0,
     "grid_on": false,
     "legend_show": true,
     "legend_frame": false,
     "legend_fontsize": 10,
     "legend_position": "best",
     "colormap": "tab10"
   }
