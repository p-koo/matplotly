AI Style Extraction
===================

matplotly can extract styling from a reference image (a screenshot of a figure
you'd like to match) and apply it to your plot. This feature requires the
``[ai]`` extra:

.. code-block:: bash

   pip install "matplotly[ai]"

Setup
-----

You need an API key for at least one supported provider:

- **Anthropic** — for Claude models
- **OpenAI** — for GPT models

Set your key as an environment variable before launching Jupyter:

.. code-block:: bash

   export ANTHROPIC_API_KEY="sk-ant-..."
   # or
   export OPENAI_API_KEY="sk-..."

The key is read from the environment — **no secrets are written to disk**.
You can also paste a key directly into the editor's API key field; it will
be kept in memory for the current session only.

Supported models
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Provider
     - Models
   * - Anthropic
     - Claude Sonnet 4, Claude Haiku 3.5
   * - OpenAI
     - GPT-5.2, GPT-5.1, GPT-5, GPT-5 mini, GPT-5 nano, GPT-4o, GPT-4o mini

How it works
------------

The extraction uses a **two-pass vision pipeline**:

**Pass 1 — Extraction**
   The vision model receives your reference image and returns structured JSON
   describing every visual property it can identify:

   - **Global settings** — font family and sizes, spine visibility, tick
     direction and spacing, grid style, legend position, background color,
     colormap, axis scales.
   - **Per-series properties** — colors (hex), line styles, markers, edge
     colors, hatches, fill, alpha.

   The model uses a relative-size estimation method (tick labels as baseline)
   to infer font sizes and uses a checklist to ensure completeness.

**Pass 2 — Verification**
   The model re-examines the reference image against the extracted parameters
   and outputs only corrections. This second pass catches errors and improves
   accuracy.

After both passes, matplotly merges the results and applies them to your
figure.

Supported image formats
-----------------------

- **Native** — PNG, JPG, GIF, WebP (sent directly to the vision API).
- **Converted** — PDF and TIFF files are converted to PNG before sending
  (PDF via PyMuPDF, TIFF via Pillow).

Saving extracted styles
-----------------------

After AI extraction, you can save the result as a style profile so you can
reapply it to other figures without calling the API again. See :doc:`profiles`.
