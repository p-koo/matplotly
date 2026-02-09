"""Shared color utilities used by LinePanel, ScatterPanel, and GlobalPanel."""
from __future__ import annotations

import ipywidgets as widgets
import matplotlib
from matplotlib.colors import ListedColormap, to_hex

_COLORMAPS = [
    # Qualitative
    "tab10", "tab20", "Set1", "Set2", "Set3", "Paired",
    "Dark2", "Accent", "Pastel1", "Pastel2",
    # Diverging
    "bwr", "coolwarm", "RdBu", "RdYlBu", "RdYlGn",
    "PiYG", "PRGn", "seismic",
    # Sequential
    "viridis", "plasma", "inferno", "magma", "cividis",
    "Blues", "Reds", "Greens", "Oranges", "Purples",
    "YlOrRd", "YlGnBu", "BuGn",
    # Cyclic
    "twilight", "hsv",
]

_DW = "48px"    # uniform description_width
_NW = "50px"    # uniform number-edit width
_SN = {"description_width": _DW}


def _slider_num(slider, desc_width=None):
    """Slider (no readout) + linked number edit box (2 dp)."""
    slider.readout = False
    slider.style = {"description_width": desc_width or _DW}
    slider.layout.flex = "1 1 auto"
    if isinstance(slider, widgets.IntSlider):
        num = widgets.IntText(value=slider.value,
                              layout=widgets.Layout(width=_NW))
    else:
        num = widgets.BoundedFloatText(
            value=round(slider.value, 2), step=slider.step,
            min=slider.min, max=slider.max,
            layout=widgets.Layout(width=_NW))
    widgets.link((slider, "value"), (num, "value"))
    return widgets.HBox([slider, num],
                        layout=widgets.Layout(width="80%"))


def _cmap_color(cmap, i, n):
    """Sample color i of n from a colormap.

    For small qualitative colormaps (tab10, Set1, etc. with N <= 20), returns
    sequential discrete colors (1st, 2nd, 3rd...).  For everything else
    (continuous or large-N listed colormaps like viridis with N=256),
    interpolates evenly across the full [0,1] range.
    """
    if isinstance(cmap, ListedColormap) and cmap.N <= 20:
        return cmap(i % cmap.N)
    return cmap(i / max(n - 1, 1))


def _get_palette_colors(cmap_name, n=10):
    """Get n hex colors from a colormap."""
    try:
        cmap = matplotlib.colormaps.get_cmap(cmap_name)
    except Exception:
        cmap = matplotlib.colormaps.get_cmap("tab10")
    return [to_hex(_cmap_color(cmap, i, n)) for i in range(n)]


def _make_color_dot(hex_color):
    return (f'<div style="width:14px;height:14px;background:{hex_color};'
            f'border:1px solid #666;border-radius:2px;flex-shrink:0"></div>')


def cmap_color_btn(initial_color, on_change_fn, cmap_panel=None):
    """Compact color button; click toggles colormap swatch row + palette.

    *on_change_fn(hex_val)* is called when a new color is picked.
    Returns ``(color_btn, swatch_row)`` widgets for flexible layout.
    """
    color_btn = widgets.Button(
        layout=widgets.Layout(width='28px', height='24px',
                              padding='0', min_width='28px'),
        tooltip="Click to choose color")
    color_btn.style.button_color = initial_color

    swatch_btns = []
    for _ in range(10):
        b = widgets.Button(
            layout=widgets.Layout(width="18px", height="16px",
                                  padding="0", margin="1px",
                                  min_width="18px"))
        swatch_btns.append(b)

    palette_btn = widgets.Button(
        icon="paint-brush", tooltip="Custom color...",
        layout=widgets.Layout(width="18px", height="16px",
                              padding="0", min_width="18px",
                              margin="1px"))
    palette_btn.style.button_color = "#e8e8e8"
    palette_btn.add_class("pb-swatch-btn")

    _pk_cls = f"pb-txtpk-{id(color_btn)}"
    picker = widgets.ColorPicker(
        value=initial_color, concise=True,
        layout=widgets.Layout(width="1px", height="1px",
                              overflow="hidden", padding="0",
                              margin="0", border="0"))
    picker.add_class(_pk_cls)

    _js = widgets.Output(
        layout=widgets.Layout(height="0px", overflow="hidden"))

    swatch_row = widgets.HBox(
        swatch_btns + [palette_btn, picker, _js],
        layout=widgets.Layout(display='none', align_items='center',
                              gap='1px'))

    _upd = [False]

    def _refresh():
        cname = 'tab10'
        if cmap_panel and hasattr(cmap_panel, '_selected'):
            cname = cmap_panel._selected
        colors = _get_palette_colors(cname, 10)
        for i, btn in enumerate(swatch_btns):
            btn.style.button_color = colors[i]

    def _apply(hex_val):
        _upd[0] = True
        color_btn.style.button_color = hex_val
        picker.value = hex_val
        _upd[0] = False
        on_change_fn(hex_val)

    for b in swatch_btns:
        def _on_sw(btn, _b=b):
            _apply(_b.style.button_color)
        b.on_click(_on_sw)

    def _toggle(btn):
        if swatch_row.layout.display == 'none':
            _refresh()
            swatch_row.layout.display = ''
        else:
            swatch_row.layout.display = 'none'
    color_btn.on_click(_toggle)

    def _on_pal(b):
        with _js:
            _js.clear_output()
            from IPython.display import display as ipy_display, Javascript
            ipy_display(Javascript(
                "setTimeout(function(){"
                "var el=document.querySelector('.%s input[type=\"color\"]');"
                "if(el)el.click();"
                "},150);" % _pk_cls))
    palette_btn.on_click(_on_pal)

    def _on_pk(change):
        if _upd[0]:
            return
        _apply(change["new"])
    picker.observe(_on_pk, names="value")

    return color_btn, swatch_row


def _refresh_legend(ax):
    """Recreate the legend so it reflects current artist properties."""
    if ax is None:
        return
    leg = ax.get_legend()
    if leg is None:
        return
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    props = {}
    try:
        props['frameon'] = leg.get_frame().get_visible()
    except Exception:
        pass
    try:
        props['fontsize'] = leg._fontsize
    except Exception:
        pass
    try:
        props['ncol'] = getattr(leg, '_ncols', 1)
    except Exception:
        pass
    try:
        props['markerfirst'] = leg._markerfirst
    except Exception:
        pass
    try:
        props['handletextpad'] = leg.handletextpad
    except Exception:
        pass
    try:
        props['handleheight'] = leg.handleheight
    except Exception:
        pass
    try:
        props['loc'] = leg._loc
    except Exception:
        pass
    try:
        if hasattr(leg, '_bbox_to_anchor') and leg._bbox_to_anchor is not None:
            inv = ax.transAxes.inverted()
            bx, by = inv.transform(
                (leg._bbox_to_anchor.x0, leg._bbox_to_anchor.y0))
            props['bbox_to_anchor'] = (round(bx, 3), round(by, 3))
    except Exception:
        pass
    ax.legend(handles, labels, **props)
    # Preserve custom legend text colors
    if hasattr(ax, '_matplotly_leg_text_colors'):
        new_leg = ax.get_legend()
        if new_leg:
            for _i, _c in enumerate(ax._matplotly_leg_text_colors):
                if _i < len(new_leg.get_texts()):
                    new_leg.get_texts()[_i].set_color(_c)
