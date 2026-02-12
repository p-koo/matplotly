"""Figure introspection — detect and classify all artists in a matplotlib figure."""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex
from matplotlib.container import BarContainer, ErrorbarContainer
from matplotlib.collections import LineCollection, PathCollection, PolyCollection, QuadMesh
from matplotlib.image import AxesImage
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Rectangle

from ._types import ArtistGroup, PlotType


# ---------------------------------------------------------------------------
# Boxplot helpers
# ---------------------------------------------------------------------------

def _get_patch_data_bounds(patch):
    """Get patch bounds in data coordinates."""
    path = patch.get_path()
    trans = patch.get_patch_transform()
    verts = trans.transform(path.vertices)
    x_min, y_min = verts[:, 0].min(), verts[:, 1].min()
    x_max, y_max = verts[:, 0].max(), verts[:, 1].max()
    return x_min, y_min, x_max, y_max


def _extract_box_stats_patch_mode(box_patches, internal_lines, n_boxes,
                                  horizontal=False):
    """Extract per-box stats when patch_artist=True (boxes are Patches)."""
    box_stats = []

    # For horizontal boxplots, swap x↔y so the existing vertical logic
    # works unchanged.  We never mutate originals — just read swapped.
    def _bounds(patch):
        bx0, by0, bx1, by1 = _get_patch_data_bounds(patch)
        return (by0, bx0, by1, bx1) if horizontal else (bx0, by0, bx1, by1)

    def _line_xy(line):
        xd, yd = line.get_xdata(), line.get_ydata()
        return (yd, xd) if horizontal else (xd, yd)

    # Sort patches by x-center in data coordinates (position axis)
    sorted_patches = sorted(
        box_patches,
        key=lambda p: (_bounds(p)[0] + _bounds(p)[2]) / 2)

    # Classify lines: medians vs whiskers vs caps vs fliers
    # Medians are short horizontal lines inside boxes.
    # Whiskers are vertical lines extending from box edges.
    # Caps are short horizontal lines at whisker ends.
    # Fliers have markers.
    medians, whiskers, caps, fliers = [], [], [], []
    for line in internal_lines:
        xd, yd = _line_xy(line)
        marker = line.get_marker()
        if marker and marker not in ("None", "none", "", " "):
            fliers.append(line)
        elif len(xd) == 2 and len(yd) == 2:
            dx = abs(xd[1] - xd[0])
            dy = abs(yd[1] - yd[0])
            if dy < 1e-10 and dx > 1e-10:
                # Horizontal line — could be median or cap
                # We'll classify after sorting
                medians.append(line)  # tentative
            elif dx < 1e-10 and dy > 1e-10:
                whiskers.append(line)
            else:
                whiskers.append(line)
        else:
            whiskers.append(line)

    # Separate medians from caps by checking if they're inside box bounds
    actual_medians = []
    actual_caps = []
    for line in medians:
        xd, yd = _line_xy(line)
        y_val = yd[0]
        x_center = (xd[0] + xd[1]) / 2
        is_median = False
        for patch in sorted_patches:
            bx0, by0, bx1, by1 = _bounds(patch)
            if (bx0 - 0.5 <= x_center <= bx1 + 0.5
                    and by0 <= y_val <= by1):
                is_median = True
                break
        if is_median:
            actual_medians.append(line)
        else:
            actual_caps.append(line)

    for i, patch in enumerate(sorted_patches):
        bx0, by0, bx1, by1 = _bounds(patch)
        x_center = (bx0 + bx1) / 2
        box_half_w = (bx1 - bx0) / 2
        q1 = by0
        q3 = by1

        # Find median for this box
        median = (q1 + q3) / 2
        for m_line in actual_medians:
            mxd, myd = _line_xy(m_line)
            mx = (mxd[0] + mxd[1]) / 2
            if abs(mx - x_center) < box_half_w + 0.5:
                median = myd[0]
                break

        # Find whiskers for this box
        whislo, whishi = q1, q3
        for w_line in whiskers:
            wx, wy = _line_xy(w_line)
            w_x_center = (wx[0] + wx[1]) / 2
            if abs(w_x_center - x_center) < box_half_w + 0.5:
                wmin, wmax = min(wy), max(wy)
                if wmax <= q1 + 0.01:
                    whislo = min(whislo, wmin)
                elif wmin >= q3 - 0.01:
                    whishi = max(whishi, wmax)
                else:
                    if wmin < q1:
                        whislo = min(whislo, wmin)
                    if wmax > q3:
                        whishi = max(whishi, wmax)

        # Find caps for this box (refine whisker ends)
        for c_line in actual_caps:
            cxd, cyd = _line_xy(c_line)
            cx = (cxd[0] + cxd[1]) / 2
            if abs(cx - x_center) < box_half_w + 0.5:
                cy = cyd[0]
                if cy < q1:
                    whislo = min(whislo, cy)
                elif cy > q3:
                    whishi = max(whishi, cy)

        # Find fliers for this box
        box_fliers = []
        for f_line in fliers:
            fxd, fyd = _line_xy(f_line)
            if len(fxd) > 0 and abs(np.mean(fxd) - x_center) < box_half_w + 0.5:
                box_fliers.extend(np.asarray(fyd).tolist())

        box_stats.append({
            "q1": float(q1), "q3": float(q3), "median": float(median),
            "whislo": float(whislo), "whishi": float(whishi),
            "fliers": box_fliers,
            "position": float(x_center),
            "mean": None,
        })

    return box_stats


def _extract_box_stats_line_mode(internal_lines, per_box, n_boxes,
                                 horizontal=False):
    """Extract per-box stats when patch_artist=False (boxes are Line2D)."""

    # For horizontal boxplots, swap x↔y so the existing vertical logic
    # works unchanged.
    def _line_xy(line):
        xd, yd = np.array(line.get_xdata()), np.array(line.get_ydata())
        return (yd, xd) if horizontal else (xd, yd)

    box_stats = []
    for i in range(n_boxes):
        chunk = internal_lines[i * per_box:(i + 1) * per_box]
        # Typical order: box, med, whi_lo, whi_hi, cap_lo, cap_hi, flier
        # But order can vary — classify by geometry
        horizontals = []
        verticals = []
        marker_lines = []

        for line in chunk:
            xd, yd = _line_xy(line)
            marker = line.get_marker()
            if marker and marker not in ("None", "none", "", " "):
                marker_lines.append(line)
                continue
            if len(xd) < 2:
                continue
            dx = abs(xd[-1] - xd[0]) if len(xd) >= 2 else 0
            dy = abs(yd[-1] - yd[0]) if len(yd) >= 2 else 0
            if len(xd) >= 4:
                # Box rectangle (4 or 5 points forming a rectangle)
                horizontals.insert(0, line)  # box first
            elif dy < 1e-10 and dx > 1e-10:
                horizontals.append(line)
            elif dx < 1e-10 and dy > 1e-10:
                verticals.append(line)
            else:
                horizontals.append(line)

        # The box line has 5 points (closed rectangle)
        box_line = None
        other_h = []
        for h in horizontals:
            xd, _ = _line_xy(h)
            if len(xd) >= 4:
                box_line = h
            else:
                other_h.append(h)

        if box_line is not None:
            xd, yd = _line_xy(box_line)
            q1 = float(np.min(yd))
            q3 = float(np.max(yd))
            x_center = float((np.min(xd) + np.max(xd)) / 2)
        else:
            # Fallback: use mean of horizontal lines
            q1, q3, x_center = 0, 1, float(i + 1)
            if other_h:
                all_y = [_line_xy(h)[1][0] for h in other_h]
                q1 = min(all_y)
                q3 = max(all_y)
                x_center = float(np.mean(_line_xy(other_h[0])[0]))

        # Median: horizontal line inside box
        median = (q1 + q3) / 2
        for h in other_h:
            _, hyd = _line_xy(h)
            hy = hyd[0]
            if q1 <= hy <= q3:
                median = float(hy)
                break

        # Whiskers
        whislo, whishi = q1, q3
        for v in verticals:
            _, vy = _line_xy(v)
            vmin, vmax = float(min(vy)), float(max(vy))
            if vmax <= q1 + 0.01:
                whislo = min(whislo, vmin)
            elif vmin >= q3 - 0.01:
                whishi = max(whishi, vmax)
            else:
                whislo = min(whislo, vmin)
                whishi = max(whishi, vmax)

        # Caps (horizontal lines outside box)
        for h in other_h:
            _, hyd = _line_xy(h)
            hy = hyd[0]
            if hy < q1:
                whislo = min(whislo, float(hy))
            elif hy > q3:
                whishi = max(whishi, float(hy))

        # Fliers
        box_fliers = []
        for m in marker_lines:
            _, myd = _line_xy(m)
            box_fliers.extend(myd.tolist())

        box_stats.append({
            "q1": q1, "q3": q3, "median": median,
            "whislo": whislo, "whishi": whishi,
            "fliers": box_fliers,
            "position": x_center,
            "mean": None,
        })

    return box_stats


def _split_boxplot_groups(box_stats, internal_lines, box_patches,
                          all_positions, n_boxes, per_box,
                          horizontal=False):
    """Split detected boxes into separate groups by position clustering.

    Returns list of (stats_list, lines_list, patches_list) tuples.
    Multiple ax.boxplot() calls at interleaved positions get separated into
    distinct groups.
    """
    if n_boxes <= 1:
        return [(box_stats, list(internal_lines), list(box_patches))]

    if per_box is not None:
        # Line mode: boxes are in creation order.
        # Each boxplot() call creates per_box lines sequentially.
        pos_arr = np.array(all_positions)

        best_split = None
        for n_grp in range(2, n_boxes + 1):
            if n_boxes % n_grp != 0:
                continue
            boxes_per_group = n_boxes // n_grp
            group_positions = []
            for g in range(n_grp):
                start = g * boxes_per_group
                gp = pos_arr[start:start + boxes_per_group]
                group_positions.append(gp)

            valid = True
            for g in range(n_grp):
                gp = sorted(group_positions[g])
                if len(gp) >= 2:
                    diffs = np.diff(gp)
                    if len(diffs) > 1 and np.std(diffs) > np.mean(diffs) * 0.5:
                        valid = False
                        break

            if valid and n_grp >= 2:
                ranges = [(min(gp), max(gp)) for gp in group_positions]
                overlapping = any(
                    ranges[a][0] < ranges[b][1] and ranges[b][0] < ranges[a][1]
                    for a in range(n_grp) for b in range(a + 1, n_grp))
                if overlapping:
                    best_split = n_grp
                    break

        if best_split and best_split > 1:
            boxes_per_group = n_boxes // best_split
            result = []
            for g in range(best_split):
                start = g * boxes_per_group
                end = start + boxes_per_group
                sg_stats = box_stats[start:end]
                sg_lines = internal_lines[start * per_box:end * per_box]
                sg_patches = []  # no patches in line mode
                result.append((sg_stats, list(sg_lines), sg_patches))
            return result

    elif box_patches:
        # Patch artist mode: box_stats are sorted by position.
        # Sort patches to match box_stats ordering.
        # For horizontal: position axis is y, so sort by y-center.
        if horizontal:
            sorted_patches = sorted(
                box_patches,
                key=lambda p: (_get_patch_data_bounds(p)[1]
                               + _get_patch_data_bounds(p)[3]) / 2)
        else:
            sorted_patches = sorted(
                box_patches,
                key=lambda p: (_get_patch_data_bounds(p)[0]
                               + _get_patch_data_bounds(p)[2]) / 2)

        # Before attempting split, check if all patches have the same
        # facecolor.  A single boxplot() call produces uniformly colored
        # boxes; interleaved calls from different groups differ.
        from matplotlib.colors import to_hex as _to_hex_split
        try:
            _fc_set = {_to_hex_split(bp.get_facecolor()) for bp in box_patches}
        except Exception:
            _fc_set = set()
        if len(_fc_set) <= 1:
            return [(box_stats, list(internal_lines), list(box_patches))]

        pos_arr = np.array(all_positions)  # already sorted by position

        best_split = None
        for n_grp in range(2, n_boxes + 1):
            if n_boxes % n_grp != 0:
                continue

            # Cycle-assign: sorted position at index k → group (k % n_grp)
            group_positions = []
            for g in range(n_grp):
                gp = pos_arr[g::n_grp]
                group_positions.append(gp)

            valid = True
            for g in range(n_grp):
                gp = group_positions[g]
                if len(gp) >= 2:
                    diffs = np.diff(gp)
                    if len(diffs) > 1 and np.std(diffs) > np.mean(diffs) * 0.5:
                        valid = False
                        break

            if valid and n_grp >= 2:
                ranges = [(float(min(gp)), float(max(gp)))
                          for gp in group_positions]
                overlapping = any(
                    ranges[a][0] < ranges[b][1] and ranges[b][0] < ranges[a][1]
                    for a in range(n_grp) for b in range(a + 1, n_grp))
                if overlapping:
                    best_split = n_grp
                    break

        if best_split and best_split > 1:
            result = []
            for g in range(best_split):
                indices = list(range(g, n_boxes, best_split))
                sg_stats = [box_stats[i] for i in indices]
                sg_patches_list = [sorted_patches[i] for i in indices]
                # All lines go to first group (all get cleared on first redraw)
                sg_lines = list(internal_lines) if g == 0 else []
                result.append((sg_stats, sg_lines, sg_patches_list))
            return result

    # No split detected — return single group
    return [(box_stats, list(internal_lines), list(box_patches))]


def _reconstruct_data_from_stats(stats, n_samples=100):
    """Approximate raw data from boxplot statistics."""
    q1, q3, med = stats['q1'], stats['q3'], stats['median']
    wlo, whi = stats['whislo'], stats['whishi']
    fliers = stats.get('fliers', [])

    n = max(n_samples - len(fliers), 20)
    q_each = n // 4
    remainder = n - 4 * q_each

    rng = np.random.RandomState(42)
    data = np.concatenate([
        rng.uniform(wlo, q1, q_each),
        rng.uniform(q1, med, q_each),
        rng.uniform(med, q3, q_each),
        rng.uniform(q3, whi, q_each + remainder),
        np.array(fliers, dtype=float),
    ])
    return data


def _reconstruct_data_from_violin(verts, n_samples=100):
    """Approximate raw data from violin KDE shape vertices."""
    # Violin shape: left half goes up, right half comes down
    # The y-axis represents the data values
    y_vals = verts[:, 1]
    x_vals = verts[:, 0]
    y_min, y_max = float(np.min(y_vals)), float(np.max(y_vals))

    # Sample y values weighted by the density (width at each y)
    n_steps = 200
    y_grid = np.linspace(y_min, y_max, n_steps)

    # Compute width (density) at each y by interpolation
    # Split into left and right halves
    mid_y = (y_min + y_max) / 2
    center_x = float(np.mean(x_vals))

    widths = np.zeros(n_steps)
    for j, y in enumerate(y_grid):
        # Find all x values at this y level
        xs_at_y = []
        for k in range(len(verts) - 1):
            y0, y1 = verts[k, 1], verts[k + 1, 1]
            if (y0 <= y <= y1) or (y1 <= y <= y0):
                if abs(y1 - y0) > 1e-10:
                    t = (y - y0) / (y1 - y0)
                    x_interp = verts[k, 0] + t * (verts[k + 1, 0] - verts[k, 0])
                    xs_at_y.append(x_interp)
        if xs_at_y:
            widths[j] = max(xs_at_y) - min(xs_at_y)

    # Normalize widths to form a probability density
    total = np.sum(widths)
    if total < 1e-10:
        return np.random.RandomState(42).uniform(y_min, y_max, n_samples)

    probs = widths / total
    rng = np.random.RandomState(42)
    indices = rng.choice(n_steps, size=n_samples, p=probs)
    # Add small jitter within each bin
    bin_width = (y_max - y_min) / n_steps
    data = y_grid[indices] + rng.uniform(-bin_width / 2, bin_width / 2, n_samples)
    return data


class FigureIntrospector:
    """Walk a Figure and return classified ArtistGroups."""

    def __init__(self, fig: Figure):
        self._fig = fig
        self._claimed: set[int] = set()  # id() of already-classified artists

    def _claim(self, artist: Any) -> None:
        self._claimed.add(id(artist))

    def _is_claimed(self, artist: Any) -> bool:
        return id(artist) in self._claimed

    def introspect(self) -> list[ArtistGroup]:
        groups: list[ArtistGroup] = []
        # Filter out marginal histogram axes and colorbar axes
        axes_list = [a for a in self._fig.get_axes()
                     if not getattr(a, '_matplotly_marginal', False)
                     and not hasattr(a, '_colorbar')]
        nrows, ncols = self._grid_shape(axes_list)

        for idx, ax in enumerate(axes_list):
            row, col = divmod(idx, max(ncols, 1))
            subplot_index = (row, col)

            # Detection order matters — earlier detectors claim artists.
            # Errorbars before boxplots: errorbar uses precise container
            # matching; boxplot uses line-count heuristics that can
            # misfire on unclaimed errorbar cap/data lines.
            groups.extend(self._detect_heatmaps(ax, subplot_index))
            groups.extend(self._detect_errorbars(ax, subplot_index))
            groups.extend(self._detect_boxplots(ax, subplot_index))
            groups.extend(self._detect_lines(ax, subplot_index))
            groups.extend(self._detect_bars(ax, subplot_index))
            groups.extend(self._detect_scatter(ax, subplot_index))
            groups.extend(self._detect_violins(ax, subplot_index))
            groups.extend(self._detect_fill(ax, subplot_index))

        return groups

    # ------------------------------------------------------------------
    # Grid shape helper
    # ------------------------------------------------------------------

    @staticmethod
    def _grid_shape(axes_list: list) -> tuple[int, int]:
        if not axes_list:
            return (1, 1)
        if hasattr(axes_list[0], "get_subplotspec"):
            try:
                spec = axes_list[0].get_subplotspec()
                if spec is not None:
                    gs = spec.get_gridspec()
                    return (gs.nrows, gs.ncols)
            except Exception:
                pass
        n = len(axes_list)
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / max(ncols, 1)))
        return (nrows, ncols)

    # ------------------------------------------------------------------
    # Detectors
    # ------------------------------------------------------------------

    def _detect_heatmaps(self, ax, subplot_index) -> list[ArtistGroup]:
        """Detect heatmaps from imshow (AxesImage) and pcolormesh (QuadMesh)."""
        groups: list[ArtistGroup] = []

        # imshow → AxesImage
        for im in ax.images:
            if self._is_claimed(im):
                continue
            if not isinstance(im, AxesImage):
                continue
            self._claim(im)
            data = im.get_array()
            if data is not None:
                data = np.asarray(data)
            cmap_name = im.get_cmap().name
            vmin, vmax = im.get_clim()
            norm_type = type(im.norm).__name__
            interp = im.get_interpolation()
            aspect = ax.get_aspect()
            alpha = im.get_alpha()
            has_cbar = bool(getattr(im, 'colorbar', None))
            cbar_label = ""
            cbar_location = "right"
            if has_cbar:
                cb = im.colorbar
                cbar_label = cb.ax.get_ylabel() or cb.ax.get_xlabel() or ""
                try:
                    cbar_location = getattr(cb, 'location', 'right') or 'right'
                except Exception:
                    pass

            # Detect seaborn-style text annotations
            annotations = []
            for txt in ax.texts:
                annotations.append({
                    'x': txt.get_position()[0],
                    'y': txt.get_position()[1],
                    'text': txt.get_text(),
                })

            metadata = {
                'heatmap_type': 'imshow',
                'data': data,
                'cmap': cmap_name,
                'vmin': float(vmin) if vmin is not None else None,
                'vmax': float(vmax) if vmax is not None else None,
                'norm_type': norm_type,
                'interpolation': interp,
                'aspect': aspect,
                'alpha': float(alpha) if alpha is not None else 1.0,
                'has_colorbar': has_cbar,
                'colorbar_label': cbar_label,
                'colorbar_location': cbar_location,
                'annotations': annotations,
                'mappable': im,
            }
            groups.append(ArtistGroup(
                plot_type=PlotType.HEATMAP, axes=ax,
                artists=[im], label="Heatmap (imshow)",
                subplot_index=subplot_index, metadata=metadata))

        # pcolormesh → QuadMesh
        for coll in ax.collections:
            if self._is_claimed(coll):
                continue
            if not isinstance(coll, QuadMesh):
                continue
            self._claim(coll)
            data = coll.get_array()
            if data is not None:
                data = np.asarray(data)
            cmap_name = coll.get_cmap().name
            vmin, vmax = coll.get_clim()
            norm_type = type(coll.norm).__name__
            alpha = coll.get_alpha()
            has_cbar = bool(getattr(coll, 'colorbar', None))
            cbar_label = ""
            cbar_location = "right"
            if has_cbar:
                cb = coll.colorbar
                cbar_label = cb.ax.get_ylabel() or cb.ax.get_xlabel() or ""
                try:
                    cbar_location = getattr(cb, 'location', 'right') or 'right'
                except Exception:
                    pass

            annotations = []
            for txt in ax.texts:
                annotations.append({
                    'x': txt.get_position()[0],
                    'y': txt.get_position()[1],
                    'text': txt.get_text(),
                })

            coords = None
            try:
                coords = coll.get_coordinates()
            except Exception:
                pass

            metadata = {
                'heatmap_type': 'pcolormesh',
                'data': data,
                'cmap': cmap_name,
                'vmin': float(vmin) if vmin is not None else None,
                'vmax': float(vmax) if vmax is not None else None,
                'norm_type': norm_type,
                'alpha': float(alpha) if alpha is not None else 1.0,
                'has_colorbar': has_cbar,
                'colorbar_label': cbar_label,
                'colorbar_location': cbar_location,
                'annotations': annotations,
                'coordinates': coords,
                'mappable': coll,
            }
            groups.append(ArtistGroup(
                plot_type=PlotType.HEATMAP, axes=ax,
                artists=[coll], label="Heatmap (pcolormesh)",
                subplot_index=subplot_index, metadata=metadata))

        return groups

    def _detect_boxplots(self, ax, subplot_index) -> list[ArtistGroup]:
        """Detect box plots structurally with per-box statistics and raw data.

        Supports both patch_artist mode (boxes are Patches) and line mode
        (boxes are Line2D rectangles). Extracts per-box stats and
        reconstructs approximate raw data for mode switching.
        """
        groups: list[ArtistGroup] = []

        # Collect all _-prefixed Line2D (internal lines)
        internal_lines = [l for l in ax.lines
                          if l.get_label().startswith("_")
                          and not self._is_claimed(l)]
        if len(internal_lines) < 5:
            return groups

        # Check for patch_artist mode (boxes are PathPatch or Rectangle)
        # In patch_artist mode, matplotlib creates PathPatch objects for boxes.
        # These have empty labels or '_'-prefixed labels, and non-zero extents.
        box_patches = []
        for p in ax.patches:
            if self._is_claimed(p):
                continue
            if not isinstance(p, (Rectangle, PathPatch)):
                continue
            lbl = p.get_label()
            if lbl and not lbl.startswith("_"):
                continue  # user-labeled patches are not box parts
            # Verify it has visible extents (filter out zero-sized patches)
            try:
                bbox = p.get_extents()
                if bbox.width < 1e-6 or bbox.height < 1e-6:
                    continue
            except Exception:
                continue
            box_patches.append(p)

        if box_patches:
            # patch_artist mode: n patches + lines for median/whiskers/caps/fliers
            # Validate: there should be at least 5*n lines (median + 2 whiskers + 2 caps)
            n_boxes = len(box_patches)
            n_lines = len(internal_lines)
            if n_boxes < 1 or n_lines < n_boxes * 5:
                # Not a boxplot — too few internal lines for the patches
                box_patches = []

        # Detect orientation early (before stats extraction) so we can
        # swap x↔y coordinates for horizontal boxplots.
        orientation = "vertical"
        if box_patches:
            bx0, by0, bx1, by1 = _get_patch_data_bounds(box_patches[0])
            if abs(bx1 - bx0) > abs(by1 - by0) * 2:
                orientation = "horizontal"
        elif internal_lines:
            # Line mode: check first box rectangle (≥4 points)
            for line in internal_lines:
                xd, yd = line.get_xdata(), line.get_ydata()
                if len(xd) >= 4:
                    dx = abs(max(xd) - min(xd))
                    dy = abs(max(yd) - min(yd))
                    if dx > dy * 2:
                        orientation = "horizontal"
                    break

        _horiz = orientation == "horizontal"

        if box_patches:
            n_boxes = len(box_patches)
            box_stats = _extract_box_stats_patch_mode(
                box_patches, internal_lines, n_boxes, horizontal=_horiz)
        else:
            # Line mode: 5, 6, or 7 lines per box
            n = len(internal_lines)
            per_box = None
            for try_per in (7, 6, 5):
                if n % try_per == 0 and n // try_per >= 1:
                    per_box = try_per
                    break
            if per_box is None:
                return groups
            n_boxes = n // per_box
            box_stats = _extract_box_stats_line_mode(
                internal_lines, per_box, n_boxes, horizontal=_horiz)

        if not box_stats:
            return groups

        # Split into separate groups by position clusters
        # (each ax.boxplot() call typically creates a set of boxes at
        # specific positions; detect these by finding position gaps)
        all_positions = [s["position"] for s in box_stats]
        sub_groups = _split_boxplot_groups(
            box_stats, internal_lines, box_patches, all_positions,
            n_boxes, per_box if not box_patches else None,
            horizontal=_horiz)

        for sg_stats, sg_lines, sg_patches in sub_groups:
            sg_n = len(sg_stats)
            # Reconstruct raw data
            sg_raw = [_reconstruct_data_from_stats(s, n_samples=100)
                      for s in sg_stats]

            for l in sg_lines:
                self._claim(l)
            for p in sg_patches:
                self._claim(p)

            metadata = {
                "n_boxes": sg_n,
                "box_stats": sg_stats,
                "raw_data": sg_raw,
                "positions": [s["position"] for s in sg_stats],
                "orientation": orientation,
                "patch_artist": bool(sg_patches),
                "box_patches": sg_patches,
            }
            groups.append(ArtistGroup(
                plot_type=PlotType.BOXPLOT, axes=ax,
                artists=sg_lines + sg_patches,
                label=f"Box plot ({sg_n} boxes)",
                subplot_index=subplot_index, metadata=metadata))
        return groups

    @staticmethod
    def _extract_error_from_segments(barlinecols, x_data, y_data, axis='y'):
        """Extract error magnitudes from LineCollection segments.

        Each segment in a vertical error bar collection is
        ``[(x, y-err_lo), (x, y+err_hi)]``.  For horizontal bars the
        roles of x/y are swapped.

        Returns a 1-D array (symmetric) or 2×N array (asymmetric).
        """
        if not barlinecols:
            return None
        try:
            segs = barlinecols[0].get_segments()
        except Exception:
            return None
        if not segs or len(segs) != len(x_data):
            return None

        err_lo = np.empty(len(segs))
        err_hi = np.empty(len(segs))
        for i, seg in enumerate(segs):
            if len(seg) < 2:
                err_lo[i] = err_hi[i] = 0.0
                continue
            if axis == 'y':
                centre = float(y_data[i])
                err_lo[i] = centre - float(seg[0, 1])
                err_hi[i] = float(seg[1, 1]) - centre
            else:
                centre = float(x_data[i])
                err_lo[i] = centre - float(seg[0, 0])
                err_hi[i] = float(seg[1, 0]) - centre

        # Symmetric when lo ≈ hi everywhere
        if np.allclose(err_lo, err_hi, atol=1e-10):
            return err_hi  # 1-D
        return np.array([err_lo, err_hi])  # 2×N

    def _detect_errorbars(self, ax, subplot_index) -> list[ArtistGroup]:
        groups: list[ArtistGroup] = []

        # Identify ErrorbarContainers owned by BarContainers (from yerr/xerr)
        # These are managed by BarPanel, not ErrorbarPanel.
        bar_owned_ebs: set[int] = set()
        for c in getattr(ax, "containers", []):
            if isinstance(c, BarContainer):
                eb = getattr(c, 'errorbar', None)
                if eb is not None:
                    bar_owned_ebs.add(id(eb))
                    # Claim errorbar artists so they don't appear in other panels
                    if eb[0] is not None:
                        self._claim(eb[0])
                    for cap in eb[1]:
                        self._claim(cap)
                    for barcol in eb[2]:
                        self._claim(barcol)

        for container in getattr(ax, "containers", []):
            if isinstance(container, ErrorbarContainer):
                if id(container) in bar_owned_ebs:
                    continue  # Skip bar-owned error bars
                artists = []
                data_line = container[0]
                if data_line is not None:
                    artists.append(data_line)
                    self._claim(data_line)
                caplines = list(container[1])
                barlinecols = list(container[2])
                for cap in caplines:
                    artists.append(cap)
                    self._claim(cap)
                for barcol in barlinecols:
                    artists.append(barcol)
                    self._claim(barcol)

                # Container holds the user-visible label; data_line is
                # typically "_nolegend_".
                label = container.get_label() if hasattr(
                    container, 'get_label') else None
                if not label or label.startswith("_"):
                    label = (data_line.get_label()
                             if data_line else "Error bars")
                if label.startswith("_"):
                    label = "Error bars"

                # --- Enrich metadata for panel + code gen ---
                metadata: dict = {"container": container}

                if data_line is not None:
                    x_data = np.asarray(data_line.get_xdata(), dtype=float)
                    y_data = np.asarray(data_line.get_ydata(), dtype=float)
                    metadata["x_data"] = x_data
                    metadata["y_data"] = y_data

                    # Data line style
                    try:
                        metadata["color"] = to_hex(data_line.get_color())
                    except Exception:
                        metadata["color"] = "#1f77b4"
                    metadata["line_width"] = float(data_line.get_linewidth())
                    ls = data_line.get_linestyle()
                    _ls_map = {
                        "solid": "-", "dashed": "--",
                        "dotted": ":", "dashdot": "-.",
                    }
                    metadata["line_style"] = _ls_map.get(ls, ls)
                    metadata["marker"] = data_line.get_marker() or ""
                    if metadata["marker"] in ("None", "none"):
                        metadata["marker"] = ""
                    metadata["marker_size"] = float(
                        data_line.get_markersize())
                    metadata["alpha"] = (
                        data_line.get_alpha()
                        if data_line.get_alpha() is not None else 1.0)

                    # Error bar line width (from barlinecols)
                    if barlinecols:
                        try:
                            lws = barlinecols[0].get_linewidths()
                            metadata["bar_lw"] = float(lws[0]) if len(
                                lws) > 0 else 1.5
                        except Exception:
                            metadata["bar_lw"] = 1.5
                    else:
                        metadata["bar_lw"] = 1.5

                    # Cap size
                    if caplines:
                        try:
                            metadata["cap_size"] = float(
                                caplines[0].get_markersize())
                        except Exception:
                            metadata["cap_size"] = 3.0
                    else:
                        metadata["cap_size"] = 0.0

                    # Extract error magnitudes
                    # Separate y-error and x-error barlinecols
                    has_yerr = container.has_yerr if hasattr(
                        container, 'has_yerr') else bool(barlinecols)
                    has_xerr = container.has_xerr if hasattr(
                        container, 'has_xerr') else False

                    metadata["has_yerr"] = has_yerr
                    metadata["has_xerr"] = has_xerr

                    if has_yerr:
                        # y-error bars are the first barlinecol
                        yerr = self._extract_error_from_segments(
                            barlinecols[:1], x_data, y_data, axis='y')
                        metadata["yerr"] = yerr
                    if has_xerr:
                        # x-error bars may be second barlinecol
                        xerr_cols = barlinecols[1:2] if has_yerr else barlinecols[:1]
                        xerr = self._extract_error_from_segments(
                            xerr_cols, x_data, y_data, axis='x')
                        metadata["xerr"] = xerr

                groups.append(ArtistGroup(
                    plot_type=PlotType.ERRORBAR,
                    axes=ax,
                    artists=artists,
                    label=f"Errorbar: {label}",
                    subplot_index=subplot_index,
                    metadata=metadata,
                ))
        return groups

    def _detect_lines(self, ax, subplot_index) -> list[ArtistGroup]:
        groups: list[ArtistGroup] = []
        for line in ax.lines:
            if self._is_claimed(line):
                continue
            label = line.get_label()
            if label.startswith("_"):
                # skip internal matplotlib lines
                continue
            self._claim(line)
            groups.append(ArtistGroup(
                plot_type=PlotType.LINE,
                axes=ax,
                artists=[line],
                label=f"Line: {label}",
                subplot_index=subplot_index,
            ))
        return groups

    @staticmethod
    def _is_histogram_container(container: BarContainer) -> bool:
        """Check if a BarContainer looks like a histogram (contiguous equal-width bins)."""
        patches = [p for p in container.patches if isinstance(p, Rectangle)]
        if len(patches) < 5:
            return False
        patches_sorted = sorted(patches, key=lambda r: r.get_x())
        widths = [r.get_width() for r in patches_sorted]
        # All bins should be roughly the same width
        if len(set(round(w, 6) for w in widths)) > 2:
            return False
        # Bins should be contiguous (no gaps)
        xs = [r.get_x() for r in patches_sorted]
        gaps = [xs[i + 1] - (xs[i] + widths[i]) for i in range(len(xs) - 1)]
        return all(abs(g) < widths[0] * 0.1 for g in gaps)

    def _detect_bars(self, ax, subplot_index) -> list[ArtistGroup]:
        groups: list[ArtistGroup] = []
        bar_containers = [c for c in getattr(ax, "containers", [])
                          if isinstance(c, BarContainer)]
        non_hist_containers = []

        # First pass: detect histograms
        for container in bar_containers:
            if self._is_histogram_container(container):
                artists = []
                for patch in container.patches:
                    if not self._is_claimed(patch):
                        artists.append(patch)
                        self._claim(patch)
                if artists:
                    sorted_artists = sorted(artists, key=lambda r: r.get_x())

                    # Build bin edges and heights
                    bin_edges = ([r.get_x() for r in sorted_artists]
                                 + [sorted_artists[-1].get_x()
                                    + sorted_artists[-1].get_width()])
                    bin_heights = [r.get_height() for r in sorted_artists]

                    # Reconstruct approximate raw data from bin geometry
                    raw_data = []
                    for i, count in enumerate(bin_heights):
                        n = int(round(count))
                        if n > 0:
                            raw_data.extend(
                                np.linspace(bin_edges[i], bin_edges[i + 1],
                                            n, endpoint=False).tolist())
                    raw_data_arr = np.array(raw_data) if raw_data else np.array([0.0])

                    # Extract label: check first patch, then container
                    label = sorted_artists[0].get_label()
                    if not label or label.startswith("_"):
                        label = container.get_label()
                    if not label or label.startswith("_"):
                        label = "Histogram"

                    # Detect orientation: vertical bars have varying height,
                    # horizontal bars have varying width
                    orientation = "vertical"
                    widths = [r.get_width() for r in sorted_artists]
                    heights = [r.get_height() for r in sorted_artists]
                    if (len(set(round(h, 6) for h in heights)) <= 2
                            and len(set(round(w, 6) for w in widths)) > 2):
                        orientation = "horizontal"

                    zorder = sorted_artists[0].get_zorder()

                    groups.append(ArtistGroup(
                        plot_type=PlotType.HISTOGRAM,
                        axes=ax,
                        artists=sorted_artists,
                        label=f"Histogram: {label}",
                        subplot_index=subplot_index,
                        metadata={
                            "container": container,
                            "bin_edges": bin_edges,
                            "bin_heights": bin_heights,
                            "raw_data": raw_data_arr,
                            "zorder": zorder,
                            "orientation": orientation,
                        },
                    ))
            else:
                non_hist_containers.append(container)

        # Second pass: remaining containers are bars
        is_grouped = len(non_hist_containers) > 1
        for container in non_hist_containers:
            artists = []
            for patch in container.patches:
                if not self._is_claimed(patch):
                    artists.append(patch)
                    self._claim(patch)
            if not artists:
                continue

            label = container.get_label()
            if label.startswith("_"):
                label = "bars"

            # Extract bar geometry
            sorted_patches = sorted(artists, key=lambda r: r.get_x())
            widths = [r.get_width() for r in sorted_patches]
            heights = [r.get_height() for r in sorted_patches]

            # Detect orientation: consistent heights + varying widths = horizontal
            if (len(set(round(h, 6) for h in heights)) <= 1
                    and len(set(round(w, 6) for w in widths)) > 1):
                orientation = "horizontal"
                sorted_patches = sorted(artists, key=lambda r: r.get_y())
                positions = [r.get_y() + r.get_height() / 2
                             for r in sorted_patches]
                values = [r.get_width() for r in sorted_patches]
                bottoms = [r.get_x() for r in sorted_patches]
                bar_width = sorted_patches[0].get_height()
            else:
                orientation = "vertical"
                positions = [r.get_x() + r.get_width() / 2
                             for r in sorted_patches]
                values = [r.get_height() for r in sorted_patches]
                bottoms = [r.get_y() for r in sorted_patches]
                bar_width = sorted_patches[0].get_width()

            metadata = {
                "container": container,
                "positions": positions,
                "values": values,
                "bottoms": bottoms,
                "bar_width": bar_width,
                "orientation": orientation,
                "zorder": sorted_patches[0].get_zorder(),
            }

            # Extract error bar info from BarContainer.errorbar (yerr/xerr)
            eb = getattr(container, 'errorbar', None)
            if eb is not None:
                errbar_artists = []
                if eb[0] is not None:
                    errbar_artists.append(eb[0])
                errbar_artists.extend(list(eb[1]))  # caplines
                errbar_artists.extend(list(eb[2]))  # barlinecols
                metadata["errbar_artists"] = errbar_artists
                metadata["errbar_container"] = eb

                barlinecols = list(eb[2])
                caplines = list(eb[1])

                # Extract error values from segments
                pos_arr = np.array(positions)
                val_arr = np.array(values)
                err_axis = 'y' if orientation == 'vertical' else 'x'
                yerr = self._extract_error_from_segments(
                    barlinecols[:1], pos_arr, val_arr, axis=err_axis)
                if yerr is not None:
                    metadata["errbar_values"] = yerr

                # Extract cap size
                if caplines:
                    try:
                        metadata["errbar_capsize"] = float(
                            caplines[0].get_markersize())
                    except Exception:
                        metadata["errbar_capsize"] = 3.0

                # Extract error bar color from barlinecols
                if barlinecols:
                    try:
                        colors = barlinecols[0].get_colors()
                        if len(colors) > 0:
                            metadata["errbar_color"] = to_hex(colors[0])
                    except Exception:
                        pass
                    try:
                        lws = barlinecols[0].get_linewidths()
                        metadata["errbar_linewidth"] = float(
                            lws[0]) if len(lws) > 0 else 1.5
                    except Exception:
                        metadata["errbar_linewidth"] = 1.5

            ptype = PlotType.GROUPED_BAR if is_grouped else PlotType.BAR
            groups.append(ArtistGroup(
                plot_type=ptype,
                axes=ax,
                artists=artists,
                label=f"Bar: {label}",
                subplot_index=subplot_index,
                metadata=metadata,
            ))
        return groups

    def _detect_scatter(self, ax, subplot_index) -> list[ArtistGroup]:
        groups: list[ArtistGroup] = []
        for coll in ax.collections:
            if self._is_claimed(coll):
                continue
            if isinstance(coll, PathCollection):
                self._claim(coll)
                label = coll.get_label()
                if label.startswith("_"):
                    label = "scatter"
                groups.append(ArtistGroup(
                    plot_type=PlotType.SCATTER,
                    axes=ax,
                    artists=[coll],
                    label=f"Scatter: {label}",
                    subplot_index=subplot_index,
                ))
        return groups

    def _detect_violins(self, ax, subplot_index) -> list[ArtistGroup]:
        """Detect violin plots: group all violin bodies + stat lines together."""
        groups: list[ArtistGroup] = []

        # Collect all unclaimed PolyCollections with fill (violin bodies)
        violin_bodies = []
        for coll in ax.collections:
            if self._is_claimed(coll):
                continue
            if isinstance(coll, PolyCollection):
                fc = coll.get_facecolor()
                if fc is not None and len(fc) > 0 and fc[0][3] > 0:
                    violin_bodies.append(coll)

        if not violin_bodies:
            return groups

        # Extract per-violin data from KDE shape
        n_violins = len(violin_bodies)
        positions = []
        raw_data_list = []
        for body in violin_bodies:
            paths = body.get_paths()
            if not paths:
                continue
            verts = paths[0].vertices
            # Violin center position = mean of x-coordinates
            pos = float(np.mean(verts[:, 0]))
            positions.append(pos)
            # Reconstruct data by inverse sampling from KDE shape
            raw = _reconstruct_data_from_violin(verts, n_samples=100)
            raw_data_list.append(raw)
            self._claim(body)

        if not positions:
            return groups

        # Find associated stat lines (LineCollections from violinplot)
        stat_lines = []
        for coll in ax.collections:
            if self._is_claimed(coll):
                continue
            if isinstance(coll, LineCollection):
                # Check if it aligns with violin positions
                try:
                    segs = coll.get_segments()
                    if segs:
                        seg_x = float(np.mean([s[:, 0].mean() for s in segs]))
                        if any(abs(seg_x - p) < 1.0 for p in positions):
                            stat_lines.append(coll)
                            self._claim(coll)
                except Exception:
                    pass

        # Detect orientation
        orientation = "vertical"
        # If violin bodies are wider along y than x, might be horizontal
        # Default to vertical for now

        metadata = {
            "n_violins": n_violins,
            "positions": positions,
            "raw_data": raw_data_list,
            "bodies": violin_bodies,
            "stat_lines": stat_lines,
            "orientation": orientation,
        }
        groups.append(ArtistGroup(
            plot_type=PlotType.VIOLIN, axes=ax,
            artists=list(violin_bodies) + stat_lines,
            label=f"Violin ({n_violins} violins)",
            subplot_index=subplot_index, metadata=metadata))
        return groups

    def _detect_fill(self, ax, subplot_index) -> list[ArtistGroup]:
        groups: list[ArtistGroup] = []
        for coll in ax.collections:
            if self._is_claimed(coll):
                continue
            if isinstance(coll, PolyCollection):
                self._claim(coll)
                groups.append(ArtistGroup(
                    plot_type=PlotType.FILL_BETWEEN,
                    axes=ax,
                    artists=[coll],
                    label="Fill",
                    subplot_index=subplot_index,
                ))
        return groups
