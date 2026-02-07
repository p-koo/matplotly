"""Figure introspection — detect and classify all artists in a matplotlib figure."""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.container import BarContainer, ErrorbarContainer
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from ._types import ArtistGroup, PlotType


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
        axes_list = self._fig.get_axes()
        nrows, ncols = self._grid_shape(axes_list)

        for idx, ax in enumerate(axes_list):
            row, col = divmod(idx, max(ncols, 1))
            subplot_index = (row, col)

            # Detection order matters — earlier detectors claim artists
            groups.extend(self._detect_boxplots(ax, subplot_index))
            groups.extend(self._detect_errorbars(ax, subplot_index))
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

    def _detect_boxplots(self, ax, subplot_index) -> list[ArtistGroup]:
        """Detect box plots structurally.

        matplotlib's boxplot() creates lines with _-prefixed labels in groups
        of 7 per box: 1 box, 2 whiskers, 2 caps, 1 median, 1 flier.
        We detect this by counting _-prefixed Line2D objects and checking
        the 7-per-box pattern.
        """
        groups: list[ArtistGroup] = []

        # Collect all _-prefixed Line2D (internal lines)
        internal_lines = [l for l in ax.lines if l.get_label().startswith("_")]
        if len(internal_lines) < 7:
            return groups

        # Check if count is a multiple of 7 (one box = 7 lines)
        # Also accept 6 per box (when means are not shown, fliers might be absent)
        n = len(internal_lines)
        if n % 7 == 0 or n % 6 == 0:
            per_box = 7 if n % 7 == 0 else 6
            n_boxes = n // per_box
            if n_boxes >= 1:
                for l in internal_lines:
                    self._claim(l)
                groups.append(ArtistGroup(
                    plot_type=PlotType.BOXPLOT,
                    axes=ax,
                    artists=internal_lines,
                    label=f"Box plot ({n_boxes} boxes)",
                    subplot_index=subplot_index,
                    metadata={"n_boxes": n_boxes},
                ))

        return groups

    def _detect_errorbars(self, ax, subplot_index) -> list[ArtistGroup]:
        groups: list[ArtistGroup] = []
        for container in getattr(ax, "containers", []):
            if isinstance(container, ErrorbarContainer):
                artists = []
                data_line = container[0]
                if data_line is not None:
                    artists.append(data_line)
                    self._claim(data_line)
                # caplines and barlinecols
                for cap in container[1]:
                    artists.append(cap)
                    self._claim(cap)
                for barcol in container[2]:
                    artists.append(barcol)
                    self._claim(barcol)

                label = data_line.get_label() if data_line else "Error bars"
                if label.startswith("_"):
                    label = "Error bars"
                groups.append(ArtistGroup(
                    plot_type=PlotType.ERRORBAR,
                    axes=ax,
                    artists=artists,
                    label=f"Errorbar: {label}",
                    subplot_index=subplot_index,
                    metadata={"container": container},
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
                    groups.append(ArtistGroup(
                        plot_type=PlotType.HISTOGRAM,
                        axes=ax,
                        artists=sorted_artists,
                        label="Histogram",
                        subplot_index=subplot_index,
                        metadata={
                            "container": container,
                            "bin_edges": [r.get_x() for r in sorted_artists]
                                         + [sorted_artists[-1].get_x()
                                            + sorted_artists[-1].get_width()],
                            "bin_heights": [r.get_height() for r in sorted_artists],
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

            ptype = PlotType.GROUPED_BAR if is_grouped else PlotType.BAR
            groups.append(ArtistGroup(
                plot_type=ptype,
                axes=ax,
                artists=artists,
                label=f"Bar: {label}",
                subplot_index=subplot_index,
                metadata={"container": container},
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
        groups: list[ArtistGroup] = []
        for coll in ax.collections:
            if self._is_claimed(coll):
                continue
            if isinstance(coll, PolyCollection):
                # Heuristic: violin bodies are PolyCollections with fill
                fc = coll.get_facecolor()
                if fc is not None and len(fc) > 0 and fc[0][3] > 0:
                    self._claim(coll)
                    groups.append(ArtistGroup(
                        plot_type=PlotType.VIOLIN,
                        axes=ax,
                        artists=[coll],
                        label="Violin",
                        subplot_index=subplot_index,
                    ))
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
