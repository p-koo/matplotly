"""Plot type classification enums and data structures."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class PlotType(Enum):
    LINE = auto()
    BAR = auto()
    GROUPED_BAR = auto()
    SCATTER = auto()
    VIOLIN = auto()
    BOXPLOT = auto()
    HISTOGRAM = auto()
    ERRORBAR = auto()
    FILL_BETWEEN = auto()
    HEATMAP = auto()


@dataclass
class ArtistGroup:
    """A group of related matplotlib artists sharing a plot type."""

    plot_type: PlotType
    axes: Any  # matplotlib.axes.Axes
    artists: list[Any] = field(default_factory=list)
    label: str = ""
    subplot_index: tuple[int, int] = (0, 0)
    metadata: dict[str, Any] = field(default_factory=dict)
