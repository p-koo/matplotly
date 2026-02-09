"""Panel registry — maps PlotType to panel class, plus factory function."""
from __future__ import annotations

from .._commands import CommandStack
from .._renderer import CanvasManager
from .._types import ArtistGroup, PlotType
from ._base import ArtistPanel
from ._bar import BarPanel
from ._distribution import DistributionPanel
from ._errorbar import ErrorbarPanel
from ._fill import FillPanel
from ._heatmap import HeatmapPanel
from ._histogram import HistogramPanel
from ._line import LinePanel
from ._scatter import ScatterPanel

PANEL_REGISTRY: dict[PlotType, type[ArtistPanel]] = {
    PlotType.LINE: LinePanel,
    PlotType.BAR: BarPanel,
    PlotType.GROUPED_BAR: BarPanel,
    PlotType.SCATTER: ScatterPanel,
    PlotType.HISTOGRAM: HistogramPanel,
    PlotType.BOXPLOT: DistributionPanel,
    PlotType.VIOLIN: DistributionPanel,
    PlotType.ERRORBAR: ErrorbarPanel,
    PlotType.FILL_BETWEEN: FillPanel,
    PlotType.HEATMAP: HeatmapPanel,
}


def create_panel(group: ArtistGroup, stack: CommandStack,
                 canvas: CanvasManager) -> ArtistPanel | None:
    """Create the appropriate panel for an ArtistGroup, or None if unsupported."""
    cls = PANEL_REGISTRY.get(group.plot_type)
    if cls is None:
        return None
    return cls(group, stack, canvas)
