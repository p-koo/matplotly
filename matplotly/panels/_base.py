"""Base class for artist-specific control panels."""
from __future__ import annotations

import abc

import ipywidgets as widgets

from .._commands import CommandStack
from .._renderer import CanvasManager
from .._types import ArtistGroup


class ArtistPanel(abc.ABC):
    """Abstract base for a panel that controls one ArtistGroup."""

    def __init__(self, group: ArtistGroup, stack: CommandStack,
                 canvas: CanvasManager):
        self._group = group
        self._stack = stack
        self._canvas = canvas
        self._widget: widgets.Widget | None = None

    @abc.abstractmethod
    def build(self) -> widgets.Widget:
        """Construct and return the ipywidgets control panel."""

    @property
    def widget(self) -> widgets.Widget:
        if self._widget is None:
            self._widget = self.build()
        return self._widget

    # -- helpers -----------------------------------------------------------

    def _execute_and_redraw(self, cmd) -> None:
        """Execute a command and refresh the canvas."""
        self._stack.execute(cmd)
        self._canvas.redraw()
