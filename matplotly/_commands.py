"""Undo/redo command system for figure modifications."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Command:
    """A single property change on a matplotlib artist."""

    artist: Any
    property_name: str
    old_value: Any
    new_value: Any
    # For complex operations that can't use simple set_*
    apply_fn: Callable[[], None] | None = None
    revert_fn: Callable[[], None] | None = None
    # Human-readable description for code gen
    description: str = ""

    def execute(self) -> None:
        if self.apply_fn is not None:
            self.apply_fn()
        else:
            setter = getattr(self.artist, f"set_{self.property_name}", None)
            if setter is not None:
                setter(self.new_value)

    def undo(self) -> None:
        if self.revert_fn is not None:
            self.revert_fn()
        else:
            setter = getattr(self.artist, f"set_{self.property_name}", None)
            if setter is not None:
                setter(self.old_value)


@dataclass
class BatchCommand:
    """A group of commands applied/reverted together as one undo step."""

    commands: list[Command] = field(default_factory=list)
    description: str = ""

    def execute(self) -> None:
        for cmd in self.commands:
            cmd.execute()

    def undo(self) -> None:
        for cmd in reversed(self.commands):
            cmd.undo()


class CommandStack:
    """Manages undo/redo stacks with a maximum depth."""

    def __init__(self, max_depth: int = 100,
                 on_change: Callable[[], None] | None = None):
        self._undo_stack: list[Command | BatchCommand] = []
        self._redo_stack: list[Command | BatchCommand] = []
        self._max_depth = max_depth
        self._on_change = on_change

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    @property
    def history(self) -> list[Command | BatchCommand]:
        return list(self._undo_stack)

    def execute(self, cmd: Command | BatchCommand) -> None:
        """Execute a command and push it onto the undo stack."""
        cmd.execute()
        self._undo_stack.append(cmd)
        if len(self._undo_stack) > self._max_depth:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        if self._on_change:
            self._on_change()

    def undo(self) -> None:
        if not self._undo_stack:
            return
        cmd = self._undo_stack.pop()
        cmd.undo()
        self._redo_stack.append(cmd)
        if self._on_change:
            self._on_change()

    def redo(self) -> None:
        if not self._redo_stack:
            return
        cmd = self._redo_stack.pop()
        cmd.execute()
        self._undo_stack.append(cmd)
        if self._on_change:
            self._on_change()

    def clear(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()
        if self._on_change:
            self._on_change()
