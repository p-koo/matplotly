"""Tests for introspection → panel → mode change → code generation pipeline.

Exercises the flow: create figure → introspect → build DistributionPanel/
SharedPanel → change mode → _redraw → _store_dist_info → generate_code →
verify generated output.

Run:  python -m pytest tests/test_code_gen.py -v
      (from the PlotBuilder/ directory)
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import matplotlib
matplotlib.use("Agg")  # headless backend

import matplotlib.pyplot as plt
import numpy as np
import pytest

# Add the parent dir so `matplotly` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from matplotly._introspect import FigureIntrospector
from matplotly._code_gen import generate_code
from matplotly._commands import CommandStack
from matplotly._types import PlotType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockCanvas:
    """Headless replacement for CanvasManager — no IPython needed."""

    def __init__(self, fig):
        self._fig = fig

    def redraw(self):
        pass

    def force_redraw(self):
        pass


def _make_panels(groups, fig, canvas, stack):
    """Build DistributionPanel + DistributionSharedPanel for boxplot/violin groups.

    Returns (panels, shared_panel) or ([], None) if no distribution groups.
    """
    from matplotly.panels._distribution import DistributionPanel, DistributionSharedPanel

    dist_groups = [g for g in groups
                   if g.plot_type in (PlotType.BOXPLOT, PlotType.VIOLIN)]
    if not dist_groups:
        return [], None

    panels = []
    for i, grp in enumerate(dist_groups):
        dp = DistributionPanel(grp, stack, canvas)
        dp._plot_number = i + 1
        dp.build()
        panels.append(dp)

    # Determine initial mode
    initial_mode = "box"
    if all(g.plot_type == PlotType.VIOLIN for g in dist_groups):
        initial_mode = "violin"

    shared = DistributionSharedPanel(panels, canvas, initial_mode=initial_mode)
    shared.build()

    return panels, shared


def _change_mode(shared, new_mode):
    """Simulate mode change the same way the UI dropdown callback does.

    Replicates the logic in DistributionSharedPanel._mode_cb that sets
    violin_inner="none" when both box and violin are active.
    """
    from matplotly.panels._distribution import _mode_has_box, _mode_has_violin

    old_mode = shared._mode
    shared._mode = new_mode
    has_both = _mode_has_box(new_mode) and _mode_has_violin(new_mode)
    had_both = _mode_has_box(old_mode) and _mode_has_violin(old_mode)
    if has_both and not had_both:
        for panel in shared._panels:
            panel._violin_inner = "none"
    for panel in shared._panels:
        panel.update_section_visibility(new_mode)
    shared._redraw()


# ---------------------------------------------------------------------------
# Test 1: Simple patch_artist boxplot (4 boxes, single group)
# ---------------------------------------------------------------------------

class TestPatchArtistBoxplot:
    """4 boxes with patch_artist=True — should detect as single group."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(42)
        self.fig, self.ax = plt.subplots()
        data = [np.random.randn(100) + i for i in range(4)]
        self.ax.boxplot(data, labels=["A", "B", "C", "D"], patch_artist=True)
        self.ax.set_title("Patch Artist Boxes")
        yield
        plt.close(self.fig)

    def test_introspect_single_group(self):
        groups = FigureIntrospector(self.fig).introspect()
        box_groups = [g for g in groups if g.plot_type == PlotType.BOXPLOT]
        assert len(box_groups) == 1, (
            f"Expected 1 box group, got {len(box_groups)}")

    def test_introspect_4_boxes(self):
        groups = FigureIntrospector(self.fig).introspect()
        box_group = [g for g in groups if g.plot_type == PlotType.BOXPLOT][0]
        n_boxes = box_group.metadata["n_boxes"]
        assert n_boxes == 4, f"Expected 4 boxes, got {n_boxes}"

    def test_patch_artist_flag(self):
        groups = FigureIntrospector(self.fig).introspect()
        box_group = [g for g in groups if g.plot_type == PlotType.BOXPLOT][0]
        assert box_group.metadata["patch_artist"] is True

    def test_code_gen_basic(self):
        stack = CommandStack()
        code = generate_code(self.fig, stack)
        assert "ax.set_title('Patch Artist Boxes')" in code
        assert "ax.set_xlim" in code
        assert "ax.set_ylim" in code


# ---------------------------------------------------------------------------
# Test 2: Line-mode boxplot (7 boxes, notched, with outliers)
#         Mirrors demo_box.ipynb Test 5
# ---------------------------------------------------------------------------

class TestManyBoxesLineMode:
    """7 notched line-mode boxes with outliers — Test 5 from demo_box.ipynb."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(42)
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        data = []
        labels = []
        for i in range(7):
            d = np.random.randn(200) * (0.5 + i * 0.3) + i
            # Inject outliers
            outliers = np.array([d.min() - 3, d.max() + 3,
                                 d.min() - 5, d.max() + 5])
            data.append(np.concatenate([d, outliers]))
            labels.append(f"Group {i+1}")
        self.data = data
        self.ax.boxplot(data, labels=labels, notch=True)
        self.ax.set_title("Many Boxes with Outliers")
        yield
        plt.close(self.fig)

    def test_introspect_single_group_7_boxes(self):
        groups = FigureIntrospector(self.fig).introspect()
        box_groups = [g for g in groups if g.plot_type == PlotType.BOXPLOT]
        assert len(box_groups) == 1, (
            f"Expected 1 box group, got {len(box_groups)}")
        assert box_groups[0].metadata["n_boxes"] == 7

    def test_line_mode_detected(self):
        groups = FigureIntrospector(self.fig).introspect()
        box_group = [g for g in groups if g.plot_type == PlotType.BOXPLOT][0]
        assert box_group.metadata["patch_artist"] is False

    def test_box_stats_extracted(self):
        groups = FigureIntrospector(self.fig).introspect()
        box_group = [g for g in groups if g.plot_type == PlotType.BOXPLOT][0]
        stats = box_group.metadata["box_stats"]
        assert len(stats) == 7
        for s in stats:
            assert "q1" in s and "q3" in s and "median" in s
            assert s["q1"] <= s["median"] <= s["q3"]
            assert s["whislo"] <= s["q1"]
            assert s["whishi"] >= s["q3"]

    def test_mode_change_box_to_box_violin(self):
        """Change mode to box+violin, verify _store_dist_info and code gen."""
        groups = FigureIntrospector(self.fig).introspect()
        stack = CommandStack()
        canvas = MockCanvas(self.fig)
        panels, shared = _make_panels(groups, self.fig, canvas, stack)

        assert shared is not None, "Shared panel should be created"
        assert len(panels) == 1

        # Change mode
        shared._mode = "box+violin"
        shared._redraw()

        # Verify dist info stored on axes
        dist_info = getattr(self.ax, '_matplotly_dist_info', [])
        assert len(dist_info) >= 1
        info = dist_info[0]
        assert info["display_mode"] == "box+violin"
        assert len(info["raw_data"]) == 7
        assert len(info["positions"]) == 7

        # Generate code and verify
        code = generate_code(self.fig, stack)
        assert "_stats" in code or "_stats_0" in code, \
            "Code should contain bxp stats"
        assert "ax.bxp(" in code, "Code should contain ax.bxp() call"
        assert "patch_artist=True" in code
        # Violin comment should be present
        assert "Violin styling" in code or "violinplot" in code

    def test_mode_change_to_violin_only(self):
        """Change to violin-only mode — should not emit bxp()."""
        groups = FigureIntrospector(self.fig).introspect()
        stack = CommandStack()
        canvas = MockCanvas(self.fig)
        panels, shared = _make_panels(groups, self.fig, canvas, stack)

        shared._mode = "violin"
        shared._redraw()

        code = generate_code(self.fig, stack)
        assert "ax.bxp(" not in code, \
            "Violin-only mode should not emit ax.bxp()"
        assert "Violin styling" in code or "violinplot" in code

    def test_positions_preserved(self):
        """Positions should be correct after mode change."""
        groups = FigureIntrospector(self.fig).introspect()
        box_group = [g for g in groups if g.plot_type == PlotType.BOXPLOT][0]
        original_positions = box_group.metadata["positions"]

        stack = CommandStack()
        canvas = MockCanvas(self.fig)
        panels, shared = _make_panels(groups, self.fig, canvas, stack)

        shared._mode = "box+violin"
        shared._redraw()

        dist_info = self.ax._matplotly_dist_info[0]
        new_positions = dist_info["positions"]
        assert len(new_positions) == len(original_positions), \
            "Position count should match"


# ---------------------------------------------------------------------------
# Test 3: Grouped boxplots (two series with different colors)
# ---------------------------------------------------------------------------

class TestGroupedBoxplots:
    """Two boxplot() calls with different colors at interleaved positions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(42)
        self.fig, self.ax = plt.subplots()
        positions1 = [1, 2, 3]
        positions2 = [1.4, 2.4, 3.4]
        data1 = [np.random.randn(100) for _ in range(3)]
        data2 = [np.random.randn(100) + 1 for _ in range(3)]
        bp1 = self.ax.boxplot(data1, positions=positions1, patch_artist=True,
                              widths=0.35)
        bp2 = self.ax.boxplot(data2, positions=positions2, patch_artist=True,
                              widths=0.35)
        for box in bp1["boxes"]:
            box.set_facecolor("#1f77b4")
        for box in bp2["boxes"]:
            box.set_facecolor("#ff7f0e")
        self.ax.set_title("Grouped Boxplots")
        yield
        plt.close(self.fig)

    def test_two_groups_detected(self):
        groups = FigureIntrospector(self.fig).introspect()
        box_groups = [g for g in groups if g.plot_type == PlotType.BOXPLOT]
        assert len(box_groups) == 2, (
            f"Expected 2 box groups (different colors), got {len(box_groups)}")

    def test_each_group_has_3_boxes(self):
        groups = FigureIntrospector(self.fig).introspect()
        box_groups = [g for g in groups if g.plot_type == PlotType.BOXPLOT]
        for grp in box_groups:
            assert grp.metadata["n_boxes"] == 3, \
                f"Expected 3 boxes per group, got {grp.metadata['n_boxes']}"


# ---------------------------------------------------------------------------
# Test 4: Uniform-color patch_artist boxplot should NOT split
# ---------------------------------------------------------------------------

class TestUniformColorNoSplit:
    """4 patch_artist boxes all same color — should stay as 1 group."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(42)
        self.fig, self.ax = plt.subplots()
        data = [np.random.randn(100) for _ in range(4)]
        self.ax.boxplot(data, patch_artist=True)
        yield
        plt.close(self.fig)

    def test_single_group(self):
        groups = FigureIntrospector(self.fig).introspect()
        box_groups = [g for g in groups if g.plot_type == PlotType.BOXPLOT]
        assert len(box_groups) == 1, (
            f"Uniform-color boxes split into {len(box_groups)} groups "
            f"(expected 1)")


# ---------------------------------------------------------------------------
# Test 5: Violinplot detection
# ---------------------------------------------------------------------------

class TestViolinplot:
    """4 violins should be detected as a single VIOLIN group."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(42)
        self.fig, self.ax = plt.subplots()
        data = [np.random.randn(200) + i for i in range(4)]
        self.ax.violinplot(data, positions=[1, 2, 3, 4])
        yield
        plt.close(self.fig)

    def test_violin_group_detected(self):
        groups = FigureIntrospector(self.fig).introspect()
        violin_groups = [g for g in groups if g.plot_type == PlotType.VIOLIN]
        assert len(violin_groups) == 1
        assert violin_groups[0].metadata["n_violins"] == 4


# ---------------------------------------------------------------------------
# Test 6: Multi-subplot (box + violin side by side)
# ---------------------------------------------------------------------------

class TestMultiSubplot:
    """2 subplots: left has 4 patch_artist boxes, right has 4 violins."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(42)
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 4))
        data = [np.random.randn(100) + i for i in range(4)]
        self.ax1.boxplot(data, labels=["A", "B", "C", "D"], patch_artist=True)
        self.ax1.set_title("Boxes")
        self.ax2.violinplot(data, positions=[1, 2, 3, 4])
        self.ax2.set_title("Violins")
        yield
        plt.close(self.fig)

    def test_introspect_both_types(self):
        groups = FigureIntrospector(self.fig).introspect()
        box_groups = [g for g in groups if g.plot_type == PlotType.BOXPLOT]
        violin_groups = [g for g in groups if g.plot_type == PlotType.VIOLIN]
        assert len(box_groups) == 1
        assert len(violin_groups) == 1

    def test_subplot_indices(self):
        groups = FigureIntrospector(self.fig).introspect()
        box_group = [g for g in groups if g.plot_type == PlotType.BOXPLOT][0]
        violin_group = [g for g in groups if g.plot_type == PlotType.VIOLIN][0]
        assert box_group.subplot_index == (0, 0)
        assert violin_group.subplot_index == (0, 1)

    def test_multi_subplot_code_gen(self):
        stack = CommandStack()
        code = generate_code(self.fig, stack)
        assert "axes = fig.get_axes()" in code
        assert "axes[0]" in code
        assert "axes[1]" in code
        assert "axes[0].set_title('Boxes')" in code
        assert "axes[1].set_title('Violins')" in code

    def test_subplot_spacing_emitted(self):
        stack = CommandStack()
        # Adjust spacing so it's non-zero
        self.fig.subplots_adjust(hspace=0.3, wspace=0.4)
        code = generate_code(self.fig, stack)
        assert "fig.subplots_adjust(" in code


# ---------------------------------------------------------------------------
# Test 7: Code gen for distribution with bxp() after mode change
# ---------------------------------------------------------------------------

class TestBxpCodeGen:
    """Verify that code gen emits correct bxp() structure after mode change."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(42)
        self.fig, self.ax = plt.subplots()
        data = [np.random.randn(100) * (1 + i * 0.5) for i in range(3)]
        self.ax.boxplot(data, labels=["X", "Y", "Z"], patch_artist=True)
        yield
        plt.close(self.fig)

    def test_bxp_stats_in_code(self):
        groups = FigureIntrospector(self.fig).introspect()
        stack = CommandStack()
        canvas = MockCanvas(self.fig)
        panels, shared = _make_panels(groups, self.fig, canvas, stack)

        shared._mode = "box"
        shared._redraw()

        code = generate_code(self.fig, stack)
        assert "_stats" in code, "Should emit stats variable"
        assert "'med'" in code, "Stats should contain 'med' key"
        assert "'q1'" in code
        assert "'q3'" in code
        assert "'whislo'" in code
        assert "'whishi'" in code
        assert "'fliers'" in code

    def test_bxp_boxprops(self):
        groups = FigureIntrospector(self.fig).introspect()
        stack = CommandStack()
        canvas = MockCanvas(self.fig)
        panels, shared = _make_panels(groups, self.fig, canvas, stack)

        # Change box color
        panels[0]._box_color = "#ff0000"
        panels[0]._store_dist_info()
        shared._mode = "box"
        shared._redraw()

        code = generate_code(self.fig, stack)
        assert "boxprops=dict(" in code
        assert "#ff0000" in code, "Custom box color should appear in code"

    def test_bxp_cleanup_code(self):
        """Code gen should emit artist cleanup before bxp() calls."""
        groups = FigureIntrospector(self.fig).introspect()
        stack = CommandStack()
        canvas = MockCanvas(self.fig)
        panels, shared = _make_panels(groups, self.fig, canvas, stack)

        shared._mode = "box"
        shared._redraw()

        code = generate_code(self.fig, stack)
        # Should cleanup original artists
        assert "_l.remove()" in code
        assert "_p.remove()" in code

    def test_bxp_label_set(self):
        """Generated code should set label on first box for legend."""
        groups = FigureIntrospector(self.fig).introspect()
        stack = CommandStack()
        canvas = MockCanvas(self.fig)
        panels, shared = _make_panels(groups, self.fig, canvas, stack)

        shared._mode = "box"
        shared._redraw()

        code = generate_code(self.fig, stack)
        assert ".set_label(" in code


# ---------------------------------------------------------------------------
# Test 8: Horizontal boxplot
# ---------------------------------------------------------------------------

class TestHorizontalBoxplot:
    """Horizontal boxplot should be detected with correct orientation,
    positions, and code generation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        rng = np.random.default_rng(33)
        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.data = [rng.normal(25, 5, 100), rng.normal(30, 4, 100),
                     rng.normal(20, 6, 100), rng.normal(35, 3, 100),
                     rng.normal(28, 5, 100)]
        self.labels = ['Team A', 'Team B', 'Team C', 'Team D', 'Team E']
        self.ax.boxplot(self.data, vert=False, tick_labels=self.labels,
                        patch_artist=True)
        self.ax.set_title('Horizontal Boxes')
        yield
        plt.close(self.fig)

    def test_horizontal_orientation(self):
        groups = FigureIntrospector(self.fig).introspect()
        box_groups = [g for g in groups if g.plot_type == PlotType.BOXPLOT]
        assert len(box_groups) == 1
        assert box_groups[0].metadata["orientation"] == "horizontal"

    def test_positions_are_category_indices(self):
        """Positions should be 1..5 (category indices), not median values."""
        groups = FigureIntrospector(self.fig).introspect()
        box_group = [g for g in groups if g.plot_type == PlotType.BOXPLOT][0]
        positions = box_group.metadata["positions"]
        # Positions should be close to [1, 2, 3, 4, 5]
        for i, pos in enumerate(positions):
            assert abs(pos - (i + 1)) < 0.01, \
                f"Position {i} should be ~{i+1}, got {pos}"

    def test_stats_values_reasonable(self):
        """Q1/median/Q3 should be in the data range, not position range."""
        groups = FigureIntrospector(self.fig).introspect()
        box_group = [g for g in groups if g.plot_type == PlotType.BOXPLOT][0]
        stats = box_group.metadata["box_stats"]
        for s in stats:
            # Data values should be in ~5-50 range, not ~1-5 position range
            assert s["median"] > 5, \
                f"Median {s['median']} looks like a position, not data"
            assert s["q1"] <= s["median"] <= s["q3"]

    def test_code_gen_vert_false(self):
        """Generated code after mode change should have vert=False."""
        groups = FigureIntrospector(self.fig).introspect()
        stack = CommandStack()
        canvas = MockCanvas(self.fig)
        panels, shared = _make_panels(groups, self.fig, canvas, stack)
        shared._mode = "box"
        shared._redraw()
        code = generate_code(self.fig, stack)
        assert "vert=False" in code, \
            f"Should have vert=False in code, got:\n{code}"

    def test_code_gen_yticks(self):
        """Tick labels should be on the y-axis for horizontal boxplots."""
        groups = FigureIntrospector(self.fig).introspect()
        stack = CommandStack()
        canvas = MockCanvas(self.fig)
        panels, shared = _make_panels(groups, self.fig, canvas, stack)
        shared._mode = "box"
        shared._redraw()
        code = generate_code(self.fig, stack)
        assert "set_yticks" in code or "set_yticklabels" in code, \
            f"Should set y-axis ticks for horizontal boxplot"

    def test_code_gen_compiles(self):
        """Generated code should compile."""
        groups = FigureIntrospector(self.fig).introspect()
        stack = CommandStack()
        canvas = MockCanvas(self.fig)
        panels, shared = _make_panels(groups, self.fig, canvas, stack)
        shared._mode = "box"
        shared._redraw()
        code = generate_code(self.fig, stack)
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as e:
            pytest.fail(f"Horizontal box code has syntax error: {e}\n{code}")


class TestHorizontalBoxplotLineMode:
    """Horizontal boxplot in line mode (no patch_artist)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(42)
        self.fig, self.ax = plt.subplots()
        data = [np.random.randn(100) for _ in range(3)]
        self.ax.boxplot(data, vert=False)
        yield
        plt.close(self.fig)

    def test_horizontal_orientation_line_mode(self):
        groups = FigureIntrospector(self.fig).introspect()
        box_groups = [g for g in groups if g.plot_type == PlotType.BOXPLOT]
        assert len(box_groups) == 1
        assert box_groups[0].metadata["orientation"] == "horizontal"

    def test_positions_line_mode(self):
        groups = FigureIntrospector(self.fig).introspect()
        box_group = [g for g in groups if g.plot_type == PlotType.BOXPLOT][0]
        positions = box_group.metadata["positions"]
        for i, pos in enumerate(positions):
            assert abs(pos - (i + 1)) < 0.01, \
                f"Position {i} should be ~{i+1}, got {pos}"


# ---------------------------------------------------------------------------
# Test 9: Lines + scatter in code gen (filtered indexing)
# ---------------------------------------------------------------------------

class TestLineScatterCodeGen:
    """Verify filtered line/scatter indexing in generated code."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(42)
        self.fig, self.ax = plt.subplots()
        self.ax.plot([1, 2, 3], [1, 4, 9], label="Quadratic", color="red")
        self.ax.plot([1, 2, 3], [1, 2, 3], label="Linear", color="blue")
        self.ax.scatter([1, 2, 3], [2, 3, 5], label="Points", c="green")
        yield
        plt.close(self.fig)

    def test_filtered_line_indexing(self):
        stack = CommandStack()
        code = generate_code(self.fig, stack)
        # Should use filtered list, not ax.lines[i]
        assert "_lines = [l for l in ax.lines" in code
        assert "_lines[0]" in code
        assert "_lines[1]" in code

    def test_scatter_filtered_indexing(self):
        stack = CommandStack()
        code = generate_code(self.fig, stack)
        assert "_scatter = [c for c in ax.collections" in code
        assert "_scatter[0]" in code

    def test_line_colors_in_code(self):
        stack = CommandStack()
        code = generate_code(self.fig, stack)
        assert "#ff0000" in code or "'red'" in code  # red
        assert "#0000ff" in code or "'blue'" in code  # blue

    def test_scatter_color_in_code(self):
        stack = CommandStack()
        code = generate_code(self.fig, stack)
        assert "#008000" in code or "'green'" in code  # green


# ---------------------------------------------------------------------------
# Test 10: End-to-end generated code is syntactically valid
# ---------------------------------------------------------------------------

class TestCodeSyntax:
    """Generated code should be valid Python syntax."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(42)
        self.fig, self.ax = plt.subplots()
        data = [np.random.randn(100) for _ in range(3)]
        self.ax.boxplot(data, labels=["A", "B", "C"], patch_artist=True)
        self.ax.set_title("Test")
        self.ax.set_xlabel("X axis")
        self.ax.set_ylabel("Y axis")
        yield
        plt.close(self.fig)

    def test_code_compiles(self):
        """Generated code should compile without SyntaxError."""
        stack = CommandStack()
        code = generate_code(self.fig, stack)
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as e:
            pytest.fail(f"Generated code has syntax error: {e}\n\n{code}")

    def test_code_compiles_after_mode_change(self):
        """Generated code after mode change should also compile."""
        groups = FigureIntrospector(self.fig).introspect()
        stack = CommandStack()
        canvas = MockCanvas(self.fig)
        panels, shared = _make_panels(groups, self.fig, canvas, stack)

        shared._mode = "box+violin"
        shared._redraw()

        code = generate_code(self.fig, stack)
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as e:
            pytest.fail(
                f"Code after mode change has syntax error: {e}\n\n{code}")

    def test_code_compiles_jitter_mode(self):
        """Code in box+jitter mode should compile."""
        groups = FigureIntrospector(self.fig).introspect()
        stack = CommandStack()
        canvas = MockCanvas(self.fig)
        panels, shared = _make_panels(groups, self.fig, canvas, stack)

        shared._mode = "box+jitter"
        shared._redraw()

        code = generate_code(self.fig, stack)
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as e:
            pytest.fail(
                f"Code in box+jitter mode has syntax error: {e}\n\n{code}")


# ---------------------------------------------------------------------------
# Test 11: Violin emitted as actual code (not comments)
# ---------------------------------------------------------------------------

class TestViolinActualCode:
    """Verify violin/jitter are emitted as working code, not placeholder
    comments.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(42)
        self.fig, self.ax = plt.subplots(figsize=(9, 5))
        labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        data = [np.random.randn(60) + i * 3 for i in range(7)]
        self.ax.boxplot(data, tick_labels=labels, notch=True)
        self.ax.set_title('Daily Response Times')
        yield
        plt.close(self.fig)

    def _setup_mode(self, mode):
        groups = FigureIntrospector(self.fig).introspect()
        stack = CommandStack()
        canvas = MockCanvas(self.fig)
        panels, shared = _make_panels(groups, self.fig, canvas, stack)
        _change_mode(shared, mode)
        return generate_code(self.fig, stack)

    def test_box_violin_emits_violinplot(self):
        """box+violin mode should emit actual ax.violinplot() call."""
        code = self._setup_mode("box+violin")
        assert "ax.violinplot(" in code, \
            "Should have actual violinplot call, not a comment"
        assert "# vp = ax.violinplot" not in code, \
            "Violin should NOT be a comment anymore"

    def test_box_violin_has_data_reconstruction(self):
        """box+violin should reconstruct data from stats for violin KDE."""
        code = self._setup_mode("box+violin")
        assert "_raw" in code, "Should have reconstructed data variable"
        assert "np.random.RandomState" in code
        assert "np.concatenate" in code

    def test_box_violin_has_violin_styling(self):
        """Violin bodies should be styled."""
        code = self._setup_mode("box+violin")
        assert "set_facecolor" in code
        assert "set_edgecolor" in code
        assert "set_alpha" in code

    def test_box_violin_hides_violin_stats(self):
        """In box+violin mode, violin inner stats should be hidden
        (the box component handles medians/whiskers)."""
        code = self._setup_mode("box+violin")
        assert "showmedians=False" in code, \
            "box+violin should hide violin medians (box handles them)"

    def test_violin_only_emits_violinplot(self):
        """violin-only mode should emit violinplot with label."""
        code = self._setup_mode("violin")
        assert "ax.violinplot(" in code
        assert "ax.bxp(" not in code
        assert "set_label(" in code, "Violin should get legend label"

    def test_box_jitter_emits_scatter(self):
        """box+jitter should emit actual scatter calls."""
        code = self._setup_mode("box+jitter")
        assert "ax.scatter(" in code, \
            "Should have actual scatter call, not a comment"
        assert "# Jitter:" not in code, \
            "Jitter should NOT be a comment anymore"

    def test_jitter_only_emits_scatter_with_label(self):
        """jitter-only mode should emit scatter with label."""
        code = self._setup_mode("jitter")
        assert "ax.scatter(" in code
        assert "label=" in code, \
            "Jitter-only should set label for legend"

    def test_all_three_modes(self):
        """box+violin+jitter should emit all three components."""
        code = self._setup_mode("box+violin+jitter")
        assert "ax.bxp(" in code, "Should have box"
        assert "ax.violinplot(" in code, "Should have violin"
        assert "ax.scatter(" in code, "Should have jitter"

    def test_generated_code_is_executable(self):
        """Full generated code should actually execute without error."""
        code = self._setup_mode("box+violin")
        # Build the full script: imports + figure creation + generated code
        full_script = textwrap.dedent("""\
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            fig, ax = plt.subplots()
            ax.boxplot([np.random.randn(50) for _ in range(7)],
                       tick_labels=['A','B','C','D','E','F','G'])
        """) + code + "\nplt.close(fig)\n"
        try:
            exec(compile(full_script, "<test>", "exec"), {})
        except Exception as e:
            pytest.fail(
                f"Generated code failed to execute: {e}\n\n{full_script}")

    def test_all_modes_compile(self):
        """Every display mode should produce compilable code."""
        for mode_label, mode in [
            ("box", "box"), ("violin", "violin"), ("jitter", "jitter"),
            ("box+violin", "box+violin"), ("box+jitter", "box+jitter"),
            ("violin+jitter", "violin+jitter"),
            ("box+violin+jitter", "box+violin+jitter"),
        ]:
            # Fresh figure for each mode (redraw destroys original artists)
            fig, ax = plt.subplots()
            ax.boxplot(
                [np.random.randn(60) for _ in range(3)],
                tick_labels=["A", "B", "C"])
            groups = FigureIntrospector(fig).introspect()
            stack = CommandStack()
            canvas = MockCanvas(fig)
            panels, shared = _make_panels(groups, fig, canvas, stack)
            _change_mode(shared, mode)
            code = generate_code(fig, stack)
            plt.close(fig)
            try:
                compile(code, "<generated>", "exec")
            except SyntaxError as e:
                pytest.fail(
                    f"Mode {mode_label!r} produced syntax error: "
                    f"{e}\n\n{code}")


# ---------------------------------------------------------------------------
# Histogram helpers
# ---------------------------------------------------------------------------

def _make_hist_panels(groups, fig, canvas, stack):
    """Build HistogramPanel + HistogramSharedPanel for histogram groups.

    Returns (panels, shared_panel) or ([], None) if no histogram groups.
    """
    from matplotly.panels._histogram import HistogramPanel, HistogramSharedPanel

    hist_groups = [g for g in groups if g.plot_type == PlotType.HISTOGRAM]
    if not hist_groups:
        return [], None

    panels = []
    for i, grp in enumerate(hist_groups):
        hp = HistogramPanel(grp, stack, canvas)
        hp._plot_number = i + 1
        hp.build()
        panels.append(hp)

    shared = HistogramSharedPanel(panels, canvas)
    shared.build()
    return panels, shared


# ---------------------------------------------------------------------------
# Histogram tests
# ---------------------------------------------------------------------------

class TestHistogramCodeGen:
    """Histogram code gen: labels, container detection, merged histograms."""

    def setup_method(self):
        rng = np.random.default_rng(42)
        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.x1 = rng.normal(0, 1, 500)
        self.x2 = rng.normal(2, 0.5, 300)
        self.ax.hist(self.x1, bins=20, alpha=0.7, label='A')
        self.ax.hist(self.x2, bins=20, alpha=0.7, label='B')
        self.stack = CommandStack()
        self.canvas = MockCanvas(self.fig)
        self.groups = FigureIntrospector(self.fig).introspect()

    def teardown_method(self):
        plt.close(self.fig)

    def test_histograms_detected(self):
        """Both histograms should be detected as HISTOGRAM type."""
        hist_groups = [g for g in self.groups
                       if g.plot_type == PlotType.HISTOGRAM]
        assert len(hist_groups) == 2

    def test_labels_from_first_patch(self):
        """Labels should be extracted from first patch, not container."""
        panels, shared = _make_hist_panels(
            self.groups, self.fig, self.canvas, self.stack)
        assert len(panels) == 2
        code = generate_code(self.fig, self.stack)
        # Should have proper labels, not "Bar group" or "Histogram 0"
        assert "Bar group" not in code
        assert "# Histogram: A" in code
        assert "# Histogram: B" in code

    def test_no_bar_group_in_histogram_code(self):
        """Histograms should not fall through to bar section."""
        panels, shared = _make_hist_panels(
            self.groups, self.fig, self.canvas, self.stack)
        code = generate_code(self.fig, self.stack)
        assert "Bar group" not in code

    def test_non_merged_style_only(self):
        """Non-merged histograms should emit style-only container code."""
        panels, shared = _make_hist_panels(
            self.groups, self.fig, self.canvas, self.stack)
        code = generate_code(self.fig, self.stack)
        assert "_bar_cs[" in code
        assert "set_facecolor" in code

    def test_non_merged_compiles(self):
        """Non-merged histogram code should be syntactically valid."""
        panels, shared = _make_hist_panels(
            self.groups, self.fig, self.canvas, self.stack)
        code = generate_code(self.fig, self.stack)
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as e:
            pytest.fail(f"Non-merged histogram code has syntax error: {e}")


class TestHistogramMerged:
    """Merged histogram code gen: should emit ax.hist() recreation."""

    def setup_method(self):
        rng = np.random.default_rng(42)
        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.x1 = rng.normal(0, 1, 500)
        self.x2 = rng.normal(2, 0.5, 300)
        self.ax.hist(self.x1, bins=20, alpha=0.7, label='A')
        self.ax.hist(self.x2, bins=20, alpha=0.7, label='B')
        self.stack = CommandStack()
        self.canvas = MockCanvas(self.fig)
        self.groups = FigureIntrospector(self.fig).introspect()
        self.panels, self.shared = _make_hist_panels(
            self.groups, self.fig, self.canvas, self.stack)

    def teardown_method(self):
        plt.close(self.fig)

    def _merge(self):
        """Simulate pressing the Merge checkbox."""
        self.shared._merged = True
        self.shared._merge_histograms()

    def test_merged_emits_hist_call(self):
        """Merged histograms should emit ax.hist() recreation."""
        self._merge()
        code = generate_code(self.fig, self.stack)
        assert "ax.hist(" in code
        assert "merged" in code.lower()

    def test_merged_no_delete_replot(self):
        """Merged code should NOT contain delete-replot pattern for histograms."""
        self._merge()
        code = generate_code(self.fig, self.stack)
        # Should not have the old remove-containers-then-replot pattern
        assert "for _p in _c: _p.remove()" not in code
        assert "ax.containers[:] =" not in code

    def test_merged_has_correct_labels(self):
        """Merged histogram labels should be 'A' and 'B'."""
        self._merge()
        code = generate_code(self.fig, self.stack)
        assert "'A'" in code
        assert "'B'" in code
        assert "Bar group" not in code

    def test_merged_no_fabricated_data(self):
        """Merged code should NOT contain fabricated data arrays."""
        self._merge()
        code = generate_code(self.fig, self.stack)
        assert "_w0 = np.array" not in code
        assert "np.repeat" not in code
        assert "_edges = np.array" not in code

    def test_merged_has_rwidth(self):
        """Merged code should include rwidth parameter."""
        self._merge()
        code = generate_code(self.fig, self.stack)
        assert "rwidth=" in code

    def test_merged_compiles(self):
        """Merged histogram code should be syntactically valid."""
        self._merge()
        code = generate_code(self.fig, self.stack)
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as e:
            pytest.fail(f"Merged histogram code has syntax error: {e}")

    def test_merged_executes_with_user_code(self):
        """Merged histogram style code should execute on user's existing histograms."""
        self._merge()
        code = generate_code(self.fig, self.stack)
        # The generated code styles existing containers.
        # Simulate the user's original code creating histograms.
        full_script = textwrap.dedent("""\
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            rng = np.random.default_rng(42)
            fig, ax = plt.subplots()
            x1 = rng.normal(0, 1, 500)
            x2 = rng.normal(2, 0.5, 300)
            ax.hist(x1, bins=20, alpha=0.7, label='A')
            ax.hist(x2, bins=20, alpha=0.7, label='B')
        """) + code + "\nplt.close(fig)\n"
        try:
            exec(compile(full_script, "<test>", "exec"), {})
        except Exception as e:
            pytest.fail(
                f"Merged histogram code failed to execute: {e}\n\n"
                f"{full_script}")

    def test_merged_no_bar_group(self):
        """Merged histograms must not produce 'Bar group' labels."""
        self._merge()
        code = generate_code(self.fig, self.stack)
        assert "Bar group" not in code

    def test_unmerge_restores_style_only(self):
        """After unmerge, code gen should revert to style-only approach."""
        self._merge()
        # Now unmerge
        self.shared._merged = False
        self.shared._unmerge_histograms()
        code = generate_code(self.fig, self.stack)
        # Should NOT have merged ax.hist() call
        assert "Histograms (merged)" not in code

    def test_merged_with_data_vars_emits_executable(self):
        """When data_vars are set (Apply path), emit executable ax.hist()."""
        self._merge()
        self.ax._matplotly_hist_data_vars = ['x1', 'x2']
        code = generate_code(self.fig, self.stack)
        # Should have executable (non-commented) ax.hist call with real vars
        assert "ax.hist([x1, x2]," in code
        # Should NOT have comment template
        assert "# ax.hist(" not in code
        # Should NOT have style-only container loop
        assert "containers[" not in code or "# for _p" in code

    def test_merged_with_data_vars_compiles(self):
        """Executable merged hist code should compile."""
        self._merge()
        self.ax._matplotly_hist_data_vars = ['x1', 'x2']
        code = generate_code(self.fig, self.stack)
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as e:
            pytest.fail(f"Merged hist with data_vars has syntax error: {e}")

    def test_merged_with_data_vars_executes(self):
        """Full end-to-end: executable merged hist code runs."""
        self._merge()
        self.ax._matplotly_hist_data_vars = ['x1', 'x2']
        code = generate_code(self.fig, self.stack)
        full_script = textwrap.dedent("""\
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            rng = np.random.default_rng(42)
            fig, ax = plt.subplots()
            x1 = rng.normal(0, 1, 500)
            x2 = rng.normal(2, 0.5, 300)
        """) + code + "\nplt.close(fig)\n"
        try:
            exec(compile(full_script, "<test>", "exec"), {})
        except Exception as e:
            pytest.fail(
                f"Executable merged hist failed: {e}\n\n{full_script}")


# ---------------------------------------------------------------------------
# AST extraction tests
# ---------------------------------------------------------------------------

class TestExtractHistCalls:
    """Tests for _extract_hist_calls and _match_hist_data_vars."""

    def test_basic_extraction(self):
        from matplotly._api import _extract_hist_calls
        src = "ax.hist(x1, bins=20, label='A')\nax.hist(x2, bins=20, label='B')"
        calls = _extract_hist_calls(src)
        assert len(calls) == 2
        assert calls[0]['data_var'] == 'x1'
        assert calls[1]['data_var'] == 'x2'
        assert calls[0]['label'] == 'A'
        assert calls[1]['label'] == 'B'

    def test_complex_data_var(self):
        from matplotly._api import _extract_hist_calls
        src = "ax.hist(data[:, 0], bins=30)"
        calls = _extract_hist_calls(src)
        assert len(calls) == 1
        assert calls[0]['data_var'] == 'data[:, 0]'

    def test_multiline_hist_call(self):
        from matplotly._api import _extract_hist_calls
        src = "ax.hist(\n    x1,\n    bins=20,\n    label='A'\n)"
        calls = _extract_hist_calls(src)
        assert len(calls) == 1
        assert calls[0]['data_var'] == 'x1'
        assert calls[0]['start_line'] == 1
        assert calls[0]['end_line'] == 5

    def test_no_hist_calls(self):
        from matplotly._api import _extract_hist_calls
        src = "ax.plot([1,2,3])\nax.set_title('test')"
        calls = _extract_hist_calls(src)
        assert len(calls) == 0

    def test_match_by_label(self):
        from matplotly._api import _extract_hist_calls, _match_hist_data_vars
        src = "ax.hist(beta, label='B')\nax.hist(alpha, label='A')"
        calls = _extract_hist_calls(src)
        infos = [{'label': 'A'}, {'label': 'B'}]
        vars_ = _match_hist_data_vars(calls, infos)
        # Should match by label, not position
        assert vars_ == ['alpha', 'beta']

    def test_match_positional_fallback(self):
        from matplotly._api import _extract_hist_calls, _match_hist_data_vars
        src = "ax.hist(x1, bins=20)\nax.hist(x2, bins=20)"
        calls = _extract_hist_calls(src)
        infos = [{'label': 'A'}, {'label': 'B'}]
        vars_ = _match_hist_data_vars(calls, infos)
        # No labels in source → positional matching
        assert vars_ == ['x1', 'x2']


# ---------------------------------------------------------------------------
# Distribution Apply-path tests (real data variable)
# ---------------------------------------------------------------------------

class TestDistWithDataVar:
    """Distribution code gen with real data variable (Apply path)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(42)
        self.fig, self.ax = plt.subplots()
        data = [np.random.randn(100) for _ in range(4)]
        self.ax.boxplot(data, labels=["A", "B", "C", "D"],
                        patch_artist=True)
        self.ax.set_title("Test Dist Apply")
        yield
        plt.close(self.fig)

    def _setup_with_data_var(self, mode="box+violin"):
        groups = FigureIntrospector(self.fig).introspect()
        stack = CommandStack()
        canvas = MockCanvas(self.fig)
        panels, shared = _make_panels(groups, self.fig, canvas, stack)
        _change_mode(shared, mode)
        # Simulate Apply path: set data_var on axes
        self.ax._matplotly_dist_data_vars = ["data"]
        return generate_code(self.fig, stack)

    def test_boxplot_with_data_var_no_cleanup(self):
        """Should NOT have artist cleanup when data_var is set."""
        code = self._setup_with_data_var("box")
        assert "_l.remove()" not in code, \
            "Apply path should not emit artist cleanup"
        assert "_p.remove()" not in code

    def test_boxplot_with_data_var_no_fabricated_data(self):
        """Should NOT have _rng.uniform or _emit_data_from_stats."""
        code = self._setup_with_data_var("box+violin")
        assert "_rng.uniform" not in code
        assert "np.concatenate" not in code
        assert "Approximate data from stats" not in code

    def test_boxplot_with_data_var_uses_boxplot(self):
        """Should use ax.boxplot(data, ...) instead of ax.bxp(stats)."""
        code = self._setup_with_data_var("box")
        assert "ax.boxplot(_data," in code or "ax.boxplot(_data_" in code
        assert "ax.bxp(" not in code

    def test_boxplot_with_data_var_uses_violinplot(self):
        """Should use ax.violinplot(data, ...) in violin mode."""
        code = self._setup_with_data_var("box+violin")
        assert "ax.violinplot(" in code

    def test_boxplot_with_data_var_uses_scatter(self):
        """Should use ax.scatter() in jitter mode."""
        code = self._setup_with_data_var("box+jitter")
        assert "ax.scatter(" in code

    def test_boxplot_with_data_var_compiles(self):
        """Generated code should compile."""
        code = self._setup_with_data_var("box+violin")
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as e:
            pytest.fail(
                f"Apply-path dist code has syntax error: {e}\n\n{code}")

    def test_boxplot_with_data_var_all_modes_compile(self):
        """Every mode should compile with data_var set."""
        for mode in ["box", "violin", "jitter", "box+violin",
                      "box+jitter", "violin+jitter",
                      "box+violin+jitter"]:
            fig, ax = plt.subplots()
            ax.boxplot([np.random.randn(60) for _ in range(3)],
                       labels=["A", "B", "C"], patch_artist=True)
            groups = FigureIntrospector(fig).introspect()
            stack = CommandStack()
            canvas = MockCanvas(fig)
            panels, shared = _make_panels(groups, fig, canvas, stack)
            _change_mode(shared, mode)
            ax._matplotly_dist_data_vars = ["data"]
            code = generate_code(fig, stack)
            plt.close(fig)
            try:
                compile(code, "<generated>", "exec")
            except SyntaxError as e:
                pytest.fail(
                    f"Mode {mode!r} Apply-path syntax error: "
                    f"{e}\n\n{code}")

    def test_boxplot_with_data_var_executes(self):
        """Full script with user data should execute."""
        code = self._setup_with_data_var("box+violin")
        full_script = textwrap.dedent("""\
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            fig, ax = plt.subplots()
            data = [np.random.randn(100) for _ in range(4)]
        """) + code + "\nplt.close(fig)\n"
        try:
            exec(compile(full_script, "<test>", "exec"), {})
        except Exception as e:
            pytest.fail(
                f"Apply-path dist code failed to execute: "
                f"{e}\n\n{full_script}")

    def test_data_indices_mapping(self):
        """Data indices should map original positions to data array."""
        code = self._setup_with_data_var("box")
        # 4 boxes at positions 1,2,3,4 → indices [0,1,2,3]
        assert "[0, 1, 2, 3]" in code

    def test_no_stats_variable(self):
        """Apply path should NOT emit _stats variable."""
        code = self._setup_with_data_var("box")
        assert "_stats" not in code or "_stats_" not in code.split(
            "ax.boxplot")[0]

    def test_multi_call_no_index_error(self):
        """Two boxplot calls with separate data vars should not IndexError."""
        fig, ax = plt.subplots()
        np.random.seed(99)
        male_data = [np.random.randn(50) for _ in range(3)]
        female_data = [np.random.randn(50) for _ in range(3)]
        bp1 = ax.boxplot(male_data, positions=[0.8, 1.8, 2.8],
                         patch_artist=True, labels=["A", "B", "C"])
        bp2 = ax.boxplot(female_data, positions=[1.2, 2.2, 3.2],
                         patch_artist=True, labels=["A", "B", "C"])
        # Different facecolors trigger separate groups in introspector
        for p in bp1['boxes']:
            p.set_facecolor('#1f77b4')
        for p in bp2['boxes']:
            p.set_facecolor('#ff7f0e')
        groups = FigureIntrospector(fig).introspect()
        stack = CommandStack()
        canvas = MockCanvas(fig)
        panels, shared = _make_panels(groups, fig, canvas, stack)
        _change_mode(shared, "box")
        # Multi-call: two separate data vars
        ax._matplotly_dist_data_vars = ["male_data", "female_data"]
        code = generate_code(fig, stack)
        plt.close(fig)
        # Each group should reference its own data var directly
        assert "male_data" in code
        assert "female_data" in code
        # Should NOT have index slicing into a single variable
        assert "for i in [" not in code
        # Should compile
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as e:
            pytest.fail(
                f"Multi-call Apply-path syntax error: {e}\n\n{code}")


# ---------------------------------------------------------------------------
# AST extraction tests for boxplot/violinplot
# ---------------------------------------------------------------------------

class TestExtractPlotCalls:
    """Tests for _extract_plot_calls and _find_dependent_lines."""

    def test_extract_boxplot_basic(self):
        from matplotly._api import _extract_plot_calls
        src = "bp = ax.boxplot(data, patch_artist=True)"
        calls = _extract_plot_calls(src, 'boxplot')
        assert len(calls) == 1
        assert calls[0]['data_var'] == 'data'
        assert calls[0]['result_var'] == 'bp'

    def test_extract_boxplot_no_assignment(self):
        from matplotly._api import _extract_plot_calls
        src = "ax.boxplot(data, patch_artist=True)"
        calls = _extract_plot_calls(src, 'boxplot')
        assert len(calls) == 1
        assert calls[0]['data_var'] == 'data'
        assert calls[0]['result_var'] is None

    def test_extract_violinplot(self):
        from matplotly._api import _extract_plot_calls
        src = "vp = ax.violinplot(data, positions=[1,2,3])"
        calls = _extract_plot_calls(src, 'violinplot')
        assert len(calls) == 1
        assert calls[0]['data_var'] == 'data'
        assert calls[0]['result_var'] == 'vp'

    def test_extract_multiline_boxplot(self):
        from matplotly._api import _extract_plot_calls
        src = "bp = ax.boxplot(\n    data,\n    patch_artist=True\n)"
        calls = _extract_plot_calls(src, 'boxplot')
        assert len(calls) == 1
        assert calls[0]['start_line'] == 1
        assert calls[0]['end_line'] == 4

    def test_find_dependent_lines_basic(self):
        from matplotly._api import _find_dependent_lines
        src = (
            "bp = ax.boxplot(data)\n"
            "for box in bp['boxes']:\n"
            "    box.set_facecolor('red')\n"
            "bp['medians'][0].set_color('blue')\n"
            "ax.set_title('test')\n"
        )
        deps = _find_dependent_lines(src, 'bp')
        # Lines 1 (assignment), 2 (for loop header), 3 (loop body), 4 (subscript)
        assert 2 in deps, "For loop referencing bp should be found"
        assert 4 in deps, "bp['medians'] line should be found"
        # Line 5 should NOT be in deps
        assert 5 not in deps

    def test_find_dependent_lines_no_refs(self):
        from matplotly._api import _find_dependent_lines
        src = "x = 1\ny = 2\nz = x + y\n"
        deps = _find_dependent_lines(src, 'bp')
        assert len(deps) == 0


# ---------------------------------------------------------------------------
# Heatmap tests
# ---------------------------------------------------------------------------

class TestHeatmapIntrospection:
    """Heatmap introspection: imshow and pcolormesh detection."""

    def test_imshow_detected(self):
        fig, ax = plt.subplots()
        ax.imshow(np.random.rand(8, 10))
        groups = FigureIntrospector(fig).introspect()
        heatmap_groups = [g for g in groups if g.plot_type == PlotType.HEATMAP]
        assert len(heatmap_groups) == 1
        assert heatmap_groups[0].metadata['heatmap_type'] == 'imshow'
        plt.close(fig)

    def test_pcolormesh_detected(self):
        fig, ax = plt.subplots()
        ax.pcolormesh(np.random.rand(8, 10))
        groups = FigureIntrospector(fig).introspect()
        heatmap_groups = [g for g in groups if g.plot_type == PlotType.HEATMAP]
        assert len(heatmap_groups) == 1
        assert heatmap_groups[0].metadata['heatmap_type'] == 'pcolormesh'
        plt.close(fig)

    def test_heatmap_metadata(self):
        fig, ax = plt.subplots()
        data = np.random.rand(5, 7)
        ax.imshow(data, cmap='coolwarm', vmin=0.0, vmax=1.0)
        groups = FigureIntrospector(fig).introspect()
        hg = [g for g in groups if g.plot_type == PlotType.HEATMAP][0]
        meta = hg.metadata
        assert meta['cmap'] == 'coolwarm'
        assert abs(meta['vmin'] - 0.0) < 1e-6
        assert abs(meta['vmax'] - 1.0) < 1e-6
        assert meta['data'].shape == (5, 7)
        plt.close(fig)

    def test_colorbar_axes_excluded(self):
        fig, ax = plt.subplots()
        im = ax.imshow(np.random.rand(8, 10))
        fig.colorbar(im, ax=ax)
        groups = FigureIntrospector(fig).introspect()
        # Should have exactly 1 heatmap group, no extra axes groups
        heatmap_groups = [g for g in groups if g.plot_type == PlotType.HEATMAP]
        assert len(heatmap_groups) == 1
        # The colorbar axes should not produce any groups
        all_axes = fig.get_axes()
        assert len(all_axes) == 2  # main + colorbar
        plt.close(fig)

    def test_imshow_not_detected_as_scatter(self):
        """imshow should not be misclassified as scatter."""
        fig, ax = plt.subplots()
        ax.imshow(np.random.rand(8, 10))
        groups = FigureIntrospector(fig).introspect()
        scatter_groups = [g for g in groups if g.plot_type == PlotType.SCATTER]
        assert len(scatter_groups) == 0
        plt.close(fig)

    def test_pcolormesh_not_detected_as_scatter(self):
        """pcolormesh QuadMesh should not be misclassified as scatter."""
        fig, ax = plt.subplots()
        ax.pcolormesh(np.random.rand(8, 10))
        groups = FigureIntrospector(fig).introspect()
        scatter_groups = [g for g in groups if g.plot_type == PlotType.SCATTER]
        assert len(scatter_groups) == 0
        plt.close(fig)


class TestHeatmapCodeGen:
    """Heatmap code generation tests."""

    def test_code_gen_cmap(self):
        fig, ax = plt.subplots()
        ax.imshow(np.random.rand(5, 5), cmap='plasma')
        groups = FigureIntrospector(fig).introspect()
        # Store heatmap info on axes
        hg = [g for g in groups if g.plot_type == PlotType.HEATMAP][0]
        ax._matplotly_heatmap_info = [{
            'mappable': hg.metadata['mappable'],
            'heatmap_type': 'imshow',
            'data': hg.metadata['data'],
            'cmap': 'plasma',
            'vmin': 0.0, 'vmax': 1.0,
            'norm_type': 'linear',
            'interpolation': 'nearest',
            'aspect': 'equal',
            'alpha': 1.0,
            'annot_enabled': False,
            'annot_fmt': '.2f', 'annot_fontsize': 8.0,
            'annot_color': 'auto',
            'grid_enabled': False,
            'grid_lw': 1.0, 'grid_color': '#ffffff',
        }]
        stack = CommandStack()
        code = generate_code(fig, stack)
        assert "'plasma'" in code
        assert "set_cmap" in code
        plt.close(fig)

    def test_code_gen_clim(self):
        fig, ax = plt.subplots()
        ax.imshow(np.random.rand(5, 5))
        groups = FigureIntrospector(fig).introspect()
        hg = [g for g in groups if g.plot_type == PlotType.HEATMAP][0]
        ax._matplotly_heatmap_info = [{
            'mappable': hg.metadata['mappable'],
            'heatmap_type': 'imshow',
            'data': hg.metadata['data'],
            'cmap': 'viridis',
            'vmin': 0.2, 'vmax': 0.8,
            'norm_type': 'linear',
            'interpolation': 'nearest',
            'aspect': 'equal',
            'alpha': 1.0,
            'annot_enabled': False,
            'annot_fmt': '.2f', 'annot_fontsize': 8.0,
            'annot_color': 'auto',
            'grid_enabled': False,
            'grid_lw': 1.0, 'grid_color': '#ffffff',
        }]
        stack = CommandStack()
        code = generate_code(fig, stack)
        assert "set_clim(0.2, 0.8)" in code
        plt.close(fig)

    def test_code_gen_compiles(self):
        fig, ax = plt.subplots()
        ax.imshow(np.random.rand(5, 5), cmap='coolwarm')
        groups = FigureIntrospector(fig).introspect()
        hg = [g for g in groups if g.plot_type == PlotType.HEATMAP][0]
        ax._matplotly_heatmap_info = [{
            'mappable': hg.metadata['mappable'],
            'heatmap_type': 'imshow',
            'data': hg.metadata['data'],
            'cmap': 'coolwarm',
            'vmin': 0.0, 'vmax': 1.0,
            'norm_type': 'linear',
            'interpolation': 'nearest',
            'aspect': 'equal',
            'alpha': 0.8,
            'annot_enabled': False,
            'annot_fmt': '.2f', 'annot_fontsize': 8.0,
            'annot_color': 'auto',
            'grid_enabled': False,
            'grid_lw': 1.0, 'grid_color': '#ffffff',
        }]
        stack = CommandStack()
        code = generate_code(fig, stack)
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as e:
            pytest.fail(f"Heatmap code has syntax error: {e}\n\n{code}")
        plt.close(fig)

    def test_code_gen_colorbar(self):
        fig, ax = plt.subplots()
        im = ax.imshow(np.random.rand(5, 5))
        fig.colorbar(im, ax=ax)
        groups = FigureIntrospector(fig).introspect()
        hg = [g for g in groups if g.plot_type == PlotType.HEATMAP][0]
        ax._matplotly_heatmap_info = [{
            'mappable': hg.metadata['mappable'],
            'heatmap_type': 'imshow',
            'data': hg.metadata['data'],
            'cmap': 'viridis',
            'vmin': 0.0, 'vmax': 1.0,
            'norm_type': 'linear',
            'interpolation': 'nearest',
            'aspect': 'equal',
            'alpha': 1.0,
            'annot_enabled': False,
            'annot_fmt': '.2f', 'annot_fontsize': 8.0,
            'annot_color': 'auto',
            'grid_enabled': False,
            'grid_lw': 1.0, 'grid_color': '#ffffff',
        }]
        ax._matplotly_colorbar_info = {
            'show': True,
            'location': 'right',
            'shrink': 1.0,
            'pad': 0.05,
            'label': 'Values',
            'label_fontsize': 12.0,
            'tick_fontsize': 10.0,
        }
        stack = CommandStack()
        code = generate_code(fig, stack)
        assert "fig.colorbar(" in code
        assert "'Values'" in code
        plt.close(fig)

    def test_code_gen_log_norm(self):
        fig, ax = plt.subplots()
        ax.imshow(np.random.rand(5, 5) * 100 + 1)
        groups = FigureIntrospector(fig).introspect()
        hg = [g for g in groups if g.plot_type == PlotType.HEATMAP][0]
        ax._matplotly_heatmap_info = [{
            'mappable': hg.metadata['mappable'],
            'heatmap_type': 'imshow',
            'data': hg.metadata['data'],
            'cmap': 'plasma',
            'vmin': 1.0, 'vmax': 100.0,
            'norm_type': 'log',
            'interpolation': 'nearest',
            'aspect': 'equal',
            'alpha': 1.0,
            'annot_enabled': False,
            'annot_fmt': '.2f', 'annot_fontsize': 8.0,
            'annot_color': 'auto',
            'grid_enabled': False,
            'grid_lw': 1.0, 'grid_color': '#ffffff',
        }]
        stack = CommandStack()
        code = generate_code(fig, stack)
        assert "LogNorm" in code
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as e:
            pytest.fail(f"LogNorm heatmap code syntax error: {e}\n\n{code}")
        plt.close(fig)

    def test_code_gen_annotations(self):
        fig, ax = plt.subplots()
        data = np.random.rand(3, 4)
        ax.imshow(data)
        groups = FigureIntrospector(fig).introspect()
        hg = [g for g in groups if g.plot_type == PlotType.HEATMAP][0]
        ax._matplotly_heatmap_info = [{
            'mappable': hg.metadata['mappable'],
            'heatmap_type': 'imshow',
            'data': data,
            'cmap': 'viridis',
            'vmin': 0.0, 'vmax': 1.0,
            'norm_type': 'linear',
            'interpolation': 'nearest',
            'aspect': 'equal',
            'alpha': 1.0,
            'annot_enabled': True,
            'annot_fmt': '.2f', 'annot_fontsize': 10.0,
            'annot_color': 'auto',
            'grid_enabled': False,
            'grid_lw': 1.0, 'grid_color': '#ffffff',
        }]
        stack = CommandStack()
        code = generate_code(fig, stack)
        assert "ax.text(" in code
        assert "ha='center'" in code
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as e:
            pytest.fail(
                f"Annotations heatmap code syntax error: {e}\n\n{code}")
        plt.close(fig)

    def test_code_gen_grid(self):
        fig, ax = plt.subplots()
        data = np.random.rand(4, 5)
        ax.imshow(data)
        groups = FigureIntrospector(fig).introspect()
        hg = [g for g in groups if g.plot_type == PlotType.HEATMAP][0]
        ax._matplotly_heatmap_info = [{
            'mappable': hg.metadata['mappable'],
            'heatmap_type': 'imshow',
            'data': data,
            'cmap': 'viridis',
            'vmin': 0.0, 'vmax': 1.0,
            'norm_type': 'linear',
            'interpolation': 'nearest',
            'aspect': 'equal',
            'alpha': 1.0,
            'annot_enabled': False,
            'annot_fmt': '.2f', 'annot_fontsize': 8.0,
            'annot_color': 'auto',
            'grid_enabled': True,
            'grid_lw': 2.0, 'grid_color': '#ffffff',
        }]
        stack = CommandStack()
        code = generate_code(fig, stack)
        assert "grid(which='minor'" in code
        assert "set_xticks" in code
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as e:
            pytest.fail(f"Grid heatmap code syntax error: {e}\n\n{code}")
        plt.close(fig)

    def test_pcolormesh_code_gen(self):
        fig, ax = plt.subplots()
        ax.pcolormesh(np.random.rand(5, 5), cmap='RdBu')
        groups = FigureIntrospector(fig).introspect()
        hg = [g for g in groups if g.plot_type == PlotType.HEATMAP][0]
        ax._matplotly_heatmap_info = [{
            'mappable': hg.metadata['mappable'],
            'heatmap_type': 'pcolormesh',
            'data': hg.metadata['data'],
            'cmap': 'RdBu',
            'vmin': 0.0, 'vmax': 1.0,
            'norm_type': 'linear',
            'alpha': 1.0,
            'annot_enabled': False,
            'annot_fmt': '.2f', 'annot_fontsize': 8.0,
            'annot_color': 'auto',
            'grid_enabled': False,
            'grid_lw': 1.0, 'grid_color': '#ffffff',
        }]
        stack = CommandStack()
        code = generate_code(fig, stack)
        assert "_qm" in code
        assert "'RdBu'" in code
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as e:
            pytest.fail(
                f"pcolormesh code syntax error: {e}\n\n{code}")
        plt.close(fig)

    def test_full_heatmap_code_executes(self):
        """Full script with heatmap code should execute."""
        fig, ax = plt.subplots()
        data = np.random.rand(5, 5)
        ax.imshow(data, cmap='coolwarm')
        groups = FigureIntrospector(fig).introspect()
        hg = [g for g in groups if g.plot_type == PlotType.HEATMAP][0]
        ax._matplotly_heatmap_info = [{
            'mappable': hg.metadata['mappable'],
            'heatmap_type': 'imshow',
            'data': data,
            'cmap': 'coolwarm',
            'vmin': 0.0, 'vmax': 1.0,
            'norm_type': 'linear',
            'interpolation': 'bilinear',
            'aspect': 'auto',
            'alpha': 0.9,
            'annot_enabled': True,
            'annot_fmt': '.1f', 'annot_fontsize': 9.0,
            'annot_color': 'auto',
            'grid_enabled': True,
            'grid_lw': 1.5, 'grid_color': '#cccccc',
        }]
        ax._matplotly_colorbar_info = {
            'show': True,
            'location': 'bottom',
            'shrink': 0.8,
            'pad': 0.1,
            'label': 'Intensity',
            'label_fontsize': 11.0,
            'tick_fontsize': 9.0,
        }
        stack = CommandStack()
        code = generate_code(fig, stack)
        plt.close(fig)

        full_script = textwrap.dedent("""\
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            fig, ax = plt.subplots()
            data = np.random.rand(5, 5)
            ax.imshow(data, cmap='viridis')
        """) + code + "\nplt.close(fig)\n"
        try:
            exec(compile(full_script, "<test>", "exec"), {})
        except Exception as e:
            pytest.fail(
                f"Heatmap code failed to execute: {e}\n\n{full_script}")

    def test_code_gen_tick_labels(self):
        """Tick label custom text should be emitted."""
        fig, ax = plt.subplots()
        data = np.random.rand(3, 4)
        ax.imshow(data)
        groups = FigureIntrospector(fig).introspect()
        hg = [g for g in groups if g.plot_type == PlotType.HEATMAP][0]
        ax._matplotly_heatmap_info = [{
            'mappable': hg.metadata['mappable'],
            'heatmap_type': 'imshow',
            'data': data,
            'cmap': 'viridis',
            'vmin': 0.0, 'vmax': 1.0,
            'norm_type': 'linear',
            'interpolation': 'nearest',
            'aspect': 'equal',
            'alpha': 1.0,
            'annot_enabled': False,
            'annot_fmt': '.2f', 'annot_fontsize': 8.0,
            'annot_color': 'auto',
            'grid_enabled': False,
            'grid_lw': 1.0, 'grid_color': '#ffffff',
            'xtick_show': True,
            'xtick_labels': 'A,B,C,D',
            'ytick_show': True,
            'ytick_labels': 'X,Y,Z',
        }]
        stack = CommandStack()
        code = generate_code(fig, stack)
        assert "set_xticklabels" in code
        assert "'A'" in code
        assert "set_yticklabels" in code
        assert "'X'" in code
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as e:
            pytest.fail(f"Tick label code syntax error: {e}\n\n{code}")
        plt.close(fig)

    def test_code_gen_tick_hidden(self):
        """Hidden ticks should emit tick_params with False."""
        fig, ax = plt.subplots()
        data = np.random.rand(3, 4)
        ax.imshow(data)
        groups = FigureIntrospector(fig).introspect()
        hg = [g for g in groups if g.plot_type == PlotType.HEATMAP][0]
        ax._matplotly_heatmap_info = [{
            'mappable': hg.metadata['mappable'],
            'heatmap_type': 'imshow',
            'data': data,
            'cmap': 'viridis',
            'vmin': 0.0, 'vmax': 1.0,
            'norm_type': 'linear',
            'interpolation': 'nearest',
            'aspect': 'equal',
            'alpha': 1.0,
            'annot_enabled': False,
            'annot_fmt': '.2f', 'annot_fontsize': 8.0,
            'annot_color': 'auto',
            'grid_enabled': False,
            'grid_lw': 1.0, 'grid_color': '#ffffff',
            'xtick_show': False,
            'xtick_labels': '',
            'ytick_show': False,
            'ytick_labels': '',
        }]
        stack = CommandStack()
        code = generate_code(fig, stack)
        assert "bottom=False" in code
        assert "left=False" in code
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as e:
            pytest.fail(f"Hidden tick code syntax error: {e}\n\n{code}")
        plt.close(fig)

    def test_colorbar_removes_existing(self):
        """Generated code should remove existing colorbar before creating new."""
        fig, ax = plt.subplots()
        im = ax.imshow(np.random.rand(5, 5))
        fig.colorbar(im, ax=ax, location='bottom')
        groups = FigureIntrospector(fig).introspect()
        hg = [g for g in groups if g.plot_type == PlotType.HEATMAP][0]
        ax._matplotly_heatmap_info = [{
            'mappable': hg.metadata['mappable'],
            'heatmap_type': 'imshow',
            'data': hg.metadata['data'],
            'cmap': 'viridis',
            'vmin': 0.0, 'vmax': 1.0,
            'norm_type': 'linear',
            'interpolation': 'nearest',
            'aspect': 'equal',
            'alpha': 1.0,
            'annot_enabled': False,
            'annot_fmt': '.2f', 'annot_fontsize': 8.0,
            'annot_color': 'auto',
            'grid_enabled': False,
            'grid_lw': 1.0, 'grid_color': '#ffffff',
        }]
        ax._matplotly_colorbar_info = {
            'show': True,
            'location': 'right',
            'shrink': 0.8,
            'pad': 0.05,
            'label': 'Test',
            'label_fontsize': 12.0,
            'tick_fontsize': 10.0,
        }
        stack = CommandStack()
        code = generate_code(fig, stack)
        assert "colorbar.remove()" in code
        assert "fig.colorbar(" in code
        plt.close(fig)

    def test_colorbar_with_existing_executes(self):
        """Full script with existing colorbar should execute cleanly."""
        fig, ax = plt.subplots()
        data = np.random.rand(5, 5)
        im = ax.imshow(data, cmap='coolwarm')
        fig.colorbar(im, ax=ax, location='bottom', shrink=0.8, pad=0.12)
        ax.set_title('Test')
        groups = FigureIntrospector(fig).introspect()
        hg = [g for g in groups if g.plot_type == PlotType.HEATMAP][0]
        ax._matplotly_heatmap_info = [{
            'mappable': hg.metadata['mappable'],
            'heatmap_type': 'imshow',
            'data': data,
            'cmap': 'Reds',
            'vmin': -2.0, 'vmax': 2.0,
            'norm_type': 'linear',
            'interpolation': 'bicubic',
            'aspect': 'equal',
            'alpha': 1.0,
            'annot_enabled': True,
            'annot_fmt': '.2g', 'annot_fontsize': 10.0,
            'annot_color': 'auto',
            'grid_enabled': False,
            'grid_lw': 1.0, 'grid_color': '#ffffff',
        }]
        ax._matplotly_colorbar_info = {
            'show': True,
            'location': 'right',
            'shrink': 0.8,
            'pad': 0.05,
            'label': 'Z-score',
            'label_fontsize': 12.0,
            'tick_fontsize': 10.0,
        }
        stack = CommandStack()
        code = generate_code(fig, stack)
        plt.close(fig)

        full_script = textwrap.dedent("""\
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            np.random.seed(42)
            data = np.random.randn(5, 5)
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(data, cmap='coolwarm', interpolation='bicubic')
            cbar = fig.colorbar(im, ax=ax, location='bottom', shrink=0.8, pad=0.12)
            cbar.set_label('Z-score', fontsize=12)
            ax.set_title('Styled Heatmap', fontsize=14, fontweight='bold')
        """) + code + "\nplt.close(fig)\n"
        try:
            exec(compile(full_script, "<test>", "exec"), {})
        except Exception as e:
            pytest.fail(
                f"Heatmap + colorbar code failed: {e}\n\n{full_script}")


# =====================================================================
# Errorbar tests
# =====================================================================

class TestErrorbarIntrospection:
    """Errorbar introspection: detection, metadata, and label extraction."""

    def test_errorbar_detected(self):
        """ax.errorbar() should produce an ERRORBAR group."""
        fig, ax = plt.subplots()
        ax.errorbar([1, 2, 3], [4, 5, 6], yerr=[0.5, 0.3, 0.4],
                    label="data1")
        groups = FigureIntrospector(fig).introspect()
        eb_groups = [g for g in groups
                     if g.plot_type == PlotType.ERRORBAR]
        assert len(eb_groups) == 1
        plt.close(fig)

    def test_errorbar_metadata_has_data(self):
        """Introspected errorbar should have x_data, y_data, yerr."""
        fig, ax = plt.subplots()
        ax.errorbar([1, 2, 3], [10, 20, 30], yerr=[1, 2, 3],
                    label="series")
        groups = FigureIntrospector(fig).introspect()
        eb = [g for g in groups if g.plot_type == PlotType.ERRORBAR][0]
        meta = eb.metadata
        assert meta["x_data"] is not None
        assert meta["y_data"] is not None
        assert meta["yerr"] is not None
        assert len(meta["x_data"]) == 3
        assert meta["has_yerr"] is True
        plt.close(fig)

    def test_errorbar_metadata_has_style(self):
        """Introspected errorbar should have color, line_width, cap_size."""
        fig, ax = plt.subplots()
        ax.errorbar([1, 2], [3, 4], yerr=[0.1, 0.2],
                    color='red', linewidth=2.0, capsize=5, label="styled")
        groups = FigureIntrospector(fig).introspect()
        eb = [g for g in groups if g.plot_type == PlotType.ERRORBAR][0]
        meta = eb.metadata
        assert "color" in meta
        assert abs(meta["line_width"] - 2.0) < 1e-6
        # cap_size comes from capline.get_markersize() which is 2x the
        # capsize kwarg in some matplotlib versions
        assert meta["cap_size"] > 0
        plt.close(fig)

    def test_errorbar_label(self):
        """Errorbar label should be 'Errorbar: <label>'."""
        fig, ax = plt.subplots()
        ax.errorbar([1, 2], [3, 4], yerr=[0.1, 0.2], label="mydata")
        groups = FigureIntrospector(fig).introspect()
        eb = [g for g in groups if g.plot_type == PlotType.ERRORBAR][0]
        assert eb.label == "Errorbar: mydata"
        plt.close(fig)


class TestErrorbarCodeGen:
    """Errorbar code generation for toggle-based modes."""

    def _setup_errorbar_fig(self, show_bars=True, show_line=True,
                            show_markers=True, show_shaded=False):
        """Create a figure with an errorbar and set up _matplotly_errorbar_info."""
        fig, ax = plt.subplots()
        ax.errorbar([1, 2, 3], [4, 5, 6], yerr=[0.5, 0.3, 0.4],
                    label="test_eb", capsize=3)
        ax._matplotly_errorbar_info = [{
            '_group_id': 0,
            'show_bars': show_bars,
            'show_line': show_line,
            'show_markers': show_markers,
            'show_shaded': show_shaded,
            'bar_color': '#ff0000',
            'marker_color': '#ff0000',
            'line_color': '#ff0000',
            'shade_color': '#ff0000',
            'bar_alpha': 0.8,
            'marker_alpha': 0.8,
            'line_alpha': 0.8,
            'line_width': 2.0,
            'line_style': '-',
            'bar_lw': 1.5,
            'cap_size': 3.0,
            'marker': 'o',
            'marker_size': 6.0,
            'shade_alpha': 0.3,
            'has_yerr': True,
            'has_xerr': False,
            'label': 'test_eb',
        }]
        return fig, ax

    def test_line_on_compiles(self):
        """Code gen with line on should produce valid Python."""
        fig, ax = self._setup_errorbar_fig(show_line=True, show_markers=True)
        stack = CommandStack()
        code = generate_code(fig, stack)
        plt.close(fig)
        assert "Errorbar" in code or "errorbar" in code.lower()
        compile(code, "<test>", "exec")

    def test_markers_only_compiles(self):
        """Code gen with markers only (no line) should produce valid Python."""
        fig, ax = self._setup_errorbar_fig(show_line=False, show_markers=True)
        stack = CommandStack()
        code = generate_code(fig, stack)
        plt.close(fig)
        compile(code, "<test>", "exec")

    def test_line_off_has_linestyle_none(self):
        """Line off should emit linestyle='none' on the data line."""
        fig, ax = self._setup_errorbar_fig(show_line=False, show_markers=True)
        stack = CommandStack()
        code = generate_code(fig, stack)
        plt.close(fig)
        assert "linestyle" in code.lower() or "set_linestyle" in code

    def test_shaded_compiles(self):
        """Code gen with shaded region should produce valid Python."""
        fig, ax = self._setup_errorbar_fig(
            show_line=True, show_markers=False, show_shaded=True)
        stack = CommandStack()
        code = generate_code(fig, stack)
        plt.close(fig)
        assert "fill_between" in code
        compile(code, "<test>", "exec")

    def test_bars_off_compiles(self):
        """Code gen with error bars toggled off should produce valid Python."""
        fig, ax = self._setup_errorbar_fig(
            show_bars=False, show_line=True, show_markers=True)
        stack = CommandStack()
        code = generate_code(fig, stack)
        plt.close(fig)
        assert "set_linewidth(0)" in code or "set_markersize(0)" in code
        compile(code, "<test>", "exec")

    def test_per_section_colors_compiles(self):
        """Code gen with different per-section colors should compile."""
        fig, ax = plt.subplots()
        ax.errorbar([1, 2, 3], [4, 5, 6], yerr=[0.5, 0.3, 0.4],
                    label="test_eb", capsize=3)
        ax._matplotly_errorbar_info = [{
            '_group_id': 0,
            'show_bars': True,
            'show_line': True,
            'show_markers': True,
            'show_shaded': False,
            'bar_color': '#e74c3c',
            'marker_color': '#2ecc71',
            'line_color': '#3498db',
            'shade_color': '#9b59b6',
            'bar_alpha': 1.0,
            'marker_alpha': 1.0,
            'line_alpha': 1.0,
            'line_width': 1.5,
            'line_style': '-',
            'bar_lw': 1.5,
            'cap_size': 3.0,
            'marker': 'o',
            'marker_size': 6.0,
            'shade_alpha': 0.3,
            'has_yerr': True,
            'has_xerr': False,
            'label': 'test_eb',
        }]
        stack = CommandStack()
        code = generate_code(fig, stack)
        plt.close(fig)
        assert '#e74c3c' in code  # bar color
        assert '#2ecc71' in code  # marker color
        assert '#3498db' in code  # line color
        compile(code, "<test>", "exec")

    def test_per_section_alphas_compiles(self):
        """Code gen with different per-section alphas should compile."""
        fig, ax = plt.subplots()
        ax.errorbar([1, 2, 3], [4, 5, 6], yerr=[0.5, 0.3, 0.4],
                    label="test_eb", capsize=3)
        ax._matplotly_errorbar_info = [{
            '_group_id': 0,
            'show_bars': True,
            'show_line': True,
            'show_markers': True,
            'show_shaded': False,
            'bar_color': '#ff0000',
            'marker_color': '#ff0000',
            'line_color': '#ff0000',
            'shade_color': '#ff0000',
            'bar_alpha': 0.5,
            'marker_alpha': 0.7,
            'line_alpha': 0.9,
            'line_width': 1.5,
            'line_style': '-',
            'bar_lw': 1.5,
            'cap_size': 3.0,
            'marker': 'o',
            'marker_size': 6.0,
            'shade_alpha': 0.3,
            'has_yerr': True,
            'has_xerr': False,
            'label': 'test_eb',
        }]
        stack = CommandStack()
        code = generate_code(fig, stack)
        plt.close(fig)
        assert 'set_alpha(0.5)' in code  # bar alpha
        assert 'set_alpha(0.9)' in code  # line alpha
        assert '0.7' in code  # marker alpha in RGBA
        compile(code, "<test>", "exec")

    def test_executes_end_to_end(self):
        """Full end-to-end: create figure + generate + execute code."""
        fig, ax = self._setup_errorbar_fig(
            show_line=True, show_markers=True)
        stack = CommandStack()
        code = generate_code(fig, stack)
        plt.close(fig)

        full_script = textwrap.dedent("""\
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            fig, ax = plt.subplots()
            ax.errorbar([1, 2, 3], [4, 5, 6], yerr=[0.5, 0.3, 0.4],
                        label='test_eb', capsize=3)
        """) + code + "\nplt.close(fig)\n"
        try:
            exec(compile(full_script, "<test>", "exec"), {})
        except Exception as e:
            pytest.fail(
                f"Errorbar code failed: {e}\n\n{full_script}")

    def test_shaded_executes_end_to_end(self):
        """Shaded region code should create fill_between when executed."""
        fig, ax = self._setup_errorbar_fig(
            show_bars=False, show_line=True, show_markers=False,
            show_shaded=True)
        stack = CommandStack()
        code = generate_code(fig, stack)
        plt.close(fig)

        full_script = textwrap.dedent("""\
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            fig, ax = plt.subplots()
            ax.errorbar([1, 2, 3], [4, 5, 6], yerr=[0.5, 0.3, 0.4],
                        label='test_eb', capsize=3)
        """) + code + "\nplt.close(fig)\n"
        try:
            exec(compile(full_script, "<test>", "exec"), {})
        except Exception as e:
            pytest.fail(
                f"Errorbar shaded code failed: {e}\n\n{full_script}")


class TestMultipleErrorbars:
    """Tests for multiple errorbars on the same axes."""

    def test_two_errorbars_detected(self):
        """Two ax.errorbar() calls should produce two ERRORBAR groups."""
        fig, ax = plt.subplots()
        ax.errorbar([1, 2, 3], [4, 5, 6], yerr=[0.5, 0.3, 0.4],
                    label="series_a")
        ax.errorbar([1, 2, 3], [7, 8, 9], yerr=[0.2, 0.1, 0.3],
                    label="series_b")
        groups = FigureIntrospector(fig).introspect()
        eb_groups = [g for g in groups
                     if g.plot_type == PlotType.ERRORBAR]
        assert len(eb_groups) == 2
        plt.close(fig)

    def test_two_errorbars_correct_labels(self):
        """Multiple errorbars should have correct labels."""
        fig, ax = plt.subplots()
        ax.errorbar([1, 2, 3], [4, 5, 6], yerr=[0.5, 0.3, 0.4],
                    label="alpha")
        ax.errorbar([1, 2, 3], [7, 8, 9], yerr=[0.2, 0.1, 0.3],
                    label="beta")
        groups = FigureIntrospector(fig).introspect()
        eb_groups = [g for g in groups
                     if g.plot_type == PlotType.ERRORBAR]
        labels = sorted([g.label for g in eb_groups])
        assert labels == ["Errorbar: alpha", "Errorbar: beta"]
        plt.close(fig)


# ---------------------------------------------------------------------------
# Bar error bar helpers
# ---------------------------------------------------------------------------

def _make_bar_panels(groups, fig, canvas, stack):
    """Build BarPanel + BarSharedPanel for bar groups.

    Returns (panels, shared_panel) or ([], None) if no bar groups.
    """
    from matplotly.panels._bar import BarPanel, BarSharedPanel

    bar_groups = [g for g in groups if g.plot_type == PlotType.BAR]
    if not bar_groups:
        return [], None

    panels = []
    for i, grp in enumerate(bar_groups):
        bp = BarPanel(grp, stack, canvas)
        bp._plot_number = i + 1
        bp.build()
        bp._store_bar_info()
        panels.append(bp)

    shared = BarSharedPanel(panels, canvas)
    shared.build()
    return panels, shared


# ---------------------------------------------------------------------------
# Bar error bar tests
# ---------------------------------------------------------------------------

class TestBarErrorBars:
    """Bar plot error bar tests — bars without yerr."""

    def setup_method(self):
        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.ax.bar(['A', 'B', 'C'], [10, 20, 15], label='Sales')
        self.stack = CommandStack()
        self.canvas = MockCanvas(self.fig)
        self.groups = FigureIntrospector(self.fig).introspect()
        self.panels, self.shared = _make_bar_panels(
            self.groups, self.fig, self.canvas, self.stack)

    def teardown_method(self):
        plt.close(self.fig)

    def test_errorbars_default_off(self):
        """Error bar toggle should default to False when no yerr."""
        assert len(self.panels) >= 1
        assert self.panels[0]._show_errorbars is False

    def test_no_errorbar_panel_without_yerr(self):
        """Bars without yerr should not create ErrorbarPanel."""
        eb_groups = [g for g in self.groups
                     if g.plot_type == PlotType.ERRORBAR]
        assert len(eb_groups) == 0

    def test_no_errorbars_in_code_without_yerr(self):
        """Bars without yerr should not emit errorbar code."""
        code = generate_code(self.fig, self.stack)
        assert "errorbar" not in code.lower()


class TestBarErrorBarsWithYerr:
    """Bar plot error bar tests — bars with yerr."""

    def setup_method(self):
        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.ax.bar(['A', 'B', 'C'], [10, 20, 15],
                    yerr=[2, 3, 1], capsize=4, label='Sales')
        self.stack = CommandStack()
        self.canvas = MockCanvas(self.fig)
        self.groups = FigureIntrospector(self.fig).introspect()
        self.panels, self.shared = _make_bar_panels(
            self.groups, self.fig, self.canvas, self.stack)

    def teardown_method(self):
        plt.close(self.fig)

    def test_errorbars_default_on(self):
        """Error bar toggle should default to True when yerr exists."""
        assert len(self.panels) >= 1
        assert self.panels[0]._show_errorbars is True

    def test_errbar_values_extracted(self):
        """Error values should be extracted from yerr."""
        panel = self.panels[0]
        assert panel._errbar_values is not None
        assert list(panel._errbar_values) == [2.0, 3.0, 1.0]

    def test_no_errorbar_panel_for_bar_yerr(self):
        """Bar-owned errorbars should not create a separate ErrorbarPanel."""
        eb_groups = [g for g in self.groups
                     if g.plot_type == PlotType.ERRORBAR]
        assert len(eb_groups) == 0

    def test_toggle_off_clears_artists(self):
        """Toggling off should remove error bar artists."""
        panel = self.panels[0]
        panel._show_errorbars = False
        panel._clear_bar_errorbars()
        tagged = [l for l in self.ax.lines
                  if getattr(l, '_matplotly_bar_errorbar', False)]
        # Original errorbar artists should also be gone
        assert len(self.ax.lines) == 0
        assert len(self.ax.collections) == 0

    def test_toggle_on_redraws_artists(self):
        """Toggling on should create tagged error bar artists."""
        panel = self.panels[0]
        panel._show_errorbars = False
        panel._clear_bar_errorbars()
        panel._show_errorbars = True
        panel._draw_bar_errorbars()
        tagged = [l for l in self.ax.lines
                  if getattr(l, '_matplotly_bar_errorbar', False)]
        tagged += [c for c in self.ax.collections
                   if getattr(c, '_matplotly_bar_errorbar', False)]
        assert len(tagged) > 0

    def test_errorbars_in_code_gen(self):
        """Bars with yerr should produce errorbar code with stored values."""
        code = generate_code(self.fig, self.stack)
        assert "errorbar" in code.lower()
        assert "2.0" in code  # actual yerr values, not fabricated

    def test_errorbars_code_compiles(self):
        """Generated code with error bars should be syntactically valid."""
        code = generate_code(self.fig, self.stack)
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as e:
            pytest.fail(f"Bar errorbar code has syntax error: {e}")

    def test_redraw_preserves_errorbars(self):
        """Changing bar width with error bars on should redraw them."""
        panel = self.panels[0]
        self.shared._bar_width = 0.5
        self.shared._redraw_bars()
        tagged = [l for l in self.ax.lines
                  if getattr(l, '_matplotly_bar_errorbar', False)]
        tagged += [c for c in self.ax.collections
                   if getattr(c, '_matplotly_bar_errorbar', False)]
        assert len(tagged) > 0