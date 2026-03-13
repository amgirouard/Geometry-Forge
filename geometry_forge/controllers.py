from __future__ import annotations

import contextlib
import logging
from typing import Any, Callable
from collections.abc import Iterator

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .models import (
    AppConstants, DrawingContext, DrawingDependencies, TransformState,
    ShapeConfig, ShapeConfigProvider, ValidationError, ScaleSpec,
)
from .labels import LabelManager
from .drawers import ShapeRegistry

logger = logging.getLogger(__name__)


class TransformController:
    """Handles flip and rotation state management (Tkinter-free)."""

    def __init__(self, on_change_callback: Callable[[], None]):
        self.on_change_callback = on_change_callback
        self.flip_h: bool = False
        self.flip_v: bool = False
        self.base_side: int = 0

    def reset(self) -> None:
        """Reset all transforms to default state."""
        self.flip_h = False
        self.flip_v = False
        self.base_side = 0

    def rotate(self, direction: int, num_sides: int) -> None:
        """Rotate by one position in the given direction."""
        if num_sides <= 0:
            num_sides = 4
        self.base_side = (self.base_side + direction) % num_sides
        self.on_change_callback()

    def get_state(self) -> TransformState:
        """Get current transform state."""
        return TransformState(
            flip_h=self.flip_h,
            flip_v=self.flip_v,
            base_side=self.base_side,
        )

    def set_state(self, state: dict) -> None:
        """Restore transform state from a snapshot dict."""
        t = state.get("transforms", {"h": False, "v": False, "side": 0})
        self.flip_h = t.get("h", False)
        self.flip_v = t.get("v", False)
        self.base_side = t.get("side", 0)


class PlotController:
    """Handles plot generation and drawing coordination (no Tkinter canvas)."""

    def __init__(self, fig: Figure, ax: Axes, label_manager: LabelManager):
        self.fig = fig
        self.ax = ax
        self.label_manager = label_manager
        self.font_size = AppConstants.DEFAULT_FONT_SIZE
        self.line_width = AppConstants.DEFAULT_LINE_WIDTH
        self.font_family = AppConstants.DEFAULT_FONT_FAMILY
        self.label_manager.font_family = AppConstants.DEFAULT_FONT_FAMILY
        self.ax.set_facecolor('white')
        self.ax.patch.set_facecolor('white')

    def set_font_size(self, size: int) -> None:
        self.font_size = size

    def set_line_width(self, width: int) -> None:
        self.line_width = width

    def set_font_family(self, family: str) -> None:
        self.font_family = family
        self.label_manager.font_family = family

    def create_drawing_context(self, aspect_ratio: float) -> DrawingContext:
        ctx = DrawingContext(
            ax=self.ax,
            aspect_ratio=aspect_ratio,
            font_size=self.font_size
        )
        lw = getattr(self, 'line_width', AppConstants.DEFAULT_LINE_WIDTH)
        ctx.line_args["linewidth"] = lw
        ctx.dash_args["linewidth"] = lw
        return ctx

    def create_drawing_deps(self) -> DrawingDependencies:
        return DrawingDependencies(
            ax=self.ax,
            label_manager=self.label_manager,
            font_size=self.font_size
        )

    def clear(self) -> None:
        """Clear the plot and reset label positions."""
        self.ax.clear()
        self.ax.set_facecolor('white')
        self.label_manager.clear_auto_positions()

    def setup_axes(self) -> None:
        """Configure axes for shape drawing to look like fixed paper."""
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_facecolor('white')
        for spine in self.ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(1.0)

    def draw_shape(self, shape: str, ctx: DrawingContext,
                   transform: TransformState, params: dict) -> str | None:
        """Draw a shape using the registry. Returns error message or None on success."""
        deps = self.create_drawing_deps()
        drawer = ShapeRegistry.get_drawer(shape, deps)
        if drawer is None:
            return f"Shape '{shape}' not implemented"
        try:
            drawer.draw(ctx, transform, params)
            return None
        except ValidationError as e:
            logger.warning("Validation error drawing %s: %s", shape, e)
            return str(e)
        except (ValueError, ZeroDivisionError):
            logger.exception("Math error drawing %s", shape)
            return f"Invalid dimensions for {shape}"
        except Exception:
            logger.exception("Unexpected error drawing %s", shape)
            return f"Error drawing {shape}"

    def draw_error(self, message: str) -> None:
        """Display an error message on the axes."""
        self.ax.text(0.5, 0.5, f"⚠ {message}\n\nPlease check your input values.",
                     ha="center", va="center", fontsize=12, color="orange",
                     transform=self.ax.transAxes,
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                               edgecolor="orange", alpha=1))

    def refresh(self) -> None:
        """No-op: Streamlit calls st.pyplot(fig) to display the figure."""
        pass


class HistoryManager:
    """Manages undo/redo stacks by capturing and restoring application state."""

    def __init__(self, max_depth: int = 50):
        self.undo_stack: list[dict] = []
        self.redo_stack: list[dict] = []
        self.max_depth = max_depth
        self._is_restoring = False

    @property
    def is_restoring(self) -> bool:
        return self._is_restoring

    @is_restoring.setter
    def is_restoring(self, value: bool) -> None:
        self._is_restoring = value

    @contextlib.contextmanager
    def restoring(self) -> Iterator[None]:
        self._is_restoring = True
        try:
            yield
        finally:
            self._is_restoring = False

    def capture_state(self, state: dict, force: bool = False) -> None:
        if self._is_restoring:
            return
        if not force and self.undo_stack:
            if self._states_equal(self.undo_stack[-1], state):
                return
        self.undo_stack.append(state)
        self.redo_stack.clear()
        if len(self.undo_stack) > self.max_depth:
            self.undo_stack.pop(0)

    def _states_equal(self, s1: dict, s2: dict) -> bool:
        try:
            return s1 == s2
        except Exception:
            return False

    def undo(self, _=None) -> dict | None:
        if len(self.undo_stack) == 0:
            return None
        self.redo_stack.append(self.undo_stack.pop())
        if len(self.undo_stack) > 0:
            return self.undo_stack[-1]
        return None

    def redo(self, _=None) -> dict | None:
        if not self.redo_stack:
            return None
        state = self.redo_stack.pop()
        self.undo_stack.append(state)
        return state

    def can_undo(self) -> bool:
        return len(self.undo_stack) > 1

    def can_redo(self) -> bool:
        return len(self.redo_stack) > 0


class ScaleManager:
    """Central manager for all slider variables/ranges/defaults (Tkinter-free)."""

    def __init__(self):
        self.specs: dict[str, ScaleSpec] = {
            "aspect":     ScaleSpec("aspect",     AppConstants.SLIDER_MIN, AppConstants.SLIDER_MAX, AppConstants.SLIDER_DEFAULT),
            "slope":      ScaleSpec("slope",      -1.0,  1.0,  0.3),
            "peak_offset": ScaleSpec("peak_offset", -0.5, 1.5,  0.5),
            "view_scale": ScaleSpec("view_scale", 0.25,  1.0,  1.0),
        }
        # Plain float values instead of tk.DoubleVar
        self._values: dict[str, float] = {
            key: spec.default for key, spec in self.specs.items()
        }

    def var(self, key: str) -> "_FloatProxy":
        """Return a proxy object with .get()/.set() API matching tk.DoubleVar."""
        if key not in self.specs:
            raise KeyError(f"Unknown scale key: {key}")
        return _FloatProxy(self._values, key)

    def get(self, key: str) -> float:
        """Direct float access."""
        return self._values.get(key, self.specs[key].default)

    def set(self, key: str, value: float) -> None:
        """Direct float setter."""
        self._values[key] = float(value)

    def reset(self, key: str) -> None:
        self._values[key] = self.specs[key].default

    def reset_many(self, keys: list[str]) -> None:
        for k in keys:
            self.reset(k)

    def get_state(self) -> dict:
        return {"scales": dict(self._values)}

    def set_state(self, state: dict) -> None:
        for k, v in state.get("scales", {}).items():
            if k in self.specs:
                try:
                    self._values[k] = float(v)
                except (ValueError, TypeError):
                    pass


class _FloatProxy:
    """Thin proxy that exposes .get()/.set() so existing code using scale_manager.var(k).get() works."""

    def __init__(self, store: dict, key: str) -> None:
        self._store = store
        self._key = key

    def get(self) -> float:
        return self._store[self._key]

    def set(self, value: float) -> None:
        self._store[self._key] = float(value)
