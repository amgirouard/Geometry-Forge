from __future__ import annotations

import contextlib
import math
import logging
import tkinter as tk
from typing import Any, Callable
from collections.abc import Iterator
from dataclasses import dataclass

from .models import (
    AppConstants, DrawingContext, DrawingDependencies, TransformState,
    ShapeConfig, ShapeConfigProvider, ValidationError, ScaleSpec,
)
from .labels import LabelManager
from .drawers import ShapeRegistry

logger = logging.getLogger(__name__)


class TransformController:
    """Handles flip and rotation state management."""
    
    def __init__(
        self,
        on_change_callback: Callable[[], None],
        flip_h_btn: tk.Button | None = None,
        flip_v_btn: tk.Button | None = None,
        rotate_ccw_btn: tk.Button | None = None,
        rotate_cw_btn: tk.Button | None = None
    ):
        self.on_change_callback = on_change_callback

        # Single source of truth
        self.flip_h: bool = False
        self.flip_v: bool = False
        self.base_side: int = 0
        
        # UI elements
        self.flip_h_btn = flip_h_btn
        self.flip_v_btn = flip_v_btn
        self.rotate_ccw_btn = rotate_ccw_btn
        self.rotate_cw_btn = rotate_cw_btn

        self._reset_button_icons()
    
    def reset(self) -> None:
        """Reset all transforms to default state."""
        self.flip_h = False
        self.flip_v = False
        self.base_side = 0
        self._reset_button_icons()
    
    def rotate(self, direction: int, num_sides: int) -> None:
        """Rotate by one position in the given direction."""
        if num_sides <= 0:
            num_sides = 4
        self.base_side = (self.base_side + direction) % num_sides
        self.on_change_callback()
    
    def _reset_button_icons(self) -> None:
        """Ensure flip button icons show their static arrow characters.
        
        Note: these buttons use a fixed icon (↔ / ↕) rather than reflecting
        active/inactive state visually — selection state is shown elsewhere
        in the UI. This method is a no-op if the buttons haven't been attached yet.
        """
        if self.flip_h_btn:
            self.flip_h_btn.config(text="↔")
        if self.flip_v_btn:
            self.flip_v_btn.config(text="↕")
    
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



class InputController:
    """Handles input field creation and management."""

    def __init__(self, input_frame: tk.Frame, label_manager: LabelManager,
                 on_change_callback: Callable[[], None]):
        self.input_frame = input_frame
        self.label_manager = label_manager
        self.on_change_callback = on_change_callback
        self.entries: dict[str, dict] = {}
        self._debounce_timers: dict[str, str] = {}  # label → after() id

    def get_entry_value(self, key: str) -> str:
        """Get the text value of an entry field."""
        if key in self.entries:
            return self.entries[key]["text"].get()
        return ""
    
    def set_entry_value(self, key: str, value: str) -> None:
        """Set the text value of an entry field."""
        if key in self.entries:
            self.entries[key]["text"].delete(0, tk.END)
            self.entries[key]["text"].insert(0, value)
    
    def clear_entry(self, key: str) -> None:
        """Clear an entry field."""
        if key in self.entries:
            self.entries[key]["text"].delete(0, tk.END)
    
    def create_header(self, slim: bool = False) -> None:
        """Create the input table header row. Slim mode omits Show column."""
        if slim:
            tk.Label(self.input_frame, text="Parameter", bg=AppConstants.BG_COLOR,
                     font=("Arial", 9, "bold")).grid(row=0, column=0, padx=(2, 3))
            tk.Label(self.input_frame, text="Value", bg=AppConstants.BG_COLOR,
                     font=("Arial", 9, "bold")).grid(row=0, column=1, padx=1)
        else:
            tk.Label(self.input_frame, text="Parameter / Label", bg=AppConstants.BG_COLOR,
                     font=("Arial", 9, "bold")).grid(row=0, column=0, padx=(2, 3))
            tk.Label(self.input_frame, text="Value", bg=AppConstants.BG_COLOR,
                     font=("Arial", 9, "bold")).grid(row=0, column=1, padx=1)
            tk.Label(self.input_frame, text="Show", bg=AppConstants.BG_COLOR,
                     font=("Arial", 9, "bold")).grid(row=0, column=2, padx=0)
    
    def create_entry_row(self, row: int, label: str, default: str = "",
                         readonly: bool = False, show_default: bool = True,
                         slim: bool = False) -> tuple[tk.Entry, tk.BooleanVar]:
        """Create a single input row. Slim mode omits the Show checkbox."""
        tk.Label(self.input_frame, text=label + ":", bg=AppConstants.BG_COLOR).grid(
            row=row, column=0, padx=(2, 3), sticky="e"
        )
        
        ent_text = tk.Entry(self.input_frame, width=AppConstants.ENTRY_WIDTH, takefocus=1)
        ent_text.grid(row=row, column=1, padx=1, pady=1)
        ent_text.insert(0, default)
        if readonly:
            ent_text.config(state="readonly")
        
        show_var = tk.BooleanVar(value=show_default)
        
        if slim:
            self.entries[label] = {
                "text": ent_text, "show": show_var
            }
        else:
            initial_state = "normal" if default.strip() else "disabled"
            show_cb = tk.Checkbutton(
                self.input_frame, variable=show_var, bg=AppConstants.BG_COLOR,
                state=initial_state, command=self.on_change_callback, takefocus=0
            )
            show_cb.grid(row=row, column=2, padx=0)
            self.entries[label] = {
                "text": ent_text, "show": show_var, "show_cb": show_cb
            }
        
        ent_text.bind("<KeyRelease>", lambda e, lbl=label: self._on_text_change(e, lbl))
        
        return ent_text, show_var
    
    def _on_text_change(self, event, label: str) -> None:
        """Handle text changes in entry fields (fires on every keystroke).

        Updates the Show checkbox state immediately. Mutual exclusivity clearing
        is debounced — the sibling field is wiped 400 ms after the last keystroke
        so it clears automatically during a natural typing pause without
        interrupting mid-entry.
        """
        if label not in self.entries:
            return
        
        entry_data = self.entries[label]
        text = entry_data["text"].get().strip()
        new_state = "normal" if text else "disabled"
        
        if "show_cb" in entry_data:
            entry_data["show_cb"].config(state=new_state)

        # Debounce mutual exclusivity: cancel any pending clear, schedule a new one.
        # The redraw callback is bundled into the debounced call so the canvas never
        # redraws while a conflicting sibling value is still present.
        if label in self._debounce_timers:
            self.input_frame.after_cancel(self._debounce_timers.pop(label))
        if text:
            def _clear_then_redraw(lbl=label):
                self._on_mutual_exclusive_clear(lbl)
                self.on_change_callback()
            timer_id = self.input_frame.after(200, _clear_then_redraw)
            self._debounce_timers[label] = timer_id
        else:
            # Field was cleared — redraw immediately (no sibling conflict possible)
            self.on_change_callback()

    def _on_mutual_exclusive_clear(self, label: str) -> None:
        """Clear sibling fields that are mutually exclusive with *label*.

        Called 400 ms after the last keystroke via debounce in _on_text_change,
        so the sibling is cleared automatically once the user pauses typing.
        """
        self._debounce_timers.pop(label, None)
        if label not in self.entries:
            return
        text = self.entries[label]["text"].get().strip()
        if not text:
            return
        if label == "Radius" and "Diameter" in self.entries:
            self.entries["Diameter"]["text"].delete(0, tk.END)
        elif label == "Diameter" and "Radius" in self.entries:
            self.entries["Radius"]["text"].delete(0, tk.END)
        elif label == "Height" and "Left Side" in self.entries:
            self.entries["Left Side"]["text"].delete(0, tk.END)
            self.entries["Right Side"]["text"].delete(0, tk.END)
        elif label in ("Left Side", "Right Side") and "Height" in self.entries:
            self.entries["Height"]["text"].delete(0, tk.END)
    
    def build_from_config(self, config: ShapeConfig, mode: str = "Default",
                          slim: bool = False) -> None:
        """Build input rows from a ShapeConfig. Slim mode omits Show checkboxes."""
        if slim and config.custom_labels:
            labels = config.custom_labels
            defaults = config.custom_values
        else:
            labels = config.labels
            defaults = config.get_defaults_for_mode(mode)
        
        # Batch widget creation to reduce redraws
        self.input_frame.update_idletasks()
        
        for i, lbl in enumerate(labels):
            default = defaults[i] if i < len(defaults) else ""
            self.create_entry_row(i + 1, lbl, default, slim=slim)
        
        self.input_frame.update_idletasks()
    
    def get_state(self) -> dict:
        """Return serialisable snapshot of all entry values."""
        return {"entries": {k: self.get_entry_value(k) for k in self.entries}}

    def set_state(self, state: dict) -> None:
        """Restore entry values from a snapshot dict."""
        for k, val in state.get("entries", {}).items():
            self.set_entry_value(k, val)

    def collect_params(self) -> dict[str, Any]:
        """Collect all entry values safely into a params dict."""
        params = {}
        for key, entry_data in self.entries.items():
            text = entry_data["text"].get().strip()
            params[key] = text
            num_key = f"{key}_num"
            
            # Strict numeric check: only convert to float if it looks like a number
            # This prevents 'a', 'b', or 'h' from crashing the math logic
            if text:
                try:
                    val = float(text)
                    # Store numeric value (including negatives) for validation
                    params[num_key] = val
                except (ValueError, TypeError):
                    # Not a number - store as None so math logic ignores it
                    params[num_key] = None
            else:
                params[num_key] = None
        return params


class PlotController:
    """Handles plot generation and drawing coordination."""
    
    def __init__(self, ax: Axes, canvas: FigureCanvasTkAgg, label_manager: LabelManager):
        self.ax = ax
        self.canvas = canvas
        self.label_manager = label_manager
        self.font_size = AppConstants.DEFAULT_FONT_SIZE
        self.line_width = AppConstants.DEFAULT_LINE_WIDTH
        self.font_family = AppConstants.DEFAULT_FONT_FAMILY
        self.label_manager.font_family = AppConstants.DEFAULT_FONT_FAMILY
        # Ensure white axes background
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
        
        # Consistent 1px border
        for spine in self.ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(1.0)
    
    def draw_shape(self, shape: str, ctx: DrawingContext, 
                   transform: TransformState, params: dict) -> str | None:
        """
        Draw a shape using the registry.
        Returns error message if drawing failed, None on success.
        """
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
        """Display an error message on the canvas."""
        self.ax.text(0.5, 0.5, f"⚠ {message}\n\nPlease check your input values.",
                    ha="center", va="center", fontsize=12, color="orange",
                    transform=self.ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", 
                             edgecolor="orange", alpha=1))
    
    def refresh(self) -> None:
        """Redraw the canvas."""
        self.canvas.draw()



class HistoryManager:
    """Manages undo/redo stacks by capturing and restoring application state."""
    def __init__(self, max_depth: int = 50):
        self.undo_stack: list[dict] = []
        self.redo_stack: list[dict] = []
        self.max_depth = max_depth
        self._is_restoring = False

    @property
    def is_restoring(self) -> bool:
        """Public read access to the restoring flag. Use this instead of _is_restoring externally."""
        return self._is_restoring

    @is_restoring.setter
    def is_restoring(self, value: bool) -> None:
        self._is_restoring = value

    @contextlib.contextmanager
    def restoring(self) -> Iterator[None]:
        """Context manager that sets is_restoring=True for the duration of the block.

        Guarantees the flag is cleared even if an exception occurs, preventing
        the undo stack from being permanently locked.

        Usage:
            with self.history_manager.restoring():
                self._apply_state(state)
        """
        self._is_restoring = True
        try:
            yield
        finally:
            self._is_restoring = False

    def capture_state(self, state: dict, force: bool = False) -> None:
        """Add new state to undo stack and clear redo stack.
        
        Args:
            state: State snapshot to push.
            force: If True, bypass deduplication and always push.  Use for
                   deliberate user actions (e.g. rotate) where the live state
                   happens to equal the stack top before the mutation fires.
        """
        if self._is_restoring:
            return

        if not force and self.undo_stack:
            prev = self.undo_stack[-1]
            # Compare all keys that exist in both states
            if self._states_equal(prev, state):
                return

        self.undo_stack.append(state)
        self.redo_stack.clear()
        
        if len(self.undo_stack) > self.max_depth:
            self.undo_stack.pop(0)
    
    def _states_equal(self, s1: dict, s2: dict) -> bool:
        """Deep comparison of two state dictionaries."""
        try:
            return s1 == s2
        except Exception:
            return False

    def undo(self, _=None) -> dict | None:
        if len(self.undo_stack) == 0:
            return None

        # Pop the current position onto the redo stack and return the entry
        # below it (one step back).  Pre-action captures ensure the stack top
        # always matches the live state before every user action, so the
        # "unsaved changes" branch that caused phantom no-op undos is gone.
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
    """Central manager for all slider variables/ranges/defaults."""
    def __init__(self, root: tk.Tk):
        self.root = root
        self.specs: dict[str, ScaleSpec] = {
            "aspect": ScaleSpec("aspect", AppConstants.SLIDER_MIN, AppConstants.SLIDER_MAX, AppConstants.SLIDER_DEFAULT),
            "slope": ScaleSpec("slope", -1.0, 1.0, 0.3),
            "peak_offset": ScaleSpec("peak_offset", -0.5, 1.5, 0.5),
            "view_scale": ScaleSpec("view_scale", 0.25, 1.0, 1.0),  # 1.0=fill (far right), 0.25=min
        }
        # Eagerly create all vars so defaults are applied at construction time
        self._vars: dict[str, tk.DoubleVar] = {
            key: tk.DoubleVar(master=self.root, value=spec.default)
            for key, spec in self.specs.items()
        }

    def var(self, key: str) -> tk.DoubleVar:
        if key not in self.specs:
            raise KeyError(f"Unknown scale key: {key}")
        return self._vars[key]

    def reset(self, key: str) -> None:
        self.var(key).set(self.specs[key].default)

    def reset_many(self, keys: list[str]) -> None:
        for k in keys:
            self.reset(k)

    def get_state(self) -> dict:
        """Return serialisable snapshot of all scale values."""
        return {"scales": {k: self.var(k).get() for k in self.specs}}

    def set_state(self, state: dict) -> None:
        """Restore scale values from a snapshot dict."""
        for k, v in state.get("scales", {}).items():
            try:
                self.var(k).set(v)
            except (KeyError, Exception):
                pass


