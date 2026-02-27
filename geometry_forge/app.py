from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Any
import math
import platform
import os
import subprocess
import logging
from io import BytesIO

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as patches

from .models import (
    Point, Polygon, StandaloneDimLine, CompositeDimLine,
    AppConstants, DrawingContext, TransformState, ShapeConfig,
    ShapeConfigProvider, ShapeFeature, TriangleType, PolygonType,
    ValidationError, SHAPE_CAPABILITIES,
)
from .validators import ShapeValidator
from .drawing import DrawingUtilities, SmartGeometryEngine, GeometricRotation
from .labels import LabelManager
from .drawers import ShapeDrawer, ShapeRegistry
from .controllers import (
    TransformController, InputController, PlotController,
    HistoryManager, ScaleManager,
)
from .widgets import CompositeTransferList
from .composite_controller import CompositeDragController
from .standalone_controller import StandaloneAnnotationController

logger = logging.getLogger(__name__)


class GeometryApp:
    """Main application class for Geometry Forge."""

    # Ordered list of (label_key, snapshot_key) pairs for toggle-based labels
    # (radius/diameter/circumference and preset dim labels).  This is the single
    # source of truth used by _build_state_snapshot, _apply_state, and
    # generate_plot — add new toggle labels here only.
    TOGGLE_LABEL_KEYS: list[tuple[str, str]] = [
        ("Circumference",  "circ"),
        ("Radius",         "radius"),
        ("Diameter",       "diameter"),
        ("Height",         "height"),
        ("Slant",          "slant"),
        ("Length (Front)", "length_front"),
        ("Width (Side)",   "width_side"),
        ("Base (Tri)",     "base_tri"),
        ("Height (Tri)",   "height_tri"),
        ("Length (Prism)", "length_prism"),
    ]

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(AppConstants.WINDOW_TITLE)
        self.font_size = AppConstants.DEFAULT_FONT_SIZE
        self.line_width = AppConstants.DEFAULT_LINE_WIDTH
        self.font_family = AppConstants.DEFAULT_FONT_FAMILY
        self._redraw_after_id: str | None = None
        self._slider_after_id: str | None = None
        self._capture_after_id: str | None = None
        # Composite drag controller — owns positions, transforms, selection,
        # labels, dim_lines, drag/marquee state, snap, and all action methods.
        # Instantiated after label_manager / history_manager (below).
        
        # Standalone annotation controller — owns labels, dim_lines, selection state,
        # and all four mouse-event handlers for non-composite shapes.
        # Instantiated after label_manager / history_manager (below) because
        # __init__ only captures a reference to self (app).
        # Accessed as self.standalone_ctrl; backward-compat delegates kept below.
        self._shape_bounds: dict | None = None
        self._standalone_event_ids: list[int] = []
        self._composite_event_ids: list[int] = []
        self._label_drag_state: dict | None = None
        self._label_hover_active: bool = False
        self._shape_pan_offset: tuple[float, float] = (0.0, 0.0)  # data-unit pan offset
        self._scale_after_id: str | None = None
        # Transform/render state — set during generate_plot; initialized here so
        # all code paths can read them without getattr fallbacks.
        # NOTE: _builtin_selected is a @property backed by standalone_ctrl and is
        # NOT assigned here — standalone_ctrl initialises it in its own __init__.
        self._suppress_preset_dim_snap: bool = False
        self._axes_pixel_aspect: float = 4.0 / 3.0
        self._canonical_pre_rotation_center: tuple[float, float] | None = None
        self._pre_rotation_center: tuple[float, float] | None = None
        self._flip_center: tuple[float, float] | None = None
        self._canvas_target_w: int = 970
        self._canvas_target_h: int = 727
        self.label_manager = LabelManager()
        self.history_manager = HistoryManager()
        # Controllers must be instantiated AFTER label_manager / history_manager
        # and BEFORE any property that routes to them is triggered.
        self.standalone_ctrl = StandaloneAnnotationController(self)
        self.composite_ctrl = CompositeDragController(self)

        # Central slider/range manager
        self.scale_manager = ScaleManager(self.root)
        
        self.triangle_type_var = tk.StringVar(value="Custom")
        self.polygon_type_var = tk.StringVar(value=PolygonType.PENTAGON.value)
        self.dimension_mode_var = tk.StringVar(value="Default")
        self.show_hashmarks_var = tk.BooleanVar(value=False)

        # UI widget attributes — initialized to None so guards can use `is not None`
        # instead of hasattr(). Set to real widgets during _setup_layout/_setup_canvas.
        self.fig = None
        self.canvas = None
        self.composite_transfer = None
        self.col_shape_type = None
        self.col_transforms = None
        self.col_inputs = None
        self.col_dimlines = None
        self.col_tools = None
        self.controls_row = None
        self.center_container = None
        self.mode_help_label = None
        self.clear_workspace_btn = None
        self.save_btn = None
        self.copy_btn = None
        self.default_btn = None
        self.font_label = None
        self.font_spin = None
        self.font_sans_btn = None
        self.font_serif_btn = None
        self.weight_label = None
        self.line_width_spin = None
        self.font_size_var = None
        self.line_width_var = None
        self._dim_status_label = None
        self._composite_label_entry = None
        self._composite_text_btn = None
        self._composite_cancel_btn = None
        self._composite_bboxes = []
        self._composite_snap_anchors = []
        self._composite_view_limits = None
        self._composite_dim_label_bboxes = []
        self._composite_label_bboxes = []
        self._composite_dim_endpoints = []
        self._composite_dim_lines = []
        self._standalone_label_entry = None
        self._standalone_text_btn = None
        self._standalone_cancel_btn = None
        self._standalone_cancel_dim_btn = None
        self._standalone_free_btn = None
        self._dim_preview_line = None
        self.tri_buttons = None
        self.poly_buttons = None

        self._setup_data()
        self._setup_layout()
        self._setup_canvas_and_controllers()
    # Button press feedback removed - macOS theming is inconsistent; use default OS behavior

    # ---------------- Error/Info messaging (standardized) ----------------
    def _ui_error(self, title: str, message: str, *, exc: Exception | None = None) -> None:
        if exc is not None:
            logger.error(
                f"{title}: {message}",
                exc_info=(type(exc), exc, exc.__traceback__)
            )
        else:
            logger.error(f"{title}: {message}")
        messagebox.showerror(title, message)

    def _ui_info(self, title: str, message: str) -> None:
        logger.info(f"{title}: {message}")
        messagebox.showinfo(title, message)

    def _ui_confirm_yesno(self, title: str, message: str) -> bool:
        logger.info(f"{title}: {message}")
        return bool(messagebox.askyesno(title, message))

    # ── CompositeDragController backward-compat properties ──────────────────
    # Route all existing self._composite_* accesses to the controller.

    @property
    def _composite_positions(self): return self.composite_ctrl.positions
    @_composite_positions.setter
    def _composite_positions(self, v): self.composite_ctrl.positions = v

    @property
    def _composite_transforms(self): return self.composite_ctrl.transforms
    @_composite_transforms.setter
    def _composite_transforms(self, v): self.composite_ctrl.transforms = v

    @property
    def _composite_selected(self): return self.composite_ctrl.selected
    @_composite_selected.setter
    def _composite_selected(self, v): self.composite_ctrl.selected = v

    @property
    def _composite_labels(self): return self.composite_ctrl.labels
    @_composite_labels.setter
    def _composite_labels(self, v): self.composite_ctrl.labels = v

    @property
    def _composite_label_drag(self): return self.composite_ctrl.label_drag
    @_composite_label_drag.setter
    def _composite_label_drag(self, v): self.composite_ctrl.label_drag = v

    @property
    def _composite_selected_label(self): return self.composite_ctrl.selected_label
    @_composite_selected_label.setter
    def _composite_selected_label(self, v): self.composite_ctrl.selected_label = v

    @property
    def _composite_dim_lines(self): return self.composite_ctrl.dim_lines
    @_composite_dim_lines.setter
    def _composite_dim_lines(self, v): self.composite_ctrl.dim_lines = v

    @property
    def _composite_selected_dim(self): return self.composite_ctrl.selected_dim
    @_composite_selected_dim.setter
    def _composite_selected_dim(self, v): self.composite_ctrl.selected_dim = v

    @property
    def _composite_dim_mode(self): return self.composite_ctrl.dim_mode
    @_composite_dim_mode.setter
    def _composite_dim_mode(self, v): self.composite_ctrl.dim_mode = v

    @property
    def _composite_dim_first_point(self): return self.composite_ctrl.dim_first_point
    @_composite_dim_first_point.setter
    def _composite_dim_first_point(self, v): self.composite_ctrl.dim_first_point = v

    @property
    def _composite_edit_mode(self): return self.composite_ctrl.edit_mode
    @_composite_edit_mode.setter
    def _composite_edit_mode(self, v): self.composite_ctrl.edit_mode = v

    @property
    def _composite_drag_state(self): return self.composite_ctrl.drag_state
    @_composite_drag_state.setter
    def _composite_drag_state(self, v): self.composite_ctrl.drag_state = v

    @property
    def _composite_marquee(self): return self.composite_ctrl.marquee
    @_composite_marquee.setter
    def _composite_marquee(self, v): self.composite_ctrl.marquee = v

    @property
    def _composite_snap_guides(self): return self.composite_ctrl.snap_guides
    @_composite_snap_guides.setter
    def _composite_snap_guides(self, v): self.composite_ctrl.snap_guides = v

    # ── CompositeDragController delegate methods ──────────────────────────────

    def _connect_composite_drag(self) -> None:
        self.composite_ctrl.connect()

    def _disconnect_composite_drag(self) -> None:
        self.composite_ctrl.disconnect()

    def _on_composite_press(self, event) -> None:
        self.composite_ctrl.on_press(event)

    def _on_composite_motion(self, event) -> None:
        self.composite_ctrl.on_motion(event)

    def _on_composite_release(self, event) -> None:
        self.composite_ctrl.on_release(event)

    def _translate_patch(self, patch, dx: float, dy: float) -> None:
        self.composite_ctrl.translate_patch(patch, dx, dy)

    def _calc_snap(self, drag_idx: int, drag_bbox: tuple,
                   all_bboxes: list[tuple]) -> tuple[float, float, list]:
        return self.composite_ctrl.calc_snap(drag_idx, drag_bbox, all_bboxes)

    def _update_composite_selection_ui(self) -> None:
        self.composite_ctrl.update_selection_ui()

    def _get_group_center(self) -> tuple[float, float]:
        return self.composite_ctrl.get_group_center()

    def _get_group_bbox(self) -> tuple[float, float, float, float]:
        return self.composite_ctrl.get_group_bbox()

    def _all_shapes_selected(self) -> bool:
        return self.composite_ctrl.all_shapes_selected()

    def _get_annotations_in_region(self, x_min: float, y_min: float,
                                    x_max: float, y_max: float,
                                    include_all: bool = False) -> tuple[list[int], list[int]]:
        return self.composite_ctrl.get_annotations_in_region(x_min, y_min, x_max, y_max, include_all)

    def _rotate_annotations(self, label_idxs: list[int], dim_idxs: list[int],
                             cx: float, cy: float, direction: int) -> None:
        self.composite_ctrl.rotate_annotations(label_idxs, dim_idxs, cx, cy, direction)

    def _flip_annotations_h(self, label_idxs: list[int], dim_idxs: list[int], cx: float) -> None:
        self.composite_ctrl.flip_annotations_h(label_idxs, dim_idxs, cx)

    def _flip_annotations_v(self, label_idxs: list[int], dim_idxs: list[int], cy: float) -> None:
        self.composite_ctrl.flip_annotations_v(label_idxs, dim_idxs, cy)

    def _start_dim_line_mode(self, mode: str) -> None:
        self.composite_ctrl.start_dim_mode(mode)

    def _cancel_dim_line_mode(self) -> None:
        self.composite_ctrl.cancel_dim_mode()

    def _handle_dim_line_click(self, mx: float, my: float) -> bool:
        return self.composite_ctrl.handle_dim_click(mx, my)

    def _add_composite_label(self) -> None:
        self.composite_ctrl.add_label()

    def _remove_composite_label(self) -> None:
        self.composite_ctrl.remove_label()

    def _confirm_composite_label(self) -> None:
        self.composite_ctrl.confirm_label()

    def _remove_selected_annotation(self) -> None:
        self.composite_ctrl.remove_selected_annotation()

    def _enter_composite_edit(self, edit_type: str, idx: int) -> None:
        self.composite_ctrl.enter_edit(edit_type, idx)

    def _apply_composite_edit(self) -> None:
        self.composite_ctrl.apply_edit()

    def _cancel_composite_edit(self) -> None:
        self.composite_ctrl.cancel_edit()

    def _composite_flip(self, axis: str) -> None:
        self.composite_ctrl.flip(axis)

    def _composite_flip_h(self) -> None:
        self.composite_ctrl.flip_h()

    def _composite_flip_v(self) -> None:
        self.composite_ctrl.flip_v()

    def _composite_rotate(self, direction: int) -> None:
        self.composite_ctrl.rotate(direction)

    # ── end CompositeDragController delegation ────────────────────────────────

    # ── StandaloneAnnotationController backward-compat properties ───────────
    # These route all existing self._standalone_* / self._builtin_selected
    # accesses to the controller without changing any call site.

    @property
    def _standalone_labels(self): return self.standalone_ctrl.labels
    @_standalone_labels.setter
    def _standalone_labels(self, v): self.standalone_ctrl.labels = v

    @property
    def _standalone_selected_label(self): return self.standalone_ctrl.selected_label
    @_standalone_selected_label.setter
    def _standalone_selected_label(self, v): self.standalone_ctrl.selected_label = v

    @property
    def _standalone_label_bboxes(self): return self.standalone_ctrl.label_bboxes
    @_standalone_label_bboxes.setter
    def _standalone_label_bboxes(self, v): self.standalone_ctrl.label_bboxes = v

    @property
    def _standalone_dim_lines(self): return self.standalone_ctrl.dim_lines
    @_standalone_dim_lines.setter
    def _standalone_dim_lines(self, v): self.standalone_ctrl.dim_lines = v

    @property
    def _standalone_selected_dim(self): return self.standalone_ctrl.selected_dim
    @_standalone_selected_dim.setter
    def _standalone_selected_dim(self, v): self.standalone_ctrl.selected_dim = v

    @property
    def _standalone_dim_endpoints(self): return self.standalone_ctrl.dim_endpoints
    @_standalone_dim_endpoints.setter
    def _standalone_dim_endpoints(self, v): self.standalone_ctrl.dim_endpoints = v

    @property
    def _standalone_dim_label_bboxes(self): return self.standalone_ctrl.dim_label_bboxes
    @_standalone_dim_label_bboxes.setter
    def _standalone_dim_label_bboxes(self, v): self.standalone_ctrl.dim_label_bboxes = v

    @property
    def _standalone_dim_mode(self): return self.standalone_ctrl.dim_mode
    @_standalone_dim_mode.setter
    def _standalone_dim_mode(self, v): self.standalone_ctrl.dim_mode = v

    @property
    def _standalone_dim_first_point(self): return self.standalone_ctrl.dim_first_point
    @_standalone_dim_first_point.setter
    def _standalone_dim_first_point(self, v): self.standalone_ctrl.dim_first_point = v

    @property
    def _standalone_dim_preview_line(self): return self.standalone_ctrl.dim_preview_line
    @_standalone_dim_preview_line.setter
    def _standalone_dim_preview_line(self, v): self.standalone_ctrl.dim_preview_line = v

    @property
    def _standalone_edit_mode(self): return self.standalone_ctrl.edit_mode
    @_standalone_edit_mode.setter
    def _standalone_edit_mode(self, v): self.standalone_ctrl.edit_mode = v

    @property
    def _builtin_selected(self): return self.standalone_ctrl.builtin_selected
    @_builtin_selected.setter
    def _builtin_selected(self, v): self.standalone_ctrl.builtin_selected = v

    @property
    def _builtin_dim_endpoints(self): return self.standalone_ctrl.builtin_dim_endpoints
    @_builtin_dim_endpoints.setter
    def _builtin_dim_endpoints(self, v): self.standalone_ctrl.builtin_dim_endpoints = v

    # ── StandaloneAnnotationController delegate methods ───────────────────────
    # Keeps all existing call sites working without modification.

    def _connect_standalone_drag(self) -> None:
        self.standalone_ctrl.connect()

    def _disconnect_standalone_drag(self) -> None:
        self.standalone_ctrl.disconnect()

    def _pixels_to_data_pad(self, px: float) -> tuple[float, float]:
        return self.standalone_ctrl.pixels_to_data_pad(px)

    def _find_nearest_label(self, x: float, y: float, threshold: float = 0.8) -> str | None:
        return self.standalone_ctrl.find_nearest_label(x, y, threshold)

    def _on_standalone_label_hover(self, event) -> None:
        self.standalone_ctrl.on_hover(event)

    def _on_standalone_label_press(self, event) -> None:
        self.standalone_ctrl.on_press(event)

    def _on_standalone_label_motion(self, event) -> None:
        self.standalone_ctrl.on_motion(event)

    def _on_standalone_label_release(self, event) -> None:
        self.standalone_ctrl.on_release(event)

    def _cancel_standalone_dim_mode(self) -> None:
        self.standalone_ctrl.cancel_dim_mode()

    def _add_standalone_label(self) -> None:
        self.standalone_ctrl.add_label()

    def _add_standalone_dim_preset(self, preset_key: str, default_text: str) -> None:
        self.standalone_ctrl.add_dim_preset(preset_key, default_text)

    def _remove_standalone_annotation(self) -> None:
        self.standalone_ctrl.remove_annotation()

    def _confirm_standalone_label(self) -> None:
        self.standalone_ctrl.confirm_label()

    def _enter_standalone_edit(self, edit_type: str, idx: int) -> None:
        self.standalone_ctrl.enter_edit(edit_type, idx)

    def _apply_standalone_edit(self) -> None:
        self.standalone_ctrl.apply_edit()

    def _cancel_standalone_edit(self) -> None:
        self.standalone_ctrl.cancel_edit()

    def _get_preset_value_key(self, preset_key: str) -> str | None:
        """Map a dim line preset key to the Custom mode entry key for value population."""
        shape = self.shape_var.get()
        tri_type = self.triangle_type_var.get()
        mapping = {
            "Rectangle": {"height": "Width", "width": "Length"},
            "Parallelogram": {
                "para_height": "Height", "para_base": "Length",
                "para_side_l": "Side", "para_side_r": "Side",
            },
            "Trapezoid": {
                "trap_height": "Height", "trap_base": "Bottom Base",
                "trap_top": "Top Base", "trap_side_l": "Left Side", "trap_side_r": "Right Side",
            },
            "Cylinder": {"height": "Height", "radius": "Radius", "diameter": "Diameter"},
            "Cone": {"height": "Height", "radius": "Radius", "diameter": "Diameter"},
            "Rectangular Prism": {
                "height": "Height", "width": "Length (Front)", "length": "Width (Side)",
            },
            "Triangular Prism": {
                "height": "Height", "tri_base": "Base", "tri_length": "Length",
            },
            "Tri Prism": {
                "height": "Height", "tri_base": "Base", "tri_length": "Length",
            },
        }
        if shape == "Triangle" and tri_type == "Custom":
            tri_mapping = {
                "height": "Height", "width": "Base Width",
                "side_l": "Left Side", "side_r": "Right Side",
            }
            return tri_mapping.get(preset_key)
        shape_map = mapping.get(shape, {})
        return shape_map.get(preset_key)

    # ── end StandaloneAnnotationController delegation ─────────────────────────

    def _setup_data(self) -> None:
        self.shape_data = {
            "2D Figures": [
                "Rectangle", "Square", "Triangle", "Circle",
                "Parallelogram", "Trapezoid",
                "Polygon"
            ],
            "3D Solids": [
                "Sphere", "Hemisphere", "Cylinder", "Cone",
                "Rectangular Prism", "Triangular Prism"
            ],
            "Angles & Lines": [
                "Angle (Adjustable)",
                "Parallel Lines & Transversal",
                "Complementary Angles", "Supplementary Angles", "Vertical Angles",
                "Line Segment"
            ],
            "Composite Figures": [
                "2D Composite", "3D Composite"
            ]
        }

    def _setup_layout(self) -> None:
        """Build layout: top bar, 5-column controls row, full-width canvas, shortcut bar.
        
        Window width is determined by the controls row content.
        Canvas uses 4:3 aspect ratio which determines window height.
        """
        # Root grid: top bar, controls, canvas, shortcut bar
        self.root.rowconfigure(0, weight=0)  # top bar
        self.root.rowconfigure(1, weight=0)  # controls row (fixed height)
        self.root.rowconfigure(2, weight=1)  # canvas (fills remaining)
        self.root.rowconfigure(3, weight=0)  # shortcut bar
        self.root.columnconfigure(0, weight=1)
        
        self._create_top_bar()
        self._create_controls_row()
        self._create_canvas_area()
        self._create_shortcut_bar()
    
    def _size_window_to_content(self) -> None:
        """Calculate window size: top bar (all buttons visible) sets width, canvas is 4:3."""
        # Temporarily show Save/Copy so they're included in width measurement
        self.save_btn.grid(row=0, column=10, padx=1, sticky="e")
        self.copy_btn.grid(row=0, column=11, padx=1, sticky="e")
        self.root.update_idletasks()
        
        # Measure top bar with all buttons visible
        top_width = self.center_container.winfo_reqwidth() + 20
        controls_width = self.controls_row.winfo_reqwidth() + 20 if self.controls_row is not None else 0
        app_width = max(top_width, controls_width)
        
        # Hide Save/Copy again (show_welcome will control visibility)
        self.save_btn.grid_remove()
        self.copy_btn.grid_remove()
        
        # Canvas fills the full app width, 4:3 ratio sets height
        canvas_w = app_width
        canvas_h = int(canvas_w * 3 / 4)
        
        # Total height
        self.root.update_idletasks()
        top_h = self.top_bar.winfo_reqheight()
        controls_h = self.controls_row.winfo_reqheight() if self.controls_row is not None else 0
        shortcut_h = 25
        total_height = top_h + controls_h + canvas_h + shortcut_h
        
        # Store for figure sizing
        self._canvas_target_w = canvas_w
        self._canvas_target_h = canvas_h
        
        self.root.geometry(f"{app_width}x{total_height}")
    
    def _create_top_bar(self) -> None:
        """Create the top bar with category, shape, font, weight, and action controls."""
        top_bar = tk.Frame(self.root, bg=AppConstants.BG_COLOR)
        top_bar.grid(row=0, column=0, sticky="ew", padx=5, pady=(3, 0))
        self.top_bar = top_bar
        
        # Center everything in one row
        self.center_container = tk.Frame(top_bar, bg=AppConstants.BG_COLOR)
        self.center_container.pack(anchor="center")
        
        self._create_category_selector()
        self._create_shape_selector()
        self._create_font_selector()
    
    def _create_controls_row(self) -> None:
        """Create the 5-column controls row above the canvas, spread full width."""
        controls = tk.Frame(self.root, bg=AppConstants.BG_COLOR)
        controls.grid(row=1, column=0, sticky="ew", padx=8, pady=(2, 0))
        self.controls_row = controls
        
        # All columns expand evenly to fill width
        controls.columnconfigure(0, weight=1)  # shape type
        controls.columnconfigure(1, weight=1)  # transforms
        controls.columnconfigure(2, weight=1)  # inputs
        controls.columnconfigure(3, weight=1)  # labels and lines
        controls.columnconfigure(4, weight=1)  # tools
        controls.rowconfigure(0, weight=0)
        
        # Col 0: Shape Type (triangle type, polygon type, default/custom)
        self.col_shape_type = tk.Frame(controls, bg=AppConstants.BG_COLOR)
        self.col_shape_type.grid(row=0, column=0, sticky="n", padx=4)
        self.col_shape_type.grid_remove()  # hidden until needed
        
        self.shape_options_frame = tk.Frame(self.col_shape_type, bg=AppConstants.BG_COLOR)
        self.shape_options_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))
        
        # Col 1: Transforms & Sliders
        self.col_transforms = tk.Frame(controls, bg=AppConstants.BG_COLOR)
        self.col_transforms.grid(row=0, column=1, sticky="n", padx=4)
        self.col_transforms.grid_remove()  # hidden until needed
        
        self.options_header = tk.Label(
            self.col_transforms, text="Options", bg=AppConstants.BG_COLOR,
            font=("Arial", 9, "bold")
        )
        self._create_transform_controls()
        self.adjust_sliders_frame = tk.Frame(self.col_transforms, bg=AppConstants.BG_COLOR)
        self.adjust_sliders_frame.pack(side=tk.TOP, pady=(0, 0), fill=tk.X)
        
        # Col 2: Inputs
        self.col_inputs = tk.Frame(controls, bg=AppConstants.BG_COLOR)
        self.col_inputs.grid(row=0, column=2, sticky="n", padx=4)
        self.col_inputs.grid_remove()  # hidden until shape selected
        
        self.input_frame = tk.Frame(self.col_inputs, bg=AppConstants.BG_COLOR)
        self.input_frame.pack(side=tk.TOP, anchor="n", fill=tk.X)
        
        # Col 3: Labels and Lines
        self.col_dimlines = tk.Frame(controls, bg=AppConstants.BG_COLOR)
        self.col_dimlines.grid(row=0, column=3, sticky="n", padx=4)
        self.col_dimlines.grid_remove()  # hidden until shape selected
        
        # Col 4: Tools
        self.col_tools = tk.Frame(controls, bg=AppConstants.BG_COLOR)
        self.col_tools.grid(row=0, column=4, sticky="n", padx=4)
        self.col_tools.grid_remove()  # hidden until shape selected
        
        self._create_right_tools_panel()
    
    def _create_canvas_area(self) -> None:
        """Create the full-width canvas frame below the controls row."""
        self.col_canvas = tk.Frame(self.root, bg='#e8e8e8')
        self.col_canvas.grid(row=2, column=0, sticky="nsew")
    
    def _create_shortcut_bar(self) -> None:
        """Create the keyboard shortcut hint bar."""
        shortcut_bar = tk.Frame(self.root, bg=AppConstants.BG_COLOR, height=20)
        shortcut_bar.grid(row=3, column=0, sticky="ew")
        shortcut_text = (
            "Reflect [H/V]     •     Rotate [R/L]     •     Undo/Redo [Ctrl+Z/Y]     •     "
            "Save [Ctrl+S]     •     Copy [Ctrl+C]"
        )
        tk.Label(
            shortcut_bar, text=shortcut_text, bg=AppConstants.BG_COLOR, fg="#555555",
            font=AppConstants.BTN_FONT
        ).pack(expand=True, pady=2)
    
    def _create_category_selector(self) -> None:
        tk.Label(self.center_container, text="Category:", bg=AppConstants.BG_COLOR, width=8, anchor="e").grid(row=0, column=0, sticky="e", pady=5)
        self.cat_var = tk.StringVar()
        self.cat_combo = ttk.Combobox(self.center_container, textvariable=self.cat_var, state="readonly", width=15)
        self.cat_combo["values"] = list(self.shape_data.keys())
        self.cat_combo.set("Select Category")
        self.cat_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.cat_combo.bind("<<ComboboxSelected>>", self.update_shape_list)
    
    def _create_shape_selector(self) -> None:
        tk.Label(self.center_container, text="Shape:", bg=AppConstants.BG_COLOR, width=8, anchor="e").grid(row=0, column=2, sticky="e", pady=5)
        self.shape_var = tk.StringVar()
        self.shape_combo = ttk.Combobox(self.center_container, textvariable=self.shape_var, state="readonly", width=15)
        self.shape_combo.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        self.shape_combo.bind("<<ComboboxSelected>>", self.update_inputs)
    
    def _create_font_selector(self) -> None:
        self.font_label = tk.Label(self.center_container, text="Font:", bg=AppConstants.BG_COLOR)
        self.font_label.grid(row=0, column=4, padx=(8, 0), pady=5)
        self.font_size_var = tk.IntVar(value=AppConstants.DEFAULT_FONT_SIZE)
        self.font_spin = tk.Spinbox(
            self.center_container, from_=AppConstants.MIN_FONT_SIZE, to=AppConstants.MAX_FONT_SIZE, width=3,
            textvariable=self.font_size_var,
            command=self.update_font_size
        )
        self.font_spin.grid(row=0, column=5, padx=2, pady=5)
        self.font_spin.bind('<FocusOut>', lambda e: self._on_font_blur())
        self.font_spin.bind('<Return>', lambda e: self.update_font_size())
        self.font_spin.bind('<KP_Enter>', lambda e: self.update_font_size())

        self._font_family = AppConstants.DEFAULT_FONT_FAMILY
        self.font_sans_btn = tk.Button(
            self.center_container, text="Aa", font=AppConstants.BTN_FONT,
            relief=tk.SUNKEN, bg=AppConstants.ACTIVE_BUTTON_COLOR,
            command=self._set_font_sans
        )
        self.font_sans_btn.grid(row=0, column=6, padx=(4, 1), pady=5)
        self.font_serif_btn = tk.Button(
            self.center_container, text="Aa", font=("Times New Roman", AppConstants.BTN_FONT[1]),
            relief=tk.RAISED, bg=AppConstants.DEFAULT_BUTTON_COLOR,
            command=self._set_font_serif
        )
        self.font_serif_btn.grid(row=0, column=7, padx=(1, 4), pady=5)
        self.weight_label = tk.Label(self.center_container, text="Weight:", bg=AppConstants.BG_COLOR)
        self.weight_label.grid(row=0, column=8, padx=(4, 0), pady=5)
        self.line_width_var = tk.IntVar(value=AppConstants.DEFAULT_LINE_WIDTH)
        self.line_width_spin = tk.Spinbox(
            self.center_container, from_=AppConstants.MIN_LINE_WIDTH, to=AppConstants.MAX_LINE_WIDTH, width=2,
            textvariable=self.line_width_var,
            command=self.update_line_width
        )
        self.line_width_spin.grid(row=0, column=9, padx=2, pady=5)
        self.line_width_spin.bind('<FocusOut>', lambda e: self._on_line_width_blur())
        self.line_width_spin.bind('<Return>', lambda e: self.update_line_width())
        self.line_width_spin.bind('<KP_Enter>', lambda e: self.update_line_width())


    def _setup_options_panel(self) -> None:
        """Create Save/Copy buttons in the top bar."""
        self.save_btn = tk.Button(self.center_container, text="Save", font=AppConstants.BTN_FONT, command=self.save_image)
        self.save_btn.grid(row=0, column=10, padx=1, sticky="e")
        self.save_btn.grid_remove()

        self.copy_btn = tk.Button(self.center_container, text="Copy", font=AppConstants.BTN_FONT, command=self.copy_to_clipboard)
        self.copy_btn.grid(row=0, column=11, padx=1, sticky="e")
        self.copy_btn.grid_remove()
    
    def _create_transform_controls(self) -> None:
        """Create flip/rotate controls inside col_transforms."""
        self.transform_frame = tk.Frame(self.col_transforms, bg=AppConstants.BG_COLOR)
        self.transform_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Reflect controls
        self.flip_row = tk.Frame(self.transform_frame, bg=AppConstants.BG_COLOR)
        self.flip_row.columnconfigure(0, weight=0)
        self.flip_row.columnconfigure(1, weight=1)
        self.flip_row.columnconfigure(2, weight=1)
        
        self.flip_label = tk.Label(self.flip_row, text="Reflect:", bg=AppConstants.BG_COLOR, anchor="e")
        self.flip_h_btn = tk.Button(
            self.flip_row, text="↔", height=1, font=("Arial", 14),
            command=lambda: self._flip_with_annotations('h')
        )
        self.flip_v_btn = tk.Button(
            self.flip_row, text="↕", height=1, font=("Arial", 14),
            command=lambda: self._flip_with_annotations('v')
        )
        self.flip_label.grid(row=0, column=0, padx=(0, 5), sticky="e")
        self.flip_h_btn.grid(row=0, column=1, padx=1, sticky="ew")
        self.flip_v_btn.grid(row=0, column=2, padx=1, sticky="ew")
        
        # Rotate controls
        self.rotate_row = tk.Frame(self.transform_frame, bg=AppConstants.BG_COLOR)
        self.rotate_row.columnconfigure(0, weight=0)
        self.rotate_row.columnconfigure(1, weight=1)
        self.rotate_row.columnconfigure(2, weight=1)
        
        self.rotate_label = tk.Label(self.rotate_row, text="Rotate:", bg=AppConstants.BG_COLOR, anchor="e")
        self.rotate_ccw_btn = tk.Button(
            self.rotate_row, text="↺", height=1, font=("Arial", 14),
            command=lambda: self._on_rotate_click(-1)
        )
        self.rotate_cw_btn = tk.Button(
            self.rotate_row, text="↻", height=1, font=("Arial", 14),
            command=lambda: self._on_rotate_click(1)
        )
        
        self.rotation_state_label = tk.Label(
            self.transform_frame, text="", bg=AppConstants.BG_COLOR, fg="blue", font=("Arial", 8, "italic")
        )
        
        self.rotate_label.grid(row=0, column=0, padx=(0, 5), sticky="e")
        self.rotate_ccw_btn.grid(row=0, column=1, padx=1, sticky="ew")
        self.rotate_cw_btn.grid(row=0, column=2, padx=1, sticky="ew")

    def _create_right_tools_panel(self) -> None:
        """Create tools panel in col_tools."""
        self.right_panel_frame = self.col_tools
        
        self.tools_header = tk.Label(
            self.right_panel_frame, text="Tools", bg=AppConstants.BG_COLOR,
            font=("Arial", 9, "bold"), width=22
        )
        
        # Undo/Redo
        self.undo_redo_frame = tk.Frame(self.right_panel_frame, bg=AppConstants.BG_COLOR)
        self.undo_redo_frame.columnconfigure(0, weight=1)
        self.undo_redo_frame.columnconfigure(1, weight=1)
        self.undo_btn = tk.Button(self.undo_redo_frame, text="Undo", font=AppConstants.BTN_FONT, state="disabled", command=self._undo_action)
        self.redo_btn = tk.Button(self.undo_redo_frame, text="Redo", font=AppConstants.BTN_FONT, state="disabled", command=self._redo_action)
        self.undo_btn.grid(row=0, column=0, padx=(0, 1), sticky="ew")
        self.redo_btn.grid(row=0, column=1, padx=(1, 0), sticky="ew")
        
        # Clear/Reset
        self.clear_workspace_btn = tk.Button(self.right_panel_frame, text="Clear Values & Labels", font=AppConstants.BTN_FONT, 
                                            command=self._clear_workspace, fg="red")
        
        # Help text
        self.mode_help_label = tk.Label(
            self.right_panel_frame, text="", bg=AppConstants.BG_COLOR, fg="gray",
            font=AppConstants.BTN_FONT, justify="left", wraplength=150
        )

        # Scale slider — created once here, shown/hidden by _pack_right_panel
        self.scale_frame = tk.Frame(self.right_panel_frame, bg=AppConstants.BG_COLOR)
        tk.Label(self.scale_frame, text="Scale:", bg=AppConstants.BG_COLOR,
                 font=("Arial", 8, "bold")).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Scale(
            self.scale_frame,
            variable=self.scale_manager.var("view_scale"),
            from_=0.25, to=1.0,
            orient="horizontal",
            command=lambda _: self._apply_view_scale_only()
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _setup_canvas_and_controllers(self) -> None:
        """Create canvas inside col_canvas (full-width) and initialize controllers."""
        # Save/Copy buttons must exist before window sizing
        self._setup_options_panel()
        
        # Size window based on top bar width, then derive canvas dimensions
        self._size_window_to_content()
        self.root.resizable(False, False)
        
        # Give col_canvas the exact target size BEFORE preventing propagation
        canvas_w = getattr(self, '_canvas_target_w', 970)
        canvas_h = getattr(self, '_canvas_target_h', 727)
        self.col_canvas.configure(width=canvas_w, height=canvas_h)
        self.col_canvas.grid_propagate(False)
        self.col_canvas.pack_propagate(False)
        
        # Pre-size figure to measured canvas dimensions so first draw is at full resolution
        _dpi = 100
        _fw = getattr(self, '_canvas_target_w', 970) / _dpi
        _fh = getattr(self, '_canvas_target_h', 727) / _dpi
        self.fig = Figure(figsize=(_fw, _fh), dpi=_dpi)
        self.fig.patch.set_facecolor('#e8e8e8')
        
        # [left, bottom, width, height] in fractions of figure size
        # Margin controls the grey border around the white paper.
        _mx = 0.03  # horizontal margin (left & right)
        _axes_w = 1 - 2 * _mx
        # Enforce true 4:3 white area: (canvas_w * _axes_w) / (canvas_h * _axes_h) = 4/3
        _axes_h = (canvas_w * _axes_w * 3.0) / (canvas_h * 4.0)
        _axes_h = min(_axes_h, 1.0 - 2 * _mx)  # safety only, should be ~0.94
        _bottom = (1.0 - _axes_h) / 2.0
        self.ax = self.fig.add_axes([_mx, _bottom, _axes_w, _axes_h])
        # Store true axes pixel aspect for use in _apply_view_scale
        self._axes_pixel_aspect = (canvas_w * _axes_w) / (canvas_h * _axes_h)
        self.ax.set_facecolor('#ffffff')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Add permanent 1px black border
        for spine in self.ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(1.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.col_canvas)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # Keep figure in sync with widget size; draw_idle() repaints after resize
        def _fit_figure(event=None):
            w = canvas_widget.winfo_width()
            h = canvas_widget.winfo_height()
            if w > 10 and h > 10:
                new_w = w / self.fig.dpi
                new_h = h / self.fig.dpi
                cur_w, cur_h = self.fig.get_size_inches()
                if abs(new_w - cur_w) > 0.05 or abs(new_h - cur_h) > 0.05:
                    self.fig.set_size_inches(new_w, new_h, forward=False)
                    # axes so the white area stays 4:3 after figure resize.
                    _mx = 0.03
                    _axes_w = 1 - 2 * _mx
                    _axes_h = (w * _axes_w * 3.0) / (h * 4.0)
                    _axes_h = min(_axes_h, 1.0 - 2 * _mx)
                    _bottom = (1.0 - _axes_h) / 2.0
                    self.ax.set_position([_mx, _bottom, _axes_w, _axes_h])
                    self._axes_pixel_aspect = (w * _axes_w) / (h * _axes_h)
                    self.canvas.draw_idle()
        
        canvas_widget.bind('<Configure>', _fit_figure)
        
        # Initialize controllers after all UI elements exist
        self.plot_controller = PlotController(self.ax, self.canvas, self.label_manager)
        self.input_controller = InputController(
            input_frame=self.input_frame,
            label_manager=self.label_manager,
            on_change_callback=self._on_input_change
        )
        self.transform_controller = TransformController(
            on_change_callback=self.generate_plot,
            flip_h_btn=self.flip_h_btn,
            flip_v_btn=self.flip_v_btn,
            rotate_ccw_btn=self.rotate_ccw_btn,
            rotate_cw_btn=self.rotate_cw_btn
        )
        
        # Connect standalone label drag events
        self._connect_standalone_drag()
        
        # Handle window close to clean up resources
        self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)
        
        # Setup keyboard shortcuts
        self._bind_shortcuts()
        
        self.show_welcome()
    
    def _on_window_close(self) -> None:
        """Clean up resources before closing the window."""
        self._disconnect_standalone_drag()

        if self.fig is not None:
            self.fig.clf()
        if self.canvas is not None:
            canvas_widget = self.canvas.get_tk_widget()
            if canvas_widget.winfo_exists():
                canvas_widget.destroy()

        # Force quit even if dropdown is open
        self.root.quit()
        self.root.destroy()

    def _on_rotate_click(self, direction: int) -> None:
        """Handle rotate button clicks."""
        if not self.shape_var.get():
            return
        if self._is_composite_shape():
            # Only transform if shapes are explicitly selected.
            # Auto-selecting all when nothing is selected is confusing:
            # pressing rotate with nothing selected should do nothing.
            if self._composite_selected:
                self._composite_rotate(direction)
            return
        shape = self.shape_var.get()
        config = ShapeConfigProvider.get(shape)
        num_sides = config.num_sides if config.num_sides > 0 else 4
        self._rotate_with_annotations(shape, direction, num_sides, config)
    
    def _rotate_base(self, direction: int) -> None:
        """Handle rotate keyboard shortcuts."""
        if not self.shape_var.get():
            return
        if self._is_composite_shape():
            # Only transform if shapes are explicitly selected.
            if self._composite_selected:
                self._composite_rotate(direction)
            return
        # Don't trigger if typing in entry field
        focused = self.root.focus_get()
        if isinstance(focused, tk.Entry):
            return
        shape = self.shape_var.get()
        config = ShapeConfigProvider.get(shape)
        num_sides = config.num_sides if config.num_sides > 0 else 4
        self._rotate_with_annotations(shape, direction, num_sides, config)
    
    def _get_shape_center(self) -> tuple[float, float] | None:
        """Return the geometry-only center of the current shape, or None.
        
        Uses _pre_rotation_center (computed from Polygon/Line2D/Arc artists only,
        excluding Text) to match the exact pivot used by rotate_axes_artists.
        This prevents annotation drift caused by text labels extending the
        bounding box asymmetrically.
        """
        gc = getattr(self, '_pre_rotation_center', None)
        if gc is not None:
            return gc
        b = getattr(self, '_shape_bounds', None)
        if not b:
            return None
        return ((b["x_min"] + b["x_max"]) / 2,
                (b["y_min"] + b["y_max"]) / 2)

    def _rotate_with_annotations(self, shape: str, direction: int,
                                  num_sides: int, config) -> None:
        """Rotate the shape and transform ALL standalone annotations to follow.

        ALL dim lines (preset and freeform) and labels are rotated by the same
        geometric angle the shape turns.  Preset re-snap is suppressed during
        rotation so the rotated coordinates are not overwritten.

        Since all shapes use the unified post-draw artist transform pipeline
        (draw canonical, then rotate/flip artists), the rotation angle is
        always the step angle -(360/n)*direction and the pivot is the bbox
        center.  This is the same approach used by composite mode.

        Flow:
        1. Snapshot ALL dim-line coords and pre-rotation center.
        2. Advance base_side silently (no callback).
        3. generate_plot() with re-snap suppressed → get new center.
        4. Compute rotation angle from step (unified for all shapes).
        5. Apply rotation to all annotation coords from snapshots.
        6. Final generate_plot() with re-snap suppressed → shows rotated annots.
        If no annotations exist, steps 3-6 collapse to a single generate_plot.
        """
        is_geometric = (shape in GeometricRotation.ALL_GEOMETRIC_SHAPES)

        if not is_geometric:
            self.transform_controller.rotate(direction, num_sides)
            self._update_rotation_label(config)
            # Capture AFTER mutating base_side — the new rotated state is now
            # distinct from the stack top, so deduplication won't suppress it.
            if not self.history_manager.is_restoring:
                self._capture_current_state()
            return

        has_annotations = bool(self._standalone_labels or self._standalone_dim_lines)

        # ── Step 0: capture BEFORE mutating so this rotation is always undoable ──
        if not self.history_manager.is_restoring:
            self._capture_current_state()

        # ── Step 1: snapshot pre-rotation coords ────────────────────────────────
        old_center = self._get_shape_center()
        old_dim_coords: list[dict] = []
        for dim in self._standalone_dim_lines:
            old_dim_coords.append({
                "x1": dim["x1"], "y1": dim["y1"],
                "x2": dim["x2"], "y2": dim["y2"],
                "label_x": dim.get("label_x", (dim["x1"] + dim["x2"]) / 2),
                "label_y": dim.get("label_y", (dim["y1"] + dim["y2"]) / 2),
            })

        # ── Step 2: advance rotation state silently ───────────────────────────
        if num_sides <= 0:
            num_sides = 4
        self.transform_controller.base_side = (
            self.transform_controller.base_side + direction) % num_sides
        self._update_rotation_label(config)

        # ── Step 3: draw new shape, suppressing preset re-snap ────────────────
        self._suppress_preset_dim_snap = True
        try:
            self.generate_plot()
        finally:
            self._suppress_preset_dim_snap = False

        # ── Steps 4-6: compute angle, rotate all annotations, final redraw ───
        if has_annotations and old_center is not None:
            # The shape is always redrawn from scratch in canonical orientation
            # then rotated around _canonical_pre_rotation_center by the cumulative angle.
            # Annotations must rotate around the SAME fixed pivot by the step angle.
            # No shift is needed because rotation is a rigid transform: if both
            # shape vertices and annotation coords rotate around the same center,
            # relative positions are preserved exactly.
            #
            # IMPORTANT: flip and rotation do NOT commute.  When an odd number of
            # flip axes are active (flip_h XOR flip_v), F(R(x)) = R⁻¹(F(x)).
            # The annotation coords already live in the flipped-and-previously-
            # rotated visual space, so we must negate the step angle to keep them
            # in sync with the shape body which is drawn canonical→rotate→flip.
            canonical = getattr(self, '_canonical_pre_rotation_center', None)
            if canonical is not None:
                pivot = canonical
            else:
                pivot = old_center

            flip_h = self.transform_controller.flip_h
            flip_v = self.transform_controller.flip_v
            flip_parity = int(flip_h) ^ int(flip_v)  # 1 if exactly one flip active
            effective_direction = -direction if flip_parity else direction
            angle = -effective_direction * (2 * math.pi / num_sides)
            rp = GeometricRotation.rotate_point

            if abs(angle) > 1e-6:
                for lbl in self._standalone_labels:
                    lbl["x"], lbl["y"] = rp(lbl["x"], lbl["y"], angle, pivot[0], pivot[1])

                for dim, old in zip(self._standalone_dim_lines, old_dim_coords):
                    for src_xk, src_yk, dst_xk, dst_yk in [
                        ("x1", "y1", "x1", "y1"),
                        ("x2", "y2", "x2", "y2"),
                        ("label_x", "label_y", "label_x", "label_y"),
                    ]:
                        rx, ry = rp(old[src_xk], old[src_yk], angle, pivot[0], pivot[1])
                        dim[dst_xk] = rx
                        dim[dst_yk] = ry

                self._suppress_preset_dim_snap = True
                try:
                    self.generate_plot()
                finally:
                    self._suppress_preset_dim_snap = False

        # Capture AFTER the full rotation + annotation transform completes.
        # force=True because rotating a shape back to its original orientation
        # produces a state equal to a prior stack entry — without force, dedup
        # would suppress it and leave the rotation unrecorded.
        if not self.history_manager.is_restoring:
            self._capture_current_state(force=True)

    def _flip_with_annotations(self, axis: str) -> None:
        """Flip the shape and transform ALL annotations to follow.

        ALL dim lines (preset and freeform) and labels are mirrored.
        Preset re-snap is suppressed so the mirrored coords are not overwritten.
        """
        shape = self.shape_var.get()
        if not shape:
            return

        # Capture BEFORE mutating so this action is always undoable.
        if not self.history_manager.is_restoring:
            self._capture_current_state()
        has_annotations = bool(self._standalone_labels or self._standalone_dim_lines)
        is_geometric = (shape in GeometricRotation.ALL_GEOMETRIC_SHAPES)

        old_center = self._get_shape_center() if (has_annotations and is_geometric) else None
        old_dim_coords: list[dict] = []
        if has_annotations and is_geometric:
            for dim in self._standalone_dim_lines:
                old_dim_coords.append({
                    "x1": dim["x1"], "y1": dim["y1"],
                    "x2": dim["x2"], "y2": dim["y2"],
                    "label_x": dim.get("label_x", (dim["x1"] + dim["x2"]) / 2),
                    "label_y": dim.get("label_y", (dim["y1"] + dim["y2"]) / 2),
                })

        flip_h = (axis == 'h')
        flip_v = (axis == 'v')
        if flip_h:
            self.transform_controller.flip_h = not self.transform_controller.flip_h
            self.transform_controller._reset_button_icons()
        else:
            self.transform_controller.flip_v = not self.transform_controller.flip_v
            self.transform_controller._reset_button_icons()

        self._suppress_preset_dim_snap = True
        try:
            self.generate_plot()
        finally:
            self._suppress_preset_dim_snap = False

        if has_annotations and is_geometric and old_center is not None:
            # Use the flip center that generate_plot used for flip_axes_artists.
            # This may differ from the canonical center when the shape is rotated
            # (the post-rotation bbox center shifts for asymmetric shapes).
            fc = getattr(self, '_flip_center', None)
            if fc is not None:
                cx, cy = fc
            else:
                cx, cy = old_center

            def mirror(px, py):
                x = (2 * cx - px) if flip_h else px
                y = (2 * cy - py) if flip_v else py
                return x, y

            for lbl in self._standalone_labels:
                lbl["x"], lbl["y"] = mirror(lbl["x"], lbl["y"])

            for dim, old in zip(self._standalone_dim_lines, old_dim_coords):
                dim["x1"], dim["y1"] = mirror(old["x1"], old["y1"])
                dim["x2"], dim["y2"] = mirror(old["x2"], old["y2"])
                dim["label_x"], dim["label_y"] = mirror(old["label_x"], old["label_y"])

            self._suppress_preset_dim_snap = True
            try:
                self.generate_plot()
            finally:
                self._suppress_preset_dim_snap = False

        # Capture AFTER the flip + annotation transform completes so the new
        # flipped state is always on the undo stack.  force=True is needed here
        # because toggling a flip back to its original value produces a state
        # that equals a previous stack entry — deduplication would otherwise
        # suppress it, leaving the redo stack without a way back.
        if not self.history_manager.is_restoring:
            self._capture_current_state(force=True)

    def _update_rotation_label(self, config: ShapeConfig | None = None) -> None:
        """Update rotation state label based on current shape and base_side."""
        if config is None:
            shape = self.shape_var.get()
            config = ShapeConfigProvider.get(shape)
        
        if config.rotation_labels:
            current_side = self.transform_controller.base_side
            label_text = config.rotation_labels[current_side % len(config.rotation_labels)]
            self.rotation_state_label.config(text=label_text)
        else:
            self.rotation_state_label.config(text="")
    
    def _set_dimension_mode(self, mode: str) -> None:
        self.dimension_mode_var.set(mode)
        self.scale_manager.reset_many(["aspect", "slope"])
        
        # Rebuild inputs without repacking the right panel frame
        self._rebuild_inputs_for_mode()
        self._update_dimension_mode_buttons()
        
        self.generate_plot()
        if not self.history_manager.is_restoring:
            self._capture_current_state()
    
    def _update_dimension_mode_buttons(self) -> None:
        # Guard against button not existing (shapes without dimension mode)
        if not hasattr(self, 'default_btn') or not self.default_btn.winfo_exists():
            return
        
        mode = self.dimension_mode_var.get()
        if mode == "Default":
            self.default_btn.config(relief="sunken", bg=AppConstants.ACTIVE_BUTTON_COLOR, fg="black")
            self.custom_btn.config(relief="raised", bg=AppConstants.BG_COLOR, fg="gray")
        else:
            self.default_btn.config(relief="raised", bg=AppConstants.BG_COLOR, fg="gray")
            self.custom_btn.config(relief="sunken", bg=AppConstants.ACTIVE_BUTTON_COLOR, fg="black")
    
    def _clear_entries(self) -> None:
        """Clear input entries without wiping custom label positions."""
        for w in self.input_frame.winfo_children():
            w.destroy()
        # Clear dim line column (col 3)
        if self.col_dimlines is not None:
            for w in self.col_dimlines.winfo_children():
                w.destroy()
        self.input_controller.entries = {}
        self._standalone_labels = []
        self._standalone_selected_label = None
        self._standalone_label_bboxes = []
        self._standalone_dim_lines = []
        self._standalone_selected_dim = None
        self._standalone_dim_mode = False
        self._standalone_dim_first_point = None
        self._builtin_dim_endpoints = []
        # Clear toggle-based labels so they don't persist across shapes
        for _k in ("Circumference", "Radius", "Diameter", "Height", "Slant",
                   "Length (Front)", "Width (Side)", "Base (Tri)", "Height (Tri)", "Length (Prism)"):
            self.label_manager.label_texts.pop(_k, None)
            self.label_manager.label_visibility.pop(_k, None)
        self._standalone_edit_mode = None
        self._standalone_dim_endpoints = []
        self._standalone_dim_label_bboxes = []
        self._builtin_dim_endpoints = []
        self._builtin_selected = None
        # Clear composite transfer list reference and disconnect drag events
        if self.composite_transfer is not None:
            self._disconnect_composite_drag()
            self.composite_transfer = None
        # Explicitly NOT calling self.label_manager.clear_positions() here
    
    def _rebuild_inputs_for_mode(self) -> None:
        shape = self.shape_var.get()
        if not shape: return
        mode = self.dimension_mode_var.get()
    
        # Ensure inputs column is visible
        self.col_inputs.grid()
        
        if shape == "Triangle":
            # Don't call _clear_entries or _create_input_header here; let build_triangle_inputs handle its own lifecycle
            self.build_triangle_inputs()
        elif shape == "Polygon":
            self.build_polygon_inputs()
        else:
            self._clear_entries()
            self._configure_transforms(shape)
            if mode == "Custom":
                self._create_input_header(slim=True)
            self._build_inputs_for_mode(shape, mode)
        
        if shape == "Parallelogram":
            if not hasattr(self, 'parallelogram_slope_var'):
                self.parallelogram_slope_var = tk.DoubleVar(value=0.5)
        
        config = ShapeConfigProvider.get(shape)
        if self.mode_help_label is not None:
            self.mode_help_label.config(text=config.help_text)
        
        # Don't pack right panel here - let caller handle it to avoid flash
    
    def _build_inputs_for_mode(self, shape: str, mode: str):
        """Build inputs for the specified mode using ShapeConfigProvider."""
        self._clear_adjust_sliders()
        
        config = ShapeConfigProvider.get(shape)

        # Add mode-specific sliders
        self._add_mode_sliders(shape, mode)

        # Build entry rows ONLY for Custom mode; radial shapes now use toggles
        if mode == "Custom":
            self.input_controller.build_from_config(config, mode, slim=True)
        
        self._build_standalone_label_ui()
    
    def _clear_adjust_sliders(self) -> None:
        """Clear all widgets from adjust sliders frame."""
        for w in self.adjust_sliders_frame.winfo_children():
            w.destroy()
    
    def _add_mode_sliders(self, shape: str, mode: str) -> None:
        """Add sliders specific to shape and mode."""
        config = ShapeConfigProvider.get(shape)
        
        if mode == "Custom" and config.has_feature(ShapeFeature.SLIDER_SLOPE):
            self._add_slope_slider()
        
        if mode == "Default" and config.has_feature(ShapeFeature.SLIDER_SHAPE):
            self._add_shape_adjust_slider()
        
        if mode == "Default" and config.has_feature(ShapeFeature.SLIDER_PEAK):
            self._add_peak_offset_slider()
    
    def _add_slope_slider(self) -> None:
        """Add parallelogram slope slider with center tick at 0.0, default position at 0.3."""
        self.parallelogram_slope_var = self.scale_manager.var("slope")
        spec = self.scale_manager.specs["slope"]
        self._add_slider("Adjust Slope:", self.parallelogram_slope_var, spec.min_val, spec.max_val, show_center=True, center_value=0.0)
    
    def _add_shape_adjust_slider(self) -> None:
        """Add default shape adjustment slider."""
        aspect_var = self.scale_manager.var("aspect")
        self._add_slider("Adjust Shape:", aspect_var, 
                         AppConstants.SLIDER_MIN, AppConstants.SLIDER_MAX, show_center=False)
    
    def _add_peak_offset_slider(self) -> None:
        """Add triangle peak offset slider."""
        self.peak_offset_var = self.scale_manager.var("peak_offset")
        spec = self.scale_manager.specs["peak_offset"]
        self._add_slider("Peak Offset:", self.peak_offset_var, spec.min_val, spec.max_val, show_center=False)
    
    def _add_slider(self, label: str, variable: tk.DoubleVar, 
                    from_val: float, to_val: float, show_center: bool = False, center_value: float = None) -> ttk.Scale:
        """Add a labeled slider with tight vertical packing."""
        lbl = tk.Label(self.adjust_sliders_frame, text=label, bg=AppConstants.BG_COLOR, font=("Arial", 8, "bold"))
        lbl.pack(side=tk.TOP, pady=(2, 0))
        
        slider = ttk.Scale(self.adjust_sliders_frame, variable=variable, from_=from_val, to=to_val, orient="horizontal", command=self.on_slider_change)
        slider.pack(side=tk.TOP, pady=0, fill=tk.X, padx=5)
        
        if show_center:
            rel_pos = (center_value - from_val) / (to_val - from_val) if center_value is not None else 0.5
            tick_container = tk.Frame(self.adjust_sliders_frame, bg=AppConstants.BG_COLOR, height=4)
            tick_container.pack(side=tk.TOP, fill=tk.X, padx=5)
            tick_container.pack_propagate(False)
            tk.Frame(tick_container, bg="gray", width=1, height=4).place(relx=rel_pos, rely=0, anchor="n")
            
        return slider
    
    def _is_composite_shape(self, shape: str = None) -> bool:
        """Check if the given (or current) shape is a composite type."""
        if shape is None:
            shape = self.shape_var.get()
        return shape in ("2D Composite", "3D Composite")

    def _reset_transforms(self) -> None:
        self.transform_controller.reset()
    
    def _on_font_blur(self) -> None:
        """Handle font spinbox losing focus - restore value if empty."""
        raw_val = self.font_spin.get()
        if not raw_val or not raw_val.strip():
            self.font_size_var.set(self.font_size)
        else:
            self.update_font_size()

    def update_font_size(self) -> None:
        if not self.root.winfo_exists(): return
        try:
            raw_val = self.font_spin.get()
            if not raw_val or not raw_val.strip():
                return
            try:
                new_size = int(raw_val)
                new_size = max(AppConstants.MIN_FONT_SIZE, min(AppConstants.MAX_FONT_SIZE, new_size))
                    
                self.font_size = new_size
                self.font_size_var.set(new_size)
                self.plot_controller.set_font_size(self.font_size)
                self.generate_plot()
                if not self.history_manager.is_restoring:
                    self._capture_current_state()
            except ValueError:
                self.font_size_var.set(self.font_size)
        except tk.TclError as e:
            logger.debug("Spinbox update failed (likely widget destroyed): %s", e)

    def _set_font_family(self, family: str) -> None:
        """Set the active font family, update button states, redraw, and capture undo state."""
        self.font_family = family
        self.plot_controller.set_font_family(family)
        is_sans = (family == "sans-serif")
        self.font_sans_btn.config(
            relief=tk.SUNKEN if is_sans else tk.RAISED,
            bg=AppConstants.ACTIVE_BUTTON_COLOR if is_sans else AppConstants.DEFAULT_BUTTON_COLOR,
        )
        self.font_serif_btn.config(
            relief=tk.RAISED if is_sans else tk.SUNKEN,
            bg=AppConstants.DEFAULT_BUTTON_COLOR if is_sans else AppConstants.ACTIVE_BUTTON_COLOR,
        )
        self.generate_plot()
        if not self.history_manager.is_restoring:
            self._capture_current_state()

    def _set_font_sans(self) -> None:
        self._set_font_family("sans-serif")

    def _set_font_serif(self) -> None:
        self._set_font_family("serif")

    def _on_line_width_blur(self) -> None:
        """Handle line weight spinbox losing focus - restore value if empty."""
        raw_val = self.line_width_spin.get()
        if not raw_val or not raw_val.strip():
            self.line_width_var.set(self.line_width)
        else:
            self.update_line_width()

    def update_line_width(self) -> None:
        if not self.root.winfo_exists(): return
        try:
            raw_val = self.line_width_spin.get()
            if not raw_val or not raw_val.strip():
                return
            try:
                new_width = int(raw_val)
                new_width = max(AppConstants.MIN_LINE_WIDTH, min(AppConstants.MAX_LINE_WIDTH, new_width))
                self.line_width = new_width
                self.line_width_var.set(new_width)
                self.plot_controller.set_line_width(self.line_width)
                self.generate_plot()
                if not self.history_manager.is_restoring:
                    self._capture_current_state()
            except ValueError:
                self.line_width_var.set(self.line_width)
        except tk.TclError as e:
            logger.debug("Line width spinbox update failed (likely widget destroyed): %s", e)

    def show_welcome(self) -> None:
        # Keep only Category (cols 0,1) and Shape (cols 2,3) visible in top bar
        if self.center_container is not None:
            for slave in self.center_container.grid_slaves(row=0):
                col = int(slave.grid_info()["column"])
                if col <= 3:
                    slave.grid()
                else:
                    slave.grid_remove()
        
        # Hide all control columns
        if self.col_shape_type is not None: self.col_shape_type.grid_remove()
        if self.col_transforms is not None: self.col_transforms.grid_remove()
        if self.col_inputs is not None: self.col_inputs.grid_remove()
        if self.col_dimlines is not None: self.col_dimlines.grid_remove()
        if self.col_tools is not None: self.col_tools.grid_remove()
        
        self.ax.clear()
        self.plot_controller.setup_axes()
        
        # Standardized welcome limits matching the paper aspect ratio
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-7, 7)
        
        self.ax.text(0, 1.5, "Welcome to Geometry Forge", 
                    ha="center", va="center", fontsize=28, fontweight='bold', color="black")
        self.ax.text(0, -1.0, "Select a Category to Begin", 
                    ha="center", va="center", fontsize=12, color="black")
        self.canvas.draw()

    def _cancel_after(self, attr_name: str) -> None:
        """Cancel a pending after callback by attribute name."""
        after_id = getattr(self, attr_name, None)
        if after_id is None:
            return
        
        try:
            self.root.after_cancel(after_id)
        except tk.TclError as e:
            logger.debug("Could not cancel after callback %s: %s", attr_name, e)
        finally:
            setattr(self, attr_name, None)
    
    def on_slider_change(self, _) -> None:
        """Handle slider changes with debouncing to reduce rapid redraws."""
        if not self.shape_var.get():
            return
        
        self._cancel_after('_slider_after_id')
        self._slider_after_id = self.root.after(AppConstants.SLIDER_DEBOUNCE_DELAY, self._finalize_slider_change)
    
    def _finalize_slider_change(self) -> None:
        """Capture state and redraw after slider movement."""
        if not self.history_manager.is_restoring:
            self._capture_current_state()
        self.generate_plot()
    
    def _apply_view_scale_only(self) -> None:
        """Apply view scale + pan without redrawing the entire shape."""
        self._apply_view_scale()
        self.canvas.draw_idle()
        self.canvas.flush_events()
        # Debounced state capture so undo works after scale/pan changes
        self._cancel_after('_scale_after_id')
        self._scale_after_id = self.root.after(400, self._finalize_scale_change)

    def _finalize_scale_change(self) -> None:
        """Capture state after scale/pan interaction settles (for undo)."""
        self._scale_after_id = None
        if not self.history_manager.is_restoring and self.shape_var.get():
            self._capture_current_state()

    def _on_input_change(self) -> None:
        """Handle input changes with debouncing."""
        self._cancel_after('_redraw_after_id')
        self._redraw_after_id = self.root.after(AppConstants.DEBOUNCE_DELAY, self._finalize_input_change)

    def _finalize_input_change(self) -> None:
        """Capture state and redraw after user stops typing."""
        if self.history_manager.is_restoring:
            return
        self._capture_current_state()
        self.generate_plot()

    def _set_triangle_type(self, tri_type: str) -> None:
        """Handle triangle type button clicks."""
        self.triangle_type_var.set(tri_type)
        if tri_type == "Scalene":
            self.scale_manager.var("peak_offset").set(AppConstants.SCALENE_DEFAULT_PEAK)
        else:
            self.scale_manager.reset("peak_offset")
        # Right triangle uses aspect slider centered at 1.0 (3-4-5 default)
        if tri_type == "Right":
            self.scale_manager.reset("aspect")
        elif tri_type != "Custom":
            self.scale_manager.reset("aspect")
        self._update_triangle_button_styles()
        self.root.after_idle(self._rebuild_triangle_ui)
        if not self.history_manager.is_restoring:
            self._capture_current_state()

    def _update_triangle_button_styles(self) -> None:
        """Highlight active triangle button using standardized active state colors."""
        if self.tri_buttons is None:
            return
        current = self.triangle_type_var.get()
        for name, btn in self.tri_buttons.items():
            if name == current:
                btn.config(relief="sunken", bg=AppConstants.ACTIVE_BUTTON_COLOR, fg="black")
            else:
                btn.config(relief="raised", bg=AppConstants.BG_COLOR, fg="gray")

    def on_triangle_type_change(self, event: Any | None = None) -> None:
        if not self.history_manager.is_restoring:
            self._capture_current_state()
        tri = self.triangle_type_var.get()
        if tri == "Scalene":
            self.scale_manager.var("peak_offset").set(AppConstants.SCALENE_DEFAULT_PEAK)
        else:
            self.scale_manager.reset("peak_offset")
        self.scale_manager.reset("aspect")
        self.root.after_idle(self._rebuild_triangle_ui)
    
    def _rebuild_triangle_ui(self) -> None:
        """Rebuild triangle UI - called via after_idle to prevent blocking."""
        self.build_triangle_inputs()
        self.generate_plot()

    def _create_input_header(self, slim: bool = False) -> None:
        self.input_controller.create_header(slim=slim)
        # Right panel packing is handled centrally by _pack_right_panel()
    
    # --------------------------------------------------
    def _configure_transforms(self, shape: str) -> None:
        """Configure shape type, transform, and slider columns for the current shape."""
        config = ShapeConfigProvider.get(shape)
        self._reset_transforms()

        # --- Column 0: Shape Type ---
        # Clear old shape-specific widgets
        for w in self.shape_options_frame.winfo_children():
            w.destroy()
        
        has_shape_type = False
        if config.has_dimension_mode:
            has_shape_type = True
            shape_type_label = f"{shape} Type"
            tk.Label(
                self.shape_options_frame, text=shape_type_label, bg=AppConstants.BG_COLOR,
                font=("Arial", 9, "bold")
            ).pack(side=tk.TOP, pady=(0, 1), anchor="center")
            self.default_btn = tk.Button(
                self.shape_options_frame, text="Default", width=10, font=AppConstants.BTN_FONT,
                command=lambda: self._set_dimension_mode("Default")
            )
            self.default_btn.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(0, 1))
            self.custom_btn = tk.Button(
                self.shape_options_frame, text="Custom", width=10, font=AppConstants.BTN_FONT,
                command=lambda: self._set_dimension_mode("Custom")
            )
            self.custom_btn.pack(side=tk.TOP, fill=tk.X, padx=5)
            self._update_dimension_mode_buttons()
        
        has_shape_specific = shape in ["Triangle", "Polygon"]
        show_col0 = has_shape_specific or has_shape_type
        is_composite = shape in ("2D Composite", "3D Composite")
        
        if show_col0 and not is_composite:
            self.col_shape_type.grid()
        else:
            self.col_shape_type.grid_remove()
        
        # --- Column 1: Transforms & Sliders ---
        # Reset transform widget visibility
        self.options_header.pack_forget()
        self.transform_frame.pack_forget()
        self.flip_row.pack_forget()
        self.rotate_row.pack_forget()
        self.rotation_state_label.pack_forget()
        self.adjust_sliders_frame.pack_forget()
        for w in self.adjust_sliders_frame.winfo_children():
            w.destroy()
        
        has_flip = config.has_feature(ShapeFeature.FLIP)
        has_rotate = config.has_feature(ShapeFeature.ROTATE)
        has_transforms = has_flip or has_rotate
        
        has_sliders = False
        if config.has_feature(ShapeFeature.SLIDER_SHAPE):
            has_sliders = True
            self._add_shape_adjust_slider()
        
        show_col1 = (has_transforms or has_sliders) and not is_composite
        
        if show_col1:
            self.col_transforms.grid()
            
            if has_transforms:
                self.options_header.pack(side=tk.TOP, pady=(2, 1), anchor="center")
                self.transform_frame.pack(side=tk.TOP, fill=tk.X)
                if has_flip:
                    self.flip_row.pack(side=tk.TOP, pady=1, fill=tk.X, padx=5)
                if has_rotate:
                    self.rotate_row.pack(side=tk.TOP, pady=1, fill=tk.X, padx=5)
                    if config.rotation_labels:
                        self.rotation_state_label.pack(side=tk.TOP, pady=1, anchor="center")
                        self._update_rotation_label(config)
            
            if has_sliders:
                self.adjust_sliders_frame.pack(side=tk.TOP, pady=0, fill=tk.X)
        else:
            self.col_transforms.grid_remove()

    # --------------------------------------------------
    def _on_escape_pressed(self, event) -> None:
        if self._standalone_dim_mode:
            self._cancel_standalone_dim_mode()
            return
        if self._composite_dim_mode is not None:
            self._cancel_dim_line_mode()
            self.generate_plot()
            return
        
    def _on_hashmarks_changed(self) -> None:
        # Suppress preset dim-line re-snap so transformed dim line positions
        # are not overwritten by canonical-geometry recalculation on redraw.
        self._suppress_preset_dim_snap = True
        try:
            self.generate_plot()
        finally:
            self._suppress_preset_dim_snap = False
        if not self.history_manager.is_restoring:
            self._capture_current_state()
    
    def _clear_workspace(self) -> None:
        if self._ui_confirm_yesno("Clear All", "Clear all parameter values and label positions?"):
            if not self.history_manager.is_restoring:
                self._capture_current_state()
            # Clear dimension value entry fields
            for key in self.input_controller.entries:
                self.input_controller.clear_entry(key)
            # Clear toggle labels (radius, diameter, height, etc.)
            self.label_manager.clear_label_texts()
            # Clear drag-offset custom positions
            self.label_manager.reset_all_custom_positions()
            # Clear standalone freeform labels and dim lines
            self._standalone_labels.clear()
            self._standalone_dim_lines.clear()
            self._standalone_selected_label = None
            self._standalone_selected_dim = None
            self._builtin_selected = None
            # Reset view
            self._shape_pan_offset = (0.0, 0.0)
            self.scale_manager.reset("view_scale")
            self.generate_plot()
            if not self.history_manager.is_restoring:
                self._capture_current_state()


    def _capture_current_state(self, force: bool = False) -> None:
        """Capture all UI and logic state into the history manager.
        
        Args:
            force: Passed through to HistoryManager.capture_state — bypasses
                   deduplication when True.  Use for deliberate user actions
                   where the snapshot equals the current stack top.
        """
        if self.history_manager.is_restoring:
            return
        shape = self.shape_var.get()
        if not shape or shape in ["", "Select Shape"]:
            return
        state = self._build_state_snapshot()
        # Ignore states without shape data to prevent undoing to blank screens
        if not state["shape"]:
            return
        self.history_manager.capture_state(state, force=force)
        self._update_undo_redo_buttons()

    def _apply_state(self, state: dict) -> None:
        """Optimized UI restoration without full rebuild if possible."""
        if not state or self.history_manager.is_restoring:
            return
        with self.history_manager.restoring():
            old_shape = self.shape_var.get()
            old_cat = self.cat_var.get()
            old_mode = self.dimension_mode_var.get()
            old_tri = self.triangle_type_var.get()
            old_poly = self.polygon_type_var.get()

            new_shape = state.get("shape", "")
            new_cat = state.get("cat", "")
            new_mode = state.get("dim_mode", "Default")
            new_tri = state.get("tri_type", "Custom")
            new_poly = state.get("poly_type", PolygonType.PENTAGON.value)

            self.dimension_mode_var.set(new_mode)
            self.triangle_type_var.set(new_tri)
            self.polygon_type_var.set(new_poly)

            if old_cat != new_cat:
                self.cat_var.set(new_cat)
                self.cat_combo.set(new_cat)
                shapes = self.shape_data.get(new_cat, [])
                self.shape_combo["values"] = shapes

            if old_shape != new_shape:
                self.shape_var.set(new_shape)
                self.shape_combo.set(new_shape if new_shape else "Select Shape")
                self._rebuild_shape_ui_for_restore(new_shape)
            elif old_mode != new_mode or old_tri != new_tri or old_poly != new_poly:
                self._rebuild_inputs_for_mode()
            else:
                self._update_dimension_mode_buttons()
            
            # Delegate controller restore
            self.scale_manager.set_state(state)
            self.input_controller.set_state(state)
            # Restore pan offset
            pan = state.get("view_pan", [0.0, 0.0])
            self._shape_pan_offset = (float(pan[0]), float(pan[1]))

            # Restore composite transfer list and positions if applicable
            composite_shapes = state.get("composite_shapes", [])
            if self._is_composite_shape(new_shape) and self.composite_transfer is not None:
                self.composite_transfer.set_selected_shapes(composite_shapes)
                self._composite_positions = dict(state.get("composite_positions", {}))
                self._composite_transforms = {k: dict(v) for k, v in state.get("composite_transforms", {}).items()}
                self._composite_labels = [dict(lbl) for lbl in state.get("composite_labels", [])]
                self._composite_dim_lines = [dict(d) for d in state.get("composite_dim_lines", [])]
                self._composite_selected.clear()
                self._composite_selected_label = None
                self._composite_selected_dim = None

            # Restore line width
            if self.line_width_var is not None:
                lw = state.get("line_width", AppConstants.DEFAULT_LINE_WIDTH)
                self.line_width = lw
                self.line_width_var.set(lw)
                self.plot_controller.set_line_width(lw)

            # Restore font size
            if self.font_size_var is not None:
                fs = state.get("font_size", AppConstants.DEFAULT_FONT_SIZE)
                self.font_size = fs
                self.font_size_var.set(fs)
                self.plot_controller.set_font_size(fs)

            # Restore font family
            ff = state.get("font_family", AppConstants.DEFAULT_FONT_FAMILY)
            self.font_family = ff
            self.plot_controller.set_font_family(ff)
            if self.font_sans_btn is not None:
                if ff == "serif":
                    self.font_sans_btn.config(relief=tk.RAISED, bg=AppConstants.DEFAULT_BUTTON_COLOR)
                    self.font_serif_btn.config(relief=tk.SUNKEN, bg=AppConstants.ACTIVE_BUTTON_COLOR)
                else:
                    self.font_sans_btn.config(relief=tk.SUNKEN, bg=AppConstants.ACTIVE_BUTTON_COLOR)
                    self.font_serif_btn.config(relief=tk.RAISED, bg=AppConstants.DEFAULT_BUTTON_COLOR)

            # Restore hashmarks checkbox
            if hasattr(self, 'show_hashmarks_var'):
                self.show_hashmarks_var.set(state.get("show_hashmarks", False))

            # Restore standalone freeform labels and dim lines
            if not self._is_composite_shape(state.get("shape", "")):
                self._standalone_labels = [dict(lbl) for lbl in state.get("standalone_labels", [])]
                self._standalone_selected_label = None
                self._standalone_dim_lines = [dict(d) for d in state.get("standalone_dim_lines", [])]
                self._standalone_selected_dim = None
                for lbl_key, st_key in self.TOGGLE_LABEL_KEYS:
                    t = state.get(f"{st_key}_text", "")
                    v = state.get(f"{st_key}_vis", False)
                    if t:
                        self.label_manager.set_label_text(lbl_key, t, v)
                    else:
                        self.label_manager.label_texts.pop(lbl_key, None)
                        self.label_manager.label_visibility.pop(lbl_key, None)

            self.transform_controller.set_state(state)
            self.label_manager.set_state(state)

        self._update_undo_redo_buttons()
        self._update_rotation_label()
        self._pack_right_panel()
        # Suppress preset dim-line re-snap during state restore: the snapshot
        # already contains correctly-transformed coordinates for the restored
        # orientation.  Re-snapping would recalculate from canonical geometry
        # (base_side=0, no flip) and overwrite them before the artist transform
        # pipeline has had a chance to move them to the right visual position.
        self._suppress_preset_dim_snap = True
        try:
            self.generate_plot()
        finally:
            self._suppress_preset_dim_snap = False

    def _undo_action(self) -> None:
        if not self.history_manager.can_undo() or self.history_manager.is_restoring:
            return
        prev = self.history_manager.undo(None)
        if prev:
            self._apply_state(prev)

    def _redo_action(self) -> None:
        if not self.history_manager.can_redo():
            return
        next_s = self.history_manager.redo(None) # Redo doesn't need current for stack swap here
        if next_s:
            self._apply_state(next_s)

    def _build_state_snapshot(self) -> dict:
        """Build current UI state snapshot for history.

        Delegates positioning/entry/scale/transform state to the four controllers
        via their get_state() methods; composite and standalone annotation state
        is captured directly here since it belongs to GeometryApp.
        """
        is_composite = self._is_composite_shape(self.shape_var.get())

        composite_selected = []
        composite_positions = {}
        composite_transforms = {}
        if is_composite and self.composite_transfer is not None:
            composite_selected = self.composite_transfer.get_selected_shapes()
            composite_positions = dict(self._composite_positions)
            composite_transforms = {k: dict(v) for k, v in self._composite_transforms.items()}

        state = {
            "cat": self.cat_var.get(),
            "shape": self.shape_var.get(),
            "font_size": self.font_size_var.get(),
            "line_width": (
                self.line_width_var.get()
                if self.line_width_var is not None
                else AppConstants.DEFAULT_LINE_WIDTH
            ),
            "font_family": self.font_family,
            "dim_mode": self.dimension_mode_var.get(),
            "tri_type": self.triangle_type_var.get(),
            "poly_type": self.polygon_type_var.get(),
            "show_hashmarks": self.show_hashmarks_var.get(),
            "view_pan": list(self._shape_pan_offset),
            "composite_shapes": composite_selected,
            "composite_positions": composite_positions,
            "composite_transforms": composite_transforms,
            "composite_labels": [dict(lbl) for lbl in self._composite_labels] if is_composite else [],
            "composite_dim_lines": [dict(d) for d in self._composite_dim_lines] if is_composite else [],
            "standalone_labels": [dict(lbl) for lbl in self._standalone_labels] if not is_composite else [],
            "standalone_dim_lines": [dict(d) for d in self._standalone_dim_lines] if not is_composite else [],
            **{
                f"{st_key}_text": self.label_manager.label_texts.get(lbl_key, "")
                for lbl_key, st_key in self.TOGGLE_LABEL_KEYS
            },
            **{
                f"{st_key}_vis": self.label_manager.label_visibility.get(lbl_key, False)
                for lbl_key, st_key in self.TOGGLE_LABEL_KEYS
            },
        }
        # Delegate controller state
        state.update(self.scale_manager.get_state())
        state.update({"transforms": {
            "h": self.transform_controller.flip_h,
            "v": self.transform_controller.flip_v,
            "side": self.transform_controller.base_side,
        }})
        state.update(self.label_manager.get_state())
        state.update(self.input_controller.get_state())
        return state

    def _update_undo_redo_buttons(self) -> None:
        can_undo = len(self.history_manager.undo_stack) > 1
        can_redo = self.history_manager.can_redo()
        self.undo_btn.config(state="normal" if can_undo else "disabled")
        self.redo_btn.config(state="normal" if can_redo else "disabled")

    def _create_transform_state(self) -> TransformState:
        return self.transform_controller.get_state()
    
    def _collect_shape_params(self) -> dict[str, Any]:
        params = self.input_controller.collect_params()
        params["triangle_type"] = self.triangle_type_var.get()
        params["polygon_type"] = self.polygon_type_var.get()
        params["dimension_mode"] = self.dimension_mode_var.get()
        
        # ScaleManager vars - always available with defaults
        params["peak_offset"] = self.scale_manager.var("peak_offset").get()
        params["parallelogram_slope"] = self.scale_manager.var("slope").get()
        
        return params

    def build_triangle_inputs(self) -> None:
        # Only clear if called directly (not from update_inputs which pre-clears)
        if self.input_frame.winfo_children():
            self._clear_entries()

        triangle_type = self.triangle_type_var.get()

        # Only destroy if widgets exist (avoid redundant destruction from _configure_transforms)
        if self.shape_options_frame.winfo_children():
            for w in self.shape_options_frame.winfo_children():
                w.destroy()
        if self.adjust_sliders_frame.winfo_children():
            for w in self.adjust_sliders_frame.winfo_children():
                w.destroy()

        self.triangle_type_label = tk.Label(
            self.shape_options_frame, text="Triangle Type", bg=AppConstants.BG_COLOR,
            font=("Arial", 9, "bold")
        )
        self.triangle_type_label.pack(side=tk.TOP, pady=(0, 2), anchor="center")

        self.tri_buttons = {}
        for name in ["Custom", "Isosceles", "Scalene", "Equilateral", "Right"]:
            btn = tk.Button(
                self.shape_options_frame, text=name, width=10, font=AppConstants.BTN_FONT,
                command=lambda n=name: self._set_triangle_type(n)
            )
            btn.pack(side=tk.TOP, fill=tk.X, padx=5, pady=1)
            self.tri_buttons[name] = btn

        self._update_triangle_button_styles()

        # Config-driven triangle sliders
        config = ShapeConfigProvider.get_triangle_config(triangle_type)
        if config.has_feature(ShapeFeature.SLIDER_PEAK):
            self._add_peak_offset_slider()
        if config.has_feature(ShapeFeature.SLIDER_SHAPE) or triangle_type in ["Isosceles", "Scalene", "Right"]:
            self._add_shape_adjust_slider()
        # Ensure adjust_sliders_frame is packed and visible
        if self.adjust_sliders_frame.winfo_children():
            self.adjust_sliders_frame.pack(side=tk.TOP, pady=0, fill=tk.X)

        # Custom triangle: build parameter input boxes
        if triangle_type == "Custom":
            self._create_input_header(slim=True)
            self.input_controller.build_from_config(config, mode="Custom", slim=True)

        self._build_standalone_label_ui()
        self._pack_right_panel(show_help=True)
            
    def build_polygon_inputs(self) -> None:
        """Build polygon type selector UI, parallel to build_triangle_inputs."""
        if self.shape_options_frame.winfo_children():
            for w in self.shape_options_frame.winfo_children():
                w.destroy()
        if self.adjust_sliders_frame.winfo_children():
            for w in self.adjust_sliders_frame.winfo_children():
                w.destroy()

        tk.Label(
            self.shape_options_frame, text="Polygon Type", bg=AppConstants.BG_COLOR,
            font=("Arial", 9, "bold")
        ).pack(side=tk.TOP, pady=(0, 2), anchor="center")

        self.poly_buttons = {}
        for name in [PolygonType.PENTAGON.value, PolygonType.HEXAGON.value, PolygonType.OCTAGON.value]:
            btn = tk.Button(
                self.shape_options_frame, text=name, width=10, font=AppConstants.BTN_FONT,
                command=lambda n=name: self._set_polygon_type(n)
            )
            btn.pack(side=tk.TOP, fill=tk.X, padx=5, pady=1)
            self.poly_buttons[name] = btn

        self._update_polygon_button_styles()
        self._build_standalone_label_ui()
        self._pack_right_panel(show_help=True)

    def _set_polygon_type(self, poly_type: str) -> None:
        """Handle polygon type button clicks."""
        self.polygon_type_var.set(poly_type)
        self._update_polygon_button_styles()
        self.root.after_idle(self._rebuild_polygon_ui)
        if not self.history_manager.is_restoring:
            self._capture_current_state()

    def _update_polygon_button_styles(self) -> None:
        """Highlight active polygon type button."""
        if self.poly_buttons is None:
            return
        current = self.polygon_type_var.get()
        for name, btn in self.poly_buttons.items():
            if name == current:
                btn.config(relief="sunken", bg=AppConstants.ACTIVE_BUTTON_COLOR, fg="black")
            else:
                btn.config(relief="raised", bg=AppConstants.BG_COLOR, fg="gray")

    def _rebuild_polygon_ui(self) -> None:
        """Rebuild polygon UI and redraw — called via after_idle."""
        self._clear_entries()
        self.build_polygon_inputs()
        self.generate_plot()
        if not self.history_manager.is_restoring:
            self._schedule_delayed_capture()

    # --------------------------------------------------
    def update_shape_list(self, _: Any | None = None) -> None:
        # Capture outgoing shape state (label moves, input changes, etc.)
        if self.shape_var.get() and not self.history_manager.is_restoring:
            self._capture_current_state()
        
        cat = self.cat_var.get()
        shapes = self.shape_data.get(cat, [])
        
        # DON'T clear history - allow undo across categories
        self.shape_combo["values"] = shapes
        # Reset selection state whenever category changes
        self.shape_var.set("")
        self.shape_combo.set("Select Shape")

        # Clear inputs and reset transforms so old state doesn't "stick" across categories
        self._clear_entries()
        self._reset_transforms()

        # Hide control columns until a shape is selected
        if self.col_shape_type is not None:
            self.col_shape_type.grid_remove()
        if self.col_transforms is not None:
            self.col_transforms.grid_remove()
        if self.col_inputs is not None:
            self.col_inputs.grid_remove()
        if self.col_dimlines is not None:
            self.col_dimlines.grid_remove()
        if self.col_tools is not None:
            self.col_tools.grid_remove()

        # Hide right-panel widgets until a shape is selected
        if self.mode_help_label is not None:
            self.mode_help_label.pack_forget()
        if self.clear_workspace_btn is not None:
            self.clear_workspace_btn.pack_forget()

        # Hide export buttons until a shape is selected
        if self.save_btn is not None:
            self.save_btn.grid_remove()
        if self.copy_btn is not None:
            self.copy_btn.grid_remove()

        # Show welcome prompt so the user clearly knows they must pick a shape
        self.show_welcome()
        
        # Auto-select first shape in category for 3.0 (optional polish)
        if shapes:
            self.shape_combo.current(0)
            self.update_inputs()
        
        self.root.focus_set()

    def _rebuild_shape_ui_for_restore(self, shape: str) -> None:
        """Rebuild shape UI during state restoration without triggering captures."""
        # Show control columns
        if self.col_inputs is not None:
            self.col_inputs.grid()
        if self.col_dimlines is not None:
            self.col_dimlines.grid()
        if self.col_tools is not None:
            self.col_tools.grid()
        
        self.save_btn.grid(row=0, column=10, padx=1, sticky="e")
        self.copy_btn.grid(row=0, column=11, padx=1, sticky="e")
        
        self._clear_entries()
        
        if not shape:
            return
        
        # Composite shapes skip normal transforms/header and use transfer list UI
        if self._is_composite_shape(shape):
            config = ShapeConfigProvider.get(shape)
            self._build_composite_ui(shape)
            self._show_help_text(config.help_text)
            self._pack_right_panel(show_help=True)
            return
        
        self._configure_transforms(shape)
        
        if shape == "Triangle":
            self.build_triangle_inputs()
            return

        if shape == "Polygon":
            self.build_polygon_inputs()
            return

        config = ShapeConfigProvider.get(shape)
        
        if config.has_dimension_mode:
            self._rebuild_inputs_for_mode()
            self._show_help_text(config.help_text)
            self._pack_right_panel(show_help=True)
        else:
            self._build_standard_inputs(config)
            self._show_help_text(config.help_text)
            self._pack_right_panel(show_help=True)
    
    def update_inputs(self, _: Any | None = None) -> None:

        if self.center_container is not None:
            # Re-show all controls in the top bar
            for slave in self.center_container.grid_slaves(row=0):
                slave.grid()
        
        # Explicitly show export buttons
        if self.save_btn is not None:
            self.save_btn.grid(row=0, column=10, padx=1, sticky="e")
        if self.copy_btn is not None:
            self.copy_btn.grid(row=0, column=11, padx=1, sticky="e")
        
        # Ensure font controls are visible
        if self.font_label.winfo_exists():
            self.font_label.grid()
        if self.font_spin.winfo_exists():
            self.font_spin.grid()
        if self.font_sans_btn.winfo_exists():
            self.font_sans_btn.grid()
        if self.font_serif_btn.winfo_exists():
            self.font_serif_btn.grid()
        if self.weight_label.winfo_exists():
            self.weight_label.grid()
        if self.line_width_spin.winfo_exists():
            self.line_width_spin.grid()

        # Show control columns
        if self.col_inputs is not None:
            self.col_inputs.grid()
        if self.col_dimlines is not None:
            self.col_dimlines.grid()
        if self.col_tools is not None:
            self.col_tools.grid()

        # Only reset sliders if NOT restoring from history
        if not self.history_manager.is_restoring:
            self.scale_manager.reset_many(["aspect", "peak_offset", "slope", "view_scale"])
            self._shape_pan_offset = (0.0, 0.0)  # reset pan on shape change
            self.label_manager.reset_all_custom_positions()
            self.show_hashmarks_var.set(False)

        self._clear_entries()
        
        shape = self.shape_var.get()
        if not shape:
            return
        
        # Initialize shape-specific variables (removed - ScaleManager handles defaults)
        
        # Composite shapes skip normal transforms/header and use transfer list UI
        if self._is_composite_shape(shape):
            config = ShapeConfigProvider.get(shape)
            self._build_composite_ui(shape)
            self._show_help_text(config.help_text)
            self._pack_right_panel(show_help=True)
            self.generate_plot()
            self.root.focus_set()
            return
        
        self._configure_transforms(shape)
        
        # Handle triangle specially due to type selector
        if shape == "Triangle":
            self.build_triangle_inputs()
            self.generate_plot()
            if not self.history_manager.is_restoring:
                self._schedule_delayed_capture()
            return

        # Handle polygon specially due to type selector
        if shape == "Polygon":
            self.build_polygon_inputs()
            self.generate_plot()
            if not self.history_manager.is_restoring:
                self._schedule_delayed_capture()
            return
        
        config = ShapeConfigProvider.get(shape)
        
        # Build inputs based on dimension mode support
        if config.has_dimension_mode:
            if hasattr(self, 'dimension_mode_var'):
                self._rebuild_inputs_for_mode()
            self._show_help_text(config.help_text)
            self._pack_right_panel(show_help=True)
        else:
            self._build_standard_inputs(config)
            self._show_help_text(config.help_text)
            self._pack_right_panel(show_help=True)
        
        self.generate_plot()
        # Capture state after shape UI is fully built and drawn
        if not self.history_manager.is_restoring:
            self._schedule_delayed_capture()
        
        self.root.focus_set()
    
    def _show_help_text(self, text: str):
        """Update help text label."""
        if self.mode_help_label is not None:
            self.mode_help_label.config(text=text)
    
    def _build_standard_inputs(self, config: ShapeConfig) -> None:
        """Build standard shape UI — no entry table for non-dimension-mode shapes.
        Labels are auto-populated from ShapeConfig defaults via _populate_label_texts."""
        self._build_standalone_label_ui()
        self._pack_right_panel(show_help=True)

    def _build_standalone_label_ui(self) -> None:
        """Build label entry controls (col 2) and dim line controls (col 3)."""
        # --- Col 2: Label entry and hashmarks ---
        container = tk.Frame(self.input_frame, bg=AppConstants.BG_COLOR)
        container.grid(row=100, column=0, columnspan=5, sticky="ew", pady=(4, 0))

        # Hashmarks checkbox (only for polygon shapes)
        shape = self.shape_var.get()
        if shape in SmartGeometryEngine.POLYGON_SHAPES:
            hm_frame = tk.Frame(container, bg=AppConstants.BG_COLOR)
            hm_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))
            if not hasattr(self, 'show_hashmarks_var'):
                self.show_hashmarks_var = tk.BooleanVar(value=False)
            tk.Checkbutton(
                hm_frame, text="Show Hashmarks", bg=AppConstants.BG_COLOR,
                variable=self.show_hashmarks_var,
                command=self._on_hashmarks_changed, takefocus=0
            ).pack(side=tk.LEFT, padx=(0, 4))

        # --- Col 3: Labels and Lines ---
        # Clear previous widgets
        for w in self.col_dimlines.winfo_children():
            w.destroy()
        
        shape = self.shape_var.get()
        preset_key = shape
        if shape == "Triangle":
            tri_type = self.triangle_type_var.get()
            tri_key = f"Triangle_{tri_type}"
            if tri_key in self._DIM_PRESETS:
                preset_key = tri_key
        presets = self._DIM_PRESETS.get(preset_key, [])
        
        # Header
        tk.Label(self.col_dimlines, text="Labels and Lines", bg=AppConstants.BG_COLOR,
                 font=("Arial", 9, "bold")).pack(side=tk.TOP, anchor="center", pady=(0, 2))
        
        # Entry row: [         ] [+ Text]
        label_frame = tk.Frame(self.col_dimlines, bg=AppConstants.BG_COLOR)
        label_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))
        
        self._standalone_label_entry = tk.Entry(label_frame, width=10, font=AppConstants.BTN_FONT)
        self._standalone_label_entry.pack(side=tk.LEFT, padx=(0, 2), fill=tk.X, expand=True)
        self._standalone_label_entry.bind('<Return>', lambda e: self._confirm_standalone_label())
        
        self._standalone_text_btn = tk.Button(label_frame, text="+ Text", font=AppConstants.BTN_FONT,
                  command=self._confirm_standalone_label)
        self._standalone_text_btn.pack(side=tk.LEFT, padx=1)
        
        self._standalone_cancel_btn = tk.Button(label_frame, text="Cancel", font=AppConstants.BTN_FONT,
                  command=self._cancel_standalone_edit)
        # Hidden by default — shown only during edit mode
        
        # 2-column grid of preset line buttons + Free
        btn_grid = tk.Frame(self.col_dimlines, bg=AppConstants.BG_COLOR)
        btn_grid.pack(side=tk.TOP, fill=tk.X)
        btn_grid.columnconfigure(0, weight=1)
        btn_grid.columnconfigure(1, weight=1)
        
        all_buttons = [(p["label"], lambda k=p["key"], t=p["default_text"]: self._add_standalone_dim_preset(k, t)) for p in presets]
        
        self._standalone_free_btn = tk.Button(btn_grid, text="Free", font=AppConstants.BTN_FONT,
                  command=self._start_standalone_dim_mode)
        self._standalone_cancel_dim_btn = tk.Button(btn_grid, text="Cancel", font=AppConstants.BTN_FONT,
                  command=self._cancel_standalone_dim_mode)
        
        row = 0
        col = 0
        for label_text, cmd in all_buttons:
            tk.Button(btn_grid, text=label_text, font=AppConstants.BTN_FONT,
                      command=cmd).grid(row=row, column=col, padx=1, pady=1, sticky="ew")
            col += 1
            if col > 1:
                col = 0
                row += 1
        
        # Free button in next slot
        self._standalone_free_btn.grid(row=row, column=col, padx=1, pady=1, sticky="ew")
        # Cancel replaces Free during dim mode (same grid cell)
        self._standalone_cancel_dim_btn.grid(row=row, column=col, padx=1, pady=1, sticky="ew")
        self._standalone_cancel_dim_btn.grid_remove()
        
        # Delete spanning both columns
        tk.Button(self.col_dimlines, text="Delete Selected", font=AppConstants.BTN_FONT,
                  command=self._remove_standalone_annotation).pack(side=tk.TOP, pady=(2, 0))
        
        # Show col 3
        self.col_dimlines.grid()

    def _start_standalone_dim_mode(self) -> None:
        """Enter freeform dimension line draw mode for standalone shapes."""
        self._standalone_dim_mode = True
        self._standalone_dim_first_point = None
        self.root.config(cursor="crosshair")
        if self._standalone_free_btn is not None:
            self._standalone_free_btn.grid_remove()
            self._standalone_cancel_dim_btn.grid()

    def _build_composite_ui(self, shape: str) -> None:
        """Build the transfer list UI for composite shapes."""
        if shape == "2D Composite":
            source_shapes = [
                "Tri Triangle" if s == "Triangle" else s
                for s in self.shape_data.get("2D Figures", [])
                if s != "Polygon"
            ]
        else:
            source_shapes = [
                "Tri Prism" if s == "Triangular Prism" else s
                for s in self.shape_data.get("3D Solids", [])
            ]

        # Transfer list in input_frame (left columns)
        self.composite_transfer = CompositeTransferList(
            parent=self.input_frame,
            available_shapes=source_shapes,
            on_change_callback=self._on_composite_change
        )
        self.composite_transfer.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ── Labels and Lines column (col_dimlines) ───────────────────────────
        # Clear and populate col_dimlines exactly like _build_standalone_label_ui does.
        for w in self.col_dimlines.winfo_children():
            w.destroy()

        tk.Label(self.col_dimlines, text="Labels and Lines", bg=AppConstants.BG_COLOR,
                 font=("Arial", 9, "bold")).pack(side=tk.TOP, anchor="center", pady=(0, 2))

        # Entry + "+ Text" row
        entry_row = tk.Frame(self.col_dimlines, bg=AppConstants.BG_COLOR)
        entry_row.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))

        self._composite_label_entry = tk.Entry(entry_row, width=10, font=AppConstants.BTN_FONT)
        self._composite_label_entry.pack(side=tk.LEFT, padx=(0, 2), fill=tk.X, expand=True)
        self._composite_label_entry.bind('<Return>', lambda e: self._confirm_composite_label())

        self._composite_text_btn = tk.Button(entry_row, text="+ Text", font=AppConstants.BTN_FONT,
                                              command=self._confirm_composite_label)
        self._composite_text_btn.pack(side=tk.LEFT, padx=1)

        self._composite_cancel_btn = tk.Button(entry_row, text="Cancel", font=AppConstants.BTN_FONT,
                                               command=self._cancel_composite_edit)
        # Hidden by default — shown only during edit mode

        # 2-column preset grid  (Height | Width  /  Radius | + Line)
        btn_grid = tk.Frame(self.col_dimlines, bg=AppConstants.BG_COLOR)
        btn_grid.pack(side=tk.TOP, fill=tk.X)
        btn_grid.columnconfigure(0, weight=1)
        btn_grid.columnconfigure(1, weight=1)

        presets = [("Height", "height"), ("Width", "width"), ("Radius", "radius")]
        for i, (label, key) in enumerate(presets):
            tk.Button(btn_grid, text=label, font=AppConstants.BTN_FONT,
                      command=lambda k=key: self._start_dim_line_mode(k)
                      ).grid(row=i // 2, column=i % 2, padx=1, pady=1, sticky="ew")

        tk.Button(btn_grid, text="+ Line", font=AppConstants.BTN_FONT,
                  command=lambda: self._start_dim_line_mode("free")
                  ).grid(row=1, column=1, padx=1, pady=1, sticky="ew")

        tk.Button(self.col_dimlines, text="Delete Selected", font=AppConstants.BTN_FONT,
                  command=self._remove_selected_annotation).pack(side=tk.TOP, pady=(2, 0))

        # Status label for dim-line placement feedback
        self._dim_status_label = tk.Label(self.col_dimlines, text="", bg=AppConstants.BG_COLOR,
                                           font=("Arial", 8, "italic"), fg="#0066cc")
        self._dim_status_label.pack(side=tk.TOP, fill=tk.X, pady=(1, 0))

        self.col_dimlines.grid()
        
        # Cached view limits — only recalculated on shape add/remove, not during drag
        self._composite_view_limits = None
        self._composite_bboxes = []
        self._composite_snap_anchors = []
        
        # Reset positions and transforms for new composite session
        self._composite_positions = {}
        self._composite_transforms = {}
        self._composite_selected = set()
        self._composite_labels = []
        self._composite_selected_label = None
        self._composite_dim_lines = []
        self._composite_selected_dim = None
        self._composite_dim_mode = None
        self._composite_dim_first_point = None
        
        # Connect drag events for composite mode
        self._connect_composite_drag()
        
        # Capture the initial empty composite state so undo can return to it
        if not self.history_manager.is_restoring:
            self._capture_current_state()
    
    def _schedule_delayed_capture(self) -> None:
        """Schedule a single delayed state capture, cancelling any pending one."""
        if self._capture_after_id is not None:
            try:
                self.root.after_cancel(self._capture_after_id)
            except tk.TclError:
                pass
        self._capture_after_id = self.root.after(100, self._delayed_capture)
    
    def _delayed_capture(self) -> None:
        """Execute the delayed capture."""
        self._capture_after_id = None
        if not self.history_manager.is_restoring:
            self._capture_current_state()

    def _on_composite_change(self, operation: tuple = None) -> None:
        self.composite_ctrl.on_change(operation)

    def _on_composite_delete(self, event=None) -> None:
        self.composite_ctrl.on_delete(event)

    def _on_delete_shortcut(self, event=None) -> None:
        """Handle Delete/Backspace key for both composite and standalone modes."""
        if isinstance(self.root.focus_get(), tk.Entry):
            return
        if self._is_composite_shape():
            self._on_composite_delete()
            return
        # Standalone mode — delete selected annotation or circumference
        if (self._standalone_selected_dim is not None or 
            self._standalone_selected_label is not None or
            self._builtin_selected is not None):
            self._remove_standalone_annotation()
    
    def _draw_composite_shapes(self, selected_shapes: list[str]) -> None:
        """Draw each selected shape, positioned by grid layout + user drag offsets."""
        n = len(selected_shapes)
        if n == 0:
            self._composite_bboxes = []
            return
        
        default_transform = TransformState()
        
        # Phase 1: Draw each shape at origin, capture its artists and bounds
        shape_data_list = []
        
        for i, shape_name in enumerate(selected_shapes):
            patches_before = len(self.ax.patches)
            texts_before = len(self.ax.texts)
            lines_before = len(self.ax.lines)
            
            ctx = self.plot_controller.create_drawing_context(
                aspect_ratio=1.0
            )
            ctx.composite_mode = True  # suppress standalone-only decorations (e.g. Square hashmarks)
            
            config = ShapeConfigProvider.get(shape_name)
            params = {}
            for label, default in zip(config.labels, config.default_values):
                params[label] = default
                params[f"{label}_num"] = None
            params["triangle_type"] = "Custom"
            params["dimension_mode"] = "Default"
            params["peak_offset"] = 0.5
            params["parallelogram_slope"] = 1.0
            
            deps = self.plot_controller.create_drawing_deps()
            drawer = ShapeRegistry.get_drawer(shape_name, deps)
            
            if drawer is None:
                shape_data_list.append({"name": shape_name, "error": True})
                continue
            
            try:
                # Apply per-shape transform if set
                shape_transform = TransformState()
                t = self._composite_transforms.get(i, {})
                if t:
                    shape_transform.flip_h = t.get("flip_h", False)
                    shape_transform.flip_v = t.get("flip_v", False)
                    shape_transform.base_side = t.get("base_side", 0)
                
                # For ALL geometric shapes (2D and 3D): draw at canonical
                # orientation, then apply post-draw artist transforms.
                # This ensures rotation/flip behave identically to standalone.
                actual_bs = shape_transform.base_side
                actual_fh = shape_transform.flip_h
                actual_fv = shape_transform.flip_v
                is_unified = shape_name in GeometricRotation.ALL_GEOMETRIC_SHAPES

                if is_unified and shape_name in GeometricRotation.ARC_SYMMETRIC_SHAPES:
                    _ns = 4  # all arc-symmetric shapes have num_sides=4
                    if actual_fv:
                        actual_bs = (actual_bs + 2) % _ns
                        actual_fv = False
                    if actual_fh:
                        # flip_h swaps left<->right orientations (1<->3);
                        # at 0 or 2 it's a visual no-op for these symmetric shapes.
                        if actual_bs in (1, 3):
                            actual_bs = _ns - actual_bs  # 1->3, 3->1
                        actual_fh = False
                if is_unified:
                    shape_transform = TransformState(flip_h=False, flip_v=False, base_side=0)
                
                drawer.draw(ctx, shape_transform, params)

                # Capture the xlim/ylim set by the drawer's set_limits() call.
                # This reflects the shape's own geometry bounds (the tight bbox the
                # drawer explicitly set) BEFORE any dimension-line or label artists
                # extend the axes limits.  Using this as the canonical center ensures
                # transform_artist_lists rotates/flips around the true visual center
                # of the shape, not a center skewed by far-reaching dimension lines.
                _post_draw_xlim = self.ax.get_xlim()
                _post_draw_ylim = self.ax.get_ylim()
                _canonical_cx = (_post_draw_xlim[0] + _post_draw_xlim[1]) / 2
                _canonical_cy = (_post_draw_ylim[0] + _post_draw_ylim[1]) / 2

                # Collect semantic snap anchors in canonical space right after draw
                snap_anchors = []
                if hasattr(drawer, 'get_snap_anchors'):
                    snap_anchors = drawer.get_snap_anchors(ctx, shape_transform, params)

                # For ALL geometric shapes: apply geometric rotate-then-flip to
                # only this shape's newly-added artists (same pipeline as
                # standalone generate_plot but scoped to avoid affecting other shapes).
                if is_unified:
                    _cfg = ShapeConfigProvider.get(shape_name)
                    _num_sides = _cfg.num_sides if _cfg.num_sides > 0 else 4
                    _bs = actual_bs
                    _fh = actual_fh
                    _fv = actual_fv
                    if _bs != 0 or _fh or _fv:
                        _new_p = list(self.ax.patches[patches_before:])
                        _new_l = list(self.ax.lines[lines_before:])
                        _new_t = list(self.ax.texts[texts_before:])
                        # Use the bounds-derived center captured right after draw().
                        # This avoids the dimension-line skew that occurs when
                        # computing center from all artist data points.
                        _cx = _canonical_cx
                        _cy = _canonical_cy
                        _angle_deg = -(360.0 / _num_sides) * _bs if _bs != 0 else 0.0
                        GeometricRotation.transform_artist_lists(
                            _new_p, _new_l, _new_t,
                            angle_deg=_angle_deg, flip_h=_fh, flip_v=_fv,
                            cx=_cx, cy=_cy)
                        # Also rotate the raw snap anchors so they match the
                        # rotated artist positions.  Without this the origin
                        # anchor (last element, representing the shape centre)
                        # stays in canonical space and the Pass-3 correction
                        # in _composite_rotate lands the shape in the wrong place.
                        if snap_anchors:
                            _angle_rad = math.radians(_angle_deg)
                            rotated = []
                            for _ax_pt, _ay_pt in snap_anchors:
                                if _angle_rad != 0.0:
                                    _ax_pt, _ay_pt = GeometricRotation.rotate_point(
                                        _ax_pt, _ay_pt, _angle_rad, _cx, _cy)
                                if _fh:
                                    _ax_pt = 2 * _cx - _ax_pt
                                if _fv:
                                    _ay_pt = 2 * _cy - _ay_pt
                                rotated.append((_ax_pt, _ay_pt))
                            snap_anchors = rotated

                # Compute shape bounds from patches + snap anchors (after any transforms).
                # Patch data covers ellipses and polygon faces; snap anchors cover
                # key geometry points (e.g. cone apex, prism corners) that may be
                # represented by Line2D artists rather than patches.  Pure Line2D
                # artists are still excluded from direct scanning since dimension
                # lines extend far beyond the shape footprint.
                new_patches = list(self.ax.patches[patches_before:])
                new_texts = list(self.ax.texts[texts_before:])
                new_lines = list(self.ax.lines[lines_before:])

                b_xs, b_ys = [], []
                for _p in new_patches:
                    _cls = type(_p).__name__
                    try:
                        if _cls == 'Polygon':
                            _xy = _p.get_xy()
                            b_xs.extend(v[0] for v in _xy)
                            b_ys.extend(v[1] for v in _xy)
                        elif _cls in ('Ellipse', 'Circle', 'Arc'):
                            _ecx, _ecy = _p.center
                            _ew = _p.width / 2
                            _eh = _p.height / 2
                            _ang = math.radians(getattr(_p, 'angle', 0))
                            _ca, _sa = math.cos(_ang), math.sin(_ang)
                            _dx = math.sqrt((_ew * _ca)**2 + (_eh * _sa)**2)
                            _dy = math.sqrt((_ew * _sa)**2 + (_eh * _ca)**2)
                            b_xs.extend([_ecx - _dx, _ecx + _dx])
                            b_ys.extend([_ecy - _dy, _ecy + _dy])
                    except Exception:
                        pass

                # Also include snap anchors in the bounds — anchors explicitly
                # encode the shape's key geometry points (e.g., cone apex, prism
                # corners) that may be represented by Line2D artists rather than
                # patches, and therefore missed by patch-only scanning.
                for _ax_pt, _ay_pt in snap_anchors:
                    b_xs.append(_ax_pt)
                    b_ys.append(_ay_pt)

                if b_xs and b_ys:
                    shape_xlim = (min(b_xs), max(b_xs))
                    shape_ylim = (min(b_ys), max(b_ys))
                else:
                    # Fallback to the canonical bounds set by the drawer's set_limits()
                    shape_xlim = tuple(_post_draw_xlim)
                    shape_ylim = tuple(_post_draw_ylim)

                shape_data_list.append({
                    "name": shape_name,
                    "error": False,
                    "xlim": shape_xlim,
                    "ylim": shape_ylim,
                    "patches": new_patches,
                    "texts": new_texts,
                    "lines": new_lines,
                    "snap_anchors": snap_anchors,
                })
            except Exception as e:
                logger.warning("Error drawing composite shape %s: %s", shape_name, e)
                shape_data_list.append({"name": shape_name, "error": True})
        
        # Phase 2: Calculate grid layout for initial positions of NEW (unplaced) shapes.
        # Shapes that already have an absolute position stored are NOT moved — this
        # prevents existing objects jumping when a new shape is added and the grid reflows.
        valid = [s for s in shape_data_list if not s.get("error")]
        if not valid:
            self._composite_bboxes = []
            return
        
        # Use a fixed 3-column grid so layout never reflows when shapes are added.
        # Columns are fixed; new rows always grow downward (negative Y), so existing
        # placed shapes are never displaced when additional shapes are appended.
        GRID_COLS = 3
        cols = min(n, GRID_COLS)
        rows = math.ceil(n / GRID_COLS)
        
        max_w = max((s["xlim"][1] - s["xlim"][0]) for s in valid)
        max_h = max((s["ylim"][1] - s["ylim"][0]) for s in valid)
        cell_w = max(max_w * 1.4, 5.0)
        cell_h = max(max_h * 1.4, 4.0)
        
        # Assign absolute positions to any shapes that haven't been placed yet.
        # Row 0 is at y=0, row 1 at y=-cell_h, row 2 at y=-2*cell_h, etc.
        # This downward-growing scheme means existing positions are never invalidated
        # when more shapes (and thus more rows) are added later.
        for i in range(n):
            if i not in self._composite_positions:
                row_idx = i // GRID_COLS
                col_idx = i % GRID_COLS
                grid_cx = (col_idx + 0.5) * cell_w
                grid_cy = -(row_idx + 0.5) * cell_h  # row 0 → -0.5*cell_h, row 1 → -1.5*cell_h
                self._composite_positions[i] = (grid_cx, grid_cy)
        
        # Phase 3: Translate each shape to its stored absolute canvas position
        self._composite_bboxes = []
        self._composite_snap_anchors = []
        
        for i, sdata in enumerate(shape_data_list):
            # Absolute target center (never recomputed from grid once set)
            target_cx, target_cy = self._composite_positions.get(i, (0.0, 0.0))
            
            if sdata.get("error"):
                self.ax.text(target_cx, target_cy, f"[{sdata['name']}]",
                            ha="center", va="center", fontsize=10, color="orange")
                self._composite_bboxes.append((0, 0, 0, 0))
                self._composite_snap_anchors.append([])
                continue
            
            # Shape's drawn center
            shape_cx = (sdata["xlim"][0] + sdata["xlim"][1]) / 2
            shape_cy = (sdata["ylim"][0] + sdata["ylim"][1]) / 2
            dx = target_cx - shape_cx
            dy = target_cy - shape_cy
            
            # Half-dimensions for bounding box
            half_w = (sdata["xlim"][1] - sdata["xlim"][0]) / 2
            half_h = (sdata["ylim"][1] - sdata["ylim"][0]) / 2
            
            # Store bounding box for hit-testing (x_min, y_min, x_max, y_max)
            self._composite_bboxes.append((
                target_cx - half_w, target_cy - half_h,
                target_cx + half_w, target_cy + half_h
            ))
            
            # Translate all patches
            for p in sdata["patches"]:
                self._translate_patch(p, dx, dy)
            
            # Translate lines
            for line in sdata["lines"]:
                xdata = np.array(line.get_xdata(), dtype=float)
                ydata = np.array(line.get_ydata(), dtype=float)
                line.set_xdata(xdata + dx)
                line.set_ydata(ydata + dy)
            
            # Translate texts
            for t in sdata["texts"]:
                if t.get_transform() != self.ax.transData:
                    if hasattr(t, 'get_unitless_position'):
                        tx, ty = t.get_position()
                        t.set_position((tx + dx, ty + dy))
                    continue
                tx, ty = t.get_position()
                t.set_position((tx + dx, ty + dy))
            
            # Translate snap anchors
            translated_anchors = [(ax + dx, ay + dy) for ax, ay in sdata.get("snap_anchors", [])]
            self._composite_snap_anchors.append(translated_anchors)
        
        # Draw selection highlight around all selected shapes
        for sel_idx in self._composite_selected:
            if sel_idx < len(self._composite_bboxes):
                sel_bbox = self._composite_bboxes[sel_idx]
                if sel_bbox != (0, 0, 0, 0):
                    pad = 0.8
                    sel_rect = patches.FancyBboxPatch(
                        (sel_bbox[0] - pad, sel_bbox[1] - pad),
                        sel_bbox[2] - sel_bbox[0] + 2 * pad,
                        sel_bbox[3] - sel_bbox[1] + 2 * pad,
                        boxstyle="round,pad=0.2",
                        edgecolor="#4488ff", facecolor="none",
                        linewidth=1.5, linestyle="--", alpha=0.7, zorder=15
                    )
                    self.ax.add_patch(sel_rect)
        
        # Phase 4: Set view — use fixed page-like limits for consistent canvas
        # Calculate bounds that encompass all shapes including drag offsets
        all_x_min = min(b[0] for b in self._composite_bboxes if b != (0, 0, 0, 0)) if self._composite_bboxes else 0
        all_x_max = max(b[2] for b in self._composite_bboxes if b != (0, 0, 0, 0)) if self._composite_bboxes else cols * cell_w
        all_y_min = min(b[1] for b in self._composite_bboxes if b != (0, 0, 0, 0)) if self._composite_bboxes else 0
        all_y_max = max(b[3] for b in self._composite_bboxes if b != (0, 0, 0, 0)) if self._composite_bboxes else rows * cell_h
        
        # Expand bounds to include annotation positions (labels and dim lines)
        for lbl in self._composite_labels:
            all_x_min = min(all_x_min, lbl["x"] - 1.0)
            all_x_max = max(all_x_max, lbl["x"] + 1.0)
            all_y_min = min(all_y_min, lbl["y"] - 1.0)
            all_y_max = max(all_y_max, lbl["y"] + 1.0)
        for dim in self._composite_dim_lines:
            for key_x, key_y in [("x1", "y1"), ("x2", "y2"), ("label_x", "label_y")]:
                if key_x in dim and key_y in dim:
                    all_x_min = min(all_x_min, dim[key_x] - 0.5)
                    all_x_max = max(all_x_max, dim[key_x] + 0.5)
                    all_y_min = min(all_y_min, dim[key_y] - 0.5)
                    all_y_max = max(all_y_max, dim[key_y] + 0.5)
        
        # Determine if we're in a drag operation (use cached limits)
        is_dragging = (self._composite_drag_state is not None 
                       or self._composite_label_drag is not None)
        
        if is_dragging and self._composite_view_limits is not None:
            # Reuse cached view limits during drag for visual stability
            vx_min, vx_max, vy_min, vy_max = self._composite_view_limits
            self.ax.set_xlim(vx_min, vx_max)
            self.ax.set_ylim(vy_min, vy_max)
        else:
            # Recalculate view bounds
            content_w = all_x_max - all_x_min
            content_h = all_y_max - all_y_min
            center_x = (all_x_min + all_x_max) / 2
            center_y = (all_y_min + all_y_max) / 2
            
            margin_x = content_w * 0.12
            margin_y = content_h * 0.12
            padded_w = content_w + 2 * margin_x
            padded_h = content_h + 2 * margin_y
            
            pixel_aspect = getattr(self, '_axes_pixel_aspect', 4.0/3.0)
            content_aspect = padded_w / padded_h if padded_h > 0 else pixel_aspect
            
            if content_aspect > pixel_aspect:
                final_w = padded_w
                final_h = final_w / pixel_aspect
            else:
                final_h = padded_h
                final_w = final_h * pixel_aspect
            
            vx_min = center_x - final_w / 2
            vx_max = center_x + final_w / 2
            vy_min = center_y - final_h / 2
            vy_max = center_y + final_h / 2
            
            self.ax.set_xlim(vx_min, vx_max)
            self.ax.set_ylim(vy_min, vy_max)
            
            # Cache for drag stability
            self._composite_view_limits = (vx_min, vx_max, vy_min, vy_max)
        
        # Determine which annotations are "in group" for visual feedback.
        # Works for both single-shape selection (highlight annotations within that
        # shape's bbox) and multi-shape groups (highlight annotations within the
        # combined bbox).  This makes it clear which labels/dim lines belong to
        # the selected shape(s) and will move if a transform is applied.
        _annotations_in_group = set()
        _dim_lines_in_group = set()
        if len(self._composite_selected) == 1:
            sel_idx = next(iter(self._composite_selected))
            if sel_idx < len(self._composite_bboxes):
                s_bb = self._composite_bboxes[sel_idx]
                if s_bb != (0, 0, 0, 0):
                    in_lbl, in_dim = self._get_annotations_in_region(*s_bb)
                    _annotations_in_group = set(in_lbl)
                    _dim_lines_in_group = set(in_dim)
        elif len(self._composite_selected) > 1:
            group_bb = self._get_group_bbox()
            if group_bb != (0, 0, 0, 0):
                all_sel = self._all_shapes_selected()
                in_lbl, in_dim = self._get_annotations_in_region(*group_bb, include_all=all_sel)
                _annotations_in_group = set(in_lbl)
                _dim_lines_in_group = set(in_dim)
        
        # Phase 5: Draw composite labels
        self._composite_label_bboxes = []
        for i, lbl in enumerate(self._composite_labels):
            is_selected = (i == self._composite_selected_label)
            in_group = (i in _annotations_in_group)
            color = "#0066cc" if (is_selected or in_group) else "black"
            weight = "bold" if is_selected else "normal"
            font_size = self.font_size_var.get() if self.font_size_var is not None else 12
            
            edge = color if (is_selected or in_group) else "none"
            txt = self.ax.text(lbl["x"], lbl["y"], lbl["text"],
                             fontsize=font_size, color=color, fontweight=weight,
                             fontfamily=getattr(self, 'font_family', AppConstants.DEFAULT_FONT_FAMILY),
                             ha="center", va="center", zorder=AppConstants.LABEL_ZORDER + 1,
                             bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                      edgecolor=edge,
                                      alpha=0.9))
            
            # Store approximate bbox for hit testing
            pad_x, pad_y = self._pixels_to_data_pad(3)
            try:
                renderer = self.fig.canvas.get_renderer()
                bb = txt.get_window_extent(renderer=renderer)
                inv = self.ax.transData.inverted()
                bl = inv.transform([bb.x0, bb.y0])
                tr = inv.transform([bb.x1, bb.y1])
                half_w = (tr[0] - bl[0]) / 2 + pad_x
                half_h = (tr[1] - bl[1]) / 2 + pad_y
            except Exception:
                char_w = font_size * 0.12
                char_h = font_size * 0.2
                half_w = (len(lbl["text"]) * char_w / 2) + pad_x
                half_h = (char_h / 2) + pad_y
            self._composite_label_bboxes.append((
                lbl["x"] - half_w, lbl["y"] - half_h,
                lbl["x"] + half_w, lbl["y"] + half_h
            ))

        # Phase 6: Draw dimension lines
        self._composite_dim_endpoints = []
        self._composite_dim_label_bboxes = []
        font_size = self.font_size_var.get() if self.font_size_var is not None else 12
        
        for i, dim in enumerate(self._composite_dim_lines):
            is_selected = (i == self._composite_selected_dim)
            in_group = (i in _dim_lines_in_group)
            color = "#0066cc" if (is_selected or in_group) else "black"
            lw = 1.5 if (is_selected or in_group) else AppConstants.DIMENSION_LINE_WIDTH
            
            x1, y1, x2, y2 = dim["x1"], dim["y1"], dim["x2"], dim["y2"]
            
            # Draw dashed dimension line
            self.ax.plot([x1, x2], [y1, y2],
                        color=color, linestyle="--", linewidth=lw, zorder=12)
            
            # Draw small endpoint markers
            marker_size = 4 if is_selected else 3
            for px, py in [(x1, y1), (x2, y2)]:
                self.ax.plot(px, py, marker="o", markersize=marker_size,
                           color=color, zorder=13)
            
            # Draw label at stored position (or midpoint if not set)
            label_x = dim.get("label_x", (x1 + x2) / 2)
            label_y = dim.get("label_y", (y1 + y2) / 2)
            
            label_color = "#0066cc" if (is_selected or in_group) else "black"
            dim_txt = self.ax.text(label_x, label_y, dim["text"],
                        fontsize=font_size, color=label_color,
                        fontweight="bold" if is_selected else "normal",
                        fontfamily=getattr(self, 'font_family', AppConstants.DEFAULT_FONT_FAMILY),
                        ha="center", va="center", zorder=AppConstants.LABEL_ZORDER + 1,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                 edgecolor=color if (is_selected or in_group) else "none", alpha=1))
            
            # Store endpoints for hit testing / dragging
            self._composite_dim_endpoints.append({
                "p1": (x1, y1),
                "p2": (x2, y2),
            })
            
            # Store label bbox for click-to-select
            pad_x, pad_y = self._pixels_to_data_pad(3)
            try:
                renderer = self.fig.canvas.get_renderer()
                bb = dim_txt.get_window_extent(renderer=renderer)
                inv = self.ax.transData.inverted()
                bl = inv.transform([bb.x0, bb.y0])
                tr = inv.transform([bb.x1, bb.y1])
                half_w = (tr[0] - bl[0]) / 2 + pad_x
                half_h = (tr[1] - bl[1]) / 2 + pad_y
            except Exception:
                char_w = font_size * 0.08
                char_h = font_size * 0.15
                half_w = len(dim["text"]) * char_w / 2 + pad_x
                half_h = char_h / 2 + pad_y
            self._composite_dim_label_bboxes.append((
                label_x - half_w, label_y - half_h,
                label_x + half_w, label_y + half_h
            ))

    # Preset dimension line configurations per shape
    _DIM_PRESETS: dict[str, list[dict]] = {
        "Rectangle": [{"key": "height", "label": "Width", "default_text": "w"},
                      {"key": "width", "label": "Length", "default_text": "l"}],
        "Square": [{"key": "side_v", "label": "Side V", "default_text": "v"},
                   {"key": "side_h", "label": "Side H", "default_text": "h"}],
        "Triangle": [{"key": "height", "label": "Height", "default_text": "h"},
                     {"key": "width", "label": "Base", "default_text": "b"},
                     {"key": "side_l", "label": "Side L", "default_text": "l"},
                     {"key": "side_r", "label": "Side R", "default_text": "r"}],
        "Circle": [{"key": "radius", "label": "Radius", "default_text": "r"},
                   {"key": "diameter", "label": "Diameter", "default_text": "d"},
                   {"key": "circumference", "label": "Circ.", "default_text": "c"}],
        "Parallelogram": [{"key": "para_height", "label": "Height", "default_text": "h"},
                          {"key": "para_base", "label": "Base", "default_text": "b"},
                          {"key": "para_top", "label": "Top", "default_text": "t"},
                          {"key": "para_side_l", "label": "Side L", "default_text": "l"},
                          {"key": "para_side_r", "label": "Side R", "default_text": "r"}],
        "Trapezoid": [{"key": "trap_height", "label": "Height", "default_text": "h"},
                      {"key": "trap_base", "label": "Base", "default_text": "b"},
                      {"key": "trap_top", "label": "Top", "default_text": "t"},
                      {"key": "trap_side_l", "label": "Side L", "default_text": "l"},
                      {"key": "trap_side_r", "label": "Side R", "default_text": "r"}],
        "Polygon": [{"key": "side", "label": "Side", "default_text": "s"}],
        "Sphere": [{"key": "radius", "label": "Radius", "default_text": "r"},
                   {"key": "diameter", "label": "Diameter", "default_text": "d"},
                   {"key": "circumference", "label": "Circ.", "default_text": "c"}],
        "Cylinder": [{"key": "height", "label": "Height", "default_text": "h"},
                     {"key": "radius", "label": "Radius", "default_text": "r"},
                     {"key": "diameter", "label": "Diameter", "default_text": "d"},
                     {"key": "circumference", "label": "Circ.", "default_text": "c"}],
        "Cone": [{"key": "height", "label": "Height", "default_text": "h"},
                 {"key": "radius", "label": "Radius", "default_text": "r"},
                 {"key": "diameter", "label": "Diameter", "default_text": "d"},
                 {"key": "circumference", "label": "Circ.", "default_text": "c"}],
        "Hemisphere": [{"key": "radius", "label": "Radius", "default_text": "r"},
                       {"key": "diameter", "label": "Diameter", "default_text": "d"},
                       {"key": "circumference", "label": "Circ.", "default_text": "c"}],
        "Rectangular Prism": [{"key": "height", "label": "Height", "default_text": "h"},
                              {"key": "length", "label": "Length", "default_text": "l"},
                              {"key": "width", "label": "Width", "default_text": "w"}],
        "Triangular Prism": [{"key": "height", "label": "Height", "default_text": "h"},
                             {"key": "tri_base", "label": "Base", "default_text": "b"},
                             {"key": "tri_length", "label": "Length", "default_text": "l"}],
        "Tri Prism": [{"key": "height", "label": "Height", "default_text": "h"},
                      {"key": "tri_base", "label": "Base", "default_text": "b"},
                      {"key": "tri_length", "label": "Length", "default_text": "l"}],
        "Tri Triangle": [{"key": "height", "label": "Height", "default_text": "h"},
                          {"key": "width", "label": "Base", "default_text": "b"},
                          {"key": "side_l", "label": "Side L", "default_text": "l"},
                          {"key": "side_r", "label": "Side R", "default_text": "r"}],
        "Triangle_Right": [{"key": "leg_b",  "label": "Leg B",       "default_text": "b"},
                            {"key": "leg_a",  "label": "Leg A",       "default_text": "a"},
                            {"key": "hyp",    "label": "Hypotenuse",  "default_text": "c"}],
    }

    # ------------------------------------------------------------------
    # Dimension-line endpoint calculators — one private method per preset key.
    # Each receives the pre-extracted common variables so there is no repeated
    # book-keeping.  _calc_dim_line_endpoints is the public dispatcher.
    # Adding a new preset key: implement _calc_dim_<key> and add it to
    # _DIM_DISPATCH (built lazily on first call to avoid forward-reference issues).
    # ------------------------------------------------------------------

    def _calc_dim_line_endpoints(self, preset_key: str) -> dict | None:
        """Dispatch to a per-preset calculator.

        Uses _shape_bounds captured during generate_plot (tight bounds before
        view scale).  Returns dict with x1,y1,x2,y2,label_x,label_y or None.
        """
        if not hasattr(self, '_shape_bounds') or not self._shape_bounds:
            return None

        b = self._shape_bounds
        hints  = self.label_manager.geometry_hints
        offset = DrawingUtilities.dim_offset_from_axes(self.ax)
        label_gap = DrawingUtilities.dim_label_gap_from_axes(self.ax)

        # Build dispatch table once per instance on first call.
        # Safe to cache permanently: all values are bound methods on `self`,
        # so they always reflect the current instance state. The shape-specific
        # _calc_dim_* methods exist by the time __init__ finishes, so no
        # partial-object risk in production. No invalidation needed.
        if not hasattr(self, '_dim_dispatch'):
            self._dim_dispatch = {
                "height":       self._calc_dim_height,
                "width":        self._calc_dim_width,
                "tri_base":     self._calc_dim_tri_base,
                "tri_length":   self._calc_dim_tri_length,
                "side_v":       self._calc_dim_side_v,
                "side_h":       self._calc_dim_side_h,
                "side_l":       self._calc_dim_side_l,
                "side_r":       self._calc_dim_side_r,
                "para_height":  self._calc_dim_para_trap_height,
                "trap_height":  self._calc_dim_para_trap_height,
                "para_base":    self._calc_dim_para_base,
                "para_top":     self._calc_dim_para_top,
                "para_side_l":  self._calc_dim_para_side_l,
                "para_side_r":  self._calc_dim_para_side_r,
                "trap_base":    self._calc_dim_trap_base,
                "trap_top":     self._calc_dim_trap_top,
                "trap_side_l":  self._calc_dim_trap_side_l,
                "trap_side_r":  self._calc_dim_trap_side_r,
                "radius":       self._calc_dim_radius,
                "diameter":     self._calc_dim_diameter,
                "slant":        self._calc_dim_slant,
                "length":       self._calc_dim_length,
                "side":         self._calc_dim_side,
                "circumference": self._calc_dim_circumference,
                "leg_a":        self._calc_dim_leg_a,
                "leg_b":        self._calc_dim_leg_b,
                "hyp":          self._calc_dim_hyp,
            }

        fn = self._dim_dispatch.get(preset_key)
        if fn is None:
            return None
        return fn(hints, b, offset, label_gap)

    # ---- shared geometry helpers ----------------------------------------

    @staticmethod
    def _dim_perp_offset(p1, p2, offset, label_gap, shape_cx, shape_cy):
        """Return endpoints + label pos for a dim line offset perp to p1→p2."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        seg_len = math.sqrt(dx**2 + dy**2)
        if seg_len <= 0.001:
            return None
        nx, ny = -dy / seg_len, dx / seg_len
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2
        if nx * (mid_x + nx - shape_cx) + ny * (mid_y + ny - shape_cy) < 0:
            nx, ny = -nx, -ny
        return {
            "x1": p1[0] + nx * offset, "y1": p1[1] + ny * offset,
            "x2": p2[0] + nx * offset, "y2": p2[1] + ny * offset,
            "label_x": mid_x + nx * (offset + label_gap),
            "label_y": mid_y + ny * (offset + label_gap),
            "constraint": None,
        }

    @staticmethod
    def _dim_edge_offset(p1, p2, offset, label_gap):
        """Offset a dim line perp to p1→p2 without shape-center correction."""
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        seg_len = math.sqrt(dx**2 + dy**2) or 1
        nx, ny = -dy / seg_len, dx / seg_len
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        return {
            "x1": p1[0] + nx * offset, "y1": p1[1] + ny * offset,
            "x2": p2[0] + nx * offset, "y2": p2[1] + ny * offset,
            "label_x": mx + nx * (offset + label_gap),
            "label_y": my + ny * (offset + label_gap),
            "constraint": None,
        }

    # ---- per-preset calculators -----------------------------------------

    def _calc_dim_height(self, hints, b, offset, label_gap):
        shape_left   = b["x_min"]; shape_right = b["x_max"]
        shape_bottom = b["y_min"]; shape_top   = b["y_max"]
        shape_cx = (shape_left + shape_right) / 2
        shape_cy = (shape_bottom + shape_top) / 2

        # Rectangular Prism: height is the left front edge (f_bl→f_tl)
        if "prism_f_bl" in hints and "prism_f_tl" in hints:
            res = self._dim_perp_offset(
                hints["prism_f_bl"], hints["prism_f_tl"],
                offset, label_gap, shape_cx, shape_cy)
            if res:
                return res

        # Tri prism (standalone): height line from foot to apex, offset from face
        if "prism_tri_p1" in hints:
            foot = hints["tri_foot"]; apex = hints["tri_apex"]
            p1, p2, p3 = hints["prism_tri_p1"], hints["prism_tri_p2"], hints["prism_tri_p3"]
            mid_x = (foot[0] + apex[0]) / 2
            mid_y = (foot[1] + apex[1]) / 2
            dx = apex[0] - foot[0]; dy = apex[1] - foot[1]
            seg_len = math.sqrt(dx**2 + dy**2) or 1
            nx, ny = -dy / seg_len, dx / seg_len
            cx = (p1[0] + p2[0] + p3[0]) / 3
            cy = (p1[1] + p2[1] + p3[1]) / 3
            if nx * (mid_x + nx) + ny * (mid_y + ny) < nx * cx + ny * cy:
                nx, ny = -nx, -ny
            off2 = offset * 1.2
            return {
                "x1": foot[0] + nx * off2, "y1": foot[1] + ny * off2,
                "x2": apex[0] + nx * off2, "y2": apex[1] + ny * off2,
                "label_x": mid_x + nx * (off2 + label_gap),
                "label_y": mid_y + ny * (off2 + label_gap),
                "constraint": None,
            }

        # 2D Triangle: in-place height line from foot to apex
        if "tri_foot" in hints:
            foot = hints["tri_foot"]; apex = hints["tri_apex"]
            mid_x = (foot[0] + apex[0]) / 2
            mid_y = (foot[1] + apex[1]) / 2
            return {
                "x1": foot[0], "y1": foot[1],
                "x2": apex[0], "y2": apex[1],
                "label_x": mid_x + label_gap, "label_y": mid_y,
                "constraint": None,
            }

        if "height_y1" in hints:
            # Rectangle (rect_pts, no prism): dim line on right side
            if "rect_pts" in hints and "prism_f_bl" not in hints and "prism_tri_p1" not in hints:
                pts = hints["rect_pts"]
                res = self._dim_perp_offset(pts[1], pts[2], offset, label_gap, shape_cx, shape_cy)
                if res:
                    return res

            # Generic: build from height_x/y1/y2 (already rotated)
            hx = hints.get("height_x", shape_cx)
            y1 = hints["height_y1"]; y2 = hints["height_y2"]
            foot = (hx, y1); top = (hx, y2)
            res = self._dim_perp_offset(foot, top, offset, label_gap, shape_cx, shape_cy)
            if res:
                return res
            # Degenerate fallback
            x = shape_right + offset
            return {
                "x1": x, "y1": y1, "x2": x, "y2": y2,
                "label_x": x + label_gap, "label_y": (y1 + y2) / 2,
                "constraint": "height",
            }

        x = shape_right + offset
        return {
            "x1": x, "y1": shape_bottom, "x2": x, "y2": shape_top,
            "label_x": x + label_gap, "label_y": shape_cy,
            "constraint": "height",
        }

    def _calc_dim_width(self, hints, b, offset, label_gap):
        shape_left   = b["x_min"]; shape_right = b["x_max"]
        shape_bottom = b["y_min"]; shape_top   = b["y_max"]
        shape_cx = (shape_left + shape_right) / 2
        shape_cy = (shape_bottom + shape_top) / 2

        # Rect prism: W on depth edge f_br→b_br, offset perp-outward with standard gap.
        if "prism_f_br" in hints and "prism_b_br" in hints:
            f_br = hints["prism_f_br"]; b_br = hints["prism_b_br"]
            mx = (f_br[0] + b_br[0]) / 2; my = (f_br[1] + b_br[1]) / 2
            dx = b_br[0] - f_br[0]; dy = b_br[1] - f_br[1]
            seg_len = math.sqrt(dx**2 + dy**2) or 1
            nx, ny = -dy / seg_len, dx / seg_len
            if nx * (mx - shape_cx) + ny * (my - shape_cy) < 0:
                nx, ny = -nx, -ny
            return {
                "x1": f_br[0] + nx * offset, "y1": f_br[1] + ny * offset,
                "x2": b_br[0] + nx * offset, "y2": b_br[1] + ny * offset,
                "label_x": mx + nx * (offset + label_gap),
                "label_y": my + ny * (offset + label_gap),
                "constraint": "width",
            }

        # Triangle base
        if "tri_foot" in hints and "tri_base_p1" in hints and "tri_base_p2" in hints:
            p1 = hints["tri_base_p1"]; p2 = hints["tri_base_p2"]
            res = self._dim_perp_offset(p1, p2, offset, label_gap, shape_cx, shape_cy)
            if res:
                return res
            y = shape_bottom - offset
            return {
                "x1": p1[0], "y1": y, "x2": p2[0], "y2": y,
                "label_x": (p1[0] + p2[0]) / 2, "label_y": y - label_gap,
                "constraint": "width",
            }

        # Rectangle: actual bottom edge from rect_pts
        pts = hints.get("rect_pts")
        if pts and len(pts) >= 4:
            res = self._dim_perp_offset(pts[0], pts[1], offset, label_gap, shape_cx, shape_cy)
            if res:
                return res

        y = shape_bottom - offset
        return {
            "x1": shape_left, "y1": y, "x2": shape_right, "y2": y,
            "label_x": shape_cx, "label_y": y - label_gap,
            "constraint": "width",
        }

    def _calc_dim_tri_base(self, hints, b, offset, label_gap):
        shape_cx = (b["x_min"] + b["x_max"]) / 2
        shape_cy = (b["y_min"] + b["y_max"]) / 2
        if "prism_tri_p1" not in hints:
            return None
        p1, p2 = hints["prism_tri_p1"], hints["prism_tri_p2"]
        res = self._dim_perp_offset(p1, p2, offset, label_gap, shape_cx, shape_cy)
        if res:
            return res
        y = min(p1[1], p2[1]) - offset
        return {
            "x1": p1[0], "y1": y, "x2": p2[0], "y2": y,
            "label_x": (p1[0] + p2[0]) / 2, "label_y": y - label_gap,
            "constraint": "width",
        }

    def _calc_dim_tri_length(self, hints, b, offset, label_gap):
        if "prism_tri_p1" not in hints:
            return None
        p1 = hints["prism_tri_p1"]; p2 = hints["prism_tri_p2"]
        b1 = hints["prism_tri_b1"]; b2 = hints["prism_tri_b2"]
        p3 = hints["prism_tri_p3"]
        flip_h = hints.get("prism_tri_flip_h", False)
        all_pts = [p1, p2, p3, b1, b2]
        shape_cy_tri = sum(p[1] for p in all_pts) / len(all_pts)
        ep1_raw, ep2_raw = (p1, b1) if flip_h else (p2, b2)
        dx = ep2_raw[0] - ep1_raw[0]; dy = ep2_raw[1] - ep1_raw[1]
        seg_len = math.sqrt(dx*dx + dy*dy) or 1
        nx, ny = -dy / seg_len, dx / seg_len
        if (ep1_raw[1] + ep2_raw[1]) / 2 + ny > shape_cy_tri:
            nx, ny = -nx, -ny
        ep1 = (ep1_raw[0] + nx * offset, ep1_raw[1] + ny * offset)
        ep2 = (ep2_raw[0] + nx * offset, ep2_raw[1] + ny * offset)
        mx = (ep1[0] + ep2[0]) / 2; my = (ep1[1] + ep2[1]) / 2
        return {
            "x1": ep1[0], "y1": ep1[1], "x2": ep2[0], "y2": ep2[1],
            "label_x": mx + nx * label_gap, "label_y": my + ny * label_gap,
            "constraint": None,
        }

    def _calc_dim_side_v(self, hints, b, offset, label_gap):
        shape_cx = (b["x_min"] + b["x_max"]) / 2
        shape_cy = (b["y_min"] + b["y_max"]) / 2
        pts = hints.get("rect_pts")
        if pts and len(pts) >= 4:
            res = self._dim_perp_offset(pts[1], pts[2], offset, label_gap, shape_cx, shape_cy)
            if res:
                return res
        x = b["x_max"] + offset
        return {
            "x1": x, "y1": b["y_min"], "x2": x, "y2": b["y_max"],
            "label_x": x + label_gap, "label_y": shape_cy,
            "constraint": "height",
        }

    def _calc_dim_side_h(self, hints, b, offset, label_gap):
        shape_cx = (b["x_min"] + b["x_max"]) / 2
        shape_cy = (b["y_min"] + b["y_max"]) / 2
        pts = hints.get("rect_pts")
        if pts and len(pts) >= 4:
            res = self._dim_perp_offset(pts[0], pts[1], offset, label_gap, shape_cx, shape_cy)
            if res:
                return res
        y = b["y_min"] - offset
        return {
            "x1": b["x_min"], "y1": y, "x2": b["x_max"], "y2": y,
            "label_x": shape_cx, "label_y": y - label_gap,
            "constraint": "width",
        }

    def _calc_dim_side_l(self, hints, b, offset, label_gap):
        if "tri_base_p1" not in hints or "tri_apex" not in hints:
            return None
        p1 = hints["tri_base_p1"]; p2 = hints["tri_apex"]
        return self._dim_edge_offset(p1, p2, offset, label_gap)

    def _calc_dim_side_r(self, hints, b, offset, label_gap):
        if "tri_base_p2" not in hints or "tri_apex" not in hints:
            return None
        p1 = hints["tri_apex"]; p2 = hints["tri_base_p2"]
        return self._dim_edge_offset(p1, p2, offset, label_gap)

    def _calc_dim_para_trap_height(self, hints, b, offset, label_gap):
        """Shared handler for para_height and trap_height."""
        # Determine which point-list key to use based on which is present
        pts = hints.get("para_pts") or hints.get("trap_pts")
        if not pts or len(pts) < 4:
            return None
        p0, p1, p2, p3 = pts[0], pts[1], pts[2], pts[3]
        bx, by = p1[0] - p0[0], p1[1] - p0[1]
        blen = math.sqrt(bx**2 + by**2)
        if blen > 0.001:
            ub = (bx / blen, by / blen)
            candidates = []
            for pt in [p2, p3]:
                v = (pt[0] - p0[0], pt[1] - p0[1])
                t = v[0] * ub[0] + v[1] * ub[1]
                is_internal = 0 <= t <= blen
                ext = 0 if is_internal else (abs(t) if t < 0 else abs(t - blen))
                candidates.append({"pt": pt, "t": t, "internal": is_internal, "ext": ext})
            internals = [c for c in candidates if c["internal"]]
            best = sorted(internals, key=lambda c: c["t"])[0] if internals else                    sorted(candidates, key=lambda c: c["ext"], reverse=True)[0]
            apex = best["pt"]
            foot = (p0[0] + ub[0] * best["t"], p0[1] + ub[1] * best["t"])
        else:
            foot, apex = p0, p3
        mid_y = (foot[1] + apex[1]) / 2
        return {
            "x1": foot[0], "y1": foot[1], "x2": apex[0], "y2": apex[1],
            "label_x": foot[0] + label_gap, "label_y": mid_y,
            "constraint": None,
        }

    def _calc_dim_para_base(self, hints, b, offset, label_gap):
        shape_cx = (b["x_min"] + b["x_max"]) / 2
        shape_cy = (b["y_min"] + b["y_max"]) / 2
        pts = hints.get("para_pts")
        if not pts or len(pts) < 4:
            return None
        p0, p1 = pts[0], pts[1]
        res = self._dim_perp_offset(p0, p1, offset, label_gap, shape_cx, shape_cy)
        if res:
            return res
        y = min(p0[1], p1[1]) - offset
        return {
            "x1": p0[0], "y1": y, "x2": p1[0], "y2": y,
            "label_x": (p0[0] + p1[0]) / 2, "label_y": y - label_gap,
            "constraint": None,
        }

    def _calc_dim_para_top(self, hints, b, offset, label_gap):
        shape_cx = (b["x_min"] + b["x_max"]) / 2
        shape_cy = (b["y_min"] + b["y_max"]) / 2
        pts = hints.get("para_pts")
        if not pts or len(pts) < 4:
            return None
        p2, p3 = pts[2], pts[3]
        res = self._dim_perp_offset(p3, p2, offset, label_gap, shape_cx, shape_cy)
        if res:
            return res
        y = max(p2[1], p3[1]) + offset
        return {
            "x1": p3[0], "y1": y, "x2": p2[0], "y2": y,
            "label_x": (p3[0] + p2[0]) / 2, "label_y": y + label_gap,
            "constraint": None,
        }

    def _calc_dim_para_side_l(self, hints, b, offset, label_gap):
        pts = hints.get("para_pts")
        if not pts or len(pts) < 4:
            return None
        return self._dim_edge_offset(pts[0], pts[3], offset, label_gap)

    def _calc_dim_para_side_r(self, hints, b, offset, label_gap):
        pts = hints.get("para_pts")
        if not pts or len(pts) < 4:
            return None
        p1, p2 = pts[1], pts[2]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        seg_len = math.sqrt(dx**2 + dy**2) or 1
        nx, ny = dy / seg_len, -dx / seg_len   # outward right normal
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        return {
            "x1": p1[0] + nx * offset, "y1": p1[1] + ny * offset,
            "x2": p2[0] + nx * offset, "y2": p2[1] + ny * offset,
            "label_x": mx + nx * (offset + label_gap),
            "label_y": my + ny * (offset + label_gap),
            "constraint": None,
        }

    def _calc_dim_trap_base(self, hints, b, offset, label_gap):
        pts = hints.get("trap_pts")
        if not pts or len(pts) < 4:
            return None
        p0, p1 = pts[0], pts[1]
        y = min(p0[1], p1[1]) - offset
        return {
            "x1": p0[0], "y1": y, "x2": p1[0], "y2": y,
            "label_x": (p0[0] + p1[0]) / 2, "label_y": y - label_gap,
            "constraint": None,
        }

    def _calc_dim_trap_top(self, hints, b, offset, label_gap):
        pts = hints.get("trap_pts")
        if not pts or len(pts) < 4:
            return None
        p2, p3 = pts[2], pts[3]
        y = max(p2[1], p3[1]) + offset
        return {
            "x1": p3[0], "y1": y, "x2": p2[0], "y2": y,
            "label_x": (p3[0] + p2[0]) / 2, "label_y": y + label_gap,
            "constraint": None,
        }

    def _calc_dim_trap_side_l(self, hints, b, offset, label_gap):
        pts = hints.get("trap_pts")
        if not pts or len(pts) < 4:
            return None
        return self._dim_edge_offset(pts[0], pts[3], offset, label_gap)

    def _calc_dim_trap_side_r(self, hints, b, offset, label_gap):
        pts = hints.get("trap_pts")
        if not pts or len(pts) < 4:
            return None
        p1, p2 = pts[1], pts[2]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        seg_len = math.sqrt(dx**2 + dy**2) or 1
        nx, ny = dy / seg_len, -dx / seg_len   # outward right normal
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        return {
            "x1": p1[0] + nx * offset, "y1": p1[1] + ny * offset,
            "x2": p2[0] + nx * offset, "y2": p2[1] + ny * offset,
            "label_x": mx + nx * (offset + label_gap),
            "label_y": my + ny * (offset + label_gap),
            "constraint": None,
        }

    def _calc_dim_radius(self, hints, b, offset, label_gap):
        shape_cx = (b["x_min"] + b["x_max"]) / 2
        shape_cy = (b["y_min"] + b["y_max"]) / 2
        shape_w  = b["x_max"] - b["x_min"]
        r = hints.get("radius", shape_w / 2)
        return {
            "x1": shape_cx, "y1": shape_cy,
            "x2": shape_cx + r, "y2": shape_cy,
            "label_x": shape_cx + r * 0.65,
            "label_y": shape_cy - offset * 2,
            "constraint": None,
        }

    def _calc_dim_diameter(self, hints, b, offset, label_gap):
        shape_cx = (b["x_min"] + b["x_max"]) / 2
        shape_cy = (b["y_min"] + b["y_max"]) / 2
        return {
            "x1": b["x_min"], "y1": shape_cy,
            "x2": b["x_max"], "y2": shape_cy,
            "label_x": shape_cx,
            "label_y": shape_cy - offset * 2,
            "constraint": None,
        }

    def _calc_dim_slant(self, hints, b, offset, label_gap):
        shape_right  = b["x_max"]
        shape_bottom = b["y_min"]
        shape_top    = b["y_max"]
        shape_cx = (b["x_min"] + b["x_max"]) / 2
        shape_cy = (b["y_min"] + b["y_max"]) / 2
        return {
            "x1": shape_right, "y1": shape_bottom,
            "x2": shape_cx,    "y2": shape_top,
            "label_x": (shape_right + shape_cx) / 2 + offset,
            "label_y": shape_cy,
            "constraint": None,
        }

    def _calc_dim_length(self, hints, b, offset, label_gap):
        shape_left   = b["x_min"]; shape_right = b["x_max"]
        shape_bottom = b["y_min"]
        shape_cx = (shape_left + shape_right) / 2

        if "prism_f_bl" in hints:
            # L: front-bottom edge f_bl→f_br, offset below with standard gap.
            f_bl = hints["prism_f_bl"]; f_br = hints["prism_f_br"]
            y = min(f_bl[1], f_br[1]) - offset
            mx = (f_bl[0] + f_br[0]) / 2
            return {
                "x1": f_bl[0], "y1": y,
                "x2": f_br[0], "y2": y,
                "label_x": mx, "label_y": y - label_gap,
                "constraint": None,
            }
        y = shape_bottom - offset
        return {
            "x1": shape_left, "y1": y, "x2": shape_right, "y2": y,
            "label_x": shape_cx, "label_y": y - label_gap,
            "constraint": "width",
        }

    def _calc_dim_side(self, hints, b, offset, label_gap):
        shape_left   = b["x_min"]; shape_right = b["x_max"]
        shape_bottom = b["y_min"]
        shape_cx = (shape_left + shape_right) / 2
        pts = hints.get("polygon_pts")
        if pts and len(pts) >= 2:
            n = len(pts)
            best_i, best_mx = 0, float("-inf")
            for i in range(n):
                p1 = pts[i]; p2 = pts[(i + 1) % n]
                mx = (p1[0] + p2[0]) / 2
                if mx > best_mx:
                    best_mx = mx; best_i = i
            p1 = pts[best_i]; p2 = pts[(best_i + 1) % n]
            mid_x = (p1[0] + p2[0]) / 2; mid_y = (p1[1] + p2[1]) / 2
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            seg_len = math.sqrt(dx**2 + dy**2)
            if seg_len > 0:
                nx, ny = dy / seg_len, -dx / seg_len  # outward normal
                return {
                    "x1": p1[0] + nx * offset, "y1": p1[1] + ny * offset,
                    "x2": p2[0] + nx * offset, "y2": p2[1] + ny * offset,
                    "label_x": mid_x + nx * (offset + label_gap),
                    "label_y": mid_y + ny * (offset + label_gap),
                    "constraint": None,
                }
        y = shape_bottom - offset
        return {
            "x1": shape_left, "y1": y, "x2": shape_right, "y2": y,
            "label_x": shape_cx, "label_y": y - label_gap,
            "constraint": None,
        }

    def _calc_dim_leg_a(self, hints, b, offset, label_gap):
        """Leg A (vertical leg) of right triangle: tri_pts[2]→tri_pts[0], offset left."""
        pts = hints.get("tri_pts")
        if pts and len(pts) >= 3:
            p0, p2 = pts[0], pts[2]
            shape_cx = (b["x_min"] + b["x_max"]) / 2
            shape_cy = (b["y_min"] + b["y_max"]) / 2
            res = self._dim_perp_offset(p2, p0, offset, label_gap, shape_cx, shape_cy)
            if res:
                return res
        return None

    def _calc_dim_leg_b(self, hints, b, offset, label_gap):
        """Leg B (base edge) of right triangle: tri_pts[0]→tri_pts[1], offset below."""
        pts = hints.get("tri_pts")
        if pts and len(pts) >= 3:
            p0, p1 = pts[0], pts[1]
            y = min(p0[1], p1[1]) - offset
            mx = (p0[0] + p1[0]) / 2
            return {
                "x1": p0[0], "y1": y, "x2": p1[0], "y2": y,
                "label_x": mx, "label_y": y - label_gap,
                "constraint": None,
            }
        return None

    def _calc_dim_hyp(self, hints, b, offset, label_gap):
        """Hypotenuse of right triangle: tri_pts[1]→tri_pts[2], offset outward away from shape."""
        pts = hints.get("tri_pts")
        if pts and len(pts) >= 3:
            p1, p2 = pts[1], pts[2]
            dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
            seg_len = math.sqrt(dx**2 + dy**2) or 1
            nx, ny = -dy / seg_len, dx / seg_len
            # Point normal away from the right-angle vertex (pts[0])
            p0 = pts[0]
            mid_x = (p1[0] + p2[0]) / 2; mid_y = (p1[1] + p2[1]) / 2
            if nx * (mid_x - p0[0]) + ny * (mid_y - p0[1]) < 0:
                nx, ny = -nx, -ny
            return {
                "x1": p1[0] + nx * offset, "y1": p1[1] + ny * offset,
                "x2": p2[0] + nx * offset, "y2": p2[1] + ny * offset,
                "label_x": mid_x + nx * (offset + label_gap),
                "label_y": mid_y + ny * (offset + label_gap),
                "constraint": None,
            }
        return None

    def _calc_dim_circumference(self, hints, b, offset, label_gap):
        # Handled as arc toggle in _add_standalone_dim_preset
        return None

    def _rotate_geometry_hints(self, angle_rad: float, cx: float, cy: float) -> None:
        """Rotate point-valued geometry_hints after 3D geometric rotation.
        
        This ensures hit-testing coordinates (builtin dim lines, snap anchors)
        match the rotated visual position of artists on screen.
        """
        rp = GeometricRotation.rotate_point
        hints = self.label_manager.geometry_hints
        
        # Rotate individual point hints (2D triangle + prism vertex points)
        point_keys = [
            "tri_foot", "tri_apex", "tri_base_p1", "tri_base_p2",
            # Rectangular prism face corners
            "prism_f_bl", "prism_f_br", "prism_f_tl", "prism_f_tr",
            "prism_b_bl", "prism_b_br", "prism_b_tl", "prism_b_tr",
            # Triangular prism face corners
            "prism_tri_p1", "prism_tri_p2", "prism_tri_p3",
            "prism_tri_b1", "prism_tri_b2", "prism_tri_b3",
        ]
        for key in point_keys:
            pt = hints.get(key)
            if pt and isinstance(pt, (tuple, list)) and len(pt) == 2:
                hints[key] = rp(pt[0], pt[1], angle_rad, cx, cy)
        
        # Rotate height endpoint hints stored as scalars.
        # height_y1 and height_y2 are both anchored at height_x, so rotate each
        # (height_x, yN) pair independently — do not carry the rotated x forward.
        hx = hints.get("height_x")
        if hx is not None:
            hy1 = hints.get("height_y1")
            hy2 = hints.get("height_y2")
            if hy1 is not None:
                rx1, ry1 = rp(hx, hy1, angle_rad, cx, cy)
                hints["height_y1"] = ry1
            else:
                rx1 = None
            if hy2 is not None:
                rx2, ry2 = rp(hx, hy2, angle_rad, cx, cy)
                hints["height_y2"] = ry2
            else:
                rx2 = None
            # height_x: use the average of both rotated x values if both present
            if rx1 is not None and rx2 is not None:
                hints["height_x"] = (rx1 + rx2) / 2
            elif rx1 is not None:
                hints["height_x"] = rx1
            elif rx2 is not None:
                hints["height_x"] = rx2
        
        # Rotate point-list hints (vertex arrays)
        list_keys = ["rect_pts", "para_pts", "trap_pts", "tri_pts", "polygon_pts"]
        for key in list_keys:
            pts = hints.get(key)
            if pts and isinstance(pts, list):
                hints[key] = [rp(p[0], p[1], angle_rad, cx, cy) for p in pts]
        
        # Rotate builtin_dimlines endpoints
        bdl = hints.get("builtin_dimlines")
        if bdl and isinstance(bdl, dict):
            for dl_key, dl in bdl.items():
                for xk, yk in [("x1", "y1"), ("x2", "y2")]:
                    if xk in dl and yk in dl:
                        rx, ry = rp(dl[xk], dl[yk], angle_rad, cx, cy)
                        dl[xk] = rx
                        dl[yk] = ry

    def _flip_geometry_hints(self, flip_h: bool, flip_v: bool,
                             cx: float, cy: float) -> None:
        """Mirror point-valued geometry_hints after a flip operation.

        Applied after _rotate_geometry_hints so hint coordinates always
        match the final visual position of artists on screen.
        """
        if not flip_h and not flip_v:
            return

        def mirror(px, py):
            x = (2 * cx - px) if flip_h else px
            y = (2 * cy - py) if flip_v else py
            return x, y

        hints = self.label_manager.geometry_hints

        point_keys = [
            "tri_foot", "tri_apex", "tri_base_p1", "tri_base_p2",
            # Rectangular prism face corners
            "prism_f_bl", "prism_f_br", "prism_f_tl", "prism_f_tr",
            "prism_b_bl", "prism_b_br", "prism_b_tl", "prism_b_tr",
            # Triangular prism face corners
            "prism_tri_p1", "prism_tri_p2", "prism_tri_p3",
            "prism_tri_b1", "prism_tri_b2", "prism_tri_b3",
        ]
        for key in point_keys:
            pt = hints.get(key)
            if pt and isinstance(pt, (tuple, list)) and len(pt) == 2:
                hints[key] = mirror(pt[0], pt[1])

        # height_x / height_y1 / height_y2 stored as scalars
        hx = hints.get("height_x")
        hy1 = hints.get("height_y1")
        hy2 = hints.get("height_y2")
        if hx is not None and hy1 is not None:
            nx, ny = mirror(hx, hy1)
            hints["height_x"] = nx
            hints["height_y1"] = ny
        if hx is not None and hy2 is not None:
            nx, ny = mirror(hx, hy2)
            hints["height_x"] = nx
            hints["height_y2"] = ny

        list_keys = ["rect_pts", "para_pts", "trap_pts", "tri_pts", "polygon_pts"]
        for key in list_keys:
            pts = hints.get(key)
            if pts and isinstance(pts, list):
                hints[key] = [mirror(p[0], p[1]) for p in pts]

        bdl = hints.get("builtin_dimlines")
        if bdl and isinstance(bdl, dict):
            for dl_key, dl in bdl.items():
                for xk, yk in [("x1", "y1"), ("x2", "y2")]:
                    if xk in dl and yk in dl:
                        dl[xk], dl[yk] = mirror(dl[xk], dl[yk])

    def _draw_builtin_dim_lines(self) -> None:
        """Track built-in dim line endpoints for hit-testing after shape renders them."""
        self._builtin_dim_endpoints = []
        hints = self.label_manager.geometry_hints
        bdl = hints.get("builtin_dimlines")
        if not bdl:
            return
        for key, dl in bdl.items():
            self._builtin_dim_endpoints.append({
                "key": key,
                "p1": (dl["x1"], dl["y1"]),
                "p2": (dl["x2"], dl["y2"])
            })

    def _draw_standalone_labels(self) -> None:
        """Draw freeform labels and dimension lines on standalone shapes."""
        self._standalone_label_bboxes = []
        font_size = self.font_size_var.get() if self.font_size_var is not None else 12
        
        for i, lbl in enumerate(self._standalone_labels):
            is_selected = (i == self._standalone_selected_label)
            color = "#0066cc" if is_selected else "black"
            weight = "bold" if is_selected else "normal"
            edge = color if is_selected else "none"
            
            lbl_txt = self.ax.text(lbl["x"], lbl["y"], lbl["text"],
                        fontsize=font_size, color=color, fontweight=weight,
                        fontfamily=getattr(self, 'font_family', AppConstants.DEFAULT_FONT_FAMILY),
                        ha="center", va="center", zorder=AppConstants.LABEL_ZORDER + 1,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                 edgecolor=edge, alpha=1))
            
            pad_x, pad_y = self._pixels_to_data_pad(3)
            try:
                renderer = self.fig.canvas.get_renderer()
                bb = lbl_txt.get_window_extent(renderer=renderer)
                inv = self.ax.transData.inverted()
                bl = inv.transform([bb.x0, bb.y0])
                tr = inv.transform([bb.x1, bb.y1])
                half_w = (tr[0] - bl[0]) / 2 + pad_x
                half_h = (tr[1] - bl[1]) / 2 + pad_y
            except Exception:
                char_w = font_size * 0.12
                char_h = font_size * 0.2
                half_w = (len(lbl["text"]) * char_w / 2) + pad_x
                half_h = (char_h / 2) + pad_y
            self._standalone_label_bboxes.append((
                lbl["x"] - half_w, lbl["y"] - half_h,
                lbl["x"] + half_w, lbl["y"] + half_h
            ))
        
        # Draw standalone dimension lines
        self._standalone_dim_endpoints = []
        self._standalone_dim_label_bboxes = []
        
        for i, dim in enumerate(self._standalone_dim_lines):
            is_selected = (i == self._standalone_selected_dim)
            color = "#0066cc" if is_selected else "black"
            lw = 1.5 if is_selected else AppConstants.DIMENSION_LINE_WIDTH

            # Live-update preset dim lines to track shape geometry changes.
            # Suppressed during rotation/flip so rotated coords are not overwritten.
            preset_key = dim.get("preset_key")
            suppress_snap = getattr(self, '_suppress_preset_dim_snap', False)
            if preset_key and not dim.get("user_dragged") and not suppress_snap:
                fresh = self._calc_dim_line_endpoints(preset_key)
                if fresh:
                    # Preserve label drag offset relative to old midpoint
                    old_mx = (dim["x1"] + dim["x2"]) / 2
                    old_my = (dim["y1"] + dim["y2"]) / 2
                    lbl_dx = dim.get("label_x", old_mx) - old_mx
                    lbl_dy = dim.get("label_y", old_my) - old_my
                    dim["x1"], dim["y1"] = fresh["x1"], fresh["y1"]
                    dim["x2"], dim["y2"] = fresh["x2"], fresh["y2"]
                    new_mx = (fresh["x1"] + fresh["x2"]) / 2
                    new_my = (fresh["y1"] + fresh["y2"]) / 2
                    dim["label_x"] = new_mx + lbl_dx
                    dim["label_y"] = new_my + lbl_dy

            x1, y1, x2, y2 = dim["x1"], dim["y1"], dim["x2"], dim["y2"]
            
            self.ax.plot([x1, x2], [y1, y2],
                        color=color, linestyle="--", linewidth=lw, zorder=12)
            
            marker_size = 4 if is_selected else 3
            for px, py in [(x1, y1), (x2, y2)]:
                self.ax.plot(px, py, marker="o", markersize=marker_size,
                           color=color, zorder=13)
            
            label_x = dim.get("label_x", (x1 + x2) / 2)
            label_y = dim.get("label_y", (y1 + y2) / 2)
            
            label_color = "#0066cc" if is_selected else "black"
            dim_lbl_txt = self.ax.text(label_x, label_y, dim["text"],
                        fontsize=font_size, color=label_color,
                        fontweight="bold" if is_selected else "normal",
                        fontfamily=getattr(self, 'font_family', AppConstants.DEFAULT_FONT_FAMILY),
                        ha="center", va="center", zorder=AppConstants.LABEL_ZORDER + 1,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                 edgecolor=color if is_selected else "none", alpha=1))
            
            self._standalone_dim_endpoints.append({"p1": (x1, y1), "p2": (x2, y2)})

            # G11: draw right-angle marker and dashed base extension for height dim lines
            pk = dim.get("preset_key", "")
            ra_at = dim.get("right_angle_at")
            if ra_at is None and pk in ("height", "para_height", "trap_height"):
                # Prisms: height is an external side edge, not an altitude — no marker
                # Rectangle: height is always perpendicular by definition — no marker needed
                hints_check = self.label_manager.geometry_hints
                is_prism = "prism_f_bl" in hints_check or "prism_tri_p1" in hints_check
                if not is_prism and "rect_pts" not in hints_check:
                    ra_at = "p1"
            if ra_at in ("p1", "p2"):
                foot = (x1, y1) if ra_at == "p1" else (x2, y2)
                tip  = (x2, y2) if ra_at == "p1" else (x1, y1)
                dx_line = tip[0] - foot[0]
                dy_line = tip[1] - foot[1]
                line_len = math.sqrt(dx_line**2 + dy_line**2)
                if line_len > 0.001:
                    u_line = (dx_line / line_len, dy_line / line_len)
                    # Determine correct perpendicular direction using shape geometry hints
                    hints = self.label_manager.geometry_hints
                    h_val = None
                    base_p0 = base_p1_pt = None

                    if "tri_foot" in hints and "tri_base_p1" in hints:
                        # Triangle: use stored base edge
                        base_p0 = hints["tri_base_p1"]
                        base_p1_pt = hints["tri_base_p2"]
                    elif "para_pts" in hints and len(hints["para_pts"]) >= 4:
                        pts_h = hints["para_pts"]
                        base_p0, base_p1_pt = pts_h[0], pts_h[1]
                    elif "trap_pts" in hints and len(hints["trap_pts"]) >= 4:
                        pts_h = hints["trap_pts"]
                        base_p0, base_p1_pt = pts_h[0], pts_h[1]

                    if base_p0 and base_p1_pt:
                        bdx = base_p1_pt[0] - base_p0[0]
                        bdy = base_p1_pt[1] - base_p0[1]
                        base_len = math.sqrt(bdx**2 + bdy**2)
                        if base_len > 0.001:
                            u_base = (bdx / base_len, bdy / base_len)
                            u_perp_candidate = (-u_base[1], u_base[0])
                            # h_val: signed distance from base to tip (positive = interior side)
                            vt = (tip[0] - base_p0[0], tip[1] - base_p0[1])
                            h_val = vt[0] * u_perp_candidate[0] + vt[1] * u_perp_candidate[1]
                            # t: position of foot along base
                            t_foot = (foot[0] - base_p0[0]) * u_base[0] + (foot[1] - base_p0[1]) * u_base[1]

                            # Dashed base extension when foot is outside base segment
                            if t_foot < -0.001:
                                self.ax.plot([base_p0[0], foot[0]], [base_p0[1], foot[1]],
                                            color=color, linestyle="--",
                                            linewidth=AppConstants.DIMENSION_LINE_WIDTH, zorder=12)
                            elif t_foot > base_len + 0.001:
                                self.ax.plot([base_p1_pt[0], foot[0]], [base_p1_pt[1], foot[1]],
                                            color=color, linestyle="--",
                                            linewidth=AppConstants.DIMENSION_LINE_WIDTH, zorder=12)

                            # Orient u_perp toward shape interior
                            if h_val is not None:
                                u_perp = u_perp_candidate if h_val > 0 else (-u_perp_candidate[0], -u_perp_candidate[1])
                                # base direction: point toward nearest base endpoint from foot
                                m_base_dir = u_base if t_foot < base_len / 2 else (-u_base[0], -u_base[1])
                            else:
                                u_perp = (-u_line[1], u_line[0])
                                m_base_dir = u_line
                        else:
                            u_perp = (-u_line[1], u_line[0])
                            m_base_dir = u_line
                    else:
                        u_perp = (-u_line[1], u_line[0])
                        m_base_dir = u_line

                    marker_size = max(0.15, min(0.40, line_len * 0.10))
                    ctx_tmp = self.plot_controller.create_drawing_context(
                        aspect_ratio=self.scale_manager.var("aspect").get()
                    )
                    DrawingUtilities.draw_right_angle_marker(
                        ctx_tmp, foot, m_base_dir, u_perp, size=marker_size
                    )
            
            pad_x, pad_y = self._pixels_to_data_pad(3)
            try:
                renderer = self.fig.canvas.get_renderer()
                bb = dim_lbl_txt.get_window_extent(renderer=renderer)
                inv = self.ax.transData.inverted()
                bl = inv.transform([bb.x0, bb.y0])
                tr = inv.transform([bb.x1, bb.y1])
                half_w = (tr[0] - bl[0]) / 2 + pad_x
                half_h = (tr[1] - bl[1]) / 2 + pad_y
            except Exception:
                char_w = font_size * 0.08
                char_h = font_size * 0.15
                half_w = len(dim["text"]) * char_w / 2 + pad_x
                half_h = char_h / 2 + pad_y
            self._standalone_dim_label_bboxes.append((
                label_x - half_w, label_y - half_h,
                label_x + half_w, label_y + half_h
            ))
        
    def generate_plot(self) -> None:
        """Top-level draw entry point. Dispatches to composite or standalone path."""
        try:
            shape = self.shape_var.get()
            if not shape:
                return
            # size before we read _axes_pixel_aspect.
            self.root.update_idletasks()
            self.plot_controller.clear()
            self.plot_controller.setup_axes()
            self._preserve_toggle_labels()
            if self._is_composite_shape(shape) and self.composite_transfer is not None:
                self._generate_composite_plot(shape)
            else:
                self._generate_standalone_plot(shape)
        except Exception as e:
            logger.exception("Unhandled error in generate_plot()")
            try:
                self.plot_controller.clear()
                self.plot_controller.setup_axes()
                self.plot_controller.draw_error(f"Error updating plot\n{type(e).__name__}: {e}")
                self.plot_controller.refresh()
            except Exception:
                pass

    def _preserve_toggle_labels(self) -> None:
        """Persist toggle-based label values across a plot clear/redraw cycle."""
        preserved = {}
        for lbl_key, _ in self.TOGGLE_LABEL_KEYS:
            if lbl_key in self.label_manager.label_texts:
                preserved[lbl_key] = (
                    self.label_manager.label_texts[lbl_key],
                    self.label_manager.label_visibility.get(lbl_key, True),
                )
        self.label_manager.clear_label_texts()
        for key, (text, vis) in preserved.items():
            if text:
                self.label_manager.set_label_text(key, text, vis)

    def _generate_composite_plot(self, shape: str) -> None:
        """Render a composite (multi-shape) canvas."""
        selected = self.composite_transfer.get_selected_shapes()
        if not selected:
            pixel_aspect = self._axes_pixel_aspect
            page_h = 20.0
            page_w = page_h * pixel_aspect
            self.ax.set_xlim(0, page_w)
            self.ax.set_ylim(0, page_h)
            self.ax.text(
                0.5, 0.5,
                "Add shapes from the Available list\nusing the → button",
                ha="center", va="center", fontsize=14, color="gray",
                transform=self.ax.transAxes,
            )
            self.plot_controller.refresh()
            return
        self._draw_composite_shapes(selected)
        self.plot_controller.refresh()

    def _generate_standalone_plot(self, shape: str) -> None:
        """Render a single standalone shape with the full transform pipeline."""
        ctx = self.plot_controller.create_drawing_context(
            aspect_ratio=self.scale_manager.var("aspect").get()
        )
        ctx.show_hashmarks = self.show_hashmarks_var.get()
        self.label_manager.builtin_selected = self._builtin_selected
        transform = self._create_transform_state()
        params = self._collect_shape_params()
        self._apply_transform_pipeline(ctx, shape, transform, params)

        # Draw smart hashmarks if enabled
        if (self.show_hashmarks_var.get()
                and shape in SmartGeometryEngine.POLYGON_SHAPES):
            self._draw_smart_hashmarks_overlay(ctx)

        # Store tight shape bounds before view scale expands them
        self._shape_bounds = {
            "x_min": self.ax.get_xlim()[0],
            "x_max": self.ax.get_xlim()[1],
            "y_min": self.ax.get_ylim()[0],
            "y_max": self.ax.get_ylim()[1],
        }
        self._apply_view_scale()
        self._draw_builtin_dim_lines()
        self._draw_standalone_labels()
        self.plot_controller.refresh()

    def _apply_transform_pipeline(self, ctx: "DrawingContext", shape: str,
                                   transform: "TransformState",
                                   params: dict) -> None:
        """Draw shape at canonical orientation then apply rotate → flip artist pipeline.

        For shapes in GeometricRotation.ALL_GEOMETRIC_SHAPES the unified pipeline is:
          1. Draw canonical (base_side=0, no flip) so vertex coords are predictable.
          2. Rotate all artists around the canonical center by the step angle.
          3. Flip all artists in the post-rotation visual space.
        For all other shapes the transform is passed directly to the drawer.

        Updates self._canonical_pre_rotation_center, _pre_rotation_center, and
        _flip_center so that _rotate_with_annotations and _flip_with_annotations
        use the correct pivot points.
        """
        actual_base_side = transform.base_side
        actual_flip_h = transform.flip_h
        actual_flip_v = transform.flip_v
        is_unified = shape in GeometricRotation.ALL_GEOMETRIC_SHAPES
        if is_unified:
            transform = TransformState(flip_h=False, flip_v=False, base_side=0)

        error = self.plot_controller.draw_shape(shape, ctx, transform, params)
        if error:
            self.plot_controller.draw_error(error)

        # Record canonical center from the tight limits set_limits() placed.
        _xlim = self.ax.get_xlim()
        _ylim = self.ax.get_ylim()
        canonical_center = ((_xlim[0] + _xlim[1]) / 2, (_ylim[0] + _ylim[1]) / 2)
        self._canonical_pre_rotation_center = canonical_center
        self._pre_rotation_center = canonical_center

        if not (is_unified and not error):
            return

        config = ShapeConfigProvider.get(shape)
        num_sides = config.num_sides if config.num_sides > 0 else 4

        # Step 2: rotate
        if actual_base_side != 0:
            angle_deg = -(360.0 / num_sides) * actual_base_side
            cx, cy = self._pre_rotation_center
            GeometricRotation.rotate_axes_artists(self.ax, angle_deg, cx, cy)
            GeometricRotation.recalculate_limits(self.ax)
            self._rotate_geometry_hints(math.radians(angle_deg), cx, cy)

        # Step 3: flip (in post-rotation visual space)
        if actual_flip_h or actual_flip_v:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            cx = (xlim[0] + xlim[1]) / 2
            cy = (ylim[0] + ylim[1]) / 2
            GeometricRotation.flip_axes_artists(self.ax, actual_flip_h, actual_flip_v, cx, cy)
            GeometricRotation.recalculate_limits(self.ax)
            self._flip_geometry_hints(actual_flip_h, actual_flip_v, cx, cy)
            self._flip_center = (cx, cy)
        else:
            self._flip_center = self._canonical_pre_rotation_center

    def _save_figure(self, path: str | BytesIO, fmt: str = "png") -> None:
        """Save the current figure to a file. Central method for all export paths."""
        # pad_inches=-0.1 is intentional: matplotlib's tight-layout leaves a small
        # whitespace ring even at pad_inches=0. A slight negative value crops that
        # ring so the exported image is flush to the shape bounds.
        self.fig.savefig(path, format=fmt, dpi=150, bbox_inches="tight", facecolor="#ffffff", pad_inches=-0.1)

    def save_image(self) -> None:
        if not self.shape_var.get():
            self._ui_info("No Shape", "Please select a shape before saving.")
            return
        
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("SVG Vector", "*.svg")]
        )
        if path:
            try:
                fmt = "svg" if path.lower().endswith(".svg") else "png"
                self._save_figure(path, fmt)
                self._ui_info("Saved", f"Image saved to {path}")
            except Exception as e:
                self._ui_error("Save Failed", "Failed to save image.", exc=e)

    def copy_to_clipboard(self) -> None:
        """Copy current plot to system clipboard as an image."""
        if not self.shape_var.get():
            self._ui_info("No Shape", "Please select a shape before copying.")
            return

        try:
            # Save figure to a bytes buffer (used by Windows/pywin32 path)
            buf = BytesIO()
            self._save_figure(buf, fmt="png")
            buf.seek(0)
            
            system = platform.system()
            
            if system == "Windows":
                # Prefer pywin32 if present; otherwise fall back to PowerShell/.NET (no pip installs)
                try:
                    import win32clipboard  # type: ignore
                    from PIL import Image  # only required for the pywin32 path

                    image = Image.open(buf)
                    output = BytesIO()
                    image.convert("RGB").save(output, "BMP")
                    data = output.getvalue()[14:]
                    output.close()
                    
                    win32clipboard.OpenClipboard()
                    win32clipboard.EmptyClipboard()
                    win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
                    win32clipboard.CloseClipboard()

                except ImportError:
                    import tempfile

                    fd, temp_path = tempfile.mkstemp(suffix=".png")
                    os.close(fd)

                    try:
                        self._save_figure(temp_path, fmt="png")
                        ps_path = temp_path.replace("'", "''")

                        ps = (
                            "Add-Type -AssemblyName System.Windows.Forms; "
                            "Add-Type -AssemblyName System.Drawing; "
                            f"$img=[System.Drawing.Image]::FromFile('{ps_path}'); "
                            "[System.Windows.Forms.Clipboard]::SetImage($img); "
                            "$img.Dispose();"
                        )

                        subprocess.run(
                            ["powershell", "-NoProfile", "-STA", "-Command", ps],
                            check=True,
                            capture_output=True,
                            text=True
                        )
                    except subprocess.CalledProcessError as ps_err:
                        stderr_msg = ps_err.stderr.strip() if ps_err.stderr else "no details"
                        messagebox.showwarning(
                            "Clipboard Error",
                            f"PowerShell failed to copy image to clipboard.\n\n{stderr_msg}"
                        )
                        return
                    finally:
                        try:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                        except Exception:
                            pass
            
            elif system == "Darwin":  # macOS
                # Write to temp file and use AppleScript to set clipboard to image
                import tempfile
                fd, temp_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                try:
                    self._save_figure(temp_path, fmt="png")
                    script = f'set the clipboard to (read (POSIX file "{temp_path}") as «class PNGf»)'
                    subprocess.run(["osascript", "-e", script], check=True)
                finally:
                    try:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    except Exception:
                        pass
            
            else:
                self._ui_info("Clipboard", f"Clipboard not yet supported on {system}. Please use 'Save Image'.")
                return

            self._ui_info("Success", "Image copied to clipboard.")
        except subprocess.CalledProcessError as e:
            self._ui_error("Error", "Failed to copy image to clipboard.", exc=e)
        except Exception as e:
            self._ui_error("Error", f"Failed to copy image: {e}", exc=e)
    
    def _draw_smart_hashmarks_overlay(self, ctx: "DrawingContext") -> None:
        """Extract polygon points from geometry hints and draw smart hashmarks."""
        hints = self.label_manager.geometry_hints
        shape = self.shape_var.get()
        points: list | None = None

        if shape == "Triangle":
            # Triangle hashmarks are drawn by _draw_triangle_common (via equal_sides + ctx.show_hashmarks).
            # Custom/Scalene/Right have no equal sides so smart detection runs here for those.
            base_p1 = hints.get("tri_base_p1")
            base_p2 = hints.get("tri_base_p2")
            apex = hints.get("tri_apex")
            tri_type = self.triangle_type_var.get()
            # Only run smart overlay for types without built-in equal_sides marks
            if base_p1 and base_p2 and apex and tri_type not in ("Isosceles", "Equilateral"):
                points = [base_p1, base_p2, apex]
        elif shape == "Polygon":
            # Polygon hashmarks are drawn by PolygonDrawer (gated on ctx.show_hashmarks).
            pass
        elif shape == "Rectangle":
            pts = hints.get("rect_pts")
            if pts:
                points = pts
            else:
                # Fall back to axis limits (rectangle is axis-aligned)
                b = self._shape_bounds
                if b:
                    points = [
                        (b["x_min"], b["y_min"]), (b["x_max"], b["y_min"]),
                        (b["x_max"], b["y_max"]), (b["x_min"], b["y_max"])
                    ]
        elif shape == "Square":
            # Square hashmarks are drawn by SquareDrawer (gated on ctx.show_hashmarks).
            # Skip overlay to avoid double-drawing.
            pass
        elif shape == "Parallelogram":
            points = hints.get("para_pts")
        elif shape == "Trapezoid":
            points = hints.get("trap_pts")

        if points and len(points) >= 3:
            SmartGeometryEngine.draw_smart_hashmarks(ctx, points)

    def _pack_right_panel(self, show_help: bool = True) -> None:
        """Pack right panel (col_tools) widgets without triggering recursive layout events."""
        # Skip entirely during state restoration to prevent flash
        if self.history_manager.is_restoring:
            return

        # Destroy any dynamically-created children that are not persistent widgets.
        # scale_frame is persistent and managed via pack/pack_forget below.
        persistent = (self.tools_header, self.undo_redo_frame,
                      self.clear_workspace_btn, self.mode_help_label,
                      self.scale_frame)
        for child in list(self.col_tools.winfo_children()):
            if child not in persistent:
                child.destroy()

        # Show the tools column
        self.col_tools.grid()

        # Pack persistent widgets in order
        self.tools_header.pack(side=tk.TOP)
        self.undo_redo_frame.pack(side=tk.TOP, fill=tk.X, pady=1)
        self.clear_workspace_btn.pack(side=tk.TOP, fill=tk.X, pady=1)

        # Scale slider: show for standalone shapes, hide for composite
        if not self._is_composite_shape():
            self.scale_frame.pack(side=tk.TOP, fill=tk.X, pady=(6, 0))
        else:
            self.scale_frame.pack_forget()

        if show_help and self.mode_help_label.cget("text"):
            self.mode_help_label.pack(side=tk.TOP, fill=tk.X, pady=1)
        else:
            self.mode_help_label.pack_forget()
    
    def _bind_shortcuts(self) -> None:
        """Centralized keyboard shortcut binding for all platforms."""
        # Basic transforms
        for key in ['h', 'H']:
            self.root.bind(f'<{key}>', lambda e: self._on_flip_shortcut('h', e))
        for key in ['v', 'V']:
            self.root.bind(f'<{key}>', lambda e: self._on_flip_shortcut('v', e))
        for key in ['r', 'R']:
            self.root.bind(f'<{key}>', lambda e: self._rotate_base(1))
        for key in ['l', 'L']:
            self.root.bind(f'<{key}>', lambda e: self._rotate_base(-1))
        
        # Delete/Backspace to remove selected shapes or annotations
        self.root.bind_all('<Delete>', lambda e: self._on_delete_shortcut() or "break")
        self.root.bind_all('<BackSpace>', lambda e: self._on_delete_shortcut() or "break")
        
        # Platform-specific modifiers
        mod = "Command" if platform.system() == "Darwin" else "Control"
        
        actions = [
            ('z', self._undo_action),
            ('y', self._redo_action),
            ('s', lambda e=None: self.save_image()),
            ('c', lambda e=None: self.copy_to_clipboard())
        ]
        
        for key, func in actions:
            self.root.bind_all(f'<{mod}-{key}>', lambda e, f=func: f() or "break")
            self.root.bind_all(f'<{mod}-{key.upper()}>', lambda e, f=func: f() or "break")

    def _apply_view_scale(self) -> None:
        """Auto-scale shape to fill canvas with scale + pan applied.

        scale=1.0 (slider far right) = shape fills white area fully.
        scale<1.0 = shape shrinks, revealing white space around it.
        Pan offset shifts the view center in data coordinates.
        No set_aspect() is used — pixel ratio is enforced via xlim/ylim math.
        """
        ax = self.plot_controller.ax
        ax.set_autoscale_on(False)

        bounds = getattr(self, '_shape_bounds', None)
        if bounds:
            shape_width    = bounds["x_max"] - bounds["x_min"]
            shape_height   = bounds["y_max"] - bounds["y_min"]
            shape_center_x = (bounds["x_min"] + bounds["x_max"]) / 2
            shape_center_y = (bounds["y_min"] + bounds["y_max"]) / 2
        else:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            shape_width    = xlim[1] - xlim[0]
            shape_height   = ylim[1] - ylim[0]
            shape_center_x = (xlim[0] + xlim[1]) / 2
            shape_center_y = (ylim[0] + ylim[1]) / 2

        # Use actual canvas pixel ratio so shapes are undistorted
        pixel_aspect = getattr(self, '_axes_pixel_aspect', 4.0/3.0)

        # At scale=1.0 the shape fills the white area (12% margin each side)
        margin_x = shape_width  * 0.12
        margin_y = shape_height * 0.12
        content_width  = shape_width  + 2 * margin_x
        content_height = shape_height + 2 * margin_y

        content_aspect = content_width / content_height if content_height > 0 else pixel_aspect
        if content_aspect > pixel_aspect:
            base_width  = content_width
            base_height = base_width / pixel_aspect
        else:
            base_height = content_height
            base_width  = base_height * pixel_aspect

        # Scale: slider 1.0=fill, lower values zoom out. Clamp to valid range.
        _raw = self.scale_manager.var("view_scale").get()
        scale = max(0.25, min(1.0, _raw))
        final_width  = base_width  / scale
        final_height = base_height / scale

        # Apply pan offset (data units), clamped so shape can't be dragged
        # fully off-screen (center must stay within ±half the base extent)
        pan_x, pan_y = getattr(self, '_shape_pan_offset', (0.0, 0.0))
        max_pan_x = final_width  * 0.35
        max_pan_y = final_height * 0.35
        pan_x = max(-max_pan_x, min(max_pan_x, pan_x))
        pan_y = max(-max_pan_y, min(max_pan_y, pan_y))
        self._shape_pan_offset = (pan_x, pan_y)

        cx = shape_center_x + pan_x
        cy = shape_center_y + pan_y

        ax.set_xlim(cx - final_width  / 2, cx + final_width  / 2)
        ax.set_ylim(cy - final_height / 2, cy + final_height / 2)
    
    def _on_flip_shortcut(self, axis: str, event=None) -> None:
        """Handle 'h'/'v' key for horizontal or vertical flip."""
        if not self.shape_var.get() or isinstance(self.root.focus_get(), tk.Entry):
            return
        if self._is_composite_shape() and self._composite_selected:
            if axis == 'h':
                self._composite_flip_h()
            else:
                self._composite_flip_v()
            return
        config = ShapeConfigProvider.get(self.shape_var.get())
        if config.has_feature(ShapeFeature.FLIP):
            self._flip_with_annotations(axis)



