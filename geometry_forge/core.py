"""geometry_forge/core.py — Framework-agnostic drawing engine.

No Tkinter. No Streamlit. Pure Python + matplotlib.
All state stored as plain Python attributes — session_state in streamlit_app.py
wraps this object and persists it across reruns.
"""
from __future__ import annotations

import math
import logging
from io import BytesIO
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
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
from .controllers import TransformController, PlotController, HistoryManager, ScaleManager

logger = logging.getLogger(__name__)


class GeometryCore:
    """Framework-agnostic geometry drawing engine.

    Instantiated once and stored in st.session_state. Every Streamlit rerun
    reads state from self.* attrs, calls generate_figure(), and displays the
    returned Figure with st.pyplot().
    """

    # Ordered (label_key, snapshot_key) pairs for toggle-based labels.
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

    def __init__(self) -> None:
        # ── Shape catalogue ───────────────────────────────────────────────────
        self.shape_data: dict[str, list[str]] = {
            "2D Figures": [
                "Rectangle", "Square", "Triangle", "Circle",
                "Parallelogram", "Trapezoid", "Polygon",
            ],
            "3D Solids": [
                "Sphere", "Hemisphere", "Cylinder", "Cone",
                "Rectangular Prism", "Triangular Prism",
            ],
            "Angles & Lines": [
                "Angle (Adjustable)",
                "Parallel Lines & Transversal",
                "Complementary Angles", "Supplementary Angles",
                "Vertical Angles", "Line Segment",
            ],
            "Composite Figures": ["2D Composite", "3D Composite"],
        }

        # ── Shape selection ───────────────────────────────────────────────────
        self.category: str | None = None
        self.shape_name: str | None = None

        # ── Appearance ────────────────────────────────────────────────────────
        self.font_size: int = AppConstants.DEFAULT_FONT_SIZE
        self.line_width: int = AppConstants.DEFAULT_LINE_WIDTH
        self.font_family: str = AppConstants.DEFAULT_FONT_FAMILY

        # ── Shape sub-type ────────────────────────────────────────────────────
        self.triangle_type: str = "Custom"
        self.polygon_type: str = PolygonType.PENTAGON.value
        self.dimension_mode: str = "Default"
        self.show_hashmarks: bool = False

        # ── Input field values (label → text string) ──────────────────────────
        self.params: dict[str, str] = {}

        # ── Toggle label state (key → {"text": str, "show": bool}) ───────────
        # Managed via label_manager; this is kept for snapshot/restore parity.

        # ── Standalone annotations ────────────────────────────────────────────
        self.standalone_labels: list[dict] = []       # [{"text", "x", "y"}]
        self.standalone_dim_lines: list[dict] = []    # StandaloneDimLine dicts

        # ── Composite state ───────────────────────────────────────────────────
        self.composite_shapes: list[str] = []
        self.composite_positions: dict[int, tuple[float, float]] = {}
        self.composite_transforms: dict[int, dict] = {}
        self.composite_labels: list[dict] = []
        self.composite_dim_lines_list: list[dict] = []

        # ── Internal draw state ───────────────────────────────────────────────
        self._shape_bounds: dict | None = None
        self._composite_bboxes: list = []
        self._composite_snap_anchors: list = []
        self._composite_geometry_hints: dict[int, dict] = {}
        self._composite_view_limits: tuple | None = None
        self._composite_dim_endpoints: list = []
        self._composite_dim_label_bboxes: list = []
        self._composite_label_bboxes: list = []
        self._suppress_preset_dim_snap: bool = False
        self._axes_pixel_aspect: float = AppConstants.PAPER_ASPECT_RATIO
        self._canonical_pre_rotation_center: tuple | None = None
        self._pre_rotation_center: tuple | None = None
        self._flip_center: tuple | None = None
        self._shape_pan_offset: tuple[float, float] = (0.0, 0.0)
        self._standalone_label_bboxes: list = []
        self._standalone_dim_endpoints: list = []
        self._standalone_dim_label_bboxes: list = []
        self._builtin_dim_endpoints: list = []

        # ── Sub-controllers ───────────────────────────────────────────────────
        self.label_manager = LabelManager()
        self.history_manager = HistoryManager()
        self.transform_controller = TransformController(on_change_callback=lambda: None)
        self.scale_manager = ScaleManager()

        # ── Matplotlib figure ─────────────────────────────────────────────────
        self.fig = Figure(figsize=(9.7, 7.27), facecolor="white")
        self.ax = self.fig.add_subplot(111)
        self.plot_controller = PlotController(
            fig=self.fig,
            ax=self.ax,
            label_manager=self.label_manager,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def get_categories(self) -> list[str]:
        return list(self.shape_data.keys())

    def get_shapes(self, category: str) -> list[str]:
        return self.shape_data.get(category, [])

    def generate_figure(self) -> Figure:
        """Top-level draw entry point. Returns the matplotlib Figure."""
        try:
            shape = self.shape_name
            if not shape:
                self.plot_controller.clear()
                self.plot_controller.setup_axes()
                self.ax.text(0.5, 0.5, "Select a shape from the sidebar",
                             ha="center", va="center", fontsize=14, color="gray",
                             transform=self.ax.transAxes)
                return self.fig

            # Sync plot_controller appearance settings
            self.plot_controller.set_font_size(self.font_size)
            self.plot_controller.set_line_width(self.line_width)
            self.plot_controller.set_font_family(self.font_family)

            self.plot_controller.clear()
            self.plot_controller.setup_axes()
            self._preserve_toggle_labels()

            if self._is_composite_shape(shape):
                self._generate_composite_plot(shape)
            else:
                self._generate_standalone_plot(shape)

        except Exception as e:
            logger.exception("Unhandled error in generate_figure()")
            try:
                self.plot_controller.clear()
                self.plot_controller.setup_axes()
                self.plot_controller.draw_error(
                    f"Error updating plot\n{type(e).__name__}: {e}"
                )
            except Exception:
                pass

        return self.fig

    def get_figure_bytes(self, fmt: str = "png") -> bytes:
        """Return the current figure as bytes in the given format."""
        buf = BytesIO()
        self.fig.savefig(buf, format=fmt, dpi=150,
                         bbox_inches="tight", facecolor="#ffffff", pad_inches=-0.1)
        buf.seek(0)
        return buf.read()

    def reset_transforms(self) -> None:
        self.transform_controller.reset()

    def reset_all(self) -> None:
        """Reset all state to defaults."""
        self.category = None
        self.shape_name = None
        self.font_size = AppConstants.DEFAULT_FONT_SIZE
        self.line_width = AppConstants.DEFAULT_LINE_WIDTH
        self.font_family = AppConstants.DEFAULT_FONT_FAMILY
        self.triangle_type = "Custom"
        self.polygon_type = PolygonType.PENTAGON.value
        self.dimension_mode = "Default"
        self.show_hashmarks = False
        self.params = {}
        self.standalone_labels = []
        self.standalone_dim_lines = []
        self.composite_shapes = []
        self.composite_positions = {}
        self.composite_transforms = {}
        self.composite_labels = []
        self.composite_dim_lines_list = []
        self._shape_pan_offset = (0.0, 0.0)
        self.transform_controller.reset()
        self.scale_manager = ScaleManager()
        self.label_manager = LabelManager()
        self.label_manager.font_family = self.font_family

    # ── State snapshot / restore (undo/redo) ──────────────────────────────────

    def _build_state_snapshot(self) -> dict:
        is_composite = self._is_composite_shape(self.shape_name)
        state = {
            "cat": self.category,
            "shape": self.shape_name,
            "font_size": self.font_size,
            "line_width": self.line_width,
            "font_family": self.font_family,
            "dim_mode": self.dimension_mode,
            "tri_type": self.triangle_type,
            "poly_type": self.polygon_type,
            "show_hashmarks": self.show_hashmarks,
            "view_pan": list(self._shape_pan_offset),
            "composite_shapes": list(self.composite_shapes) if is_composite else [],
            "composite_positions": dict(self.composite_positions) if is_composite else {},
            "composite_transforms": {k: dict(v) for k, v in self.composite_transforms.items()} if is_composite else {},
            "composite_labels": [dict(lbl) for lbl in self.composite_labels] if is_composite else [],
            "composite_dim_lines": [dict(d) for d in self.composite_dim_lines_list] if is_composite else [],
            "standalone_labels": [dict(lbl) for lbl in self.standalone_labels] if not is_composite else [],
            "standalone_dim_lines": [dict(d) for d in self.standalone_dim_lines] if not is_composite else [],
            **{
                f"{st_key}_text": self.label_manager.label_texts.get(lbl_key, "")
                for lbl_key, st_key in self.TOGGLE_LABEL_KEYS
            },
            **{
                f"{st_key}_vis": self.label_manager.label_visibility.get(lbl_key, False)
                for lbl_key, st_key in self.TOGGLE_LABEL_KEYS
            },
            "entries": dict(self.params),
        }
        state.update(self.scale_manager.get_state())
        state.update({"transforms": {
            "h": self.transform_controller.flip_h,
            "v": self.transform_controller.flip_v,
            "side": self.transform_controller.base_side,
        }})
        state.update(self.label_manager.get_state())
        return state

    def _apply_state(self, state: dict) -> None:
        """Restore all state from a snapshot dict (used for undo/redo)."""
        if not state:
            return
        with self.history_manager.restoring():
            self.category = state.get("cat", self.category)
            self.shape_name = state.get("shape", self.shape_name)
            self.font_size = state.get("font_size", self.font_size)
            self.line_width = state.get("line_width", self.line_width)
            self.font_family = state.get("font_family", self.font_family)
            self.dimension_mode = state.get("dim_mode", self.dimension_mode)
            self.triangle_type = state.get("tri_type", self.triangle_type)
            self.polygon_type = state.get("poly_type", self.polygon_type)
            self.show_hashmarks = state.get("show_hashmarks", self.show_hashmarks)
            pan = state.get("view_pan", [0.0, 0.0])
            self._shape_pan_offset = (float(pan[0]), float(pan[1]))
            self.params = dict(state.get("entries", {}))

            new_shape = state.get("shape", "")
            if self._is_composite_shape(new_shape):
                self.composite_shapes = list(state.get("composite_shapes", []))
                self.composite_positions = {
                    int(k): tuple(v) for k, v in state.get("composite_positions", {}).items()
                }
                self.composite_transforms = {
                    int(k): dict(v) for k, v in state.get("composite_transforms", {}).items()
                }
                self.composite_labels = [dict(lbl) for lbl in state.get("composite_labels", [])]
                self.composite_dim_lines_list = [dict(d) for d in state.get("composite_dim_lines", [])]
            else:
                self.standalone_labels = [dict(lbl) for lbl in state.get("standalone_labels", [])]
                self.standalone_dim_lines = [dict(d) for d in state.get("standalone_dim_lines", [])]
                for lbl_key, st_key in self.TOGGLE_LABEL_KEYS:
                    t = state.get(f"{st_key}_text", "")
                    v = state.get(f"{st_key}_vis", False)
                    if t:
                        self.label_manager.set_label_text(lbl_key, t, v)
                    else:
                        self.label_manager.label_texts.pop(lbl_key, None)
                        self.label_manager.label_visibility.pop(lbl_key, None)

            self.scale_manager.set_state(state)
            self.transform_controller.set_state(state)
            self.label_manager.set_state(state)

        self._suppress_preset_dim_snap = True
        try:
            self.generate_figure()
        finally:
            self._suppress_preset_dim_snap = False

    # ── Private helpers ───────────────────────────────────────────────────────

    def _is_composite_shape(self, shape: str | None = None) -> bool:
        if shape is None:
            shape = self.shape_name
        return shape in ("2D Composite", "3D Composite")

    def _create_transform_state(self) -> TransformState:
        return self.transform_controller.get_state()

    def _collect_shape_params(self) -> dict[str, Any]:
        """Collect shape parameters from self.params with numeric conversion."""
        result: dict[str, Any] = {}
        for key, text in self.params.items():
            text = text.strip() if isinstance(text, str) else str(text)
            result[key] = text
            num_key = f"{key}_num"
            if text:
                try:
                    result[num_key] = float(text)
                except (ValueError, TypeError):
                    result[num_key] = None
            else:
                result[num_key] = None

        result["triangle_type"] = self.triangle_type
        result["polygon_type"] = self.polygon_type
        result["dimension_mode"] = self.dimension_mode
        result["peak_offset"] = self.scale_manager.get("peak_offset")
        result["parallelogram_slope"] = self.scale_manager.get("slope")
        return result

    def _pixels_to_data_pad(self, px: float) -> tuple[float, float]:
        """Convert pixel size to data-unit padding based on current axis state."""
        fig_w = self.fig.get_figwidth() * self.fig.dpi
        fig_h = self.fig.get_figheight() * self.fig.dpi
        ax_pos = self.ax.get_position()
        ax_px_w = fig_w * ax_pos.width
        ax_px_h = fig_h * ax_pos.height
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        data_w = xlim[1] - xlim[0]
        data_h = ylim[1] - ylim[0]
        pad_x = px * data_w / ax_px_w if ax_px_w > 0 else 0.1
        pad_y = px * data_h / ax_px_h if ax_px_h > 0 else 0.1
        return pad_x, pad_y

    # ── Plot generation ───────────────────────────────────────────────────────

    def _preserve_toggle_labels(self) -> None:
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
        selected = self.composite_shapes
        if not selected:
            pixel_aspect = self._axes_pixel_aspect
            page_h = 20.0
            page_w = page_h * pixel_aspect
            free_dims = [d for d in self.composite_dim_lines_list
                         if d.get("shape_owner") is None]
            if free_dims:
                all_x = [d["x1"] for d in free_dims] + [d["x2"] for d in free_dims]
                all_y = [d["y1"] for d in free_dims] + [d["y2"] for d in free_dims]
                pad = max(page_w, page_h) * 0.1
                self.ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
                self.ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
                for i, dim in enumerate(self.composite_dim_lines_list):
                    if dim.get("shape_owner") is not None:
                        continue
                    color = "black"
                    lw = AppConstants.DIMENSION_LINE_WIDTH
                    x1, y1, x2, y2 = dim["x1"], dim["y1"], dim["x2"], dim["y2"]
                    self.ax.plot([x1, x2], [y1, y2],
                                 color=color, linestyle="--", linewidth=lw, zorder=12)
                    for px, py in [(x1, y1), (x2, y2)]:
                        self.ax.plot(px, py, marker="o", markersize=3,
                                     color=color, zorder=13)
                    lx = dim.get("label_x", (x1 + x2) / 2)
                    ly = dim.get("label_y", (y1 + y2) / 2)
                    self.ax.text(lx, ly, dim["text"], fontsize=self.font_size,
                                 color=color, fontfamily=self.font_family,
                                 ha="center", va="center",
                                 zorder=AppConstants.LABEL_ZORDER + 1,
                                 bbox=dict(boxstyle="round,pad=0.2",
                                           facecolor="white", edgecolor="none", alpha=1))
            else:
                self.ax.set_xlim(0, page_w)
                self.ax.set_ylim(0, page_h)
                self.ax.text(0.5, 0.5, "Add shapes from the Available list",
                             ha="center", va="center", fontsize=14, color="gray",
                             transform=self.ax.transAxes)
            return
        self._draw_composite_shapes(selected)

    def _generate_standalone_plot(self, shape: str) -> None:
        ctx = self.plot_controller.create_drawing_context(
            aspect_ratio=self.scale_manager.get("aspect")
        )
        ctx.show_hashmarks = self.show_hashmarks
        self.label_manager.builtin_selected = None  # no canvas selection in Streamlit
        transform = self._create_transform_state()
        params = self._collect_shape_params()
        self._apply_transform_pipeline(ctx, shape, transform, params)

        if self.show_hashmarks and shape in SmartGeometryEngine.POLYGON_SHAPES:
            self._draw_smart_hashmarks_overlay(ctx)

        self._shape_bounds = {
            "x_min": self.ax.get_xlim()[0],
            "x_max": self.ax.get_xlim()[1],
            "y_min": self.ax.get_ylim()[0],
            "y_max": self.ax.get_ylim()[1],
        }
        self._apply_view_scale()
        self._draw_builtin_dim_lines()
        self._draw_standalone_labels()

    def _apply_transform_pipeline(self, ctx: DrawingContext, shape: str,
                                   transform: TransformState, params: dict) -> None:
        actual_base_side = transform.base_side
        actual_flip_h = transform.flip_h
        actual_flip_v = transform.flip_v
        is_unified = shape in GeometricRotation.ALL_GEOMETRIC_SHAPES
        if is_unified:
            transform = TransformState(flip_h=False, flip_v=False, base_side=0)

        error = self.plot_controller.draw_shape(shape, ctx, transform, params)
        if error:
            self.plot_controller.draw_error(error)

        _xlim = self.ax.get_xlim()
        _ylim = self.ax.get_ylim()
        canonical_center = ((_xlim[0] + _xlim[1]) / 2, (_ylim[0] + _ylim[1]) / 2)
        self._canonical_pre_rotation_center = canonical_center
        self._pre_rotation_center = canonical_center

        if not (is_unified and not error):
            return

        config = ShapeConfigProvider.get(shape)
        num_sides = config.num_sides if config.num_sides > 0 else 4

        if actual_base_side != 0:
            angle_deg = -(360.0 / num_sides) * actual_base_side
            cx, cy = self._pre_rotation_center
            GeometricRotation.rotate_axes_artists(self.ax, angle_deg, cx, cy)
            GeometricRotation.recalculate_limits(self.ax)
            self._rotate_geometry_hints(math.radians(angle_deg), cx, cy)

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

    def _apply_view_scale(self) -> None:
        ax = self.plot_controller.ax
        ax.set_autoscale_on(False)

        bounds = self._shape_bounds
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

        pixel_aspect = self._axes_pixel_aspect
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

        _raw = self.scale_manager.get("view_scale")
        scale = max(0.25, min(1.0, _raw))
        final_width  = base_width  / scale
        final_height = base_height / scale

        pan_x, pan_y = self._shape_pan_offset
        max_pan_x = final_width  * 0.35
        max_pan_y = final_height * 0.35
        pan_x = max(-max_pan_x, min(max_pan_x, pan_x))
        pan_y = max(-max_pan_y, min(max_pan_y, pan_y))
        self._shape_pan_offset = (pan_x, pan_y)

        cx = shape_center_x + pan_x
        cy = shape_center_y + pan_y
        ax.set_xlim(cx - final_width  / 2, cx + final_width  / 2)
        ax.set_ylim(cy - final_height / 2, cy + final_height / 2)

    def _draw_builtin_dim_lines(self) -> None:
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
        self._standalone_label_bboxes = []
        font_size = self.font_size

        for i, lbl in enumerate(self.standalone_labels):
            color = "black"
            lbl_txt = self.ax.text(lbl["x"], lbl["y"], lbl["text"],
                        fontsize=font_size, color=color, fontweight="normal",
                        fontfamily=self.font_family,
                        ha="center", va="center", zorder=AppConstants.LABEL_ZORDER + 1,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor="none", alpha=1))
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

        self._standalone_dim_endpoints = []
        self._standalone_dim_label_bboxes = []
        for i, dim in enumerate(self.standalone_dim_lines):
            color = "black"
            lw = AppConstants.DIMENSION_LINE_WIDTH

            preset_key = dim.get("preset_key")
            if preset_key and not dim.get("user_dragged") and not self._suppress_preset_dim_snap:
                fresh = self._calc_dim_line_endpoints(preset_key)
                if fresh:
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
            for px, py in [(x1, y1), (x2, y2)]:
                self.ax.plot(px, py, marker="o", markersize=3, color=color, zorder=13)

            label_x = dim.get("label_x", (x1 + x2) / 2)
            label_y = dim.get("label_y", (y1 + y2) / 2)
            dim_lbl_txt = self.ax.text(label_x, label_y, dim["text"],
                        fontsize=font_size, color=color, fontweight="normal",
                        fontfamily=self.font_family,
                        ha="center", va="center", zorder=AppConstants.LABEL_ZORDER + 1,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor="none", alpha=1))

            self._standalone_dim_endpoints.append({"p1": (x1, y1), "p2": (x2, y2)})
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

    def _draw_smart_hashmarks_overlay(self, ctx: DrawingContext) -> None:
        hints = self.label_manager.geometry_hints
        shape = self.shape_name
        points: list | None = None

        if shape == "Triangle":
            base_p1 = hints.get("tri_base_p1")
            base_p2 = hints.get("tri_base_p2")
            apex = hints.get("tri_apex")
            tri_type = self.triangle_type
            if base_p1 and base_p2 and apex and tri_type not in ("Isosceles", "Equilateral"):
                points = [base_p1, base_p2, apex]
        elif shape == "Rectangle":
            pts = hints.get("rect_pts")
            if pts:
                points = pts
            else:
                b = self._shape_bounds
                if b:
                    points = [
                        (b["x_min"], b["y_min"]), (b["x_max"], b["y_min"]),
                        (b["x_max"], b["y_max"]), (b["x_min"], b["y_max"])
                    ]
        elif shape == "Parallelogram":
            points = hints.get("para_pts")
        elif shape == "Trapezoid":
            points = hints.get("trap_pts")

        if points and len(points) >= 3:
            SmartGeometryEngine.draw_smart_hashmarks(ctx, points)

    # ── Composite drawing ─────────────────────────────────────────────────────

    def _draw_composite_shapes(self, selected_shapes: list[str]) -> None:
        n = len(selected_shapes)
        if n == 0:
            self._composite_bboxes = []
            return

        # Phase 1: Draw each shape at origin, capture artists
        shape_data_list = []
        self._composite_geometry_hints = {}

        for i, shape_name in enumerate(selected_shapes):
            patches_before = len(self.ax.patches)
            texts_before = len(self.ax.texts)
            lines_before = len(self.ax.lines)

            ctx = self.plot_controller.create_drawing_context(aspect_ratio=1.0)
            ctx.composite_mode = True

            config = ShapeConfigProvider.get(shape_name)
            params: dict[str, Any] = {}
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
                shape_transform = TransformState()
                t = self.composite_transforms.get(i, {})
                if t:
                    shape_transform.flip_h = t.get("flip_h", False)
                    shape_transform.flip_v = t.get("flip_v", False)
                    shape_transform.base_side = t.get("base_side", 0)

                actual_bs = shape_transform.base_side
                actual_fh = shape_transform.flip_h
                actual_fv = shape_transform.flip_v
                is_unified = shape_name in GeometricRotation.ALL_GEOMETRIC_SHAPES

                if is_unified and shape_name in GeometricRotation.ARC_SYMMETRIC_SHAPES:
                    _ns = 4
                    if actual_fv:
                        actual_bs = (actual_bs + 2) % _ns
                        actual_fv = False
                    if actual_fh:
                        if actual_bs in (1, 3):
                            actual_bs = _ns - actual_bs
                        actual_fh = False
                if is_unified:
                    shape_transform = TransformState(flip_h=False, flip_v=False, base_side=0)

                drawer.draw(ctx, shape_transform, params)

                _post_draw_xlim = self.ax.get_xlim()
                _post_draw_ylim = self.ax.get_ylim()
                _canonical_cx = (_post_draw_xlim[0] + _post_draw_xlim[1]) / 2
                _canonical_cy = (_post_draw_ylim[0] + _post_draw_ylim[1]) / 2

                snap_anchors = []
                if hasattr(drawer, 'get_snap_anchors'):
                    snap_anchors = drawer.get_snap_anchors(ctx, shape_transform, params)

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
                        _cx = _canonical_cx
                        _cy = _canonical_cy
                        _angle_deg = -(360.0 / _num_sides) * _bs if _bs != 0 else 0.0
                        GeometricRotation.transform_artist_lists(
                            _new_p, _new_l, _new_t,
                            angle_deg=_angle_deg, flip_h=_fh, flip_v=_fv,
                            cx=_cx, cy=_cy)
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

                for _ax_pt, _ay_pt in snap_anchors:
                    b_xs.append(_ax_pt)
                    b_ys.append(_ay_pt)

                if b_xs and b_ys:
                    shape_xlim = (min(b_xs), max(b_xs))
                    shape_ylim = (min(b_ys), max(b_ys))
                else:
                    shape_xlim = tuple(_post_draw_xlim)
                    shape_ylim = tuple(_post_draw_ylim)

                self._composite_geometry_hints[i] = dict(self.label_manager.geometry_hints)
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

        # Phase 2: Grid layout for new (unplaced) shapes
        valid = [s for s in shape_data_list if not s.get("error")]
        if not valid:
            self._composite_bboxes = []
            return

        GRID_COLS = 3
        cols = min(n, GRID_COLS)
        rows = math.ceil(n / GRID_COLS)

        max_w = max((s["xlim"][1] - s["xlim"][0]) for s in valid)
        max_h = max((s["ylim"][1] - s["ylim"][0]) for s in valid)
        cell_w = max(max_w * 1.4, 5.0)
        cell_h = max(max_h * 1.4, 4.0)

        for i in range(n):
            if i not in self.composite_positions:
                row_idx = i // GRID_COLS
                col_idx = i % GRID_COLS
                grid_cx = (col_idx + 0.5) * cell_w
                grid_cy = -(row_idx + 0.5) * cell_h
                self.composite_positions[i] = (grid_cx, grid_cy)

        # Phase 3: Translate each shape to stored position
        self._composite_bboxes = []
        self._composite_snap_anchors = []

        for i, sdata in enumerate(shape_data_list):
            target_cx, target_cy = self.composite_positions.get(i, (0.0, 0.0))
            if sdata.get("error"):
                self.ax.text(target_cx, target_cy, f"[{sdata['name']}]",
                             ha="center", va="center", fontsize=10, color="orange")
                self._composite_bboxes.append((0, 0, 0, 0))
                self._composite_snap_anchors.append([])
                continue

            shape_cx = (sdata["xlim"][0] + sdata["xlim"][1]) / 2
            shape_cy = (sdata["ylim"][0] + sdata["ylim"][1]) / 2
            dx = target_cx - shape_cx
            dy = target_cy - shape_cy
            half_w = (sdata["xlim"][1] - sdata["xlim"][0]) / 2
            half_h = (sdata["ylim"][1] - sdata["ylim"][0]) / 2
            self._composite_bboxes.append((
                target_cx - half_w, target_cy - half_h,
                target_cx + half_w, target_cy + half_h
            ))

            for p in sdata["patches"]:
                self._translate_patch(p, dx, dy)
            for line in sdata["lines"]:
                xdata = np.array(line.get_xdata(), dtype=float)
                ydata = np.array(line.get_ydata(), dtype=float)
                line.set_xdata(xdata + dx)
                line.set_ydata(ydata + dy)
            for t in sdata["texts"]:
                if t.get_transform() != self.ax.transData:
                    if hasattr(t, 'get_unitless_position'):
                        tx, ty = t.get_position()
                        t.set_position((tx + dx, ty + dy))
                    continue
                tx, ty = t.get_position()
                t.set_position((tx + dx, ty + dy))

            translated_anchors = [(ax + dx, ay + dy) for ax, ay in sdata.get("snap_anchors", [])]
            self._composite_snap_anchors.append(translated_anchors)

            _POINT_KEYS = {"tri_foot", "tri_apex", "tri_base_p1", "tri_base_p2",
                           "prism_f_bl", "prism_f_br", "prism_f_tl", "prism_f_tr",
                           "prism_b_bl", "prism_b_br", "prism_b_tl", "prism_b_tr"}
            _LIST_KEYS = {"rect_pts", "tri_pts", "para_pts", "trap_pts", "polygon_pts"}
            if i in self._composite_geometry_hints:
                th = {}
                for k, v in self._composite_geometry_hints[i].items():
                    if k in _POINT_KEYS and isinstance(v, (tuple, list)) and len(v) == 2:
                        th[k] = (v[0] + dx, v[1] + dy)
                    elif k in _LIST_KEYS and isinstance(v, list):
                        th[k] = [(p[0] + dx, p[1] + dy) for p in v]
                    elif k in ("height_y1", "height_y2"):
                        th[k] = v + dy
                    elif k == "height_x":
                        th[k] = v + dx
                    else:
                        th[k] = v
                self._composite_geometry_hints[i] = th

        # Phase 4: Set view bounds
        all_x_min = min(b[0] for b in self._composite_bboxes if b != (0, 0, 0, 0)) if self._composite_bboxes else 0
        all_x_max = max(b[2] for b in self._composite_bboxes if b != (0, 0, 0, 0)) if self._composite_bboxes else cols * cell_w
        all_y_min = min(b[1] for b in self._composite_bboxes if b != (0, 0, 0, 0)) if self._composite_bboxes else 0
        all_y_max = max(b[3] for b in self._composite_bboxes if b != (0, 0, 0, 0)) if self._composite_bboxes else rows * cell_h

        for lbl in self.composite_labels:
            all_x_min = min(all_x_min, lbl["x"] - 1.0)
            all_x_max = max(all_x_max, lbl["x"] + 1.0)
            all_y_min = min(all_y_min, lbl["y"] - 1.0)
            all_y_max = max(all_y_max, lbl["y"] + 1.0)
        for dim in self.composite_dim_lines_list:
            for key_x, key_y in [("x1", "y1"), ("x2", "y2"), ("label_x", "label_y")]:
                if key_x in dim and key_y in dim:
                    all_x_min = min(all_x_min, dim[key_x] - 0.5)
                    all_x_max = max(all_x_max, dim[key_x] + 0.5)
                    all_y_min = min(all_y_min, dim[key_y] - 0.5)
                    all_y_max = max(all_y_max, dim[key_y] + 0.5)

        content_w = all_x_max - all_x_min
        content_h = all_y_max - all_y_min
        center_x = (all_x_min + all_x_max) / 2
        center_y = (all_y_min + all_y_max) / 2
        margin_x = content_w * 0.12
        margin_y = content_h * 0.12
        padded_w = content_w + 2 * margin_x
        padded_h = content_h + 2 * margin_y
        pixel_aspect = self._axes_pixel_aspect
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
        self._composite_view_limits = (vx_min, vx_max, vy_min, vy_max)

        # Phase 5: Draw composite labels
        self._composite_label_bboxes = []
        for i, lbl in enumerate(self.composite_labels):
            color = "black"
            txt = self.ax.text(lbl["x"], lbl["y"], lbl["text"],
                               fontsize=self.font_size, color=color, fontweight="normal",
                               fontfamily=self.font_family,
                               ha="center", va="center", zorder=AppConstants.LABEL_ZORDER + 1,
                               bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                         edgecolor="none", alpha=0.9))
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
                char_w = self.font_size * 0.12
                char_h = self.font_size * 0.2
                half_w = (len(lbl["text"]) * char_w / 2) + pad_x
                half_h = (char_h / 2) + pad_y
            self._composite_label_bboxes.append((
                lbl["x"] - half_w, lbl["y"] - half_h,
                lbl["x"] + half_w, lbl["y"] + half_h
            ))

        # Phase 6: Draw composite dimension lines
        self._composite_dim_endpoints = []
        self._composite_dim_label_bboxes = []
        font_size = self.font_size

        for i, dim in enumerate(self.composite_dim_lines_list):
            color = "black"
            lw = AppConstants.DIMENSION_LINE_WIDTH

            preset_key = dim.get("preset_key")
            owner_idx = dim.get("shape_owner")
            if (preset_key and not dim.get("user_dragged") and not self._suppress_preset_dim_snap
                    and owner_idx is not None
                    and owner_idx < len(self._composite_bboxes)):
                owner_bbox = self._composite_bboxes[owner_idx]
                if owner_bbox != (0, 0, 0, 0):
                    x_min, y_min, x_max, y_max = owner_bbox
                    saved_bounds = self._shape_bounds
                    saved_hints = dict(self.label_manager.geometry_hints)
                    saved_xlim = self.ax.get_xlim()
                    saved_ylim = self.ax.get_ylim()
                    try:
                        self._shape_bounds = {
                            "x_min": x_min, "x_max": x_max,
                            "y_min": y_min, "y_max": y_max,
                        }
                        shape_hints = self._composite_geometry_hints.get(owner_idx, {})
                        self.label_manager.geometry_hints.clear()
                        self.label_manager.geometry_hints.update(shape_hints)
                        pad = max((x_max - x_min), (y_max - y_min)) * 0.15 or 0.5
                        self.ax.set_xlim(x_min - pad, x_max + pad)
                        self.ax.set_ylim(y_min - pad, y_max + pad)
                        fresh = self._calc_dim_line_endpoints(preset_key)
                    finally:
                        self._shape_bounds = saved_bounds
                        self.label_manager.geometry_hints.clear()
                        self.label_manager.geometry_hints.update(saved_hints)
                        self.ax.set_xlim(*saved_xlim)
                        self.ax.set_ylim(*saved_ylim)
                    if fresh:
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
            for px, py in [(x1, y1), (x2, y2)]:
                self.ax.plot(px, py, marker="o", markersize=3, color=color, zorder=13)

            label_x = dim.get("label_x", (x1 + x2) / 2)
            label_y = dim.get("label_y", (y1 + y2) / 2)
            dim_txt = self.ax.text(label_x, label_y, dim["text"],
                        fontsize=font_size, color=color, fontweight="normal",
                        fontfamily=self.font_family,
                        ha="center", va="center", zorder=AppConstants.LABEL_ZORDER + 1,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor="none", alpha=1))
            self._composite_dim_endpoints.append({"p1": (x1, y1), "p2": (x2, y2)})
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

    @staticmethod
    def _translate_patch(patch, dx: float, dy: float) -> None:
        """Translate a matplotlib patch by (dx, dy) in data coordinates."""
        if isinstance(patch, (patches.Polygon, patches.FancyArrow)):
            xy = patch.get_xy()
            patch.set_xy(xy + np.array([dx, dy]))
        elif isinstance(patch, (patches.Circle, patches.Ellipse, patches.Arc, patches.Wedge)):
            cx, cy = patch.center
            patch.set_center((cx + dx, cy + dy))
        elif isinstance(patch, patches.Rectangle):
            x, y = patch.get_xy()
            patch.set_xy((x + dx, y + dy))
        else:
            if hasattr(patch, 'get_xy'):
                xy = patch.get_xy()
                if hasattr(xy, '__add__'):
                    patch.set_xy(xy + np.array([dx, dy]))
                else:
                    patch.set_xy((xy[0] + dx, xy[1] + dy))
            elif hasattr(patch, 'center'):
                cx, cy = patch.center
                patch.set_center((cx + dx, cy + dy))

    # ── Geometry hint transforms ──────────────────────────────────────────────

    def _rotate_geometry_hints(self, angle_rad: float, cx: float, cy: float) -> None:
        rp = GeometricRotation.rotate_point
        hints = self.label_manager.geometry_hints
        point_keys = [
            "tri_foot", "tri_apex", "tri_base_p1", "tri_base_p2",
            "prism_f_bl", "prism_f_br", "prism_f_tl", "prism_f_tr",
            "prism_b_bl", "prism_b_br", "prism_b_tl", "prism_b_tr",
            "prism_tri_p1", "prism_tri_p2", "prism_tri_p3",
            "prism_tri_b1", "prism_tri_b2", "prism_tri_b3",
        ]
        for key in point_keys:
            pt = hints.get(key)
            if pt and isinstance(pt, (tuple, list)) and len(pt) == 2:
                hints[key] = rp(pt[0], pt[1], angle_rad, cx, cy)
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
            if rx1 is not None and rx2 is not None:
                hints["height_x"] = (rx1 + rx2) / 2
            elif rx1 is not None:
                hints["height_x"] = rx1
            elif rx2 is not None:
                hints["height_x"] = rx2
        list_keys = ["rect_pts", "para_pts", "trap_pts", "tri_pts", "polygon_pts"]
        for key in list_keys:
            pts = hints.get(key)
            if pts and isinstance(pts, list):
                hints[key] = [rp(p[0], p[1], angle_rad, cx, cy) for p in pts]
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
        if not flip_h and not flip_v:
            return

        def mirror(px, py):
            x = (2 * cx - px) if flip_h else px
            y = (2 * cy - py) if flip_v else py
            return x, y

        hints = self.label_manager.geometry_hints
        point_keys = [
            "tri_foot", "tri_apex", "tri_base_p1", "tri_base_p2",
            "prism_f_bl", "prism_f_br", "prism_f_tl", "prism_f_tr",
            "prism_b_bl", "prism_b_br", "prism_b_tl", "prism_b_tr",
            "prism_tri_p1", "prism_tri_p2", "prism_tri_p3",
            "prism_tri_b1", "prism_tri_b2", "prism_tri_b3",
        ]
        for key in point_keys:
            pt = hints.get(key)
            if pt and isinstance(pt, (tuple, list)) and len(pt) == 2:
                hints[key] = mirror(pt[0], pt[1])
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

    # ── Dimension line endpoint calculators ───────────────────────────────────

    def _calc_dim_line_endpoints(self, preset_key: str) -> dict | None:
        if not self._shape_bounds:
            return None
        b = self._shape_bounds
        hints = self.label_manager.geometry_hints
        offset = DrawingUtilities.dim_offset_from_axes(self.ax)
        label_gap = DrawingUtilities.dim_offset_from_axes(
            self.ax, px=AppConstants.PRESET_DIM_OFFSET_PX * 0.5)
        if not hasattr(self, '_dim_dispatch'):
            self._dim_dispatch = {
                "height":        self._calc_dim_height,
                "width":         self._calc_dim_width,
                "tri_base":      self._calc_dim_tri_base,
                "tri_length":    self._calc_dim_tri_length,
                "side_v":        self._calc_dim_side_v,
                "side_h":        self._calc_dim_side_h,
                "side_l":        self._calc_dim_side_l,
                "side_r":        self._calc_dim_side_r,
                "para_height":   self._calc_dim_para_trap_height,
                "trap_height":   self._calc_dim_para_trap_height,
                "para_base":     self._calc_dim_para_base,
                "para_top":      self._calc_dim_para_top,
                "para_side_l":   self._calc_dim_para_side_l,
                "para_side_r":   self._calc_dim_para_side_r,
                "trap_base":     self._calc_dim_trap_base,
                "trap_top":      self._calc_dim_trap_top,
                "trap_side_l":   self._calc_dim_trap_side_l,
                "trap_side_r":   self._calc_dim_trap_side_r,
                "radius":        self._calc_dim_radius,
                "diameter":      self._calc_dim_diameter,
                "slant":         self._calc_dim_slant,
                "length":        self._calc_dim_length,
                "side":          self._calc_dim_side,
                "circumference": self._calc_dim_circumference,
                "leg_a":         self._calc_dim_leg_a,
                "leg_b":         self._calc_dim_leg_b,
                "hyp":           self._calc_dim_hyp,
            }
        fn = self._dim_dispatch.get(preset_key)
        if fn is None:
            return None
        return fn(hints, b, offset, label_gap)

    @staticmethod
    def _dim_perp_offset(p1, p2, offset, label_gap, shape_cx, shape_cy):
        dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
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

    def _calc_dim_height(self, hints, b, offset, label_gap):
        shape_left = b["x_min"]; shape_right = b["x_max"]
        shape_bottom = b["y_min"]; shape_top = b["y_max"]
        shape_cx = (shape_left + shape_right) / 2
        shape_cy = (shape_bottom + shape_top) / 2
        if "prism_f_bl" in hints and "prism_f_tl" in hints:
            res = self._dim_perp_offset(hints["prism_f_bl"], hints["prism_f_tl"],
                                        offset, label_gap, shape_cx, shape_cy)
            if res: return res
        if "prism_tri_p1" in hints:
            foot = hints["tri_foot"]; apex = hints["tri_apex"]
            p1, p2, p3 = hints["prism_tri_p1"], hints["prism_tri_p2"], hints["prism_tri_p3"]
            mid_x = (foot[0] + apex[0]) / 2; mid_y = (foot[1] + apex[1]) / 2
            dx = apex[0] - foot[0]; dy = apex[1] - foot[1]
            seg_len = math.sqrt(dx**2 + dy**2) or 1
            nx, ny = -dy / seg_len, dx / seg_len
            cx_tri = (p1[0] + p2[0] + p3[0]) / 3; cy_tri = (p1[1] + p2[1] + p3[1]) / 3
            if nx * (mid_x + nx) + ny * (mid_y + ny) < nx * cx_tri + ny * cy_tri:
                nx, ny = -nx, -ny
            off2 = offset * 1.2
            return {"x1": foot[0] + nx * off2, "y1": foot[1] + ny * off2,
                    "x2": apex[0] + nx * off2, "y2": apex[1] + ny * off2,
                    "label_x": mid_x + nx * (off2 + label_gap),
                    "label_y": mid_y + ny * (off2 + label_gap), "constraint": None}
        if "tri_foot" in hints:
            foot = hints["tri_foot"]; apex = hints["tri_apex"]
            mid_x = (foot[0] + apex[0]) / 2; mid_y = (foot[1] + apex[1]) / 2
            return {"x1": foot[0], "y1": foot[1], "x2": apex[0], "y2": apex[1],
                    "label_x": mid_x + label_gap, "label_y": mid_y, "constraint": None}
        if "height_y1" in hints:
            if "rect_pts" in hints and "prism_f_bl" not in hints and "prism_tri_p1" not in hints:
                pts = hints["rect_pts"]
                res = self._dim_perp_offset(pts[1], pts[2], offset, label_gap, shape_cx, shape_cy)
                if res: return res
            hx = hints.get("height_x", shape_cx)
            y1 = hints["height_y1"]; y2 = hints["height_y2"]
            foot = (hx, y1); top = (hx, y2)
            res = self._dim_perp_offset(foot, top, offset, label_gap, shape_cx, shape_cy)
            if res: return res
            x = shape_right + offset
            return {"x1": x, "y1": y1, "x2": x, "y2": y2,
                    "label_x": x + label_gap, "label_y": (y1 + y2) / 2, "constraint": "height"}
        x = shape_right + offset
        return {"x1": x, "y1": shape_bottom, "x2": x, "y2": shape_top,
                "label_x": x + label_gap, "label_y": shape_cy, "constraint": "height"}

    def _calc_dim_width(self, hints, b, offset, label_gap):
        shape_left = b["x_min"]; shape_right = b["x_max"]
        shape_bottom = b["y_min"]; shape_top = b["y_max"]
        shape_cx = (shape_left + shape_right) / 2
        shape_cy = (shape_bottom + shape_top) / 2
        if "prism_f_br" in hints and "prism_b_br" in hints:
            f_br = hints["prism_f_br"]; b_br = hints["prism_b_br"]
            mx = (f_br[0] + b_br[0]) / 2; my = (f_br[1] + b_br[1]) / 2
            dx = b_br[0] - f_br[0]; dy = b_br[1] - f_br[1]
            seg_len = math.sqrt(dx**2 + dy**2) or 1
            nx, ny = -dy / seg_len, dx / seg_len
            if nx * (mx - shape_cx) + ny * (my - shape_cy) < 0:
                nx, ny = -nx, -ny
            return {"x1": f_br[0] + nx * offset, "y1": f_br[1] + ny * offset,
                    "x2": b_br[0] + nx * offset, "y2": b_br[1] + ny * offset,
                    "label_x": mx + nx * (offset + label_gap),
                    "label_y": my + ny * (offset + label_gap), "constraint": "width"}
        if "tri_foot" in hints and "tri_base_p1" in hints and "tri_base_p2" in hints:
            p1 = hints["tri_base_p1"]; p2 = hints["tri_base_p2"]
            res = self._dim_perp_offset(p1, p2, offset, label_gap, shape_cx, shape_cy)
            if res: return res
            y = shape_bottom - offset
            return {"x1": p1[0], "y1": y, "x2": p2[0], "y2": y,
                    "label_x": (p1[0] + p2[0]) / 2, "label_y": y - label_gap, "constraint": "width"}
        pts = hints.get("rect_pts")
        if pts and len(pts) >= 4:
            res = self._dim_perp_offset(pts[0], pts[1], offset, label_gap, shape_cx, shape_cy)
            if res: return res
        y = shape_bottom - offset
        return {"x1": shape_left, "y1": y, "x2": shape_right, "y2": y,
                "label_x": shape_cx, "label_y": y - label_gap, "constraint": "width"}

    def _calc_dim_tri_base(self, hints, b, offset, label_gap):
        shape_cx = (b["x_min"] + b["x_max"]) / 2
        shape_cy = (b["y_min"] + b["y_max"]) / 2
        if "prism_tri_p1" not in hints: return None
        p1, p2 = hints["prism_tri_p1"], hints["prism_tri_p2"]
        res = self._dim_perp_offset(p1, p2, offset, label_gap, shape_cx, shape_cy)
        if res: return res
        y = min(p1[1], p2[1]) - offset
        return {"x1": p1[0], "y1": y, "x2": p2[0], "y2": y,
                "label_x": (p1[0] + p2[0]) / 2, "label_y": y - label_gap, "constraint": "width"}

    def _calc_dim_tri_length(self, hints, b, offset, label_gap):
        if "prism_tri_p1" not in hints: return None
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
        if (ep1_raw[1] + ep2_raw[1]) / 2 + ny > shape_cy_tri: nx, ny = -nx, -ny
        ep1 = (ep1_raw[0] + nx * offset, ep1_raw[1] + ny * offset)
        ep2 = (ep2_raw[0] + nx * offset, ep2_raw[1] + ny * offset)
        mx = (ep1[0] + ep2[0]) / 2; my = (ep1[1] + ep2[1]) / 2
        return {"x1": ep1[0], "y1": ep1[1], "x2": ep2[0], "y2": ep2[1],
                "label_x": mx + nx * label_gap, "label_y": my + ny * label_gap, "constraint": None}

    def _calc_dim_side_v(self, hints, b, offset, label_gap):
        shape_cx = (b["x_min"] + b["x_max"]) / 2
        shape_cy = (b["y_min"] + b["y_max"]) / 2
        pts = hints.get("rect_pts")
        if pts and len(pts) >= 4:
            res = self._dim_perp_offset(pts[1], pts[2], offset, label_gap, shape_cx, shape_cy)
            if res: return res
        x = b["x_max"] + offset
        return {"x1": x, "y1": b["y_min"], "x2": x, "y2": b["y_max"],
                "label_x": x + label_gap, "label_y": shape_cy, "constraint": "height"}

    def _calc_dim_side_h(self, hints, b, offset, label_gap):
        shape_cx = (b["x_min"] + b["x_max"]) / 2
        shape_cy = (b["y_min"] + b["y_max"]) / 2
        pts = hints.get("rect_pts")
        if pts and len(pts) >= 4:
            res = self._dim_perp_offset(pts[0], pts[1], offset, label_gap, shape_cx, shape_cy)
            if res: return res
        y = b["y_min"] - offset
        return {"x1": b["x_min"], "y1": y, "x2": b["x_max"], "y2": y,
                "label_x": shape_cx, "label_y": y - label_gap, "constraint": "width"}

    def _calc_dim_side_l(self, hints, b, offset, label_gap):
        if "tri_base_p1" not in hints or "tri_apex" not in hints: return None
        return self._dim_edge_offset(hints["tri_base_p1"], hints["tri_apex"], offset, label_gap)

    def _calc_dim_side_r(self, hints, b, offset, label_gap):
        if "tri_base_p2" not in hints or "tri_apex" not in hints: return None
        return self._dim_edge_offset(hints["tri_apex"], hints["tri_base_p2"], offset, label_gap)

    def _calc_dim_para_trap_height(self, hints, b, offset, label_gap):
        pts = hints.get("para_pts") or hints.get("trap_pts")
        if not pts or len(pts) < 4: return None
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
            best = sorted(internals, key=lambda c: c["t"])[0] if internals else \
                   sorted(candidates, key=lambda c: c["ext"], reverse=True)[0]
            apex = best["pt"]
            foot = (p0[0] + ub[0] * best["t"], p0[1] + ub[1] * best["t"])
        else:
            foot, apex = p0, p3
        mid_y = (foot[1] + apex[1]) / 2
        return {"x1": foot[0], "y1": foot[1], "x2": apex[0], "y2": apex[1],
                "label_x": foot[0] + label_gap, "label_y": mid_y, "constraint": None}

    def _calc_dim_para_base(self, hints, b, offset, label_gap):
        shape_cx = (b["x_min"] + b["x_max"]) / 2
        shape_cy = (b["y_min"] + b["y_max"]) / 2
        pts = hints.get("para_pts")
        if not pts or len(pts) < 4: return None
        p0, p1 = pts[0], pts[1]
        res = self._dim_perp_offset(p0, p1, offset, label_gap, shape_cx, shape_cy)
        if res: return res
        y = min(p0[1], p1[1]) - offset
        return {"x1": p0[0], "y1": y, "x2": p1[0], "y2": y,
                "label_x": (p0[0] + p1[0]) / 2, "label_y": y - label_gap, "constraint": None}

    def _calc_dim_para_top(self, hints, b, offset, label_gap):
        shape_cx = (b["x_min"] + b["x_max"]) / 2
        shape_cy = (b["y_min"] + b["y_max"]) / 2
        pts = hints.get("para_pts")
        if not pts or len(pts) < 4: return None
        p2, p3 = pts[2], pts[3]
        res = self._dim_perp_offset(p3, p2, offset, label_gap, shape_cx, shape_cy)
        if res: return res
        y = max(p2[1], p3[1]) + offset
        return {"x1": p3[0], "y1": y, "x2": p2[0], "y2": y,
                "label_x": (p3[0] + p2[0]) / 2, "label_y": y + label_gap, "constraint": None}

    def _calc_dim_para_side_l(self, hints, b, offset, label_gap):
        pts = hints.get("para_pts")
        if not pts or len(pts) < 4: return None
        return self._dim_edge_offset(pts[0], pts[3], offset, label_gap)

    def _calc_dim_para_side_r(self, hints, b, offset, label_gap):
        pts = hints.get("para_pts")
        if not pts or len(pts) < 4: return None
        p1, p2 = pts[1], pts[2]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        seg_len = math.sqrt(dx**2 + dy**2) or 1
        nx, ny = dy / seg_len, -dx / seg_len
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        return {"x1": p1[0] + nx * offset, "y1": p1[1] + ny * offset,
                "x2": p2[0] + nx * offset, "y2": p2[1] + ny * offset,
                "label_x": mx + nx * (offset + label_gap),
                "label_y": my + ny * (offset + label_gap), "constraint": None}

    def _calc_dim_trap_base(self, hints, b, offset, label_gap):
        pts = hints.get("trap_pts")
        if not pts or len(pts) < 4: return None
        p0, p1 = pts[0], pts[1]
        y = min(p0[1], p1[1]) - offset
        return {"x1": p0[0], "y1": y, "x2": p1[0], "y2": y,
                "label_x": (p0[0] + p1[0]) / 2, "label_y": y - label_gap, "constraint": None}

    def _calc_dim_trap_top(self, hints, b, offset, label_gap):
        pts = hints.get("trap_pts")
        if not pts or len(pts) < 4: return None
        p2, p3 = pts[2], pts[3]
        y = max(p2[1], p3[1]) + offset
        return {"x1": p3[0], "y1": y, "x2": p2[0], "y2": y,
                "label_x": (p3[0] + p2[0]) / 2, "label_y": y + label_gap, "constraint": None}

    def _calc_dim_trap_side_l(self, hints, b, offset, label_gap):
        pts = hints.get("trap_pts")
        if not pts or len(pts) < 4: return None
        return self._dim_edge_offset(pts[0], pts[3], offset, label_gap)

    def _calc_dim_trap_side_r(self, hints, b, offset, label_gap):
        pts = hints.get("trap_pts")
        if not pts or len(pts) < 4: return None
        p1, p2 = pts[1], pts[2]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        seg_len = math.sqrt(dx**2 + dy**2) or 1
        nx, ny = dy / seg_len, -dx / seg_len
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        return {"x1": p1[0] + nx * offset, "y1": p1[1] + ny * offset,
                "x2": p2[0] + nx * offset, "y2": p2[1] + ny * offset,
                "label_x": mx + nx * (offset + label_gap),
                "label_y": my + ny * (offset + label_gap), "constraint": None}

    def _calc_dim_radius(self, hints, b, offset, label_gap):
        shape_cx = (b["x_min"] + b["x_max"]) / 2
        shape_cy = (b["y_min"] + b["y_max"]) / 2
        shape_w = b["x_max"] - b["x_min"]
        r = hints.get("radius", shape_w / 2)
        return {"x1": shape_cx, "y1": shape_cy,
                "x2": shape_cx + r, "y2": shape_cy,
                "label_x": shape_cx + r * 0.65,
                "label_y": shape_cy - offset * 2, "constraint": None}

    def _calc_dim_diameter(self, hints, b, offset, label_gap):
        shape_cx = (b["x_min"] + b["x_max"]) / 2
        shape_cy = (b["y_min"] + b["y_max"]) / 2
        return {"x1": b["x_min"], "y1": shape_cy,
                "x2": b["x_max"], "y2": shape_cy,
                "label_x": shape_cx, "label_y": shape_cy - offset * 2, "constraint": None}

    def _calc_dim_slant(self, hints, b, offset, label_gap):
        shape_right = b["x_max"]; shape_bottom = b["y_min"]; shape_top = b["y_max"]
        shape_cx = (b["x_min"] + b["x_max"]) / 2; shape_cy = (shape_bottom + shape_top) / 2
        return {"x1": shape_right, "y1": shape_bottom, "x2": shape_cx, "y2": shape_top,
                "label_x": (shape_right + shape_cx) / 2 + offset, "label_y": shape_cy,
                "constraint": None}

    def _calc_dim_length(self, hints, b, offset, label_gap):
        shape_left = b["x_min"]; shape_right = b["x_max"]
        shape_bottom = b["y_min"]
        shape_cx = (shape_left + shape_right) / 2
        if "prism_f_bl" in hints:
            f_bl = hints["prism_f_bl"]; f_br = hints["prism_f_br"]
            y = min(f_bl[1], f_br[1]) - offset
            mx = (f_bl[0] + f_br[0]) / 2
            return {"x1": f_bl[0], "y1": y, "x2": f_br[0], "y2": y,
                    "label_x": mx, "label_y": y - label_gap, "constraint": None}
        y = shape_bottom - offset
        return {"x1": shape_left, "y1": y, "x2": shape_right, "y2": y,
                "label_x": shape_cx, "label_y": y - label_gap, "constraint": "width"}

    def _calc_dim_side(self, hints, b, offset, label_gap):
        shape_left = b["x_min"]; shape_right = b["x_max"]
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
                nx, ny = dy / seg_len, -dx / seg_len
                return {"x1": p1[0] + nx * offset, "y1": p1[1] + ny * offset,
                        "x2": p2[0] + nx * offset, "y2": p2[1] + ny * offset,
                        "label_x": mid_x + nx * (offset + label_gap),
                        "label_y": mid_y + ny * (offset + label_gap), "constraint": None}
        y = shape_bottom - offset
        return {"x1": shape_left, "y1": y, "x2": shape_right, "y2": y,
                "label_x": shape_cx, "label_y": y - label_gap, "constraint": None}

    def _calc_dim_leg_a(self, hints, b, offset, label_gap):
        pts = hints.get("tri_pts")
        if pts and len(pts) >= 3:
            p0, p2 = pts[0], pts[2]
            shape_cx = (b["x_min"] + b["x_max"]) / 2
            shape_cy = (b["y_min"] + b["y_max"]) / 2
            res = self._dim_perp_offset(p2, p0, offset, label_gap, shape_cx, shape_cy)
            if res: return res
        return None

    def _calc_dim_leg_b(self, hints, b, offset, label_gap):
        pts = hints.get("tri_pts")
        if pts and len(pts) >= 3:
            p0, p1 = pts[0], pts[1]
            y = min(p0[1], p1[1]) - offset
            mx = (p0[0] + p1[0]) / 2
            return {"x1": p0[0], "y1": y, "x2": p1[0], "y2": y,
                    "label_x": mx, "label_y": y - label_gap, "constraint": None}
        return None

    def _calc_dim_hyp(self, hints, b, offset, label_gap):
        pts = hints.get("tri_pts")
        if pts and len(pts) >= 3:
            p1, p2 = pts[1], pts[2]
            dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
            seg_len = math.sqrt(dx**2 + dy**2) or 1
            nx, ny = -dy / seg_len, dx / seg_len
            p0 = pts[0]
            mid_x = (p1[0] + p2[0]) / 2; mid_y = (p1[1] + p2[1]) / 2
            if nx * (mid_x - p0[0]) + ny * (mid_y - p0[1]) < 0:
                nx, ny = -nx, -ny
            return {"x1": p1[0] + nx * offset, "y1": p1[1] + ny * offset,
                    "x2": p2[0] + nx * offset, "y2": p2[1] + ny * offset,
                    "label_x": mid_x + nx * (offset + label_gap),
                    "label_y": mid_y + ny * (offset + label_gap), "constraint": None}
        return None

    def _calc_dim_circumference(self, hints, b, offset, label_gap):
        return None  # Handled as arc toggle
