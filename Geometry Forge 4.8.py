from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Optional, Any, TypedDict
import contextlib
from dataclasses import dataclass, field
from enum import Enum
import math
import functools
import numpy as np
import subprocess
import platform
import os
import logging
from io import BytesIO

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.patches as patches


# Enable High-DPI scaling for Windows
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

# Type aliases for geometry primitives (kept near top for readability)
Point = tuple[float, float]
Polygon = list[Point]

__all__ = [
    # Primary application entry point
    "GeometryApp",
    # Extension points — classes a downstream integrator might subclass or inspect
    "ShapeDrawer",
    "ShapeRegistry",
    "AppConstants",
    "TriangleType",
    "ShapeConfig",
    "ShapeConfigProvider",
    "ShapeFeature",
    # Public value types
    "Point",
    "Polygon",
]

# Configure logging — must be before any class that uses logger
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
# Quiet matplotlib font manager debug spam
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


class StandaloneDimLine(TypedDict, total=False):
    """A freeform or preset dimension line on the standalone canvas.

    Required fields (total=True subset via separate TypedDict not needed here;
    callers must always supply x1/y1/x2/y2/text).
    Optional fields default to None / False when absent.
    """
    x1: float
    y1: float
    x2: float
    y2: float
    text: str
    label_x: float | None   # None → midpoint fallback used at render time
    label_y: float | None
    preset_key: str | None  # None for freeform lines
    user_dragged: bool
    constraint: str | None  # "height", "width", or None


class CompositeDimLine(TypedDict, total=False):
    """A dimension line on the composite canvas."""
    x1: float
    y1: float
    x2: float
    y2: float
    text: str
    label_x: float | None   # None → midpoint fallback used at render time
    label_y: float | None
    constraint: str | None  # "height", "width", "free", or None


class ValidationError(Exception):
    """Raised when shape parameters are invalid."""
    pass

class ShapeValidator:
    """Centralized validation for shape parameters."""
    
    @staticmethod
    def validate_positive(name: str, value: Optional[float]) -> Optional[str]:
        """Check if value is positive. Returns error message or None."""
        if value is None:
            return None
        if not math.isfinite(value):
            return f"{name} must be a finite number"
        if value <= 0:
            return f"{name} must be positive"
        return None
    
    @staticmethod
    def validate_equal(name1: str, val1: Optional[float], 
                       name2: str, val2: Optional[float],
                       tolerance: float = 0.001) -> Optional[str]:
        """Check if two values are equal. Returns error message or None."""
        if val1 is not None and not math.isfinite(val1):
            return f"{name1} must be a finite number"
        if val2 is not None and not math.isfinite(val2):
            return f"{name2} must be a finite number"
        if val1 is not None and val1 > 0 and val2 is not None and val2 > 0:
            if abs(val1 - val2) > tolerance:
                return f"{name1} and {name2} must be equal"
        return None
    
    @staticmethod
    def validate_all_equal(values: dict[str, Optional[float]], 
                          tolerance: float = 0.001) -> Optional[str]:
        """Check if all provided values are equal. Returns error message or None."""
        for k, v in values.items():
            if v is not None and not math.isfinite(v):
                return f"{k} must be a finite number"
        numeric_values = [(k, v) for k, v in values.items() 
                         if v is not None and v > 0]
        if len(numeric_values) > 1:
            first_val = numeric_values[0][1]
            for name, val in numeric_values[1:]:
                if abs(val - first_val) > tolerance:
                    return f"All values must be equal ({', '.join(values.keys())})"
        return None
    
    @staticmethod
    def validate_mutually_exclusive(group1_name: str, group1_has: bool,
                                    group2_name: str, group2_has: bool) -> Optional[str]:
        """Check that only one of two groups has values. Returns error message or None."""
        if group1_has and group2_has:
            return f"Enter {group1_name} OR {group2_name}, not both"
        return None
    
    @staticmethod
    def validate_diameter_radius(radius: Any, diameter: Any, 
                                  default_radius: float = 3.0) -> tuple[float, Optional[str]]:
        """Validate radius/diameter relationship. Returns (radius, error_message)."""
        try:
            r_val = float(radius) if radius not in [None, ""] else None
            d_val = float(diameter) if diameter not in [None, ""] else None
        except (ValueError, TypeError):
            return default_radius, "Radius and Diameter must be valid numbers"

        if r_val is not None and not math.isfinite(r_val):
            return default_radius, "Radius must be a finite number"
        if d_val is not None and not math.isfinite(d_val):
            return default_radius, "Diameter must be a finite number"

        if r_val is not None and r_val <= 0:
            return default_radius, "Radius must be positive"
        if d_val is not None and d_val <= 0:
            return default_radius, "Diameter must be positive"
        
        has_radius = r_val is not None and r_val > 0
        has_diameter = d_val is not None and d_val > 0
        
        if has_radius and has_diameter:
            if abs(d_val - (r_val * 2)) > 0.001:
                return default_radius, "Diameter must be exactly 2x the radius"
        
        if has_diameter:
            return d_val / 2, None
        if has_radius:
            return r_val, None
        
        return default_radius, None


class DrawingUtilities:
    """Common drawing utilities for shapes."""
    
    # Centralized Z-Order Layers
    Z_BACK = 1
    Z_BODY = 2
    Z_FRONT = 3
    Z_LINES = 4
    Z_LABELS = 5
    
    @staticmethod
    def draw_right_angle_marker(ctx: "DrawingContext", vertex: "Point",
                                 p1_dir: "Point", p2_dir: "Point",
                                 size: Optional[float] = None) -> None:
        """Draw a right angle marker (small square) at a vertex with global scaling logic."""
        if size is None:
            # Standardized engine scaling: 10% of the minor dimension or 0.3 units fixed
            d1_len = math.sqrt(p1_dir[0]**2 + p1_dir[1]**2)
            d2_len = math.sqrt(p2_dir[0]**2 + p2_dir[1]**2)
            # Use the smaller adjacent side as the reference to prevent marker overlap
            ref_len = min(d1_len, d2_len) if (d1_len > 0 and d2_len > 0) else 1.0
            size = max(0.15, min(ref_len * 0.15, 0.4))
        
        # Normalize direction vectors
        d1 = DrawingUtilities.normalize_vector(p1_dir)
        d2 = DrawingUtilities.normalize_vector(p2_dir)
        
        # Check if directions are perpendicular (dot product near zero)
        dot = abs(d1[0] * d2[0] + d1[1] * d2[1])
        if dot > AppConstants.RIGHT_ANGLE_PERPENDICULAR_TOLERANCE:
            return
        
        # Calculate the three points of the right angle marker
        p1 = (vertex[0] + d1[0] * size, vertex[1] + d1[1] * size)
        p2 = (vertex[0] + d1[0] * size + d2[0] * size, 
              vertex[1] + d1[1] * size + d2[1] * size)
        p3 = (vertex[0] + d2[0] * size, vertex[1] + d2[1] * size)
        
        ctx.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                    color=ctx.line_color, lw=AppConstants.DIMENSION_LINE_WIDTH)
        ctx.ax.plot([p2[0], p3[0]], [p2[1], p3[1]], 
                    color=ctx.line_color, lw=AppConstants.DIMENSION_LINE_WIDTH)
    
    @staticmethod
    def normalize_vector(v: tuple) -> tuple:
        """Normalize a 2D vector. Returns (0, 0) if zero-length."""
        length = math.sqrt(v[0]**2 + v[1]**2)
        if length == 0:
            return (0, 0)
        return (v[0] / length, v[1] / length)

    @staticmethod
    def draw_hash_marks(ctx: "DrawingContext", points: list["Point"],
                       hash_len: Optional[float] = None, count: int = 1) -> None:
        """Draw hash marks using consistent context styling."""
        if hash_len is None:
            hash_len = AppConstants.HASH_MARK_LENGTH
        
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]
            mid_x, mid_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            side_len = math.sqrt(dx**2 + dy**2)
            
            if side_len > 0:
                perp_x, perp_y = -dy / side_len, dx / side_len
                # Offset multiple marks so they don't overlap
                for c in range(count):
                    offset = (c - (count - 1) / 2) * 0.1
                    ox, oy = mid_x + dx/side_len * offset, mid_y + dy/side_len * offset
                    ctx.ax.plot(
                        [ox - perp_x * hash_len, ox + perp_x * hash_len],
                        [oy - perp_y * hash_len, oy + perp_y * hash_len],
                        color=ctx.line_color, lw=AppConstants.DEFAULT_LINE_WIDTH,
                        zorder=DrawingUtilities.Z_LINES
                    )
    

    
    @staticmethod
    def dim_offset_from_axes(ax, px: float = None) -> float:
        """Convert a fixed pixel distance to data-coordinate units using current axes state."""
        if px is None:
            px = AppConstants.PRESET_DIM_OFFSET_PX
        try:
            fig = ax.get_figure()
            fig_w = fig.get_figwidth() * fig.dpi
            fig_h = fig.get_figheight() * fig.dpi
            ax_pos = ax.get_position()
            ax_px_w = fig_w * ax_pos.width
            ax_px_h = fig_h * ax_pos.height
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            data_w = xlim[1] - xlim[0]
            data_h = ylim[1] - ylim[0]
            scale_x = data_w / ax_px_w if ax_px_w > 0 else 0.01
            scale_y = data_h / ax_px_h if ax_px_h > 0 else 0.01
            return px * (scale_x + scale_y) / 2
        except Exception:
            return 0.3

    @staticmethod
    def dim_label_gap_from_axes(ax) -> float:
        """Get the label gap (dim line to label text) in data units."""
        return DrawingUtilities.dim_offset_from_axes(ax, AppConstants.PRESET_DIM_LABEL_GAP_PX)


class SmartGeometryEngine:
    """Detects congruent sides and right angles for smart hashmark/marker drawing."""

    TOL_SIDE = 0.01       # Fractional tolerance for side-length equality
    TOL_ANGLE = 0.1       # Degrees tolerance for right angle detection

    # Shapes that have meaningful polygon vertices to analyze
    POLYGON_SHAPES = frozenset([
        "Rectangle", "Square", "Parallelogram", "Trapezoid",
        "Triangle",
        "Polygon",
    ])

    @staticmethod
    def _side_lengths(points: "Polygon") -> list[float]:
        n = len(points)
        lengths = []
        for i in range(n):
            p1 = points[i]
            p2 = points[(i + 1) % n]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            lengths.append(math.sqrt(dx * dx + dy * dy))
        return lengths

    @staticmethod
    def detect_congruence(points: "Polygon") -> dict[int, int]:
        """Return a mapping of side_index → group_id for equal-length sides.

        Groups are assigned in order of first appearance.
        Sides that are unique (no match) still get a group_id but the caller
        uses the group count to decide whether to draw incremental marks.
        Returns {} if points is empty or has < 3 vertices.
        """
        if not points or len(points) < 3:
            return {}
        lengths = SmartGeometryEngine._side_lengths(points)
        tol = SmartGeometryEngine.TOL_SIDE
        groups: dict[int, int] = {}
        next_group = 0
        ref_lengths: list[tuple[float, int]] = []  # (representative_length, group_id)
        for i, L in enumerate(lengths):
            matched = -1
            for ref_L, gid in ref_lengths:
                if ref_L > 0 and abs(L - ref_L) / ref_L <= tol:
                    matched = gid
                    break
            if matched >= 0:
                groups[i] = matched
            else:
                groups[i] = next_group
                ref_lengths.append((L, next_group))
                next_group += 1
        return groups

    @staticmethod
    def detect_right_angles(points: "Polygon") -> list[int]:
        """Return list of vertex indices where the interior angle is 90° ± TOL_ANGLE."""
        if not points or len(points) < 3:
            return []
        n = len(points)
        right_angle_verts = []
        for i in range(n):
            prev_pt = points[(i - 1) % n]
            curr_pt = points[i]
            next_pt = points[(i + 1) % n]
            v1 = (prev_pt[0] - curr_pt[0], prev_pt[1] - curr_pt[1])
            v2 = (next_pt[0] - curr_pt[0], next_pt[1] - curr_pt[1])
            len1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
            len2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
            if len1 < 1e-9 or len2 < 1e-9:
                continue
            cos_a = (v1[0] * v2[0] + v1[1] * v2[1]) / (len1 * len2)
            cos_a = max(-1.0, min(1.0, cos_a))
            angle_deg = math.degrees(math.acos(cos_a))
            if abs(angle_deg - 90.0) <= SmartGeometryEngine.TOL_ANGLE:
                right_angle_verts.append(i)
        return right_angle_verts

    @staticmethod
    def draw_smart_hashmarks(ctx: "DrawingContext", points: "Polygon") -> None:
        """Draw hashmarks and right-angle markers based on geometry analysis.

        Rules (matching F01/G10 spec):
        - Congruent side groups get I, II, III... tick marks.
        - If ALL sides are unique (all in different groups) AND no right angles → scalene
          increment marks (I, II, III... one per side to show inequality).
        - If a right angle is detected in a triangle, scalene marks are suppressed.
        - Right-angle markers drawn at 90° vertices whenever they are detected.
        """
        if not points or len(points) < 3:
            return

        n = len(points)
        groups = SmartGeometryEngine.detect_congruence(points)
        right_verts = SmartGeometryEngine.detect_right_angles(points)

        # Count how many sides are in each group
        group_counts: dict[int, int] = {}
        for gid in groups.values():
            group_counts[gid] = group_counts.get(gid, 0) + 1

        # Determine if any group has more than one side (true congruency)
        has_congruent = any(c > 1 for c in group_counts.values())
        all_unique = not has_congruent
        is_triangle = (n == 3)

        # Case A: draw congruent groups with I / II / III marks
        if has_congruent:
            # Map each group to a 1-based tick count (only groups with ≥2 sides)
            congruent_groups = sorted(
                set(gid for gid, cnt in group_counts.items() if cnt > 1)
            )
            group_to_ticks: dict[int, int] = {
                gid: idx + 1 for idx, gid in enumerate(congruent_groups)
            }
            for side_idx, gid in groups.items():
                if gid in group_to_ticks:
                    ticks = group_to_ticks[gid]
                    p1 = points[side_idx]
                    p2 = points[(side_idx + 1) % n]
                    DrawingUtilities.draw_hash_marks(ctx, [p1, p2], count=ticks)

        # Case B: all sides unique — incremental marks unless right angle in triangle
        elif all_unique:
            suppress = is_triangle and len(right_verts) > 0
            if not suppress:
                for side_idx in range(n):
                    p1 = points[side_idx]
                    p2 = points[(side_idx + 1) % n]
                    DrawingUtilities.draw_hash_marks(ctx, [p1, p2], count=side_idx + 1)

        # Draw right-angle markers at detected 90° vertices
        for vi in right_verts:
            vertex = points[vi]
            p_prev = points[(vi - 1) % n]
            p_next = points[(vi + 1) % n]
            d1 = (p_prev[0] - vertex[0], p_prev[1] - vertex[1])
            d2 = (p_next[0] - vertex[0], p_next[1] - vertex[1])
            DrawingUtilities.draw_right_angle_marker(ctx, vertex, d1, d2)


class TriangleType(Enum):
    CUSTOM = "Custom"
    ISOSCELES = "Isosceles"
    SCALENE = "Scalene"
    EQUILATERAL = "Equilateral"


class PolygonType(Enum):
    PENTAGON = "Pentagon"
    HEXAGON = "Hexagon"
    OCTAGON = "Octagon"


class AppConstants:
    # UI Settings
    WINDOW_TITLE: str = "Geometry Forge 4.8"
    
    DEFAULT_FONT_SIZE: int = 12
    MIN_FONT_SIZE: int = 6
    MAX_FONT_SIZE: int = 48
    DEFAULT_FONT_FAMILY: str = "serif"
    
    SLIDER_MIN: float = 0.1
    SLIDER_MAX: float = 1.9
    SLIDER_DEFAULT: float = 1.0
    
    ENTRY_WIDTH: int = 8
    
    DEFAULT_LINE_WIDTH: int = 2
    MIN_LINE_WIDTH: int = 1
    MAX_LINE_WIDTH: int = 6
    DIMENSION_LINE_WIDTH: int = 1
    PRESET_DIM_OFFSET_PX: float = 20.0  # Fixed pixel distance for all preset dim line offsets
    PRESET_DIM_LABEL_GAP_PX: float = 12.0  # Fixed pixel distance from dim line to its label
    HASH_MARK_LENGTH: float = 0.125
    RIGHT_ANGLE_PERPENDICULAR_TOLERANCE: float = 0.05
    POINT_SIZE_LARGE: int = 10
    POINT_SIZE_SMALL: int = 8
    DIM_ENDPOINT_SIZE: int = 3
    ARROW_HEAD_WIDTH: float = 0.2
    
    BG_COLOR: str = "#f0f0f0"
    DISABLED_COLOR: str = "#d0d0d0"
    ACTIVE_BUTTON_COLOR: str = "#a0d0ff"
    DEFAULT_BUTTON_COLOR: str = "SystemButtonFace"
    
    # Drawing canvas paper background
    CANVAS_BG_COLOR: str = "#e8e8e8"
    PAPER_COLOR: str = "#ffffff"
    
    DEBOUNCE_DELAY: int = 300
    SLIDER_DEBOUNCE_DELAY: int = 100
    
    # Shape default dimensions (standardized: base unit x=5)
    # Square=x, Rect=2x*x, Triangle=equilateral x, Circle=diameter x
    RECT_DEFAULT_WIDTH: float = 10.0
    RECT_DEFAULT_HEIGHT: float = 5.0
    SQUARE_DEFAULT_SIDE: float = 5.0
    TRIANGLE_DEFAULT_BASE: float = 5.0
    TRIANGLE_DEFAULT_HEIGHT: float = 4.33
    CIRCLE_DEFAULT_RADIUS: float = 2.5
    TRAPEZOID_DEFAULT_HEIGHT: float = 4.0
    TRAPEZOID_DEFAULT_BOTTOM: float = 7.5
    TRAPEZOID_DEFAULT_TOP: float = 3.75
    PARALLELOGRAM_DEFAULT_BASE: float = 7.5
    PARALLELOGRAM_DEFAULT_HEIGHT: float = 4.0
    PARALLELOGRAM_DEFAULT_SLOPE: float = 1.0
    POLYGON_DEFAULT_RADIUS: float = 2.5
    CYLINDER_DEFAULT_RADIUS: float = 3.0
    CYLINDER_DEFAULT_HEIGHT: float = 6.0
    PRISM_DEFAULT_DEPTH_SCALE: float = 0.4
    PRISM_DEFAULT_LENGTH: float = 5.0
    PRISM_DEFAULT_WIDTH: float = 4.0
    PRISM_DEFAULT_HEIGHT: float = 4.0
    TRI_PRISM_DEFAULT_BASE: float = 5.0
    TRI_PRISM_DEFAULT_HEIGHT: float = 4.0
    TRI_PRISM_DEFAULT_LENGTH: float = 4.0
    ISOSCELES_DEFAULT_BASE: float = 5.0
    ISOSCELES_DEFAULT_HEIGHT: float = 4.33
    SCALENE_DEFAULT_BASE: float = 7.0
    EQUILATERAL_DEFAULT_SIDE: float = 5.0
    
    # 3D projection ratios
    PRISM_DEPTH_X_RATIO: float = 0.4      # Horizontal offset for 3D depth effect (shared by all prism types)
    PRISM_DEPTH_Y_RATIO: float = 0.25     # Vertical offset for 3D depth effect (shared by all prism types)
    BASE_ELLIPSE_RATIO: float = 0.6         # Minor/Major axis ratio for 3D base ellipses
    RADIAL_LABEL_ELLIPSE_RATIO: float = 0.2        # Ratio for positioning radial labels within the base gap
    

    # Label positioning - centralized offset configuration
    # These values control how far labels appear from shapes (in coordinate units)
    # Smaller = closer to shape, Larger = farther from shape
    # Recommended range: 0.1 to 0.3 for most cases
    
    LABEL_OFFSET_SIDE: float = 0.3          # Standard side offset
    LABEL_OFFSET_DIMENSION: float = 0.4     # Offset for height/slant lines
    LABEL_OFFSET_RADIAL: float = 0.35       # Further centered within the 3D base gap
    LABEL_OFFSET_PRISM: float = 0.45        # Standard 3D offset
    SMART_LABEL_BUFFER: float = 0.3         # Standard smart buffer
    
    # Label appearance
    LABEL_USE_BACKGROUND: bool = True  # Add white background to prevent line clipping
    LABEL_BBOX_PAD: float = 0.15       # Matplotlib bbox padding for label backgrounds
    LABEL_ZORDER: int = 15             # Labels always on top (above dim lines at 12, dots at 13)
    
    # Composite snap settings
    SNAP_THRESHOLD: float = 1.8        # Data-coordinate distance to trigger snap
    SNAP_LINE_COLOR: str = "#4488ff"   # Blue guide lines
    SNAP_LINE_STYLE: str = "--"
    SNAP_LINE_WIDTH: float = 0.8
    SNAP_LINE_ALPHA: float = 0.6
    
    # Canvas layout
    PAPER_ASPECT_RATIO: float = 4.0 / 3.0  # Canvas drawing area aspect ratio

    # All built-in dimension label keys that support selection, drag, and double-click edit.
    # Used in _on_canvas_press to distinguish built-in labels from freeform annotations.
    BUILTIN_LABEL_KEYS: frozenset[str] = frozenset({
        "Circumference", "Radius", "Diameter", "Height", "Slant",
        "Length (Front)", "Width (Side)",
        "Base (Tri)", "Height (Tri)", "Length (Prism)",
    })


class ShapeFeature:
    """Feature key constants for ShapeConfig.features dicts."""
    FLIP = "reflect"
    ROTATE = "rotate"
    HASH_MARKS = "hash_marks"       # Only implemented for regular polygons (Pentagon, Hexagon, Octagon)
    SLIDER_SHAPE = "slider_shape"   # "Adjust Shape" slider (aspect_var)
    SLIDER_SLOPE = "slider_slope"   # "Adjust Slope" slider (parallelogram_slope_var)
    SLIDER_PEAK = "slider_peak"     # "Peak Offset" slider (peak_offset_var)



@dataclass
class ShapeConfig:
    """Configuration for a single shape."""
    labels: list[str] = field(default_factory=list)
    default_values: list[str] = field(default_factory=list)
    custom_values: list[str] = field(default_factory=list)
    custom_labels: list[str] = field(default_factory=list)
    features: dict[str, bool] = field(default_factory=dict)
    num_sides: int = 0
    rotation_labels: list[str] = field(default_factory=list)
    has_dimension_mode: bool = False
    uses_base_side_flip: bool = True  # False for shapes that don't follow the base_side orientation convention (e.g. prisms, Tri Triangle)
    help_text: str = ""
    
    def get_defaults_for_mode(self, mode: str) -> list[str]:
        """Get default values for the specified mode."""
        if mode == "Custom" and self.custom_values:
            return self.custom_values
        return self.default_values
    
    def has_feature(self, feature: str) -> bool:
        """Check if shape has a specific feature. Use ShapeFeature constants."""
        return self.features.get(feature, False)

    # Named helpers delegate to has_feature — single point of truth.
    def has_flip(self) -> bool:
        return self.has_feature(ShapeFeature.FLIP)

    def has_rotate(self) -> bool:
        return self.has_feature(ShapeFeature.ROTATE)

    def has_hash_marks(self) -> bool:
        return self.has_feature(ShapeFeature.HASH_MARKS)

    def has_slider_shape(self) -> bool:
        return self.has_feature(ShapeFeature.SLIDER_SHAPE)

    def has_slider_slope(self) -> bool:
        return self.has_feature(ShapeFeature.SLIDER_SLOPE)

    def has_slider_peak(self) -> bool:
        return self.has_feature(ShapeFeature.SLIDER_PEAK)


class ShapeConfigProvider:
    """Single source of truth for all shape configurations."""
    
    _configs: dict[str, ShapeConfig] = {}
    _initialized: bool = False
    
    @classmethod
    def _ensure_initialized(cls) -> None:
        if cls._initialized:
            return
        cls._init_2d_shapes()
        cls._init_3d_shapes()
        cls._init_angle_shapes()
        cls._init_composite_shapes()
        cls._initialized = True
    
    @classmethod
    def _init_2d_shapes(cls) -> None:
        cls._configs["Rectangle"] = ShapeConfig(
            labels=["Top", "Bottom", "Left", "Right"],
            default_values=["a", "a", "b", "b"],
            custom_values=["10", "5"],
            custom_labels=["Length", "Width"],
            features={ShapeFeature.SLIDER_SHAPE: True, ShapeFeature.FLIP: True, ShapeFeature.ROTATE: True},
            num_sides=4,
            has_dimension_mode=True,
            help_text="Default: Clean shape, add labels with presets.\nCustom: Enter Length and Width values."
        )
        cls._configs["Square"] = ShapeConfig(
            labels=["Top", "Bottom", "Left", "Right"],
            default_values=["a", "a", "a", "a"],
            features={ShapeFeature.FLIP: False, ShapeFeature.ROTATE: False},
            has_dimension_mode=False,
            help_text="All sides equal. Labels only mode."
        )
        cls._configs["Circle"] = ShapeConfig(
            labels=["Radius", "Diameter", "Circumference"],
            default_values=["r", "", ""],
            features={ShapeFeature.FLIP: False, ShapeFeature.ROTATE: False},
            has_dimension_mode=False,
            help_text="Defined by radius or diameter. Leave unused fields blank."
        )
        cls._configs["Parallelogram"] = ShapeConfig(
            labels=["Top", "Bottom", "Left Side", "Right Side", "Height"],
            default_values=["a", "a", "b", "b", "h"],
            custom_values=["6", "4", ""],
            custom_labels=["Length", "Height", "Side"],
            features={ShapeFeature.SLIDER_SHAPE: True, ShapeFeature.SLIDER_SLOPE: True, ShapeFeature.FLIP: True, ShapeFeature.ROTATE: True},
            num_sides=4,
            has_dimension_mode=True,
            help_text="Enter Length + Height, OR Length + Side.\nCustom: Slope slider adjusts lean."
        )
        cls._configs["Trapezoid"] = ShapeConfig(
            labels=["Top", "Bottom", "Left Leg", "Right Leg", "Height"],
            default_values=["a", "b", "c", "d", "h"],
            custom_values=["3", "6", "4", "", ""],
            custom_labels=["Top Base", "Bottom Base", "Height", "Left Side", "Right Side"],
            features={ShapeFeature.SLIDER_SHAPE: True, ShapeFeature.FLIP: True, ShapeFeature.ROTATE: True},
            num_sides=4,
            has_dimension_mode=True,
            help_text="Enter Top + Bottom + Height, OR Top + Bottom + both Sides.\nCustom: Numeric values control exact dimensions."
        )
        cls._configs["Polygon"] = ShapeConfig(
            labels=["Side Length"],
            default_values=["a"],
            features={ShapeFeature.HASH_MARKS: True},
            has_dimension_mode=False,
            help_text="Regular polygon. Choose Pentagon, Hexagon, or Octagon using the type buttons."
        )
        # Triangle configurations by type
        _tri_base = dict(
            labels=["Base Width", "Height", "Left Label", "Right Label"],
            default_values=["b", "h", "c", "a"],
            features={ShapeFeature.FLIP: True, ShapeFeature.ROTATE: True},
            has_dimension_mode=False,
        )
        cls._configs["Triangle"] = ShapeConfig(
            **_tri_base,
            num_sides=3,
            help_text="Custom triangle defined by base and height."
        )
        # Composite-only triangle: identical draw to Triangle but num_sides=4 so
        # rotation steps are 90° (matching Rectangle/Parallelogram) instead of 120°.
        cls._configs["Tri Triangle"] = ShapeConfig(
            **_tri_base,
            num_sides=4,
            uses_base_side_flip=False,
            rotation_labels=["Base Down", "Left Side Down", "Base Up", "Right Side Down"],
            help_text="Triangle (composite). Rotates in 90° steps."
        )
        cls._configs["Triangle_Custom"] = ShapeConfig(
            labels=["Base Width", "Height", "Left Side", "Right Side"],
            default_values=["b", "h", "c", "a"],
            custom_values=["5", "4", "", ""],
            custom_labels=["Base Width", "Height", "Left Side", "Right Side"],
            features={ShapeFeature.FLIP: True, ShapeFeature.ROTATE: True},
            num_sides=3,
            has_dimension_mode=True,
            help_text="Enter Base + Height OR Base + both Sides (mutually exclusive)."
        )
        cls._configs["Triangle_Isosceles"] = ShapeConfig(
            labels=["Base Label", "Left Label", "Right Label", "Height"],
            default_values=["a", "b", "b", "h"],
            features={ShapeFeature.SLIDER_SHAPE: True, ShapeFeature.FLIP: True, ShapeFeature.ROTATE: True},
            num_sides=3,
            has_dimension_mode=False,
            help_text="Isosceles triangle with two equal sides. Slider adjusts proportions."
        )
        cls._configs["Triangle_Scalene"] = ShapeConfig(
            labels=["Side A (Bottom)", "Side B (Left)", "Side C (Right)", "Height"],
            default_values=["a", "b", "c", "h"],
            features={ShapeFeature.SLIDER_SHAPE: True, ShapeFeature.FLIP: True, ShapeFeature.ROTATE: True, ShapeFeature.SLIDER_PEAK: True},
            num_sides=3,
            has_dimension_mode=False,
            help_text="Scalene triangle with all different sides. Sliders adjust shape proportions."
        )
        cls._configs["Triangle_Equilateral"] = ShapeConfig(
            labels=["Side A (Bottom)", "Side B (Left)", "Side C (Right)"],
            default_values=["a", "a", "a"],
            features={ShapeFeature.FLIP: True, ShapeFeature.ROTATE: True},
            num_sides=3,
            has_dimension_mode=False,
            help_text="Equilateral triangle with all equal sides."
        )
    
    @classmethod
    def _init_3d_shapes(cls) -> None:
        cls._configs["Sphere"] = ShapeConfig(
            labels=["Radius", "Diameter", "Circumference"],
            default_values=["r", "", ""],
            has_dimension_mode=False,
            help_text="Defined by radius or diameter. Slider adjusts scale."
        )
        cls._configs["Hemisphere"] = ShapeConfig(
            labels=["Radius", "Diameter", "Circumference"],
            default_values=["r", "", ""],
            features={ShapeFeature.SLIDER_SHAPE: True, ShapeFeature.FLIP: True, ShapeFeature.ROTATE: True},
            num_sides=4,
            rotation_labels=[],
            has_dimension_mode=False,
            help_text="Defined by radius or diameter. Adjust Shape slider changes dome proportions (squat/tall)."
        )
        cls._configs["Cylinder"] = ShapeConfig(
            labels=["Radius", "Diameter", "Height", "Circumference"],
            default_values=["r", "", "h", ""],
            custom_values=["5", "", "5"],
            custom_labels=["Radius", "Diameter", "Height"],
            features={ShapeFeature.SLIDER_SHAPE: True, ShapeFeature.FLIP: False, ShapeFeature.ROTATE: True},
            num_sides=4,
            rotation_labels=[],
            has_dimension_mode=True,
            help_text="Default: Clean shape, add labels with presets.\nCustom: Enter Radius or Diameter + Height."
        )
        cls._configs["Cone"] = ShapeConfig(
            labels=["Radius", "Diameter", "Height", "Circumference"],
            default_values=["r", "", "h", ""],
            custom_values=["4", "", "9"],
            custom_labels=["Radius", "Diameter", "Height"],
            features={ShapeFeature.SLIDER_SHAPE: True, ShapeFeature.FLIP: True, ShapeFeature.ROTATE: True},
            num_sides=4,
            rotation_labels=[],
            has_dimension_mode=True,
            help_text="Default: Clean shape, add labels with presets.\nCustom: Enter Radius or Diameter + Height."
        )
        cls._configs["Rectangular Prism"] = ShapeConfig(
            labels=["Length (Front)", "Width (Side)", "Height"],
            default_values=["l", "w", "h"],
            custom_values=["5", "5", "10"],
            custom_labels=["Length (Front)", "Width (Side)", "Height"],
            features={ShapeFeature.SLIDER_SHAPE: True, ShapeFeature.FLIP: True, ShapeFeature.ROTATE: True},
            num_sides=4,
            uses_base_side_flip=False,
            rotation_labels=[],
            has_dimension_mode=True,
            help_text="Default: Clean shape, add labels with presets.\nCustom: Enter Length, Width, and Height."
        )
        _tri_prism_base = dict(
            labels=["Base (Tri)", "Height (Tri)", "Length (Prism)", "Left Side", "Right Side"],
            default_values=["b", "h", "l", "", ""],
            custom_values=["5", "4", "6", "", ""],
            custom_labels=["Base", "Height", "Length", "Left Side", "Right Side"],
            features={ShapeFeature.SLIDER_SHAPE: True, ShapeFeature.FLIP: True, ShapeFeature.ROTATE: True, ShapeFeature.SLIDER_PEAK: True},
            has_dimension_mode=True,
            help_text="Enter Base + Height OR Base + both Sides.\nCustom: Peak slider adjusts triangle apex."
        )
        cls._configs["Triangular Prism"] = ShapeConfig(
            **_tri_prism_base,
            num_sides=3,
            rotation_labels=["Base Down", "Left Side Down", "Base Up"],
        )
        cls._configs["Tri Prism"] = ShapeConfig(
            **_tri_prism_base,
            num_sides=4,
            uses_base_side_flip=False,
            rotation_labels=["Base Down", "Left Side Down", "Base Up", "Right Side Down"],
        )
    
    @classmethod
    def _init_angle_shapes(cls) -> None:
        cls._configs["Angle (Adjustable)"] = ShapeConfig(
            labels=["Angle Label"],
            default_values=["x°"],
            features={ShapeFeature.SLIDER_SHAPE: True, ShapeFeature.FLIP: True},
            has_dimension_mode=False,
            help_text="Adjustable angle. Slider controls angle size (5° to 350°)."
        )
        cls._configs["Parallel Lines & Transversal"] = ShapeConfig(
            labels=["Top Left", "Top Right", "Bottom Left", "Bottom Right"],
            default_values=["x°", "x°", "x°", "x°"],
            features={ShapeFeature.SLIDER_SHAPE: True, ShapeFeature.FLIP: True},
            has_dimension_mode=False,
            help_text="Parallel lines cut by transversal. Slider adjusts gap between parallel lines."
        )
        cls._configs["Complementary Angles"] = ShapeConfig(
            labels=["Angle 1", "Angle 2"],
            default_values=["x°", "x°"],
            features={ShapeFeature.SLIDER_SHAPE: True, ShapeFeature.FLIP: True},
            has_dimension_mode=False,
            help_text="Two angles that sum to 90°. Slider adjusts split between angles."
        )
        cls._configs["Supplementary Angles"] = ShapeConfig(
            labels=["Left Angle", "Right Angle"],
            default_values=["x°", "x°"],
            features={ShapeFeature.SLIDER_SHAPE: True, ShapeFeature.FLIP: True},
            has_dimension_mode=False,
            help_text="Two angles that sum to 180°. Slider adjusts split between angles."
        )
        cls._configs["Vertical Angles"] = ShapeConfig(
            labels=["Top", "Bottom", "Left", "Right"],
            default_values=["x°", "x°", "x°", "x°"],
            features={ShapeFeature.SLIDER_SHAPE: True},
            has_dimension_mode=False,
            help_text="Opposite angles formed by two intersecting lines. Slider adjusts line angle."
        )
        cls._configs["Line Segment"] = ShapeConfig(
            labels=["Point A", "Point B", "Point C", "Left Segment", "Right Segment"],
            default_values=["A", "B", "C", "", ""],
            features={ShapeFeature.SLIDER_SHAPE: True, ShapeFeature.FLIP: True},
            has_dimension_mode=False,
            help_text="Line segment with three labeled points. Slider moves point B along segment."
        )
    
    @classmethod
    def _init_composite_shapes(cls) -> None:
        cls._configs["2D Composite"] = ShapeConfig(
            labels=[],
            default_values=[],
            features={},
            has_dimension_mode=False,
            help_text="Add 2D shapes using the transfer list, then they appear on the canvas."
        )
        cls._configs["3D Composite"] = ShapeConfig(
            labels=[],
            default_values=[],
            features={},
            has_dimension_mode=False,
            help_text="Add 3D shapes using the transfer list, then they appear on the canvas."
        )
    
    @classmethod
    def get(cls, shape_name: str) -> ShapeConfig:
        """Get configuration for a shape. Returns empty config if not found."""
        cls._ensure_initialized()
        return cls._configs.get(shape_name, ShapeConfig())
    
    @classmethod
    def get_triangle_config(cls, triangle_type: str) -> ShapeConfig:
        """Get configuration for a specific triangle type."""
        cls._ensure_initialized()
        key = f"Triangle_{triangle_type}"
        if key in cls._configs:
            return cls._configs[key]
        return cls._configs.get("Triangle", ShapeConfig())
    
    @classmethod
    def has_dimension_mode(cls, shape_name: str) -> bool:
        """Check if shape supports dimension mode switching."""
        return cls.get(shape_name).has_dimension_mode
    


class GeometricRotation:
    """Computes geometric transforms for annotation tracking during rotation.
    
    2D polygon shapes (Category A) rotate their geometry when base_side changes.
    This class computes the matching transform so freeform labels and dimension
    lines can follow the shape.  3D orientation shapes (Category B) redraw with
    fundamentally different geometry per orientation, so annotations are NOT
    transformed for those.
    """
    
    # 2D polygon shapes where annotations should follow geometric rotation.
    # These all use rotate_polygon_to_base or equivalent vertex reordering
    # that rotates CW for direction=1.  Polygon is excluded (no ROTATE
    # feature, centered at origin, uses additive angle offset instead).
    GEOMETRIC_SHAPES: frozenset[str] = frozenset({
        "Rectangle", "Triangle", "Parallelogram", "Trapezoid",
        "Tri Triangle",  # composite-only 4-step triangle
    })
    
    # 3D shapes that use flat geometric rotation of the entire drawing
    # instead of per-orientation redrawing.  The drawer always draws at
    # base_side=0 (canonical), then rotate_axes_artists rotates all
    # matplotlib artists by the appropriate angle.
    GEOMETRIC_3D_SHAPES: frozenset[str] = frozenset({
        "Hemisphere", "Cylinder", "Cone", "Rectangular Prism",
        "Triangular Prism", "Tri Prism",
    })

    # Shapes whose geometry is purely Arc-based and radially symmetric around
    # their rotation axis.  For these, flip_h / flip_v is always equivalent to
    # a pure base_side step, so we can safely normalise the stored flip flags
    # into base_side before calling transform_artist_lists.  This avoids the
    # two-pass (rotate-then-flip) Arc angle bug (B30) where a flip applied
    # AFTER rotation produces the wrong theta/angle combination on Arc patches.
    ARC_SYMMETRIC_SHAPES: frozenset[str] = frozenset({
        "Hemisphere", "Cylinder", "Cone", "Sphere",
    })
    
    # Unified set: ALL shapes that use the post-draw artist transform pipeline.
    # Both 2D and 3D shapes are drawn at canonical orientation (base_side=0,
    # no flip), then rotate_axes_artists + flip_axes_artists transform all
    # matplotlib artists.  This ensures rotation and flip behave identically
    # for standalone and composite shapes — everything moves as a unit.
    ALL_GEOMETRIC_SHAPES: frozenset[str] = frozenset(
        # Note: bare class-body names (GEOMETRIC_SHAPES, GEOMETRIC_3D_SHAPES) are
        # valid here — Python evaluates the class body top-to-bottom so both sets
        # are already defined by this point.
        GEOMETRIC_SHAPES | GEOMETRIC_3D_SHAPES
    )
    
    # Map of shape name → geometry_hints key that stores display vertices.
    _VERTEX_HINT_KEYS: dict[str, str] = {
        "Rectangle": "rect_pts",
        "Triangle": "tri_pts",
        "Tri Triangle": "tri_pts",
        "Parallelogram": "para_pts",
        "Trapezoid": "trap_pts",
    }

    @staticmethod
    def rotate_point(px: float, py: float,
                     angle: float,
                     cx: float, cy: float) -> tuple[float, float]:
        """Rotate point (px, py) by *angle* radians around center (cx, cy)."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        dx = px - cx
        dy = py - cy
        return (cx + dx * cos_a - dy * sin_a,
                cy + dx * sin_a + dy * cos_a)

    @staticmethod
    def compute_angle_from_vertices(old_pts: list, new_pts: list,
                                     new_base_side: int) -> Optional[float]:
        """Compute the actual rotation angle from corresponding vertex pairs.
        
        After rotate_polygon_to_base(pts, new_base_side), vertex mapping is:
          old[new_base_side] → new[0]
          old[(new_base_side + 1) % n] → new[1]
        We use these two pairs to compute the rotation angle via atan2.
        Returns the angle in radians, or None if computation fails.
        """
        n_old = len(old_pts)
        n_new = len(new_pts)
        if n_old < 2 or n_new < 2:
            return None
        
        idx_a_old = new_base_side % n_old
        idx_b_old = (new_base_side + 1) % n_old
        
        old_a = old_pts[idx_a_old]
        old_b = old_pts[idx_b_old]
        new_a = new_pts[0]
        new_b = new_pts[1]
        
        # Compute the angle between the old edge vector and new edge vector
        old_dx = old_b[0] - old_a[0]
        old_dy = old_b[1] - old_a[1]
        new_dx = new_b[0] - new_a[0]
        new_dy = new_b[1] - new_a[1]
        
        old_angle = math.atan2(old_dy, old_dx)
        new_angle = math.atan2(new_dy, new_dx)
        
        return new_angle - old_angle

    @staticmethod
    def compute_rotation_center(old_pt: tuple, new_pt: tuple,
                                angle: float) -> Optional[tuple[float, float]]:
        """Compute the rotation center given that old_pt maps to new_pt under angle.

        Solves the 2×2 system:
            (Ax - cx)*(cosθ-1) - (Ay - cy)*sinθ = Ax' - Ax
            (Ax - cx)*sinθ + (Ay - cy)*(cosθ-1) = Ay' - Ay

        Returns (cx, cy) or None if the system is degenerate (angle ≈ 0 or 2π).
        """
        c1 = math.cos(angle) - 1.0
        s  = math.sin(angle)
        det = c1 * c1 + s * s          # = 2 - 2*cos(angle)
        if abs(det) < 1e-9:            # angle ≈ 0 — no unique center
            return None
        dx = new_pt[0] - old_pt[0]
        dy = new_pt[1] - old_pt[1]
        u = (dx * c1 + dy * s) / det
        v = (-dx * s + dy * c1) / det
        return (old_pt[0] - u, old_pt[1] - v)

    @classmethod
    def compute_transform(cls, shape: str, direction: int, num_sides: int,
                          new_base_side: int,
                          old_hints: dict, new_hints: dict,
                          old_center: tuple[float, float],
                          new_center: tuple[float, float]
                          ) -> tuple[float, tuple[float, float], tuple[float, float]]:
        """Compute (angle, pivot, pivot) for annotation transform.

        For 2D polygon shapes the vertices are redrawn in a new orientation
        rather than geometrically rotated, so we must derive both the rotation
        angle AND the true pivot from the vertex correspondence.

        Returns (angle_radians, pivot, pivot) — caller rotates around pivot
        with no additional shift.  For the fallback case the old bbox-center /
        new bbox-center pair is returned so the caller can also apply a shift.
        """
        hint_key = cls._VERTEX_HINT_KEYS.get(shape)
        if hint_key:
            old_pts = old_hints.get(hint_key)
            new_pts = new_hints.get(hint_key)
            if old_pts and new_pts:
                angle = cls.compute_angle_from_vertices(old_pts, new_pts, new_base_side)
                if angle is not None:
                    # Derive the true pivot from the vertex that moved
                    n_old = len(old_pts)
                    idx_a = new_base_side % n_old
                    old_a = old_pts[idx_a]
                    new_a = new_pts[0]
                    pivot = cls.compute_rotation_center(old_a, new_a, angle)
                    if pivot is not None:
                        return (angle, pivot, pivot)
                    # Degenerate (180° or 0°): fall through to shift-based path
                    return (angle, old_center, new_center)

        # Fallback for 3D shapes and shapes without vertex hints:
        # use step angle and bbox-center shift.
        angle = -direction * (2 * math.pi / num_sides) if num_sides > 0 else 0.0
        return (angle, old_center, new_center)

    @classmethod
    def transform_annotations(cls,
                              labels: list[dict],
                              dim_lines: list[dict],
                              angle: float,
                              old_center: tuple[float, float],
                              new_center: tuple[float, float]) -> None:
        """Transform annotation positions in-place after a rotation step.
        
        Each point is rotated by *angle* around *old_center*, then shifted so
        the old center maps to *new_center*.
        """
        shift_x = new_center[0] - old_center[0]
        shift_y = new_center[1] - old_center[1]
        
        for lbl in labels:
            rx, ry = cls.rotate_point(lbl["x"], lbl["y"], angle,
                                      old_center[0], old_center[1])
            lbl["x"] = rx + shift_x
            lbl["y"] = ry + shift_y
        
        for dim in dim_lines:
            for xk, yk in [("x1", "y1"), ("x2", "y2"), ("label_x", "label_y")]:
                if xk in dim and yk in dim:
                    rx, ry = cls.rotate_point(dim[xk], dim[yk], angle,
                                              old_center[0], old_center[1])
                    dim[xk] = rx + shift_x
                    dim[yk] = ry + shift_y

    @classmethod
    def flip_annotations(cls,
                         labels: list[dict],
                         dim_lines: list[dict],
                         flip_h: bool, flip_v: bool,
                         cx: float, cy: float) -> None:
        """Mirror annotation positions in-place after a flip.

        Applied in the current (post-rotation) coordinate space so the
        annotation mirror matches what the user sees.
        """
        def mirror(px, py):
            x = (2 * cx - px) if flip_h else px
            y = (2 * cy - py) if flip_v else py
            return x, y

        for lbl in labels:
            lbl["x"], lbl["y"] = mirror(lbl["x"], lbl["y"])

        for dim in dim_lines:
            for xk, yk in [("x1", "y1"), ("x2", "y2"), ("label_x", "label_y")]:
                if xk in dim and yk in dim:
                    dim[xk], dim[yk] = mirror(dim[xk], dim[yk])

    @classmethod
    def rotate_axes_artists(cls, ax, angle_deg: float, cx: float, cy: float) -> None:
        """Rotate all drawing artists on *ax* by *angle_deg* around (cx, cy).
        
        Handles:  Polygon  → rotate vertex xy data
                  Ellipse / Arc / Circle → rotate center + add angle offset
                  Line2D   → rotate xdata/ydata points
                  Text     → rotate position
                  FancyArrowPatch → rotate posA/posB endpoints
        
        Infrastructure artists (Spines, Axes background Rectangle) are skipped.
        """
        angle_rad = math.radians(angle_deg)
        
        # Collect axes infrastructure to skip
        skip = set()
        for spine in ax.spines.values():
            skip.add(id(spine))
        skip.add(id(ax.patch))  # axes background rectangle
        skip.add(id(ax.xaxis))
        skip.add(id(ax.yaxis))
        # Title / axis label Text objects
        skip.add(id(ax.title))
        if hasattr(ax, '_left_title'):
            skip.add(id(ax._left_title))
        if hasattr(ax, '_right_title'):
            skip.add(id(ax._right_title))
        
        for artist in list(ax.get_children()):
            if id(artist) in skip:
                continue
            
            cls_name = type(artist).__name__
            
            try:
                if cls_name == 'Polygon':
                    xy = artist.get_xy()
                    new_xy = [cls.rotate_point(x, y, angle_rad, cx, cy) for x, y in xy]
                    artist.set_xy(new_xy)
                
                elif cls_name in ('Ellipse', 'Circle'):
                    ecx, ecy = artist.center
                    artist.center = cls.rotate_point(ecx, ecy, angle_rad, cx, cy)
                    artist.angle = getattr(artist, 'angle', 0) + angle_deg
                
                elif cls_name == 'Arc':
                    ecx, ecy = artist.center
                    artist.center = cls.rotate_point(ecx, ecy, angle_rad, cx, cy)
                    artist.angle = getattr(artist, 'angle', 0) + angle_deg
                
                elif cls_name == 'Line2D':
                    xd = artist.get_xdata()
                    yd = artist.get_ydata()
                    pts = [cls.rotate_point(x, y, angle_rad, cx, cy)
                           for x, y in zip(xd, yd)]
                    if pts:
                        artist.set_xdata([p[0] for p in pts])
                        artist.set_ydata([p[1] for p in pts])
                
                elif cls_name == 'Text':
                    # Skip text that uses axes-fraction transforms (e.g. error msgs)
                    if artist.get_transform() != ax.transData:
                        if not hasattr(artist, '_original_position'):
                            continue
                    px, py = artist.get_position()
                    artist.set_position(cls.rotate_point(px, py, angle_rad, cx, cy))
                
                elif cls_name == 'FancyArrowPatch':
                    posA, posB = artist._posA_posB
                    artist._posA_posB = [
                        cls.rotate_point(posA[0], posA[1], angle_rad, cx, cy),
                        cls.rotate_point(posB[0], posB[1], angle_rad, cx, cy),
                    ]
                
                elif cls_name == 'FancyBboxPatch':
                    # Text bbox backgrounds — they follow their parent Text
                    pass
                
                elif cls_name == 'Rectangle':
                    # Axes background rectangle — skip
                    pass
                    
            except Exception:
                logger.debug("Could not rotate artist %s", cls_name)

    @classmethod
    def _transform_artist(cls, artist, angle_rad: float = 0.0,
                          flip_h: bool = False, flip_v: bool = False,
                          cx: float = 0.0, cy: float = 0.0) -> None:
        """Apply rotate and/or flip to a single artist around (cx, cy).
        Rotation is applied first, then flip (if any).
        Used by rotate_artist_list / flip_artist_list for targeted transforms.
        """
        def rp(px, py):
            if angle_rad != 0.0:
                px, py = cls.rotate_point(px, py, angle_rad, cx, cy)
            if flip_h:
                px = 2 * cx - px
            if flip_v:
                py = 2 * cy - py
            return px, py

        cls_name = type(artist).__name__
        try:
            if cls_name == 'Polygon':
                xy = artist.get_xy()
                artist.set_xy([rp(x, y) for x, y in xy])

            elif cls_name in ('Ellipse', 'Circle'):
                ecx, ecy = artist.center
                artist.center = rp(ecx, ecy)
                if angle_rad != 0.0:
                    artist.angle = getattr(artist, 'angle', 0) + math.degrees(angle_rad)
                if flip_h ^ flip_v:
                    artist.angle = -getattr(artist, 'angle', 0)

            elif cls_name == 'Arc':
                ecx, ecy = artist.center
                artist.center = rp(ecx, ecy)
                # Rotation: only update artist.angle (the patch's tilt in its own
                # coordinate frame), exactly like rotate_axes_artists does.
                # Do NOT modify theta1/theta2 — those define the arc span in the
                # patch's local frame and must stay fixed during rotation, or the
                # visible arc segment jumps to a wrong position (hemisphere "explodes").
                a1, a2 = artist.theta1, artist.theta2
                if angle_rad != 0.0:
                    ad = math.degrees(angle_rad)
                    artist.angle = getattr(artist, 'angle', 0) + ad
                    # a1, a2 are intentionally NOT updated here
                if flip_h and not flip_v:
                    artist.theta1 = 180 - a2
                    artist.theta2 = 180 - a1
                    artist.angle = -getattr(artist, 'angle', 0)
                elif flip_v and not flip_h:
                    artist.theta1 = -a2
                    artist.theta2 = -a1
                    artist.angle = -getattr(artist, 'angle', 0)
                elif flip_h and flip_v:
                    artist.theta1 = a1 + 180
                    artist.theta2 = a2 + 180

            elif cls_name == 'Line2D':
                xd = artist.get_xdata()
                yd = artist.get_ydata()
                pts = [rp(x, y) for x, y in zip(xd, yd)]
                if pts:
                    artist.set_xdata([p[0] for p in pts])
                    artist.set_ydata([p[1] for p in pts])

            elif cls_name == 'Text':
                px, py = artist.get_position()
                artist.set_position(rp(px, py))

            elif cls_name == 'FancyArrowPatch':
                posA, posB = artist._posA_posB
                artist._posA_posB = [
                    rp(posA[0], posA[1]),
                    rp(posB[0], posB[1]),
                ]
        except Exception:
            logger.debug("Could not transform artist %s", cls_name)

    @classmethod
    def transform_artist_lists(cls, patches: list, lines: list, texts: list,
                               angle_deg: float = 0.0,
                               flip_h: bool = False, flip_v: bool = False,
                               cx: float = 0.0, cy: float = 0.0) -> None:
        """Apply rotate-then-flip to explicit artist lists (composite mode).
        Rotation is applied first at angle_deg, then flips in the rotated space.
        cx, cy should be the center of the shape's canonical bounding box.
        """
        if angle_deg == 0.0 and not flip_h and not flip_v:
            return
        angle_rad = math.radians(angle_deg)

        # Step 1: rotation only
        if angle_deg != 0.0:
            for artist in patches + lines + texts:
                cls._transform_artist(artist, angle_rad=angle_rad, cx=cx, cy=cy)
            # Recompute center after rotation for the flip step
            xs, ys = [], []
            for p in patches:
                try:
                    xy = p.get_xy()
                    xs.extend(v[0] for v in xy)
                    ys.extend(v[1] for v in xy)
                except Exception:
                    pass
            for l in lines:
                try:
                    xs.extend(l.get_xdata())
                    ys.extend(l.get_ydata())
                except Exception:
                    pass
            if xs and ys:
                cx = (min(xs) + max(xs)) / 2
                cy = (min(ys) + max(ys)) / 2

        # Step 2: flip in current (post-rotation) space
        if flip_h or flip_v:
            for artist in patches + lines + texts:
                cls._transform_artist(artist, flip_h=flip_h, flip_v=flip_v, cx=cx, cy=cy)

    @classmethod
    def recalculate_limits(cls, ax, padding: float = 0.5) -> None:
        """Recalculate axes limits from all artist bounding boxes after rotation.
        
        Collects all data-coordinate extents and sets xlim/ylim to contain
        everything with a small padding.
        """
        x_vals = []
        y_vals = []
        
        for artist in ax.get_children():
            cls_name = type(artist).__name__
            try:
                if cls_name == 'Polygon':
                    xy = artist.get_xy()
                    x_vals.extend(p[0] for p in xy)
                    y_vals.extend(p[1] for p in xy)
                elif cls_name in ('Ellipse', 'Circle', 'Arc'):
                    ecx, ecy = artist.center
                    # Use the bounding box of the rotated ellipse
                    w = artist.width / 2
                    h = artist.height / 2
                    ang = math.radians(getattr(artist, 'angle', 0))
                    # Rotated ellipse extents
                    cos_a = math.cos(ang)
                    sin_a = math.sin(ang)
                    dx = math.sqrt((w * cos_a)**2 + (h * sin_a)**2)
                    dy = math.sqrt((w * sin_a)**2 + (h * cos_a)**2)
                    x_vals.extend([ecx - dx, ecx + dx])
                    y_vals.extend([ecy - dy, ecy + dy])
                elif cls_name == 'Line2D':
                    xd = list(artist.get_xdata())
                    yd = list(artist.get_ydata())
                    x_vals.extend(xd)
                    y_vals.extend(yd)
                elif cls_name == 'Text':
                    px, py = artist.get_position()
                    if artist.get_text().strip():
                        x_vals.append(px)
                        y_vals.append(py)
            except Exception:
                pass
        
        if x_vals and y_vals:
            ax.set_xlim(min(x_vals) - padding, max(x_vals) + padding)
            ax.set_ylim(min(y_vals) - padding, max(y_vals) + padding)

    @classmethod
    def flip_axes_artists(cls, ax, flip_h: bool, flip_v: bool,
                          cx: float, cy: float) -> None:
        """Mirror all drawing artists on *ax* around (cx, cy).

        Applied AFTER rotate_axes_artists so flips operate in the current
        (already-rotated) screen orientation rather than canonical space.

        Handles: Polygon, Ellipse/Circle/Arc, Line2D, Text, FancyArrowPatch.
        Infrastructure artists (spines, axes background) are skipped.
        """
        if not flip_h and not flip_v:
            return

        skip = set()
        for spine in ax.spines.values():
            skip.add(id(spine))
        skip.add(id(ax.patch))
        skip.add(id(ax.xaxis))
        skip.add(id(ax.yaxis))
        skip.add(id(ax.title))
        if hasattr(ax, '_left_title'):
            skip.add(id(ax._left_title))
        if hasattr(ax, '_right_title'):
            skip.add(id(ax._right_title))

        def mirror(px, py):
            x = (2 * cx - px) if flip_h else px
            y = (2 * cy - py) if flip_v else py
            return x, y

        for artist in list(ax.get_children()):
            if id(artist) in skip:
                continue
            cls_name = type(artist).__name__
            try:
                if cls_name == 'Polygon':
                    xy = artist.get_xy()
                    artist.set_xy([mirror(x, y) for x, y in xy])

                elif cls_name in ('Ellipse', 'Circle'):
                    ecx, ecy = artist.center
                    artist.center = mirror(ecx, ecy)
                    # Flip the tilt angle: H-flip negates the angle, V-flip also negates it
                    if flip_h ^ flip_v:
                        artist.angle = -getattr(artist, 'angle', 0)

                elif cls_name == 'Arc':
                    ecx, ecy = artist.center
                    artist.center = mirror(ecx, ecy)
                    # Mirror arc angles around horizontal or vertical axis
                    a1 = artist.theta1
                    a2 = artist.theta2
                    if flip_h and not flip_v:
                        artist.theta1 = 180 - a2
                        artist.theta2 = 180 - a1
                        artist.angle = -getattr(artist, 'angle', 0)
                    elif flip_v and not flip_h:
                        artist.theta1 = -a2
                        artist.theta2 = -a1
                        artist.angle = -getattr(artist, 'angle', 0)
                    elif flip_h and flip_v:
                        artist.theta1 = a1 + 180
                        artist.theta2 = a2 + 180

                elif cls_name == 'Line2D':
                    xd = artist.get_xdata()
                    yd = artist.get_ydata()
                    pts = [mirror(x, y) for x, y in zip(xd, yd)]
                    if pts:
                        artist.set_xdata([p[0] for p in pts])
                        artist.set_ydata([p[1] for p in pts])

                elif cls_name == 'Text':
                    if artist.get_transform() != ax.transData:
                        if not hasattr(artist, '_original_position'):
                            continue
                    px, py = artist.get_position()
                    artist.set_position(mirror(px, py))

                elif cls_name == 'FancyArrowPatch':
                    posA, posB = artist._posA_posB
                    artist._posA_posB = [
                        mirror(posA[0], posA[1]),
                        mirror(posB[0], posB[1]),
                    ]

                elif cls_name in ('FancyBboxPatch', 'Rectangle'):
                    pass  # follow parent Text / skip background

            except Exception:
                logger.debug("Could not flip artist %s", cls_name)



@dataclass
class DrawingContext:
    """Holds common drawing parameters."""
    ax: Axes
    aspect_ratio: float
    font_size: int
    view_scale: float = 1.0
    composite_mode: bool = False  # True when drawing inside _draw_composite_shapes
    line_color: str = field(init=False, default="black")
    line_args: dict = field(init=False)
    dash_args: dict = field(init=False)
    dim_args: dict = field(init=False)
    
    def __post_init__(self):
        self.line_color = "black"
        self.line_args = {
            "edgecolor": "black",
            "facecolor": "none",
            "linewidth": AppConstants.DEFAULT_LINE_WIDTH
        }
        self.dash_args = {
            "color": "black",
            "linewidth": AppConstants.DEFAULT_LINE_WIDTH,
            "linestyle": "--"
        }
        self.dim_args = {
            "color": "black",
            "linewidth": AppConstants.DIMENSION_LINE_WIDTH
        }



@dataclass
class TransformState:
    """Holds current transformation state for shape drawing."""
    flip_h: bool = False
    flip_v: bool = False
    base_side: int = 0



@dataclass
class DrawingDependencies:
    """Dependencies required for shape drawing, passed explicitly to drawers."""
    ax: Axes
    label_manager: LabelManager
    font_size: int


class LabelManager:
    """Manages label positioning, custom positions, and label rendering."""
    def __init__(self):
        self.custom_positions: dict[str, tuple[float, float]] = {}
        self.auto_positions: dict[str, tuple[float, float, dict]] = {}
        self.label_texts: dict[str, str] = {}
        self.label_visibility: dict[str, bool] = {}
        # Geometry hints from shape drawers for preset dim line placement
        self.geometry_hints: dict[str, Any] = {}
        # Built-in label selection state (for highlight color)
        self.circ_selected: bool = False
        self.builtin_selected: Optional[str] = None
        # User-dragged offsets for built-in dim lines {key: (dx, dy)}
        self.custom_dim_offsets: dict[str, tuple[float, float]] = {}
    
    def set_label_text(self, key: str, text: str, visible: bool = True) -> None:
        """Set label text and visibility directly (bypasses entries)."""
        self.label_texts[key] = text
        self.label_visibility[key] = visible
    
    def clear_label_texts(self) -> None:
        """Clear all directly-set label texts."""
        self.label_texts.clear()
        self.label_visibility.clear()
        self.geometry_hints.clear()
    
    def clear_positions(self) -> None:
        self.custom_positions.clear()
        self.auto_positions.clear()
    
    def clear_auto_positions(self) -> None:
        self.auto_positions.clear()
    
    def set_custom_position(self, label: str, x: float, y: float) -> None:
        self.custom_positions[label] = (x, y)
    
    def reset_custom_position(self, label: str) -> None:
        if label in self.custom_positions:
            del self.custom_positions[label]
    
    def reset_all_custom_positions(self) -> None:
        self.custom_positions.clear()
        self.custom_dim_offsets.clear()
    
    def has_custom_position(self, label: str) -> bool:
        return label in self.custom_positions
    
    def get_entry_values(self, key: str) -> tuple[str, bool]:
        """Get text and show values from label_texts only."""
        if key in self.label_texts:
            text = self.label_texts[key]
            show = self.label_visibility.get(key, True)
            return text, show
        return "", False

    def draw_label(self, ax: Axes, x: float, y: float, key: str, font_size: int,
                   use_background: bool = True,
                   **kwargs) -> None:
        """Draw a label, using custom position if set."""
        text, show = self.get_entry_values(key)
        if not (text and show):
            return

        self.auto_positions[key] = (x, y, kwargs.copy())
        if key in self.custom_positions:
            x, y = self.custom_positions[key]
            kwargs['ha'] = 'center'
            kwargs['va'] = 'center'

        # Built-in label selection highlight
        is_circ_sel = (key == getattr(self, 'builtin_selected', None) and key is not None)

        should_use_bg = use_background and AppConstants.LABEL_USE_BACKGROUND
        kwargs['zorder'] = AppConstants.LABEL_ZORDER
        kwargs.setdefault('fontfamily', getattr(self, 'font_family', AppConstants.DEFAULT_FONT_FAMILY))

        if is_circ_sel:
            bbox_props = dict(
                boxstyle=f"round,pad={AppConstants.LABEL_BBOX_PAD}",
                facecolor='white',
                edgecolor='#0066cc',
                alpha=1
            )
            ax.text(x, y, text, fontsize=font_size, color="#0066cc",
                    fontweight="bold", bbox=bbox_props, **kwargs)
        elif should_use_bg:
            bbox_props = dict(
                boxstyle=f"round,pad={AppConstants.LABEL_BBOX_PAD}",
                facecolor='white',
                edgecolor='none',
                alpha=1
            )
            ax.text(x, y, text, fontsize=font_size, bbox=bbox_props, **kwargs)
        else:
            ax.text(x, y, text, fontsize=font_size, **kwargs)
    
    def get_smart_label_pos(self, p1: Point, p2: Point, centroid: Point, 
                            buffer: float = None) -> tuple[float, float, str, str]:
        """Calculate smart label position for a polygon side.
        
        Uses AppConstants.SMART_LABEL_BUFFER if buffer is None.
        """
        if buffer is None:
            buffer = AppConstants.SMART_LABEL_BUFFER
        
        mx = (p1[0] + p2[0]) / 2
        my = (p1[1] + p2[1]) / 2
        vx = mx - centroid[0]
        vy = my - centroid[1]
        length = math.sqrt(vx * vx + vy * vy)
        
        if length == 0:
            return mx, my, "center", "center"
        
        ux = vx / length
        uy = vy / length
        nx = mx + (ux * buffer)
        ny = my + (uy * buffer)
        ha = "left" if ux > 0.2 else "right" if ux < -0.2 else "center"
        va = "bottom" if uy > 0.2 else "top" if uy < -0.2 else "center"
        
        return nx, ny, ha, va


class PolygonLabelMixin:
    """Mixin for shapes with polygon side labels."""

    def draw_smart_labels(self, points: Polygon, label_keys: list[str], ctx: DrawingContext) -> None:
        """Draw labels using smart positioning relative to centroid."""
        if not points or len(points) < 2: return
        centroid = self.calculate_centroid(points)
        for i, key in enumerate(label_keys):
            p1, p2 = points[i], points[(i + 1) % len(points)]
            nx, ny, ha, va = self.get_smart_label_pos(p1, p2, centroid)
            self.draw_text(nx, ny, key, ha=ha, va=va)
    
    def draw_side_labels(self, points: Polygon, label_keys: list[str],
                         side_aligns: list[tuple[str, str]], 
                         offsets: Optional[list[float]] = None) -> None:
        """Draw labels for each side of a polygon."""
        if offsets is None:
            offsets = [AppConstants.LABEL_OFFSET_SIDE] * len(label_keys)
        n = len(points)
        for i, key in enumerate(label_keys):
            p1 = points[i]
            p2 = points[(i + 1) % n]
            mx = (p1[0] + p2[0]) / 2
            my = (p1[1] + p2[1]) / 2
            ha, va = side_aligns[i % len(side_aligns)]
            offset = offsets[i % len(offsets)]
            if va == "top":
                my -= offset
            elif va == "bottom":
                my += offset
            if ha == "left":
                mx += offset
            elif ha == "right":
                mx -= offset
            self.draw_text(mx, my, key, ha=ha, va=va)


class RadialLabelMixin:
    """Mixin for shapes with radius/diameter labels."""
    
    def draw_circumference_arc(self, ctx: DrawingContext, center: Point,
                                radius: float, orientation: str = "horizontal",
                                label_offset: float = None,
                                ellipse_ratio: float = None,
                                base_position: str = "bottom") -> None:
        """Draw circumference indicator arc with arrowheads pointing away.

        All arc segments and arrowheads use ctx.line_color for consistency.

        Args:
            ctx: Drawing context
            center: Center point of the circle/ellipse
            radius: Radius of the circle (or semi-major axis for ellipse)
            orientation: "horizontal" for upright shapes, "vertical" for shapes on side
            label_offset: Offset for label positioning
            ellipse_ratio: For 3D bases, the ratio of minor/major axis (e.g., 0.3 for foreshortened circle)
            base_position: Where the circular base is located.
                           horizontal orientation: "bottom" or "top"
                           vertical orientation:   "left" or "right"
        """
        if label_offset is None:
            label_offset = AppConstants.LABEL_OFFSET_RADIAL

        valid_positions = {"bottom", "top", "left", "right"}
        if base_position not in valid_positions:
            logger.warning("draw_circumference_arc: unrecognized base_position %r, defaulting to 'bottom'", base_position)
            base_position = "bottom"

        txt_c, show_c = self.label_manager.get_entry_values("Circumference")
        if not (txt_c.strip() and show_c):
            return
        
        arc_color = "#0066cc" if self.label_manager.circ_selected else ctx.line_color
        arc_lw = AppConstants.DIMENSION_LINE_WIDTH * 1.5 if self.label_manager.circ_selected else AppConstants.DIMENSION_LINE_WIDTH
        
        arrow_size = min(0.15, radius * 0.06)  # Scale arrow size with radius
        
        if orientation == "horizontal" and ellipse_ratio is None:
            # True circle (2D Circle or Sphere) - full 270° arc outside the shape
            arc_radius = radius + label_offset
            start_angle = 45
            end_angle = 315
            
            # Draw the arc (explicitly using ctx.line_color)
            ctx.ax.add_patch(patches.Arc(
                center, arc_radius * 2, arc_radius * 2,
                theta1=start_angle, theta2=end_angle,
                color=arc_color, lw=arc_lw,
                linestyle="--"
            ))
            
            # Calculate arrowhead positions and directions
            start_rad = np.radians(start_angle)
            start_pt = (center[0] + arc_radius * np.cos(start_rad),
                       center[1] + arc_radius * np.sin(start_rad))
            # Arrow points into gap (CW direction at start)
            start_tangent = (np.sin(start_rad), -np.cos(start_rad))
            
            end_rad = np.radians(end_angle)
            end_pt = (center[0] + arc_radius * np.cos(end_rad),
                     center[1] + arc_radius * np.sin(end_rad))
            # Arrow points into gap (CCW direction at end)
            end_tangent = (-np.sin(end_rad), np.cos(end_rad))
            
            self._draw_arrowhead(ctx, start_pt, start_tangent, arrow_size, centered=True, color=arc_color)
            self._draw_arrowhead(ctx, end_pt, end_tangent, arrow_size, centered=True, color=arc_color)
            
            # Label offset to the left of the arc
            label_x = center[0] - arc_radius - label_offset * 0.3
            label_y = center[1]
            self.draw_text(label_x, label_y, "Circumference", 
                          use_background=True, ha="right", va="center")
        
        elif orientation == "horizontal" and ellipse_ratio is not None:
            # Elliptical base (3D shapes) - arc on opposite side from shape body
            gap = radius * 0.15
            base_width = radius * 2
            base_height = radius * 2 * ellipse_ratio
            # Arc wraps AROUND the base - add gap to both dimensions
            arc_width = base_width + (gap * 2)
            arc_height = base_height + (gap * 2)
            back_arc_size = gap * 2
            small_arrow = min(0.1, radius * 0.04)
            
            if base_position == "bottom":
                # Arc below the base, curving up at ends
                arc_center_y = center[1] - gap
                
                # Draw bottom half of ellipse (180° to 360°) - using ctx.line_color
                ctx.ax.add_patch(patches.Arc(
                    (center[0], arc_center_y), arc_width, arc_height,
                    theta1=180, theta2=360,
                    color=arc_color, lw=arc_lw,
                    linestyle="--"
                ))
                
                half_width = arc_width / 2
                left_x = center[0] - half_width
                right_x = center[0] + half_width
                
                # Back continuation arcs curving up - using ctx.line_color
                ctx.ax.add_patch(patches.Arc(
                    (left_x + gap, arc_center_y), back_arc_size, back_arc_size,
                    theta1=135, theta2=180,
                    color=arc_color, lw=arc_lw,
                    linestyle="--"
                ))
                ctx.ax.add_patch(patches.Arc(
                    (right_x - gap, arc_center_y), back_arc_size, back_arc_size,
                    theta1=0, theta2=45,
                    color=arc_color, lw=arc_lw,
                    linestyle="--"
                ))
                
                # Arrowheads
                left_arrow_angle = np.radians(130)
                left_arrow_x = left_x + gap + gap * np.cos(left_arrow_angle)
                left_arrow_y = arc_center_y + gap * np.sin(left_arrow_angle)
                left_tangent = (np.sin(left_arrow_angle), -np.cos(left_arrow_angle))
                
                right_arrow_angle = np.radians(50)
                right_arrow_x = right_x - gap + gap * np.cos(right_arrow_angle)
                right_arrow_y = arc_center_y + gap * np.sin(right_arrow_angle)
                right_tangent = (-np.sin(right_arrow_angle), np.cos(right_arrow_angle))
                
                self._draw_arrowhead(ctx, (left_arrow_x, left_arrow_y), left_tangent, small_arrow, color=arc_color)
                self._draw_arrowhead(ctx, (right_arrow_x, right_arrow_y), right_tangent, small_arrow, color=arc_color)
                
                # Label below
                label_x = center[0]
                label_y = arc_center_y - (arc_height / 2) - label_offset * 0.3
                self.draw_text(label_x, label_y, "Circumference",
                              use_background=True, ha="center", va="top")
            
            elif base_position == "top":
                # Arc above the base, curving down at ends
                arc_center_y = center[1] + gap
                
                # Draw top half of ellipse (0° to 180°) - using ctx.line_color
                ctx.ax.add_patch(patches.Arc(
                    (center[0], arc_center_y), arc_width, arc_height,
                    theta1=0, theta2=180,
                    color=arc_color, lw=arc_lw,
                    linestyle="--"
                ))
                
                half_width = arc_width / 2
                left_x = center[0] - half_width
                right_x = center[0] + half_width
                
                # Back continuation arcs curving down - using ctx.line_color
                ctx.ax.add_patch(patches.Arc(
                    (left_x + gap, arc_center_y), back_arc_size, back_arc_size,
                    theta1=180, theta2=225,
                    color=arc_color, lw=arc_lw,
                    linestyle="--"
                ))
                ctx.ax.add_patch(patches.Arc(
                    (right_x - gap, arc_center_y), back_arc_size, back_arc_size,
                    theta1=315, theta2=360,
                    color=arc_color, lw=arc_lw,
                    linestyle="--"
                ))
                
                # Arrowheads - pointing down and inward (continuing around back)
                left_arrow_angle = np.radians(220)
                left_arrow_x = left_x + gap + gap * np.cos(left_arrow_angle)
                left_arrow_y = arc_center_y + gap * np.sin(left_arrow_angle)
                # Arrow points down and to the right (into the back)
                left_tangent = (0.7, -0.7)
                
                right_arrow_angle = np.radians(320)
                right_arrow_x = right_x - gap + gap * np.cos(right_arrow_angle)
                right_arrow_y = arc_center_y + gap * np.sin(right_arrow_angle)
                # Arrow points down and to the left (into the back)
                right_tangent = (-0.7, -0.7)
                
                self._draw_arrowhead(ctx, (left_arrow_x, left_arrow_y), left_tangent, small_arrow, color=arc_color)
                self._draw_arrowhead(ctx, (right_arrow_x, right_arrow_y), right_tangent, small_arrow, color=arc_color)
                
                # Label above
                label_x = center[0]
                label_y = arc_center_y + (arc_height / 2) + label_offset * 0.3
                self.draw_text(label_x, label_y, "Circumference",
                              use_background=True, ha="center", va="bottom")
        
        elif orientation == "vertical" and ellipse_ratio is not None:
            # Vertical elliptical base (shapes on side)
            gap = radius * 0.15
            base_width = radius * 2 * ellipse_ratio
            base_height = radius * 2
            # Arc wraps AROUND the base - add gap to both dimensions
            arc_width = base_width + (gap * 2)
            arc_height = base_height + (gap * 2)
            back_arc_size = gap * 2
            small_arrow = min(0.2, radius * 0.08)
            
            if base_position == "left":
                # Arc to the left of the base, curving right at ends
                arc_center_x = center[0] - gap
                
                # Draw left half of ellipse (90° to 270°) - using ctx.line_color
                ctx.ax.add_patch(patches.Arc(
                    (arc_center_x, center[1]), arc_width, arc_height,
                    theta1=90, theta2=270,
                    color=arc_color, lw=arc_lw,
                    linestyle="--"
                ))
                
                half_height = arc_height / 2
                top_y = center[1] + half_height
                bottom_y = center[1] - half_height
                
                # Back continuation arcs curving right - using ctx.line_color
                ctx.ax.add_patch(patches.Arc(
                    (arc_center_x, top_y - gap), back_arc_size, back_arc_size,
                    theta1=45, theta2=90,
                    color=arc_color, lw=arc_lw,
                    linestyle="--"
                ))
                ctx.ax.add_patch(patches.Arc(
                    (arc_center_x, bottom_y + gap), back_arc_size, back_arc_size,
                    theta1=270, theta2=315,
                    color=arc_color, lw=arc_lw,
                    linestyle="--"
                ))
                
                # Arrowheads - pointing right and inward (continuing around back)
                top_arrow_angle = np.radians(50)
                top_arrow_x = arc_center_x + gap * np.cos(top_arrow_angle)
                top_arrow_y = top_y - gap + gap * np.sin(top_arrow_angle)
                # Arrow points right and down (into the back)
                top_tangent = (0.7, -0.7)
                
                bottom_arrow_angle = np.radians(310)
                bottom_arrow_x = arc_center_x + gap * np.cos(bottom_arrow_angle)
                bottom_arrow_y = bottom_y + gap + gap * np.sin(bottom_arrow_angle)
                # Arrow points right and up (into the back)
                bottom_tangent = (0.7, 0.7)
                
                self._draw_arrowhead(ctx, (top_arrow_x, top_arrow_y), top_tangent, small_arrow, color=arc_color)
                self._draw_arrowhead(ctx, (bottom_arrow_x, bottom_arrow_y), bottom_tangent, small_arrow, color=arc_color)
                
                # Label to the left
                label_x = arc_center_x - (arc_width / 2) - label_offset * 0.3
                label_y = center[1]
                self.draw_text(label_x, label_y, "Circumference",
                              use_background=True, ha="right", va="center")
            
            elif base_position == "right":
                # Arc to the right of the base, curving left at ends
                arc_center_x = center[0] + gap
                
                # Draw right half of ellipse (270° to 90°, i.e., -90° to 90°) - using ctx.line_color
                ctx.ax.add_patch(patches.Arc(
                    (arc_center_x, center[1]), arc_width, arc_height,
                    theta1=-90, theta2=90,
                    color=arc_color, lw=arc_lw,
                    linestyle="--"
                ))
                
                half_height = arc_height / 2
                top_y = center[1] + half_height
                bottom_y = center[1] - half_height
                
                # Back continuation arcs curving left - using ctx.line_color
                ctx.ax.add_patch(patches.Arc(
                    (arc_center_x, top_y - gap), back_arc_size, back_arc_size,
                    theta1=90, theta2=135,
                    color=arc_color, lw=arc_lw,
                    linestyle="--"
                ))
                ctx.ax.add_patch(patches.Arc(
                    (arc_center_x, bottom_y + gap), back_arc_size, back_arc_size,
                    theta1=225, theta2=270,
                    color=arc_color, lw=arc_lw,
                    linestyle="--"
                ))
                
                # Arrowheads - pointing left and inward (continuing around back)
                top_arrow_angle = np.radians(130)
                top_arrow_x = arc_center_x + gap * np.cos(top_arrow_angle)
                top_arrow_y = top_y - gap + gap * np.sin(top_arrow_angle)
                # Arrow points left and down (into the back)
                top_tangent = (-0.7, -0.7)
                
                bottom_arrow_angle = np.radians(230)
                bottom_arrow_x = arc_center_x + gap * np.cos(bottom_arrow_angle)
                bottom_arrow_y = bottom_y + gap + gap * np.sin(bottom_arrow_angle)
                # Arrow points left and up (into the back)
                bottom_tangent = (-0.7, 0.7)
                
                self._draw_arrowhead(ctx, (top_arrow_x, top_arrow_y), top_tangent, small_arrow, color=arc_color)
                self._draw_arrowhead(ctx, (bottom_arrow_x, bottom_arrow_y), bottom_tangent, small_arrow, color=arc_color)
                
                # Label to the right
                label_x = arc_center_x + (arc_width / 2) + label_offset * 0.3
                label_y = center[1]
                self.draw_text(label_x, label_y, "Circumference",
                              use_background=True, ha="left", va="center")
        
        else:
            # Fallback for vertical without ellipse_ratio
            arc_radius_x = label_offset
            arc_radius_y = radius + label_offset
            
            start_angle = -90
            end_angle = 90
            
            ctx.ax.add_patch(patches.Arc(
                center, arc_radius_x * 2, arc_radius_y * 2,
                theta1=start_angle, theta2=end_angle,
                color=arc_color, lw=arc_lw,
                linestyle="--"
            ))
            
            top_pt = (center[0], center[1] + arc_radius_y)
            top_tangent = (-1, 0)
            
            bottom_pt = (center[0], center[1] - arc_radius_y)
            bottom_tangent = (-1, 0)
            
            self._draw_arrowhead(ctx, top_pt, top_tangent, arrow_size, color=arc_color)
            self._draw_arrowhead(ctx, bottom_pt, bottom_tangent, arrow_size, color=arc_color)
            
            label_x = center[0] + arc_radius_x + label_offset
            label_y = center[1]
            self.draw_text(label_x, label_y, "Circumference",
                          use_background=True, ha="left", va="center")
    
    def _draw_arrowhead(self, ctx: DrawingContext, tip: Point, 
                        direction: tuple[float, float], size: float,
                        centered: bool = True, color: str = None) -> None:
        """Draw a triangular arrowhead.
        
        Args:
            ctx: Drawing context
            tip: Point where the arrow tip (or center if centered=True) is located
            direction: Unit vector pointing in the direction the arrow points
            size: Size of the arrowhead
            centered: If True, center the arrow on the point; if False, point is at tip
            color: Override color (defaults to ctx.line_color)
        """
        arrow_color = color if color is not None else ctx.line_color
        # Normalize direction
        length = math.sqrt(direction[0]**2 + direction[1]**2)
        if length == 0:
            return
        dx, dy = direction[0] / length, direction[1] / length
        
        # Perpendicular vector
        px, py = -dy, dx
        
        # Calculate triangle points based on centered flag
        if centered:
            tip_pos = (tip[0] + dx * size * 0.5, tip[1] + dy * size * 0.5)
            base_center = (tip[0] - dx * size * 0.5, tip[1] - dy * size * 0.5)
        else:
            tip_pos = tip
            base_center = (tip[0] - dx * size, tip[1] - dy * size)
        
        p1 = (base_center[0] + px * size * 0.5, base_center[1] + py * size * 0.5)
        p2 = (base_center[0] - px * size * 0.5, base_center[1] - py * size * 0.5)
        
        # Draw filled triangle
        triangle = patches.Polygon(
            [tip_pos, p1, p2],
            closed=True,
            facecolor=arrow_color,
            edgecolor=arrow_color,
            lw=1
        )
        ctx.ax.add_patch(triangle)
    
    def draw_radial_dimension_labels(self, ctx: DrawingContext, radius: float,
                                     base_pos: float, orientation: str = "horizontal",
                                     label_offset: float = None,
                                     ellipse_ratio: float = None,
                                     base_position: str = "bottom") -> None:
        if label_offset is None:
            label_offset = AppConstants.LABEL_OFFSET_RADIAL
        
        txt_r, show_r = self.label_manager.get_entry_values("Radius")
        txt_d, show_d = self.label_manager.get_entry_values("Diameter")
        
        if orientation == "horizontal":
            if txt_d.strip() and show_d:
                self.draw_dim_line(ctx, "Diameter", (-radius, base_pos), (radius, base_pos))
                d_off = label_offset * 0.5 if not (txt_r.strip() and show_r) else label_offset * 1.2
                self.draw_text(0, base_pos - d_off, "Diameter", use_background=True, ha="center", va="center")
            
            if txt_r.strip() and show_r:
                self.draw_dim_line(ctx, "Radius", (0, base_pos), (radius, base_pos))
                self.draw_text(radius / 2, base_pos - label_offset * 0.5, "Radius", use_background=True, ha="center", va="center")
            
            # Draw circumference arc
            self.draw_circumference_arc(ctx, (0, base_pos), radius, "horizontal", label_offset, ellipse_ratio, base_position)
        
        else:
            # Vertical layout (horizontal shape like cylinder on side)
            dynamic_offset = label_offset * 0.5 if not (txt_r.strip() and show_r) else label_offset * 1.2
            
            if txt_d.strip() and show_d:
                self.draw_dim_line(ctx, "Diameter", (base_pos, -radius), (base_pos, radius))
                self.draw_text(base_pos - dynamic_offset, 0, "Diameter", use_background=True, ha="right", va="center")
            
            if txt_r.strip() and show_r:
                self.draw_dim_line(ctx, "Radius", (base_pos, 0), (base_pos, radius))
                self.draw_text(base_pos - label_offset * 0.5, radius / 2, "Radius", use_background=True, ha="right", va="center")
            
            # Draw circumference arc
            self.draw_circumference_arc(ctx, (base_pos, 0), radius, "vertical", label_offset, ellipse_ratio, base_position)


class ArrowMixin:
    """Mixin for shapes that draw arrows."""
    
    def draw_arrow(self, ctx: DrawingContext, start: Point, end: Point,
                   head_width: float = None, head_length: float = None) -> None:
        if head_width is None:
            head_width = AppConstants.ARROW_HEAD_WIDTH
        if head_length is None:
            head_length = head_width * 1.5
        
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        # Shorten arrow shaft so head ends at 'end' point
        length = math.sqrt(dx**2 + dy**2)
        if length > head_length:
            scale = (length - head_length) / length
            dx *= scale
            dy *= scale
        
        ctx.ax.arrow(start[0], start[1], dx, dy,
                    head_width=head_width, head_length=head_length,
                    fc=ctx.line_color, ec=ctx.line_color)


class ShapeDrawer:
    """Base class for all shape drawers."""
    
    def __init__(self, deps: DrawingDependencies):
        self._deps = deps
    
    @property
    def ax(self) -> Axes:
        return self._deps.ax
    
    @property
    def label_manager(self) -> LabelManager:
        return self._deps.label_manager
    
    @property
    def font_size(self) -> int:
        return self._deps.font_size
    
    def draw(self, ctx: DrawingContext, transform: TransformState, params: dict) -> None:
        """Draw the shape. Subclasses must implement this."""
        raise NotImplementedError
    
    def get_snap_anchors(self, ctx: DrawingContext, transform: TransformState, params: dict) -> list[tuple[float, float]]:
        """Return semantic snap points for composite snapping.
        
        Override in subclasses to provide shape-specific anchor points
        (e.g., ellipse centers, apex, base midpoints) that enable clean
        joins between 3D shapes in composite mode.
        Default: empty list (falls back to bbox-only snapping).
        """
        return []
    
    def parse_dimension(self, key: str, default_base: float, 
                        aspect_ratio: float) -> tuple[float, bool, bool]:
        """Parse a dimension from entries, returns (value, is_numeric, show)."""
        text, show = self.label_manager.get_entry_values(key)
        if not text.strip():
            return default_base * aspect_ratio, False, show
        try:
            val = float(text)
            return val, True, show
        except (ValueError, TypeError):
            return default_base * aspect_ratio, False, show
    
    def get_param_num(self, params: dict, key: str, default: float,
                      aspect_ratio: float) -> float:
        """Get numeric parameter from params dict with fallback to scaled default."""
        num_key = f"{key}_num"
        if num_key in params and params[num_key] is not None:
            value = params[num_key]
            if not math.isfinite(value):
                return default * aspect_ratio
            if value == 0:
                raise ValidationError(f"{key} must be positive")
            if value > 0:
                return value
        return default * aspect_ratio
    
    # -------------------- Validation Helpers --------------------
    
    def validate_positive_params(self, params: dict, keys: list[str]) -> Optional[str]:
        """Validate that all specified parameters are positive if present.
        Returns first error message or None (does NOT raise)."""
        for key in keys:
            err = ShapeValidator.validate_positive(key, params.get(f"{key}_num"))
            if err:
                return err
        return None
    
    def validate_pairs_equal(self, params: dict, 
                             pairs: list[tuple[str, str]]) -> Optional[str]:
        """Validate that pairs are equal. Returns first error message or None."""
        for key1, key2 in pairs:
            err = ShapeValidator.validate_equal(
                key1, params.get(f"{key1}_num"),
                key2, params.get(f"{key2}_num")
            )
            if err:
                return err
        return None
    
    def validate_radius_diameter(self, params: dict, default_radius: float = None) -> float:
        """Validate and extract radius from Radius/Diameter params.
        Raises ValidationError if invalid. Returns radius value."""
        if default_radius is None:
            default_radius = AppConstants.CIRCLE_DEFAULT_RADIUS
        r_num = params.get("Radius_num")
        d_num = params.get("Diameter_num")
        
        # Reject zero explicitly — 0 is not a valid radius or diameter
        if r_num is not None and r_num == 0:
            raise ValidationError("Radius must be positive")
        if d_num is not None and d_num == 0:
            raise ValidationError("Diameter must be positive")
        
        # Only validate if user has entered actual numeric values
        # Allow defaults/empty values to pass through without error
        has_r = r_num is not None
        has_d = d_num is not None
        
        if not has_r and not has_d:
            # No values entered, use default without validation
            return default_radius
        
        radius, err = ShapeValidator.validate_diameter_radius(r_num, d_num, default_radius)
        if err:
            raise ValidationError(err)
        return radius
    
    def has_numeric_value(self, params: dict, key: str) -> bool:
        """Check if a parameter has a positive numeric value."""
        val = params.get(f"{key}_num")
        return val is not None and val > 0
    
    def collect_numeric_values(self, params: dict, keys: list[str]) -> dict[str, float]:
        """Collect all positive numeric values for the given keys."""
        result = {}
        for key in keys:
            val = params.get(f"{key}_num")
            if val is not None and val > 0:
                result[key] = val
        return result
    
    def draw_text(self, x: float, y: float, key: str, use_background: bool = True, 
                  **kwargs) -> None:
        self.label_manager.draw_label(self.ax, x, y, key, self.font_size, use_background, 
                                      **kwargs)
    
    def get_smart_label_pos(self, p1: tuple, p2: tuple, centroid: tuple,
                            buffer: float = None) -> tuple[float, float, str, str]:
        if buffer is None:
            buffer = AppConstants.SMART_LABEL_BUFFER
        return self.label_manager.get_smart_label_pos(p1, p2, centroid, buffer)
    
    def transform_points(self, points: Polygon, center: Optional[Point] = None,
                         transform: Optional[TransformState] = None) -> Polygon:
        """Apply flip transforms to points."""
        if not points:
            return points
        
        if transform is None:
            return points
        
        flip_h = transform.flip_h
        flip_v = transform.flip_v
        
        if not flip_h and not flip_v:
            return points
        
        if center is None:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            center = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
        
        transformed = []
        for x, y in points:
            x -= center[0]
            y -= center[1]
            
            if flip_h:
                x = -x
            if flip_v:
                y = -y
            
            x += center[0]
            y += center[1]
            
            transformed.append((x, y))
        
        return transformed
    
    def calculate_centroid(self, points: Polygon) -> Point:
        """Calculate centroid of a polygon."""
        n = len(points)
        if n == 0:
            return (0, 0)
        cx = sum(p[0] for p in points) / n
        cy = sum(p[1] for p in points) / n
        return (cx, cy)
    
    def calculate_bounds(self, points: Polygon) -> tuple[Point, Point]:
        """Calculate bounding box of points, returns ((min_x, max_x), (min_y, max_y))."""
        if not points:
            return ((0, 0), (0, 0))
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return ((min(xs), max(xs)), (min(ys), max(ys)))
    
    def set_limits(self, ctx: DrawingContext, x_range: tuple, y_range: tuple,
                   padding: float = 0.1) -> None:
        # B29 fix: expand bounds by at least the label buffer so preset labels
        # placed outside shape geometry are still within the tight bounds that
        # _apply_view_scale uses to compute margins. Without this, small shapes
        # have labels clipped because the fixed buffer (0.3) exceeds 12% of a
        # tiny shape extent.
        label_margin = AppConstants.SMART_LABEL_BUFFER + 0.15  # buffer + bbox pad
        x_min = x_range[0] - label_margin
        x_max = x_range[1] + label_margin
        y_min = y_range[0] - label_margin
        y_max = y_range[1] + label_margin
        ctx.ax.set_xlim(x_min, x_max)
        ctx.ax.set_ylim(y_min, y_max)
    
    def draw_polygon(self, ctx: DrawingContext, points: Polygon, closed: bool = True) -> None:
        if not points or len(points) < 2:
            return
        ctx.ax.add_patch(patches.Polygon(points, closed=closed, **ctx.line_args))
    
    def draw_circle(self, ctx: DrawingContext, center: Point, radius: float) -> None:
        if radius <= 0:
            radius = 1
        ctx.ax.add_patch(patches.Circle(center, radius, **ctx.line_args))
    
    def draw_ellipse(self, ctx: DrawingContext, center: Point, width: float,
                     height: float) -> None:
        if width <= 0 or height <= 0:
            width = max(width, 1)
            height = max(height, 1)
        ctx.ax.add_patch(patches.Ellipse(center, width, height, **ctx.line_args))
    
    def draw_line(self, ctx: DrawingContext, p1: Point, p2: Point,
                  dashed: bool = False) -> None:
        if dashed:
            style = {**ctx.dim_args, "linestyle": "--"}
        else:
            style = {"color": ctx.line_color, "lw": AppConstants.DEFAULT_LINE_WIDTH}
        ctx.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **style)

    def draw_dim_line(self, ctx: DrawingContext, key: str, p1: Point, p2: Point) -> None:
        """Draw a dashed dimension line that highlights blue when its label is selected."""
        dx, dy = self.label_manager.custom_dim_offsets.get(key, (0.0, 0.0))
        p1 = (p1[0] + dx, p1[1] + dy)
        p2 = (p2[0] + dx, p2[1] + dy)
        bdl = self.label_manager.geometry_hints.setdefault("builtin_dimlines", {})
        bdl[key] = {"x1": p1[0], "y1": p1[1], "x2": p2[0], "y2": p2[1]}
        is_sel = (self.label_manager.builtin_selected == key)
        color = "#0066cc" if is_sel else ctx.line_color
        lw = AppConstants.DIMENSION_LINE_WIDTH * 1.5 if is_sel else AppConstants.DIMENSION_LINE_WIDTH
        ctx.ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color=color, linewidth=lw, linestyle="--", zorder=12)

    def draw_labeled_dimension(self, ctx: DrawingContext, key: str,
                               p1: Point, p2: Point,
                               label_x: float, label_y: float,
                               label_ha: str = "center", label_va: str = "center") -> None:
        """Draw a dimension line with endpoint dots and positioned label.
        Checks visibility via label_manager; draws nothing if label is hidden.
        Applies custom_dim_offsets so dots and label stay with the line when dragged."""
        text, show = self.label_manager.get_entry_values(key)
        if not (text and show):
            return
        dx, dy = self.label_manager.custom_dim_offsets.get(key, (0.0, 0.0))
        self.draw_dim_line(ctx, key, p1, p2)
        self.draw_point(ctx, (p1[0] + dx, p1[1] + dy), AppConstants.DIM_ENDPOINT_SIZE)
        self.draw_point(ctx, (p2[0] + dx, p2[1] + dy), AppConstants.DIM_ENDPOINT_SIZE)
        self.draw_text(label_x + dx, label_y + dy, key, use_background=True, ha=label_ha, va=label_va)

    def draw_point(self, ctx: DrawingContext, point: Point, size: int = 5) -> None:
        ctx.ax.plot(point[0], point[1], marker="o", markersize=size, 
                    color=ctx.line_color)
    
    def draw_hash_marks(self, ctx: DrawingContext, points: Polygon,
                        hash_len: Optional[float] = None, count: int = 1) -> None:
        """Convenience wrapper around DrawingUtilities.draw_hash_marks."""
        DrawingUtilities.draw_hash_marks(ctx, points, hash_len, count)

    def draw_right_angle_marker(self, ctx: DrawingContext, vertex: Point,
                                 p1_dir: Point, p2_dir: Point,
                                 size: Optional[float] = None) -> None:
        """Convenience wrapper around DrawingUtilities.draw_right_angle_marker."""
        DrawingUtilities.draw_right_angle_marker(ctx, vertex, p1_dir, p2_dir, size)

    @staticmethod
    def calc_height_marker_size(foot: "Point", apex: "Point") -> float:
        """Unified G11 marker size: 10% of height line length, clamped [0.15, 0.40]."""
        dx = apex[0] - foot[0]
        dy = apex[1] - foot[1]
        line_len = math.sqrt(dx * dx + dy * dy)
        return max(0.15, min(0.40, line_len * 0.10))
    
    def rotate_list(self, items: list, positions: int) -> list:
        """Rotate list elements by n positions."""
        if not items or positions == 0:
            return items
        n = len(items)
        positions = positions % n
        return items[positions:] + items[:positions]
    
    def rotate_polygon_to_base(self, points: Polygon, base_side: int) -> Polygon:
        """Rotate polygon so specified side becomes the base (bottom)."""
        if base_side == 0:
            return points
        
        n = len(points)
        p1 = points[base_side]
        p2 = points[(base_side + 1) % n]
        
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = math.atan2(dy, dx)
        
        cos_a = math.cos(-angle)
        sin_a = math.sin(-angle)
        
        rotated = []
        for px, py in points:
            tx = px - p1[0]
            ty = py - p1[1]
            rx = tx * cos_a - ty * sin_a
            ry = tx * sin_a + ty * cos_a
            rotated.append((rx, ry))
        
        rotated = rotated[base_side:] + rotated[:base_side]
        
        min_x = min(p[0] for p in rotated)
        min_y = min(p[1] for p in rotated)
        rotated = [(p[0] - min_x, p[1] - min_y) for p in rotated]
        
        return rotated
    
    def draw_hidden_arc(self, ctx: DrawingContext, center: Point,
                        width: float, height: float,
                        solid_angles: tuple[float, float],
                        dashed_angles: tuple[float, float]) -> None:
        """Draw the solid and dashed portions of an ellipse arc with proper occlusion."""
        # Use centralized Z-orders: solid arcs go with lines, dashed arcs go to the back
        ctx.ax.add_patch(patches.Arc(
            center, width, height,
            theta1=solid_angles[0], theta2=solid_angles[1],
            color=ctx.line_color, lw=AppConstants.DEFAULT_LINE_WIDTH, zorder=DrawingUtilities.Z_LINES))
        dash_args_with_z = {**ctx.dash_args, "zorder": DrawingUtilities.Z_BACK}
        ctx.ax.add_patch(patches.Arc(
            center, width, height,
            theta1=dashed_angles[0], theta2=dashed_angles[1],
            **dash_args_with_z))


class TransformController:
    """Handles flip and rotation state management."""
    
    def __init__(
        self,
        on_change_callback: callable,
        flip_h_btn: Optional[tk.Button] = None,
        flip_v_btn: Optional[tk.Button] = None,
        rotate_ccw_btn: Optional[tk.Button] = None,
        rotate_cw_btn: Optional[tk.Button] = None
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
    
    def toggle_flip_h(self) -> None:
        """Toggle horizontal flip."""
        self.flip_h = not self.flip_h
        self._reset_button_icons()
        self.on_change_callback()
    
    def toggle_flip_v(self) -> None:
        """Toggle vertical flip."""
        self.flip_v = not self.flip_v
        self._reset_button_icons()
        self.on_change_callback()
    
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
            base_side=self.base_side
        )


class InputController:
    """Handles input field creation and management."""

    def __init__(self, input_frame: tk.Frame, label_manager: LabelManager,
                 on_change_callback: callable):
        self.input_frame = input_frame
        self.label_manager = label_manager
        self.on_change_callback = on_change_callback
        self.entries: dict[str, dict] = {}

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
    
    def clear_entries(self) -> None:
        """Clear all entry widgets and reset entries dict."""
        for w in self.input_frame.winfo_children():
            w.destroy()
        
        self.entries = {}
        self.label_manager.clear_positions()
    
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
        """Handle text changes in entry fields."""
        if label not in self.entries:
            return
        
        entry_data = self.entries[label]
        text = entry_data["text"].get().strip()
        new_state = "normal" if text else "disabled"
        
        if "show_cb" in entry_data:
            entry_data["show_cb"].config(state=new_state)

        # Mutual exclusivity for circular and triangular dimensions
        if text:
            if label == "Radius" and "Diameter" in self.entries:
                self.entries["Diameter"]["text"].delete(0, tk.END)
            elif label == "Diameter" and "Radius" in self.entries:
                self.entries["Radius"]["text"].delete(0, tk.END)
            elif label == "Height" and "Left Side" in self.entries:
                self.entries["Left Side"]["text"].delete(0, tk.END)
                self.entries["Right Side"]["text"].delete(0, tk.END)
            elif label in ("Left Side", "Right Side") and "Height" in self.entries:
                self.entries["Height"]["text"].delete(0, tk.END)
        
        # Notify parent of change (debouncing handled by parent)
        self.on_change_callback()
    
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

class CompositeTransferList(tk.Frame):
    """Transfer list widget for composite figure shape selection.

    Source list (available shapes) on the left, arrow controls in the center,
    destination list (selected shapes) with reorder handles on the right.
    """

    # Maps internal shape name -> user-facing display name in the transfer list.
    # INTERNAL_NAMES is derived automatically — edit only DISPLAY_NAMES.
    DISPLAY_NAMES: dict[str, str] = {"Tri Prism": "Triangular Prism", "Tri Triangle": "Triangle"}
    INTERNAL_NAMES: dict[str, str] = {v: k for k, v in DISPLAY_NAMES.items()}

    def __init__(self, parent: tk.Frame, available_shapes: list[str],
                 on_change_callback: callable,
                 on_before_change_callback: callable = None, **kwargs):
        super().__init__(parent, bg=AppConstants.BG_COLOR, **kwargs)
        self.available_shapes = list(available_shapes)
        self.on_change_callback = on_change_callback
        self.on_before_change_callback = on_before_change_callback
        self._build_ui()
    
    def _build_ui(self) -> None:
        """Build the three-column transfer list layout."""
        # --- Source column ---
        src_frame = tk.Frame(self, bg=AppConstants.BG_COLOR)
        src_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 2))
        
        tk.Label(src_frame, text="Available", bg=AppConstants.BG_COLOR,
                 font=("Arial", 8, "bold")).pack(side=tk.TOP)
        
        self.source_listbox = tk.Listbox(src_frame, width=18, height=10,
                                          selectmode=tk.SINGLE, exportselection=False)
        self.source_listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.source_listbox.bind("<Double-Button-1>", lambda e: self._add_selected())
        
        for shape in self.available_shapes:
            self.source_listbox.insert(tk.END, self.DISPLAY_NAMES.get(shape, shape))
        
        # --- Arrow buttons column ---
        btn_frame = tk.Frame(self, bg=AppConstants.BG_COLOR)
        btn_frame.pack(side=tk.LEFT, fill=tk.Y, padx=4)
        
        # Vertical centering spacer
        tk.Frame(btn_frame, bg=AppConstants.BG_COLOR, height=30).pack(side=tk.TOP)
        
        self.add_btn = tk.Button(btn_frame, text="→", width=3, font=("Arial", 12),
                                  command=self._add_selected)
        self.add_btn.pack(side=tk.TOP, pady=2)
        
        self.remove_btn = tk.Button(btn_frame, text="←", width=3, font=("Arial", 12),
                                     command=self._remove_selected)
        self.remove_btn.pack(side=tk.TOP, pady=2)
        
        # --- Destination column ---
        dest_outer = tk.Frame(self, bg=AppConstants.BG_COLOR)
        dest_outer.pack(side=tk.LEFT, fill=tk.BOTH, padx=(2, 0))
        
        tk.Label(dest_outer, text="Selected", bg=AppConstants.BG_COLOR,
                 font=("Arial", 8, "bold")).pack(side=tk.TOP)
        
        dest_row = tk.Frame(dest_outer, bg=AppConstants.BG_COLOR)
        dest_row.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.dest_listbox = tk.Listbox(dest_row, width=18, height=10,
                                        selectmode=tk.SINGLE, exportselection=False)
        self.dest_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.dest_listbox.bind("<Double-Button-1>", lambda e: self._remove_selected())
        
        # Reorder buttons
        reorder_frame = tk.Frame(dest_row, bg=AppConstants.BG_COLOR)
        reorder_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(2, 0))
        
        tk.Frame(reorder_frame, bg=AppConstants.BG_COLOR, height=20).pack(side=tk.TOP)
        
        self.up_btn = tk.Button(reorder_frame, text="▲", width=2, font=("Arial", 10),
                                 command=self._move_up)
        self.up_btn.pack(side=tk.TOP, pady=1)
        
        self.down_btn = tk.Button(reorder_frame, text="▼", width=2, font=("Arial", 10),
                                   command=self._move_down)
        self.down_btn.pack(side=tk.TOP, pady=1)
    
    def _add_selected(self) -> None:
        """Copy selected item from source to destination (source is a palette, not depleted)."""
        sel = self.source_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        shape = self.source_listbox.get(idx)
        self.dest_listbox.insert(tk.END, shape)
        # Keep source selection for quick multi-add
        self.source_listbox.selection_set(idx)
        self.on_change_callback(("add", self.dest_listbox.size() - 1))
    
    def _remove_selected(self) -> None:
        """Remove selected item from destination list."""
        sel = self.dest_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        self.dest_listbox.delete(idx)
        # Re-select next item in dest
        if self.dest_listbox.size() > 0:
            new_idx = min(idx, self.dest_listbox.size() - 1)
            self.dest_listbox.selection_set(new_idx)
        self.on_change_callback(("remove", idx))
    
    def _move_up(self) -> None:
        """Move selected destination item up one position."""
        sel = self.dest_listbox.curselection()
        if not sel or sel[0] == 0:
            return
        idx = sel[0]
        shape = self.dest_listbox.get(idx)
        self.dest_listbox.delete(idx)
        self.dest_listbox.insert(idx - 1, shape)
        self.dest_listbox.selection_set(idx - 1)
        self.on_change_callback(("swap", idx, idx - 1))
    
    def _move_down(self) -> None:
        """Move selected destination item down one position."""
        sel = self.dest_listbox.curselection()
        if not sel or sel[0] >= self.dest_listbox.size() - 1:
            return
        idx = sel[0]
        shape = self.dest_listbox.get(idx)
        self.dest_listbox.delete(idx)
        self.dest_listbox.insert(idx + 1, shape)
        self.dest_listbox.selection_set(idx + 1)
        self.on_change_callback(("swap", idx, idx + 1))
    
    def get_selected_shapes(self) -> list[str]:
        """Return the ordered list of shapes in the destination list (internal names)."""
        return [self.INTERNAL_NAMES.get(self.dest_listbox.get(i), self.dest_listbox.get(i))
                for i in range(self.dest_listbox.size())]
    
    def set_selected_shapes(self, shapes: list[str]) -> None:
        """Restore destination list state (for undo/redo). Accepts internal names."""
        self.dest_listbox.delete(0, tk.END)
        for s in shapes:
            self.dest_listbox.insert(tk.END, self.DISPLAY_NAMES.get(s, s))


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
                   transform: TransformState, params: dict) -> Optional[str]:
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


class ShapeRegistry:
    """Registry for shape drawer classes using a simple factory pattern."""
    
    _drawers: dict[str, type] = {}
    
    @classmethod
    def register(cls, shape_name: str):
        """Decorator to register a shape drawer class."""
        def decorator(drawer_class: type):
            cls._drawers[shape_name] = drawer_class
            return drawer_class
        return decorator
    
    @classmethod
    def get_drawer(cls, shape_name: str, deps: DrawingDependencies) -> Optional[ShapeDrawer]:
        """Get a new drawer instance for a shape, or None if not registered."""
        if shape_name not in cls._drawers:
            return None
        
        return cls._drawers[shape_name](deps)
    
    @classmethod
    def has_drawer(cls, shape_name: str) -> bool:
        return shape_name in cls._drawers
    
    @classmethod
    def get_registered_shapes(cls) -> list[str]:
        return list(cls._drawers.keys())


@ShapeRegistry.register("Rectangle")
class RectangleDrawer(ShapeDrawer, PolygonLabelMixin):
    """Draws rectangles with rotation and flip support."""
    
    def get_snap_anchors(self, ctx, transform, params):
        mode = params.get("dimension_mode", "Custom")
        if mode == "Default":
            ratio = ctx.aspect_ratio / AppConstants.SLIDER_DEFAULT
            w = AppConstants.RECT_DEFAULT_WIDTH * ratio
            h = AppConstants.RECT_DEFAULT_HEIGHT / ratio
        else:
            w = self.get_param_num(params, "Length", 6, ctx.aspect_ratio)
            h = self.get_param_num(params, "Width", 4, ctx.aspect_ratio)
        base_points, _ = self._get_rotated_geometry(w, h, transform.base_side)
        center = self.calculate_centroid(base_points)
        points = self.transform_points(base_points, center, transform)
        anchors = list(points)
        n = len(points)
        for i in range(n):
            mx = (points[i][0] + points[(i+1) % n][0]) / 2
            my = (points[i][1] + points[(i+1) % n][1]) / 2
            anchors.append((mx, my))
        anchors.append(center)
        return anchors
    
    SIDE_ALIGNS = [
        ("center", "top"),     # Bottom label
        ("left", "center"),    # Right label
        ("center", "bottom"),  # Top label
        ("right", "center"),   # Left label
    ]
    DIMENSION_KEYS = ["Top", "Bottom", "Left", "Right", "Length", "Width"]
    EQUAL_PAIRS = [("Top", "Bottom"), ("Left", "Right")]
    
    def draw(self, ctx: DrawingContext, transform: TransformState, params: dict) -> None:
        mode = params.get("dimension_mode", "Custom")
        
        # Validate dimensions
        if err := self.validate_positive_params(params, self.DIMENSION_KEYS):
            raise ValidationError(err)
        if err := self.validate_pairs_equal(params, self.EQUAL_PAIRS):
            raise ValidationError(err)
        
        if mode == "Default":
            ratio = ctx.aspect_ratio / AppConstants.SLIDER_DEFAULT
            w = AppConstants.RECT_DEFAULT_WIDTH * ratio
            h = AppConstants.RECT_DEFAULT_HEIGHT / ratio
        else:
            w = self.get_param_num(params, "Length", 6, ctx.aspect_ratio)
            h = self.get_param_num(params, "Width", 4, ctx.aspect_ratio)
        
        base_points, label_keys = self._get_rotated_geometry(w, h, transform.base_side)
        center = self.calculate_centroid(base_points)
        points = self.transform_points(base_points, center, transform)
        self.draw_polygon(ctx, points)
        x_range, y_range = self.calculate_bounds(points)
        self.set_limits(ctx, x_range, y_range)
        self.draw_side_labels(points, label_keys, self.SIDE_ALIGNS)
        # Store geometry hints for preset dim lines
        self.label_manager.geometry_hints["height_y1"] = min(p[1] for p in points)
        self.label_manager.geometry_hints["height_y2"] = max(p[1] for p in points)
        self.label_manager.geometry_hints["height_x"] = (min(p[0] for p in points) + max(p[0] for p in points)) / 2
        self.label_manager.geometry_hints["rect_pts"] = list(points)
    
    def _get_rotated_geometry(self, w: float, h: float, 
                               base_side: int) -> tuple[Polygon, list[str]]:
        # Cases 0 and 2 share the same point layout (w×h rectangle at origin);
        # cases 1 and 3 swap width and height.  Only the label ordering differs
        # across all four orientations — labels rotate while geometry stays fixed.
        if base_side == 0:
            points = [(0, 0), (w, 0), (w, h), (0, h)]
            label_keys = ["Bottom", "Right", "Top", "Left"]
        elif base_side == 1:
            points = [(0, 0), (h, 0), (h, w), (0, w)]
            label_keys = ["Right", "Top", "Left", "Bottom"]
        elif base_side == 2:
            points = [(0, 0), (w, 0), (w, h), (0, h)]  # same geometry as case 0; labels rotate
            label_keys = ["Top", "Left", "Bottom", "Right"]
        else:
            points = [(0, 0), (h, 0), (h, w), (0, w)]
            label_keys = ["Left", "Bottom", "Right", "Top"]
        
        return points, label_keys


@ShapeRegistry.register("Square")
class SquareDrawer(ShapeDrawer, PolygonLabelMixin):
    """Draws squares with flip support."""
    
    def get_snap_anchors(self, ctx, transform, params):
        s = self._get_side_length(params, ctx)
        base_points = [(0, 0), (s, 0), (s, s), (0, s)]
        center = (s / 2, s / 2)
        points = self.transform_points(base_points, center, transform)
        anchors = list(points)
        n = len(points)
        for i in range(n):
            mx = (points[i][0] + points[(i+1) % n][0]) / 2
            my = (points[i][1] + points[(i+1) % n][1]) / 2
            anchors.append((mx, my))
        anchors.append(center)
        return anchors
    
    SIDE_LABELS = ["Bottom", "Right", "Top", "Left"]
    SIDE_ALIGNS = [
        ("center", "top"),
        ("left", "center"),
        ("center", "bottom"),
        ("right", "center"),
    ]
    SIDE_KEYS = ["Top", "Bottom", "Left", "Right"]
    
    def draw(self, ctx: DrawingContext, transform: TransformState, params: dict) -> None:
        s = self._get_side_length(params, ctx)
        
        base_points = [(0, 0), (s, 0), (s, s), (0, s)]
        center = (s / 2, s / 2)
        points = self.transform_points(base_points, center, transform)
        self.draw_polygon(ctx, points)
        
        # Only show hash marks in standalone mode (not in composite).
        # In composite, the hashmarks checkbox doesn't apply and clutters the canvas.
        if not ctx.composite_mode:
            DrawingUtilities.draw_hash_marks(ctx, points + [points[0]])
        
        x_range, y_range = self.calculate_bounds(points)
        self.set_limits(ctx, x_range, y_range)
        self.draw_side_labels(points, self.SIDE_LABELS, self.SIDE_ALIGNS)
        # Store rect_pts so side_v/side_h preset dim lines use actual rotated edge points
        self.label_manager.geometry_hints["rect_pts"] = list(points)
    
    def _get_side_length(self, params: dict, ctx: DrawingContext) -> float:
        """Returns side length. Raises ValidationError if invalid."""
        default = AppConstants.SQUARE_DEFAULT_SIDE * ctx.aspect_ratio
        
        for key in self.SIDE_KEYS:
            if err := ShapeValidator.validate_positive(key, params.get(f"{key}_num")):
                raise ValidationError(err)
        
        sides = self.collect_numeric_values(params, self.SIDE_KEYS)
        
        if err := ShapeValidator.validate_all_equal(sides):
            raise ValidationError(err)
        
        return next(iter(sides.values()), default)


@ShapeRegistry.register("Triangle")
class TriangleDrawer(ShapeDrawer):
    """Draws triangles of all types: Custom, Isosceles, Scalene, Equilateral."""
    
    def get_snap_anchors(self, ctx, transform, params):
        triangle_type = params.get("triangle_type", "Custom")
        base = self.get_param_num(params, "Base Width", AppConstants.TRIANGLE_DEFAULT_BASE, ctx.aspect_ratio)
        height = self.get_param_num(params, "Height", AppConstants.TRIANGLE_DEFAULT_HEIGHT, ctx.aspect_ratio)
        peak = params.get("peak_offset", 0.5)
        apex_x = base * peak
        p1 = (0, 0)
        p2 = (base, 0)
        p3 = (apex_x, height)
        center = ((p1[0] + p2[0] + p3[0]) / 3, (p1[1] + p2[1] + p3[1]) / 3)
        points = self.transform_points([p1, p2, p3], center, transform)
        mid_base = ((points[0][0] + points[1][0]) / 2, (points[0][1] + points[1][1]) / 2)
        cx = (points[0][0] + points[1][0] + points[2][0]) / 3
        cy = (points[0][1] + points[1][1] + points[2][1]) / 3
        return [points[0], points[1], points[2], mid_base, (cx, cy)]
    
    def draw(self, ctx: DrawingContext, transform: TransformState, params: dict) -> None:
        triangle_type = params.get("triangle_type", TriangleType.CUSTOM.value)
        self._validate_triangle(params, triangle_type)
        
        draw_methods = {
            TriangleType.CUSTOM.value: self._draw_custom,
            TriangleType.ISOSCELES.value: self._draw_isosceles,
            TriangleType.SCALENE.value: self._draw_scalene,
            TriangleType.EQUILATERAL.value: self._draw_equilateral,
        }
        
        method = draw_methods.get(triangle_type, self._draw_custom)
        method(ctx, transform, params)
    
    def _validate_triangle(self, params: dict, triangle_type: str) -> None:
        """Validate triangle dimensions. Raises ValidationError if invalid."""
        # Only validate if user has entered numeric values - shapes use defaults otherwise
        if triangle_type == TriangleType.ISOSCELES.value:
            # Only validate if user entered values
            if self.has_numeric_value(params, "Base Label") or \
               self.has_numeric_value(params, "Left Label") or \
               self.has_numeric_value(params, "Right Label"):
                if err := self.validate_positive_params(params, ["Base Label", "Left Label", "Right Label"]):
                    raise ValidationError(err)
                if err := self.validate_pairs_equal(params, [("Left Label", "Right Label")]):
                    raise ValidationError(err)
        elif triangle_type == TriangleType.SCALENE.value:
            # Only validate if user entered values
            if self.has_numeric_value(params, "Side A (Bottom)") or \
               self.has_numeric_value(params, "Side B (Left)") or \
               self.has_numeric_value(params, "Side C (Right)"):
                if err := self.validate_positive_params(params, ["Side A (Bottom)", "Side B (Left)", "Side C (Right)"]):
                    raise ValidationError(err)
        elif triangle_type == TriangleType.EQUILATERAL.value:
            side_keys = ["Side A (Bottom)", "Side B (Left)", "Side C (Right)"]
            # Only validate if user entered values
            if any(self.has_numeric_value(params, k) for k in side_keys):
                if err := self.validate_positive_params(params, side_keys):
                    raise ValidationError(err)
                sides = self.collect_numeric_values(params, side_keys)
                if len(sides) > 1:  # Only check equality if multiple values entered
                    if err := ShapeValidator.validate_all_equal(sides):
                        raise ValidationError("All sides of an equilateral triangle must be equal")
    
    def _get_base_vertices(self, base: float, height: float, 
                           peak_x: float) -> Polygon:
        return [(0, 0), (base, 0), (peak_x, height)]
    
    def _calculate_side_lengths(self, vertices: Polygon) -> tuple[float, float, float]:
        a, b, c = vertices[0], vertices[1], vertices[2]
        ab = math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
        bc = math.sqrt((c[0] - b[0])**2 + (c[1] - b[1])**2)
        ca = math.sqrt((a[0] - c[0])**2 + (a[1] - c[1])**2)
        return (ab, bc, ca)
    
    def _rotate_to_base(self, vertices: Polygon, label_keys: list[str], 
                        base_side: int) -> tuple[Polygon, list[str], bool]:
        # Simplify: Use standard polygon rotation for triangles to maintain parity with other shapes
        if base_side == 0:
            return vertices, label_keys, True
            
        # Rotate points and labels using the standard list rotation
        rotated_pts = self.rotate_polygon_to_base(vertices, base_side)
        rotated_labels = self.rotate_list(label_keys, base_side)
        
        # Height is only conventionally shown for the default base (base_side 0)
        return rotated_pts, rotated_labels, (base_side == 0)
    
    def _draw_triangle_common(self, ctx: DrawingContext, transform: TransformState,
                               base_pts: Polygon, label_keys: list[str], 
                               show_height: bool = False,
                               equal_sides: Optional[list[int]] = None,
                               padding: float = 1.0,
                               height_label_key: str = "Height") -> Polygon:
        xs = [p[0] for p in base_pts]
        ys = [p[1] for p in base_pts]
        center = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
        
        pts = self.transform_points(base_pts, center, transform)
        self.draw_polygon(ctx, pts)
        x_range, y_range = self.calculate_bounds(pts)
        self.set_limits(ctx, x_range, y_range, padding)
        if equal_sides:
            for side_idx in equal_sides:
                p1 = pts[side_idx]
                p2 = pts[(side_idx + 1) % 3]
                self.draw_hash_marks(ctx, [p1, p2])
        centroid = self.calculate_centroid(pts)
        sides = [(pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[0])]
        
        for i, key in enumerate(label_keys):
            p1, p2 = sides[i]
            nx, ny, ha, va = self.get_smart_label_pos(p1, p2, centroid)
            self.draw_text(nx, ny, key, ha=ha, va=va)
        if show_height:
            self._draw_height_line(ctx, pts, transform, height_label_key)
        
        # Store geometry hints for preset dim lines
        base_p1, base_p2, apex = pts[0], pts[1], pts[2]
        base_dx = base_p2[0] - base_p1[0]
        base_dy = base_p2[1] - base_p1[1]
        base_len = math.sqrt(base_dx**2 + base_dy**2)
        if base_len > 0.001:
            u_base = (base_dx / base_len, base_dy / base_len)
            v = (apex[0] - base_p1[0], apex[1] - base_p1[1])
            t = v[0] * u_base[0] + v[1] * u_base[1]
            foot = (base_p1[0] + u_base[0] * t, base_p1[1] + u_base[1] * t)
            self.label_manager.geometry_hints["tri_foot"] = foot
            self.label_manager.geometry_hints["tri_apex"] = apex
            self.label_manager.geometry_hints["tri_base_p1"] = base_p1
            self.label_manager.geometry_hints["tri_base_p2"] = base_p2
            self.label_manager.geometry_hints["tri_pts"] = [base_p1, base_p2, apex]
            self.label_manager.geometry_hints["height_y1"] = foot[1]
            self.label_manager.geometry_hints["height_y2"] = apex[1]
        
        return pts
    
    def _draw_height_line(self, ctx: DrawingContext, points: Polygon, 
                          transform: TransformState, label_key: str) -> None:
        text, show = self.label_manager.get_entry_values(label_key)
        if not (show and text.strip()):
            return
        
        # Height line always draws in canonical orientation.
        # The post-draw artist transform pipeline rotates it with the shape.
        
        # Base is edge from points[0] to points[1], apex is points[2]
        base_p1, base_p2, apex = points[0], points[1], points[2]
        
        # Calculate base edge vector
        base_dx = base_p2[0] - base_p1[0]
        base_dy = base_p2[1] - base_p1[1]
        base_len = math.sqrt(base_dx**2 + base_dy**2)
        
        if base_len < 0.001:
            return
        
        # Unit vectors
        u_base = (base_dx / base_len, base_dy / base_len)
        u_perp = (-u_base[1], u_base[0])
        
        # Project apex onto base line
        v = (apex[0] - base_p1[0], apex[1] - base_p1[1])
        t = v[0] * u_base[0] + v[1] * u_base[1]  # Distance along base
        h = v[0] * u_perp[0] + v[1] * u_perp[1]  # Perpendicular distance
        
        # Skip if height is too small
        if abs(h) < 0.1:
            return
        
        # Foot of perpendicular (may be outside base segment)
        foot = (base_p1[0] + u_base[0] * t, base_p1[1] + u_base[1] * t)
        
        # Draw height line from foot to apex
        self.draw_line(ctx, foot, apex, dashed=True)
        
        # If foot is outside base segment, draw extension line
        if t < 0:
            self.draw_line(ctx, base_p1, foot, dashed=True)
        elif t > base_len:
            self.draw_line(ctx, base_p2, foot, dashed=True)
        
        # Base direction - point TOWARD the base segment (inside corner of L)
        if t < 0:
            # Foot is left of base, point right toward base_p1
            marker_base_dir = u_base
        elif t > base_len:
            # Foot is right of base, point left toward base_p2
            marker_base_dir = (-u_base[0], -u_base[1])
        else:
            # Foot within base - default
            marker_base_dir = u_base
        
        # Perpendicular direction - point TOWARD apex (inside corner of L)
        if h > 0:
            marker_perp_dir = u_perp
        else:
            marker_perp_dir = (-u_perp[0], -u_perp[1])
        
        # G11: Proportional height marker size (10% of height line length, clamped)
        marker_size = ShapeDrawer.calc_height_marker_size(foot, apex)
        self.draw_right_angle_marker(ctx, foot, marker_base_dir, marker_perp_dir, size=marker_size)
        
        # Label positioned close to midpoint of height line
        offset = AppConstants.LABEL_OFFSET_DIMENSION * 0.5
        mid_x = (foot[0] + apex[0]) / 2 + u_base[0] * offset
        mid_y = (foot[1] + apex[1]) / 2 + u_base[1] * offset
        self.draw_text(mid_x, mid_y, label_key, use_background=True, ha="left", va="center")
    
    def _draw_custom(self, ctx: DrawingContext, transform: TransformState, 
                     params: dict) -> None:
        b = self.get_param_num(params, "Base Width", AppConstants.TRIANGLE_DEFAULT_BASE, 1.0)
        peak_offset = params.get("peak_offset", 0.5)
        
        l_side = params.get("Left Side_num")
        r_side = params.get("Right Side_num")
        if l_side is not None and l_side == 0:
            raise ValidationError("Left Side must be positive")
        if r_side is not None and r_side == 0:
            raise ValidationError("Right Side must be positive")
        # Both sides provided → compute height from law of cosines
        if l_side and r_side:
            if l_side + r_side <= b:
                raise ValidationError("Sides too short for given base")
            s = (b + l_side + r_side) / 2
            h = (2 * math.sqrt(max(0, s * (s - b) * (s - l_side) * (s - r_side)))) / b
            peak_x = (b**2 + l_side**2 - r_side**2) / (2 * b)
            peak_offset = peak_x / b
        elif l_side or r_side:
            # Exactly one side entered — ambiguous, user likely forgot the other
            raise ValidationError("Enter both Left Side and Right Side, or leave both blank to use Height")
        else:
            h = self.get_param_num(params, "Height", AppConstants.TRIANGLE_DEFAULT_HEIGHT, 1.0)

        top_x = b * peak_offset
        base_pts = self._get_base_vertices(b, h, top_x)
        label_keys = ["Base Width", "Right Side", "Left Side"]
        base_pts, label_keys, _ = self._rotate_to_base(base_pts, label_keys, transform.base_side)
        
        self._draw_triangle_common(
            ctx, transform, base_pts, label_keys,
            show_height=True, height_label_key="Height"
        )
        # Only draw right angle marker if peak is at left corner and shape not flipped
        if transform.base_side == 0 and abs(peak_offset) < 0.01:
            if not transform.flip_h and not transform.flip_v:
                # Calculate actual edge directions from transformed points
                base_dir = (base_pts[1][0] - base_pts[0][0], base_pts[1][1] - base_pts[0][1])
                left_dir = (base_pts[2][0] - base_pts[0][0], base_pts[2][1] - base_pts[0][1])
                self.draw_right_angle_marker(ctx, base_pts[0], base_dir, left_dir)
    
    def _draw_isosceles(self, ctx: DrawingContext, transform: TransformState,
                        params: dict) -> None:
        aspect = ctx.aspect_ratio
        b = AppConstants.ISOSCELES_DEFAULT_BASE * aspect
        h = AppConstants.ISOSCELES_DEFAULT_HEIGHT / aspect
        top_x = b / 2
        base_pts = self._get_base_vertices(b, h, top_x)
        label_keys = ["Base Label", "Right Label", "Left Label"]
        base_pts, label_keys, _ = self._rotate_to_base(
            base_pts, label_keys, transform.base_side
        )
        self._draw_triangle_common(
            ctx, transform, base_pts, label_keys,
            equal_sides=[1, 2],
            padding=1.5,
            show_height=True,
            height_label_key="Height"
        )
    
    def _draw_scalene(self, ctx: DrawingContext, transform: TransformState,
                      params: dict) -> None:
        aspect = ctx.aspect_ratio
        # Derive geometry from reference side lengths (b=7, l=3, r=5)
        # so the default shape matches Custom mode with those values
        b_ref = AppConstants.SCALENE_DEFAULT_BASE   # 7.0
        l_ref, r_ref = 3.0, 5.0
        s = (b_ref + l_ref + r_ref) / 2
        h_ref = (2 * math.sqrt(max(0, s * (s - b_ref) * (s - l_ref) * (s - r_ref)))) / b_ref
        peak_ref = (b_ref**2 + l_ref**2 - r_ref**2) / (2 * b_ref) / b_ref
        # Apply aspect slider as a fine-tune scale on the whole shape
        b = b_ref * aspect
        h = h_ref * aspect
        peak = params.get("peak_offset", peak_ref)
        top_x = b * peak
        base_pts = self._get_base_vertices(b, h, top_x)
        label_keys = ["Side A (Bottom)", "Side C (Right)", "Side B (Left)"]
        base_pts, label_keys, _ = self._rotate_to_base(
            base_pts, label_keys, transform.base_side
        )
        self._draw_triangle_common(
            ctx, transform, base_pts, label_keys,
            padding=1.5,
            show_height=True,
            height_label_key="Height"
        )
    
    def _draw_equilateral(self, ctx: DrawingContext, transform: TransformState,
                          params: dict) -> None:
        side = AppConstants.EQUILATERAL_DEFAULT_SIDE
        h = side * math.sqrt(3) / 2
        top_x = side / 2
        base_pts = self._get_base_vertices(side, h, top_x)
        all_labels = ["Side A (Bottom)", "Side C (Right)", "Side B (Left)"]
        label_keys = self.rotate_list(all_labels, transform.base_side)
        self._draw_triangle_common(
            ctx, transform, base_pts, label_keys,
            equal_sides=[0, 1, 2],
            padding=2
        )


@ShapeRegistry.register("Circle")
class CircleDrawer(ShapeDrawer, RadialLabelMixin):
    """Draws circles with radius and diameter labels."""

    def get_snap_anchors(self, ctx, transform, params):
        r = self.validate_radius_diameter(params)
        return [(0, r), (0, -r), (-r, 0), (r, 0), (0, 0)]

    def draw(self, ctx: DrawingContext, transform: TransformState, params: dict) -> None:
        radius = self.validate_radius_diameter(params)
        self.draw_circle(ctx, (0, 0), radius)
        self.set_limits(ctx, (-radius, radius), (-radius, radius))
        self.draw_radial_dimension_labels(ctx, radius, 0, "horizontal")


@ShapeRegistry.register("Parallelogram")
class ParallelogramDrawer(ShapeDrawer, PolygonLabelMixin):
    """Draws parallelograms with slope, flip, and height line support."""

    def get_snap_anchors(self, ctx, transform, params):
        mode = params.get("dimension_mode", "Custom")
        base = self.get_param_num(params, "Length", AppConstants.PARALLELOGRAM_DEFAULT_BASE, ctx.aspect_ratio)
        height = self.get_param_num(params, "Height", AppConstants.PARALLELOGRAM_DEFAULT_HEIGHT, ctx.aspect_ratio)
        slope = params.get("parallelogram_slope", AppConstants.PARALLELOGRAM_DEFAULT_SLOPE)
        offset = height * slope
        p1 = (0, 0)
        p2 = (base, 0)
        p3 = (base + offset, height)
        p4 = (offset, height)
        center = self.calculate_centroid([p1, p2, p3, p4])
        points = self.transform_points([p1, p2, p3, p4], center, transform)
        anchors = list(points)
        n = len(points)
        for i in range(n):
            mx = (points[i][0] + points[(i+1) % n][0]) / 2
            my = (points[i][1] + points[(i+1) % n][1]) / 2
            anchors.append((mx, my))
        anchors.append(self.calculate_centroid(points))
        return anchors
    
    def draw(self, ctx: DrawingContext, transform: TransformState, params: dict) -> None:
        mode = params.get("dimension_mode", "Default")
        slope = params.get("parallelogram_slope", 0.3)
        base_pts = self._get_parallelogram_points(params, mode, ctx, slope)
        label_keys = ["Bottom", "Right Side", "Top", "Left Side"]
        
        xs = [p[0] for p in base_pts]
        ys = [p[1] for p in base_pts]
        center = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)

        if transform.base_side != 0:
            base_pts = self.rotate_polygon_to_base(base_pts, transform.base_side)
            label_keys = self.rotate_list(label_keys, transform.base_side)
        
        points = self.transform_points(base_pts, center, transform)
        self.draw_polygon(ctx, points)
        x_range, y_range = self.calculate_bounds(points)
        self.set_limits(ctx, x_range, y_range)
        
        self._draw_height_line(ctx, points)
        self.draw_smart_labels(points, label_keys, ctx)
        
        # Store geometry hints for preset dim lines
        # points: [bottom-left, bottom-right, top-right, top-left] (default orientation)
        self.label_manager.geometry_hints["para_pts"] = list(points)

    def _draw_height_line(self, ctx: DrawingContext, points: Polygon) -> None:
        """Draw height line: internal by preference, otherwise from the furthest outer peak."""
        height_text, show_height = self.label_manager.get_entry_values("Height")
        if not (height_text.strip() and show_height) or len(points) < 4:
            return
        try:
            # Base is defined by points[0] and points[1]
            p0, p1 = points[0], points[1]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            base_len = math.sqrt(dx**2 + dy**2)
            if base_len < 0.01: return
            
            u_base = (dx/base_len, dy/base_len)
            u_perp = (-u_base[1], u_base[0])

            # Evaluate top vertices (points 2 and 3)
            candidates = []
            for i in [2, 3]:
                v = (points[i][0] - p0[0], points[i][1] - p0[1])
                t = v[0] * u_base[0] + v[1] * u_base[1]
                h = v[0] * u_perp[0] + v[1] * u_perp[1]
                is_internal = 0 <= t <= base_len
                # For external vertices, measure absolute distance from nearest endpoint
                if t < 0:
                    ext_dist = abs(t)
                elif t > base_len:
                    ext_dist = abs(t - base_len)
                else:
                    ext_dist = 0
                candidates.append({'pt': points[i], 't': t, 'h': h, 'internal': is_internal, 'ext': ext_dist})
            
            # Prefer internal vertices, then vertex with LARGEST extension (furthest outside)
            internals = [c for c in candidates if c['internal']]
            if internals:
                best = sorted(internals, key=lambda x: x['t'])[0]
            else:
                best = sorted(candidates, key=lambda x: x['ext'], reverse=True)[0]

            apex, t, h_val = best['pt'], best['t'], best['h']
            foot = (p0[0] + u_base[0] * t, p0[1] + u_base[1] * t)
            
            # Draw height line
            self.draw_line(ctx, foot, apex, dashed=True)
            
            # Draw base extension if needed
            if t < 0:
                self.draw_line(ctx, p0, foot, dashed=True)
            elif t > base_len:
                self.draw_line(ctx, p1, foot, dashed=True)

            # Right angle marker
            m_base_dir = u_base if t < base_len / 2 else (-u_base[0], -u_base[1])
            m_perp_dir = u_perp if h_val > 0 else (-u_perp[0], -u_perp[1])
            marker_size = ShapeDrawer.calc_height_marker_size(foot, apex)
            self.draw_right_angle_marker(ctx, foot, m_base_dir, m_perp_dir, size=marker_size)
            
            # Label
            h_off = AppConstants.LABEL_OFFSET_DIMENSION * 0.4
            lx = (foot[0] + apex[0]) / 2 + u_base[0] * h_off
            ly = (foot[1] + apex[1]) / 2 + u_base[1] * h_off
            self.draw_text(lx, ly, "Height", ha="left" if u_base[0] >= 0 else "right", va="center")
        except Exception as e:
            logger.error(f"Parallelogram height line error: {e}")

    def _get_parallelogram_points(self, params: dict, mode: str, ctx: DrawingContext, slope: float) -> Polygon:
        if mode == "Default": return self._get_default_points(ctx.aspect_ratio, slope)
        # Custom mode uses simplified inputs: Length, Height, Side
        base = params.get("Length_num")
        height = params.get("Height_num")
        side = params.get("Side_num")
        if err := self.validate_positive_params(params, ["Length", "Height", "Side"]):
            raise ValidationError(err)
        # Height and Side are mutually exclusive
        has_h = height is not None and height > 0
        has_s = side is not None and side > 0
        if has_h and has_s:
            raise ValidationError("Enter Height OR Side, not both")
        if base and has_h: return self._calculate_from_height(base, height, slope)
        if base and has_s: return self._calculate_from_side(base, side, slope)
        return self._get_default_points(ctx.aspect_ratio, slope)

    def _get_default_points(self, aspect_ratio: float, slope: float = 0.5) -> Polygon:
        base = AppConstants.PARALLELOGRAM_DEFAULT_BASE * aspect_ratio
        h = AppConstants.PARALLELOGRAM_DEFAULT_HEIGHT
        offset = h * slope
        return [(0, 0), (base, 0), (base + offset, h), (offset, h)]

    def _calculate_from_height(self, base: float, height: float, slope: float) -> Polygon:
        return [(0, 0), (base, 0), (base + (height * slope), height), (height * slope, height)]

    def _calculate_from_side(self, base: float, side: float, slope: float) -> Polygon:
        angle = math.pi / 4 + (slope * math.pi / 4)
        h = side * math.sin(angle)
        horz = side * math.cos(angle)
        if h < 0.5 or not math.isfinite(h):
            raise ValidationError("Invalid side length for given slope (height would be too small or infinite)")
        if abs(horz) > base * 10:
            raise ValidationError("Side length too long for given slope and base (creates extreme skew)")
        return [(0, 0), (base, 0), (base + horz, h), (horz, h)]

@ShapeRegistry.register("Trapezoid")
class TrapezoidDrawer(ShapeDrawer, PolygonLabelMixin):
    """Draws trapezoids with rotation, flip, and height line support."""

    def get_snap_anchors(self, ctx, transform, params):
        top = self.get_param_num(params, "Top Base", AppConstants.TRAPEZOID_DEFAULT_TOP, ctx.aspect_ratio)
        bot = self.get_param_num(params, "Bottom Base", AppConstants.TRAPEZOID_DEFAULT_BOTTOM, ctx.aspect_ratio)
        height = self.get_param_num(params, "Height", AppConstants.TRAPEZOID_DEFAULT_HEIGHT, ctx.aspect_ratio)
        offset = (bot - top) / 2
        p1 = (0, 0)
        p2 = (bot, 0)
        p3 = (bot - offset, height)
        p4 = (offset, height)
        center = self.calculate_centroid([p1, p2, p3, p4])
        points = self.transform_points([p1, p2, p3, p4], center, transform)
        anchors = list(points)
        n = len(points)
        for i in range(n):
            mx = (points[i][0] + points[(i+1) % n][0]) / 2
            my = (points[i][1] + points[(i+1) % n][1]) / 2
            anchors.append((mx, my))
        anchors.append(self.calculate_centroid(points))
        return anchors
    
    def draw(self, ctx: DrawingContext, transform: TransformState, params: dict) -> None:
        mode = params.get("dimension_mode", "Default")
        
        # Get base geometry - will raise ValidationError if invalid
        base_pts = self._get_trapezoid_points(params, mode, ctx.aspect_ratio)

        # Setup labels (Bottom is at y=0, Top is at y=h)
        side_labels = ["Bottom", "Right Leg", "Top", "Left Leg"]
        
        # Apply rotation if needed
        if transform.base_side != 0:
            base_pts = self.rotate_polygon_to_base(base_pts, transform.base_side)
            side_labels = self.rotate_list(side_labels, transform.base_side)
        
        # Recompute center from the (possibly rotated) points so that flip
        # pivots around the correct axis rather than the pre-rotation center.
        xs = [p[0] for p in base_pts]
        ys = [p[1] for p in base_pts]
        center = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
        
        # Apply flip transformations
        points = self.transform_points(base_pts, center, transform)
        
        # Draw shape
        self.draw_polygon(ctx, points)
        x_range, y_range = self.calculate_bounds(points)
        self.set_limits(ctx, x_range, y_range)
        
        # Draw height line and labels (errors should surface via PlotController for consistency)
        self._draw_height_line(ctx, points, transform.base_side)
        self.draw_smart_labels(points, side_labels, ctx)
        
        # Store geometry hints for preset dim lines
        self.label_manager.geometry_hints["trap_pts"] = list(points)

    def _draw_height_line(self, ctx: DrawingContext, points: Polygon, base_side: int = 0) -> None:
        """Draw height line using robust shortest-path logic inherited from Parallelogram."""
        height_text, show_height = self.label_manager.get_entry_values("Height")
        if not (height_text.strip() and show_height) or len(points) < 4:
            return
        try:
            p0, p1 = points[0], points[1]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            base_len = math.sqrt(dx**2 + dy**2)
            if base_len < 0.001: return
            u_base, u_perp = (dx/base_len, dy/base_len), (-dy/base_len, dx/base_len)

            projs = []
            for i in [3, 2]: # Evaluate top vertices
                v = (points[i][0] - p0[0], points[i][1] - p0[1])
                t, h = v[0] * u_base[0] + v[1] * u_base[1], v[0] * u_perp[0] + v[1] * u_perp[1]
                ext_dist = 0 if (0 <= t <= base_len) else min(abs(t), abs(t - base_len))
                projs.append({'pt': points[i], 't': t, 'h': h, 'ext': ext_dist})
            
            best = sorted(projs, key=lambda x: (x['ext'], x['t']))[0]
            apex, t, h_val = best['pt'], best['t'], best['h']
            foot = (p0[0] + u_base[0] * t, p0[1] + u_base[1] * t)
            
            self.draw_line(ctx, foot, apex, dashed=True)
            if t < -0.001: self.draw_line(ctx, p0, foot, dashed=True)
            elif t > base_len + 0.001: self.draw_line(ctx, p1, foot, dashed=True)

            m_base_dir = u_base if t < base_len / 2 else (-u_base[0], -u_base[1])
            m_perp_dir = u_perp if h_val > 0 else (-u_perp[0], -u_perp[1])
            marker_size = ShapeDrawer.calc_height_marker_size(foot, apex)
            self.draw_right_angle_marker(ctx, foot, m_base_dir, m_perp_dir, size=marker_size)
            
            h_off = AppConstants.LABEL_OFFSET_DIMENSION * 0.4
            self.draw_text((foot[0]+apex[0])/2 + u_base[0]*h_off, (foot[1]+apex[1])/2 + u_base[1]*h_off, 
                           "Height", ha="left" if u_base[0] >= 0 else "right", va="center")
        except Exception as e:
            logger.error(f"Trapezoid height line error: {e}")

    def _get_trapezoid_points(self, params: dict, mode: str,
                              aspect_ratio: float) -> Polygon:
        """Returns trapezoid points. Raises ValidationError if invalid."""
        # Default mode - no validation needed
        if mode == "Default":
            return self._get_default_points(aspect_ratio)
        
        # Custom mode - validate inputs (custom_labels: Top Base, Bottom Base, Height, Left Side, Right Side)
        t = params.get("Top Base_num")
        b = params.get("Bottom Base_num")
        h = params.get("Height_num")
        l = params.get("Left Side_num")
        r = params.get("Right Side_num")
        
        # Validate positive values
        for name, val in [("Top Base", t), ("Bottom Base", b), ("Height", h), 
                         ("Left Side", l), ("Right Side", r)]:
            if err := ShapeValidator.validate_positive(name, val):
                raise ValidationError(err)
        
        # Check for mutually exclusive inputs
        has_h = h is not None and h > 0
        has_one_side = (l is not None and l > 0) or (r is not None and r > 0)
        has_both_sides = (l is not None and l > 0) and (r is not None and r > 0)
        
        error = ShapeValidator.validate_mutually_exclusive("Height", has_h, "Sides", has_one_side)
        if error:
            raise ValidationError(error)
        
        # If using sides, both are required
        if has_one_side and not has_both_sides:
            raise ValidationError("Enter both Left Side and Right Side, or use Height instead")
        
        # Custom mode - need both parallel bases (already read as t and b above)
        if t is None or t <= 0 or b is None or b <= 0:
            raise ValidationError("Enter Top Base and Bottom Base lengths")
        
        if abs(t - b) < 0.001:
            raise ValidationError("Top and Bottom must be different")
        
        # Calculate from height - ensure top is at top (higher y)
        if h is not None:
            if h <= 0.01:
                raise ValidationError("Height too small")
            off = (b - t) / 2
            # Always draw with bottom at y=0 and top at y=h
            return [(0, 0), (b, 0), (b - off, h), (off, h)]
        
        # Calculate from legs
        if l and r:
            diff = b - t
            abs_diff = abs(diff)
            
            # Validate triangle inequality: sum of two sides must exceed third
            if l + r <= abs_diff + 0.01:
                raise ValidationError("Legs too short to connect Top and Bottom (violates triangle inequality)")
            
            # Check for extreme leg length that would create invalid geometry
            if l > abs_diff + 100 or r > abs_diff + 100:
                raise ValidationError("Leg length too large for given Top and Bottom widths")
            
            # Using absolute difference to support t > b
            try:
                off = (diff**2 + l**2 - r**2) / (2 * diff) if abs(diff) > 0.001 else 0
                h_sq = l**2 - off**2
                
                if h_sq < 0.0001 or not math.isfinite(h_sq):
                    raise ValidationError("Invalid leg lengths (no valid trapezoid possible)")
                
                h_val = math.sqrt(h_sq)
                if h_val < 0.01 or not math.isfinite(h_val):
                    raise ValidationError("Height too small or invalid")
            except (ValueError, ZeroDivisionError) as e:
                raise ValidationError(f"Cannot construct trapezoid with given dimensions: {str(e)}")
            
            return [(0, 0), (b, 0), (off + t, h_val), (off, h_val)]
        
        raise ValidationError("Provide Height or both Legs")

    def _get_default_points(self, aspect_ratio: float) -> Polygon:
        """Generate default trapezoid."""
        # Ensure default custom mode also has a shape if fields are empty
        h = AppConstants.TRAPEZOID_DEFAULT_HEIGHT * aspect_ratio
        b = AppConstants.TRAPEZOID_DEFAULT_BOTTOM
        t = AppConstants.TRAPEZOID_DEFAULT_TOP
        off = (b - t) / 2
        return [(0, 0), (b, 0), (b - off, h), (off, h)]


@ShapeRegistry.register("Polygon")
class PolygonDrawer(ShapeDrawer):
    """Draws regular polygons (pentagon, hexagon, octagon) selected via type buttons."""

    THETA_OFFSETS = {
        5: 1.5707963267948966,   # Pentagon (pi/2)
        6: 0,                    # Hexagon
        8: 0.39269908169872414,  # Octagon (pi/8)
    }
    NUM_SIDES_MAP = {
        PolygonType.PENTAGON.value: 5,
        PolygonType.HEXAGON.value: 6,
        PolygonType.OCTAGON.value: 8,
    }

    def get_snap_anchors(self, ctx, transform, params):
        num_sides = self._get_num_sides(params)
        base_points = self._calculate_vertices(num_sides, transform.base_side)
        points = self.transform_points(base_points, (0, 0), transform)
        anchors = [tuple(p) for p in points[:-1]]
        anchors.append((0.0, 0.0))
        return anchors

    def draw(self, ctx: DrawingContext, transform: TransformState, params: dict) -> None:
        num_sides = self._get_num_sides(params)
        base_points = self._calculate_vertices(num_sides, transform.base_side)
        points = self.transform_points(base_points, (0, 0), transform)
        self.draw_polygon(ctx, points)
        DrawingUtilities.draw_hash_marks(ctx, points[:-1] + [points[0]])
        x_range, y_range = self.calculate_bounds(points)
        self.set_limits(ctx, x_range, y_range)
        # Store geometry hints for dim line placement (exclude duplicate closing point)
        self.label_manager.geometry_hints["polygon_pts"] = list(points[:-1])

    def _get_num_sides(self, params: dict) -> int:
        polygon_type = params.get("polygon_type", PolygonType.PENTAGON.value)
        return self.NUM_SIDES_MAP.get(polygon_type, 5)

    def _calculate_vertices(self, num_sides: int, base_side: int) -> Polygon:
        base_theta_offset = self.THETA_OFFSETS.get(num_sides, 0)
        side_rotation = (2 * np.pi / num_sides) * base_side
        theta_offset = base_theta_offset + side_rotation
        theta = np.linspace(0, 2 * np.pi, num_sides + 1) + theta_offset
        radius = AppConstants.POLYGON_DEFAULT_RADIUS
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        return list(zip(x, y))


@ShapeRegistry.register("Sphere")
class SphereDrawer(ShapeDrawer, RadialLabelMixin):
    """Draws spheres with equator line and radius/diameter labels."""

    def get_snap_anchors(self, ctx, transform, params):
        r = self.validate_radius_diameter(params)
        return [(0, r), (0, -r), (-r, 0), (r, 0), (0, 0)]
    
    def draw(self, ctx: DrawingContext, transform: TransformState, params: dict) -> None:
        radius = self.validate_radius_diameter(params)

        self.draw_circle(ctx, (0, 0), radius)
        # Equator: back half (top, concave) is dashed, front half (bottom, convex) is solid
        ctx.ax.add_patch(patches.Arc((0, 0), radius * 2, radius * 0.6,
                        theta1=0, theta2=180, **ctx.dash_args))
        ctx.ax.add_patch(patches.Arc((0, 0), radius * 2, radius * 0.6,
                        theta1=180, theta2=360, **ctx.dim_args))
        
        self.set_limits(ctx, (-radius, radius), (-radius, radius))
        self.draw_radial_dimension_labels(ctx, radius, 0, "horizontal")


@ShapeRegistry.register("Hemisphere")
class HemisphereDrawer(ShapeDrawer, RadialLabelMixin):
    """Draws a hemisphere. Canonical orientation: dome facing up, flat base at bottom.
    All rotations and flips are applied geometrically by generate_plot after this draw.
    """

    def get_snap_anchors(self, ctx, transform, params):
        """Snap anchors in canonical (dome-up) space. generate_plot transforms them."""
        r = self.validate_radius_diameter(params)
        ratio = ctx.aspect_ratio
        dome_y = r * ratio
        return [(0, dome_y), (0, 0), (-r, 0), (r, 0)]

    def draw(self, ctx: DrawingContext, transform: TransformState, params: dict) -> None:
        """Draw canonical hemisphere: dome up, base at y=0. No flip/rotation logic."""
        r = self.validate_radius_diameter(params)
        ratio = ctx.aspect_ratio
        dome_y = r * ratio

        # Base ellipse: back half (top) dashed, front half (bottom) solid
        ctx.ax.add_patch(patches.Arc(
            (0, 0), r * 2, r * 0.6,
            theta1=0, theta2=180, **ctx.dash_args))
        ctx.ax.add_patch(patches.Arc(
            (0, 0), r * 2, r * 0.6,
            theta1=180, theta2=360,
            color=ctx.line_color, lw=AppConstants.DEFAULT_LINE_WIDTH))

        # Dome arc (upper half of ellipse scaled by ratio)
        ctx.ax.add_patch(patches.Arc(
            (0, 0), r * 2, r * 2 * ratio,
            theta1=0, theta2=180,
            color=ctx.line_color, lw=AppConstants.DEFAULT_LINE_WIDTH))

        # Bounds: dome top to base ellipse bottom
        cir_values = self.label_manager.get_entry_values("Circumference")
        show_cir = bool(cir_values[0].strip() and cir_values[1])
        ellipse_h = r * AppConstants.BASE_ELLIPSE_RATIO
        y_min = -(ellipse_h if show_cir else ellipse_h / 2)
        y_max = max(ellipse_h, dome_y)
        self.set_limits(ctx, (-r, r), (y_min, y_max))
        self.draw_radial_dimension_labels(
            ctx, r, 0, "horizontal",
            ellipse_ratio=AppConstants.RADIAL_LABEL_ELLIPSE_RATIO,
            base_position="bottom")


@ShapeRegistry.register("Cylinder")
class CylinderDrawer(ShapeDrawer, RadialLabelMixin):
    """Draws a cylinder. Canonical orientation: upright, open top facing up.
    All rotations and flips are applied geometrically by generate_plot after this draw.
    """

    def get_snap_anchors(self, ctx, transform, params):
        """Snap anchors in canonical (upright) space. generate_plot transforms them."""
        r = self.validate_radius_diameter(params)
        h = self.get_param_num(params, "Height", AppConstants.CYLINDER_DEFAULT_HEIGHT, ctx.aspect_ratio)
        hy = h / 2
        return [(0, hy), (0, -hy), (-r, 0), (r, 0), (0, 0)]

    def draw(self, ctx: DrawingContext, transform: TransformState, params: dict) -> None:
        """Draw canonical cylinder: upright, centred at origin. No flip/rotation logic."""
        radius = self.validate_radius_diameter(params)
        height = self.get_param_num(params, "Height", AppConstants.CYLINDER_DEFAULT_HEIGHT, ctx.aspect_ratio)
        top_y = height / 2
        bottom_y = -height / 2

        # Top ellipse (fully visible)
        self.draw_ellipse(ctx, (0, top_y), radius * 2, radius * AppConstants.BASE_ELLIPSE_RATIO)
        # Bottom ellipse: back half dashed, front half solid
        self.draw_hidden_arc(ctx, (0, bottom_y), radius * 2, radius * AppConstants.BASE_ELLIPSE_RATIO,
                             solid_angles=(180, 360), dashed_angles=(0, 180))
        # Side edges
        self.draw_line(ctx, (-radius, top_y), (-radius, bottom_y))
        self.draw_line(ctx, (radius, top_y), (radius, bottom_y))

        # Built-in height dim line
        self.draw_labeled_dimension(ctx, "Height", (0, top_y), (0, bottom_y),
                                    AppConstants.LABEL_OFFSET_DIMENSION, 0, "left", "center")

        # Geometry hints for preset dim lines
        self.label_manager.geometry_hints["height_y1"] = bottom_y
        self.label_manager.geometry_hints["height_y2"] = top_y
        self.label_manager.geometry_hints["height_x"] = 0
        self.label_manager.geometry_hints["radius"] = radius

        cir_values = self.label_manager.get_entry_values("Circumference")
        show_cir = cir_values[0].strip() != "" and cir_values[1]
        ellipse_h = radius * AppConstants.BASE_ELLIPSE_RATIO / 2
        y_min = bottom_y - (ellipse_h * 2 if show_cir else ellipse_h)
        y_max = top_y + ellipse_h
        self.set_limits(ctx, (-radius, radius), (y_min, y_max))
        self.draw_radial_dimension_labels(
            ctx, radius, bottom_y, "horizontal",
            ellipse_ratio=AppConstants.RADIAL_LABEL_ELLIPSE_RATIO,
            base_position="bottom")


@ShapeRegistry.register("Cone")
class ConeDrawer(ShapeDrawer, RadialLabelMixin):
    """Draws a cone. Canonical orientation: point up, base at bottom.
    All rotations and flips are applied geometrically by generate_plot after this draw.
    """

    def get_snap_anchors(self, ctx, transform, params):
        """Snap anchors in canonical (point-up) space. generate_plot transforms them."""
        r = self.validate_radius_diameter(params)
        h = self.get_param_num(params, "Height", AppConstants.CYLINDER_DEFAULT_HEIGHT, ctx.aspect_ratio)
        apex_y = h / 2
        base_y = -h / 2
        return [(-r, base_y), (r, base_y), (0, apex_y), (0, base_y), (0, 0)]

    def draw(self, ctx: DrawingContext, transform: TransformState, params: dict) -> None:
        """Draw canonical cone: point up, base centred at bottom. No flip/rotation logic."""
        radius = self.validate_radius_diameter(params)
        height = self.get_param_num(params, "Height", AppConstants.CYLINDER_DEFAULT_HEIGHT, ctx.aspect_ratio)
        apex_y = height / 2
        base_y = -height / 2

        # Base ellipse: back half dashed, front half solid
        self.draw_hidden_arc(ctx, (0, base_y), radius * 2, radius * AppConstants.BASE_ELLIPSE_RATIO,
                             solid_angles=(180, 360), dashed_angles=(0, 180))

        # Slant lines from base rim to apex (small offset to avoid overlapping ellipse)
        self.draw_line(ctx, (-radius, base_y + 0.1), (0, apex_y))
        self.draw_line(ctx, (radius, base_y + 0.1), (0, apex_y))

        # Built-in height dim line
        self.draw_labeled_dimension(ctx, "Height", (0, base_y), (0, apex_y),
                                    AppConstants.LABEL_OFFSET_DIMENSION, (base_y + apex_y) / 2,
                                    "left", "center")

        # Geometry hints for preset dim lines
        self.label_manager.geometry_hints["height_y1"] = base_y
        self.label_manager.geometry_hints["height_y2"] = apex_y
        self.label_manager.geometry_hints["height_x"] = 0
        self.label_manager.geometry_hints["radius"] = radius

        cir_values = self.label_manager.get_entry_values("Circumference")
        show_cir = cir_values[0].strip() != "" and cir_values[1]
        base_h = radius * AppConstants.BASE_ELLIPSE_RATIO
        if not show_cir:
            base_h /= 2
        self.set_limits(ctx, (-radius, radius), (base_y - base_h, apex_y))
        self.draw_radial_dimension_labels(
            ctx, radius, base_y, "horizontal",
            ellipse_ratio=AppConstants.RADIAL_LABEL_ELLIPSE_RATIO,
            base_position="bottom")


@ShapeRegistry.register("Rectangular Prism")
class RectangularPrismDrawer(ShapeDrawer):
    """Draws a rectangular prism. Canonical orientation: upright (length horizontal,
    height vertical, depth going back-right). All rotations and flips are applied
    geometrically by generate_plot after this draw.
    """
    DIMENSION_KEYS = ["Length (Front)", "Width (Side)", "Height"]

    def get_snap_anchors(self, ctx, transform, params):
        """Snap anchors in canonical space. generate_plot transforms them.

        Includes the outermost dimension-line endpoints so the composite
        selection box (built from these anchors) encloses the full drawn
        shape — including the Height, Length and Width dim lines that extend
        beyond the raw geometry corners.
        """
        l = self.get_param_num(params, "Length (Front)", AppConstants.PRISM_DEFAULT_LENGTH, ctx.aspect_ratio)
        w = self.get_param_num(params, "Width (Side)", AppConstants.PRISM_DEFAULT_WIDTH, ctx.aspect_ratio)
        h = self.get_param_num(params, "Height", AppConstants.PRISM_DEFAULT_HEIGHT, ctx.aspect_ratio)
        ds = ctx.aspect_ratio
        ddx = w * AppConstants.PRISM_DEPTH_X_RATIO * ds
        ddy = w * AppConstants.PRISM_DEPTH_Y_RATIO * ds
        # Canonical corners (no flip)
        f_bl = (0, 0)
        f_br = (l, 0)
        f_tr = (l, h)
        f_tl = (0, h)
        b_bl = (f_bl[0] + ddx, f_bl[1] + ddy)
        b_br = (f_br[0] + ddx, f_br[1] + ddy)
        b_tr = (f_tr[0] + ddx, f_tr[1] + ddy)
        cx = (f_bl[0] + b_tr[0]) / 2
        cy = (f_bl[1] + b_tr[1]) / 2

        all_ax = [f_bl[0], f_br[0], f_tr[0], f_tl[0], b_bl[0], b_br[0], b_tr[0]]
        all_ay = [f_bl[1], f_br[1], f_tr[1], f_tl[1], b_bl[1], b_br[1], b_tr[1]]
        sw_a = max(all_ax) - min(all_ax)
        sh_a = max(all_ay) - min(all_ay)
        off = max(sw_a, sh_a) * 0.08
        x_h_label = f_bl[0] - off * 1.5
        y_len_label = f_bl[1] - off * 1.5

        # Width (Side) dim line: replicate _draw_labels perpendicular-offset calculation
        ex = b_br[0] - f_br[0]
        ey = b_br[1] - f_br[1]
        seg_len = math.sqrt(ex * ex + ey * ey) or 1
        front_cx = (f_bl[0] + f_br[0] + f_tl[0] + f_tr[0]) / 4
        front_cy = (f_bl[1] + f_br[1] + f_tl[1] + f_tr[1]) / 4
        mid_raw_x = (f_br[0] + b_br[0]) / 2
        mid_raw_y = (f_br[1] + b_br[1]) / 2
        perp_x = -ey / seg_len * off
        perp_y =  ex / seg_len * off
        if (perp_x * (mid_raw_x - front_cx) + perp_y * (mid_raw_y - front_cy)) < 0:
            perp_x, perp_y = -perp_x, -perp_y
        w_label_x = mid_raw_x + perp_x * 1.5
        w_label_y = mid_raw_y + perp_y * 1.5

        return [
            f_bl, f_br, f_tr, f_tl,
            ((f_bl[0] + f_br[0]) / 2, f_bl[1]),
            ((f_tl[0] + f_tr[0]) / 2, f_tl[1]),
            (f_bl[0], (f_bl[1] + f_tl[1]) / 2),
            (f_br[0], (f_br[1] + f_tr[1]) / 2),
            (cx, cy),
            # Back-face vertices — required for bbox to cover full 3D depth projection
            b_bl, b_br, b_tr,
            # Dim-line extent anchors so the composite selection box covers the
            # full drawing, not just the raw geometry face corners.
            (x_h_label,  (f_bl[1] + f_tl[1]) / 2),  # height label left edge
            ((f_bl[0] + f_br[0]) / 2, y_len_label),   # length label bottom
            (w_label_x, w_label_y),                    # width label outer tip
        ]

    def draw(self, ctx: DrawingContext, transform: TransformState, params: dict) -> None:
        """Draw canonical prism: upright, in positive quadrant. No flip/rotation logic."""
        if error := self.validate_positive_params(params, self.DIMENSION_KEYS):
            raise ValidationError(error)

        l = self.get_param_num(params, "Length (Front)", AppConstants.PRISM_DEFAULT_LENGTH, ctx.aspect_ratio)
        w = self.get_param_num(params, "Width (Side)", AppConstants.PRISM_DEFAULT_WIDTH, ctx.aspect_ratio)
        h = self.get_param_num(params, "Height", AppConstants.PRISM_DEFAULT_HEIGHT, ctx.aspect_ratio)
        ds = ctx.aspect_ratio

        dx = w * AppConstants.PRISM_DEPTH_X_RATIO * ds
        dy = w * AppConstants.PRISM_DEPTH_Y_RATIO * ds

        f_bl = (0, 0)
        f_br = (l, 0)
        f_tr = (l, h)
        f_tl = (0, h)
        b_bl = (f_bl[0] + dx, f_bl[1] + dy)
        b_br = (f_br[0] + dx, f_br[1] + dy)
        b_tr = (f_tr[0] + dx, f_tr[1] + dy)
        b_tl = (f_tl[0] + dx, f_tl[1] + dy)

        # Front face
        ctx.ax.add_patch(patches.Polygon([f_bl, f_br, f_tr, f_tl], **ctx.line_args))
        # Visible back edges
        self.draw_line(ctx, f_tl, b_tl)
        self.draw_line(ctx, f_tr, b_tr)
        self.draw_line(ctx, b_tl, b_tr)
        self.draw_line(ctx, f_br, b_br)
        self.draw_line(ctx, b_br, b_tr)
        # Hidden back edges
        self.draw_line(ctx, f_bl, b_bl, dashed=True)
        self.draw_line(ctx, b_bl, b_br, dashed=True)
        self.draw_line(ctx, b_bl, b_tl, dashed=True)

        all_x = [f_bl[0], f_br[0], f_tr[0], f_tl[0], b_bl[0], b_br[0], b_tr[0], b_tl[0]]
        all_y = [f_bl[1], f_br[1], f_tr[1], f_tl[1], b_bl[1], b_br[1], b_tr[1], b_tl[1]]

        # Geometry hints for preset dim lines
        self.label_manager.geometry_hints["height_y1"] = f_bl[1]
        self.label_manager.geometry_hints["height_y2"] = f_tl[1]
        self.label_manager.geometry_hints["height_x"] = f_bl[0]
        self.label_manager.geometry_hints["prism_f_bl"] = f_bl
        self.label_manager.geometry_hints["prism_f_br"] = f_br
        self.label_manager.geometry_hints["prism_f_tl"] = f_tl
        self.label_manager.geometry_hints["prism_f_tr"] = f_tr
        self.label_manager.geometry_hints["prism_b_br"] = b_br
        self.label_manager.geometry_hints["prism_b_bl"] = b_bl
        self.label_manager.geometry_hints["prism_flip_h"] = False  # canonical
        self.set_limits(ctx, (min(all_x), max(all_x)), (min(all_y), max(all_y)))


@ShapeRegistry.register("Triangular Prism")
class TriangularPrismDrawer(ShapeDrawer):
    """Draws a triangular prism. Canonical orientation: triangular base facing front,
    triangle base edge at bottom, apex pointing up. 3-step rotation (120° steps).
    All rotations and flips are applied geometrically by generate_plot after this draw.
    """
    DIMENSION_KEYS = ["Base (Tri)", "Height (Tri)", "Length (Prism)"]

    def _compute_canonical_geometry(self, base: float, tri_h: float, length: float,
                                     depth_scale: float, peak_offset: float = 0.5) -> dict:
        """Compute canonical triangle vertices and back-face vertices (no rotation, no flip)."""
        peak_x = base * peak_offset
        p1 = (0.0, 0.0)
        p2 = (base, 0.0)
        p3 = (peak_x, tri_h)

        dx = length * AppConstants.PRISM_DEPTH_X_RATIO * depth_scale
        dy = length * AppConstants.PRISM_DEPTH_Y_RATIO * depth_scale

        b1 = (p1[0] + dx, p1[1] + dy)
        b2 = (p2[0] + dx, p2[1] + dy)
        b3 = (p3[0] + dx, p3[1] + dy)

        return {"p1": p1, "p2": p2, "p3": p3, "b1": b1, "b2": b2, "b3": b3}

    def get_snap_anchors(self, ctx, transform, params):
        """Snap anchors in canonical space. generate_plot transforms them.

        Includes the outermost dimension-line endpoints so the composite
        selection box encloses the full drawn shape including its built-in
        Base, Height and Length dim lines.
        """
        b = self.get_param_num(params, "Base (Tri)", AppConstants.TRI_PRISM_DEFAULT_BASE, ctx.aspect_ratio)
        th = self.get_param_num(params, "Height (Tri)", AppConstants.TRI_PRISM_DEFAULT_HEIGHT, ctx.aspect_ratio)
        ln = self.get_param_num(params, "Length (Prism)", AppConstants.TRI_PRISM_DEFAULT_LENGTH, ctx.aspect_ratio)
        peak = params.get("peak_offset", 0.5)
        g = self._compute_canonical_geometry(b, th, ln, ctx.aspect_ratio, peak)
        p1, p2, p3 = g["p1"], g["p2"], g["p3"]
        b1_s, b2, b3_s = g["b1"], g["b2"], g["b3"]
        mid_base = ((p1[0] + p2[0]) / 2, p1[1])
        cx = (p1[0] + p2[0] + p3[0]) / 3
        cy = (p1[1] + p2[1] + p3[1]) / 3

        all_xs = [p1[0], p2[0], p3[0], b1_s[0], b2[0], b3_s[0]]
        all_ys = [p1[1], p2[1], p3[1], b1_s[1], b2[1], b3_s[1]]
        sw = max(all_xs) - min(all_xs)
        sh = max(all_ys) - min(all_ys)
        off = max(sw, sh) * 0.08
        base_edge_y = min(p1[1], p2[1])
        y_base_label = base_edge_y - off * 1.5

        # Length (Prism) dim line: perpendicular offset from p2→b2 edge
        ex = b2[0] - p2[0]
        ey = b2[1] - p2[1]
        seg_len = math.sqrt(ex * ex + ey * ey) or 1
        tri_cx = (p1[0] + p2[0] + p3[0]) / 3
        tri_cy = (p1[1] + p2[1] + p3[1]) / 3
        cross = ex * (tri_cy - p2[1]) - ey * (tri_cx - p2[0])
        if cross > 0:
            perp_x =  ey / seg_len * off
            perp_y = -ex / seg_len * off
        else:
            perp_x = -ey / seg_len * off
            perp_y =  ex / seg_len * off
        mid_raw_x = (p2[0] + b2[0]) / 2
        mid_raw_y = (p2[1] + b2[1]) / 2
        l_label_x = mid_raw_x + perp_x * 1.5
        l_label_y = mid_raw_y + perp_y * 1.5

        return [
            p1, p2, p3, mid_base, (cx, cy),
            # Back-face vertices — required for bbox to cover full 3D depth projection
            b1_s, b2, b3_s,
            # Dim-line extent anchors
            ((p1[0] + p2[0]) / 2, y_base_label),  # base label bottom
            (l_label_x, l_label_y),                # length label outer tip
        ]

    def draw(self, ctx: DrawingContext, transform: TransformState, params: dict) -> None:
        """Draw canonical triangular prism. No flip/rotation logic."""
        mode = params.get("dimension_mode", "Default")
        peak = params.get("peak_offset", 0.5)

        if mode == "Custom":
            b = self.get_param_num(params, "Base", AppConstants.TRI_PRISM_DEFAULT_BASE, 1.0)
            l = self.get_param_num(params, "Length", AppConstants.TRI_PRISM_DEFAULT_LENGTH, 1.0)
            ls = params.get("Left Side_num")
            rs = params.get("Right Side_num")
            if ls is not None and ls == 0:
                raise ValidationError("Left Side must be positive")
            if rs is not None and rs == 0:
                raise ValidationError("Right Side must be positive")
            if ls and rs:
                if ls + rs <= b:
                    raise ValidationError("Sides too short")
                s = (b + ls + rs) / 2
                h = (2 * math.sqrt(max(0, s * (s - b) * (s - ls) * (s - rs)))) / b
                peak = ((b**2 + ls**2 - rs**2) / (2 * b)) / b
            else:
                h = self.get_param_num(params, "Height", AppConstants.TRI_PRISM_DEFAULT_HEIGHT, 1.0)
        else:
            b = self.get_param_num(params, "Base (Tri)", AppConstants.TRI_PRISM_DEFAULT_BASE, 1.0)
            l = self.get_param_num(params, "Length (Prism)", AppConstants.TRI_PRISM_DEFAULT_LENGTH, 1.0)
            h = self.get_param_num(params, "Height (Tri)", AppConstants.TRI_PRISM_DEFAULT_HEIGHT, 1.0)

        g = self._compute_canonical_geometry(b, h, l, ctx.aspect_ratio, peak)
        p1, p2, p3 = g["p1"], g["p2"], g["p3"]
        b1, b2, b3 = g["b1"], g["b2"], g["b3"]

        # All back edges dashed (standard simple prism — visibility computed by TriPrism)
        self.draw_line(ctx, p1, p2)
        self.draw_line(ctx, p2, p3)
        self.draw_line(ctx, p3, p1)
        self.draw_line(ctx, p3, b3)
        self.draw_line(ctx, p2, b2)
        self.draw_line(ctx, b3, b2)
        self.draw_line(ctx, p1, b1, dashed=True)
        self.draw_line(ctx, b3, b1, dashed=True)
        self.draw_line(ctx, b1, b2, dashed=True)

        all_x = [p1[0], p2[0], p3[0], b1[0], b2[0], b3[0]]
        all_y = [p1[1], p2[1], p3[1], b1[1], b2[1], b3[1]]
        self.set_limits(ctx, (min(all_x), max(all_x)), (min(all_y), max(all_y)))
        self._draw_tri_labels(ctx, p1, p2, p3, b1, b2, b3)

    def _draw_tri_labels(self, ctx: DrawingContext,
                         p1: tuple, p2: tuple, p3: tuple,
                         b1: tuple, b2: tuple, b3: tuple) -> None:
        """Store geometry hints. Base/Length use centralized standalone system.
        Only the internal Height (Tri) dashed line is drawn here."""
        dx_base = p2[0] - p1[0]
        dy_base = p2[1] - p1[1]
        base_len_sq = dx_base**2 + dy_base**2
        if base_len_sq > 0:
            t = ((p3[0] - p1[0]) * dx_base + (p3[1] - p1[1]) * dy_base) / base_len_sq
            foot = (p1[0] + t * dx_base, p1[1] + t * dy_base)
        else:
            foot = p1

        self.label_manager.geometry_hints["height_y1"] = foot[1]
        self.label_manager.geometry_hints["height_y2"] = p3[1]
        self.label_manager.geometry_hints["height_x"] = foot[0]
        self.label_manager.geometry_hints["tri_foot"] = foot
        self.label_manager.geometry_hints["tri_apex"] = p3
        self.label_manager.geometry_hints["prism_tri_p1"] = p1
        self.label_manager.geometry_hints["prism_tri_p2"] = p2
        self.label_manager.geometry_hints["prism_tri_p3"] = p3
        self.label_manager.geometry_hints["prism_tri_b1"] = b1
        self.label_manager.geometry_hints["prism_tri_b2"] = b2
        self.label_manager.geometry_hints["prism_tri_flip_h"] = False

        # Height (Tri): internal dashed line (stays builtin)
        ht_text, show_ht = self.label_manager.get_entry_values("Height (Tri)")
        if ht_text and show_ht:
            label_gap = DrawingUtilities.dim_label_gap_from_axes(ctx.ax)
            base_mid_x = (p1[0] + p2[0]) / 2
            base_mid_y = (p1[1] + p2[1]) / 2
            ctx.ax.plot([base_mid_x, p3[0]], [base_mid_y, p3[1]],
                        color="black", linewidth=AppConstants.DEFAULT_LINE_WIDTH,
                        linestyle="--", zorder=10)
            h_mid_x = (base_mid_x + p3[0]) / 2
            h_mid_y = (base_mid_y + p3[1]) / 2
            self.draw_text(h_mid_x + label_gap, h_mid_y, "Height (Tri)",
                           use_background=True, ha="left", va="center")


@ShapeRegistry.register("Tri Prism")
class TriPrismCompositeDrawer(TriangularPrismDrawer):
    """Triangular prism for composite figures with 4-step (90°) rotation.
    Inherits canonical draw from TriangularPrismDrawer. Only difference:
    - num_sides=4 in ShapeConfig (4 rotation orientations)
    - Snap anchors include bounding-box centre for alignment with RectangularPrism
    - draw() uses proper back-face visibility (outward normals)
    """

    def get_snap_anchors(self, ctx, transform, params):
        """Snap anchors in canonical space. generate_plot transforms them."""
        b = self.get_param_num(params, "Base (Tri)", AppConstants.TRI_PRISM_DEFAULT_BASE, ctx.aspect_ratio)
        th = self.get_param_num(params, "Height (Tri)", AppConstants.TRI_PRISM_DEFAULT_HEIGHT, ctx.aspect_ratio)
        ln = self.get_param_num(params, "Length (Prism)", AppConstants.TRI_PRISM_DEFAULT_LENGTH, ctx.aspect_ratio)
        peak = params.get("peak_offset", 0.5)
        g = self._compute_canonical_geometry(b, th, ln, ctx.aspect_ratio, peak)
        p1, p2, p3 = g["p1"], g["p2"], g["p3"]
        b1, b2, b3 = g["b1"], g["b2"], g["b3"]
        mid_base = ((p1[0] + p2[0]) / 2, p1[1])
        all_pts = [p1, p2, p3, b1, b2, b3]
        cx = (min(p[0] for p in all_pts) + max(p[0] for p in all_pts)) / 2
        cy = (min(p[1] for p in all_pts) + max(p[1] for p in all_pts)) / 2
        return [
            p1, p2, p3, mid_base, (cx, cy),
            # Back-face vertices — required for bbox to cover full 3D depth projection
            b1, b2, b3,
        ]

    def draw(self, ctx: DrawingContext, transform: TransformState, params: dict) -> None:
        """Draw canonical prism with correct back-face visibility via outward normals."""
        mode = params.get("dimension_mode", "Default")
        peak = params.get("peak_offset", 0.5)

        if mode == "Custom":
            b = self.get_param_num(params, "Base", AppConstants.TRI_PRISM_DEFAULT_BASE, 1.0)
            l = self.get_param_num(params, "Length", AppConstants.TRI_PRISM_DEFAULT_LENGTH, 1.0)
            ls = params.get("Left Side_num")
            rs = params.get("Right Side_num")
            if ls is not None and ls == 0:
                raise ValidationError("Left Side must be positive")
            if rs is not None and rs == 0:
                raise ValidationError("Right Side must be positive")
            if ls and rs:
                if ls + rs <= b:
                    raise ValidationError("Sides too short")
                s = (b + ls + rs) / 2
                h = (2 * math.sqrt(max(0, s * (s - b) * (s - ls) * (s - rs)))) / b
                peak = ((b**2 + ls**2 - rs**2) / (2 * b)) / b
            else:
                h = self.get_param_num(params, "Height", AppConstants.TRI_PRISM_DEFAULT_HEIGHT, 1.0)
        else:
            b = self.get_param_num(params, "Base (Tri)", AppConstants.TRI_PRISM_DEFAULT_BASE, 1.0)
            l = self.get_param_num(params, "Length (Prism)", AppConstants.TRI_PRISM_DEFAULT_LENGTH, 1.0)
            h = self.get_param_num(params, "Height (Tri)", AppConstants.TRI_PRISM_DEFAULT_HEIGHT, 1.0)

        g = self._compute_canonical_geometry(b, h, l, ctx.aspect_ratio, peak)
        p1, p2, p3 = g["p1"], g["p2"], g["p3"]
        b1, b2, b3 = g["b1"], g["b2"], g["b3"]

        depth_x = b1[0] - p1[0]
        depth_y = b1[1] - p1[1]

        def is_visible(pa, pb):
            nx = pb[1] - pa[1]
            ny = -(pb[0] - pa[0])
            return (nx * depth_x + ny * depth_y) > -1e-4

        v1 = is_visible(p1, p2)
        v2 = is_visible(p2, p3)
        v3 = is_visible(p3, p1)

        self.draw_line(ctx, p1, p2)
        self.draw_line(ctx, p2, p3)
        self.draw_line(ctx, p3, p1)
        self.draw_line(ctx, p1, b1, dashed=not (v1 or v3))
        self.draw_line(ctx, p2, b2, dashed=not (v1 or v2))
        self.draw_line(ctx, p3, b3, dashed=not (v2 or v3))
        self.draw_line(ctx, b1, b2, dashed=not v1)
        self.draw_line(ctx, b2, b3, dashed=not v2)
        self.draw_line(ctx, b3, b1, dashed=not v3)

        all_x = [p1[0], p2[0], p3[0], b1[0], b2[0], b3[0]]
        all_y = [p1[1], p2[1], p3[1], b1[1], b2[1], b3[1]]
        self.set_limits(ctx, (min(all_x), max(all_x)), (min(all_y), max(all_y)))
        self._draw_tri_labels(ctx, p1, p2, p3, b1, b2, b3)

@ShapeRegistry.register("Tri Triangle")
class TriTriangleCompositeDrawer(TriangleDrawer):
    """Triangle for composite figures with 4-step (90-degree) rotation.
    Inherits all draw logic from TriangleDrawer. The only difference is that
    ShapeConfig "Tri Triangle" uses num_sides=4, so the composite rotation
    pipeline applies 90-degree steps instead of 120-degree, matching Rectangle etc.
    """
    pass


@ShapeRegistry.register("Angle (Adjustable)")
class AdjustableAngleDrawer(ShapeDrawer, ArrowMixin):
    """Draws an adjustable angle with arc and label."""
    
    def draw(self, ctx: DrawingContext, transform: TransformState, params: dict) -> None:
        deg = (ctx.aspect_ratio / 3.5) * 350
        deg = max(5, deg)
        rad = np.radians(deg)
        self.draw_arrow(ctx, (0, 0), (3, 0))
        self.draw_arrow(ctx, (0, 0), (3 * np.cos(rad), 3 * np.sin(rad)))
        # Scale arc radius so label stays clear of rays at extreme angles
        arc_radius = max(0.8, min(1.5, 90 / max(deg, 1)))
        self._draw_angle_arc(ctx, (0, 0), arc_radius, 0, deg, "Angle Label")
        self.set_limits(ctx, (-4, 5), (-4, 5))
    
    def _draw_angle_arc(self, ctx: DrawingContext, center: Point, radius: float,
                        start_angle: float, end_angle: float, 
                        label_key: Optional[str] = None) -> None:
        arc = patches.Arc(center, radius * 2, radius * 2,
                          theta1=start_angle, theta2=end_angle,
                          ec=ctx.line_color, lw=AppConstants.DEFAULT_LINE_WIDTH)
        ctx.ax.add_patch(arc)
        
        if label_key:
            mid_angle = np.radians((start_angle + end_angle) / 2)
            label_r = radius + (AppConstants.LABEL_OFFSET_DIMENSION * 3)
            lx = center[0] + label_r * np.cos(mid_angle)
            ly = center[1] + label_r * np.sin(mid_angle)
            self.draw_text(lx, ly, label_key, ha="center", va="center", use_background=True)


@ShapeRegistry.register("Parallel Lines & Transversal")
class ParallelTransversalDrawer(ShapeDrawer):
    """Draws parallel lines with a transversal and angle labels."""
    
    def draw(self, ctx: DrawingContext, transform: TransformState, params: dict) -> None:
        gap = 2 * ctx.aspect_ratio
        self.draw_line(ctx, (-4, gap), (4, gap))
        self.draw_line(ctx, (-4, -gap), (4, -gap))
        
        # Line from (-2, gap+1) to (2, -gap-1)
        x1, y1 = -2, gap + 1
        x2, y2 = 2, -gap - 1
        self.draw_line(ctx, (x1, y1), (x2, y2))
        
        # Intersections
        # Line 1: y = gap
        # Transversal slope: m = (y2-y1)/(x2-x1) = (-2*gap-2)/4
        m = (-2 * gap - 2) / 4
        # Intersect 1: gap = m*(x - x1) + y1 => x = (gap - y1)/m + x1
        ix1 = (gap - y1) / m + x1
        # Intersect 2: -gap = m*(x - x1) + y1 => x = (-gap - y1)/m + x1
        ix2 = (-gap - y1) / m + x1
        
        self.set_limits(ctx, (-5, 5), (-gap - 2, gap + 2))
        
        off = 0.6
        self.draw_text(ix1 - off, gap + off/2, "Top Left", ha="right")
        self.draw_text(ix1 + off, gap - off/2, "Top Right", ha="left")
        self.draw_text(ix2 - off, -gap + off/2, "Bottom Left", ha="right")
        self.draw_text(ix2 + off, -gap - off/2, "Bottom Right", ha="left")


@ShapeRegistry.register("Complementary Angles")
class ComplementaryAnglesDrawer(ShapeDrawer, ArrowMixin):
    """Draws complementary angles (sum to 90°)."""
    
    def draw(self, ctx: DrawingContext, transform: TransformState, params: dict) -> None:
        deg = min(80, max(10, 45 * ctx.aspect_ratio))
        rad = np.radians(deg)
        self.draw_arrow(ctx, (0, 0), (4, 0))
        self.draw_arrow(ctx, (0, 0), (0, 4))
        self.draw_arrow(ctx, (0, 0), (4 * np.cos(rad), 4 * np.sin(rad)))
        DrawingUtilities.draw_right_angle_marker(ctx, (0, 0), (1, 0), (0, 1), size=0.5)
        self.set_limits(ctx, (-4, 5), (-4, 5))
        
        # Centered label positioning
        r_label = 2.5
        a1_mid = np.radians(deg / 2)
        a2_mid = np.radians((90 + deg) / 2)
        
        self.draw_text(r_label * np.cos(a1_mid), r_label * np.sin(a1_mid), "Angle 1", ha="center", va="center")
        self.draw_text(r_label * np.cos(a2_mid), r_label * np.sin(a2_mid), "Angle 2", ha="center", va="center")


@ShapeRegistry.register("Supplementary Angles")
class SupplementaryAnglesDrawer(ShapeDrawer, ArrowMixin):
    """Draws supplementary angles (sum to 180°)."""
    
    def draw(self, ctx: DrawingContext, transform: TransformState, params: dict) -> None:
        deg = min(170, max(10, 90 * ctx.aspect_ratio))
        rad = np.radians(deg)
        self.draw_arrow(ctx, (-4, 0), (4, 0))
        self.draw_point(ctx, (0, 0))
        self.draw_arrow(ctx, (0, 0), (3 * np.cos(rad), 3 * np.sin(rad)))
        self.set_limits(ctx, (-4, 5), (-4, 5))
        
        # Centered label positioning
        r_label = 2.0
        a_right_mid = np.radians(deg / 2)
        a_left_mid = np.radians((180 + deg) / 2)
        
        self.draw_text(r_label * np.cos(a_right_mid), r_label * np.sin(a_right_mid), "Left Angle", ha="center", va="center")
        self.draw_text(r_label * np.cos(a_left_mid), r_label * np.sin(a_left_mid), "Right Angle", ha="center", va="center")


@ShapeRegistry.register("Vertical Angles")
class VerticalAnglesDrawer(ShapeDrawer):
    """Draws vertical angles (opposite angles when two lines cross)."""
    
    def draw(self, ctx: DrawingContext, transform: TransformState, params: dict) -> None:
        dev = min(80, max(10, 30 * ctx.aspect_ratio))
        rad = np.radians(dev)
        x = 3
        y = 3 * np.tan(rad)
        self.draw_line(ctx, (-x, -y), (x, y))
        self.draw_line(ctx, (-x, y), (x, -y))
        self.set_limits(ctx, (-4, 5), (-4, 5))
        offset = AppConstants.LABEL_OFFSET_DIMENSION * 10
        self.draw_text(0, offset, "Top")
        self.draw_text(0, -offset, "Bottom")
        self.draw_text(-2, 0, "Left")
        self.draw_text(2, 0, "Right")


@ShapeRegistry.register("Line Segment")
class LineSegmentDrawer(ShapeDrawer):
    """Draws a line segment with labeled points."""
    
    def draw(self, ctx: DrawingContext, transform: TransformState, params: dict) -> None:
        a_pos = -4
        c_pos = 4
        y_pos = 0
        b_ratio = (ctx.aspect_ratio - AppConstants.SLIDER_MIN) / \
                  (AppConstants.SLIDER_MAX - AppConstants.SLIDER_MIN)
        b_pos = a_pos + b_ratio * (c_pos - a_pos)
        left_seg_x = (a_pos + b_pos) / 2
        right_seg_x = (b_pos + c_pos) / 2
        self.draw_line(ctx, (a_pos, y_pos), (c_pos, y_pos))
        self.draw_point(ctx, (a_pos, y_pos), size=AppConstants.POINT_SIZE_LARGE)
        self.draw_point(ctx, (b_pos, y_pos), size=AppConstants.POINT_SIZE_SMALL)
        self.draw_point(ctx, (c_pos, y_pos), size=AppConstants.POINT_SIZE_LARGE)
        point_offset = AppConstants.LABEL_OFFSET_DIMENSION * 3
        segment_offset = AppConstants.LABEL_OFFSET_DIMENSION * 4
        self._draw_point_label(ctx, a_pos, y_pos, "Point A", va="top", y_offset=-point_offset)
        self._draw_point_label(ctx, b_pos, y_pos, "Point B", va="top", y_offset=-point_offset)
        self._draw_point_label(ctx, c_pos, y_pos, "Point C", va="top", y_offset=-point_offset)
        self._draw_point_label(ctx, left_seg_x, y_pos, "Left Segment", 
                              va="bottom", y_offset=segment_offset)
        self._draw_point_label(ctx, right_seg_x, y_pos, "Right Segment", 
                              va="bottom", y_offset=segment_offset)
        self.set_limits(ctx, (-6, 6), (-3, 3))
    
    def _draw_point_label(self, ctx: DrawingContext, x: float, y: float,
                          key: str, va: str = "top", y_offset: float = 0) -> None:
        """Draw label for a specific point on a line segment."""
        self.draw_text(x, y + y_offset, key, ha="center", va=va)


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
    def restoring(self):
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

    def undo(self, _=None) -> Optional[dict]:
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

    def redo(self, _=None) -> Optional[dict]:
        if not self.redo_stack:
            return None
            
        state = self.redo_stack.pop()
        self.undo_stack.append(state)
        return state

    def can_undo(self) -> bool:
        return len(self.undo_stack) > 1

    def can_redo(self) -> bool:
        return len(self.redo_stack) > 0


@dataclass(frozen=True)
class ScaleSpec:
    key: str
    min_val: float
    max_val: float
    default: float


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


def capture_state(func):
    """Decorator to capture state before executing a function."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.history_manager.is_restoring:
            self._capture_current_state()
        return func(self, *args, **kwargs)
    return wrapper


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
        self._redraw_after_id: Optional[str] = None
        self._slider_after_id: Optional[str] = None
        self._capture_after_id: Optional[str] = None
        self._composite_positions: dict[int, tuple[float, float]] = {}
        self._composite_transforms: dict[int, dict] = {}  # {idx: {"flip_h": bool, "flip_v": bool, "base_side": int}}
        self._composite_selected: set[int] = set()
        self._composite_labels: list[dict] = []  # [{"text": "h", "x": 5.0, "y": 3.0}, ...]
        self._composite_label_drag: Optional[dict] = None
        self._composite_selected_label: Optional[int] = None
        self._composite_dim_lines: list[CompositeDimLine] = []  # [{"x1","y1","x2","y2","text","constraint":None/"v"/"h"}]
        self._composite_selected_dim: Optional[int] = None
        self._composite_dim_mode: Optional[str] = None  # None, "free", "height", "width", "radius"
        self._composite_dim_first_point: Optional[tuple] = None  # (x, y) of first click
        self._composite_edit_mode: Optional[dict] = None  # {"type": "label"/"dim", "idx": int}
        
        # Standalone freeform labels (composite-style for non-composite shapes)
        self._standalone_labels: list[dict] = []  # [{"text": "h", "x": 5.0, "y": 3.0}, ...]
        self._standalone_selected_label: Optional[int] = None
        self._standalone_label_bboxes: list[tuple] = []
        self._standalone_dim_lines: list[StandaloneDimLine] = []
        self._standalone_selected_dim: Optional[int] = None
        self._standalone_dim_endpoints: list[dict] = []
        self._standalone_dim_label_bboxes: list[tuple] = []
        self._standalone_dim_mode: bool = False
        self._standalone_dim_first_point: Optional[tuple] = None
        self._standalone_dim_preview_line = None
        self._builtin_dim_endpoints: list[dict] = []  # [{key, p1, p2}] populated per render
        self._standalone_edit_mode: Optional[dict] = None  # {"type": "label"/"dim"/"builtin", "idx": int}
        self._builtin_selected: Optional[str] = None  # Key name of selected built-in label, or None
        self._shape_bounds: Optional[dict] = None
        self._composite_drag_state: Optional[dict] = None
        self._composite_marquee: Optional[dict] = None  # {"start_x", "start_y", "rect_artist"}
        self._composite_event_ids: list[int] = []
        self._composite_snap_guides: list = []
        self._standalone_event_ids: list[int] = []
        self._label_drag_state: Optional[dict] = None
        self._label_hover_active: bool = False
        self._shape_pan_offset: tuple[float, float] = (0.0, 0.0)  # data-unit pan offset
        self._scale_after_id: Optional[str] = None
        self.label_manager = LabelManager()
        self.history_manager = HistoryManager()

        # Central slider/range manager
        self.scale_manager = ScaleManager(self.root)
        
        self.triangle_type_var = tk.StringVar(value="Custom")
        self.polygon_type_var = tk.StringVar(value=PolygonType.PENTAGON.value)
        self.dimension_mode_var = tk.StringVar(value="Default")
        self.show_hashmarks_var = tk.BooleanVar(value=False)

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
        controls_width = self.controls_row.winfo_reqwidth() + 20 if hasattr(self, 'controls_row') else 0
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
        controls_h = self.controls_row.winfo_reqheight() if hasattr(self, 'controls_row') else 0
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
            font=("Arial", 8)
        ).pack(expand=True, pady=2)
        
        self._create_category_selector()
        self._create_shape_selector()
        
        self._create_font_selector()
    
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
            self.center_container, text="Aa", font=("Arial", 9),
            relief=tk.SUNKEN, bg=AppConstants.ACTIVE_BUTTON_COLOR,
            command=self._set_font_sans
        )
        self.font_sans_btn.grid(row=0, column=6, padx=(4, 1), pady=5)
        self.font_serif_btn = tk.Button(
            self.center_container, text="Aa", font=("Times New Roman", 9),
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
        self.save_btn = tk.Button(self.center_container, text="Save", command=self.save_image)
        self.save_btn.grid(row=0, column=10, padx=1, sticky="e")
        self.save_btn.grid_remove()

        self.copy_btn = tk.Button(self.center_container, text="Copy", command=self.copy_to_clipboard)
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
        self.undo_btn = tk.Button(self.undo_redo_frame, text="Undo", state="disabled", command=self._undo_action)
        self.redo_btn = tk.Button(self.undo_redo_frame, text="Redo", state="disabled", command=self._redo_action)
        self.undo_btn.grid(row=0, column=0, padx=(0, 1), sticky="ew")
        self.redo_btn.grid(row=0, column=1, padx=(1, 0), sticky="ew")
        
        # Clear/Reset
        self.clear_workspace_btn = tk.Button(self.right_panel_frame, text="Clear Values & Labels", 
                                            command=self._clear_workspace, fg="red")
        
        # Help text
        self.mode_help_label = tk.Label(
            self.right_panel_frame, text="", bg=AppConstants.BG_COLOR, fg="gray",
            font=("Arial", 10), justify="left", wraplength=150
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
                    # B23/B28 fix: recalculate pixel aspect ratio AND reposition
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

        if hasattr(self, 'fig'):
            self.fig.clf()
        if hasattr(self, 'canvas'):
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
    
    def _get_shape_center(self) -> Optional[tuple[float, float]]:
        """Return the geometry-only center of the current shape, or None.
        
        Uses _geometry_center (computed from Polygon/Line2D/Arc artists only,
        excluding Text) to match the exact pivot used by rotate_axes_artists.
        This prevents annotation drift caused by text labels extending the
        bounding box asymmetrically.
        """
        gc = getattr(self, '_geometry_center', None)
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
            # then rotated around _canonical_geometry_center by the cumulative angle.
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
            canonical = getattr(self, '_canonical_geometry_center', None)
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

    def _update_rotation_label(self, config: Optional[ShapeConfig] = None) -> None:
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
        if hasattr(self, 'col_dimlines'):
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
        if hasattr(self, 'composite_transfer') and self.composite_transfer is not None:
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
        if hasattr(self, 'mode_help_label'):
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
            logger.debug(f"Spinbox update failed (likely widget destroyed): {e}")

    def _set_font_sans(self) -> None:
        self.font_family = "sans-serif"
        self.plot_controller.set_font_family("sans-serif")
        self.font_sans_btn.config(relief=tk.SUNKEN, bg=AppConstants.ACTIVE_BUTTON_COLOR)
        self.font_serif_btn.config(relief=tk.RAISED, bg=AppConstants.DEFAULT_BUTTON_COLOR)
        self.generate_plot()
        if not self.history_manager.is_restoring:
            self._capture_current_state()

    def _set_font_serif(self) -> None:
        self.font_family = "serif"
        self.plot_controller.set_font_family("serif")
        self.font_sans_btn.config(relief=tk.RAISED, bg=AppConstants.DEFAULT_BUTTON_COLOR)
        self.font_serif_btn.config(relief=tk.SUNKEN, bg=AppConstants.ACTIVE_BUTTON_COLOR)
        self.generate_plot()
        if not self.history_manager.is_restoring:
            self._capture_current_state()

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
            logger.debug(f"Line width spinbox update failed (likely widget destroyed): {e}")

    def show_welcome(self) -> None:
        # Keep only Category (cols 0,1) and Shape (cols 2,3) visible in top bar
        if hasattr(self, 'center_container'):
            for slave in self.center_container.grid_slaves(row=0):
                col = int(slave.grid_info()["column"])
                if col <= 3:
                    slave.grid()
                else:
                    slave.grid_remove()
        
        # Hide all control columns
        if hasattr(self, 'col_shape_type'): self.col_shape_type.grid_remove()
        if hasattr(self, 'col_transforms'): self.col_transforms.grid_remove()
        if hasattr(self, 'col_inputs'): self.col_inputs.grid_remove()
        if hasattr(self, 'col_dimlines'): self.col_dimlines.grid_remove()
        if hasattr(self, 'col_tools'): self.col_tools.grid_remove()
        
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
            logger.debug(f"Could not cancel after callback {attr_name}: {e}")
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
        if hasattr(self, '_scale_after_id') and self._scale_after_id:
            try:
                self.root.after_cancel(self._scale_after_id)
            except Exception:
                pass
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
        self.scale_manager.reset("peak_offset")
        self._update_triangle_button_styles()
        self.root.after_idle(self._rebuild_triangle_ui)
        if not self.history_manager.is_restoring:
            self._capture_current_state()

    def _update_triangle_button_styles(self) -> None:
        """Highlight active triangle button using standardized active state colors."""
        if not hasattr(self, 'tri_buttons'):
            return
        current = self.triangle_type_var.get()
        for name, btn in self.tri_buttons.items():
            if name == current:
                btn.config(relief="sunken", bg=AppConstants.ACTIVE_BUTTON_COLOR, fg="black")
            else:
                btn.config(relief="raised", bg=AppConstants.BG_COLOR, fg="gray")

    def on_triangle_type_change(self, event: Optional[Any] = None) -> None:
        if not self.history_manager.is_restoring:
            self._capture_current_state()
        self.scale_manager.reset("peak_offset")
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
                self.shape_options_frame, text="Default", width=10,
                command=lambda: self._set_dimension_mode("Default")
            )
            self.default_btn.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(0, 1))
            self.custom_btn = tk.Button(
                self.shape_options_frame, text="Custom", width=10,
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
    def _connect_standalone_drag(self) -> None:
        """Connect mouse events for standalone label dragging."""
        self._disconnect_standalone_drag()
        self._standalone_event_ids = [
            self.canvas.mpl_connect('button_press_event', self._on_standalone_label_press),
            self.canvas.mpl_connect('motion_notify_event', self._on_standalone_label_motion),
            self.canvas.mpl_connect('button_release_event', self._on_standalone_label_release),
            self.canvas.mpl_connect('motion_notify_event', self._on_standalone_label_hover),
        ]
    
    def _disconnect_standalone_drag(self) -> None:
        """Disconnect standalone label drag events."""
        for eid in self._standalone_event_ids:
            try:
                self.canvas.mpl_disconnect(eid)
            except Exception:
                pass
        self._standalone_event_ids = []
        self._label_drag_state = None
    
    def _pixels_to_data_pad(self, px: float) -> tuple[float, float]:
        """Convert a pixel size to data-unit padding (x, y) based on current axis state."""
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

    def _find_nearest_label(self, x: float, y: float, threshold: float = 0.8) -> Optional[str]:
        """Find the nearest label key to the given data coordinates."""
        if not self.label_manager.auto_positions:
            return None
        
        best_key = None
        best_dist = threshold
        
        # Get current axis limits to scale threshold proportionally
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        pad_x, pad_y = self._pixels_to_data_pad(3)
        scale = max(pad_x, pad_y)
        if scale > 0:
            best_dist = scale
        
        for key, pos_data in self.label_manager.auto_positions.items():
            lx, ly = pos_data[0], pos_data[1]
            # Use custom position if set
            if key in self.label_manager.custom_positions:
                lx, ly = self.label_manager.custom_positions[key]
            # Check if label is visible
            text, show = self.label_manager.get_entry_values(key)
            if not (text and show):
                continue
            dist = math.sqrt((x - lx)**2 + (y - ly)**2)
            if dist < best_dist:
                best_dist = dist
                best_key = key
        
        return best_key
    
    def _on_standalone_label_hover(self, event) -> None:
        """Change cursor when hovering over a draggable label or dim line."""
        if self._label_drag_state is not None:
            return
        if self._standalone_dim_mode:
            return
        if self._is_composite_shape():
            if getattr(self, '_label_hover_active', False):
                self.root.config(cursor="arrow")
                self._label_hover_active = False
            return
        if not self.shape_var.get():
            if getattr(self, '_label_hover_active', False):
                self.root.config(cursor="arrow")
                self._label_hover_active = False
            return
        if event.inaxes != self.ax:
            if getattr(self, '_label_hover_active', False):
                self.root.config(cursor="arrow")
                self._label_hover_active = False
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            if getattr(self, '_label_hover_active', False):
                self.root.config(cursor="arrow")
                self._label_hover_active = False
            return

        hit = False

        # Auto-positioned shape labels
        if self._find_nearest_label(x, y):
            hit = True

        # Standalone freeform label bboxes
        if not hit:
            for bbox in self._standalone_label_bboxes:
                if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                    hit = True
                    break

        # Standalone dim label bboxes
        if not hit:
            for bbox in self._standalone_dim_label_bboxes:
                if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                    hit = True
                    break

        # Standalone dim endpoints and body
        if not hit:
            hit_radius = max((self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) * 0.015, 0.2)
            for ep in self._standalone_dim_endpoints:
                p1, p2 = ep["p1"], ep["p2"]
                for pt in (p1, p2):
                    if math.sqrt((x - pt[0])**2 + (y - pt[1])**2) < hit_radius:
                        hit = True
                        break
                if not hit:
                    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                    seg_len_sq = dx*dx + dy*dy
                    if seg_len_sq > 0:
                        t = max(0.0, min(1.0, ((x - p1[0])*dx + (y - p1[1])*dy) / seg_len_sq))
                        dist = math.sqrt((x - p1[0] - t*dx)**2 + (y - p1[1] - t*dy)**2)
                        if dist < hit_radius:
                            hit = True
                if hit:
                    break

        # Builtin dim line endpoints and body
        if not hit:
            hit_radius = max((self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) * 0.015, 0.2)
            for ep in self._builtin_dim_endpoints:
                p1, p2 = ep["p1"], ep["p2"]
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                seg_len_sq = dx*dx + dy*dy
                if seg_len_sq > 0:
                    t = max(0.0, min(1.0, ((x - p1[0])*dx + (y - p1[1])*dy) / seg_len_sq))
                    dist = math.sqrt((x - p1[0] - t*dx)**2 + (y - p1[1] - t*dy)**2)
                    if dist < hit_radius:
                        hit = True
                        break

        if hit:
            if not getattr(self, '_label_hover_active', False):
                self.root.config(cursor="fleur")
                self._label_hover_active = True
        else:
            if getattr(self, '_label_hover_active', False):
                self.root.config(cursor="arrow")
                self._label_hover_active = False
    
    def _on_standalone_label_press(self, event) -> None:
        """Handle mouse press — start label drag if a label is hit."""
        if event.inaxes != self.ax or event.button != 1:
            return
        # Skip if in composite mode
        if self._is_composite_shape():
            return
        if not self.shape_var.get():
            return

        # Handle freeform dim line draw mode
        if self._standalone_dim_mode:
            x, y = event.xdata, event.ydata
            if x is None or y is None:
                return
            if self._standalone_dim_first_point is None:
                self._standalone_dim_first_point = (x, y)
            else:
                x1, y1 = self._standalone_dim_first_point
                text = self._standalone_label_entry.get().strip() or "f"
                mid_x = (x1 + x) / 2
                mid_y = (y1 + y) / 2
                if not self.history_manager.is_restoring:
                    self._capture_current_state()
                self._standalone_dim_lines.append({
                    "x1": x1, "y1": y1, "x2": x, "y2": y,
                    "text": text,
                    "preset_key": None,
                    "user_dragged": True,
                    "label_x": mid_x,
                    "label_y": mid_y,
                })
                self._standalone_selected_dim = len(self._standalone_dim_lines) - 1
                self._cancel_standalone_dim_mode()
                if not self.history_manager.is_restoring:
                    self._capture_current_state()
            return
        
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        
        # Double-click on a label or dim line label → enter edit mode
        if getattr(event, 'dblclick', False):
            # Check built-in labels (Circumference, Radius, Diameter, Height, etc.)
            hit_key = self._find_nearest_label(x, y)
            if hit_key and hit_key in AppConstants.BUILTIN_LABEL_KEYS:
                t, s = self.label_manager.get_entry_values(hit_key)
                if t and s:
                    self._builtin_edit_key = hit_key
                    self._enter_standalone_edit("builtin", 0)
                    return
            for idx in reversed(range(len(self._standalone_label_bboxes))):
                bbox = self._standalone_label_bboxes[idx]
                if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                    self._enter_standalone_edit("label", idx)
                    return
            for idx in reversed(range(len(self._standalone_dim_label_bboxes))):
                bbox = self._standalone_dim_label_bboxes[idx]
                if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                    self._enter_standalone_edit("dim", idx)
                    return
        
        # Check standalone dim line labels first
        for idx in reversed(range(len(self._standalone_dim_label_bboxes))):
            bbox = self._standalone_dim_label_bboxes[idx]
            if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                dim = self._standalone_dim_lines[idx]
                self._builtin_selected = None
                self._label_drag_state = {
                    "type": "standalone_dim_label",
                    "idx": idx,
                    "started": False,
                    "start_x": x,
                    "start_y": y,
                    "orig_x": dim.get("label_x", (dim["x1"] + dim["x2"]) / 2),
                    "orig_y": dim.get("label_y", (dim["y1"] + dim["y2"]) / 2),
                }
                return
        
        # Check standalone dim line endpoints
        for idx in reversed(range(len(self._standalone_dim_endpoints))):
            ep = self._standalone_dim_endpoints[idx]
            for ep_name, ep_pos in [("p1", ep["p1"]), ("p2", ep["p2"])]:
                if math.sqrt((x - ep_pos[0])**2 + (y - ep_pos[1])**2) < max(
                    (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) * 0.02, 0.3):
                    self._standalone_selected_dim = idx
                    self._standalone_selected_label = None
                    self._builtin_selected = None
                    self.generate_plot()
                    dim = self._standalone_dim_lines[idx]
                    self._label_drag_state = {
                        "type": "standalone_dim_endpoint",
                        "idx": idx,
                        "endpoint": ep_name,
                        "started": False,
                        "start_x": x,
                        "start_y": y,
                        "orig_x": ep_pos[0],
                        "orig_y": ep_pos[1],
                    }
                    return
        
        # Check standalone dim line body (for whole-line drag)
        for idx in reversed(range(len(self._standalone_dim_endpoints))):
            ep = self._standalone_dim_endpoints[idx]
            p1, p2 = ep["p1"], ep["p2"]
            # Point-to-segment distance check
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq > 0:
                t = max(0, min(1, ((x - p1[0]) * dx + (y - p1[1]) * dy) / seg_len_sq))
                proj_x = p1[0] + t * dx
                proj_y = p1[1] + t * dy
                dist = math.sqrt((x - proj_x)**2 + (y - proj_y)**2)
                hit_radius = max((self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) * 0.015, 0.2)
                if dist < hit_radius and 0.05 < t < 0.95:
                    self._standalone_selected_dim = idx
                    self._standalone_selected_label = None
                    self._builtin_selected = None
                    self.generate_plot()
                    dim = self._standalone_dim_lines[idx]
                    self._label_drag_state = {
                        "type": "standalone_dim_body",
                        "idx": idx,
                        "started": False,
                        "start_x": x,
                        "start_y": y,
                        "orig_x1": dim["x1"], "orig_y1": dim["y1"],
                        "orig_x2": dim["x2"], "orig_y2": dim["y2"],
                        "orig_lx": dim.get("label_x", (dim["x1"] + dim["x2"]) / 2),
                        "orig_ly": dim.get("label_y", (dim["y1"] + dim["y2"]) / 2),
                    }
                    return
        
        # Check built-in dim lines (body drag)
        hit_radius = max((self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) * 0.015, 0.2)
        for ep in self._builtin_dim_endpoints:
            p1, p2 = ep["p1"], ep["p2"]
            dx_seg, dy_seg = p2[0] - p1[0], p2[1] - p1[1]
            seg_len_sq = dx_seg * dx_seg + dy_seg * dy_seg
            if seg_len_sq > 0:
                t = max(0.0, min(1.0, ((x - p1[0]) * dx_seg + (y - p1[1]) * dy_seg) / seg_len_sq))
                proj_x = p1[0] + t * dx_seg
                proj_y = p1[1] + t * dy_seg
                dist = math.sqrt((x - proj_x) ** 2 + (y - proj_y) ** 2)
                if dist < hit_radius:
                    self._builtin_selected = ep["key"]
                    self._standalone_selected_dim = None
                    self._standalone_selected_label = None
                    self.generate_plot()
                    cur_off = self.label_manager.custom_dim_offsets.get(ep["key"], (0.0, 0.0))
                    self._label_drag_state = {
                        "type": "builtin_dim_body",
                        "key": ep["key"],
                        "started": False,
                        "start_x": x,
                        "start_y": y,
                        "orig_dx": cur_off[0],
                        "orig_dy": cur_off[1],
                    }
                    return

        # Check freeform standalone labels
        for idx in reversed(range(len(self._standalone_label_bboxes))):
            bbox = self._standalone_label_bboxes[idx]
            if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                self._label_drag_state = {
                    "type": "standalone_label",
                    "idx": idx,
                    "started": False,
                    "start_x": x,
                    "start_y": y,
                    "orig_x": self._standalone_labels[idx]["x"],
                    "orig_y": self._standalone_labels[idx]["y"],
                }
                return
        
        # Then check auto-positioned shape labels
        key = self._find_nearest_label(x, y)
        if key:
            if key in AppConstants.BUILTIN_LABEL_KEYS:
                # Toggle selection for built-in labels
                if self._builtin_selected == key:
                    self._builtin_selected = None
                else:
                    self._builtin_selected = key
                self._standalone_selected_dim = None
                self._standalone_selected_label = None
                self.generate_plot()
            else:
                self._builtin_selected = None
            self._label_drag_state = {"type": "auto_label", "key": key, "started": False, "start_x": x, "start_y": y}
            return
        
        # Clicked empty space — start a pan drag (drag shape around canvas)
        if self._standalone_edit_mode is not None:
            self._cancel_standalone_edit()
        if self._standalone_selected_label is not None or self._standalone_selected_dim is not None or self._builtin_selected is not None:
            self._standalone_selected_label = None
            self._standalone_selected_dim = None
            self._builtin_selected = None
            self.generate_plot()
        # Begin pan: record start position in PIXEL coords (not data coords)
        # B25 fix: xdata/ydata shift as the view is panned, so using them for
        # delta computation causes runaway shrinking. Pixel coords are stable.
        self._label_drag_state = {
            "type": "pan_shape",
            "started": False,
            "start_x": x, "start_y": y,
            "start_px": event.x, "start_py": event.y,
            "orig_pan_x": self._shape_pan_offset[0],
            "orig_pan_y": self._shape_pan_offset[1],
        }
    
    def _on_standalone_label_motion(self, event) -> None:
        """Handle mouse drag — reposition label or pan canvas."""
        # Ghost preview for freeform dim line draw mode
        if self._standalone_dim_mode and self._standalone_dim_first_point is not None:
            if event.inaxes == self.ax and event.xdata is not None:
                x1, y1 = self._standalone_dim_first_point
                x2, y2 = event.xdata, event.ydata
                if self._standalone_dim_preview_line:
                    try:
                        self._standalone_dim_preview_line.remove()
                    except (ValueError, AttributeError):
                        pass
                self._standalone_dim_preview_line = self.ax.plot(
                    [x1, x2], [y1, y2],
                    color="#4488ff", linestyle="--", linewidth=1.0, alpha=0.6, zorder=20
                )[0]
                self.canvas.draw_idle()
            return

        if self._label_drag_state is None:
            return
        if event.inaxes != self.ax:
            return
        
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        
        # Require minimum movement to start drag (prevents accidental repositioning on click)
        if not self._label_drag_state["started"]:
            dx = abs(x - self._label_drag_state["start_x"])
            dy = abs(y - self._label_drag_state["start_y"])
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            min_move = max(xlim[1] - xlim[0], ylim[1] - ylim[0]) * 0.01
            if dx < min_move and dy < min_move:
                return
            self._label_drag_state["started"] = True
            # Capture state before drag begins
            if not self.history_manager.is_restoring:
                self._capture_current_state()
            self.root.config(cursor="fleur")
        
        drag_type = self._label_drag_state.get("type", "auto_label")

        if drag_type == "pan_shape":
            # B25 fix: compute delta in pixels, convert to data units using current axis scale.
            # Using data coords (xdata) would drift because _apply_view_scale changes xlim/ylim
            # between events, shifting xdata for the same physical pixel position.
            dpx = event.x - self._label_drag_state["start_px"]
            dpy = event.y - self._label_drag_state["start_py"]
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            ax_bbox = self.ax.get_window_extent()
            ax_w_px = ax_bbox.width  if ax_bbox.width  > 1 else 1
            ax_h_px = ax_bbox.height if ax_bbox.height > 1 else 1
            data_per_px_x = (xlim[1] - xlim[0]) / ax_w_px
            data_per_px_y = (ylim[1] - ylim[0]) / ax_h_px
            # pan_x is added to the view CENTER in _apply_view_scale, so positive pan_x
            # shifts the window right — making the shape appear to move LEFT. To move the
            # shape right when dragging right (positive dpx), pan_x must decrease.
            # Y: screen y increases downward but data y increases upward, so signs match.
            self._shape_pan_offset = (
                self._label_drag_state["orig_pan_x"] - dpx * data_per_px_x,
                self._label_drag_state["orig_pan_y"] - dpy * data_per_px_y,
            )
            self._apply_view_scale_only()
            return
        
        if drag_type == "standalone_label":
            idx = self._label_drag_state["idx"]
            dx = x - self._label_drag_state["start_x"]
            dy = y - self._label_drag_state["start_y"]
            if idx < len(self._standalone_labels):
                self._standalone_labels[idx]["x"] = self._label_drag_state["orig_x"] + dx
                self._standalone_labels[idx]["y"] = self._label_drag_state["orig_y"] + dy
                self._standalone_selected_label = idx
                self.generate_plot()
        elif drag_type == "standalone_dim_endpoint":
            idx = self._label_drag_state["idx"]
            dx = x - self._label_drag_state["start_x"]
            dy = y - self._label_drag_state["start_y"]
            if idx < len(self._standalone_dim_lines):
                dim = self._standalone_dim_lines[idx]
                new_x = self._label_drag_state["orig_x"] + dx
                new_y = self._label_drag_state["orig_y"] + dy
                if dim.get("constraint") == "height":
                    new_x = self._label_drag_state["orig_x"]
                elif dim.get("constraint") == "width":
                    new_y = self._label_drag_state["orig_y"]
                ep = self._label_drag_state["endpoint"]
                if ep == "p1":
                    dim["x1"], dim["y1"] = new_x, new_y
                elif ep == "p2":
                    dim["x2"], dim["y2"] = new_x, new_y
                # Mark as user-positioned so snap doesn't overwrite the
                # dragged endpoint on the next generate_plot() call.
                dim["user_dragged"] = True
                self._standalone_selected_dim = idx
                self.generate_plot()
        elif drag_type == "standalone_dim_label":
            idx = self._label_drag_state["idx"]
            dx = x - self._label_drag_state["start_x"]
            dy = y - self._label_drag_state["start_y"]
            if idx < len(self._standalone_dim_lines):
                self._standalone_dim_lines[idx]["label_x"] = self._label_drag_state["orig_x"] + dx
                self._standalone_dim_lines[idx]["label_y"] = self._label_drag_state["orig_y"] + dy
                self._standalone_selected_dim = idx
                self.generate_plot()
        elif drag_type == "standalone_dim_body":
            idx = self._label_drag_state["idx"]
            dx = x - self._label_drag_state["start_x"]
            dy = y - self._label_drag_state["start_y"]
            if idx < len(self._standalone_dim_lines):
                dim = self._standalone_dim_lines[idx]
                dim["x1"] = self._label_drag_state["orig_x1"] + dx
                dim["y1"] = self._label_drag_state["orig_y1"] + dy
                dim["x2"] = self._label_drag_state["orig_x2"] + dx
                dim["y2"] = self._label_drag_state["orig_y2"] + dy
                dim["label_x"] = self._label_drag_state["orig_lx"] + dx
                dim["label_y"] = self._label_drag_state["orig_ly"] + dy
                dim["user_dragged"] = True
                self._standalone_selected_dim = idx
                self.generate_plot()
        elif drag_type == "builtin_dim_body":
            key = self._label_drag_state["key"]
            dx = x - self._label_drag_state["start_x"]
            dy = y - self._label_drag_state["start_y"]
            new_dx = self._label_drag_state["orig_dx"] + dx
            new_dy = self._label_drag_state["orig_dy"] + dy
            self.label_manager.custom_dim_offsets[key] = (new_dx, new_dy)
            self.generate_plot()
        else:
            key = self._label_drag_state["key"]
            self.label_manager.set_custom_position(key, x, y)
            self.generate_plot()
            
            # Draw blue selection highlight matching composite label style
            text, show = self.label_manager.get_entry_values(key)
            if text and show:
                font_size = self.font_size_var.get() if hasattr(self, 'font_size_var') else 12
                self.ax.text(x, y, text,
                            fontsize=font_size, color="#0066cc", fontweight="bold",
                            fontfamily=getattr(self, 'font_family', AppConstants.DEFAULT_FONT_FAMILY),
                            ha="center", va="center", zorder=AppConstants.LABEL_ZORDER + 2,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                     edgecolor="#0066cc", alpha=0.9))
                self.plot_controller.refresh()
    
    def _on_standalone_label_release(self, event) -> None:
        """Handle mouse release — finalize label position or toggle selection."""
        if self._label_drag_state is None:
            return
        
        drag_type = self._label_drag_state.get("type", "auto_label")

        if drag_type == "pan_shape":
            # Pan is fully handled in motion; on release just clear drag state.
            self._label_drag_state = None
            return
        was_dragged = self._label_drag_state.get("started", False)
        
        if drag_type == "standalone_label":
            idx = self._label_drag_state.get("idx")
            if was_dragged:
                self.generate_plot()
                if not self.history_manager.is_restoring:
                    self._capture_current_state()
            else:
                if self._standalone_selected_label == idx:
                    self._standalone_selected_label = None
                else:
                    self._standalone_selected_label = idx
                self._standalone_selected_dim = None
                self.generate_plot()
        elif drag_type in ("standalone_dim_endpoint", "standalone_dim_label", "standalone_dim_body"):
            idx = self._label_drag_state.get("idx")
            if was_dragged:
                # After dragging a freeform line, update its rel coords so it continues
                # to track shape transforms from the new dragged position.
                self.generate_plot()
                if not self.history_manager.is_restoring:
                    self._capture_current_state()
            else:
                if self._standalone_selected_dim == idx:
                    self._standalone_selected_dim = None
                else:
                    self._standalone_selected_dim = idx
                self._standalone_selected_label = None
                self.generate_plot()
        elif drag_type == "builtin_dim_body":
            if was_dragged:
                self.generate_plot()
                if not self.history_manager.is_restoring:
                    self._capture_current_state()
        else:
            if was_dragged:
                self.generate_plot()
                if not self.history_manager.is_restoring:
                    self._capture_current_state()
        
        self._label_drag_state = None
        self.root.config(cursor="arrow")
        self._label_hover_active = False
    
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
            
            for k, v in state.get("scales", {}).items():
                try: self.scale_manager.var(k).set(v)
                except Exception: pass
            # Restore pan offset
            pan = state.get("view_pan", [0.0, 0.0])
            self._shape_pan_offset = (float(pan[0]), float(pan[1]))

            for k, val in state.get("entries", {}).items():
                self.input_controller.set_entry_value(k, val)

            # Restore composite transfer list and positions if applicable
            composite_shapes = state.get("composite_shapes", [])
            if self._is_composite_shape(new_shape) and hasattr(self, 'composite_transfer') and self.composite_transfer is not None:
                self.composite_transfer.set_selected_shapes(composite_shapes)
                self._composite_positions = dict(state.get("composite_positions", {}))
                self._composite_transforms = {k: dict(v) for k, v in state.get("composite_transforms", {}).items()}
                self._composite_labels = [dict(lbl) for lbl in state.get("composite_labels", [])]
                self._composite_dim_lines = [dict(d) for d in state.get("composite_dim_lines", [])]
                self._composite_selected.clear()
                self._composite_selected_label = None
                self._composite_selected_dim = None

            # Restore line width
            if hasattr(self, 'line_width_var'):
                lw = state.get("line_width", AppConstants.DEFAULT_LINE_WIDTH)
                self.line_width = lw
                self.line_width_var.set(lw)
                self.plot_controller.set_line_width(lw)

            # Restore font size
            if hasattr(self, 'font_size_var'):
                fs = state.get("font_size", AppConstants.DEFAULT_FONT_SIZE)
                self.font_size = fs
                self.font_size_var.set(fs)
                self.plot_controller.set_font_size(fs)

            # Restore font family
            ff = state.get("font_family", AppConstants.DEFAULT_FONT_FAMILY)
            self.font_family = ff
            self.plot_controller.set_font_family(ff)
            if hasattr(self, 'font_sans_btn'):
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
                self.label_manager.circ_selected = False

            t = state.get("transforms", {"h": False, "v": False, "side": 0})
            self.transform_controller.flip_h, self.transform_controller.flip_v = t["h"], t["v"]
            self.transform_controller.base_side = t["side"]
            self.label_manager.custom_positions = state.get("custom_labels", {}).copy()
            self.label_manager.custom_dim_offsets = {k: tuple(v) for k, v in state.get("custom_dim_offsets", {}).items()}

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
        """Build current UI state snapshot for history."""
        # Include all scales (including view_scale) in history so undo restores zoom/pan
        scales_to_save = {k: self.scale_manager.var(k).get() for k in self.scale_manager.specs}
        
        # Capture composite transfer list contents only for composite shapes
        composite_selected = []
        composite_positions = {}
        composite_transforms = {}
        if (self._is_composite_shape(self.shape_var.get())
                and hasattr(self, 'composite_transfer') and self.composite_transfer is not None):
            composite_selected = self.composite_transfer.get_selected_shapes()
            composite_positions = dict(self._composite_positions)
            composite_transforms = {k: dict(v) for k, v in self._composite_transforms.items()}
        
        return {
            "cat": self.cat_var.get(),
            "shape": self.shape_var.get(),
            "font_size": self.font_size_var.get(),
            "line_width": self.line_width_var.get() if hasattr(self, 'line_width_var') else AppConstants.DEFAULT_LINE_WIDTH,
            "font_family": getattr(self, 'font_family', AppConstants.DEFAULT_FONT_FAMILY),
            "dim_mode": self.dimension_mode_var.get(),
            "tri_type": self.triangle_type_var.get(),
            "poly_type": self.polygon_type_var.get(),
            "scales": scales_to_save,
            "transforms": {
                "h": self.transform_controller.flip_h,
                "v": self.transform_controller.flip_v,
                "side": self.transform_controller.base_side
            },
            "custom_labels": self.label_manager.custom_positions.copy(),
            "custom_dim_offsets": {k: list(v) for k, v in self.label_manager.custom_dim_offsets.items()},
            "entries": {k: self.input_controller.get_entry_value(k) for k in self.input_controller.entries.keys()},
            "composite_shapes": composite_selected,
            "composite_positions": composite_positions,
            "composite_transforms": composite_transforms,
            "composite_labels": [dict(lbl) for lbl in self._composite_labels] if self._is_composite_shape(self.shape_var.get()) else [],
            "composite_dim_lines": [dict(d) for d in self._composite_dim_lines] if self._is_composite_shape(self.shape_var.get()) else [],
            "show_hashmarks": self.show_hashmarks_var.get() if hasattr(self, 'show_hashmarks_var') else False,
            "standalone_labels": [dict(lbl) for lbl in self._standalone_labels] if not self._is_composite_shape(self.shape_var.get()) else [],
            "standalone_dim_lines": [dict(d) for d in self._standalone_dim_lines] if not self._is_composite_shape(self.shape_var.get()) else [],
            "view_pan": list(getattr(self, '_shape_pan_offset', (0.0, 0.0))),
            **{
                f"{st_key}_text": self.label_manager.label_texts.get(lbl_key, "")
                for lbl_key, st_key in self.TOGGLE_LABEL_KEYS
            },
            **{
                f"{st_key}_vis": self.label_manager.label_visibility.get(lbl_key, False)
                for lbl_key, st_key in self.TOGGLE_LABEL_KEYS
            },
        }

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
        for name in ["Custom", "Isosceles", "Scalene", "Equilateral"]:
            btn = tk.Button(
                self.shape_options_frame, text=name, width=10, font=("Arial", 9),
                command=lambda n=name: self._set_triangle_type(n)
            )
            btn.pack(side=tk.TOP, fill=tk.X, padx=5, pady=1)
            self.tri_buttons[name] = btn

        self._update_triangle_button_styles()

        # Config-driven triangle sliders
        config = ShapeConfigProvider.get_triangle_config(triangle_type)
        if config.has_feature(ShapeFeature.SLIDER_PEAK):
            self._add_peak_offset_slider()
        if config.has_feature(ShapeFeature.SLIDER_SHAPE) or triangle_type in ["Isosceles", "Scalene"]:
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
                self.shape_options_frame, text=name, width=10, font=("Arial", 9),
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
        if not hasattr(self, 'poly_buttons'):
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
    def update_shape_list(self, _: Optional[Any] = None) -> None:
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
        if hasattr(self, 'col_shape_type'):
            self.col_shape_type.grid_remove()
        if hasattr(self, 'col_transforms'):
            self.col_transforms.grid_remove()
        if hasattr(self, 'col_inputs'):
            self.col_inputs.grid_remove()
        if hasattr(self, 'col_dimlines'):
            self.col_dimlines.grid_remove()
        if hasattr(self, 'col_tools'):
            self.col_tools.grid_remove()

        # Hide right-panel widgets until a shape is selected
        if hasattr(self, 'mode_help_label'):
            self.mode_help_label.pack_forget()
        if hasattr(self, 'clear_workspace_btn'):
            self.clear_workspace_btn.pack_forget()

        # Hide export buttons until a shape is selected
        if hasattr(self, 'save_btn'):
            self.save_btn.grid_remove()
        if hasattr(self, 'copy_btn'):
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
        if hasattr(self, 'col_inputs'):
            self.col_inputs.grid()
        if hasattr(self, 'col_dimlines'):
            self.col_dimlines.grid()
        if hasattr(self, 'col_tools'):
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
    
    def update_inputs(self, _: Optional[Any] = None) -> None:

        if hasattr(self, 'center_container'):
            # Re-show all controls in the top bar
            for slave in self.center_container.grid_slaves(row=0):
                slave.grid()
        
        # Explicitly show export buttons
        if hasattr(self, 'save_btn'):
            self.save_btn.grid(row=0, column=10, padx=1, sticky="e")
        if hasattr(self, 'copy_btn'):
            self.copy_btn.grid(row=0, column=11, padx=1, sticky="e")
        
        # Ensure font controls are visible
        if hasattr(self, 'font_label') and self.font_label.winfo_exists():
            self.font_label.grid()
        if hasattr(self, 'font_spin') and self.font_spin.winfo_exists():
            self.font_spin.grid()
        if hasattr(self, 'font_sans_btn') and self.font_sans_btn.winfo_exists():
            self.font_sans_btn.grid()
        if hasattr(self, 'font_serif_btn') and self.font_serif_btn.winfo_exists():
            self.font_serif_btn.grid()
        if hasattr(self, 'weight_label') and self.weight_label.winfo_exists():
            self.weight_label.grid()
        if hasattr(self, 'line_width_spin') and self.line_width_spin.winfo_exists():
            self.line_width_spin.grid()

        # Show control columns
        if hasattr(self, 'col_inputs'):
            self.col_inputs.grid()
        if hasattr(self, 'col_dimlines'):
            self.col_dimlines.grid()
        if hasattr(self, 'col_tools'):
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
        if hasattr(self, 'mode_help_label'):
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
        presets = self._DIM_PRESETS.get(shape, [])
        
        # Header
        tk.Label(self.col_dimlines, text="Labels and Lines", bg=AppConstants.BG_COLOR,
                 font=("Arial", 9, "bold")).pack(side=tk.TOP, anchor="center", pady=(0, 2))
        
        # Entry row: [         ] [+ Text]
        label_frame = tk.Frame(self.col_dimlines, bg=AppConstants.BG_COLOR)
        label_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))
        
        self._standalone_label_entry = tk.Entry(label_frame, width=10, font=("Arial", 10))
        self._standalone_label_entry.pack(side=tk.LEFT, padx=(0, 2), fill=tk.X, expand=True)
        self._standalone_label_entry.bind('<Return>', lambda e: self._confirm_standalone_label())
        
        self._standalone_text_btn = tk.Button(label_frame, text="+ Text", font=("Arial", 9),
                  command=self._confirm_standalone_label)
        self._standalone_text_btn.pack(side=tk.LEFT, padx=1)
        
        self._standalone_cancel_btn = tk.Button(label_frame, text="Cancel", font=("Arial", 8),
                  command=self._cancel_standalone_edit)
        # Hidden by default — shown only during edit mode
        
        # 2-column grid of preset line buttons + Free
        btn_grid = tk.Frame(self.col_dimlines, bg=AppConstants.BG_COLOR)
        btn_grid.pack(side=tk.TOP, fill=tk.X)
        btn_grid.columnconfigure(0, weight=1)
        btn_grid.columnconfigure(1, weight=1)
        
        all_buttons = [(p["label"], lambda k=p["key"], t=p["default_text"]: self._add_standalone_dim_preset(k, t)) for p in presets]
        
        self._standalone_free_btn = tk.Button(btn_grid, text="Free", font=("Arial", 8),
                  command=self._start_standalone_dim_mode)
        self._standalone_cancel_dim_btn = tk.Button(btn_grid, text="Cancel", font=("Arial", 8),
                  command=self._cancel_standalone_dim_mode)
        
        row = 0
        col = 0
        for label_text, cmd in all_buttons:
            tk.Button(btn_grid, text=label_text, font=("Arial", 8),
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
        tk.Button(self.col_dimlines, text="Delete Selected", font=("Arial", 8),
                  command=self._remove_standalone_annotation).pack(side=tk.TOP, pady=(2, 0))
        
        # Show col 3
        self.col_dimlines.grid()

    def _start_standalone_dim_mode(self) -> None:
        """Enter freeform dimension line draw mode for standalone shapes."""
        self._standalone_dim_mode = True
        self._standalone_dim_first_point = None
        self.root.config(cursor="crosshair")
        if hasattr(self, '_standalone_free_btn'):
            self._standalone_free_btn.grid_remove()
            self._standalone_cancel_dim_btn.grid()

    def _cancel_standalone_dim_mode(self) -> None:
        """Exit freeform dimension line draw mode."""
        self._standalone_dim_mode = False
        self._standalone_dim_first_point = None
        if self._standalone_dim_preview_line:
            try:
                self._standalone_dim_preview_line.remove()
            except (ValueError, AttributeError):
                pass
            self._standalone_dim_preview_line = None
        self.root.config(cursor="arrow")
        if hasattr(self, '_standalone_cancel_dim_btn'):
            self._standalone_cancel_dim_btn.grid_remove()
            self._standalone_free_btn.grid()
        self.generate_plot()

    def _add_standalone_label(self) -> None:
        """Add a freeform text label to the standalone canvas."""
        if not hasattr(self, '_standalone_label_entry'):
            return
        text = self._standalone_label_entry.get().strip()
        if not text:
            return
        
        # Place label at the center of the current view
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        cx = (xlim[0] + xlim[1]) / 2
        cy = (ylim[0] + ylim[1]) / 2
        
        self._standalone_labels.append({"text": text, "x": cx, "y": cy})
        self._standalone_label_entry.delete(0, tk.END)
        self._standalone_selected_label = len(self._standalone_labels) - 1
        
        self.generate_plot()
        if not self.history_manager.is_restoring:
            self._capture_current_state()

    def _add_standalone_dim_preset(self, preset_key: str, default_text: str) -> None:
        """Add a preset dimension line to the standalone canvas."""
        # Use text from entry if provided, otherwise try Custom value, otherwise preset default
        text = default_text
        if hasattr(self, '_standalone_label_entry'):
            entry_text = self._standalone_label_entry.get().strip()
            if entry_text:
                text = entry_text
                self._standalone_label_entry.delete(0, tk.END)
            else:
                # Populate from the matching Custom mode entry value (works for all shapes
                # with Custom mode, including Triangle Custom type)
                value_key = self._get_preset_value_key(preset_key)
                if value_key:
                    val = self.input_controller.get_entry_value(value_key).strip()
                    if val:
                        text = val
        
        # Circumference uses the built-in arc (not a straight dim line)
        if preset_key == "circumference":
            current_text, current_show = self.label_manager.get_entry_values("Circumference")
            if current_text.strip() and current_show:
                self.label_manager.set_label_text("Circumference", "", False)
            else:
                self.label_manager.set_label_text("Circumference", text, True)
            self._builtin_selected = None
            self._standalone_selected_dim = None
            self._standalone_selected_label = None
            self.generate_plot()
            if not self.history_manager.is_restoring:
                self._capture_current_state(force=True)
            return
        
        # Radius/Diameter/Height on radial shapes use built-in rendering with mutual exclusivity
        radial_shapes = {"Circle", "Sphere", "Hemisphere", "Cylinder", "Cone"}
        shape = self.shape_var.get()
        if preset_key in ("radius", "diameter") and shape in radial_shapes:
            label_key = "Radius" if preset_key == "radius" else "Diameter"
            other_key = "Diameter" if preset_key == "radius" else "Radius"
            
            current_text, current_show = self.label_manager.get_entry_values(label_key)
            
            if current_text.strip() and current_show:
                # Toggle OFF
                self.label_manager.set_label_text(label_key, "", False)
            else:
                # Toggle ON and turn the other OFF
                self.label_manager.set_label_text(label_key, text, True)
                self.label_manager.set_label_text(other_key, "", False)
                
            self._standalone_selected_dim = None
            self._standalone_selected_label = None
            self.generate_plot()
            if not self.history_manager.is_restoring:
                self._capture_current_state(force=True)
            return
        
        # Height/Slant on 3D shapes use built-in rendering
        builtin_3d_height = {"Cylinder", "Cone", "Hemisphere"}
        if preset_key == "height" and shape in builtin_3d_height:
            current_text, current_show = self.label_manager.get_entry_values("Height")
            if current_text.strip() and current_show:
                self.label_manager.set_label_text("Height", "", False)
            else:
                self.label_manager.set_label_text("Height", text, True)
            self._standalone_selected_dim = None
            self._standalone_selected_label = None
            self.generate_plot()
            if not self.history_manager.is_restoring:
                self._capture_current_state(force=True)
            return
        
        # Slant on Cone uses built-in rendering
        if preset_key == "slant" and shape == "Cone":
            current_text, current_show = self.label_manager.get_entry_values("Slant")
            if current_text.strip() and current_show:
                self.label_manager.set_label_text("Slant", "", False)
            else:
                self.label_manager.set_label_text("Slant", text, True)
            self._standalone_selected_dim = None
            self._standalone_selected_label = None
            self.generate_plot()
            if not self.history_manager.is_restoring:
                self._capture_current_state(force=True)
            return
        
        # Triangular Prism height uses built-in rendering (internal dashed line)
        if preset_key == "height" and shape in ("Triangular Prism", "Tri Prism"):
            label_key = "Height (Tri)"
            current_text, current_show = self.label_manager.get_entry_values(label_key)
            if current_text.strip() and current_show:
                self.label_manager.set_label_text(label_key, "", False)
            else:
                self.label_manager.set_label_text(label_key, text, True)
            self._standalone_selected_dim = None
            self._standalone_selected_label = None
            self.generate_plot()
            if not self.history_manager.is_restoring:
                self._capture_current_state(force=True)
            return
        
        # Rectangular Prism: all dim lines use centralized standalone system
        
        # Need a drawn shape to calculate endpoints — generate first if needed
        endpoints = self._calc_dim_line_endpoints(preset_key)
        if endpoints is None:
            return
        
        # G11: determine which endpoint touches the shape (for right-angle marker)
        # Exclude prisms — their "height" is an external side edge, not an internal altitude
        _perp_presets = {
            "height",
            "para_height", "trap_height",
        }
        _is_prism = self.shape_var.get() in ("Rectangular Prism", "Triangular Prism", "Tri Prism")
        right_angle_at = "p1" if (preset_key in _perp_presets and not _is_prism) else None

        self._standalone_dim_lines.append({
            "x1": endpoints["x1"], "y1": endpoints["y1"],
            "x2": endpoints["x2"], "y2": endpoints["y2"],
            "text": text,
            "constraint": endpoints.get("constraint"),
            "label_x": endpoints["label_x"],
            "label_y": endpoints["label_y"],
            "preset_key": preset_key,
            "user_dragged": False,
        })
        self._standalone_selected_dim = len(self._standalone_dim_lines) - 1
        self._standalone_selected_label = None
        self._builtin_selected = None
        
        self.generate_plot()
        if not self.history_manager.is_restoring:
            self._capture_current_state(force=True)

    def _get_preset_value_key(self, preset_key: str) -> Optional[str]:
        """Map a dim line preset key to the Custom mode entry key for value population."""
        shape = self.shape_var.get()
        tri_type = self.triangle_type_var.get() if hasattr(self, 'triangle_type_var') else "Custom"
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
        # Triangle Custom mode uses specific entry labels
        if shape == "Triangle" and tri_type == "Custom":
            tri_mapping = {
                "height": "Height", "width": "Base Width",
                "side_l": "Left Side", "side_r": "Right Side",
            }
            return tri_mapping.get(preset_key)
        shape_map = mapping.get(shape, {})
        return shape_map.get(preset_key)

    def _remove_standalone_annotation(self) -> None:
        """Remove the selected standalone label, dimension line, or circumference arc."""
        if self._builtin_selected:
            key = self._builtin_selected
            self.label_manager.label_texts.pop(key, None)
            self.label_manager.label_visibility.pop(key, None)
            self.label_manager.custom_positions.pop(key, None)
            self._builtin_selected = None
            self.generate_plot()
            if not self.history_manager.is_restoring:
                self._capture_current_state()
            return
        
        if self._standalone_selected_dim is not None:
            if self._standalone_selected_dim < len(self._standalone_dim_lines):
                self._standalone_dim_lines.pop(self._standalone_selected_dim)
            self._standalone_selected_dim = None
            self.generate_plot()
            if not self.history_manager.is_restoring:
                self._capture_current_state()
            return
        
        if self._standalone_selected_label is not None:
            if self._standalone_selected_label < len(self._standalone_labels):
                self._standalone_labels.pop(self._standalone_selected_label)
            self._standalone_selected_label = None
            self.generate_plot()
            if not self.history_manager.is_restoring:
                self._capture_current_state()

    def _confirm_standalone_label(self) -> None:
        """Add or update a standalone label/dim depending on edit mode."""
        if self._standalone_edit_mode is not None:
            self._apply_standalone_edit()
        else:
            self._add_standalone_label()

    def _enter_standalone_edit(self, edit_type: str, idx: int) -> None:
        """Enter edit mode: populate entry with existing text, swap button to Update."""
        if edit_type == "builtin":
            bk = getattr(self, '_builtin_edit_key', None)
            existing = self.label_manager.label_texts.get(bk, "") if bk else ""
        elif edit_type == "circ":
            existing = self.label_manager.label_texts.get("Circumference", "c")
        elif edit_type == "label" and idx < len(self._standalone_labels):
            existing = self._standalone_labels[idx]["text"]
        elif edit_type == "dim" and idx < len(self._standalone_dim_lines):
            existing = self._standalone_dim_lines[idx]["text"]
        else:
            return
        
        self._standalone_edit_mode = {"type": edit_type, "idx": idx}
        
        if hasattr(self, '_standalone_label_entry'):
            self._standalone_label_entry.delete(0, tk.END)
            self._standalone_label_entry.insert(0, existing)
            self._standalone_label_entry.focus_set()
            self._standalone_label_entry.select_range(0, tk.END)
        
        if hasattr(self, '_standalone_text_btn'):
            self._standalone_text_btn.config(text="Update")
        if hasattr(self, '_standalone_cancel_btn'):
            self._standalone_cancel_btn.pack(side=tk.LEFT, padx=1)

    def _apply_standalone_edit(self) -> None:
        """Apply the edited text to the label or dim line."""
        if self._standalone_edit_mode is None:
            return
        if not hasattr(self, '_standalone_label_entry'):
            return
        
        new_text = self._standalone_label_entry.get().strip()
        if not new_text:
            self._cancel_standalone_edit()
            return
        
        edit_type = self._standalone_edit_mode["type"]
        idx = self._standalone_edit_mode["idx"]
        
        if edit_type == "builtin":
            bk = getattr(self, '_builtin_edit_key', None)
            if bk:
                self.label_manager.set_label_text(bk, new_text, True)
        elif edit_type == "circ":
            self.label_manager.set_label_text("Circumference", new_text, True)
        elif edit_type == "label" and idx < len(self._standalone_labels):
            self._standalone_labels[idx]["text"] = new_text
        elif edit_type == "dim" and idx < len(self._standalone_dim_lines):
            self._standalone_dim_lines[idx]["text"] = new_text
        
        self._cancel_standalone_edit()
        self.generate_plot()
        if not self.history_manager.is_restoring:
            self._capture_current_state()

    def _cancel_standalone_edit(self) -> None:
        """Exit edit mode, restore + Text button."""
        self._standalone_edit_mode = None
        if hasattr(self, '_standalone_label_entry'):
            self._standalone_label_entry.delete(0, tk.END)
        if hasattr(self, '_standalone_text_btn'):
            self._standalone_text_btn.config(text="+ Text")
        if hasattr(self, '_standalone_cancel_btn'):
            self._standalone_cancel_btn.pack_forget()

    def _build_composite_ui(self, shape: str) -> None:
        """Build the transfer list UI for composite shapes."""
        # Determine which source shapes to offer
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
        
        # Create and pack the transfer list widget into the input_frame
        self.composite_transfer = CompositeTransferList(
            parent=self.input_frame,
            available_shapes=source_shapes,
            on_change_callback=self._on_composite_change
        )
        self.composite_transfer.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Label entry row
        label_frame = tk.Frame(self.input_frame, bg=AppConstants.BG_COLOR)
        label_frame.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))
        
        tk.Label(label_frame, text="Label:", bg=AppConstants.BG_COLOR,
                 font=("Arial", 9)).pack(side=tk.LEFT, padx=(0, 2))
        
        self._composite_label_entry = tk.Entry(label_frame, width=8, font=("Arial", 10))
        self._composite_label_entry.pack(side=tk.LEFT, padx=2)
        self._composite_label_entry.bind('<Return>', lambda e: self._confirm_composite_label())
        
        self._composite_text_btn = tk.Button(label_frame, text="+ Text", font=("Arial", 9),
                  command=self._confirm_composite_label)
        self._composite_text_btn.pack(side=tk.LEFT, padx=1)
        
        self._composite_cancel_btn = tk.Button(label_frame, text="Cancel", font=("Arial", 8),
                  command=self._cancel_composite_edit)
        # Hidden by default — shown only during edit mode
        
        tk.Button(label_frame, text="+ Line", font=("Arial", 9),
                  command=lambda: self._start_dim_line_mode("free")).pack(side=tk.LEFT, padx=1)
        
        # Dimension line preset buttons
        dim_frame = tk.Frame(self.input_frame, bg=AppConstants.BG_COLOR)
        dim_frame.pack(side=tk.TOP, fill=tk.X, pady=(2, 0))
        
        tk.Label(dim_frame, text="Presets:", bg=AppConstants.BG_COLOR,
                 font=("Arial", 8)).pack(side=tk.LEFT, padx=(0, 2))
        
        for preset_name, preset_key in [("Height", "height"), ("Width", "width"), ("Radius", "radius")]:
            tk.Button(dim_frame, text=preset_name, font=("Arial", 8),
                      command=lambda k=preset_key: self._start_dim_line_mode(k)).pack(side=tk.LEFT, padx=1)
        
        tk.Button(dim_frame, text="Delete", font=("Arial", 8),
                  command=self._remove_selected_annotation).pack(side=tk.LEFT, padx=(4, 1))
        
        # Status label for dimension line placement
        self._dim_status_label = tk.Label(self.input_frame, text="", bg=AppConstants.BG_COLOR,
                                           font=("Arial", 8, "italic"), fg="#0066cc")
        self._dim_status_label.pack(side=tk.TOP, fill=tk.X, pady=(1, 0))
        
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
        
        # Track shape list for removal index detection
        self._composite_prev_count = 0
        self._composite_prev_shapes = []
        
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
        """Called when the composite transfer list selection changes (after the change).
        
        Args:
            operation: Explicit operation tuple from the transfer list:
                ("add", new_index) — a shape was added at the given index
                ("remove", removed_index) — a shape was removed at the given index
                ("swap", idx_a, idx_b) — two shapes were swapped (reorder)
                None — bulk restore (e.g. undo/redo via set_selected_shapes)
        """
        if hasattr(self, 'composite_transfer') and self.composite_transfer is not None:
            new_shapes = self.composite_transfer.get_selected_shapes()
            n = len(new_shapes)
            
            if operation is not None:
                op_type = operation[0]
                
                if op_type == "swap":
                    swap_a, swap_b = operation[1], operation[2]
                    # Swap positions
                    pos_a = self._composite_positions.get(swap_a, (0.0, 0.0))
                    pos_b = self._composite_positions.get(swap_b, (0.0, 0.0))
                    self._composite_positions[swap_a] = pos_b
                    self._composite_positions[swap_b] = pos_a
                    # Swap transforms
                    tfm_a = self._composite_transforms.get(swap_a, {"flip_h": False, "flip_v": False, "base_side": 0})
                    tfm_b = self._composite_transforms.get(swap_b, {"flip_h": False, "flip_v": False, "base_side": 0})
                    self._composite_transforms[swap_a] = tfm_b
                    self._composite_transforms[swap_b] = tfm_a
                
                elif op_type == "remove":
                    removed_idx = operation[1]
                    # Shift indices above removed_idx down by 1
                    new_positions = {}
                    new_transforms = {}
                    for k, v in self._composite_positions.items():
                        if k < removed_idx:
                            new_positions[k] = v
                        elif k > removed_idx:
                            new_positions[k - 1] = v
                    for k, v in self._composite_transforms.items():
                        if k < removed_idx:
                            new_transforms[k] = v
                        elif k > removed_idx:
                            new_transforms[k - 1] = v
                    self._composite_positions = new_positions
                    self._composite_transforms = new_transforms
                
                elif op_type == "add":
                    new_idx = operation[1]
                    # Explicitly clear any stale position/transform for the new index
                    # so Phase 2 grid assignment always gives it a fresh cell.
                    # Without this, if a shape was previously at this index then
                    # removed (and indices shifted), the old absolute position can
                    # survive the shift and cause the new shape to appear on top of
                    # an existing shape rather than in an empty grid cell.
                    self._composite_positions.pop(new_idx, None)
                    self._composite_transforms.pop(new_idx, None)
            
            # Clean up stale entries
            self._composite_positions = {k: v for k, v in self._composite_positions.items() if k < n}
            self._composite_transforms = {k: v for k, v in self._composite_transforms.items() if k < n}
            self._composite_selected.clear()
            self._composite_prev_count = n
            self._composite_prev_shapes = list(new_shapes)
            self._composite_view_limits = None  # Force view recalculation
        
        self.generate_plot()
        if not self.history_manager.is_restoring:
            self._capture_current_state()

    def _connect_composite_drag(self) -> None:
        """Connect mouse events for composite shape dragging."""
        self._disconnect_composite_drag()
        self._composite_event_ids = [
            self.canvas.mpl_connect('button_press_event', self._on_composite_press),
            self.canvas.mpl_connect('motion_notify_event', self._on_composite_motion),
            self.canvas.mpl_connect('button_release_event', self._on_composite_release),
        ]
        # Note: Delete/BackSpace are handled by the global _on_delete_shortcut binding
        # (set in _bind_shortcuts), which already dispatches to _on_composite_delete
        # when _is_composite_shape() is True. No separate binding needed here.
    
    def _disconnect_composite_drag(self) -> None:
        """Disconnect composite drag mouse events."""
        for eid in self._composite_event_ids:
            try:
                self.canvas.mpl_disconnect(eid)
            except Exception:
                pass
        self._composite_event_ids = []
        self._composite_drag_state = None
        self._composite_marquee = None
    
    def _on_composite_press(self, event) -> None:
        """Handle mouse press in composite mode — start dragging if a shape is hit."""
        if event.inaxes != self.ax or event.button != 1:
            return
        if not self._is_composite_shape():
            return
        if not hasattr(self, '_composite_bboxes') or not self._composite_bboxes:
            return
        
        mx, my = event.xdata, event.ydata
        if mx is None or my is None:
            return
        
        # Double-click on a dim label or text label → enter edit mode
        # Check dim labels first (same priority as single-click hit testing)
        if getattr(event, 'dblclick', False):
            if hasattr(self, '_composite_dim_label_bboxes'):
                for idx in reversed(range(len(self._composite_dim_label_bboxes))):
                    bbox = self._composite_dim_label_bboxes[idx]
                    if bbox[0] <= mx <= bbox[2] and bbox[1] <= my <= bbox[3]:
                        self._enter_composite_edit("dim", idx)
                        return
            if hasattr(self, '_composite_label_bboxes'):
                for idx in reversed(range(len(self._composite_label_bboxes))):
                    bbox = self._composite_label_bboxes[idx]
                    if bbox[0] <= mx <= bbox[2] and bbox[1] <= my <= bbox[3]:
                        self._enter_composite_edit("label", idx)
                        return
        
        # Handle dimension line placement mode
        if self._composite_dim_mode is not None:
            if self._handle_dim_line_click(mx, my):
                return
        
        # Check for dimension line endpoint hit (for dragging endpoints)
        if hasattr(self, '_composite_dim_endpoints'):
            for idx, endpoints in enumerate(self._composite_dim_endpoints):
                for ep_key in ("p1", "p2"):
                    if ep_key not in endpoints:
                        continue
                    ex, ey = endpoints[ep_key]
                    if abs(mx - ex) < 0.5 and abs(my - ey) < 0.5:
                        self._composite_label_drag = {
                            "type": "dim_endpoint",
                            "idx": idx,
                            "endpoint": ep_key,  # "p1", "p2", or "label"
                            "start_x": mx,
                            "start_y": my,
                            "orig_x": ex,
                            "orig_y": ey,
                            "dragged": False
                        }
                        # Select this dimension line
                        self._composite_selected_dim = idx
                        self._composite_selected_label = None
                        self._composite_selected.clear()
                        if not self.history_manager.is_restoring:
                            self._capture_current_state()
                        self.root.config(cursor="fleur")
                        return

        # Check for dim line label hit first
        if hasattr(self, '_composite_dim_label_bboxes'):
            for idx in reversed(range(len(self._composite_dim_label_bboxes))):
                bbox = self._composite_dim_label_bboxes[idx]
                if bbox[0] <= mx <= bbox[2] and bbox[1] <= my <= bbox[3]:
                    # Set up drag state for the dimension line label
                    dim = self._composite_dim_lines[idx]
                    # Use stored label position, or default to midpoint if not set
                    label_x = dim.get("label_x", (dim["x1"] + dim["x2"]) / 2)
                    label_y = dim.get("label_y", (dim["y1"] + dim["y2"]) / 2)
                    
                    self._composite_label_drag = {
                        "type": "dim_label",
                        "idx": idx,
                        "start_x": mx,
                        "start_y": my,
                        "orig_x": label_x,
                        "orig_y": label_y,
                        "dragged": False
                    }
                    
                    # Select this dimension line
                    self._composite_selected_dim = idx
                    self._composite_selected_label = None
                    self._composite_selected.clear()
                    
                    if not self.history_manager.is_restoring:
                        self._capture_current_state()
                    self.root.config(cursor="fleur")
                    return

        # Check for dim line BODY hit (draggable line itself)
        if hasattr(self, '_composite_dim_lines'):
            for idx in reversed(range(len(self._composite_dim_lines))):
                dim = self._composite_dim_lines[idx]
                x1, y1 = dim["x1"], dim["y1"]
                x2, y2 = dim["x2"], dim["y2"]

                dx = x2 - x1
                dy = y2 - y1
                seg_len_sq = dx*dx + dy*dy
                if seg_len_sq == 0:
                    continue

                # Projection factor
                t = max(0.0, min(1.0, ((mx - x1)*dx + (my - y1)*dy) / seg_len_sq))
                proj_x = x1 + t*dx
                proj_y = y1 + t*dy

                dist = math.sqrt((mx - proj_x)**2 + (my - proj_y)**2)

                hit_radius = max((self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) * 0.015, 0.2)

                if dist < hit_radius and 0.05 < t < 0.95:
                    # Start dragging the whole dim line
                    self._composite_label_drag = {
                        "type": "dim_body",
                        "idx": idx,
                        "start_x": mx,
                        "start_y": my,
                        "orig_x1": x1,
                        "orig_y1": y1,
                        "orig_x2": x2,
                        "orig_y2": y2,
                        "orig_lx": dim.get("label_x", (x1 + x2) / 2),
                        "orig_ly": dim.get("label_y", (y1 + y2) / 2),
                        "dragged": False
                    }

                    self._composite_selected_dim = idx
                    self._composite_selected_label = None
                    self._composite_selected.clear()

                    if not self.history_manager.is_restoring:
                        self._capture_current_state()
                    self.root.config(cursor="fleur")
                    return

        
        # Check for label hit first (labels are on top)
        if hasattr(self, '_composite_label_bboxes'):
            for idx in reversed(range(len(self._composite_label_bboxes))):
                bbox = self._composite_label_bboxes[idx]
                if bbox[0] <= mx <= bbox[2] and bbox[1] <= my <= bbox[3]:
                    self._composite_label_drag = {
                        "idx": idx,
                        "start_x": mx,
                        "start_y": my,
                        "orig_x": self._composite_labels[idx]["x"],
                        "orig_y": self._composite_labels[idx]["y"],
                        "dragged": False
                    }
                    if not self.history_manager.is_restoring:
                        self._capture_current_state()
                    self.root.config(cursor="fleur")
                    return
        
        # Check for modifier key (Ctrl or Cmd)
        multi = bool(event.key and ("control" in event.key or "super" in event.key
                                     or "ctrl" in str(event.key).lower()))
        
        # Hit test: find which shape's bounding box contains the click
        for idx in reversed(range(len(self._composite_bboxes))):
            bbox = self._composite_bboxes[idx]
            if bbox[0] <= mx <= bbox[2] and bbox[1] <= my <= bbox[3]:

                # Record whether this shape was already selected BEFORE press so
                # the release handler can correctly decide whether a no-drag click
                # should select (first click) or deselect (second click on same shape).
                was_selected_before = idx in self._composite_selected

                # --- Selection correctness ---
                # If clicking inside an existing group selection, preserve the group
                # Otherwise, update selection normally
                if multi:
                    # Ctrl/Cmd click toggles membership
                    if idx in self._composite_selected:
                        self._composite_selected.discard(idx)
                    else:
                        self._composite_selected.add(idx)
                else:
                    # Plain click:
                    # If already part of a group, keep the group
                    if idx not in self._composite_selected:
                        self._composite_selected = {idx}

                # Clear label/dim selection when selecting shapes
                self._composite_selected_label = None
                self._composite_selected_dim = None
                self._update_composite_selection_ui()

                # Store original positions for ALL selected shapes (for group drag)
                orig_positions = {}
                for sel_idx in self._composite_selected:
                    orig_positions[sel_idx] = self._composite_positions.get(sel_idx, (0.0, 0.0))

                # For group drag: capture group bbox and original annotation positions
                # so labels/dim lines inside the group move as a unit with the shapes.
                orig_labels = []
                orig_dims = []
                if len(self._composite_selected) > 1:
                    all_sel = self._all_shapes_selected()
                    grp_bb = self._get_group_bbox()
                    if grp_bb != (0, 0, 0, 0):
                        lbl_idxs, dim_idxs = self._get_annotations_in_region(*grp_bb, include_all=all_sel)
                        orig_labels = [(i, self._composite_labels[i]["x"],
                                        self._composite_labels[i]["y"]) for i in lbl_idxs]
                        orig_dims = [(i,
                                      self._composite_dim_lines[i]["x1"], self._composite_dim_lines[i]["y1"],
                                      self._composite_dim_lines[i]["x2"], self._composite_dim_lines[i]["y2"],
                                      self._composite_dim_lines[i].get("label_x"),
                                      self._composite_dim_lines[i].get("label_y"))
                                     for i in dim_idxs]
                # For single-shape drag: capture annotations within this shape's bbox
                elif idx < len(self._composite_bboxes):
                    s_bbox = self._composite_bboxes[idx]
                    if s_bbox != (0, 0, 0, 0):
                        lbl_idxs, dim_idxs = self._get_annotations_in_region(*s_bbox)
                        orig_labels = [(i, self._composite_labels[i]["x"],
                                        self._composite_labels[i]["y"]) for i in lbl_idxs]
                        orig_dims = [(i,
                                      self._composite_dim_lines[i]["x1"], self._composite_dim_lines[i]["y1"],
                                      self._composite_dim_lines[i]["x2"], self._composite_dim_lines[i]["y2"],
                                      self._composite_dim_lines[i].get("label_x"),
                                      self._composite_dim_lines[i].get("label_y"))
                                     for i in dim_idxs]

                self._composite_drag_state = {
                    "idx": idx,
                    "start_x": mx,
                    "start_y": my,
                    "orig_pos": self._composite_positions.get(idx, (0.0, 0.0)),
                    "orig_positions": orig_positions,
                    "dragged": False,
                    "multi": multi,
                    "was_selected_before": was_selected_before,
                    "orig_labels": orig_labels,
                    "orig_dims": orig_dims,
                }

                # Capture state before drag begins
                if not self.history_manager.is_restoring:
                    self._capture_current_state()

                self.root.config(cursor="fleur")
                return
        
        # Deselect any selected label or dim line
        if self._composite_selected_label is not None or self._composite_selected_dim is not None:
            self._composite_selected_label = None
            self._composite_selected_dim = None
            self._update_composite_selection_ui()
            self.generate_plot()
        
        # Clicked empty space — start marquee selection
        self._composite_marquee = {
            "start_x": mx,
            "start_y": my,
            "rect_artist": None
        }
    
    def _on_composite_motion(self, event) -> None:
        """Handle mouse motion — update shape position during drag with snap, or draw marquee."""
        # Handle label or dimension endpoint drag
        if self._composite_label_drag is not None:
            if event.inaxes != self.ax:
                return
            mx, my = event.xdata, event.ydata
            if mx is None or my is None:
                return
            drag = self._composite_label_drag
            drag["dragged"] = True
            dx = mx - drag["start_x"]
            dy = my - drag["start_y"]
            idx = drag["idx"]
            
            if drag.get("type") == "dim_endpoint":
                # Dragging a dimension line endpoint
                if idx < len(self._composite_dim_lines):
                    dim = self._composite_dim_lines[idx]
                    ep = drag["endpoint"]
                    new_x = drag["orig_x"] + dx
                    new_y = drag["orig_y"] + dy
                    
                    # Apply constraints for preset lines
                    if dim.get("constraint") == "height":
                        # Vertical line: lock x coordinate
                        new_x = drag["orig_x"]
                    elif dim.get("constraint") == "width":
                        # Horizontal line: lock y coordinate
                        new_y = drag["orig_y"]
                    
                    if ep == "p1":
                        dim["x1"], dim["y1"] = new_x, new_y
                    elif ep == "p2":
                        dim["x2"], dim["y2"] = new_x, new_y
                    
                    # Force update of hitboxes/endpoints for the drag session
                    self.generate_plot()
                    self._composite_label_drag["dragged"] = True
            elif drag.get("type") == "dim_label":
                # Dragging the dimension line label (moves label only, not the line)
                if idx < len(self._composite_dim_lines):
                    dim = self._composite_dim_lines[idx]
                    # Update label position independently
                    dim["label_x"] = drag["orig_x"] + dx
                    dim["label_y"] = drag["orig_y"] + dy
                    self.generate_plot()

            elif drag.get("type") == "dim_body":
                # Dragging the ENTIRE dimension line (moves both endpoints together)
                if idx < len(self._composite_dim_lines):
                    dim = self._composite_dim_lines[idx]

                    # Move both endpoints by the same delta
                    dim["x1"] = drag["orig_x1"] + dx
                    dim["y1"] = drag["orig_y1"] + dy
                    dim["x2"] = drag["orig_x2"] + dx
                    dim["y2"] = drag["orig_y2"] + dy

                    # Also move the label if it was manually positioned
                    if "label_x" in dim and "label_y" in dim:
                        dim["label_x"] = drag["orig_lx"] + dx
                        dim["label_y"] = drag["orig_ly"] + dy

                    self.generate_plot()

            else:
                # Dragging a text label
                if idx < len(self._composite_labels):
                    self._composite_labels[idx]["x"] = drag["orig_x"] + dx
                    self._composite_labels[idx]["y"] = drag["orig_y"] + dy
                    self.generate_plot()
            return
        
        # Handle dimension line preview
        if self._composite_dim_mode is not None and self._composite_dim_first_point is not None:
            if event.inaxes == self.ax and event.xdata is not None:
                mx, my = event.xdata, event.ydata
                x1, y1 = self._composite_dim_first_point
                x2, y2 = mx, my
                
                if self._composite_dim_mode == "height":
                    x2 = x1
                elif self._composite_dim_mode == "width":
                    y2 = y1
                
                # Remove old preview
                if hasattr(self, '_dim_preview_line') and self._dim_preview_line:
                    try:
                        self._dim_preview_line.remove()
                    except (ValueError, AttributeError):
                        pass
                
                self._dim_preview_line = self.ax.plot(
                    [x1, x2], [y1, y2],
                    color="#4488ff", linestyle="--", linewidth=1.0, alpha=0.6, zorder=20
                )[0]
                self.canvas.draw_idle()
            return
        
        # Handle marquee selection drawing
        if self._composite_marquee is not None:
            if event.inaxes != self.ax:
                return
            mx, my = event.xdata, event.ydata
            if mx is None or my is None:
                return
            
            m = self._composite_marquee
            x0, y0 = m["start_x"], m["start_y"]
            
            # Remove old rectangle if any
            if m["rect_artist"] is not None:
                try:
                    m["rect_artist"].remove()
                except (ValueError, AttributeError):
                    pass
            
            # Draw selection rectangle
            rx = min(x0, mx)
            ry = min(y0, my)
            rw = abs(mx - x0)
            rh = abs(my - y0)
            m["rect_artist"] = self.ax.add_patch(patches.Rectangle(
                (rx, ry), rw, rh,
                edgecolor="#4488ff", facecolor="#4488ff",
                alpha=0.12, linewidth=1.2, linestyle="-", zorder=25
            ))
            self.canvas.draw_idle()
            return
        
        if self._composite_drag_state is None:
            # Hover cursor: fleur over any draggable target, arrow otherwise
            if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
                mx, my = event.xdata, event.ydata
                hit = False
                hit_radius = max((self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) * 0.015, 0.2)
                if hasattr(self, '_composite_label_bboxes'):
                    for bbox in self._composite_label_bboxes:
                        if bbox[0] <= mx <= bbox[2] and bbox[1] <= my <= bbox[3]:
                            hit = True
                            break
                if not hit and hasattr(self, '_composite_dim_label_bboxes'):
                    for bbox in self._composite_dim_label_bboxes:
                        if bbox[0] <= mx <= bbox[2] and bbox[1] <= my <= bbox[3]:
                            hit = True
                            break
                if not hit and hasattr(self, '_composite_dim_endpoints'):
                    for ep in self._composite_dim_endpoints:
                        p1, p2 = ep["p1"], ep["p2"]
                        for pt in (p1, p2):
                            if math.sqrt((mx - pt[0])**2 + (my - pt[1])**2) < hit_radius:
                                hit = True
                                break
                        if not hit:
                            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                            seg_len_sq = dx*dx + dy*dy
                            if seg_len_sq > 0:
                                t = max(0.0, min(1.0, ((mx - p1[0])*dx + (my - p1[1])*dy) / seg_len_sq))
                                dist = math.sqrt((mx - p1[0] - t*dx)**2 + (my - p1[1] - t*dy)**2)
                                if dist < hit_radius:
                                    hit = True
                        if hit:
                            break
                if not hit and hasattr(self, '_composite_bboxes'):
                    for bbox in self._composite_bboxes:
                        if bbox != (0, 0, 0, 0) and bbox[0] <= mx <= bbox[2] and bbox[1] <= my <= bbox[3]:
                            hit = True
                            break
                desired = "fleur" if hit else "arrow"
                if self.root.cget("cursor") != desired:
                    self.root.config(cursor=desired)
            else:
                if self.root.cget("cursor") not in ("crosshair", "fleur"):
                    self.root.config(cursor="arrow")
            return
        if event.inaxes != self.ax:
            return
        mx, my = event.xdata, event.ydata
        if mx is None or my is None:
            return
        
        drag = self._composite_drag_state
        idx = drag["idx"]
        dx = mx - drag["start_x"]
        dy = my - drag["start_y"]
        
        new_x = drag["orig_pos"][0] + dx
        new_y = drag["orig_pos"][1] + dy
        
        # Cache the shape's half-dimensions on first motion event.
        # With absolute position storage, the shape's current canvas center is
        # simply (bbox_cx, bbox_cy) and matches orig_pos directly.
        if "half_w" not in drag and hasattr(self, '_composite_bboxes') and idx < len(self._composite_bboxes):
            bbox = self._composite_bboxes[idx]
            if bbox != (0, 0, 0, 0):
                drag["half_w"] = (bbox[2] - bbox[0]) / 2
                drag["half_h"] = (bbox[3] - bbox[1]) / 2
        
        # Calculate snap against non-selected shapes
        snap_dx = 0.0
        snap_dy = 0.0
        self._composite_snap_guides = []
        is_group = idx in self._composite_selected and len(self._composite_selected) > 1
        
        if "half_w" in drag:
            # new_x/new_y are the provisional absolute canvas center for the dragged shape
            prov_cx = new_x
            prov_cy = new_y
            hw, hh = drag["half_w"], drag["half_h"]
            prov_bbox = (prov_cx - hw, prov_cy - hh, prov_cx + hw, prov_cy + hh)
            
            # Exclude dragged shape AND all other selected shapes from snap targets
            excluded = self._composite_selected if is_group else {idx}
            other_bboxes = []
            for i, bbox in enumerate(self._composite_bboxes):
                if i in excluded:
                    other_bboxes.append((0, 0, 0, 0))
                else:
                    other_bboxes.append(bbox)
            
            # Shift dragged shape's snap anchors to provisional position
            if hasattr(self, '_composite_snap_anchors') and idx < len(self._composite_snap_anchors):
                cur_pos = self._composite_positions.get(idx, (0.0, 0.0))
                anchor_shift_x = new_x - cur_pos[0]
                anchor_shift_y = new_y - cur_pos[1]
                orig_anchors = self._composite_snap_anchors[idx]
                self._composite_snap_anchors[idx] = [(ax + anchor_shift_x, ay + anchor_shift_y) for ax, ay in orig_anchors]
            
            snap_dx, snap_dy, guides = self._calc_snap(idx, prov_bbox, other_bboxes)
            
            # Restore original anchors (will be properly set on next generate_plot)
            if hasattr(self, '_composite_snap_anchors') and idx < len(self._composite_snap_anchors):
                self._composite_snap_anchors[idx] = orig_anchors
            new_x += snap_dx
            new_y += snap_dy
            self._composite_snap_guides = guides
        
        drag["dragged"] = True
        
        # Compute the actual translation delta applied (after snap)
        actual_dx = new_x - drag["orig_pos"][0]
        actual_dy = new_y - drag["orig_pos"][1]
        
        # Group drag: move all selected shapes by the same delta
        if is_group:
            raw_dx = actual_dx
            raw_dy = actual_dy
            for sel_idx in self._composite_selected:
                orig = drag["orig_positions"].get(sel_idx, (0.0, 0.0))
                self._composite_positions[sel_idx] = (orig[0] + raw_dx, orig[1] + raw_dy)
        else:
            self._composite_positions[idx] = (new_x, new_y)
        
        # Move annotations that were captured at drag-start alongside the shapes
        for lbl_i, ox, oy in drag.get("orig_labels", []):
            if lbl_i < len(self._composite_labels):
                self._composite_labels[lbl_i]["x"] = ox + actual_dx
                self._composite_labels[lbl_i]["y"] = oy + actual_dy
        for dim_entry in drag.get("orig_dims", []):
            dim_i, ox1, oy1, ox2, oy2, olx, oly = dim_entry
            if dim_i < len(self._composite_dim_lines):
                d = self._composite_dim_lines[dim_i]
                d["x1"] = ox1 + actual_dx
                d["y1"] = oy1 + actual_dy
                d["x2"] = ox2 + actual_dx
                d["y2"] = oy2 + actual_dy
                if olx is not None:
                    d["label_x"] = olx + actual_dx
                if oly is not None:
                    d["label_y"] = oly + actual_dy
        
        self.generate_plot()
        
        # Draw snap guide lines on top
        if self._composite_snap_guides:
            for guide in self._composite_snap_guides:
                x1, y1, x2, y2, _ = guide
                self.ax.plot([x1, x2], [y1, y2],
                           color=AppConstants.SNAP_LINE_COLOR,
                           linestyle=AppConstants.SNAP_LINE_STYLE,
                           linewidth=AppConstants.SNAP_LINE_WIDTH,
                           alpha=AppConstants.SNAP_LINE_ALPHA,
                           zorder=20)
            self.canvas.draw_idle()
    
    def _on_composite_release(self, event) -> None:
        """Handle mouse release — finalize marquee, drag, or select/multi-select shape."""
        # Handle label drag/click release (including dimension line endpoints and labels)
        if self._composite_label_drag is not None:
            drag = self._composite_label_drag
            was_drag = drag.get("dragged", False)
            idx = drag.get("idx")
            drag_type = drag.get("type")
            is_dim_endpoint = drag_type == "dim_endpoint"
            is_dim_label = drag_type == "dim_label"
            is_dim_body = drag_type == "dim_body"
            self._composite_label_drag = None
            self.root.config(cursor="arrow")
            
            if was_drag:
                self.generate_plot()
                if not self.history_manager.is_restoring:
                    self._capture_current_state()
            else:
                # Click without drag
                if is_dim_endpoint or is_dim_label or is_dim_body:
                    # Already selected the dimension line in press handler
                    self._update_composite_selection_ui()
                    self.generate_plot()
                else:
                    # Click without drag — select/deselect this label
                    if self._composite_selected_label == idx:
                        self._composite_selected_label = None
                    else:
                        self._composite_selected_label = idx
                    # Deselect shapes when selecting a label
                    self._composite_selected.clear()
                    self._update_composite_selection_ui()
                    self.generate_plot()
            return
        
        # Handle marquee selection finalization
        if self._composite_marquee is not None:
            m = self._composite_marquee
            self._composite_marquee = None
            
            # Remove the rectangle artist
            if m["rect_artist"] is not None:
                try:
                    m["rect_artist"].remove()
                except (ValueError, AttributeError):
                    pass
            
            # Get final mouse position
            mx, my = None, None
            if event.inaxes == self.ax and event.xdata is not None:
                mx, my = event.xdata, event.ydata
            
            if mx is not None:
                x0, y0 = m["start_x"], m["start_y"]
                # Marquee rectangle bounds
                sel_x_min = min(x0, mx)
                sel_x_max = max(x0, mx)
                sel_y_min = min(y0, my)
                sel_y_max = max(y0, my)
                
                # Only treat as marquee if dragged a meaningful distance
                if abs(mx - x0) > 0.3 or abs(my - y0) > 0.3:
                    # Check for modifier key for additive selection
                    additive = False
                    if hasattr(event, 'guiEvent') and event.guiEvent is not None:
                        state = event.guiEvent.state
                        additive = bool(state & 0x4) or bool(state & 0x10)
                    
                    if not additive:
                        self._composite_selected.clear()
                    
                    # Select all shapes whose bbox intersects the marquee
                    if hasattr(self, '_composite_bboxes'):
                        for idx, bbox in enumerate(self._composite_bboxes):
                            if bbox == (0, 0, 0, 0):
                                continue
                            # Check intersection: two rects intersect if they overlap on both axes
                            if (bbox[0] <= sel_x_max and bbox[2] >= sel_x_min and
                                    bbox[1] <= sel_y_max and bbox[3] >= sel_y_min):
                                self._composite_selected.add(idx)
                    
                    self._update_composite_selection_ui()
                    self.generate_plot()
                else:
                    # Tiny drag = click on empty space — deselect
                    if self._composite_selected:
                        self._composite_selected.clear()
                        self._update_composite_selection_ui()
                        self.generate_plot()
            else:
                # Released outside axes
                if self._composite_selected:
                    self._composite_selected.clear()
                    self._update_composite_selection_ui()
                    self.generate_plot()
            
            self.canvas.draw_idle()
            return
        
        if self._composite_drag_state is None:
            return
        
        drag = self._composite_drag_state
        was_drag = drag.get("dragged", False)
        multi = drag.get("multi", False)
        idx = drag["idx"]
        
        self._composite_drag_state = None
        self._composite_snap_guides = []
        self.root.config(cursor="arrow")
        
        if was_drag:
            # Finalize drag
            self.generate_plot()
            if not self.history_manager.is_restoring:
                self._capture_current_state()
        else:
            # Click without drag — use the pre-press selection state to decide.
            # Press already updated _composite_selected; here we apply toggle logic
            # based on whether the shape was selected BEFORE the press.
            was_selected_before = drag.get("was_selected_before", False)
            if multi:
                # Ctrl+Click: already toggled in press handler — nothing more to do.
                # The press set the correct final state for Ctrl+Click.
                pass
            else:
                # Plain click:
                # If shape was NOT selected before → keep it selected (press already did this).
                # If shape WAS already the sole selection → deselect (toggle off).
                # If shape was part of a group → collapse to solo selection of this shape.
                if was_selected_before and self._composite_selected == {idx}:
                    # Was already the sole selection: second click deselects
                    self._composite_selected.clear()
                else:
                    # Newly selected or collapsing a group to this shape
                    self._composite_selected = {idx}
            self._update_composite_selection_ui()
            self.generate_plot()

    def _translate_patch(self, patch, dx: float, dy: float) -> None:
        """Translate a matplotlib patch by (dx, dy) in data coordinates."""
        if isinstance(patch, (patches.Polygon, patches.FancyArrow)):
            # Polygon, Rectangle, FancyArrow all support get_xy/set_xy
            xy = patch.get_xy()
            patch.set_xy(xy + np.array([dx, dy]))
        elif isinstance(patch, (patches.Circle, patches.Ellipse, patches.Arc, patches.Wedge)):
            # All have a center property
            cx, cy = patch.center
            patch.set_center((cx + dx, cy + dy))
        elif isinstance(patch, patches.Rectangle):
            x, y = patch.get_xy()
            patch.set_xy((x + dx, y + dy))
        else:
            # Fallback: try get_xy first, then center
            if hasattr(patch, 'get_xy'):
                xy = patch.get_xy()
                if hasattr(xy, '__add__'):
                    patch.set_xy(xy + np.array([dx, dy]))
                else:
                    patch.set_xy((xy[0] + dx, xy[1] + dy))
            elif hasattr(patch, 'center'):
                cx, cy = patch.center
                patch.set_center((cx + dx, cy + dy))

    def _calc_snap(self, drag_idx: int, drag_bbox: tuple,
                   all_bboxes: list[tuple]) -> tuple[float, float, list]:
        """Calculate snap offsets and guide lines for a dragged shape.
        
        Compares the dragged shape's edges, centers, and semantic anchor
        points against all other shapes.  Anchor-to-anchor snaps use a
        tighter threshold and win ties with bbox snaps so that 3D
        ellipse centres align cleanly.
        Returns (snap_dx, snap_dy, guide_lines) where guide_lines are
        [(x1,y1,x2,y2,axis), ...] for visual feedback.
        """
        threshold = AppConstants.SNAP_THRESHOLD
        # Anchor snaps use a slightly tighter threshold but get a priority
        # bonus: when an anchor snap and a bbox snap are within 0.3 of each
        # other, the anchor snap wins.
        anchor_threshold = threshold * 0.9
        snap_dx = 0.0
        snap_dy = 0.0
        guides = []
        
        d_left, d_bottom, d_right, d_top = drag_bbox
        d_cx = (d_left + d_right) / 2
        d_cy = (d_bottom + d_top) / 2
        
        best_x_dist = threshold
        best_y_dist = threshold
        best_x_is_anchor = False
        best_y_is_anchor = False
        
        # Collect semantic anchors if available
        has_anchors = hasattr(self, '_composite_snap_anchors') and self._composite_snap_anchors
        drag_anchors = []
        if has_anchors and drag_idx < len(self._composite_snap_anchors):
            # Shift anchors by the same provisional position used for drag_bbox
            drag_anchors = self._composite_snap_anchors[drag_idx]
        
        for i, bbox in enumerate(all_bboxes):
            if i == drag_idx or bbox == (0, 0, 0, 0):
                continue
            
            o_left, o_bottom, o_right, o_top = bbox
            o_cx = (o_left + o_right) / 2
            o_cy = (o_bottom + o_top) / 2
            
            # --- Bbox edge snaps ---
            x_snaps = [
                (d_left, o_left), (d_left, o_right),
                (d_right, o_left), (d_right, o_right),
                (d_cx, o_cx),
                (d_left, o_cx), (d_right, o_cx),
                (d_cx, o_left), (d_cx, o_right),
            ]
            
            for d_edge, o_edge in x_snaps:
                dist = abs(d_edge - o_edge)
                # Anchor snap wins ties: only replace anchor with bbox if bbox is strictly closer by margin
                if dist < best_x_dist and not (best_x_is_anchor and dist > best_x_dist - 0.3):
                    best_x_dist = dist
                    best_x_is_anchor = False
                    snap_dx = o_edge - d_edge
                    all_y = [d_bottom, d_top, o_bottom, o_top]
                    guides = [g for g in guides if g[4] != "x"]
                    guides.append((o_edge, min(all_y) - 0.5, o_edge, max(all_y) + 0.5, "x"))
            
            y_snaps = [
                (d_bottom, o_bottom), (d_bottom, o_top),
                (d_top, o_bottom), (d_top, o_top),
                (d_cy, o_cy),
                (d_bottom, o_cy), (d_top, o_cy),
                (d_cy, o_bottom), (d_cy, o_top),
            ]
            
            for d_edge, o_edge in y_snaps:
                dist = abs(d_edge - o_edge)
                if dist < best_y_dist and not (best_y_is_anchor and dist > best_y_dist - 0.3):
                    best_y_dist = dist
                    best_y_is_anchor = False
                    snap_dy = o_edge - d_edge
                    all_x = [d_left, d_right, o_left, o_right]
                    guides = [g for g in guides if g[4] != "y"]
                    guides.append((min(all_x) - 0.5, o_edge, max(all_x) + 0.5, o_edge, "y"))
            
            # --- Semantic anchor snaps (higher priority) ---
            if has_anchors and i < len(self._composite_snap_anchors):
                other_anchors = self._composite_snap_anchors[i]
                if drag_anchors and other_anchors:
                    for dax, day in drag_anchors:
                        for oax, oay in other_anchors:
                            # X anchor snap
                            dist_x = abs(dax - oax)
                            if dist_x < anchor_threshold:
                                if dist_x < best_x_dist or (best_x_is_anchor and dist_x <= best_x_dist):
                                    best_x_dist = dist_x
                                    best_x_is_anchor = True
                                    snap_dx = oax - dax
                                    all_y = [d_bottom, d_top, o_bottom, o_top]
                                    guides = [g for g in guides if g[4] != "x"]
                                    guides.append((oax, min(all_y) - 0.5, oax, max(all_y) + 0.5, "x"))
                            # Y anchor snap
                            dist_y = abs(day - oay)
                            if dist_y < anchor_threshold:
                                if dist_y < best_y_dist or (best_y_is_anchor and dist_y <= best_y_dist):
                                    best_y_dist = dist_y
                                    best_y_is_anchor = True
                                    snap_dy = oay - day
                                    all_x = [d_left, d_right, o_left, o_right]
                                    guides = [g for g in guides if g[4] != "y"]
                                    guides.append((min(all_x) - 0.5, oay, max(all_x) + 0.5, oay, "y"))
        
        return snap_dx, snap_dy, guides

    def _on_composite_delete(self, event=None) -> None:
        """Delete selected composite shapes or label."""
        if not self._is_composite_shape():
            return
        
        # Prevent deletion while typing in the label entry
        if isinstance(self.root.focus_get(), tk.Entry):
            return

        if self._composite_selected_label is not None or self._composite_selected_dim is not None:
            self._remove_selected_annotation()
            return
        
        if not self._composite_selected:
            return
        if not hasattr(self, 'composite_transfer') or self.composite_transfer is None:
            return
        
        # Remove shapes from highest index to lowest (to avoid shifting issues)
        for idx in sorted(self._composite_selected, reverse=True):
            if idx < self.composite_transfer.dest_listbox.size():
                self.composite_transfer.dest_listbox.delete(idx)
                
                # Shift positions and transforms down
                new_positions = {}
                new_transforms = {}
                for k, v in self._composite_positions.items():
                    if k < idx:
                        new_positions[k] = v
                    elif k > idx:
                        new_positions[k - 1] = v
                for k, v in self._composite_transforms.items():
                    if k < idx:
                        new_transforms[k] = v
                    elif k > idx:
                        new_transforms[k - 1] = v
                self._composite_positions = new_positions
                self._composite_transforms = new_transforms
        
        self._composite_selected.clear()
        
        # Update tracking
        new_shapes = self.composite_transfer.get_selected_shapes()
        self._composite_prev_count = len(new_shapes)
        self._composite_prev_shapes = list(new_shapes)
        
        self._update_composite_selection_ui()
        self.generate_plot()
        if not self.history_manager.is_restoring:
            self._capture_current_state()

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
    
    def _add_composite_label(self) -> None:
        """Add a custom label to the composite canvas."""
        if not hasattr(self, '_composite_label_entry'):
            return
        text = self._composite_label_entry.get().strip()
        if not text:
            return
        
        # Place label at the center of the current view
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        cx = (xlim[0] + xlim[1]) / 2
        cy = (ylim[0] + ylim[1]) / 2
        
        self._composite_labels.append({"text": text, "x": cx, "y": cy})
        self._composite_label_entry.delete(0, tk.END)
        self._composite_selected_label = len(self._composite_labels) - 1
        
        self.generate_plot()
        if not self.history_manager.is_restoring:
            self._capture_current_state()
    
    def _remove_composite_label(self) -> None:
        """Remove the selected composite label."""
        if self._composite_selected_label is None:
            return
        if self._composite_selected_label >= len(self._composite_labels):
            return
        
        self._composite_labels.pop(self._composite_selected_label)
        self._composite_selected_label = None
        
        self.generate_plot()
        if not self.history_manager.is_restoring:
            self._capture_current_state()

    def _confirm_composite_label(self) -> None:
        """Add or update a composite label depending on edit mode."""
        if self._composite_edit_mode is not None:
            self._apply_composite_edit()
        else:
            self._add_composite_label()

    def _enter_composite_edit(self, edit_type: str, idx: int) -> None:
        """Enter edit mode for a composite label or dim line label."""
        if edit_type == "label" and idx < len(self._composite_labels):
            existing = self._composite_labels[idx]["text"]
        elif edit_type == "dim" and idx < len(self._composite_dim_lines):
            existing = self._composite_dim_lines[idx]["text"]
        else:
            return
        
        self._composite_edit_mode = {"type": edit_type, "idx": idx}
        
        if hasattr(self, '_composite_label_entry'):
            self._composite_label_entry.delete(0, tk.END)
            self._composite_label_entry.insert(0, existing)
            self._composite_label_entry.focus_set()
            self._composite_label_entry.select_range(0, tk.END)
        
        if hasattr(self, '_composite_text_btn'):
            self._composite_text_btn.config(text="Update")
        if hasattr(self, '_composite_cancel_btn'):
            self._composite_cancel_btn.pack(side=tk.LEFT, padx=1)

    def _apply_composite_edit(self) -> None:
        """Apply edited text to composite label or dim line."""
        if self._composite_edit_mode is None:
            return
        if not hasattr(self, '_composite_label_entry'):
            return
        
        new_text = self._composite_label_entry.get().strip()
        if not new_text:
            self._cancel_composite_edit()
            return
        
        edit_type = self._composite_edit_mode["type"]
        idx = self._composite_edit_mode["idx"]
        
        if edit_type == "label" and idx < len(self._composite_labels):
            self._composite_labels[idx]["text"] = new_text
        elif edit_type == "dim" and idx < len(self._composite_dim_lines):
            self._composite_dim_lines[idx]["text"] = new_text
        
        self._cancel_composite_edit()
        self.generate_plot()
        if not self.history_manager.is_restoring:
            self._capture_current_state()

    def _cancel_composite_edit(self) -> None:
        """Exit composite edit mode, restore + Text button."""
        self._composite_edit_mode = None
        if hasattr(self, '_composite_label_entry'):
            self._composite_label_entry.delete(0, tk.END)
        if hasattr(self, '_composite_text_btn'):
            self._composite_text_btn.config(text="+ Text")
        if hasattr(self, '_composite_cancel_btn'):
            self._composite_cancel_btn.pack_forget()

    def _start_dim_line_mode(self, mode: str) -> None:
        """Enter dimension line placement mode."""
        text = self._composite_label_entry.get().strip()
        if not text:
            # Default labels for presets
            defaults = {"height": "h", "width": "w", "radius": "r", "free": "d"}
            text = defaults.get(mode, "d")
        
        self._composite_dim_mode = mode
        self._composite_dim_first_point = None
        self._dim_status_label.config(text=f"Click first point for {mode} dimension line...")
        self.root.config(cursor="crosshair")
    
    def _cancel_dim_line_mode(self) -> None:
        """Exit dimension line placement mode."""
        self._composite_dim_mode = None
        self._composite_dim_first_point = None
        if hasattr(self, '_dim_status_label'):
            self._dim_status_label.config(text="")
        if hasattr(self, '_dim_preview_line') and self._dim_preview_line:
            try:
                self._dim_preview_line.remove()
            except (ValueError, AttributeError):
                pass
            self._dim_preview_line = None
        self.root.config(cursor="arrow")
    
    def _handle_dim_line_click(self, mx: float, my: float) -> bool:
        """Handle a click during dimension line placement. Returns True if handled."""
        if self._composite_dim_mode is None:
            return False
        
        mode = self._composite_dim_mode
        
        if self._composite_dim_first_point is None:
            # First click — record the starting point
            self._composite_dim_first_point = (mx, my)
            self._dim_status_label.config(text=f"Click second point for {mode} dimension line...")
            return True
        else:
            # Second click — create the dimension line
            x1, y1 = self._composite_dim_first_point
            x2, y2 = mx, my
            
            # Apply constraints for presets
            if mode == "height":
                x2 = x1  # vertical line
            elif mode == "width":
                y2 = y1  # horizontal line
            elif mode == "radius":
                pass  # free direction from center
            
            text = self._composite_label_entry.get().strip()
            if not text:
                defaults = {"height": "h", "width": "w", "radius": "r", "free": "d"}
                text = defaults.get(mode, "d")
            
            # Calculate initial label position at midpoint
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            self._composite_dim_lines.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "text": text,
                "constraint": mode if mode != "free" else None,
                "label_x": mid_x,
                "label_y": mid_y
            })
            
            self._composite_label_entry.delete(0, tk.END)
            self._composite_selected_dim = len(self._composite_dim_lines) - 1
            self._cancel_dim_line_mode()
            
            self.generate_plot()
            if not self.history_manager.is_restoring:
                self._capture_current_state()
            return True
    
    def _remove_selected_annotation(self) -> None:
        """Remove the selected label or dimension line."""
        if self._composite_selected_label is not None:
            self._remove_composite_label()
        elif self._composite_selected_dim is not None:
            if self._composite_selected_dim < len(self._composite_dim_lines):
                self._composite_dim_lines.pop(self._composite_selected_dim)
            self._composite_selected_dim = None
            self.generate_plot()
            if not self.history_manager.is_restoring:
                self._capture_current_state()

    def _update_composite_selection_ui(self) -> None:
        """Update the UI to reflect which composite shapes are selected."""
        if hasattr(self, 'composite_transfer') and self.composite_transfer is not None:
            dest = self.composite_transfer.dest_listbox
            dest.selection_clear(0, tk.END)
            for idx in self._composite_selected:
                if idx < dest.size():
                    dest.selection_set(idx)
            # Scroll to show the most recently selected
            if self._composite_selected:
                last = max(self._composite_selected)
                if last < dest.size():
                    dest.see(last)
    
    def _get_group_center(self) -> tuple[float, float]:
        """Calculate the center of all selected shapes' bounding boxes."""
        if not self._composite_selected or not hasattr(self, '_composite_bboxes'):
            return (0.0, 0.0)
        xs, ys = [], []
        for idx in self._composite_selected:
            if idx < len(self._composite_bboxes):
                bbox = self._composite_bboxes[idx]
                if bbox != (0, 0, 0, 0):
                    xs.append((bbox[0] + bbox[2]) / 2)
                    ys.append((bbox[1] + bbox[3]) / 2)
        if not xs:
            return (0.0, 0.0)
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def _get_annotations_in_region(self, x_min: float, y_min: float,
                                    x_max: float, y_max: float,
                                    include_all: bool = False) -> tuple[list[int], list[int]]:
        """Return (label_indices, dim_line_indices) whose positions fall within the given region.
        If include_all is True, return all annotations regardless of position."""
        label_idxs = []
        for i, lbl in enumerate(self._composite_labels):
            if include_all or (x_min <= lbl["x"] <= x_max and y_min <= lbl["y"] <= y_max):
                label_idxs.append(i)
        
        dim_idxs = []
        for i, dim in enumerate(self._composite_dim_lines):
            if include_all:
                dim_idxs.append(i)
            else:
                # Include if both endpoints are within the region
                p1_in = x_min <= dim["x1"] <= x_max and y_min <= dim["y1"] <= y_max
                p2_in = x_min <= dim["x2"] <= x_max and y_min <= dim["y2"] <= y_max
                if p1_in and p2_in:
                    dim_idxs.append(i)
        
        return label_idxs, dim_idxs

    def _all_shapes_selected(self) -> bool:
        """Check if every shape in the composite is selected."""
        if not hasattr(self, 'composite_transfer') or self.composite_transfer is None:
            return False
        total = self.composite_transfer.dest_listbox.size()
        return total > 0 and len(self._composite_selected) >= total

    def _get_group_bbox(self) -> tuple[float, float, float, float]:
        """Get the bounding box encompassing all selected shapes with padding."""
        if not self._composite_selected or not hasattr(self, '_composite_bboxes'):
            return (0, 0, 0, 0)
        x_mins, y_mins, x_maxs, y_maxs = [], [], [], []
        for idx in self._composite_selected:
            if idx < len(self._composite_bboxes):
                bbox = self._composite_bboxes[idx]
                if bbox != (0, 0, 0, 0):
                    x_mins.append(bbox[0])
                    y_mins.append(bbox[1])
                    x_maxs.append(bbox[2])
                    y_maxs.append(bbox[3])
        if not x_mins:
            return (0, 0, 0, 0)
        pad = 0.5
        return (min(x_mins) - pad, min(y_mins) - pad, max(x_maxs) + pad, max(y_maxs) + pad)

    def _rotate_annotations(self, label_idxs: list[int], dim_idxs: list[int],
                             cx: float, cy: float, direction: int) -> None:
        """Rotate annotations around (cx, cy) by 90 degrees. direction: 1=CW, -1=CCW."""
        def rot(x, y):
            rx, ry = x - cx, y - cy
            if direction == 1:
                return cx + ry, cy - rx
            else:
                return cx - ry, cy + rx
        
        for i in label_idxs:
            lbl = self._composite_labels[i]
            lbl["x"], lbl["y"] = rot(lbl["x"], lbl["y"])
        
        for i in dim_idxs:
            dim = self._composite_dim_lines[i]
            dim["x1"], dim["y1"] = rot(dim["x1"], dim["y1"])
            dim["x2"], dim["y2"] = rot(dim["x2"], dim["y2"])
            if "label_x" in dim and "label_y" in dim:
                dim["label_x"], dim["label_y"] = rot(dim["label_x"], dim["label_y"])
            # Update constraint direction after rotation
            c = dim.get("constraint")
            if c == "height":
                dim["constraint"] = "width"
            elif c == "width":
                dim["constraint"] = "height"

    def _flip_annotations_h(self, label_idxs: list[int], dim_idxs: list[int],
                              cx: float) -> None:
        """Mirror annotations horizontally around x=cx."""
        for i in label_idxs:
            lbl = self._composite_labels[i]
            lbl["x"] = 2 * cx - lbl["x"]
        
        for i in dim_idxs:
            dim = self._composite_dim_lines[i]
            dim["x1"] = 2 * cx - dim["x1"]
            dim["x2"] = 2 * cx - dim["x2"]
            if "label_x" in dim:
                dim["label_x"] = 2 * cx - dim["label_x"]

    def _flip_annotations_v(self, label_idxs: list[int], dim_idxs: list[int],
                              cy: float) -> None:
        """Mirror annotations vertically around y=cy."""
        for i in label_idxs:
            lbl = self._composite_labels[i]
            lbl["y"] = 2 * cy - lbl["y"]
        
        for i in dim_idxs:
            dim = self._composite_dim_lines[i]
            dim["y1"] = 2 * cy - dim["y1"]
            dim["y2"] = 2 * cy - dim["y2"]
            if "label_y" in dim:
                dim["label_y"] = 2 * cy - dim["label_y"]

    def _composite_flip(self, axis: str) -> None:
        """Shared implementation for horizontal ('h') and vertical ('v') composite flip.

        axis='h': mirrors left↔right around the group's X centre.
        axis='v': mirrors up↔down around the group's Y centre.
        """
        if not self._composite_selected:
            return
        if not self.history_manager.is_restoring:
            self._capture_current_state()

        # Capture pre-flip state for annotation transforms
        pre_flip_bbox = self._get_group_bbox() if len(self._composite_selected) > 1 else (0, 0, 0, 0)
        pre_flip_gcx, pre_flip_gcy = self._get_group_center() if len(self._composite_selected) > 1 else (0, 0)
        pivot = pre_flip_gcx if axis == 'h' else pre_flip_gcy

        selected = self.composite_transfer.get_selected_shapes() if hasattr(self, 'composite_transfer') and self.composite_transfer else []

        if len(self._composite_selected) > 1:
            for idx in self._composite_selected:
                pos = self._composite_positions.get(idx, (0.0, 0.0))
                if idx < len(self._composite_bboxes):
                    bbox = self._composite_bboxes[idx]
                    if bbox != (0, 0, 0, 0):
                        if axis == 'h':
                            shape_c = (bbox[0] + bbox[2]) / 2
                            new_c = 2 * pivot - shape_c
                            self._composite_positions[idx] = (pos[0] + (new_c - shape_c), pos[1])
                        else:
                            shape_c = (bbox[1] + bbox[3]) / 2
                            new_c = 2 * pivot - shape_c
                            self._composite_positions[idx] = (pos[0], pos[1] + (new_c - shape_c))

                t = self._composite_transforms.setdefault(idx, {"flip_h": False, "flip_v": False, "base_side": 0})
                num_sides = 4
                shape_name_for_flip = ""
                if idx < len(selected):
                    shape_name_for_flip = selected[idx]
                    config = ShapeConfigProvider.get(shape_name_for_flip)
                    num_sides = config.num_sides if config.num_sides > 0 else 4

                effective_side = t["base_side"]
                if num_sides >= 4 and config.uses_base_side_flip:
                    # Normalise accumulated flag flips into base_side first
                    if t["flip_v"]:
                        effective_side = (effective_side + 2) % num_sides
                        t["flip_v"] = False
                    if t["flip_h"]:
                        if effective_side in (1, 3):
                            effective_side = 4 - effective_side
                        t["flip_h"] = False

                    if axis == 'h':
                        # Horizontal mirror: left↔right sides (1↔3)
                        if effective_side in (1, 3):
                            effective_side = 4 - effective_side
                    else:
                        # Vertical mirror: top↔bottom sides (0↔2)
                        if effective_side in (0, 2):
                            effective_side = 2 - effective_side
                    t["base_side"] = effective_side
                else:
                    # 3-sided shapes and prisms: toggle the flag directly
                    if axis == 'h':
                        t["flip_h"] = not t["flip_h"]
                    else:
                        t["flip_v"] = not t["flip_v"]
        else:
            for idx in self._composite_selected:
                t = self._composite_transforms.setdefault(idx, {"flip_h": False, "flip_v": False, "base_side": 0})
                if axis == 'h':
                    t["flip_h"] = not t["flip_h"]
                else:
                    t["flip_v"] = not t["flip_v"]
                # Flip annotations within this single shape's bbox
                if idx < len(self._composite_bboxes):
                    bbox = self._composite_bboxes[idx]
                    if bbox != (0, 0, 0, 0):
                        if axis == 'h':
                            shape_c = (bbox[0] + bbox[2]) / 2
                            lbl_idxs, dim_idxs = self._get_annotations_in_region(*bbox)
                            if lbl_idxs or dim_idxs:
                                self._flip_annotations_h(lbl_idxs, dim_idxs, shape_c)
                        else:
                            shape_c = (bbox[1] + bbox[3]) / 2
                            lbl_idxs, dim_idxs = self._get_annotations_in_region(*bbox)
                            if lbl_idxs or dim_idxs:
                                self._flip_annotations_v(lbl_idxs, dim_idxs, shape_c)

        # Flip annotations within the pre-flip group area (multi-shape case)
        if len(self._composite_selected) > 1 and pre_flip_bbox != (0, 0, 0, 0):
            all_sel = self._all_shapes_selected()
            lbl_idxs, dim_idxs = self._get_annotations_in_region(*pre_flip_bbox, include_all=all_sel)
            if lbl_idxs or dim_idxs:
                if axis == 'h':
                    self._flip_annotations_h(lbl_idxs, dim_idxs, pre_flip_gcx)
                else:
                    self._flip_annotations_v(lbl_idxs, dim_idxs, pre_flip_gcy)

        self.generate_plot()
        if not self.history_manager.is_restoring:
            self._capture_current_state(force=True)

    def _composite_flip_h(self) -> None:
        """Flip selected shapes horizontally (left↔right) around their collective centre."""
        self._composite_flip('h')

    def _composite_flip_v(self) -> None:
        """Flip selected shapes vertically (up↔down) around their collective centre."""
        self._composite_flip('v')
    
    def _composite_rotate(self, direction: int) -> None:
        """Rotate selected shapes as a group around their collective center.
        
        For group rotation, this is a two-pass process:
        1. Record pre-rotation shape centers, compute target positions by rotating
           around group center, then apply individual shape rotations.
        2. Redraw to get post-rotation bounding boxes, then correct positions so
           each shape's new center lands at its target position.
        """
        if not self._composite_selected:
            return
        if not hasattr(self, 'composite_transfer') or self.composite_transfer is None:
            return
        selected = self.composite_transfer.get_selected_shapes()
        
        if not self.history_manager.is_restoring:
            self._capture_current_state()
        
        # Capture pre-rotation center and bbox for annotation transforms
        pre_rot_center = None
        pre_rot_bbox = (0, 0, 0, 0)
        if len(self._composite_selected) > 1:
            pre_rot_bbox = self._get_group_bbox()
            centers_pre = []
            for idx in self._composite_selected:
                if idx < len(self._composite_bboxes):
                    bbox = self._composite_bboxes[idx]
                    if bbox != (0, 0, 0, 0):
                        centers_pre.append(((bbox[0] + bbox[2]) / 2,
                                            (bbox[1] + bbox[3]) / 2))
            if centers_pre:
                pre_rot_center = (sum(o[0] for o in centers_pre) / len(centers_pre),
                                  sum(o[1] for o in centers_pre) / len(centers_pre))
        
        if len(self._composite_selected) > 1:
            # Always use bbox centers for consistent group rotation.
            # Snap-anchor "last = canonical origin" is unreliable for 3D shapes
            # whose visual center differs from their canonical (0,0) origin.
            shape_centers = {}
            for idx in self._composite_selected:
                if idx < len(self._composite_bboxes):
                    bbox = self._composite_bboxes[idx]
                    if bbox != (0, 0, 0, 0):
                        shape_centers[idx] = ((bbox[0] + bbox[2]) / 2,
                                              (bbox[1] + bbox[3]) / 2)

            if not shape_centers:
                return

            # Group center = centroid of all selected shape centers
            gcx = sum(o[0] for o in shape_centers.values()) / len(shape_centers)
            gcy = sum(o[1] for o in shape_centers.values()) / len(shape_centers)

            # Pass 1: Compute where each shape's bbox center should land after
            # rotating 90° around the group center, then advance each shape's
            # individual orientation by one step.
            target_centers = {}
            for idx in self._composite_selected:
                if idx in shape_centers:
                    ox, oy = shape_centers[idx]
                    rx = ox - gcx
                    ry = oy - gcy
                    if direction == 1:  # CW
                        new_rx, new_ry = ry, -rx
                    else:  # CCW
                        new_rx, new_ry = -ry, rx
                    target_centers[idx] = (gcx + new_rx, gcy + new_ry)

                # Apply individual shape rotation with flip normalization
                if idx < len(selected):
                    shape_name = selected[idx]
                    config = ShapeConfigProvider.get(shape_name)
                    num_sides = config.num_sides if config.num_sides > 0 else 4
                    t = self._composite_transforms.setdefault(idx, {"flip_h": False, "flip_v": False, "base_side": 0})

                    # Advance orientation by one step, preserving flip state exactly
                    # as single-shape rotation does. Normalizing flips into base_side
                    # was discarding flip state and causing visible reversion.
                    t["base_side"] = (t["base_side"] + direction) % num_sides

            # Pass 2: Redraw with rotated shapes to get new bounding boxes.
            # Do NOT reset _composite_view_limits here so grid positions stay frozen.
            self.generate_plot()

            # Pass 3: Correct positions so each shape's new bbox center matches target
            for idx, (target_cx, target_cy) in target_centers.items():
                actual_cx, actual_cy = None, None
                if idx < len(self._composite_bboxes):
                    new_bbox = self._composite_bboxes[idx]
                    if new_bbox != (0, 0, 0, 0):
                        actual_cx = (new_bbox[0] + new_bbox[2]) / 2
                        actual_cy = (new_bbox[1] + new_bbox[3]) / 2

                if actual_cx is not None:
                    correction_dx = target_cx - actual_cx
                    correction_dy = target_cy - actual_cy
                    pos = self._composite_positions.get(idx, (0.0, 0.0))
                    self._composite_positions[idx] = (pos[0] + correction_dx, pos[1] + correction_dy)
        else:
            # Single shape: rotate it and move any annotations within its bbox
            for idx in self._composite_selected:
                if idx >= len(selected):
                    continue
                shape_name = selected[idx]
                config = ShapeConfigProvider.get(shape_name)
                num_sides = config.num_sides if config.num_sides > 0 else 4
                t = self._composite_transforms.setdefault(idx, {"flip_h": False, "flip_v": False, "base_side": 0})
                t["base_side"] = (t["base_side"] + direction) % num_sides
                
                # Rotate annotations that fall within this shape's pre-rotation bbox
                if idx < len(self._composite_bboxes):
                    bbox = self._composite_bboxes[idx]
                    if bbox != (0, 0, 0, 0):
                        shape_cx = (bbox[0] + bbox[2]) / 2
                        shape_cy = (bbox[1] + bbox[3]) / 2
                        lbl_idxs, dim_idxs = self._get_annotations_in_region(*bbox)
                        if lbl_idxs or dim_idxs:
                            self._rotate_annotations(lbl_idxs, dim_idxs, shape_cx, shape_cy, direction)
        
        # Rotate annotations using the pre-rotation center and region
        if len(self._composite_selected) > 1 and pre_rot_center is not None:
            rot_cx, rot_cy = pre_rot_center
            all_sel = self._all_shapes_selected()
            lbl_idxs, dim_idxs = self._get_annotations_in_region(*pre_rot_bbox, include_all=all_sel)
            if lbl_idxs or dim_idxs:
                self._rotate_annotations(lbl_idxs, dim_idxs, rot_cx, rot_cy, direction)
        
        # Recalculate view after transform
        self._composite_view_limits = None
        
        # Final redraw with corrected positions
        self.generate_plot()
        if not self.history_manager.is_restoring:
            self._capture_current_state(force=True)

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

                # B30 fix: for Arc-based radially-symmetric shapes (Hemisphere,
                # Cylinder, Cone, Sphere) the two-pass rotate-then-flip in
                # transform_artist_lists produces wrong Arc theta/angle values
                # because the flip formula assumes artist.angle == 0.  For these
                # shapes flip_v is always equivalent to base_side+2 and flip_h
                # is equivalent to swapping side orientations 1<->3, so we can
                # absorb all flip state into base_side here and pass only a pure
                # rotation angle to transform_artist_lists.
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
        
        if is_dragging and hasattr(self, '_composite_view_limits') and self._composite_view_limits is not None:
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
            font_size = self.font_size_var.get() if hasattr(self, 'font_size_var') else 12
            
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
        font_size = self.font_size_var.get() if hasattr(self, 'font_size_var') else 12
        
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
                              {"key": "length", "label": "Width", "default_text": "w"},
                              {"key": "width", "label": "Length", "default_text": "l"}],
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
    }

    # ------------------------------------------------------------------
    # Dimension-line endpoint calculators — one private method per preset key.
    # Each receives the pre-extracted common variables so there is no repeated
    # book-keeping.  _calc_dim_line_endpoints is the public dispatcher.
    # Adding a new preset key: implement _calc_dim_<key> and add it to
    # _DIM_DISPATCH (built lazily on first call to avoid forward-reference issues).
    # ------------------------------------------------------------------

    def _calc_dim_line_endpoints(self, preset_key: str) -> Optional[dict]:
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

        # Build dispatch table once per instance (avoids forward-ref problems)
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

        # Rect prism: front-face bottom edge
        if "prism_f_bl" in hints:
            res = self._dim_perp_offset(
                hints["prism_f_bl"], hints["prism_f_br"],
                offset, label_gap, shape_cx, shape_cy)
            if res:
                return res
            f_bl = hints["prism_f_bl"]; f_br = hints["prism_f_br"]
            y = min(f_bl[1], f_br[1]) - offset
            return {
                "x1": f_bl[0], "y1": y, "x2": f_br[0], "y2": y,
                "label_x": (f_bl[0] + f_br[0]) / 2, "label_y": y - label_gap,
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
            f_br = hints["prism_f_br"]; b_br = hints["prism_b_br"]
            f_bl = hints["prism_f_bl"]; b_bl = hints["prism_b_bl"]
            flip_h = hints.get("prism_flip_h", False)
            all_pts = [hints["prism_f_bl"], hints["prism_f_br"],
                       hints["prism_f_tl"], hints["prism_f_tr"],
                       hints["prism_b_br"], hints["prism_b_bl"]]
            shape_cy_prism = sum(p[1] for p in all_pts) / len(all_pts)
            ep1_raw, ep2_raw = (f_bl, b_bl) if flip_h else (f_br, b_br)
            dx = ep2_raw[0] - ep1_raw[0]; dy = ep2_raw[1] - ep1_raw[1]
            seg_len = math.sqrt(dx*dx + dy*dy) or 1
            nx, ny = -dy / seg_len, dx / seg_len
            if (ep1_raw[1] + ep2_raw[1]) / 2 + ny > shape_cy_prism:
                nx, ny = -nx, -ny
            ep1 = (ep1_raw[0] + nx * offset, ep1_raw[1] + ny * offset)
            ep2 = (ep2_raw[0] + nx * offset, ep2_raw[1] + ny * offset)
            mx = (ep1[0] + ep2[0]) / 2; my = (ep1[1] + ep2[1]) / 2
            return {
                "x1": ep1[0], "y1": ep1[1], "x2": ep2[0], "y2": ep2[1],
                "label_x": mx + nx * label_gap, "label_y": my + ny * label_gap,
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
        font_size = self.font_size_var.get() if hasattr(self, 'font_size_var') else 12
        
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
        try:
            # Sync entries to label manager before any state capture or drawing
            shape = self.shape_var.get()
            if not shape:
                return
            
            # B28 fix: flush pending geometry changes (e.g. control columns
            # appearing on first shape selection) so the canvas widget has its
            # true size BEFORE we read _axes_pixel_aspect. Without this, the
            # very first draw (e.g. navigating directly to Sphere) uses the
            # pre-layout canvas dimensions, stretching equal-aspect shapes.
            self.root.update_idletasks()
            
            self.plot_controller.clear()
            self.plot_controller.setup_axes()
            
            # Preserve toggle-based labels across clear
            _preserved = {}
            for lbl_key, _ in self.TOGGLE_LABEL_KEYS:
                if lbl_key in self.label_manager.label_texts:
                    _preserved[lbl_key] = (self.label_manager.label_texts[lbl_key],
                                           self.label_manager.label_visibility.get(lbl_key, True))
            
            self.label_manager.clear_label_texts()
            
            # Restore preserved toggle labels
            for key, (text, vis) in _preserved.items():
                if text:
                    self.label_manager.set_label_text(key, text, vis)
            
            # --- Composite shape rendering ---
            if self._is_composite_shape(shape) and hasattr(self, 'composite_transfer'):
                selected = self.composite_transfer.get_selected_shapes()
                if not selected:
                    # Set paper-proportioned limits for empty canvas
                    pixel_aspect = getattr(self, '_axes_pixel_aspect', 4.0/3.0)
                    page_h = 20.0
                    page_w = page_h * pixel_aspect
                    self.ax.set_xlim(0, page_w)
                    self.ax.set_ylim(0, page_h)
                    self.ax.text(0.5, 0.5, "Add shapes from the Available list\nusing the → button",
                                ha="center", va="center", fontsize=14, color="gray",
                                transform=self.ax.transAxes)
                    self.plot_controller.refresh()
                    return
                
                self._draw_composite_shapes(selected)
                self.plot_controller.refresh()
                return
            
            ctx = self.plot_controller.create_drawing_context(
                aspect_ratio=self.scale_manager.var("aspect").get()
            )

            self.label_manager.circ_selected = getattr(self, '_builtin_selected', None) == "Circumference"
            self.label_manager.builtin_selected = getattr(self, '_builtin_selected', None)
            
            transform = self._create_transform_state()
            params = self._collect_shape_params()
            
            # Unified geometric transform for ALL shapes (2D and 3D):
            # 1. Draw at canonical orientation (base_side=0, no flip).
            # 2. Rotate all artists by the requested base_side angle.
            # 3. Flip all artists in the current (post-rotation) orientation.
            # This is the same pipeline used by composite mode, ensuring
            # rotation and flip behave identically — everything moves as a unit.
            # Flip always operates on the visible shape so "flip H" always
            # mirrors left<->right as currently displayed, regardless of rotation.
            actual_base_side = transform.base_side
            actual_flip_h = transform.flip_h
            actual_flip_v = transform.flip_v
            is_unified_rotate = shape in GeometricRotation.ALL_GEOMETRIC_SHAPES
            if is_unified_rotate:
                # Always draw canonical: no rotation, no flip
                transform = TransformState(flip_h=False, flip_v=False, base_side=0)

            error = self.plot_controller.draw_shape(shape, ctx, transform, params)
            if error:
                self.plot_controller.draw_error(error)

            # Canonical geometry center: use the axes limits that the drawer's
            # set_limits() set from shape vertex bounds.  This excludes dimension
            # line endpoints and label positions that extend the bbox asymmetrically
            # and would shift the center on asymmetric shapes (triangles, prisms).
            _xlim = self.ax.get_xlim()
            _ylim = self.ax.get_ylim()
            _canonical_center = ((_xlim[0] + _xlim[1]) / 2, (_ylim[0] + _ylim[1]) / 2)
            self._canonical_geometry_center = _canonical_center
            self._geometry_center = _canonical_center

            if is_unified_rotate and not error:
                config = ShapeConfigProvider.get(shape)
                num_sides = config.num_sides if config.num_sides > 0 else 4

                # Step 2: rotate around the geometry center
                if actual_base_side != 0:
                    angle_deg = -(360.0 / num_sides) * actual_base_side
                    if self._geometry_center:
                        cx, cy = self._geometry_center
                    else:
                        xlim = self.ax.get_xlim()
                        ylim = self.ax.get_ylim()
                        cx = (xlim[0] + xlim[1]) / 2
                        cy = (ylim[0] + ylim[1]) / 2
                    GeometricRotation.rotate_axes_artists(self.ax, angle_deg, cx, cy)
                    GeometricRotation.recalculate_limits(self.ax)
                    self._rotate_geometry_hints(math.radians(angle_deg), cx, cy)

                # Step 3: flip in current orientation (after rotation)
                if actual_flip_h or actual_flip_v:
                    # Flip operates in post-rotation visual space, so use the
                    # current axis limits center (set by recalculate_limits after
                    # rotation), NOT the canonical center.
                    xlim = self.ax.get_xlim()
                    ylim = self.ax.get_ylim()
                    cx = (xlim[0] + xlim[1]) / 2
                    cy = (ylim[0] + ylim[1]) / 2
                    GeometricRotation.flip_axes_artists(
                        self.ax, actual_flip_h, actual_flip_v, cx, cy)
                    GeometricRotation.recalculate_limits(self.ax)
                    self._flip_geometry_hints(actual_flip_h, actual_flip_v, cx, cy)
                    # Store flip center for _flip_with_annotations to use
                    self._flip_center = (cx, cy)
                else:
                    # No flip: use canonical center (annotations will pivot here)
                    self._flip_center = self._canonical_geometry_center
            
            # Draw smart hashmarks if enabled
            if (getattr(self, 'show_hashmarks_var', None) is not None
                    and self.show_hashmarks_var.get()
                    and shape in SmartGeometryEngine.POLYGON_SHAPES):
                self._draw_smart_hashmarks_overlay(ctx)

            # Store tight shape bounds before view scale expands them
            self._shape_bounds = {
                "x_min": self.ax.get_xlim()[0],
                "x_max": self.ax.get_xlim()[1],
                "y_min": self.ax.get_ylim()[0],
                "y_max": self.ax.get_ylim()[1]
            }
            
            # Apply view scale AFTER shape sets its limits
            self._apply_view_scale()
            
            # Track built-in dim line endpoints for drag hit-testing
            self._draw_builtin_dim_lines()

            # Draw standalone freeform labels
            self._draw_standalone_labels()
            
            self.plot_controller.refresh()
        except Exception as e:
            logger.exception("Unhandled error in generate_plot()")
            try:
                self.plot_controller.clear()
                self.plot_controller.setup_axes()
                self.plot_controller.draw_error(f"Error updating plot\n{type(e).__name__}: {e}")
                self.plot_controller.refresh()
            except Exception:
                pass

    def _save_figure(self, path: str | BytesIO, fmt: str = "png") -> None:
        """Save the current figure to a file. Central method for all export paths."""
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
        points: Optional[list] = None

        if shape == "Triangle":
            # Use the stored triangle vertices from geometry hints
            base_p1 = hints.get("tri_base_p1")
            base_p2 = hints.get("tri_base_p2")
            apex = hints.get("tri_apex")
            if base_p1 and base_p2 and apex:
                points = [base_p1, base_p2, apex]
        elif shape == "Polygon":
            points = hints.get("polygon_pts")
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

        # B24/B25 fix: always use the TIGHT shape bounds stored right after drawing,
        # not the current axes limits. The current limits have already been expanded
        # by a previous _apply_view_scale call, so reading them back causes the
        # view to grow unboundedly each time the slider moves or a pan event fires.
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
        # B24 fix: don't call .set() to snap back — that re-fires the command callback
        # and fights the ttk.Scale widget, allowing the variable to drift below 0.25.
        # The widget's from_=0.25 already enforces the minimum visually.
        _raw = self.scale_manager.var("view_scale").get()
        scale = max(0.25, min(1.0, _raw))
        final_width  = base_width  / scale
        final_height = base_height / scale

        # Apply pan offset (data units), clamped so shape can't be dragged
        # fully off-screen (center must stay within ±half the base extent)
        pan_x, pan_y = getattr(self, '_shape_pan_offset', (0.0, 0.0))
        max_pan_x = final_width  * 0.5
        max_pan_y = final_height * 0.5
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
        if config.has_flip():
            self._flip_with_annotations(axis)



if __name__ == "__main__":
    root = tk.Tk()
    app = GeometryApp(root)
    root.mainloop()



# ================== GEOMETRY FORGE - TASK TRACKER ====================
# Legend: [B]=Bug, [Q]=Quality, [G]=Engine, [U]=UI, [F]=Future, [R]=Regression, [D]=Dead Code
# Version: 4.8

# ---------------------------- [BUGS] ----------------------------------
# B29. the w dim line for rect prism is inside the shape
# B30. delete button doesn't delete shapes in composite
# B31. triangle doesn't rotate or flip with keys

# ----------------- [CODE QUALITY & ARCHITECTURE] ----------------------

# ----------------------- [GEOMETRY ENGINE] ----------------------------
# G02. add more smart geometry features
# G03. add right triangle preset

# ------------------------- [UI/UX POLISH] -----------------------------
# U02. polish menu UI
# U03. design a menu UI of equal size regardless of shape (doesn't stretch down and change drawing are) which allows a set drawing area
# U01. [low] Replace OS messagebox dialogs with custom styled Toplevel dialogs (blue theme matching selector color #0066cc)

# ------------------------ [FUTURE IDEAS] --------------------------------
# F13. [high] Batch export: generate multiple variants (rotations/flips) in single operation
# F14. [low] Snapping: Add coordinate snapping for "Move" mode to align labels perfectly


# ================== REFACTOR SESSION NOTES =============================
# Current file: Geometry_Forge_4_11.py
# Last completed: Batch 3 (Structural Fixes) — all tests passed.
#
# COMPLETED BATCHES
# -----------------
# Batch 1 — Safe Deletions (v4.9)
#   - Moved logging setup to top of file (before all class definitions)
#   - Removed unused n = len(lengths) in SmartGeometryEngine.detect_congruence
#   - Removed AppConstants.SCALENE_DEFAULT_HEIGHT and SCALENE_DEFAULT_OFFSET (never read)
#   - Removed dead _remove_standalone_label method (replaced by _remove_standalone_annotation)
#
# Batch 2 — Consolidation (v4.10)
#   - Extracted AppConstants.BUILTIN_LABEL_KEYS frozenset; replaced two inline dicts in
#     _on_canvas_press. Also fixed silent bug: "Height (Tri)" was missing from the
#     double-click branch so double-clicking that label never opened edit mode.
#   - Removed _on_flip_h_shortcut / _on_flip_v_shortcut pass-throughs; inlined directly
#     into _bind_shortcuts as lambda e: self._on_flip_shortcut('h'/'v', e)
#   - CompositeTransferList.INTERNAL_NAMES is now derived automatically from DISPLAY_NAMES
#     via {v: k for k, v in DISPLAY_NAMES.items()} — edit only DISPLAY_NAMES going forward
#   - Removed dead current_row variable in InputController.build_from_config; inlined i+1
#   - Triangle / Tri Triangle configs now share _tri_base dict (differ only in num_sides,
#     uses_base_side_flip, rotation_labels, help_text)
#   - Triangular Prism / Tri Prism configs now share _tri_prism_base dict (same pattern)
#
# Batch 3 — Structural Fixes (v4.11)
#   - Fixed misplaced comment in _on_composite_press (line was indented inside the
#     for ep_key loop body instead of at the if-hasattr level — cosmetic but misleading)
#   - Removed redundant <Delete>/<BackSpace> bindings from _connect_composite_drag.
#     _on_delete_shortcut (bound globally via bind_all in _bind_shortcuts) already
#     dispatches to _on_composite_delete() when _is_composite_shape() is True.
#     The duplicate root.bind() calls competed with bind_all and the winner depended
#     on binding order. Also removed the never-read _composite_key_bindings list.
#   - Added HistoryManager.restoring() context manager (contextlib.contextmanager).
#     _apply_state now uses `with self.history_manager.restoring():` instead of manually
#     setting is_restoring = True / finally: is_restoring = False. Flag cannot get stuck.
#   - Added StandaloneDimLine and CompositeDimLine TypedDicts (top of file, after __all__).
#     _standalone_dim_lines and _composite_dim_lines list declarations now use these types.
#     All existing .get("label_x/y", midpoint_fallback) call sites are unchanged and correct
#     — label_x/label_y are typed float | None, making the fallback intent explicit.
#
# REMAINING BATCHES
# -----------------
# Batch 4 — Safety Hardening (medium risk — next up)
#   [4a] draw_circumference_arc arrow tangents: replace hardcoded (±0.7, ±0.7) magic
#        numbers with tangent vectors derived mathematically from the arc angle.
#        Location: RadialLabelMixin.draw_circumference_arc — the "top" / "right" /
#        "left" / "bottom" orientation branches each have hardcoded tuples like
#        left_tangent = (0.7, -0.7). Derive as (sin(angle), -cos(angle)) instead.
#   [4b] copy_to_clipboard failure feedback: replace silent failure path with
#        messagebox.showwarning on subprocess error; ensure Windows temp file is
#        cleaned up in a finally block. Location: GeometryApp.copy_to_clipboard.
#   [4c] _apply_view_scale pan clamping: tighten max pan from final_width * 0.5
#        to ~0.35 so shape center cannot be dragged fully off-screen.
#        Location: GeometryApp._apply_view_scale, pan_x/pan_y clamping block.
#   [4d] _on_text_change mutual exclusivity: debounce field-clearing so sibling
#        fields (Height ↔ Left/Right Side, Radius ↔ Diameter) are only cleared
#        on focus-out or Enter, not on every keystroke.
#        Location: InputController._on_text_change.
#
# Batch 5 — Set and State Consolidation (low-medium risk)
#   [5a] Unify the four overlapping frozensets into one capability map:
#          SmartGeometryEngine.POLYGON_SHAPES
#          GeometricRotation.GEOMETRIC_SHAPES
#          GeometricRotation.GEOMETRIC_3D_SHAPES
#          GeometricRotation.ARC_SYMMETRIC_SHAPES
#        Replace with a single dict mapping shape name → capability flags, then
#        derive all four frozensets from it.
#   [5b] Move GeometricRotation.ALL_GEOMETRIC_SHAPES computation to module level
#        (after the class definition) to eliminate class-body evaluation order risk.
#   [5c] Resolve ValidationError: either catch it specifically in generate_plot's
#        except block, or replace the raise sites in triangle drawers with the
#        ShapeValidator return-string pattern used everywhere else.
#
# Batch 6 — Architecture (highest risk, multi-session — do last)
#   [6a] Extract CompositeDragController from GeometryApp: move all
#        _on_composite_press/motion/release logic, _composite_dim_lines,
#        _composite_bboxes, and related state into a dedicated class.
#   [6b] Extract StandaloneAnnotationController from GeometryApp: same treatment
#        for _standalone_labels, _standalone_dim_lines, and their mouse handlers.
#   [6c] Convert ShapeConfigProvider class-level mutable _configs dict to a
#        module-level constant (initialized once, never mutated after startup).
#   [6d] Audit GeometricRotation.rotate_axes_artists / flip_axes_artists for
#        matplotlib private API access (_posA_posB, _path, etc.) and replace
#        with supported public equivalents where available.