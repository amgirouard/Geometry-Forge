from __future__ import annotations

import math
import logging
import numpy as np
import matplotlib.patches as patches
from matplotlib.axes import Axes

from .models import (
    Point, Polygon, AppConstants, DrawingContext, DrawingDependencies,
    TransformState, ShapeConfig, ShapeConfigProvider, ShapeFeature,
    TriangleType, PolygonType, ValidationError,
    SHAPE_CAPABILITIES,
)
from .validators import ShapeValidator
from .drawing import DrawingUtilities, SmartGeometryEngine
from .labels import LabelManager, PolygonLabelMixin, RadialLabelMixin, ArrowMixin

logger = logging.getLogger(__name__)


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
        """Get numeric parameter from params dict with fallback to scaled default.

        Returns the stored value when positive and finite.
        Falls back to `default * aspect_ratio` for missing / non-finite inputs.
        Raises ValidationError for zero or negative values — callers must not
        pass dimension params that are intentionally non-positive.
        """
        num_key = f"{key}_num"
        if num_key in params and params[num_key] is not None:
            value = params[num_key]
            if not math.isfinite(value):
                return default * aspect_ratio
            if value == 0:
                raise ValidationError(f"{key} must be positive")
            if value < 0:
                raise ValidationError(f"{key} must be positive")
            if value > 0:
                return value
        return default * aspect_ratio
    
    # -------------------- Validation Helpers --------------------
    
    def validate_positive_params(self, params: dict, keys: list[str]) -> str | None:
        """Validate that all specified parameters are positive if present.
        Returns first error message or None (does NOT raise)."""
        for key in keys:
            err = ShapeValidator.validate_positive(key, params.get(f"{key}_num"))
            if err:
                return err
        return None
    
    def validate_pairs_equal(self, params: dict, 
                             pairs: list[tuple[str, str]]) -> str | None:
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
    
    def transform_points(self, points: Polygon, center: Point | None = None,
                         transform: TransformState | None = None) -> Polygon:
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
    
    def set_limits(self, ctx: DrawingContext, x_range: tuple, y_range: tuple) -> None:
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
            raise ValidationError(f"Circle radius must be positive, got {radius}")
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
            style = ctx.dash_args
        else:
            style = {"color": ctx.line_color, "lw": AppConstants.DEFAULT_LINE_WIDTH}
        ctx.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **style)

    def draw_dim_line(self, ctx: DrawingContext, key: str, p1: Point, p2: Point) -> None:
        """Draw a dashed dimension line that highlights blue when its label is selected.

        Args:
            ctx: Drawing context (provides axes and line color).
            key: The label key this dim line belongs to (e.g. "Height", "Radius").
                 Used to look up any user-dragged offset and to check selection state.
            p1: Start point in data coordinates (before drag offset is applied).
            p2: End point in data coordinates (before drag offset is applied).
        """
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
        """Draw a dimension line with endpoint dots and a positioned label.

        Checks visibility via label_manager; draws nothing if the label is hidden
        or has no text.  Applies custom_dim_offsets so the dots and label move
        together with the line when the user drags it.

        Args:
            ctx: Drawing context.
            key: Label key (e.g. "Height", "Radius") — drives visibility check,
                 selection highlight, and drag-offset lookup.
            p1: First endpoint of the dimension line (before drag offset).
            p2: Second endpoint of the dimension line (before drag offset).
            label_x: Default label X position in data coordinates (before drag offset).
            label_y: Default label Y position in data coordinates (before drag offset).
            label_ha: Horizontal alignment of the label text ("left"/"center"/"right").
            label_va: Vertical alignment of the label text ("top"/"center"/"bottom").
        """
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
                        hash_len: float | None = None, count: int = 1) -> None:
        """Draw tick marks perpendicular to each segment of *points*.

        Convenience wrapper around DrawingUtilities.draw_hash_marks.

        Args:
            ctx: Drawing context (provides axes and line color).
            points: List of 2-D points defining the segment(s) to mark.
                    Each consecutive pair (points[i], points[i+1]) receives marks.
            hash_len: Length of each tick in data units.
                      Defaults to AppConstants.HASH_MARK_LENGTH when None.
            count: Number of parallel ticks to draw per segment (1 = single mark,
                   2 = double, etc.).  Used to indicate congruence groups.
        """
        DrawingUtilities.draw_hash_marks(ctx, points, hash_len, count)

    def draw_right_angle_marker(self, ctx: DrawingContext, vertex: Point,
                                 p1_dir: Point, p2_dir: Point,
                                 size: float | None = None) -> None:
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
        """Rotate polygon so the specified side becomes the base (bottom).

        The result is always normalised so that (min_x, min_y) == (0, 0).
        This is correct for standalone shape drawing where the absolute position
        is irrelevant, but it DESTROYS position information.  Do NOT use this
        method for shapes that are already placed at a specific position in a
        composite canvas — the normalisation would move them to the origin.
        """
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



class ShapeRegistry:
    """Registry for shape drawer classes using a simple factory pattern."""
    
    _drawers: dict[str, type[ShapeDrawer]] = {}
    
    @classmethod
    def register(cls, shape_name: str):
        """Decorator to register a shape drawer class."""
        def decorator(drawer_class: type):
            cls._drawers[shape_name] = drawer_class
            return drawer_class
        return decorator
    
    @classmethod
    def get_drawer(cls, shape_name: str, deps: DrawingDependencies) -> ShapeDrawer | None:
        """Get a new drawer instance for a shape, or None if not registered."""
        if shape_name not in cls._drawers:
            return None
        
        return cls._drawers[shape_name](deps)


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
        
        # Only show hash marks when the checkbox is enabled and not in composite mode.
        if ctx.show_hashmarks and not ctx.composite_mode:
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
            TriangleType.RIGHT.value: self._draw_right,
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
                               equal_sides: list[int] | None = None,
                               height_label_key: str = "Height") -> Polygon:
        xs = [p[0] for p in base_pts]
        ys = [p[1] for p in base_pts]
        center = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
        
        pts = self.transform_points(base_pts, center, transform)
        self.draw_polygon(ctx, pts)
        x_range, y_range = self.calculate_bounds(pts)
        self.set_limits(ctx, x_range, y_range)
        if equal_sides and ctx.show_hashmarks:
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
            show_height=True,
            height_label_key="Height"
        )
    
    def _draw_scalene(self, ctx: DrawingContext, transform: TransformState,
                      params: dict) -> None:
        aspect = ctx.aspect_ratio
        # Derive geometry from reference side lengths (b=9, l=5, r=7)
        # so the default shape matches Custom mode with those values.
        # All three sides are intentionally different; no Pythagorean triple; l+r > b (12 > 9).
        b_ref = AppConstants.SCALENE_DEFAULT_BASE   # 9.0
        l_ref, r_ref = AppConstants.SCALENE_DEFAULT_L, AppConstants.SCALENE_DEFAULT_R
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
            equal_sides=[0, 1, 2]
        )


    def _draw_right(self, ctx: DrawingContext, transform: TransformState,
                    params: dict) -> None:
        """Right triangle with 90° at bottom-left.

        Default 3-4-5: base=4 (Leg 1), height=3 (Leg 2), hypotenuse=5.
        Slider (aspect) adjusts the ratio: left = taller, right = wider.
          b = RIGHT_DEFAULT_BASE * aspect
          l = RIGHT_DEFAULT_LEG  / aspect
        The right angle stays at bottom-left regardless of slider position.
        """
        aspect = ctx.aspect_ratio
        b = AppConstants.RIGHT_DEFAULT_BASE * aspect
        l = AppConstants.RIGHT_DEFAULT_LEG / aspect
        # Apex directly above origin → right angle at (0,0)
        base_pts = self._get_base_vertices(b, l, 0.0)
        label_keys = ["Leg B", "Hypotenuse", "Leg A"]
        base_pts, label_keys, _ = self._rotate_to_base(
            base_pts, label_keys, transform.base_side
        )
        self._draw_triangle_common(
            ctx, transform, base_pts, label_keys,
            show_height=False,
        )
        # Draw right angle marker at the 90° vertex (index 0 in canonical orientation)
        if transform.base_side == 0:
            base_dir = (base_pts[1][0] - base_pts[0][0], base_pts[1][1] - base_pts[0][1])
            left_dir = (base_pts[2][0] - base_pts[0][0], base_pts[2][1] - base_pts[0][1])
            self.draw_right_angle_marker(ctx, base_pts[0], base_dir, left_dir)


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
        if ctx.show_hashmarks:
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
                        label_key: str | None = None) -> None:
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



