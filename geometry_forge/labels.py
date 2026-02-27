from __future__ import annotations

import math
from typing import Any
import numpy as np
import matplotlib.patches as patches

from .models import (
    Point, Polygon, AppConstants, DrawingContext,
)

logger = __import__('logging').getLogger(__name__)


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
        self.builtin_selected: str | None = None
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
    
    def reset_all_custom_positions(self) -> None:
        self.custom_positions.clear()
        self.custom_dim_offsets.clear()

    def get_state(self) -> dict:
        """Return serialisable snapshot of label positioning state."""
        return {
            "custom_positions": self.custom_positions.copy(),
            "custom_dim_offsets": {k: list(v) for k, v in self.custom_dim_offsets.items()},
        }

    def set_state(self, state: dict) -> None:
        """Restore label positioning state from a snapshot dict."""
        self.custom_positions = state.get("custom_positions", {}).copy()
        self.custom_dim_offsets = {
            k: tuple(v) for k, v in state.get("custom_dim_offsets", {}).items()
        }

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
                         offsets: list[float] | None = None) -> None:
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
                                label_offset: float | None = None,
                                ellipse_ratio: float | None = None,
                                base_position: str = "bottom") -> None:
        """Draw circumference indicator arc with arrowheads pointing away.

        Dispatches to one of three private helpers depending on orientation
        and whether an ellipse_ratio is provided:
          _arc_flat         — true circle (2D Circle / Sphere)
          _arc_3d_horiz     — elliptical base, horizontal orientation (base top or bottom)
          _arc_3d_vert      — elliptical base, vertical orientation (base left or right)

        All arc segments and arrowheads use ctx.line_color for consistency.

        Args:
            ctx:           Drawing context.
            center:        Center point of the circle/ellipse.
            radius:        Radius (or semi-major axis for ellipse).
            orientation:   "horizontal" for upright shapes, "vertical" for shapes on side.
            label_offset:  Offset for label positioning.
            ellipse_ratio: For 3D bases, minor/major axis ratio (e.g. 0.3 for foreshortened circle).
            base_position: Where the circular base is located.
                           horizontal: "bottom" or "top"
                           vertical:   "left"   or "right"
        """
        if label_offset is None:
            label_offset = AppConstants.LABEL_OFFSET_RADIAL

        valid_positions = {"bottom", "top", "left", "right"}
        if base_position not in valid_positions:
            logger.warning(
                "draw_circumference_arc: unrecognized base_position %r, defaulting to 'bottom'",
                base_position,
            )
            base_position = "bottom"

        txt_c, show_c = self.label_manager.get_entry_values("Circumference")
        if not (txt_c.strip() and show_c):
            return

        arc_color = (
            "#0066cc"
            if self.label_manager.builtin_selected == "Circumference"
            else ctx.line_color
        )
        arc_lw = (
            AppConstants.DIMENSION_LINE_WIDTH * 1.5
            if self.label_manager.builtin_selected == "Circumference"
            else AppConstants.DIMENSION_LINE_WIDTH
        )
        arrow_size = min(0.15, radius * 0.06)

        if orientation == "horizontal" and ellipse_ratio is None:
            self._arc_flat(ctx, center, radius, label_offset, arc_color, arc_lw, arrow_size)
        elif orientation == "horizontal" and ellipse_ratio is not None:
            self._arc_3d_horiz(ctx, center, radius, label_offset, ellipse_ratio,
                               base_position, arc_color, arc_lw)
        elif orientation == "vertical" and ellipse_ratio is not None:
            self._arc_3d_vert(ctx, center, radius, label_offset, ellipse_ratio,
                              base_position, arc_color, arc_lw)
        else:
            # Fallback: vertical orientation without ellipse_ratio
            arc_radius_x = label_offset
            arc_radius_y = radius + label_offset
            ctx.ax.add_patch(patches.Arc(
                center, arc_radius_x * 2, arc_radius_y * 2,
                theta1=-90, theta2=90,
                color=arc_color, lw=arc_lw, linestyle="--",
            ))
            top_pt = (center[0], center[1] + arc_radius_y)
            bottom_pt = (center[0], center[1] - arc_radius_y)
            self._draw_arrowhead(ctx, top_pt, (-1, 0), arrow_size, color=arc_color)
            self._draw_arrowhead(ctx, bottom_pt, (-1, 0), arrow_size, color=arc_color)
            label_x = center[0] + arc_radius_x + label_offset
            self.draw_text(label_x, center[1], "Circumference",
                           use_background=True, ha="left", va="center")

    def _arc_flat(self, ctx: DrawingContext, center: Point, radius: float,
                  label_offset: float, arc_color: str, arc_lw: float,
                  arrow_size: float) -> None:
        """True-circle circumference arc (2D Circle / Sphere): 270° dashed arc."""
        arc_radius = radius + label_offset
        ctx.ax.add_patch(patches.Arc(
            center, arc_radius * 2, arc_radius * 2,
            theta1=45, theta2=315,
            color=arc_color, lw=arc_lw, linestyle="--",
        ))
        start_rad = np.radians(45)
        start_pt = (center[0] + arc_radius * np.cos(start_rad),
                    center[1] + arc_radius * np.sin(start_rad))
        end_rad = np.radians(315)
        end_pt = (center[0] + arc_radius * np.cos(end_rad),
                  center[1] + arc_radius * np.sin(end_rad))
        self._draw_arrowhead(ctx, start_pt,
                             (np.sin(start_rad), -np.cos(start_rad)),
                             arrow_size, centered=True, color=arc_color)
        self._draw_arrowhead(ctx, end_pt,
                             (-np.sin(end_rad), np.cos(end_rad)),
                             arrow_size, centered=True, color=arc_color)
        label_x = center[0] - arc_radius - label_offset * 0.3
        self.draw_text(label_x, center[1], "Circumference",
                       use_background=True, ha="right", va="center")

    def _arc_3d_horiz(self, ctx: DrawingContext, center: Point, radius: float,
                      label_offset: float, ellipse_ratio: float,
                      base_position: str, arc_color: str, arc_lw: float) -> None:
        """Elliptical circumference arc for horizontal-base 3D shapes (top or bottom)."""
        gap = radius * 0.15
        base_width = radius * 2
        base_height = radius * 2 * ellipse_ratio
        arc_width = base_width + gap * 2
        arc_height = base_height + gap * 2
        back_arc_size = gap * 2
        small_arrow = min(0.1, radius * 0.04)
        half_width = arc_width / 2
        left_x = center[0] - half_width
        right_x = center[0] + half_width

        if base_position == "bottom":
            arc_center_y = center[1] - gap
            # Main half-ellipse (bottom) + back-continuation corners
            ctx.ax.add_patch(patches.Arc(
                (center[0], arc_center_y), arc_width, arc_height,
                theta1=180, theta2=360,
                color=arc_color, lw=arc_lw, linestyle="--",
            ))
            ctx.ax.add_patch(patches.Arc(
                (left_x + gap, arc_center_y), back_arc_size, back_arc_size,
                theta1=135, theta2=180,
                color=arc_color, lw=arc_lw, linestyle="--",
            ))
            ctx.ax.add_patch(patches.Arc(
                (right_x - gap, arc_center_y), back_arc_size, back_arc_size,
                theta1=0, theta2=45,
                color=arc_color, lw=arc_lw, linestyle="--",
            ))
            la = np.radians(130)
            ra = np.radians(50)
            self._draw_arrowhead(ctx,
                (left_x + gap + gap * np.cos(la), arc_center_y + gap * np.sin(la)),
                (np.sin(la), -np.cos(la)), small_arrow, color=arc_color)
            self._draw_arrowhead(ctx,
                (right_x - gap + gap * np.cos(ra), arc_center_y + gap * np.sin(ra)),
                (-np.sin(ra), np.cos(ra)), small_arrow, color=arc_color)
            self.draw_text(center[0],
                           arc_center_y - arc_height / 2 - label_offset * 0.3,
                           "Circumference", use_background=True, ha="center", va="top")

        else:  # base_position == "top"
            arc_center_y = center[1] + gap
            ctx.ax.add_patch(patches.Arc(
                (center[0], arc_center_y), arc_width, arc_height,
                theta1=0, theta2=180,
                color=arc_color, lw=arc_lw, linestyle="--",
            ))
            ctx.ax.add_patch(patches.Arc(
                (left_x + gap, arc_center_y), back_arc_size, back_arc_size,
                theta1=180, theta2=225,
                color=arc_color, lw=arc_lw, linestyle="--",
            ))
            ctx.ax.add_patch(patches.Arc(
                (right_x - gap, arc_center_y), back_arc_size, back_arc_size,
                theta1=315, theta2=360,
                color=arc_color, lw=arc_lw, linestyle="--",
            ))
            la = np.radians(220)
            ra = np.radians(320)
            self._draw_arrowhead(ctx,
                (left_x + gap + gap * np.cos(la), arc_center_y + gap * np.sin(la)),
                (np.sin(la), -np.cos(la)), small_arrow, color=arc_color)
            self._draw_arrowhead(ctx,
                (right_x - gap + gap * np.cos(ra), arc_center_y + gap * np.sin(ra)),
                (-np.sin(ra), np.cos(ra)), small_arrow, color=arc_color)
            self.draw_text(center[0],
                           arc_center_y + arc_height / 2 + label_offset * 0.3,
                           "Circumference", use_background=True, ha="center", va="bottom")

    def _arc_3d_vert(self, ctx: DrawingContext, center: Point, radius: float,
                     label_offset: float, ellipse_ratio: float,
                     base_position: str, arc_color: str, arc_lw: float) -> None:
        """Elliptical circumference arc for vertical-base 3D shapes (left or right)."""
        gap = radius * 0.15
        base_width = radius * 2 * ellipse_ratio
        base_height = radius * 2
        arc_width = base_width + gap * 2
        arc_height = base_height + gap * 2
        back_arc_size = gap * 2
        small_arrow = min(0.2, radius * 0.08)
        half_height = arc_height / 2
        top_y = center[1] + half_height
        bottom_y = center[1] - half_height

        if base_position == "left":
            arc_center_x = center[0] - gap
            ctx.ax.add_patch(patches.Arc(
                (arc_center_x, center[1]), arc_width, arc_height,
                theta1=90, theta2=270,
                color=arc_color, lw=arc_lw, linestyle="--",
            ))
            ctx.ax.add_patch(patches.Arc(
                (arc_center_x, top_y - gap), back_arc_size, back_arc_size,
                theta1=45, theta2=90,
                color=arc_color, lw=arc_lw, linestyle="--",
            ))
            ctx.ax.add_patch(patches.Arc(
                (arc_center_x, bottom_y + gap), back_arc_size, back_arc_size,
                theta1=270, theta2=315,
                color=arc_color, lw=arc_lw, linestyle="--",
            ))
            ta = np.radians(50)
            ba = np.radians(310)
            self._draw_arrowhead(ctx,
                (arc_center_x + gap * np.cos(ta), top_y - gap + gap * np.sin(ta)),
                (np.sin(ta), -np.cos(ta)), small_arrow, color=arc_color)
            self._draw_arrowhead(ctx,
                (arc_center_x + gap * np.cos(ba), bottom_y + gap + gap * np.sin(ba)),
                (-np.sin(ba), np.cos(ba)), small_arrow, color=arc_color)
            self.draw_text(arc_center_x - arc_width / 2 - label_offset * 0.3,
                           center[1], "Circumference",
                           use_background=True, ha="right", va="center")

        else:  # base_position == "right"
            arc_center_x = center[0] + gap
            ctx.ax.add_patch(patches.Arc(
                (arc_center_x, center[1]), arc_width, arc_height,
                theta1=-90, theta2=90,
                color=arc_color, lw=arc_lw, linestyle="--",
            ))
            ctx.ax.add_patch(patches.Arc(
                (arc_center_x, top_y - gap), back_arc_size, back_arc_size,
                theta1=90, theta2=135,
                color=arc_color, lw=arc_lw, linestyle="--",
            ))
            ctx.ax.add_patch(patches.Arc(
                (arc_center_x, bottom_y + gap), back_arc_size, back_arc_size,
                theta1=225, theta2=270,
                color=arc_color, lw=arc_lw, linestyle="--",
            ))
            ta = np.radians(130)
            ba = np.radians(230)
            self._draw_arrowhead(ctx,
                (arc_center_x + gap * np.cos(ta), top_y - gap + gap * np.sin(ta)),
                (np.sin(ta), -np.cos(ta)), small_arrow, color=arc_color)
            self._draw_arrowhead(ctx,
                (arc_center_x + gap * np.cos(ba), bottom_y + gap + gap * np.sin(ba)),
                (-np.sin(ba), np.cos(ba)), small_arrow, color=arc_color)
            self.draw_text(arc_center_x + arc_width / 2 + label_offset * 0.3,
                           center[1], "Circumference",
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
                                     label_offset: float | None = None,
                                     ellipse_ratio: float | None = None,
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


