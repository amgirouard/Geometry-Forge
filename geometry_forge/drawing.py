from __future__ import annotations

import math
import logging
import numpy as np
import matplotlib.patches as patches

from .models import (
    Point, Polygon, AppConstants, DrawingContext,
    SHAPE_CAPABILITIES,
)

logger = logging.getLogger(__name__)


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
                                 size: float | None = None) -> None:
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
                       hash_len: float | None = None, count: int = 1) -> None:
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
    def dim_offset_from_axes(ax, px: float | None = None) -> float:
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


class SmartGeometryEngine:
    """Detects congruent sides and right angles for smart hashmark/marker drawing."""

    TOL_SIDE = 0.01       # Fractional tolerance for side-length equality
    TOL_ANGLE = 0.1       # Degrees tolerance for right angle detection

    # Shapes that have meaningful polygon vertices to analyze
    # Derived from SHAPE_CAPABILITIES — do not edit directly.
    POLYGON_SHAPES = frozenset(
        name for name, caps in SHAPE_CAPABILITIES.items() if caps["polygon"]
    )

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


class GeometricRotation:
    """Computes geometric transforms for annotation tracking during rotation.
    
    2D polygon shapes (Category A) rotate their geometry when base_side changes.
    This class computes the matching transform so freeform labels and dimension
    lines can follow the shape.  3D orientation shapes (Category B) redraw with
    fundamentally different geometry per orientation, so annotations are NOT
    transformed for those.
    """
    
    # Derived from SHAPE_CAPABILITIES — do not edit directly.
    # 2D polygon shapes where annotations should follow geometric rotation.
    GEOMETRIC_SHAPES: frozenset[str] = frozenset(
        name for name, caps in SHAPE_CAPABILITIES.items() if caps["geometric_2d"]
    )
    # 3D shapes using the post-draw artist transform pipeline.
    GEOMETRIC_3D_SHAPES: frozenset[str] = frozenset(
        name for name, caps in SHAPE_CAPABILITIES.items() if caps["geometric_3d"]
    )
    # Arc-based radially symmetric shapes — flip normalises to base_side.
    ARC_SYMMETRIC_SHAPES: frozenset[str] = frozenset(
        name for name, caps in SHAPE_CAPABILITIES.items() if caps["arc_symmetric"]
    )
    # Union of all shapes that use the post-draw artist transform pipeline.
    # Defined here (after GEOMETRIC_SHAPES and GEOMETRIC_3D_SHAPES) so the
    # class body is the single source of truth — no post-class monkey-patch needed.
    ALL_GEOMETRIC_SHAPES: frozenset[str] = (
        frozenset(name for name, caps in SHAPE_CAPABILITIES.items() if caps["geometric_2d"])
        | frozenset(name for name, caps in SHAPE_CAPABILITIES.items() if caps["geometric_3d"])
    )
    
    # Map of shape name → geometry_hints key that stores display vertices.
    # Entries here must cover every shape in SHAPE_CAPABILITIES with "polygon": True
    # that uses post-draw artist rotation (i.e. also has "geometric_2d": True).
    # WARNING: if a new polygon shape is added to SHAPE_CAPABILITIES with both
    # "polygon" and "geometric_2d" set to True, a matching entry must be added
    # here AND the drawer must store its vertices in label_manager.geometry_hints
    # under the corresponding key — otherwise _rotate_geometry_hints will silently
    # skip rotating that shape's hint vertices after a keyboard rotation.
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
                                     new_base_side: int) -> float | None:
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
                                angle: float) -> tuple[float, float] | None:
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
                    if artist._posA_posB is None:
                        continue
                    posA, posB = artist._posA_posB
                    artist.set_positions(
                        cls.rotate_point(posA[0], posA[1], angle_rad, cx, cy),
                        cls.rotate_point(posB[0], posB[1], angle_rad, cx, cy),
                    )
                
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
                if artist._posA_posB is None:
                    pass
                else:
                    posA, posB = artist._posA_posB
                    artist.set_positions(rp(posA[0], posA[1]), rp(posB[0], posB[1]))
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
            except Exception as exc:
                logger.debug("recalculate_limits: skipping artist %r: %s", artist, exc)
        
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
                    if artist._posA_posB is None:
                        continue
                    posA, posB = artist._posA_posB
                    artist.set_positions(
                        mirror(posA[0], posA[1]),
                        mirror(posB[0], posB[1]),
                    )

                elif cls_name in ('FancyBboxPatch', 'Rectangle'):
                    pass  # follow parent Text / skip background

            except Exception:
                logger.debug("Could not flip artist %s", cls_name)




