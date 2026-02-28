from __future__ import annotations

import math
import logging
import tkinter as tk
import numpy as np
import matplotlib.patches as patches
from typing import TYPE_CHECKING

from .models import (
    Point, CompositeDimLine, AppConstants, DrawingContext,
    TransformState, ShapeConfigProvider, ShapeConfig,
)
from .drawing import DrawingUtilities, GeometricRotation

if TYPE_CHECKING:
    from .app import GeometryApp

logger = logging.getLogger(__name__)


class CompositeDragController:
    """Manages drag, selection, snap, and annotation state for composite shapes.

    Owns all composite-specific state (positions, transforms, selection, labels,
    dim lines, drag/marquee state, snap guides) and the three mouse-event handlers
    plus all composite action methods (flip, rotate, delete, dim line placement).

    Calls back into GeometryApp via self.app for shared resources (canvas, ax,
    history_manager, generate_plot, root, composite_transfer, etc.).

    Instantiated as self.composite_ctrl in GeometryApp.__init__.
    GeometryApp retains thin @property aliases for all state attrs and one-liner
    delegate methods for all moved methods.
    """

    def __init__(self, app: GeometryApp) -> None:
        self.app = app
        # Shape placement state
        self.positions: dict[int, tuple[float, float]] = {}
        self.transforms: dict[int, dict] = {}
        self.selected: set[int] = set()
        # Label / annotation state
        self.labels: list[dict] = []
        self.label_drag: dict | None = None
        self.selected_label: int | None = None
        # Dim line state
        self.dim_lines: list[CompositeDimLine] = []
        self.selected_dim: int | None = None
        self.dim_mode: str | None = None
        self.dim_first_point: tuple | None = None
        self.edit_mode: dict | None = None
        # Drag / marquee state
        self.drag_state: dict | None = None
        self.marquee: dict | None = None
        self.snap_guides: list = []

    # ── Event connection ──────────────────────────────────────────────────────

    def connect(self) -> None:
        """Connect mouse events for composite shape dragging."""
        self.disconnect()
        self.app._composite_event_ids = [
            self.app.canvas.mpl_connect('button_press_event', self.on_press),
            self.app.canvas.mpl_connect('motion_notify_event', self.on_motion),
            self.app.canvas.mpl_connect('button_release_event', self.on_release),
        ]

    def disconnect(self) -> None:
        """Disconnect composite drag mouse events."""
        for eid in self.app._composite_event_ids:
            try:
                self.app.canvas.mpl_disconnect(eid)
            except Exception:
                pass
        self.app._composite_event_ids = []
        self.drag_state = None
        self.marquee = None

    # ── Mouse event handlers ──────────────────────────────────────────────────

    def on_press(self, event) -> None:
        """Handle mouse press in composite mode."""
        app = self.app
        # Move tkinter focus from any Entry to the canvas so keyboard
        # shortcuts (r/l/h/v) are not blocked by the Entry focus guard.
        try:
            app.canvas.get_tk_widget().focus_set()
        except Exception:
            pass
        if event.inaxes != app.ax:
            return
        if not app._is_composite_shape():
            return
        if not app._composite_bboxes:
            return
        mx, my = event.xdata, event.ydata
        if mx is None or my is None:
            return

        # Right-click on a dim line → toggle shape_owner on/off.
        # Unlinks an owned line so it won't move with its shape (and won't be
        # deleted with it). Right-clicking an unowned line while exactly one
        # shape is selected re-links it to that shape.
        if event.button == 3:
            # Right-click a dim line → toggle ownership
            hit_idx = self._hit_test_dim_line(mx, my, app)
            if hit_idx is not None:
                dim = self.dim_lines[hit_idx]
                if dim.get("shape_owner") is not None:
                    dim["shape_owner"] = None   # unlink
                    dim.pop("preset_key", None)  # no longer auto-snaps
                    # Clear selection so the line immediately loses its blue
                    # highlight — makes it visually obvious it's now free.
                    self.selected_dim = None
                elif len(self.selected) == 1:
                    dim["shape_owner"] = next(iter(self.selected))  # re-link
                    self.selected_dim = hit_idx  # confirm re-link with highlight
                app.generate_plot()
                if not app.history_manager.is_restoring:
                    app._capture_current_state()
                return
            # Right-click a shape → toggle it in/out of the current selection
            # (mirrors how right-clicking lines toggles ownership, and how
            # Ctrl+click works for left-click multi-select)
            if app._composite_bboxes:
                for idx in reversed(range(len(app._composite_bboxes))):
                    bbox = app._composite_bboxes[idx]
                    if bbox != (0, 0, 0, 0) and bbox[0] <= mx <= bbox[2] and bbox[1] <= my <= bbox[3]:
                        if idx in self.selected:
                            self.selected.discard(idx)
                        else:
                            self.selected.add(idx)
                        self.update_selection_ui()
                        app.generate_plot()
                        return
            return

        if event.button != 1:
            return

        # Double-click → enter edit mode
        if getattr(event, 'dblclick', False):
            if app._composite_dim_label_bboxes is not None:
                for idx in reversed(range(len(app._composite_dim_label_bboxes))):
                    bbox = app._composite_dim_label_bboxes[idx]
                    if bbox[0] <= mx <= bbox[2] and bbox[1] <= my <= bbox[3]:
                        self.enter_edit("dim", idx)
                        return
            if app._composite_label_bboxes is not None:
                for idx in reversed(range(len(app._composite_label_bboxes))):
                    bbox = app._composite_label_bboxes[idx]
                    if bbox[0] <= mx <= bbox[2] and bbox[1] <= my <= bbox[3]:
                        self.enter_edit("label", idx)
                        return

        # Dim placement mode
        if self.dim_mode is not None:
            if self.handle_dim_click(mx, my):
                return

        # Dim endpoint hit
        if app._composite_dim_endpoints is not None:
            for idx, endpoints in enumerate(app._composite_dim_endpoints):
                for ep_key in ("p1", "p2"):
                    if ep_key not in endpoints:
                        continue
                    ex, ey = endpoints[ep_key]
                    if abs(mx - ex) < 0.5 and abs(my - ey) < 0.5:
                        self.label_drag = {
                            "type": "dim_endpoint",
                            "idx": idx,
                            "endpoint": ep_key,
                            "start_x": mx, "start_y": my,
                            "orig_x": ex, "orig_y": ey,
                            "dragged": False
                        }
                        self.selected_dim = idx
                        self.selected_label = None
                        self.selected.clear()
                        if not app.history_manager.is_restoring:
                            app._capture_current_state()
                        app.root.config(cursor="fleur")
                        return

        # Dim line label hit
        if app._composite_dim_label_bboxes is not None:
            for idx in reversed(range(len(app._composite_dim_label_bboxes))):
                bbox = app._composite_dim_label_bboxes[idx]
                if bbox[0] <= mx <= bbox[2] and bbox[1] <= my <= bbox[3]:
                    dim = self.dim_lines[idx]
                    label_x = dim.get("label_x", (dim["x1"] + dim["x2"]) / 2)
                    label_y = dim.get("label_y", (dim["y1"] + dim["y2"]) / 2)
                    self.label_drag = {
                        "type": "dim_label",
                        "idx": idx,
                        "start_x": mx, "start_y": my,
                        "orig_x": label_x, "orig_y": label_y,
                        "dragged": False
                    }
                    self.selected_dim = idx
                    self.selected_label = None
                    self.selected.clear()
                    if not app.history_manager.is_restoring:
                        app._capture_current_state()
                    app.root.config(cursor="fleur")
                    return

        # Dim line body hit
        if True:  # always draw dim lines overlay
            for idx in reversed(range(len(self.dim_lines))):
                dim = self.dim_lines[idx]
                x1, y1 = dim["x1"], dim["y1"]
                x2, y2 = dim["x2"], dim["y2"]
                dx = x2 - x1; dy = y2 - y1
                seg_len_sq = dx*dx + dy*dy
                if seg_len_sq == 0:
                    continue
                t = max(0.0, min(1.0, ((mx - x1)*dx + (my - y1)*dy) / seg_len_sq))
                proj_x = x1 + t*dx; proj_y = y1 + t*dy
                dist = math.sqrt((mx - proj_x)**2 + (my - proj_y)**2)
                hit_radius = max((app.ax.get_xlim()[1] - app.ax.get_xlim()[0]) * 0.015, 0.2)
                if dist < hit_radius and 0.05 < t < 0.95:
                    self.label_drag = {
                        "type": "dim_body",
                        "idx": idx,
                        "start_x": mx, "start_y": my,
                        "orig_x1": x1, "orig_y1": y1,
                        "orig_x2": x2, "orig_y2": y2,
                        "orig_lx": dim.get("label_x", (x1 + x2) / 2),
                        "orig_ly": dim.get("label_y", (y1 + y2) / 2),
                        "dragged": False
                    }
                    self.selected_dim = idx
                    self.selected_label = None
                    self.selected.clear()
                    if not app.history_manager.is_restoring:
                        app._capture_current_state()
                    app.root.config(cursor="fleur")
                    return

        # Label hit
        if app._composite_label_bboxes is not None:
            for idx in reversed(range(len(app._composite_label_bboxes))):
                bbox = app._composite_label_bboxes[idx]
                if bbox[0] <= mx <= bbox[2] and bbox[1] <= my <= bbox[3]:
                    self.label_drag = {
                        "idx": idx,
                        "start_x": mx, "start_y": my,
                        "orig_x": self.labels[idx]["x"],
                        "orig_y": self.labels[idx]["y"],
                        "dragged": False
                    }
                    if not app.history_manager.is_restoring:
                        app._capture_current_state()
                    app.root.config(cursor="fleur")
                    return

        # Shape hit
        multi = bool(event.key and ("control" in event.key or "super" in event.key
                                     or "ctrl" in str(event.key).lower()))
        for idx in reversed(range(len(app._composite_bboxes))):
            bbox = app._composite_bboxes[idx]
            if bbox[0] <= mx <= bbox[2] and bbox[1] <= my <= bbox[3]:
                was_selected_before = idx in self.selected
                if multi:
                    if idx in self.selected:
                        self.selected.discard(idx)
                    else:
                        self.selected.add(idx)
                else:
                    if idx not in self.selected:
                        self.selected = {idx}
                self.selected_label = None
                self.selected_dim = None
                self.update_selection_ui()

                orig_positions = {}
                for sel_idx in self.selected:
                    orig_positions[sel_idx] = self.positions.get(sel_idx, (0.0, 0.0))

                orig_labels = []
                orig_dims = []

                # Collect dim lines by ownership so offset/user-dragged lines
                # always travel with their shape, regardless of position.
                owned_dim_idxs = [
                    i for i, d in enumerate(self.dim_lines)
                    if d.get("shape_owner") in self.selected
                ]
                orig_dims = [
                    (i, self.dim_lines[i]["x1"], self.dim_lines[i]["y1"],
                     self.dim_lines[i]["x2"], self.dim_lines[i]["y2"],
                     self.dim_lines[i].get("label_x"), self.dim_lines[i].get("label_y"))
                    for i in owned_dim_idxs
                ]

                # Labels don't have ownership yet — use spatial fallback
                if len(self.selected) > 1:
                    all_sel = self.all_shapes_selected()
                    grp_bb = self.get_group_bbox()
                    if grp_bb != (0, 0, 0, 0):
                        lbl_idxs, _ = self.get_annotations_in_region(*grp_bb, include_all=all_sel)
                        orig_labels = [(i, self.labels[i]["x"], self.labels[i]["y"]) for i in lbl_idxs]
                elif idx < len(app._composite_bboxes):
                    s_bbox = app._composite_bboxes[idx]
                    if s_bbox != (0, 0, 0, 0):
                        lbl_idxs, _ = self.get_annotations_in_region(*s_bbox)
                        orig_labels = [(i, self.labels[i]["x"], self.labels[i]["y"]) for i in lbl_idxs]

                self.drag_state = {
                    "idx": idx,
                    "start_x": mx, "start_y": my,
                    "orig_pos": self.positions.get(idx, (0.0, 0.0)),
                    "orig_positions": orig_positions,
                    "dragged": False,
                    "multi": multi,
                    "was_selected_before": was_selected_before,
                    "orig_labels": orig_labels,
                    "orig_dims": orig_dims,
                }
                if not app.history_manager.is_restoring:
                    app._capture_current_state()
                app.root.config(cursor="fleur")
                return

        # Deselect label/dim
        if self.selected_label is not None or self.selected_dim is not None:
            self.selected_label = None
            self.selected_dim = None
            self.update_selection_ui()
            app.generate_plot()

        # Start marquee
        self.marquee = {"start_x": mx, "start_y": my, "rect_artist": None}

    def on_motion(self, event) -> None:
        """Handle mouse motion — drag shapes/labels, draw marquee, or hover cursor."""
        app = self.app

        # Label or dim drag
        if self.label_drag is not None:
            if event.inaxes != app.ax:
                return
            mx, my = event.xdata, event.ydata
            if mx is None or my is None:
                return
            drag = self.label_drag
            drag["dragged"] = True
            dx = mx - drag["start_x"]
            dy = my - drag["start_y"]
            idx = drag["idx"]
            dtype = drag.get("type")

            if dtype == "dim_endpoint":
                if idx < len(self.dim_lines):
                    dim = self.dim_lines[idx]
                    ep = drag["endpoint"]
                    new_x = drag["orig_x"] + dx
                    new_y = drag["orig_y"] + dy
                    if dim.get("constraint") == "height":
                        new_x = drag["orig_x"]
                    elif dim.get("constraint") == "width":
                        new_y = drag["orig_y"]
                    if ep == "p1":
                        dim["x1"], dim["y1"] = new_x, new_y
                    elif ep == "p2":
                        dim["x2"], dim["y2"] = new_x, new_y
                    # Mark as user-dragged so the re-snap loop doesn't overwrite position
                    dim["user_dragged"] = True
                    app.generate_plot()
            elif dtype == "dim_label":
                if idx < len(self.dim_lines):
                    self.dim_lines[idx]["label_x"] = drag["orig_x"] + dx
                    self.dim_lines[idx]["label_y"] = drag["orig_y"] + dy
                    app.generate_plot()
            elif dtype == "dim_body":
                if idx < len(self.dim_lines):
                    dim = self.dim_lines[idx]
                    dim["x1"] = drag["orig_x1"] + dx
                    dim["y1"] = drag["orig_y1"] + dy
                    dim["x2"] = drag["orig_x2"] + dx
                    dim["y2"] = drag["orig_y2"] + dy
                    if "label_x" in dim and "label_y" in dim:
                        dim["label_x"] = drag["orig_lx"] + dx
                        dim["label_y"] = drag["orig_ly"] + dy
                    # Mark as user-dragged so the re-snap loop doesn't overwrite position
                    dim["user_dragged"] = True
                    app.generate_plot()
            else:
                if idx < len(self.labels):
                    self.labels[idx]["x"] = drag["orig_x"] + dx
                    self.labels[idx]["y"] = drag["orig_y"] + dy
                    app.generate_plot()
            return

        # Dim mode preview
        if self.dim_mode is not None and self.dim_first_point is not None:
            if event.inaxes == app.ax and event.xdata is not None:
                mx, my = event.xdata, event.ydata
                x1, y1 = self.dim_first_point
                x2, y2 = mx, my
                if self.dim_mode == "height":
                    x2 = x1
                elif self.dim_mode == "width":
                    y2 = y1
                if app._dim_preview_line:
                    try:
                        app._dim_preview_line.remove()
                    except (ValueError, AttributeError):
                        pass
                app._dim_preview_line = app.ax.plot(
                    [x1, x2], [y1, y2],
                    color="#4488ff", linestyle="--", linewidth=1.0, alpha=0.6, zorder=20
                )[0]
                app.canvas.draw_idle()
            return

        # Marquee
        if self.marquee is not None:
            if event.inaxes != app.ax:
                return
            mx, my = event.xdata, event.ydata
            if mx is None or my is None:
                return
            m = self.marquee
            x0, y0 = m["start_x"], m["start_y"]
            if m["rect_artist"] is not None:
                try:
                    m["rect_artist"].remove()
                except (ValueError, AttributeError):
                    pass
            rx = min(x0, mx); ry = min(y0, my)
            rw = abs(mx - x0); rh = abs(my - y0)
            m["rect_artist"] = app.ax.add_patch(patches.Rectangle(
                (rx, ry), rw, rh,
                edgecolor="#4488ff", facecolor="#4488ff",
                alpha=0.12, linewidth=1.2, linestyle="-", zorder=25
            ))
            app.canvas.draw_idle()
            return

        if self.drag_state is None:
            # Hover cursor
            if event.inaxes == app.ax and event.xdata is not None and event.ydata is not None:
                mx, my = event.xdata, event.ydata
                hit = False
                hit_radius = max((app.ax.get_xlim()[1] - app.ax.get_xlim()[0]) * 0.015, 0.2)
                if app._composite_label_bboxes is not None:
                    for bbox in app._composite_label_bboxes:
                        if bbox[0] <= mx <= bbox[2] and bbox[1] <= my <= bbox[3]:
                            hit = True; break
                if not hit and app._composite_dim_label_bboxes is not None:
                    for bbox in app._composite_dim_label_bboxes:
                        if bbox[0] <= mx <= bbox[2] and bbox[1] <= my <= bbox[3]:
                            hit = True; break
                if not hit and app._composite_dim_endpoints is not None:
                    for ep in app._composite_dim_endpoints:
                        p1, p2 = ep["p1"], ep["p2"]
                        for pt in (p1, p2):
                            if math.sqrt((mx - pt[0])**2 + (my - pt[1])**2) < hit_radius:
                                hit = True; break
                        if not hit:
                            dx2, dy2 = p2[0] - p1[0], p2[1] - p1[1]
                            seg_len_sq = dx2*dx2 + dy2*dy2
                            if seg_len_sq > 0:
                                t2 = max(0.0, min(1.0, ((mx - p1[0])*dx2 + (my - p1[1])*dy2) / seg_len_sq))
                                dist2 = math.sqrt((mx - p1[0] - t2*dx2)**2 + (my - p1[1] - t2*dy2)**2)
                                if dist2 < hit_radius:
                                    hit = True
                        if hit:
                            break
                if not hit and app._composite_bboxes is not None:
                    for bbox in app._composite_bboxes:
                        if bbox != (0, 0, 0, 0) and bbox[0] <= mx <= bbox[2] and bbox[1] <= my <= bbox[3]:
                            hit = True; break
                desired = "fleur" if hit else "arrow"
                if app.root.cget("cursor") != desired:
                    app.root.config(cursor=desired)
            else:
                if app.root.cget("cursor") not in ("crosshair", "fleur"):
                    app.root.config(cursor="arrow")
            return

        if event.inaxes != app.ax:
            return
        mx, my = event.xdata, event.ydata
        if mx is None or my is None:
            return

        drag = self.drag_state
        idx = drag["idx"]
        dx = mx - drag["start_x"]
        dy = my - drag["start_y"]
        new_x = drag["orig_pos"][0] + dx
        new_y = drag["orig_pos"][1] + dy

        if "half_w" not in drag and idx < len(app._composite_bboxes):
            bbox = app._composite_bboxes[idx]
            if bbox != (0, 0, 0, 0):
                drag["half_w"] = (bbox[2] - bbox[0]) / 2
                drag["half_h"] = (bbox[3] - bbox[1]) / 2

        snap_dx = 0.0; snap_dy = 0.0
        self.snap_guides = []
        is_group = idx in self.selected and len(self.selected) > 1

        if "half_w" in drag:
            prov_cx = new_x; prov_cy = new_y
            hw, hh = drag["half_w"], drag["half_h"]
            prov_bbox = (prov_cx - hw, prov_cy - hh, prov_cx + hw, prov_cy + hh)
            excluded = self.selected if is_group else {idx}
            other_bboxes = []
            for i, bbox in enumerate(app._composite_bboxes):
                if i in excluded:
                    other_bboxes.append((0, 0, 0, 0))
                else:
                    other_bboxes.append(bbox)
            if idx < len(app._composite_snap_anchors):
                cur_pos = self.positions.get(idx, (0.0, 0.0))
                anchor_shift_x = new_x - cur_pos[0]
                anchor_shift_y = new_y - cur_pos[1]
                orig_anchors = app._composite_snap_anchors[idx]
                app._composite_snap_anchors[idx] = [(ax2 + anchor_shift_x, ay2 + anchor_shift_y) for ax2, ay2 in orig_anchors]
            snap_dx, snap_dy, guides = self.calc_snap(idx, prov_bbox, other_bboxes)
            if idx < len(app._composite_snap_anchors):
                app._composite_snap_anchors[idx] = orig_anchors
            new_x += snap_dx; new_y += snap_dy
            self.snap_guides = guides

        drag["dragged"] = True
        actual_dx = new_x - drag["orig_pos"][0]
        actual_dy = new_y - drag["orig_pos"][1]

        if is_group:
            for sel_idx in self.selected:
                orig = drag["orig_positions"].get(sel_idx, (0.0, 0.0))
                self.positions[sel_idx] = (orig[0] + actual_dx, orig[1] + actual_dy)
        else:
            self.positions[idx] = (new_x, new_y)

        for lbl_i, ox, oy in drag.get("orig_labels", []):
            if lbl_i < len(self.labels):
                self.labels[lbl_i]["x"] = ox + actual_dx
                self.labels[lbl_i]["y"] = oy + actual_dy
        for dim_entry in drag.get("orig_dims", []):
            dim_i, ox1, oy1, ox2, oy2, olx, oly = dim_entry
            if dim_i < len(self.dim_lines):
                d = self.dim_lines[dim_i]
                d["x1"] = ox1 + actual_dx; d["y1"] = oy1 + actual_dy
                d["x2"] = ox2 + actual_dx; d["y2"] = oy2 + actual_dy
                if olx is not None:
                    d["label_x"] = olx + actual_dx
                if oly is not None:
                    d["label_y"] = oly + actual_dy

        app.generate_plot()

        if self.snap_guides:
            for guide in self.snap_guides:
                x1, y1, x2, y2, _ = guide
                app.ax.plot([x1, x2], [y1, y2],
                           color=AppConstants.SNAP_LINE_COLOR,
                           linestyle=AppConstants.SNAP_LINE_STYLE,
                           linewidth=AppConstants.SNAP_LINE_WIDTH,
                           alpha=AppConstants.SNAP_LINE_ALPHA, zorder=20)
            app.canvas.draw_idle()

    def on_release(self, event) -> None:
        """Handle mouse release — finalize drag, marquee, or selection."""
        app = self.app

        # Label/dim drag release
        if self.label_drag is not None:
            drag = self.label_drag
            was_drag = drag.get("dragged", False)
            idx = drag.get("idx")
            dtype = drag.get("type")
            is_dim_endpoint = dtype == "dim_endpoint"
            is_dim_label = dtype == "dim_label"
            is_dim_body = dtype == "dim_body"
            self.label_drag = None
            app.root.config(cursor="arrow")
            if was_drag:
                app.generate_plot()
                if not app.history_manager.is_restoring:
                    app._capture_current_state()
            else:
                if is_dim_endpoint or is_dim_label or is_dim_body:
                    self.update_selection_ui()
                    app.generate_plot()
                else:
                    if self.selected_label == idx:
                        self.selected_label = None
                    else:
                        self.selected_label = idx
                    self.selected.clear()
                    self.update_selection_ui()
                    app.generate_plot()
            return

        # Marquee finalization
        if self.marquee is not None:
            m = self.marquee
            self.marquee = None
            if m["rect_artist"] is not None:
                try:
                    m["rect_artist"].remove()
                except (ValueError, AttributeError):
                    pass
            mx, my = None, None
            if event.inaxes == app.ax and event.xdata is not None:
                mx, my = event.xdata, event.ydata
            if mx is not None:
                x0, y0 = m["start_x"], m["start_y"]
                sel_x_min = min(x0, mx); sel_x_max = max(x0, mx)
                sel_y_min = min(y0, my); sel_y_max = max(y0, my)
                if abs(mx - x0) > 0.3 or abs(my - y0) > 0.3:
                    additive = False
                    if hasattr(event, 'guiEvent') and event.guiEvent is not None:
                        state = event.guiEvent.state
                        additive = bool(state & 0x4) or bool(state & 0x10)
                    if not additive:
                        self.selected.clear()
                    if app._composite_bboxes is not None:
                        for i, bbox in enumerate(app._composite_bboxes):
                            if bbox == (0, 0, 0, 0):
                                continue
                            if (bbox[0] <= sel_x_max and bbox[2] >= sel_x_min and
                                    bbox[1] <= sel_y_max and bbox[3] >= sel_y_min):
                                self.selected.add(i)
                    self.update_selection_ui()
                    app.generate_plot()
                else:
                    if self.selected:
                        self.selected.clear()
                        self.update_selection_ui()
                        app.generate_plot()
            else:
                if self.selected:
                    self.selected.clear()
                    self.update_selection_ui()
                    app.generate_plot()
            return

        if self.drag_state is None:
            return

        drag = self.drag_state
        was_drag = drag.get("dragged", False)
        multi = drag.get("multi", False)
        idx = drag["idx"]

        self.drag_state = None
        self.snap_guides = []
        app.root.config(cursor="arrow")

        if was_drag:
            app.generate_plot()
            if not app.history_manager.is_restoring:
                app._capture_current_state()
        else:
            was_selected_before = drag.get("was_selected_before", False)
            if multi:
                pass
            else:
                if was_selected_before and self.selected == {idx}:
                    self.selected.clear()
                else:
                    self.selected = {idx}
            self.update_selection_ui()
            app.generate_plot()

    # ── Snap calculation ──────────────────────────────────────────────────────

    def calc_snap(self, drag_idx: int, drag_bbox: tuple,
                  all_bboxes: list[tuple]) -> tuple[float, float, list]:
        """Calculate snap offsets and guide lines for a dragged shape."""
        app = self.app
        threshold = AppConstants.SNAP_THRESHOLD
        anchor_threshold = threshold * 0.9
        snap_dx = 0.0; snap_dy = 0.0
        guides = []
        d_left, d_bottom, d_right, d_top = drag_bbox
        d_cx = (d_left + d_right) / 2; d_cy = (d_bottom + d_top) / 2
        best_x_dist = threshold; best_y_dist = threshold
        best_x_is_anchor = False; best_y_is_anchor = False
        has_anchors = bool(app._composite_snap_anchors)
        drag_anchors = []
        if has_anchors and drag_idx < len(app._composite_snap_anchors):
            drag_anchors = app._composite_snap_anchors[drag_idx]
        for i, bbox in enumerate(all_bboxes):
            if i == drag_idx or bbox == (0, 0, 0, 0):
                continue
            o_left, o_bottom, o_right, o_top = bbox
            o_cx = (o_left + o_right) / 2; o_cy = (o_bottom + o_top) / 2
            x_snaps = [
                (d_left, o_left), (d_left, o_right),
                (d_right, o_left), (d_right, o_right),
                (d_cx, o_cx),
                (d_left, o_cx), (d_right, o_cx),
                (d_cx, o_left), (d_cx, o_right),
            ]
            for d_edge, o_edge in x_snaps:
                dist = abs(d_edge - o_edge)
                if dist < best_x_dist and not (best_x_is_anchor and dist > best_x_dist - 0.3):
                    best_x_dist = dist; best_x_is_anchor = False
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
                    best_y_dist = dist; best_y_is_anchor = False
                    snap_dy = o_edge - d_edge
                    all_x = [d_left, d_right, o_left, o_right]
                    guides = [g for g in guides if g[4] != "y"]
                    guides.append((min(all_x) - 0.5, o_edge, max(all_x) + 0.5, o_edge, "y"))
            if has_anchors and i < len(app._composite_snap_anchors):
                other_anchors = app._composite_snap_anchors[i]
                if drag_anchors and other_anchors:
                    for dax, day in drag_anchors:
                        for oax, oay in other_anchors:
                            dist_x = abs(dax - oax)
                            if dist_x < anchor_threshold:
                                if dist_x < best_x_dist or (best_x_is_anchor and dist_x <= best_x_dist):
                                    best_x_dist = dist_x; best_x_is_anchor = True
                                    snap_dx = oax - dax
                                    all_y = [d_bottom, d_top, o_bottom, o_top]
                                    guides = [g for g in guides if g[4] != "x"]
                                    guides.append((oax, min(all_y) - 0.5, oax, max(all_y) + 0.5, "x"))
                            dist_y = abs(day - oay)
                            if dist_y < anchor_threshold:
                                if dist_y < best_y_dist or (best_y_is_anchor and dist_y <= best_y_dist):
                                    best_y_dist = dist_y; best_y_is_anchor = True
                                    snap_dy = oay - day
                                    all_x = [d_left, d_right, o_left, o_right]
                                    guides = [g for g in guides if g[4] != "y"]
                                    guides.append((min(all_x) - 0.5, oay, max(all_x) + 0.5, oay, "y"))
        return snap_dx, snap_dy, guides

    # ── Selection helpers ─────────────────────────────────────────────────────

    def update_selection_ui(self) -> None:
        """Update listbox selection to match self.selected."""
        app = self.app
        if app.composite_transfer is not None:
            dest = app.composite_transfer.dest_listbox
            dest.selection_clear(0, tk.END)
            for idx in self.selected:
                if idx < dest.size():
                    dest.selection_set(idx)
            if self.selected:
                last = max(self.selected)
                if last < dest.size():
                    dest.see(last)
        # Rebuild the Presets button panel to match the current selection
        if hasattr(app, '_rebuild_composite_preset_buttons'):
            app._rebuild_composite_preset_buttons()

    def get_group_center(self) -> tuple[float, float]:
        """Calculate the center of all selected shapes' bounding boxes."""
        app = self.app
        if not self.selected or app._composite_bboxes is None:
            return (0.0, 0.0)
        xs, ys = [], []
        for idx in self.selected:
            if idx < len(app._composite_bboxes):
                bbox = app._composite_bboxes[idx]
                if bbox != (0, 0, 0, 0):
                    xs.append((bbox[0] + bbox[2]) / 2)
                    ys.append((bbox[1] + bbox[3]) / 2)
        if not xs:
            return (0.0, 0.0)
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def get_group_bbox(self) -> tuple[float, float, float, float]:
        """Get the bounding box encompassing all selected shapes."""
        app = self.app
        if not self.selected or app._composite_bboxes is None:
            return (0, 0, 0, 0)
        x_mins, y_mins, x_maxs, y_maxs = [], [], [], []
        for idx in self.selected:
            if idx < len(app._composite_bboxes):
                bbox = app._composite_bboxes[idx]
                if bbox != (0, 0, 0, 0):
                    x_mins.append(bbox[0]); y_mins.append(bbox[1])
                    x_maxs.append(bbox[2]); y_maxs.append(bbox[3])
        if not x_mins:
            return (0, 0, 0, 0)
        pad = 0.5
        return (min(x_mins) - pad, min(y_mins) - pad, max(x_maxs) + pad, max(y_maxs) + pad)

    def _hit_test_dim_line(self, mx: float, my: float, app) -> "int | None":
        """Return the index of the dim line nearest to (mx, my), or None."""
        hit_radius = max((app.ax.get_xlim()[1] - app.ax.get_xlim()[0]) * 0.015, 0.2)
        # Check label bboxes first (easiest to click)
        if app._composite_dim_label_bboxes is not None:
            for idx in reversed(range(len(app._composite_dim_label_bboxes))):
                bbox = app._composite_dim_label_bboxes[idx]
                if bbox[0] <= mx <= bbox[2] and bbox[1] <= my <= bbox[3]:
                    return idx
        # Check endpoints
        if app._composite_dim_endpoints is not None:
            for idx, endpoints in enumerate(app._composite_dim_endpoints):
                for ep_key in ("p1", "p2"):
                    if ep_key not in endpoints:
                        continue
                    ex, ey = endpoints[ep_key]
                    if abs(mx - ex) < hit_radius and abs(my - ey) < hit_radius:
                        return idx
        # Check line body
        for idx in reversed(range(len(self.dim_lines))):
            dim = self.dim_lines[idx]
            x1, y1, x2, y2 = dim["x1"], dim["y1"], dim["x2"], dim["y2"]
            ddx, ddy = x2 - x1, y2 - y1
            seg_len_sq = ddx*ddx + ddy*ddy
            if seg_len_sq == 0:
                continue
            t = max(0.0, min(1.0, ((mx - x1)*ddx + (my - y1)*ddy) / seg_len_sq))
            dist = math.sqrt((mx - (x1 + t*ddx))**2 + (my - (y1 + t*ddy))**2)
            if dist < hit_radius and 0.05 < t < 0.95:
                return idx
        return None

    def all_shapes_selected(self) -> bool:
        """Check if every shape in the composite is selected."""
        app = self.app
        if app.composite_transfer is None:
            return False
        total = app.composite_transfer.dest_listbox.size()
        return total > 0 and len(self.selected) >= total

    def get_annotations_in_region(self, x_min: float, y_min: float,
                                   x_max: float, y_max: float,
                                   include_all: bool = False) -> tuple[list[int], list[int]]:
        """Return (label_indices, dim_line_indices) within the given region."""
        label_idxs = []
        for i, lbl in enumerate(self.labels):
            if include_all or (x_min <= lbl["x"] <= x_max and y_min <= lbl["y"] <= y_max):
                label_idxs.append(i)
        dim_idxs = []
        for i, dim in enumerate(self.dim_lines):
            if include_all:
                dim_idxs.append(i)
            else:
                p1_in = x_min <= dim["x1"] <= x_max and y_min <= dim["y1"] <= y_max
                p2_in = x_min <= dim["x2"] <= x_max and y_min <= dim["y2"] <= y_max
                if p1_in and p2_in:
                    dim_idxs.append(i)
        return label_idxs, dim_idxs

    # ── Annotation transform helpers ──────────────────────────────────────────

    def rotate_annotations(self, label_idxs: list[int], dim_idxs: list[int],
                            cx: float, cy: float, direction: int) -> None:
        """Rotate annotations around (cx, cy) by 90 degrees."""
        def rot(x, y):
            rx, ry = x - cx, y - cy
            if direction == 1:
                return cx + ry, cy - rx
            else:
                return cx - ry, cy + rx
        for i in label_idxs:
            lbl = self.labels[i]
            lbl["x"], lbl["y"] = rot(lbl["x"], lbl["y"])
        for i in dim_idxs:
            dim = self.dim_lines[i]
            dim["x1"], dim["y1"] = rot(dim["x1"], dim["y1"])
            dim["x2"], dim["y2"] = rot(dim["x2"], dim["y2"])
            if "label_x" in dim and "label_y" in dim:
                dim["label_x"], dim["label_y"] = rot(dim["label_x"], dim["label_y"])
            c = dim.get("constraint")
            if c == "height":
                dim["constraint"] = "width"
            elif c == "width":
                dim["constraint"] = "height"

    def flip_annotations_h(self, label_idxs: list[int], dim_idxs: list[int], cx: float) -> None:
        """Mirror annotations horizontally around x=cx."""
        for i in label_idxs:
            lbl = self.labels[i]; lbl["x"] = 2 * cx - lbl["x"]
        for i in dim_idxs:
            dim = self.dim_lines[i]
            dim["x1"] = 2 * cx - dim["x1"]; dim["x2"] = 2 * cx - dim["x2"]
            if "label_x" in dim:
                dim["label_x"] = 2 * cx - dim["label_x"]

    def flip_annotations_v(self, label_idxs: list[int], dim_idxs: list[int], cy: float) -> None:
        """Mirror annotations vertically around y=cy."""
        for i in label_idxs:
            lbl = self.labels[i]; lbl["y"] = 2 * cy - lbl["y"]
        for i in dim_idxs:
            dim = self.dim_lines[i]
            dim["y1"] = 2 * cy - dim["y1"]; dim["y2"] = 2 * cy - dim["y2"]
            if "label_y" in dim:
                dim["label_y"] = 2 * cy - dim["label_y"]

    # ── Patch translation ─────────────────────────────────────────────────────

    @staticmethod
    def translate_patch(patch, dx: float, dy: float) -> None:
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

    # ── Dim line placement mode ───────────────────────────────────────────────

    def start_dim_mode(self, mode: str) -> None:
        """Enter dimension line placement mode."""
        app = self.app
        text = app._composite_label_entry.get().strip() if app._composite_label_entry is not None else ""
        if not text:
            defaults = {"height": "h", "width": "w", "radius": "r", "free": "d"}
            text = defaults.get(mode, "d")
        self.dim_mode = mode
        self.dim_first_point = None
        if app._dim_status_label is not None:
            app._dim_status_label.config(text=f"Click first point for {mode} dimension line...")
        app.root.config(cursor="crosshair")

    def cancel_dim_mode(self) -> None:
        """Exit dimension line placement mode."""
        app = self.app
        self.dim_mode = None
        self.dim_first_point = None
        if app._dim_status_label is not None:
            app._dim_status_label.config(text="")
        if app._dim_preview_line:
            try:
                app._dim_preview_line.remove()
            except (ValueError, AttributeError):
                pass
            app._dim_preview_line = None
        app.root.config(cursor="arrow")

    def handle_dim_click(self, mx: float, my: float) -> bool:
        """Handle a click during dimension line placement. Returns True if handled."""
        app = self.app
        if self.dim_mode is None:
            return False
        mode = self.dim_mode
        if self.dim_first_point is None:
            self.dim_first_point = (mx, my)
            if app._dim_status_label is not None:
                app._dim_status_label.config(text=f"Click second point for {mode} dimension line...")
            return True
        else:
            x1, y1 = self.dim_first_point
            x2, y2 = mx, my
            if mode == "height":
                x2 = x1
            elif mode == "width":
                y2 = y1
            text = app._composite_label_entry.get().strip() if app._composite_label_entry is not None else ""
            if not text:
                defaults = {"height": "h", "width": "w", "radius": "r", "free": "d"}
                text = defaults.get(mode, "d")
            mid_x = (x1 + x2) / 2; mid_y = (y1 + y2) / 2
            # Auto-assign shape_owner when exactly one shape is selected.
            # This ensures manually drawn lines around a shape travel with it
            # (and are deleted with it) just like preset lines.
            auto_owner = next(iter(self.selected)) if len(self.selected) == 1 else None
            self.dim_lines.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "text": text,
                "constraint": mode if mode != "free" else None,
                "label_x": mid_x, "label_y": mid_y,
                "shape_owner": auto_owner,
            })
            if app._composite_label_entry is not None:
                app._composite_label_entry.delete(0, tk.END)
            self.selected_dim = len(self.dim_lines) - 1
            self.cancel_dim_mode()
            app.generate_plot()
            if not app.history_manager.is_restoring:
                app._capture_current_state()
            return True

    # ── Label management ──────────────────────────────────────────────────────

    def add_label(self) -> None:
        """Add a custom label to the composite canvas."""
        app = self.app
        if app._composite_label_entry is None:
            return
        text = app._composite_label_entry.get().strip()
        if not text:
            return
        xlim = app.ax.get_xlim(); ylim = app.ax.get_ylim()
        cx = (xlim[0] + xlim[1]) / 2; cy = (ylim[0] + ylim[1]) / 2
        self.labels.append({"text": text, "x": cx, "y": cy})
        app._composite_label_entry.delete(0, tk.END)
        self.selected_label = len(self.labels) - 1
        app.generate_plot()
        if not app.history_manager.is_restoring:
            app._capture_current_state()

    def remove_label(self) -> None:
        """Remove the selected composite label."""
        if self.selected_label is None or self.selected_label >= len(self.labels):
            return
        self.labels.pop(self.selected_label)
        self.selected_label = None
        self.app.generate_plot()
        if not self.app.history_manager.is_restoring:
            self.app._capture_current_state()

    def confirm_label(self) -> None:
        """Add or update depending on edit mode."""
        if self.edit_mode is not None:
            self.apply_edit()
        else:
            self.add_label()

    def remove_selected_annotation(self) -> None:
        """Remove the selected label or dimension line."""
        if self.selected_label is not None:
            self.remove_label()
        elif self.selected_dim is not None:
            if self.selected_dim < len(self.dim_lines):
                self.dim_lines.pop(self.selected_dim)
            self.selected_dim = None
            self.app.generate_plot()
            if not self.app.history_manager.is_restoring:
                self.app._capture_current_state()

    def enter_edit(self, edit_type: str, idx: int) -> None:
        """Enter edit mode for a label or dim line."""
        app = self.app
        if edit_type == "label" and idx < len(self.labels):
            existing = self.labels[idx]["text"]
        elif edit_type == "dim" and idx < len(self.dim_lines):
            existing = self.dim_lines[idx]["text"]
        else:
            return
        self.edit_mode = {"type": edit_type, "idx": idx}
        if app._composite_label_entry is not None:
            app._composite_label_entry.delete(0, tk.END)
            app._composite_label_entry.insert(0, existing)
            app._composite_label_entry.focus_set()
            app._composite_label_entry.select_range(0, tk.END)
        if app._composite_text_btn is not None:
            app._composite_text_btn.config(text="Update")
        if app._composite_cancel_btn is not None:
            app._composite_cancel_btn.pack(side=tk.LEFT, padx=1)

    def apply_edit(self) -> None:
        """Apply edited text to label or dim line."""
        app = self.app
        if self.edit_mode is None:
            return
        if app._composite_label_entry is None:
            return
        new_text = app._composite_label_entry.get().strip()
        if not new_text:
            self.cancel_edit()
            return
        edit_type = self.edit_mode["type"]
        idx = self.edit_mode["idx"]
        if edit_type == "label" and idx < len(self.labels):
            self.labels[idx]["text"] = new_text
        elif edit_type == "dim" and idx < len(self.dim_lines):
            self.dim_lines[idx]["text"] = new_text
        self.cancel_edit()
        app.generate_plot()
        if not app.history_manager.is_restoring:
            app._capture_current_state()

    def cancel_edit(self) -> None:
        """Exit edit mode."""
        app = self.app
        self.edit_mode = None
        if app._composite_label_entry is not None:
            app._composite_label_entry.delete(0, tk.END)
        if app._composite_text_btn is not None:
            app._composite_text_btn.config(text="+ Text")
        if app._composite_cancel_btn is not None:
            app._composite_cancel_btn.pack_forget()

    # ── Flip and rotate ───────────────────────────────────────────────────────

    def flip(self, axis: str) -> None:
        """Shared implementation for horizontal ('h') and vertical ('v') flip."""
        app = self.app
        if not self.selected:
            return
        if not app.history_manager.is_restoring:
            app._capture_current_state()

        pre_flip_bbox = self.get_group_bbox() if len(self.selected) > 1 else (0, 0, 0, 0)
        pre_flip_gcx, pre_flip_gcy = self.get_group_center() if len(self.selected) > 1 else (0, 0)
        pivot = pre_flip_gcx if axis == 'h' else pre_flip_gcy

        selected = app.composite_transfer.get_selected_shapes() if app.composite_transfer else []

        if len(self.selected) > 1:
            for idx in self.selected:
                pos = self.positions.get(idx, (0.0, 0.0))
                if idx < len(app._composite_bboxes):
                    bbox = app._composite_bboxes[idx]
                    if bbox != (0, 0, 0, 0):
                        if axis == 'h':
                            shape_c = (bbox[0] + bbox[2]) / 2
                            new_c = 2 * pivot - shape_c
                            self.positions[idx] = (pos[0] + (new_c - shape_c), pos[1])
                        else:
                            shape_c = (bbox[1] + bbox[3]) / 2
                            new_c = 2 * pivot - shape_c
                            self.positions[idx] = (pos[0], pos[1] + (new_c - shape_c))

                t = self.transforms.setdefault(idx, {"flip_h": False, "flip_v": False, "base_side": 0})
                num_sides = 4
                shape_name_for_flip = ""
                config = ShapeConfig()
                if idx < len(selected):
                    shape_name_for_flip = selected[idx]
                    config = ShapeConfigProvider.get(shape_name_for_flip)
                    num_sides = config.num_sides if config.num_sides > 0 else 4

                effective_side = t["base_side"]
                if num_sides >= 4 and config.uses_base_side_flip:
                    if t["flip_v"]:
                        effective_side = (effective_side + 2) % num_sides
                        t["flip_v"] = False
                    if t["flip_h"]:
                        if effective_side in (1, 3):
                            effective_side = 4 - effective_side
                        t["flip_h"] = False
                    if axis == 'h':
                        if effective_side in (1, 3):
                            effective_side = 4 - effective_side
                    else:
                        if effective_side in (0, 2):
                            effective_side = 2 - effective_side
                    t["base_side"] = effective_side
                else:
                    if axis == 'h':
                        t["flip_h"] = not t["flip_h"]
                    else:
                        t["flip_v"] = not t["flip_v"]
        else:
            for idx in self.selected:
                t = self.transforms.setdefault(idx, {"flip_h": False, "flip_v": False, "base_side": 0})
                if axis == 'h':
                    t["flip_h"] = not t["flip_h"]
                else:
                    t["flip_v"] = not t["flip_v"]
                if idx < len(app._composite_bboxes):
                    bbox = app._composite_bboxes[idx]
                    if bbox != (0, 0, 0, 0):
                        if axis == 'h':
                            shape_c = (bbox[0] + bbox[2]) / 2
                            lbl_idxs, dim_idxs = self.get_annotations_in_region(*bbox)
                            if lbl_idxs or dim_idxs:
                                self.flip_annotations_h(lbl_idxs, dim_idxs, shape_c)
                        else:
                            shape_c = (bbox[1] + bbox[3]) / 2
                            lbl_idxs, dim_idxs = self.get_annotations_in_region(*bbox)
                            if lbl_idxs or dim_idxs:
                                self.flip_annotations_v(lbl_idxs, dim_idxs, shape_c)

        if len(self.selected) > 1 and pre_flip_bbox != (0, 0, 0, 0):
            all_sel = self.all_shapes_selected()
            lbl_idxs, dim_idxs = self.get_annotations_in_region(*pre_flip_bbox, include_all=all_sel)
            if lbl_idxs or dim_idxs:
                if axis == 'h':
                    self.flip_annotations_h(lbl_idxs, dim_idxs, pre_flip_gcx)
                else:
                    self.flip_annotations_v(lbl_idxs, dim_idxs, pre_flip_gcy)

        app.generate_plot()
        if not app.history_manager.is_restoring:
            app._capture_current_state(force=True)

    def flip_h(self) -> None:
        """Flip selected shapes horizontally."""
        self.flip('h')

    def flip_v(self) -> None:
        """Flip selected shapes vertically."""
        self.flip('v')

    def rotate(self, direction: int) -> None:
        """Rotate selected shapes as a group around their collective center."""
        app = self.app
        if not self.selected:
            return
        if app.composite_transfer is None:
            return
        selected = app.composite_transfer.get_selected_shapes()
        if not app.history_manager.is_restoring:
            app._capture_current_state()

        pre_rot_center = None
        pre_rot_bbox = (0, 0, 0, 0)
        if len(self.selected) > 1:
            pre_rot_bbox = self.get_group_bbox()
            centers_pre = []
            for idx in self.selected:
                if idx < len(app._composite_bboxes):
                    bbox = app._composite_bboxes[idx]
                    if bbox != (0, 0, 0, 0):
                        centers_pre.append(((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2))
            if centers_pre:
                pre_rot_center = (sum(o[0] for o in centers_pre) / len(centers_pre),
                                  sum(o[1] for o in centers_pre) / len(centers_pre))

        if len(self.selected) > 1:
            shape_centers = {}
            for idx in self.selected:
                if idx < len(app._composite_bboxes):
                    bbox = app._composite_bboxes[idx]
                    if bbox != (0, 0, 0, 0):
                        shape_centers[idx] = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            if not shape_centers:
                return
            gcx = sum(o[0] for o in shape_centers.values()) / len(shape_centers)
            gcy = sum(o[1] for o in shape_centers.values()) / len(shape_centers)
            target_centers = {}
            for idx in self.selected:
                if idx in shape_centers:
                    ox, oy = shape_centers[idx]
                    rx = ox - gcx; ry = oy - gcy
                    if direction == 1:
                        new_rx, new_ry = ry, -rx
                    else:
                        new_rx, new_ry = -ry, rx
                    target_centers[idx] = (gcx + new_rx, gcy + new_ry)
                if idx < len(selected):
                    shape_name = selected[idx]
                    config = ShapeConfigProvider.get(shape_name)
                    num_sides = config.num_sides if config.num_sides > 0 else 4
                    t = self.transforms.setdefault(idx, {"flip_h": False, "flip_v": False, "base_side": 0})
                    t["base_side"] = (t["base_side"] + direction) % num_sides
            app.generate_plot()
            for idx, (target_cx, target_cy) in target_centers.items():
                actual_cx, actual_cy = None, None
                if idx < len(app._composite_bboxes):
                    new_bbox = app._composite_bboxes[idx]
                    if new_bbox != (0, 0, 0, 0):
                        actual_cx = (new_bbox[0] + new_bbox[2]) / 2
                        actual_cy = (new_bbox[1] + new_bbox[3]) / 2
                if actual_cx is not None:
                    correction_dx = target_cx - actual_cx
                    correction_dy = target_cy - actual_cy
                    pos = self.positions.get(idx, (0.0, 0.0))
                    self.positions[idx] = (pos[0] + correction_dx, pos[1] + correction_dy)
        else:
            for idx in self.selected:
                if idx >= len(selected):
                    continue
                shape_name = selected[idx]
                config = ShapeConfigProvider.get(shape_name)
                num_sides = config.num_sides if config.num_sides > 0 else 4
                t = self.transforms.setdefault(idx, {"flip_h": False, "flip_v": False, "base_side": 0})
                t["base_side"] = (t["base_side"] + direction) % num_sides
                if idx < len(app._composite_bboxes):
                    bbox = app._composite_bboxes[idx]
                    if bbox != (0, 0, 0, 0):
                        shape_cx = (bbox[0] + bbox[2]) / 2
                        shape_cy = (bbox[1] + bbox[3]) / 2
                        lbl_idxs, dim_idxs = self.get_annotations_in_region(*bbox)
                        if lbl_idxs or dim_idxs:
                            self.rotate_annotations(lbl_idxs, dim_idxs, shape_cx, shape_cy, direction)

        if len(self.selected) > 1 and pre_rot_center is not None:
            rot_cx, rot_cy = pre_rot_center
            all_sel = self.all_shapes_selected()
            lbl_idxs, dim_idxs = self.get_annotations_in_region(*pre_rot_bbox, include_all=all_sel)
            if lbl_idxs or dim_idxs:
                self.rotate_annotations(lbl_idxs, dim_idxs, rot_cx, rot_cy, direction)

        app._composite_view_limits = None
        app.generate_plot()
        if not app.history_manager.is_restoring:
            app._capture_current_state(force=True)

    def on_change(self, operation: tuple = None) -> None:
        """Called when the composite transfer list selection changes (after the change).

        Args:
            operation: Explicit operation tuple from the transfer list:
                ("add", new_index) — a shape was added at the given index
                ("remove", removed_index) — a shape was removed at the given index
                ("swap", idx_a, idx_b) — two shapes were swapped (reorder)
                None — bulk restore (e.g. undo/redo via set_selected_shapes)
        """
        app = self.app
        if app.composite_transfer is not None:
            new_shapes = app.composite_transfer.get_selected_shapes()
            n = len(new_shapes)

            if operation is not None:
                op_type = operation[0]

                if op_type == "swap":
                    swap_a, swap_b = operation[1], operation[2]
                    pos_a = self.positions.get(swap_a, (0.0, 0.0))
                    pos_b = self.positions.get(swap_b, (0.0, 0.0))
                    self.positions[swap_a] = pos_b
                    self.positions[swap_b] = pos_a
                    tfm_a = self.transforms.get(swap_a, {"flip_h": False, "flip_v": False, "base_side": 0})
                    tfm_b = self.transforms.get(swap_b, {"flip_h": False, "flip_v": False, "base_side": 0})
                    self.transforms[swap_a] = tfm_b
                    self.transforms[swap_b] = tfm_a
                    # Keep shape_owner in sync when shapes are reordered
                    for dim in self.dim_lines:
                        owner = dim.get("shape_owner")
                        if owner == swap_a:
                            dim["shape_owner"] = swap_b
                        elif owner == swap_b:
                            dim["shape_owner"] = swap_a

                elif op_type == "remove":
                    removed_idx = operation[1]
                    new_positions = {}
                    new_transforms = {}
                    for k, v in self.positions.items():
                        if k < removed_idx:
                            new_positions[k] = v
                        elif k > removed_idx:
                            new_positions[k - 1] = v
                    for k, v in self.transforms.items():
                        if k < removed_idx:
                            new_transforms[k] = v
                        elif k > removed_idx:
                            new_transforms[k - 1] = v
                    self.positions = new_positions
                    self.transforms = new_transforms
                    # Remove owned dim lines; reindex survivors
                    new_dim_lines = []
                    for dim in self.dim_lines:
                        owner = dim.get("shape_owner")
                        if owner == removed_idx:
                            continue
                        if owner is not None and owner > removed_idx:
                            dim = dict(dim)
                            dim["shape_owner"] = owner - 1
                        new_dim_lines.append(dim)
                    self.dim_lines = new_dim_lines
                    if self.selected_dim is not None and self.selected_dim >= len(self.dim_lines):
                        self.selected_dim = None

                elif op_type == "add":
                    new_idx = operation[1]
                    # Explicitly clear any stale position/transform for the new index
                    # so Phase 2 grid assignment always gives it a fresh cell.
                    # Without this, if a shape was previously at this index then
                    # removed (and indices shifted), the old absolute position can
                    # survive the shift and cause the new shape to appear on top of
                    # an existing shape rather than in an empty grid cell.
                    self.positions.pop(new_idx, None)
                    self.transforms.pop(new_idx, None)

            # Clean up stale entries
            self.positions = {k: v for k, v in self.positions.items() if k < n}
            self.transforms = {k: v for k, v in self.transforms.items() if k < n}

            # Auto-select newly added shape so presets and toolbar reflect it immediately.
            # For all other operations (remove, swap, bulk restore) clear selection as before.
            if operation is not None and operation[0] == "add":
                self.selected = {operation[1]}
            else:
                self.selected.clear()
            self.update_selection_ui()
            app._composite_view_limits = None  # Force view recalculation

        app.generate_plot()
        if not app.history_manager.is_restoring:
            app._capture_current_state()

    def on_delete(self, event=None) -> None:
        """Delete selected composite shapes or label."""
        app = self.app
        if not app._is_composite_shape():
            return

        # Prevent deletion while typing in the label entry
        if isinstance(app.root.focus_get(), tk.Entry):
            return

        if self.selected_label is not None or self.selected_dim is not None:
            self.remove_selected_annotation()
            return

        if not self.selected:
            return
        if app.composite_transfer is None:
            return

        # Remove shapes from highest index to lowest (to avoid shifting issues)
        for idx in sorted(self.selected, reverse=True):
            if idx < app.composite_transfer.dest_listbox.size():
                app.composite_transfer.dest_listbox.delete(idx)

                # Shift positions and transforms down
                new_positions = {}
                new_transforms = {}
                for k, v in self.positions.items():
                    if k < idx:
                        new_positions[k] = v
                    elif k > idx:
                        new_positions[k - 1] = v
                for k, v in self.transforms.items():
                    if k < idx:
                        new_transforms[k] = v
                    elif k > idx:
                        new_transforms[k - 1] = v
                self.positions = new_positions
                self.transforms = new_transforms

                # Remove dim lines owned by deleted shape; reindex survivors
                new_dim_lines = []
                for dim in self.dim_lines:
                    owner = dim.get("shape_owner")
                    if owner == idx:
                        continue
                    if owner is not None and owner > idx:
                        dim = dict(dim)
                        dim["shape_owner"] = owner - 1
                    new_dim_lines.append(dim)
                self.dim_lines = new_dim_lines
                if self.selected_dim is not None and self.selected_dim >= len(self.dim_lines):
                    self.selected_dim = None

        # Force grid re-layout so next added shape lands in its own cell
        self.positions.clear()
        app._composite_view_limits = None

        self.selected.clear()

        self.update_selection_ui()
        app.generate_plot()
        if not app.history_manager.is_restoring:
            app._capture_current_state()

    def reset(self) -> None:
        """Clear all composite annotation and selection state."""
        self.positions = {}
        self.transforms = {}
        self.selected = set()
        self.labels = []
        self.label_drag = None
        self.selected_label = None
        self.dim_lines = []
        self.selected_dim = None
        self.dim_mode = None
        self.dim_first_point = None
        self.edit_mode = None
        self.drag_state = None
        self.marquee = None
        self.snap_guides = []