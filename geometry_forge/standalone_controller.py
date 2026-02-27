from __future__ import annotations

import math
import logging
import tkinter as tk

from .models import (
    Point, StandaloneDimLine, AppConstants, DrawingContext,
)
from .drawing import DrawingUtilities

logger = logging.getLogger(__name__)


class StandaloneAnnotationController:
    """Manages freeform labels and dimension lines on the standalone (non-composite) canvas.

    Owns all standalone annotation state and the four mouse-event handlers.
    Calls back into GeometryApp via self.app for shared resources (canvas,
    ax, history_manager, generate_plot, label_manager, etc.).

    Instantiated once in GeometryApp.__init__ and stored as self.standalone_ctrl.
    GeometryApp retains thin delegate methods so all existing call sites work
    without modification.
    """

    def __init__(self, app: "GeometryApp") -> None:
        self.app = app
        # Annotation data
        self.labels: list[dict] = []               # [{"text": str, "x": float, "y": float}]
        self.selected_label: int | None = None
        self.label_bboxes: list[tuple] = []
        self.dim_lines: list[StandaloneDimLine] = []
        self.selected_dim: int | None = None
        self.dim_endpoints: list[dict] = []
        self.dim_label_bboxes: list[tuple] = []
        self.dim_mode: bool = False
        self.dim_first_point: tuple | None = None
        self.dim_preview_line = None
        self.edit_mode: dict | None = None      # {"type": "label"/"dim"/"builtin", "idx": int}
        # Shared drag/hover state (also used by composite path — kept on app)
        # We access app._label_drag_state, app._label_hover_active directly.
        # Builtin selection state
        self.builtin_selected: str | None = None
        self.builtin_dim_endpoints: list[dict] = []  # [{key, p1, p2}] populated per render

    # ── Event connection ──────────────────────────────────────────────────────

    def connect(self) -> None:
        """Connect mouse events for standalone label dragging."""
        self.disconnect()
        self.app._standalone_event_ids = [
            self.app.canvas.mpl_connect('button_press_event', self.on_press),
            self.app.canvas.mpl_connect('motion_notify_event', self.on_motion),
            self.app.canvas.mpl_connect('button_release_event', self.on_release),
            self.app.canvas.mpl_connect('motion_notify_event', self.on_hover),
        ]

    def disconnect(self) -> None:
        """Disconnect standalone label drag events."""
        for eid in self.app._standalone_event_ids:
            try:
                self.app.canvas.mpl_disconnect(eid)
            except Exception:
                pass
        self.app._standalone_event_ids = []
        self.app._label_drag_state = None

    # ── Geometry helpers ──────────────────────────────────────────────────────

    def pixels_to_data_pad(self, px: float) -> tuple[float, float]:
        """Convert a pixel size to data-unit padding (x, y) based on current axis state."""
        app = self.app
        fig_w = app.fig.get_figwidth() * app.fig.dpi
        fig_h = app.fig.get_figheight() * app.fig.dpi
        ax_pos = app.ax.get_position()
        ax_px_w = fig_w * ax_pos.width
        ax_px_h = fig_h * ax_pos.height
        xlim = app.ax.get_xlim()
        ylim = app.ax.get_ylim()
        data_w = xlim[1] - xlim[0]
        data_h = ylim[1] - ylim[0]
        pad_x = px * data_w / ax_px_w if ax_px_w > 0 else 0.1
        pad_y = px * data_h / ax_px_h if ax_px_h > 0 else 0.1
        return pad_x, pad_y

    def find_nearest_label(self, x: float, y: float, threshold: float = 0.8) -> str | None:
        """Find the nearest built-in label key to the given data coordinates."""
        app = self.app
        if not app.label_manager.auto_positions:
            return None
        best_key = None
        best_dist = threshold
        xlim = app.ax.get_xlim()
        ylim = app.ax.get_ylim()
        pad_x, pad_y = self.pixels_to_data_pad(3)
        scale = max(pad_x, pad_y)
        if scale > 0:
            best_dist = scale
        for key, pos_data in app.label_manager.auto_positions.items():
            lx, ly = pos_data[0], pos_data[1]
            if key in app.label_manager.custom_positions:
                lx, ly = app.label_manager.custom_positions[key]
            text, show = app.label_manager.get_entry_values(key)
            if not (text and show):
                continue
            dist = math.sqrt((x - lx)**2 + (y - ly)**2)
            if dist < best_dist:
                best_dist = dist
                best_key = key
        return best_key

    # ── Mouse event handlers ──────────────────────────────────────────────────

    def on_hover(self, event) -> None:
        """Change cursor when hovering over a draggable label or dim line."""
        app = self.app
        if app._label_drag_state is not None:
            return
        if self.dim_mode:
            return
        if app._is_composite_shape():
            if getattr(app, '_label_hover_active', False):
                app.root.config(cursor="arrow")
                app._label_hover_active = False
            return
        if not app.shape_var.get():
            if getattr(app, '_label_hover_active', False):
                app.root.config(cursor="arrow")
                app._label_hover_active = False
            return
        if event.inaxes != app.ax:
            if getattr(app, '_label_hover_active', False):
                app.root.config(cursor="arrow")
                app._label_hover_active = False
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            if getattr(app, '_label_hover_active', False):
                app.root.config(cursor="arrow")
                app._label_hover_active = False
            return
        hit = False
        if self.find_nearest_label(x, y):
            hit = True
        if not hit:
            for bbox in self.label_bboxes:
                if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                    hit = True
                    break
        if not hit:
            for bbox in self.dim_label_bboxes:
                if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                    hit = True
                    break
        if not hit:
            hit_radius = max((app.ax.get_xlim()[1] - app.ax.get_xlim()[0]) * 0.015, 0.2)
            for ep in self.dim_endpoints:
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
        if not hit:
            hit_radius = max((app.ax.get_xlim()[1] - app.ax.get_xlim()[0]) * 0.015, 0.2)
            for ep in self.builtin_dim_endpoints:
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
            if not getattr(app, '_label_hover_active', False):
                app.root.config(cursor="fleur")
                app._label_hover_active = True
        else:
            if getattr(app, '_label_hover_active', False):
                app.root.config(cursor="arrow")
                app._label_hover_active = False

    def on_press(self, event) -> None:
        """Handle mouse press — start label drag if a label is hit."""
        app = self.app
        if event.inaxes != app.ax or event.button != 1:
            return
        if app._is_composite_shape():
            return
        if not app.shape_var.get():
            return
        if self.dim_mode:
            x, y = event.xdata, event.ydata
            if x is None or y is None:
                return
            if self.dim_first_point is None:
                self.dim_first_point = (x, y)
            else:
                x1, y1 = self.dim_first_point
                text = app._standalone_label_entry.get().strip() if app._standalone_label_entry is not None else "f"
                if not text:
                    text = "f"
                mid_x = (x1 + x) / 2
                mid_y = (y1 + y) / 2
                if not app.history_manager.is_restoring:
                    app._capture_current_state()
                self.dim_lines.append({
                    "x1": x1, "y1": y1, "x2": x, "y2": y,
                    "text": text,
                    "preset_key": None,
                    "user_dragged": True,
                    "label_x": mid_x,
                    "label_y": mid_y,
                })
                self.selected_dim = len(self.dim_lines) - 1
                self.cancel_dim_mode()
                if not app.history_manager.is_restoring:
                    app._capture_current_state()
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        if getattr(event, 'dblclick', False):
            hit_key = self.find_nearest_label(x, y)
            if hit_key and hit_key in AppConstants.BUILTIN_LABEL_KEYS:
                t, s = app.label_manager.get_entry_values(hit_key)
                if t and s:
                    app._builtin_edit_key = hit_key
                    self.enter_edit("builtin", 0)
                    return
            for idx in reversed(range(len(self.label_bboxes))):
                bbox = self.label_bboxes[idx]
                if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                    self.enter_edit("label", idx)
                    return
            for idx in reversed(range(len(self.dim_label_bboxes))):
                bbox = self.dim_label_bboxes[idx]
                if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                    self.enter_edit("dim", idx)
                    return
        for idx in reversed(range(len(self.dim_label_bboxes))):
            bbox = self.dim_label_bboxes[idx]
            if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                dim = self.dim_lines[idx]
                self.builtin_selected = None
                app._label_drag_state = {
                    "type": "standalone_dim_label",
                    "idx": idx,
                    "started": False,
                    "start_x": x,
                    "start_y": y,
                    "orig_x": dim.get("label_x", (dim["x1"] + dim["x2"]) / 2),
                    "orig_y": dim.get("label_y", (dim["y1"] + dim["y2"]) / 2),
                }
                return
        for idx in reversed(range(len(self.dim_endpoints))):
            ep = self.dim_endpoints[idx]
            for ep_name, ep_pos in [("p1", ep["p1"]), ("p2", ep["p2"])]:
                if math.sqrt((x - ep_pos[0])**2 + (y - ep_pos[1])**2) < max(
                        (app.ax.get_xlim()[1] - app.ax.get_xlim()[0]) * 0.02, 0.3):
                    self.selected_dim = idx
                    self.selected_label = None
                    self.builtin_selected = None
                    app.generate_plot()
                    dim = self.dim_lines[idx]
                    app._label_drag_state = {
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
        for idx in reversed(range(len(self.dim_endpoints))):
            ep = self.dim_endpoints[idx]
            p1, p2 = ep["p1"], ep["p2"]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq > 0:
                t = max(0, min(1, ((x - p1[0]) * dx + (y - p1[1]) * dy) / seg_len_sq))
                proj_x = p1[0] + t * dx
                proj_y = p1[1] + t * dy
                dist = math.sqrt((x - proj_x)**2 + (y - proj_y)**2)
                hit_radius = max((app.ax.get_xlim()[1] - app.ax.get_xlim()[0]) * 0.015, 0.2)
                if dist < hit_radius and 0.05 < t < 0.95:
                    self.selected_dim = idx
                    self.selected_label = None
                    self.builtin_selected = None
                    app.generate_plot()
                    dim = self.dim_lines[idx]
                    app._label_drag_state = {
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
        hit_radius = max((app.ax.get_xlim()[1] - app.ax.get_xlim()[0]) * 0.015, 0.2)
        for ep in self.builtin_dim_endpoints:
            p1, p2 = ep["p1"], ep["p2"]
            dx_seg, dy_seg = p2[0] - p1[0], p2[1] - p1[1]
            seg_len_sq = dx_seg * dx_seg + dy_seg * dy_seg
            if seg_len_sq > 0:
                t = max(0.0, min(1.0, ((x - p1[0]) * dx_seg + (y - p1[1]) * dy_seg) / seg_len_sq))
                proj_x = p1[0] + t * dx_seg
                proj_y = p1[1] + t * dy_seg
                dist = math.sqrt((x - proj_x) ** 2 + (y - proj_y) ** 2)
                if dist < hit_radius:
                    self.builtin_selected = ep["key"]
                    self.selected_dim = None
                    self.selected_label = None
                    app.generate_plot()
                    cur_off = app.label_manager.custom_dim_offsets.get(ep["key"], (0.0, 0.0))
                    app._label_drag_state = {
                        "type": "builtin_dim_body",
                        "key": ep["key"],
                        "started": False,
                        "start_x": x,
                        "start_y": y,
                        "orig_dx": cur_off[0],
                        "orig_dy": cur_off[1],
                    }
                    return
        for idx in reversed(range(len(self.label_bboxes))):
            bbox = self.label_bboxes[idx]
            if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                app._label_drag_state = {
                    "type": "standalone_label",
                    "idx": idx,
                    "started": False,
                    "start_x": x,
                    "start_y": y,
                    "orig_x": self.labels[idx]["x"],
                    "orig_y": self.labels[idx]["y"],
                }
                return
        key = self.find_nearest_label(x, y)
        if key:
            if key in AppConstants.BUILTIN_LABEL_KEYS:
                if self.builtin_selected == key:
                    self.builtin_selected = None
                else:
                    self.builtin_selected = key
                self.selected_dim = None
                self.selected_label = None
                app.generate_plot()
            else:
                self.builtin_selected = None
            app._label_drag_state = {"type": "auto_label", "key": key, "started": False, "start_x": x, "start_y": y}
            return
        if self.edit_mode is not None:
            self.cancel_edit()
        if self.selected_label is not None or self.selected_dim is not None or self.builtin_selected is not None:
            self.selected_label = None
            self.selected_dim = None
            self.builtin_selected = None
            app.generate_plot()
        app._label_drag_state = {
            "type": "pan_shape",
            "started": False,
            "start_x": x, "start_y": y,
            "start_px": event.x, "start_py": event.y,
            "orig_pan_x": app._shape_pan_offset[0],
            "orig_pan_y": app._shape_pan_offset[1],
        }

    def on_motion(self, event) -> None:
        """Handle mouse drag — reposition label or pan canvas."""
        app = self.app
        if self.dim_mode and self.dim_first_point is not None:
            if event.inaxes == app.ax and event.xdata is not None:
                x1, y1 = self.dim_first_point
                x2, y2 = event.xdata, event.ydata
                if self.dim_preview_line:
                    try:
                        self.dim_preview_line.remove()
                    except (ValueError, AttributeError):
                        pass
                self.dim_preview_line = app.ax.plot(
                    [x1, x2], [y1, y2],
                    color="#4488ff", linestyle="--", linewidth=1.0, alpha=0.6, zorder=20
                )[0]
                app.canvas.draw_idle()
            return
        if app._label_drag_state is None:
            return
        if event.inaxes != app.ax:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        if not app._label_drag_state["started"]:
            dx = abs(x - app._label_drag_state["start_x"])
            dy = abs(y - app._label_drag_state["start_y"])
            xlim = app.ax.get_xlim()
            ylim = app.ax.get_ylim()
            min_move = max(xlim[1] - xlim[0], ylim[1] - ylim[0]) * 0.01
            if dx < min_move and dy < min_move:
                return
            app._label_drag_state["started"] = True
            if not app.history_manager.is_restoring:
                app._capture_current_state()
            app.root.config(cursor="fleur")
        drag_type = app._label_drag_state.get("type", "auto_label")
        if drag_type == "pan_shape":
            dpx = event.x - app._label_drag_state["start_px"]
            dpy = event.y - app._label_drag_state["start_py"]
            xlim = app.ax.get_xlim()
            ylim = app.ax.get_ylim()
            ax_bbox = app.ax.get_window_extent()
            ax_w_px = ax_bbox.width  if ax_bbox.width  > 1 else 1
            ax_h_px = ax_bbox.height if ax_bbox.height > 1 else 1
            data_per_px_x = (xlim[1] - xlim[0]) / ax_w_px
            data_per_px_y = (ylim[1] - ylim[0]) / ax_h_px
            app._shape_pan_offset = (
                app._label_drag_state["orig_pan_x"] - dpx * data_per_px_x,
                app._label_drag_state["orig_pan_y"] - dpy * data_per_px_y,
            )
            app._apply_view_scale_only()
            return
        if drag_type == "standalone_label":
            idx = app._label_drag_state["idx"]
            dx = x - app._label_drag_state["start_x"]
            dy = y - app._label_drag_state["start_y"]
            if idx < len(self.labels):
                self.labels[idx]["x"] = app._label_drag_state["orig_x"] + dx
                self.labels[idx]["y"] = app._label_drag_state["orig_y"] + dy
                self.selected_label = idx
                app.generate_plot()
        elif drag_type == "standalone_dim_endpoint":
            idx = app._label_drag_state["idx"]
            dx = x - app._label_drag_state["start_x"]
            dy = y - app._label_drag_state["start_y"]
            if idx < len(self.dim_lines):
                dim = self.dim_lines[idx]
                new_x = app._label_drag_state["orig_x"] + dx
                new_y = app._label_drag_state["orig_y"] + dy
                if dim.get("constraint") == "height":
                    new_x = app._label_drag_state["orig_x"]
                elif dim.get("constraint") == "width":
                    new_y = app._label_drag_state["orig_y"]
                ep = app._label_drag_state["endpoint"]
                if ep == "p1":
                    dim["x1"], dim["y1"] = new_x, new_y
                elif ep == "p2":
                    dim["x2"], dim["y2"] = new_x, new_y
                dim["user_dragged"] = True
                self.selected_dim = idx
                app.generate_plot()
        elif drag_type == "standalone_dim_label":
            idx = app._label_drag_state["idx"]
            dx = x - app._label_drag_state["start_x"]
            dy = y - app._label_drag_state["start_y"]
            if idx < len(self.dim_lines):
                self.dim_lines[idx]["label_x"] = app._label_drag_state["orig_x"] + dx
                self.dim_lines[idx]["label_y"] = app._label_drag_state["orig_y"] + dy
                self.selected_dim = idx
                app.generate_plot()
        elif drag_type == "standalone_dim_body":
            idx = app._label_drag_state["idx"]
            dx = x - app._label_drag_state["start_x"]
            dy = y - app._label_drag_state["start_y"]
            if idx < len(self.dim_lines):
                dim = self.dim_lines[idx]
                dim["x1"] = app._label_drag_state["orig_x1"] + dx
                dim["y1"] = app._label_drag_state["orig_y1"] + dy
                dim["x2"] = app._label_drag_state["orig_x2"] + dx
                dim["y2"] = app._label_drag_state["orig_y2"] + dy
                dim["label_x"] = app._label_drag_state["orig_lx"] + dx
                dim["label_y"] = app._label_drag_state["orig_ly"] + dy
                dim["user_dragged"] = True
                self.selected_dim = idx
                app.generate_plot()
        elif drag_type == "builtin_dim_body":
            key = app._label_drag_state["key"]
            dx = x - app._label_drag_state["start_x"]
            dy = y - app._label_drag_state["start_y"]
            new_dx = app._label_drag_state["orig_dx"] + dx
            new_dy = app._label_drag_state["orig_dy"] + dy
            app.label_manager.custom_dim_offsets[key] = (new_dx, new_dy)
            app.generate_plot()
        else:
            key = app._label_drag_state["key"]
            app.label_manager.set_custom_position(key, x, y)
            app.generate_plot()
            text, show = app.label_manager.get_entry_values(key)
            if text and show:
                font_size = app.font_size_var.get() if app.font_size_var is not None else 12
                app.ax.text(x, y, text,
                            fontsize=font_size, color="#0066cc", fontweight="bold",
                            fontfamily=getattr(app, 'font_family', AppConstants.DEFAULT_FONT_FAMILY),
                            ha="center", va="center", zorder=AppConstants.LABEL_ZORDER + 2,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                     edgecolor="#0066cc", alpha=0.9))
                app.plot_controller.refresh()

    def on_release(self, event) -> None:
        """Handle mouse release — finalize label position or toggle selection."""
        app = self.app
        if app._label_drag_state is None:
            return
        drag_type = app._label_drag_state.get("type", "auto_label")
        if drag_type == "pan_shape":
            app._label_drag_state = None
            return
        was_dragged = app._label_drag_state.get("started", False)
        if drag_type == "standalone_label":
            idx = app._label_drag_state.get("idx")
            if was_dragged:
                app.generate_plot()
                if not app.history_manager.is_restoring:
                    app._capture_current_state()
            else:
                if self.selected_label == idx:
                    self.selected_label = None
                else:
                    self.selected_label = idx
                self.selected_dim = None
                app.generate_plot()
        elif drag_type in ("standalone_dim_endpoint", "standalone_dim_label", "standalone_dim_body"):
            idx = app._label_drag_state.get("idx")
            if was_dragged:
                app.generate_plot()
                if not app.history_manager.is_restoring:
                    app._capture_current_state()
            else:
                if self.selected_dim == idx:
                    self.selected_dim = None
                else:
                    self.selected_dim = idx
                self.selected_label = None
                app.generate_plot()
        elif drag_type == "builtin_dim_body":
            if was_dragged:
                app.generate_plot()
                if not app.history_manager.is_restoring:
                    app._capture_current_state()
        else:
            if was_dragged:
                app.generate_plot()
                if not app.history_manager.is_restoring:
                    app._capture_current_state()
        app._label_drag_state = None
        app.root.config(cursor="arrow")
        app._label_hover_active = False

    # ── Dim mode ──────────────────────────────────────────────────────────────

    def cancel_dim_mode(self) -> None:
        """Exit freeform dimension line draw mode."""
        app = self.app
        self.dim_mode = False
        self.dim_first_point = None
        if self.dim_preview_line:
            try:
                self.dim_preview_line.remove()
            except (ValueError, AttributeError):
                pass
            self.dim_preview_line = None
        app.root.config(cursor="arrow")
        if app._standalone_cancel_dim_btn is not None:
            app._standalone_cancel_dim_btn.grid_remove()
            app._standalone_free_btn.grid()
        app.generate_plot()

    # ── Label / annotation management ────────────────────────────────────────

    def add_label(self) -> None:
        """Add a freeform text label to the standalone canvas."""
        app = self.app
        if app._standalone_label_entry is None:
            return
        text = app._standalone_label_entry.get().strip()
        if not text:
            return
        xlim = app.ax.get_xlim()
        ylim = app.ax.get_ylim()
        cx = (xlim[0] + xlim[1]) / 2
        cy = (ylim[0] + ylim[1]) / 2
        self.labels.append({"text": text, "x": cx, "y": cy})
        app._standalone_label_entry.delete(0, tk.END)
        self.selected_label = len(self.labels) - 1
        app.generate_plot()
        if not app.history_manager.is_restoring:
            app._capture_current_state()

    def add_dim_preset(self, preset_key: str, default_text: str) -> None:
        """Add a preset dimension line to the standalone canvas."""
        app = self.app
        text = default_text
        if app._standalone_label_entry is not None:
            entry_text = app._standalone_label_entry.get().strip()
            if entry_text:
                text = entry_text
                app._standalone_label_entry.delete(0, tk.END)
            else:
                value_key = app._get_preset_value_key(preset_key)
                if value_key:
                    val = app.input_controller.get_entry_value(value_key).strip()
                    if val:
                        text = val
        if preset_key == "circumference":
            current_text, current_show = app.label_manager.get_entry_values("Circumference")
            if current_text.strip() and current_show:
                app.label_manager.set_label_text("Circumference", "", False)
            else:
                app.label_manager.set_label_text("Circumference", text, True)
            self.builtin_selected = None
            self.selected_dim = None
            self.selected_label = None
            app.generate_plot()
            if not app.history_manager.is_restoring:
                app._capture_current_state(force=True)
            return
        radial_shapes = {"Circle", "Sphere", "Hemisphere", "Cylinder", "Cone"}
        shape = app.shape_var.get()
        if preset_key in ("radius", "diameter") and shape in radial_shapes:
            label_key = "Radius" if preset_key == "radius" else "Diameter"
            other_key = "Diameter" if preset_key == "radius" else "Radius"
            current_text, current_show = app.label_manager.get_entry_values(label_key)
            if current_text.strip() and current_show:
                app.label_manager.set_label_text(label_key, "", False)
            else:
                app.label_manager.set_label_text(label_key, text, True)
                app.label_manager.set_label_text(other_key, "", False)
            self.selected_dim = None
            self.selected_label = None
            app.generate_plot()
            if not app.history_manager.is_restoring:
                app._capture_current_state(force=True)
            return
        builtin_3d_height = {"Cylinder", "Cone", "Hemisphere"}
        if preset_key == "height" and shape in builtin_3d_height:
            current_text, current_show = app.label_manager.get_entry_values("Height")
            if current_text.strip() and current_show:
                app.label_manager.set_label_text("Height", "", False)
            else:
                app.label_manager.set_label_text("Height", text, True)
            self.selected_dim = None
            self.selected_label = None
            app.generate_plot()
            if not app.history_manager.is_restoring:
                app._capture_current_state(force=True)
            return
        if preset_key == "slant" and shape == "Cone":
            current_text, current_show = app.label_manager.get_entry_values("Slant")
            if current_text.strip() and current_show:
                app.label_manager.set_label_text("Slant", "", False)
            else:
                app.label_manager.set_label_text("Slant", text, True)
            self.selected_dim = None
            self.selected_label = None
            app.generate_plot()
            if not app.history_manager.is_restoring:
                app._capture_current_state(force=True)
            return
        if preset_key == "height" and shape in ("Triangular Prism", "Tri Prism"):
            label_key = "Height (Tri)"
            current_text, current_show = app.label_manager.get_entry_values(label_key)
            if current_text.strip() and current_show:
                app.label_manager.set_label_text(label_key, "", False)
            else:
                app.label_manager.set_label_text(label_key, text, True)
            self.selected_dim = None
            self.selected_label = None
            app.generate_plot()
            if not app.history_manager.is_restoring:
                app._capture_current_state(force=True)
            return
        endpoints = app._calc_dim_line_endpoints(preset_key)
        if endpoints is None:
            return
        _perp_presets = {"height", "para_height", "trap_height"}
        _is_prism = shape in ("Rectangular Prism", "Triangular Prism", "Tri Prism")
        right_angle_at = "p1" if (preset_key in _perp_presets and not _is_prism) else None
        self.dim_lines.append({
            "x1": endpoints["x1"], "y1": endpoints["y1"],
            "x2": endpoints["x2"], "y2": endpoints["y2"],
            "text": text,
            "constraint": endpoints.get("constraint"),
            "label_x": endpoints["label_x"],
            "label_y": endpoints["label_y"],
            "preset_key": preset_key,
            "user_dragged": False,
        })
        self.selected_dim = len(self.dim_lines) - 1
        self.selected_label = None
        self.builtin_selected = None
        app.generate_plot()
        if not app.history_manager.is_restoring:
            app._capture_current_state(force=True)

    def remove_annotation(self) -> None:
        """Remove the selected standalone label, dimension line, or circumference arc."""
        app = self.app
        if self.builtin_selected:
            key = self.builtin_selected
            app.label_manager.label_texts.pop(key, None)
            app.label_manager.label_visibility.pop(key, None)
            app.label_manager.custom_positions.pop(key, None)
            self.builtin_selected = None
            app.generate_plot()
            if not app.history_manager.is_restoring:
                app._capture_current_state()
            return
        if self.selected_dim is not None:
            if self.selected_dim < len(self.dim_lines):
                self.dim_lines.pop(self.selected_dim)
            self.selected_dim = None
            app.generate_plot()
            if not app.history_manager.is_restoring:
                app._capture_current_state()
            return
        if self.selected_label is not None:
            if self.selected_label < len(self.labels):
                self.labels.pop(self.selected_label)
            self.selected_label = None
            app.generate_plot()
            if not app.history_manager.is_restoring:
                app._capture_current_state()

    def confirm_label(self) -> None:
        """Add or update a standalone label/dim depending on edit mode."""
        if self.edit_mode is not None:
            self.apply_edit()
        else:
            self.add_label()

    def enter_edit(self, edit_type: str, idx: int) -> None:
        """Enter edit mode: populate entry with existing text, swap button to Update."""
        app = self.app
        if edit_type == "builtin":
            bk = getattr(app, '_builtin_edit_key', None)
            existing = app.label_manager.label_texts.get(bk, "") if bk else ""
        elif edit_type == "circ":
            existing = app.label_manager.label_texts.get("Circumference", "c")
        elif edit_type == "label" and idx < len(self.labels):
            existing = self.labels[idx]["text"]
        elif edit_type == "dim" and idx < len(self.dim_lines):
            existing = self.dim_lines[idx]["text"]
        else:
            return
        self.edit_mode = {"type": edit_type, "idx": idx}
        if app._standalone_label_entry is not None:
            app._standalone_label_entry.delete(0, tk.END)
            app._standalone_label_entry.insert(0, existing)
            app._standalone_label_entry.focus_set()
            app._standalone_label_entry.select_range(0, tk.END)
        if app._standalone_text_btn is not None:
            app._standalone_text_btn.config(text="Update")
        if app._standalone_cancel_btn is not None:
            app._standalone_cancel_btn.pack(side=tk.LEFT, padx=1)

    def apply_edit(self) -> None:
        """Apply the edited text to the label or dim line."""
        app = self.app
        if self.edit_mode is None:
            return
        if app._standalone_label_entry is None:
            return
        new_text = app._standalone_label_entry.get().strip()
        if not new_text:
            self.cancel_edit()
            return
        edit_type = self.edit_mode["type"]
        idx = self.edit_mode["idx"]
        if edit_type == "builtin":
            bk = getattr(app, '_builtin_edit_key', None)
            if bk:
                app.label_manager.set_label_text(bk, new_text, True)
        elif edit_type == "circ":
            app.label_manager.set_label_text("Circumference", new_text, True)
        elif edit_type == "label" and idx < len(self.labels):
            self.labels[idx]["text"] = new_text
        elif edit_type == "dim" and idx < len(self.dim_lines):
            self.dim_lines[idx]["text"] = new_text
        self.cancel_edit()
        app.generate_plot()
        if not app.history_manager.is_restoring:
            app._capture_current_state()

    def cancel_edit(self) -> None:
        """Exit edit mode, restore + Text button."""
        app = self.app
        self.edit_mode = None
        if app._standalone_label_entry is not None:
            app._standalone_label_entry.delete(0, tk.END)
        if app._standalone_text_btn is not None:
            app._standalone_text_btn.config(text="+ Text")
        if app._standalone_cancel_btn is not None:
            app._standalone_cancel_btn.pack_forget()

    def reset(self) -> None:
        """Clear all standalone annotation state (called on shape change / clear)."""
        self.labels = []
        self.selected_label = None
        self.label_bboxes = []
        self.dim_lines = []
        self.selected_dim = None
        self.dim_endpoints = []
        self.dim_label_bboxes = []
        self.dim_mode = False
        self.dim_first_point = None
        self.dim_preview_line = None
        self.edit_mode = None
        self.builtin_selected = None
        self.builtin_dim_endpoints = []



