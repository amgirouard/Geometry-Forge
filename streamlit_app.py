"""streamlit_app.py — Geometry Forge Streamlit UI entry point.

Run with:  streamlit run streamlit_app.py
"""
from __future__ import annotations

import io
import math
import streamlit as st
from PIL import Image as _PILImage
from streamlit_image_coordinates import streamlit_image_coordinates

from geometry_forge.core import GeometryCore
from geometry_forge.models import (
    AppConstants, ShapeConfigProvider, ShapeFeature,
    TriangleType, PolygonType,
)

CANVAS_DPI: int = 100  # DPI used when rendering the canvas PNG for click capture

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Geometry Forge",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Reduce vertical spacing between sidebar widgets */
section[data-testid="stSidebar"] .block-container {
    padding-top: 0.5rem;
}
section[data-testid="stSidebar"] .stElementContainer {
    margin-bottom: -0.4rem;
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stTextInput,
section[data-testid="stSidebar"] .stCheckbox,
section[data-testid="stSidebar"] .stSlider,
section[data-testid="stSidebar"] .stSelectbox,
section[data-testid="stSidebar"] .stRadio,
section[data-testid="stSidebar"] .stButton {
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div {
    gap: 0.25rem;
}
</style>
""", unsafe_allow_html=True)

# ── Helper functions ──────────────────────────────────────────────────────────

# Preset dim line definitions per shape.
# Each entry: (preset_key, display_label, default_text)
# preset_key must match a key in GeometryCore._dim_dispatch.
_SHAPE_PRESET_DIM_LINES: dict[str, list[tuple[str, str, str]]] = {
    "Rectangle": [
        ("height", "Length", "l"),
        ("width",  "Width",  "w"),
    ],
    "Square": [
        ("side_v", "Side", "s"),
    ],
    "Triangle_Custom": [
        ("height", "Height",     "h"),
        ("width",  "Base",       "b"),
        ("side_l", "Left Side",  "s"),
        ("side_r", "Right Side", "s"),
    ],
    "Triangle_Isosceles": [
        ("width",  "Base",   "a"),
        ("side_l", "Side",   "s"),
        ("height", "Height", "h"),
    ],
    "Triangle_Scalene": [
        ("width",  "Base",       "a"),
        ("side_l", "Left Side",  "s"),
        ("side_r", "Right Side", "s"),
        ("height", "Height",     "h"),
    ],
    "Triangle_Equilateral": [
        ("side_l", "Side", "s"),
    ],
    "Triangle_Right": [
        ("leg_a", "Leg A",      "a"),
        ("leg_b", "Leg B",      "b"),
        ("hyp",   "Hypotenuse", "c"),
    ],
    "Parallelogram": [
        ("para_height", "Height",     "h"),
        ("para_base",   "Base",       "a"),
        ("para_side_l", "Left Side",  "s"),
        ("para_side_r", "Right Side", "s"),
    ],
    "Trapezoid": [
        ("trap_height", "Height",     "h"),
        ("trap_top",    "Top",        "a"),
        ("trap_base",   "Base",       "b"),
        ("trap_side_l", "Left Side",  "s"),
        ("trap_side_r", "Right Side", "s"),
    ],
    "Circle": [],
    "Sphere": [],
    "Hemisphere": [],
    "Cylinder": [],
    "Cone": [],
    "Rectangular Prism": [
        ("length", "Length", "l"),
        ("width",  "Width",  "w"),
        ("height", "Height", "h"),
    ],
    "Triangular Prism": [
        ("tri_base",   "Base",   "b"),
        ("height",     "Height", "h"),
        ("tri_length", "Length", "l"),
    ],
    "Polygon": [
        ("side", "Side", "s"),
    ],
    "Line Segment": [
        ("width", "Length", "l"),
    ],
}


# Short default text for each Dimension Label toggle (used when user hasn't entered custom text)
_TOGGLE_LABEL_DEFAULTS: dict[str, str] = {
    "Circumference":  "c",
    "Radius":         "r",
    "Diameter":       "d",
    "Height":         "h",
    "Slant":          "l",
    "Length (Front)": "l",
    "Width (Side)":   "w",
    "Base (Tri)":     "b",
    "Height (Tri)":   "h",
    "Length (Prism)": "l",
}


def _get_preset_dim_lines(shape: str, triangle_type: str) -> list[tuple[str, str, str]]:
    """Return the preset dim line specs for the current shape (and triangle sub-type)."""
    if shape == "Triangle":
        key = f"Triangle_{triangle_type}"
        return _SHAPE_PRESET_DIM_LINES.get(key, _SHAPE_PRESET_DIM_LINES.get("Triangle_Custom", []))
    return _SHAPE_PRESET_DIM_LINES.get(shape, [])


def _preset_exists(core: GeometryCore, preset_key: str) -> int:
    """Return the index of an existing preset dim line, or -1 if not present."""
    for i, dl in enumerate(core.standalone_dim_lines):
        if dl.get("preset_key") == preset_key:
            return i
    return -1


def _render_preset_dim_lines(core: GeometryCore, shape: str, capture_fn) -> None:
    """Render the preset dimension line checkboxes + label text inputs (no subheader — caller adds it)."""
    specs = _get_preset_dim_lines(shape, core.triangle_type)
    if not specs:
        return
    changed = False
    for preset_key, display_label, default_text in specs:
        idx = _preset_exists(core, preset_key)
        cur_checked = idx >= 0
        cur_text = core.standalone_dim_lines[idx]["text"] if cur_checked else default_text

        col_chk, col_txt = st.columns([2, 1])
        with col_chk:
            new_checked = st.checkbox(
                display_label,
                value=cur_checked,
                key=f"preset_dl_chk_{shape}_{preset_key}",
            )
        with col_txt:
            new_text = st.text_input(
                display_label,
                value=cur_text,
                key=f"preset_dl_txt_{shape}_{preset_key}",
                label_visibility="collapsed",
                placeholder=default_text,
                disabled=not new_checked,
            )

        if new_checked and not cur_checked:
            # Add preset dim line (endpoints will be computed on first draw)
            capture_fn()
            core.standalone_dim_lines.append({
                "text": new_text or default_text,
                "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 0.0,
                "label_x": None, "label_y": None,
                "preset_key": preset_key,
                "user_dragged": False,
                "constraint": None,
            })
            changed = True
        elif not new_checked and cur_checked:
            capture_fn()
            core.standalone_dim_lines.pop(idx)
            changed = True
        elif cur_checked and new_text != cur_text:
            capture_fn()
            core.standalone_dim_lines[idx]["text"] = new_text
            changed = True

    if changed:
        st.rerun()


def _get_relevant_toggle_keys(shape: str, config) -> list[tuple[str, str]]:
    """Return TOGGLE_LABEL_KEYS subset that applies to the current shape."""
    TOGGLE_SHAPE_MAP: dict[str, set[str]] = {
        "Circumference":  {"Circle", "Sphere", "Cylinder", "Cone", "Hemisphere"},
        "Radius":         {"Circle", "Sphere", "Cylinder", "Cone", "Hemisphere"},
        "Diameter":       {"Circle", "Sphere", "Cylinder", "Cone", "Hemisphere"},
        "Height":         {"Cylinder", "Cone"},
        "Slant":          {"Cone"},
        "Length (Front)": {"Rectangular Prism"},
        "Width (Side)":   {"Rectangular Prism"},
        "Base (Tri)":     {"Triangular Prism"},
        "Height (Tri)":   {"Triangular Prism"},
        "Length (Prism)": {"Triangular Prism"},
    }
    result = []
    for lbl_key, st_key in GeometryCore.TOGGLE_LABEL_KEYS:
        allowed = TOGGLE_SHAPE_MAP.get(lbl_key)
        if allowed is None or shape in allowed:
            result.append((lbl_key, st_key))
    return result


def _render_composite_controls(core: GeometryCore, shape: str, capture_fn) -> None:
    """Render composite shape list + per-shape controls."""
    st.subheader("Composite Shapes")

    available = (
        ["Rectangle", "Square", "Triangle", "Circle",
         "Parallelogram", "Trapezoid", "Polygon"]
        if shape == "2D Composite"
        else ["Sphere", "Hemisphere", "Cylinder", "Cone",
              "Rectangular Prism", "Triangular Prism"]
    )

    new_composite = st.multiselect(
        "Selected Shapes",
        options=available,
        default=[s for s in core.composite_shapes if s in available],
        key="ms_composite_shapes",
    )
    if new_composite != core.composite_shapes:
        capture_fn()
        new_positions: dict[int, tuple[float, float]] = {}
        new_transforms: dict[int, dict] = {}
        for i, s in enumerate(new_composite):
            if i < len(core.composite_shapes) and core.composite_shapes[i] == s:
                new_positions[i] = core.composite_positions.get(i, (0.0, 0.0))
                new_transforms[i] = core.composite_transforms.get(
                    i, {"h": False, "v": False, "side": 0}
                )
            else:
                new_positions[i] = (float(i) * 6.0, 0.0)
                new_transforms[i] = {"h": False, "v": False, "side": 0}
        core.composite_shapes = new_composite
        core.composite_positions = new_positions
        core.composite_transforms = new_transforms

    # Per-shape controls
    for i, s in enumerate(core.composite_shapes):
        with st.expander(f"Shape {i + 1}: {s}", expanded=False):
            pos = core.composite_positions.get(i, (0.0, 0.0))
            c1, c2 = st.columns(2)
            with c1:
                new_x = st.number_input("X", value=float(pos[0]), step=0.5,
                                         key=f"comp_x_{i}_{s}")
            with c2:
                new_y = st.number_input("Y", value=float(pos[1]), step=0.5,
                                         key=f"comp_y_{i}_{s}")
            if (new_x, new_y) != (float(pos[0]), float(pos[1])):
                capture_fn()
                core.composite_positions[i] = (new_x, new_y)

            tr = core.composite_transforms.get(i, {"h": False, "v": False, "side": 0})
            tc1, tc2 = st.columns(2)
            with tc1:
                new_fh = st.checkbox("Flip H", value=tr.get("h", False),
                                      key=f"comp_fh_{i}_{s}")
            with tc2:
                new_fv = st.checkbox("Flip V", value=tr.get("v", False),
                                      key=f"comp_fv_{i}_{s}")

            comp_cfg = ShapeConfigProvider.get(s)
            num_sides = comp_cfg.num_sides if comp_cfg.num_sides > 0 else 4
            cur_side = tr.get("side", 0)
            tr3, tr4 = st.columns(2)
            with tr3:
                if st.button("↺ CCW", key=f"comp_ccw_{i}_{s}", width="stretch"):
                    capture_fn()
                    core.composite_transforms[i] = {
                        **tr, "side": (cur_side - 1) % num_sides
                    }
                    st.rerun()
            with tr4:
                if st.button("↻ CW", key=f"comp_cw_{i}_{s}", width="stretch"):
                    capture_fn()
                    core.composite_transforms[i] = {
                        **tr, "side": (cur_side + 1) % num_sides
                    }
                    st.rerun()

            if new_fh != tr.get("h", False) or new_fv != tr.get("v", False):
                capture_fn()
                core.composite_transforms[i] = {**tr, "h": new_fh, "v": new_fv}



# ── Canvas interaction helpers ────────────────────────────────────────────────

def _pixel_to_data(
    px: int, py: int, fig, ax
) -> tuple[float, float] | None:
    """Convert pixel coords (from streamlit_image_coordinates) to axes data coords."""
    fig_w_px = fig.get_figwidth() * CANVAS_DPI
    fig_h_px = fig.get_figheight() * CANVAS_DPI
    norm_x = px / fig_w_px
    norm_y = py / fig_h_px
    ax_pos = ax.get_position()
    if not (ax_pos.x0 <= norm_x <= ax_pos.x0 + ax_pos.width):
        return None
    if not (ax_pos.y0 <= norm_y <= ax_pos.y0 + ax_pos.height):
        return None
    ax_rel_x = (norm_x - ax_pos.x0) / ax_pos.width
    ax_rel_y = (norm_y - ax_pos.y0) / ax_pos.height
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    data_x = xlim[0] + ax_rel_x * (xlim[1] - xlim[0])
    data_y = ylim[0] + (1.0 - ax_rel_y) * (ylim[1] - ylim[0])  # Y axis is inverted
    return round(data_x, 2), round(data_y, 2)


def _find_nearest_label(data_x: float, data_y: float, ax) -> dict | None:
    """Return a selection dict for the label nearest the click, or None."""
    # Check preset dim line label bboxes (padded hit area)
    pad = 0.15
    for i, (bx0, by0, bx1, by1) in enumerate(core._standalone_dim_label_bboxes):
        x_lo = min(bx0, bx1) - pad
        x_hi = max(bx0, bx1) + pad
        y_lo = min(by0, by1) - pad
        y_hi = max(by0, by1) + pad
        if x_lo <= data_x <= x_hi and y_lo <= data_y <= y_hi:
            if i < len(core.standalone_dim_lines):
                dim = core.standalone_dim_lines[i]
                return {"type": "preset_dl", "idx": i, "name": dim["text"]}

    # Check built-in labels from auto_positions (distance threshold)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    view_range = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    threshold = max(0.5, view_range * 0.08)

    best_key, best_dist = None, float("inf")
    for key, pos in core.label_manager.auto_positions.items():
        lx, ly = pos[0], pos[1]
        if key in core.label_manager.custom_positions:
            lx, ly = core.label_manager.custom_positions[key]
        dist = math.sqrt((data_x - lx) ** 2 + (data_y - ly) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_key = key

    if best_key and best_dist < threshold:
        return {"type": "builtin", "key": best_key, "name": best_key}

    return None


def _label_canvas_x_frac(annotation: dict, fig, ax) -> float:
    """Return the 0-1 horizontal fraction of the canvas where the label sits."""
    if annotation["type"] == "builtin":
        key = annotation["key"]
        if key in core.label_manager.custom_positions:
            data_x = core.label_manager.custom_positions[key][0]
        elif key in core.label_manager.auto_positions:
            data_x = core.label_manager.auto_positions[key][0]
        else:
            return 0.5
    else:  # preset_dl
        idx = annotation["idx"]
        if idx >= len(core.standalone_dim_lines):
            return 0.5
        dim = core.standalone_dim_lines[idx]
        lx = dim.get("label_x")
        data_x = lx if lx is not None else (dim["x1"] + dim["x2"]) / 2

    xlim = ax.get_xlim()
    ax_pos = ax.get_position()
    if xlim[1] == xlim[0]:
        return 0.5
    ax_rel = (data_x - xlim[0]) / (xlim[1] - xlim[0])
    return max(0.05, min(0.95, ax_pos.x0 + ax_rel * ax_pos.width))


def _render_nudge_panel(fig, ax, capture_fn) -> None:
    """Render arrow-nudge controls below the canvas, positioned under the selected label."""
    ann = st.session_state.get("selected_annotation")
    if not ann:
        return

    step = st.session_state.get("nudge_step", 0.3)

    def _nudge_builtin(dx: float, dy: float) -> None:
        key = ann["key"]
        lm = core.label_manager
        if key in lm.custom_positions:
            cx, cy = lm.custom_positions[key]
        elif key in lm.auto_positions:
            cx, cy = lm.auto_positions[key][0], lm.auto_positions[key][1]
        else:
            return
        capture_fn()
        lm.custom_positions[key] = (cx + dx, cy + dy)

    def _nudge_dl_label(dx: float, dy: float) -> None:
        dim = core.standalone_dim_lines[ann["idx"]]
        lx = dim.get("label_x")
        ly = dim.get("label_y")
        if lx is None:
            lx = (dim["x1"] + dim["x2"]) / 2
        if ly is None:
            ly = (dim["y1"] + dim["y2"]) / 2
        capture_fn()
        dim["label_x"] = lx + dx
        dim["label_y"] = ly + dy

    def _nudge_dl_line(dx: float, dy: float) -> None:
        dim = core.standalone_dim_lines[ann["idx"]]
        capture_fn()
        dim["x1"] += dx
        dim["y1"] += dy
        dim["x2"] += dx
        dim["y2"] += dy
        if dim.get("label_x") is not None:
            dim["label_x"] += dx
        if dim.get("label_y") is not None:
            dim["label_y"] += dy
        dim["user_dragged"] = True

    # Column layout: position panel under the label's x position
    x_frac = _label_canvas_x_frac(ann, fig, ax)
    panel_w = 0.18
    left_w = max(0.01, x_frac - panel_w / 2)
    right_w = max(0.01, 1.0 - x_frac - panel_w / 2)
    total = left_w + panel_w + right_w
    _, pcol, _ = st.columns([left_w / total, panel_w / total, right_w / total])

    with pcol:
        with st.container(border=True):
            hc1, hc2 = st.columns([4, 1])
            with hc1:
                st.caption(f"**{ann['name']}**")
            with hc2:
                if st.button("✕", key="btn_nudge_close"):
                    core.label_manager.builtin_selected = None
                    st.session_state.selected_annotation = None
                    st.rerun()

            # Step size
            step_options = ["Small", "Medium", "Large"]
            step_values = {"Small": 0.1, "Medium": 0.3, "Large": 0.7}
            cur_label = "Small" if step <= 0.15 else "Medium" if step <= 0.5 else "Large"
            new_step_label = st.radio(
                "Step",
                step_options,
                index=step_options.index(cur_label),
                horizontal=True,
                label_visibility="collapsed",
                key="nudge_step_radio",
            )
            if step_values[new_step_label] != step:
                st.session_state.nudge_step = step_values[new_step_label]
                st.rerun()

            # ── Label nudge arrows ──
            st.caption("Label")
            _nudge_label = _nudge_builtin if ann["type"] == "builtin" else _nudge_dl_label

            _, uc, _ = st.columns(3)
            with uc:
                if st.button("↑", key="nl_u", use_container_width=True):
                    _nudge_label(0, step)
                    st.rerun()
            lc, cc, rc = st.columns(3)
            with lc:
                if st.button("←", key="nl_l", use_container_width=True):
                    _nudge_label(-step, 0)
                    st.rerun()
            with cc:
                if st.button("○", key="nl_reset", use_container_width=True, help="Reset label"):
                    capture_fn()
                    if ann["type"] == "builtin":
                        core.label_manager.custom_positions.pop(ann["key"], None)
                    else:
                        dim = core.standalone_dim_lines[ann["idx"]]
                        dim["label_x"] = None
                        dim["label_y"] = None
                    st.rerun()
            with rc:
                if st.button("→", key="nl_r", use_container_width=True):
                    _nudge_label(step, 0)
                    st.rerun()
            _, dc, _ = st.columns(3)
            with dc:
                if st.button("↓", key="nl_d", use_container_width=True):
                    _nudge_label(0, -step)
                    st.rerun()

            # ── Line nudge arrows (preset dim lines only) ──
            if ann["type"] == "preset_dl":
                st.caption("Line")
                _, luc, _ = st.columns(3)
                with luc:
                    if st.button("↑", key="line_u", use_container_width=True):
                        _nudge_dl_line(0, step)
                        st.rerun()
                llc, lcc, lrc = st.columns(3)
                with llc:
                    if st.button("←", key="line_l", use_container_width=True):
                        _nudge_dl_line(-step, 0)
                        st.rerun()
                with lcc:
                    if st.button("○", key="line_reset", use_container_width=True, help="Reset line"):
                        capture_fn()
                        dim = core.standalone_dim_lines[ann["idx"]]
                        dim["user_dragged"] = False
                        dim["label_x"] = None
                        dim["label_y"] = None
                        st.rerun()
                with lrc:
                    if st.button("→", key="line_r", use_container_width=True):
                        _nudge_dl_line(step, 0)
                        st.rerun()
                _, ldc, _ = st.columns(3)
                with ldc:
                    if st.button("↓", key="line_d", use_container_width=True):
                        _nudge_dl_line(0, -step)
                        st.rerun()



# ── Session state bootstrap ───────────────────────────────────────────────────
if "core" not in st.session_state:
    st.session_state.core = GeometryCore()
if "history" not in st.session_state:
    st.session_state.history: list[dict] = []
if "redo_stack" not in st.session_state:
    st.session_state.redo_stack: list[dict] = []
if "selected_annotation" not in st.session_state:
    st.session_state.selected_annotation = None
if "nudge_step" not in st.session_state:
    st.session_state.nudge_step = 0.3
if "last_click_pos" not in st.session_state:
    st.session_state.last_click_pos = None

core: GeometryCore = st.session_state.core


# ── Undo/redo helpers ─────────────────────────────────────────────────────────

def capture_state() -> None:
    """Push current state onto the undo stack and clear redo stack."""
    snapshot = core._build_state_snapshot()
    if st.session_state.history and st.session_state.history[-1] == snapshot:
        return
    st.session_state.history.append(snapshot)
    st.session_state.redo_stack.clear()


def do_undo() -> None:
    if len(st.session_state.history) < 2:
        return
    st.session_state.redo_stack.append(st.session_state.history.pop())
    core._apply_state(st.session_state.history[-1])


def do_redo() -> None:
    if not st.session_state.redo_stack:
        return
    state = st.session_state.redo_stack.pop()
    st.session_state.history.append(state)
    core._apply_state(state)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📐 Geometry Forge")

    # ── 1. Shape Selection ────────────────────────────────────────────────────
    st.subheader("Shape")

    categories = core.get_categories()
    current_cat_idx = (
        categories.index(core.category)
        if core.category in categories else 0
    )
    selected_cat = st.selectbox(
        "Category",
        categories,
        index=current_cat_idx,
        key="sb_category",
    )

    shapes_in_cat = core.get_shapes(selected_cat)

    # Category change → reset to first shape
    if core.category != selected_cat:
        capture_state()
        core.category = selected_cat
        core.shape_name = shapes_in_cat[0] if shapes_in_cat else None
        core.params = {}
        core.transform_controller.reset()
        core.scale_manager.reset_many(["aspect", "slope", "peak_offset", "view_scale"])
        core.dimension_mode = "Default"
        core.triangle_type = "Custom"
        st.session_state.selected_annotation = None
        core.label_manager.builtin_selected = None

    current_shape_idx = (
        shapes_in_cat.index(core.shape_name)
        if core.shape_name in shapes_in_cat else 0
    )
    selected_shape = st.selectbox(
        "Shape",
        shapes_in_cat,
        index=current_shape_idx,
        key="sb_shape",
    )

    # Shape change → reset dependent state
    if core.shape_name != selected_shape:
        capture_state()
        core.shape_name = selected_shape
        core.params = {}
        core.transform_controller.reset()
        core.scale_manager.reset_many(["aspect", "slope", "peak_offset", "view_scale"])
        core.dimension_mode = "Default"
        core.triangle_type = "Custom"
        core.standalone_labels = []
        core.standalone_dim_lines = []
        st.session_state.selected_annotation = None
        core.label_manager.builtin_selected = None

    shape = core.shape_name
    config = ShapeConfigProvider.get(shape) if shape else None
    is_composite = core._is_composite_shape(shape)

    st.divider()

    # Under-construction notice for Angles & Lines and Composite Figures
    if core.category in ("Angles & Lines", "Composite Figures"):
        st.caption("🚧 Under construction")

    # ── 2. Shape Sub-type ──────────────────────────────────────────────────────
    if shape == "Triangle":
        st.subheader("Triangle Type")
        tri_types = [t.value for t in TriangleType]
        tri_idx = tri_types.index(core.triangle_type) if core.triangle_type in tri_types else 0
        new_tri = st.radio(
            "Triangle Type",
            tri_types,
            index=tri_idx,
            key="rb_tri_type",
            label_visibility="collapsed",
            horizontal=True,
        )
        if core.triangle_type != new_tri:
            capture_state()
            core.triangle_type = new_tri
            core.params = {}
            core.scale_manager.reset_many(["aspect", "slope", "peak_offset"])
            st.session_state.selected_annotation = None
            core.label_manager.builtin_selected = None
        # Use triangle sub-type config for all following sections
        config = ShapeConfigProvider.get_triangle_config(core.triangle_type)

    elif shape == "Polygon":
        st.subheader("Polygon Type")
        poly_types = [t.value for t in PolygonType]
        poly_idx = (
            poly_types.index(core.polygon_type)
            if core.polygon_type in poly_types else 0
        )
        new_poly = st.radio(
            "Polygon Type",
            poly_types,
            index=poly_idx,
            key="rb_poly_type",
            label_visibility="collapsed",
        )
        if core.polygon_type != new_poly:
            capture_state()
            core.polygon_type = new_poly

    # ── 3. Dimension Mode ──────────────────────────────────────────────────────
    if config and config.has_dimension_mode and shape != "Triangle":
        st.subheader("Dimension Mode")
        dim_modes = ["Default", "Custom"]
        dim_idx = dim_modes.index(core.dimension_mode) if core.dimension_mode in dim_modes else 0
        new_dim_mode = st.radio(
            "Mode",
            dim_modes,
            index=dim_idx,
            key="rb_dim_mode",
            label_visibility="collapsed",
            horizontal=True,
        )
        if core.dimension_mode != new_dim_mode:
            capture_state()
            core.dimension_mode = new_dim_mode
            defaults = config.get_defaults_for_mode(new_dim_mode)
            active_labels = (
                config.custom_labels
                if new_dim_mode == "Custom" and config.custom_labels
                else config.labels
            )
            core.params = dict(zip(active_labels, defaults))

        if config.help_text:
            st.caption(config.help_text)

    # ── 4. Shape Parameters — only shown in Custom mode (or no dimension mode) ──
    _SHAPES_NO_PARAMS = {"Circle", "Square", "Polygon", "Sphere", "Hemisphere"}
    _is_equilateral = (shape == "Triangle" and core.triangle_type == "Equilateral")
    if config and not is_composite and shape not in _SHAPES_NO_PARAMS and not _is_equilateral:
        # For shapes with a dimension mode, only show params in Custom mode.
        # For shapes without a dimension mode, always show params.
        show_params = (not config.has_dimension_mode) or (core.dimension_mode == "Custom")

        if show_params:
            if core.dimension_mode == "Custom" and config.custom_labels:
                active_labels = config.custom_labels
                default_vals = config.custom_values
            else:
                active_labels = config.labels
                default_vals = config.default_values

            if active_labels:
                st.subheader("Parameters")
                params_changed = False
                for lbl, default_val in zip(active_labels, default_vals):
                    current_val = core.params.get(lbl, default_val)
                    pl, pi = st.columns([2, 1])
                    with pl:
                        st.write(lbl)
                    with pi:
                        new_val = st.text_input(
                            lbl,
                            value=current_val,
                            key=f"param_{shape}_{core.triangle_type}_{core.dimension_mode}_{lbl}",
                            label_visibility="collapsed",
                        )
                    if new_val != current_val:
                        params_changed = True
                        core.params[lbl] = new_val
                if params_changed:
                    capture_state()

    # ── 5. Sliders — Adjust Shape/Slope/Peak only in Default mode ──────────────
    if config and not is_composite and shape:
        slider_config = config  # already triangle sub-config if applicable

        has_slider_shape = slider_config.has_feature(ShapeFeature.SLIDER_SHAPE)
        has_slider_slope = slider_config.has_feature(ShapeFeature.SLIDER_SLOPE)
        has_slider_peak = slider_config.has_feature(ShapeFeature.SLIDER_PEAK)

        # Only show adjust sliders in Default mode (or shapes without dimension mode)
        show_adjust = (not config.has_dimension_mode) or (core.dimension_mode == "Default")

        if show_adjust and (has_slider_shape or has_slider_slope or has_slider_peak):
            st.subheader("Adjust Shape")

        if show_adjust and has_slider_shape:
            spec = core.scale_manager.specs["aspect"]
            new_aspect = st.slider(
                "Adjust Shape",
                min_value=float(spec.min_val),
                max_value=float(spec.max_val),
                value=float(core.scale_manager.get("aspect")),
                step=0.01,
                key="sl_aspect",
            )
            if abs(new_aspect - core.scale_manager.get("aspect")) > 0.001:
                core.scale_manager.set("aspect", new_aspect)

        if show_adjust and has_slider_slope:
            spec = core.scale_manager.specs["slope"]
            new_slope = st.slider(
                "Adjust Slope",
                min_value=float(spec.min_val),
                max_value=float(spec.max_val),
                value=float(core.scale_manager.get("slope")),
                step=0.01,
                key="sl_slope",
            )
            if abs(new_slope - core.scale_manager.get("slope")) > 0.001:
                core.scale_manager.set("slope", new_slope)

        if show_adjust and has_slider_peak:
            spec = core.scale_manager.specs["peak_offset"]
            new_peak = st.slider(
                "Peak Offset",
                min_value=float(spec.min_val),
                max_value=float(spec.max_val),
                value=float(core.scale_manager.get("peak_offset")),
                step=0.01,
                key="sl_peak",
            )
            if abs(new_peak - core.scale_manager.get("peak_offset")) > 0.001:
                core.scale_manager.set("peak_offset", new_peak)

        # View scale only for non-2D/3D categories
        if core.category not in ("2D Figures", "3D Solids"):
            spec_vs = core.scale_manager.specs["view_scale"]
            new_view = st.slider(
                "View Scale",
                min_value=float(spec_vs.min_val),
                max_value=float(spec_vs.max_val),
                value=float(core.scale_manager.get("view_scale")),
                step=0.01,
                key="sl_view_scale",
            )
            if abs(new_view - core.scale_manager.get("view_scale")) > 0.001:
                core.scale_manager.set("view_scale", new_view)

    # ── 6. Transformations ─────────────────────────────────────────────────────
    if config and not is_composite and (
        config.has_feature(ShapeFeature.FLIP) or config.has_feature(ShapeFeature.ROTATE)
    ):
        st.divider()
        st.subheader("Transformations")

        if config.has_feature(ShapeFeature.FLIP):
            c1, c2 = st.columns(2)
            with c1:
                label_h = "↔ Flip H ✓" if core.transform_controller.flip_h else "↔ Flip H"
                if st.button(label_h, key="btn_flip_h", width="stretch"):
                    capture_state()
                    core.transform_controller.flip_h = not core.transform_controller.flip_h
                    st.rerun()
            with c2:
                label_v = "↕ Flip V ✓" if core.transform_controller.flip_v else "↕ Flip V"
                if st.button(label_v, key="btn_flip_v", width="stretch"):
                    capture_state()
                    core.transform_controller.flip_v = not core.transform_controller.flip_v
                    st.rerun()

        if config.has_feature(ShapeFeature.ROTATE):
            num_sides = config.num_sides if config.num_sides > 0 else 4
            rot_labels = config.rotation_labels
            current_side = core.transform_controller.base_side

            c3, c4 = st.columns(2)
            with c3:
                if st.button("↺ CCW", key="btn_rot_ccw", width="stretch"):
                    capture_state()
                    core.transform_controller.rotate(-1, num_sides)
                    st.rerun()
            with c4:
                if st.button("↻ CW", key="btn_rot_cw", width="stretch"):
                    capture_state()
                    core.transform_controller.rotate(1, num_sides)
                    st.rerun()

            if rot_labels and 0 <= current_side < len(rot_labels):
                st.caption(f"Rotation: {rot_labels[current_side]}")
            else:
                st.caption(f"Rotation: position {current_side}")

        if st.button("Reset Transformations", key="btn_reset_xf", width="stretch"):
            capture_state()
            core.transform_controller.reset()
            st.rerun()

    # ── 7. Dimension Lines & Labels (merged) ───────────────────────────────────
    if shape and not is_composite:
        preset_specs = _get_preset_dim_lines(shape, core.triangle_type)
        relevant_toggles = _get_relevant_toggle_keys(shape, config) if config else []
        if preset_specs or relevant_toggles:
            st.divider()
            st.subheader("Dimension Lines")

        # Preset dim lines (dashed measurement lines on the canvas)
        if preset_specs:
            _render_preset_dim_lines(core, shape, capture_state)

        # Label toggles (arcs, built-in dim lines drawn by the shape itself)
        for lbl_key, st_key in relevant_toggles:
            stored_text = core.label_manager.label_texts.get(lbl_key, "")
            short_default = _TOGGLE_LABEL_DEFAULTS.get(lbl_key, lbl_key)
            cur_text = stored_text if stored_text else short_default
            cur_vis = core.label_manager.label_visibility.get(lbl_key, False)

            tc, ti = st.columns([2, 1])
            with tc:
                new_vis = st.checkbox(lbl_key, value=cur_vis,
                                      key=f"toggle_vis_{st_key}")
            with ti:
                new_text = st.text_input(
                    lbl_key,
                    value=cur_text,
                    key=f"toggle_text_{st_key}",
                    label_visibility="collapsed",
                    placeholder=short_default,
                )

            if new_vis != cur_vis or new_text != cur_text:
                capture_state()
                display_text = new_text.strip() if new_text.strip() else short_default
                if new_vis or new_text.strip():
                    core.label_manager.set_label_text(lbl_key, display_text, new_vis)
                else:
                    core.label_manager.label_texts.pop(lbl_key, None)
                    core.label_manager.label_visibility.pop(lbl_key, None)

    # ── 9. Hash Marks (Polygon only) ──────────────────────────────────────────
    if config and config.has_feature(ShapeFeature.HASH_MARKS):
        st.subheader("Options")
        new_hash = st.checkbox(
            "Show Hash Marks", value=core.show_hashmarks, key="chk_hashmarks"
        )
        if new_hash != core.show_hashmarks:
            capture_state()
            core.show_hashmarks = new_hash

    # ── 10. Composite Shape Controls ────────────────────────────────────────────
    if is_composite:
        _render_composite_controls(core, shape, capture_state)

    # ── 11. Appearance ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Appearance")

    fa_col, fb_col = st.columns([1, 1])
    with fa_col:
        new_font_size = st.slider(
            "Font Size",
            min_value=AppConstants.MIN_FONT_SIZE,
            max_value=AppConstants.MAX_FONT_SIZE,
            value=core.font_size,
            step=1,
            key="sl_font_size",
        )
        if new_font_size != core.font_size:
            core.font_size = new_font_size
    with fb_col:
        font_families = ["serif", "sans-serif", "monospace"]
        ff_idx = (
            font_families.index(core.font_family)
            if core.font_family in font_families else 0
        )
        new_ff = st.selectbox(
            "Font Family", font_families, index=ff_idx, key="sb_font_family"
        )
        if new_ff != core.font_family:
            core.font_family = new_ff

    new_lw = st.slider(
        "Line Width",
        min_value=AppConstants.MIN_LINE_WIDTH,
        max_value=AppConstants.MAX_LINE_WIDTH,
        value=core.line_width,
        step=1,
        key="sl_line_width",
    )
    if new_lw != core.line_width:
        core.line_width = new_lw

    # ── 13. Actions ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Actions")
    ca, cb = st.columns(2)
    with ca:
        can_undo = len(st.session_state.history) >= 2
        if st.button("↩ Undo", key="btn_undo",
                     disabled=not can_undo, width="stretch"):
            do_undo()
            st.rerun()
    with cb:
        can_redo = bool(st.session_state.redo_stack)
        if st.button("↪ Redo", key="btn_redo",
                     disabled=not can_redo, width="stretch"):
            do_redo()
            st.rerun()

    if st.button("↺ Reset All", key="btn_reset_all", width="stretch"):
        capture_state()
        core.reset_all()
        st.rerun()

    st.divider()

    # Generate bytes for download (only when shape is selected)
    if shape:
        _fig_for_dl = core.generate_figure()
        _png = core.get_figure_bytes("png")
        _svg = core.get_figure_bytes("svg")
        st.download_button(
            "⬇ Save PNG",
            data=_png,
            file_name="geometry_forge.png",
            mime="image/png",
            width="stretch",
            key="dl_png",
        )
        st.download_button(
            "⬇ Save SVG",
            data=_svg,
            file_name="geometry_forge.svg",
            mime="image/svg+xml",
            width="stretch",
            key="dl_svg",
        )


# ── Main canvas ───────────────────────────────────────────────────────────────
fig = core.generate_figure()
ax = fig.axes[0] if fig.axes else None

# Nudge panel appears above the canvas, positioned under the selected label
if ax:
    _render_nudge_panel(fig, ax, capture_state)

# Render to PIL Image for click-capture component (BytesIO not accepted)
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=CANVAS_DPI)
buf.seek(0)
pil_img = _PILImage.open(buf)

coords = streamlit_image_coordinates(pil_img, key="canvas_click", use_column_width=True)
if coords and ax:
    pos = _pixel_to_data(coords["x"], coords["y"], fig, ax)
    if pos is not None and pos != st.session_state.last_click_pos:
        st.session_state.last_click_pos = pos
        ann = _find_nearest_label(pos[0], pos[1], ax)
        prev_ann = st.session_state.selected_annotation
        core.label_manager.builtin_selected = ann["key"] if ann and ann["type"] == "builtin" else None
        st.session_state.selected_annotation = ann
        if ann != prev_ann:
            st.rerun()

with st.expander("🐛 Debug info", expanded=False):
    st.write("**raw coords:**", coords)
    if coords and ax:
        pos_dbg = _pixel_to_data(coords["x"], coords["y"], fig, ax)
        st.write("**_pixel_to_data:**", pos_dbg)
        st.write("**ax xlim/ylim:**", ax.get_xlim(), ax.get_ylim())
        st.write("**ax position:**", ax.get_position().bounds)
    st.write("**auto_positions:**", {k: (round(v[0],3), round(v[1],3)) for k, v in core.label_manager.auto_positions.items()})
    st.write("**dim_label_bboxes:**", [(round(b[0],3), round(b[1],3), round(b[2],3), round(b[3],3)) for b in core._standalone_dim_label_bboxes])
    st.write("**selected_annotation:**", st.session_state.get("selected_annotation"))
    st.write("**last_click_pos:**", st.session_state.get("last_click_pos"))
