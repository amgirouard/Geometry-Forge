"""streamlit_app.py — Geometry Forge Streamlit UI entry point.

Run with:  streamlit run streamlit_app.py
"""
from __future__ import annotations

import streamlit as st

from geometry_forge.core import GeometryCore
from geometry_forge.models import (
    AppConstants, ShapeConfigProvider, ShapeFeature,
    TriangleType, PolygonType,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Geometry Forge",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helper functions ──────────────────────────────────────────────────────────

# Preset dim line definitions per shape.
# Each entry: (preset_key, display_label, default_text)
# preset_key must match a key in GeometryCore._dim_dispatch.
_SHAPE_PRESET_DIM_LINES: dict[str, list[tuple[str, str, str]]] = {
    "Rectangle": [
        ("height", "Height", "h"),
        ("width",  "Width",  "w"),
        ("side_v", "Right Side", "b"),
        ("side_h", "Bottom Side", "a"),
    ],
    "Square": [
        ("height", "Height", "a"),
        ("width",  "Width",  "a"),
        ("side_v", "Right Side", "a"),
        ("side_h", "Bottom Side", "a"),
    ],
    "Triangle_Custom": [
        ("height", "Height",     "h"),
        ("width",  "Base Width", "b"),
        ("side_l", "Left Side",  "c"),
        ("side_r", "Right Side", "a"),
    ],
    "Triangle_Isosceles": [
        ("height", "Height",     "h"),
        ("width",  "Base Width", "a"),
        ("side_l", "Left Side",  "b"),
        ("side_r", "Right Side", "b"),
    ],
    "Triangle_Scalene": [
        ("height", "Height",     "h"),
        ("width",  "Base Width", "a"),
        ("side_l", "Left Side",  "b"),
        ("side_r", "Right Side", "c"),
    ],
    "Triangle_Equilateral": [
        ("height", "Height",     "h"),
        ("width",  "Base Width", "a"),
        ("side_l", "Left Side",  "a"),
        ("side_r", "Right Side", "a"),
    ],
    "Triangle_Right": [
        ("leg_a", "Leg A",       "a"),
        ("leg_b", "Leg B",       "b"),
        ("hyp",   "Hypotenuse",  "c"),
        ("height","Height",      "h"),
    ],
    "Parallelogram": [
        ("para_height", "Height",     "h"),
        ("para_base",   "Base",       "a"),
        ("para_top",    "Top",        "a"),
        ("para_side_l", "Left Side",  "b"),
        ("para_side_r", "Right Side", "b"),
    ],
    "Trapezoid": [
        ("trap_height", "Height",   "h"),
        ("trap_base",   "Bottom",   "b"),
        ("trap_top",    "Top",      "a"),
        ("trap_side_l", "Left Leg", "c"),
        ("trap_side_r", "Right Leg","d"),
    ],
    "Circle": [
        ("radius",        "Radius",        "r"),
        ("diameter",      "Diameter",      "d"),
        ("circumference", "Circumference", "C"),
    ],
    "Sphere": [
        ("radius",        "Radius",        "r"),
        ("diameter",      "Diameter",      "d"),
        ("circumference", "Circumference", "C"),
    ],
    "Hemisphere": [
        ("radius",   "Radius",   "r"),
        ("diameter", "Diameter", "d"),
    ],
    "Cylinder": [
        ("radius",        "Radius",        "r"),
        ("diameter",      "Diameter",      "d"),
        ("height",        "Height",        "h"),
        ("circumference", "Circumference", "C"),
    ],
    "Cone": [
        ("radius",   "Radius",       "r"),
        ("diameter", "Diameter",     "d"),
        ("height",   "Height",       "h"),
        ("slant",    "Slant Height", "l"),
    ],
    "Rectangular Prism": [
        ("height", "Height", "h"),
        ("length", "Length", "l"),
        ("width",  "Width",  "w"),
    ],
    "Triangular Prism": [
        ("height",     "Height",         "h"),
        ("tri_base",   "Triangle Base",  "b"),
        ("tri_length", "Prism Length",   "l"),
    ],
    "Polygon": [
        ("side", "Side Length", "a"),
    ],
    "Line Segment": [
        ("width", "Length", "l"),
    ],
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
    """Render the preset dimension line checkboxes + label text inputs."""
    specs = _get_preset_dim_lines(shape, core.triangle_type)
    if not specs:
        return
    st.subheader("Dimension Lines")
    changed = False
    for preset_key, display_label, default_text in specs:
        idx = _preset_exists(core, preset_key)
        cur_checked = idx >= 0
        cur_text = core.standalone_dim_lines[idx]["text"] if cur_checked else default_text

        col_chk, col_txt = st.columns([1, 2])
        with col_chk:
            new_checked = st.checkbox(
                display_label,
                value=cur_checked,
                key=f"preset_dl_chk_{shape}_{preset_key}",
            )
        with col_txt:
            new_text = st.text_input(
                "Label",
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

    # Composite freeform labels
    st.markdown("**Labels**")
    for idx, lbl in enumerate(list(core.composite_labels)):
        ca, cb = st.columns([3, 1])
        with ca:
            st.caption(
                f'"{lbl.get("text", "")}" '
                f'@ ({lbl.get("x", 0):.1f}, {lbl.get("y", 0):.1f})'
            )
        with cb:
            if st.button("✕", key=f"del_comp_lbl_{idx}"):
                capture_fn()
                core.composite_labels.pop(idx)
                st.rerun()

    with st.expander("Add Label", expanded=False):
        cl_text = st.text_input("Text", key="new_comp_lbl_text")
        cl_x = st.number_input("X", value=0.0, step=0.5, key="new_comp_lbl_x")
        cl_y = st.number_input("Y", value=0.0, step=0.5, key="new_comp_lbl_y")
        if st.button("Add Label", key="btn_add_comp_lbl") and cl_text:
            capture_fn()
            core.composite_labels.append({"text": cl_text, "x": cl_x, "y": cl_y})
            st.rerun()

    # Composite dim lines
    st.markdown("**Dimension Lines**")
    for idx, dl in enumerate(list(core.composite_dim_lines_list)):
        da, db = st.columns([3, 1])
        with da:
            st.caption(
                f'"{dl.get("text", "")}" '
                f'({dl.get("x1", 0):.1f},{dl.get("y1", 0):.1f})'
                f'→({dl.get("x2", 0):.1f},{dl.get("y2", 0):.1f})'
            )
        with db:
            if st.button("✕", key=f"del_comp_dl_{idx}"):
                capture_fn()
                core.composite_dim_lines_list.pop(idx)
                st.rerun()

    with st.expander("Add Dimension Line", expanded=False):
        cdl_text = st.text_input("Label", key="new_comp_dl_text")
        cdl_c1, cdl_c2 = st.columns(2)
        with cdl_c1:
            cdl_x1 = st.number_input("X1", value=0.0, step=0.5, key="new_comp_dl_x1")
            cdl_y1 = st.number_input("Y1", value=0.0, step=0.5, key="new_comp_dl_y1")
        with cdl_c2:
            cdl_x2 = st.number_input("X2", value=5.0, step=0.5, key="new_comp_dl_x2")
            cdl_y2 = st.number_input("Y2", value=0.0, step=0.5, key="new_comp_dl_y2")
        if st.button("Add Dim Line", key="btn_add_comp_dl"):
            capture_fn()
            core.composite_dim_lines_list.append({
                "text": cdl_text,
                "x1": cdl_x1, "y1": cdl_y1,
                "x2": cdl_x2, "y2": cdl_y2,
                "label_x": None, "label_y": None,
                "constraint": "free",
            })
            st.rerun()


def _render_annotation_controls(core: GeometryCore, capture_fn) -> None:
    """Render freeform label and dimension line controls for standalone mode."""
    has_annotations = bool(core.standalone_labels or core.standalone_dim_lines)
    if has_annotations:
        st.subheader("Annotations")
    else:
        # Collapsed expander when nothing is added yet
        with st.expander("Annotations", expanded=False):
            _annotation_form(core, capture_fn)
        return
    _annotation_form(core, capture_fn)


def _annotation_form(core: GeometryCore, capture_fn) -> None:
    """Shared annotation add/remove widgets."""
    # Existing freeform labels
    for idx, lbl in enumerate(list(core.standalone_labels)):
        la, lb = st.columns([3, 1])
        with la:
            st.caption(
                f'"{lbl.get("text", "")}" '
                f'@ ({lbl.get("x", 0):.1f}, {lbl.get("y", 0):.1f})'
            )
        with lb:
            if st.button("✕", key=f"del_sa_lbl_{idx}"):
                capture_fn()
                core.standalone_labels.pop(idx)
                st.rerun()

    with st.expander("Add Label", expanded=False):
        sa_text = st.text_input("Text", key="new_sa_lbl_text")
        sa_x = st.number_input("X", value=0.0, step=0.5, key="new_sa_lbl_x")
        sa_y = st.number_input("Y", value=0.0, step=0.5, key="new_sa_lbl_y")
        if st.button("Add Label", key="btn_add_sa_lbl") and sa_text:
            capture_fn()
            core.standalone_labels.append({"text": sa_text, "x": sa_x, "y": sa_y})
            st.rerun()

    # Existing dim lines
    for idx, dl in enumerate(list(core.standalone_dim_lines)):
        da, db = st.columns([3, 1])
        with da:
            st.caption(
                f'"{dl.get("text", "")}" '
                f'({dl.get("x1", 0):.1f},{dl.get("y1", 0):.1f})'
                f'→({dl.get("x2", 0):.1f},{dl.get("y2", 0):.1f})'
            )
        with db:
            if st.button("✕", key=f"del_sa_dl_{idx}"):
                capture_fn()
                core.standalone_dim_lines.pop(idx)
                st.rerun()

    with st.expander("Add Dimension Line", expanded=False):
        sa_dl_text = st.text_input("Label", key="new_sa_dl_text")
        dl_c1, dl_c2 = st.columns(2)
        with dl_c1:
            sa_dl_x1 = st.number_input("X1", value=0.0, step=0.5, key="new_sa_dl_x1")
            sa_dl_y1 = st.number_input("Y1", value=0.0, step=0.5, key="new_sa_dl_y1")
        with dl_c2:
            sa_dl_x2 = st.number_input("X2", value=5.0, step=0.5, key="new_sa_dl_x2")
            sa_dl_y2 = st.number_input("Y2", value=0.0, step=0.5, key="new_sa_dl_y2")
        if st.button("Add Dim Line", key="btn_add_sa_dl"):
            capture_fn()
            core.standalone_dim_lines.append({
                "text": sa_dl_text,
                "x1": sa_dl_x1, "y1": sa_dl_y1,
                "x2": sa_dl_x2, "y2": sa_dl_y2,
                "label_x": None, "label_y": None,
                "preset_key": None,
                "user_dragged": False,
                "constraint": None,
            })
            st.rerun()


# ── Session state bootstrap ───────────────────────────────────────────────────
if "core" not in st.session_state:
    st.session_state.core = GeometryCore()
if "history" not in st.session_state:
    st.session_state.history: list[dict] = []
if "redo_stack" not in st.session_state:
    st.session_state.redo_stack: list[dict] = []

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

    shape = core.shape_name
    config = ShapeConfigProvider.get(shape) if shape else None
    is_composite = core._is_composite_shape(shape)

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
        )
        if core.triangle_type != new_tri:
            capture_state()
            core.triangle_type = new_tri
            core.params = {}
            core.scale_manager.reset_many(["aspect", "slope", "peak_offset"])
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
    if config and config.has_dimension_mode:
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

    # ── 4. Transforms ──────────────────────────────────────────────────────────
    if config and not is_composite and (
        config.has_feature(ShapeFeature.FLIP) or config.has_feature(ShapeFeature.ROTATE)
    ):
        st.subheader("Transforms")

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

            if rot_labels and 0 <= current_side < len(rot_labels):
                st.caption(f"Rotation: {rot_labels[current_side]}")
            else:
                st.caption(f"Rotation: position {current_side}")

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

        if st.button("Reset Transforms", key="btn_reset_xf", width="stretch"):
            capture_state()
            core.transform_controller.reset()
            st.rerun()

    # ── 5. Shape Parameters ────────────────────────────────────────────────────
    if config and not is_composite:
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
                new_val = st.text_input(
                    lbl,
                    value=current_val,
                    key=f"param_{shape}_{core.triangle_type}_{core.dimension_mode}_{lbl}",
                )
                if new_val != current_val:
                    params_changed = True
                    core.params[lbl] = new_val
            if params_changed:
                capture_state()

    # ── 6. Sliders ─────────────────────────────────────────────────────────────
    if config and not is_composite and shape:
        slider_config = config  # already triangle sub-config if applicable

        has_slider_shape = slider_config.has_feature(ShapeFeature.SLIDER_SHAPE)
        has_slider_slope = slider_config.has_feature(ShapeFeature.SLIDER_SLOPE)
        has_slider_peak = slider_config.has_feature(ShapeFeature.SLIDER_PEAK)

        if has_slider_shape or has_slider_slope or has_slider_peak:
            st.subheader("Adjust Shape")

        if has_slider_shape:
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

        if has_slider_slope:
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

        if has_slider_peak:
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

        # View scale is always available for standalone shapes
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

    # ── 7. Preset Dimension Lines ──────────────────────────────────────────────
    if shape and not is_composite:
        _render_preset_dim_lines(core, shape, capture_state)

    # ── 8. Shape-Label Toggles ────────────────────────────────────
    if shape and not is_composite and config:
        relevant_toggles = _get_relevant_toggle_keys(shape, config)
        if relevant_toggles:
            st.subheader("Dimension Labels")
            for lbl_key, st_key in relevant_toggles:
                cur_text = core.label_manager.label_texts.get(lbl_key, "")
                cur_vis = core.label_manager.label_visibility.get(lbl_key, False)

                tc, ti = st.columns([1, 2])
                with tc:
                    new_vis = st.checkbox(lbl_key, value=cur_vis,
                                          key=f"toggle_vis_{st_key}")
                with ti:
                    new_text = st.text_input(
                        "Label",
                        value=cur_text,
                        key=f"toggle_text_{st_key}",
                        label_visibility="collapsed",
                        placeholder=lbl_key,
                    )

                if new_vis != cur_vis or new_text != cur_text:
                    capture_state()
                    if new_vis or new_text:
                        core.label_manager.set_label_text(lbl_key, new_text, new_vis)
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

    # ── 11. Freeform Annotations (standalone mode only) ────────────────────────
    if shape and not is_composite:
        _render_annotation_controls(core, capture_state)

    # ── 12. Appearance ─────────────────────────────────────────────────────────
    st.subheader("Appearance")

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

    if st.button("🔄 Reset All", key="btn_reset_all", width="stretch"):
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
st.pyplot(fig, width="stretch")
