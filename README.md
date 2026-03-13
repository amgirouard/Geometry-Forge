# Geometry Forge

Interactive geometry drawing application built with Streamlit and Matplotlib.

## Running

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the app:

```bash
streamlit run streamlit_app.py
```

Or via the convenience launchers:

```bash
python run.py
# or
python -m geometry_forge
```

The app opens in your browser at `http://localhost:8501`.

## Project Structure

```
Geometry Forge/
├── streamlit_app.py               ← Streamlit UI entry point
├── run.py                         ← Convenience launcher
├── requirements.txt               ← Python dependencies
└── geometry_forge/                ← The package
    ├── __init__.py                ← Package setup (logging, DPI, public API)
    ├── __main__.py                ← Enables `python -m geometry_forge`
    ├── core.py                    ← Framework-agnostic drawing engine (GeometryCore)
    ├── models.py                  ← Data types, constants, shape configs
    ├── validators.py              ← Input validation logic
    ├── drawing.py                 ← Low-level drawing utilities and transforms
    ├── labels.py                  ← Label positioning and rendering
    ├── drawers.py                 ← All shape drawer classes
    └── controllers.py             ← State controllers (transforms, plot, history, sliders)
```

## Which File Do I Edit?

### I want to change the UI layout, sidebar controls, or add a new widget
**→ `streamlit_app.py`**

The full Streamlit UI lives here. Sidebar sections (in order): shape selection, shape sub-type, dimension mode, transforms, parameters, sliders, preset dimension lines, shape-label toggles, hash marks, composite controls, freeform annotations, appearance, actions/download. The main canvas calls `core.generate_figure()` and displays it with `st.pyplot()`.

### I want to change drawing logic, state, or add a new shape feature
**→ `geometry_forge/core.py`**

`GeometryCore` is the framework-agnostic drawing engine stored in `st.session_state`. It owns all application state (shape selection, params, transforms, annotations, composite shapes) as plain Python attributes and exposes:
- `generate_figure()` — top-level draw entry point, returns a `matplotlib.figure.Figure`
- `get_figure_bytes(fmt)` — PNG/SVG bytes for download buttons
- `_build_state_snapshot()` / `_apply_state()` — undo/redo serialisation
- All `_generate_*`, `_draw_*`, `_calc_dim_*`, `_apply_*` drawing helpers

### I want to change a constant, default value, or color
**→ `geometry_forge/models.py`**

`AppConstants` has all the magic numbers: default dimensions, colors, font sizes, offsets, snap thresholds, z-order layers, and canvas settings. Shape feature flags, `SHAPE_CAPABILITIES`, and `ShapeConfigProvider` (which maps shape names → `ShapeConfig`) are all here. Triangle sub-type configs live in `_build_triangle_sub_configs()`.

### I want to add a new shape or fix how an existing shape draws
**→ `geometry_forge/drawers.py`**

Every shape has its own drawer class (e.g. `RectangleDrawer`, `ConeDrawer`). To add a new shape:
1. Create a class that inherits from `ShapeDrawer` and decorate it with `@ShapeRegistry.register("My Shape")`
2. Implement `draw(ctx, transform, params)` — use `ctx.ax` to draw on the matplotlib axes
3. Add a `ShapeConfig` entry in `models.py` → `_build_shape_configs()`
4. Add it to `GeometryCore.shape_data` in `core.py`
5. Add preset dim line specs to `_SHAPE_PRESET_DIM_LINES` in `streamlit_app.py` if applicable

### I want to change how labels are positioned or rendered
**→ `geometry_forge/labels.py`**

`LabelManager` handles label text storage, visibility, custom positions, and geometry hints passed from drawers. `PolygonLabelMixin` and `RadialLabelMixin` are mixed into drawer classes to provide shape-specific label layouts. `ArrowMixin` handles arrow drawing for angle shapes.

### I want to change the congruence detection, right-angle markers, or hashmarks
**→ `geometry_forge/drawing.py`**

`SmartGeometryEngine` detects congruent sides and right angles. `DrawingUtilities` draws hashmarks, right-angle markers, and handles coordinate conversions. `GeometricRotation` manages the artist-level rotate/flip transform pipeline applied after drawing.

### I want to change input validation (e.g. radius/diameter rules)
**→ `geometry_forge/validators.py`**

`ShapeValidator` has all the reusable validation methods: positive checks, equality checks, mutual exclusivity, and radius/diameter consistency.

### I want to change how transforms, history, or sliders work
**→ `geometry_forge/controllers.py`**

- `TransformController` — flip/rotate state with `get_state()` / `set_state()`
- `ScaleManager` — slider values (aspect, slope, peak_offset, view_scale) as plain floats; `var(key)` returns a `_FloatProxy` with `.get()/.set()` for backward compatibility
- `PlotController` — wraps the matplotlib Figure/Axes; `refresh()` is a no-op (Streamlit calls `st.pyplot()`)
- `HistoryManager` — undo/redo stack with a `restoring()` context manager

### I want to add a new preset dimension line type
**→ `geometry_forge/core.py`** + **`streamlit_app.py`**

1. In `core.py`, add a `_calc_dim_<key>` method and register it in `_dim_dispatch` inside `_calc_dim_line_endpoints()`
2. In `streamlit_app.py`, add `("<key>", "Label", "default")` to `_SHAPE_PRESET_DIM_LINES` for the relevant shape(s)

## How Imports Work

Files import from each other using relative imports:

```python
from .models import AppConstants, DrawingContext
from .drawing import DrawingUtilities
from .controllers import TransformController, ScaleManager
from .core import GeometryCore
```

The dot (`.`) means "from this same package." The `__init__.py` file is what makes the folder a Python package.

## Dependencies

- Python 3.10+
- `streamlit >= 1.30.0`
- `matplotlib >= 3.7.0`
- `numpy >= 1.24.0`
