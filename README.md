# Geometry Forge

Interactive geometry drawing application built with Tkinter and Matplotlib.

## Running

```bash
python run.py
```

Or as a package:

```bash
python -m geometry_forge
```

## Project Structure
```
Geometry Forge/
├── run.py                             ← Launch the app
└── geometry_forge/                    ← The package
    ├── __init__.py                    ← Package setup (logging, DPI, public API)
    ├── __main__.py                    ← Enables `python -m geometry_forge`
    ├── models.py                      ← Data types, constants, shape configs
    ├── validators.py                  ← Input validation logic
    ├── drawing.py                     ← Low-level drawing utilities and transforms
    ├── labels.py                      ← Label positioning and rendering
    ├── drawers.py                     ← All shape drawer classes
    ├── controllers.py                 ← UI controllers (input, plot, history, sliders)
    ├── forge_widgets.py               ← Shared styled UI widgets (buttons, entries, sliders, etc.)
    ├── widgets.py                     ← App-specific Tkinter widgets
    ├── composite_controller.py        ← Multi-shape composite canvas controller
    ├── standalone_controller.py       ← Single-shape annotation controller
    └── app.py                         ← Main application class (GeometryApp)
```

## Which File Do I Edit?

### I want to change a constant, default value, or color
**→ `models.py`**

`AppConstants` has all the magic numbers: default dimensions, colors, font sizes, offsets, snap thresholds, z-order layers, and canvas settings. Shape feature flags and the `SHAPE_CAPABILITIES` table are here too.

### I want to add a new shape or fix how an existing shape draws
**→ `drawers.py`**

Every shape has its own drawer class (e.g. `RectangleDrawer`, `ConeDrawer`). To add a new shape, create a class that inherits from `ShapeDrawer`, decorate it with `@ShapeRegistry.register("My Shape")`, and implement the `draw()` method. You'll also need to add a config entry in `models.py` inside `_build_shape_configs()`.

### I want to change how labels are positioned or rendered
**→ `labels.py`**

`LabelManager` handles label text storage, custom positions, and rendering. `PolygonLabelMixin` and `RadialLabelMixin` are mixed into drawer classes to provide shape-specific label layouts. `ArrowMixin` handles arrow drawing for angle shapes.

### I want to change the congruence detection, right-angle markers, or hashmarks
**→ `drawing.py`**

`SmartGeometryEngine` detects congruent sides and right angles. `DrawingUtilities` draws hashmarks, right-angle markers, and handles coordinate conversions. `GeometricRotation` manages the artist-level rotate/flip transform pipeline.

### I want to change input validation (e.g. radius/diameter rules)
**→ `validators.py`**

`ShapeValidator` has all the reusable validation methods: positive checks, equality checks, mutual exclusivity, and radius/diameter consistency.

### I want to change the UI layout, menus, buttons, or keyboard shortcuts
**→ `app.py`**

`GeometryApp` owns the full Tkinter layout: the shape selector, input panel, toolbar, canvas setup, options panel, save/copy, and keyboard bindings. This is the biggest file and the most likely target for UI polish work.

### I want to change how the input fields or sliders work
**→ `controllers.py`**

`InputController` manages entry field creation, debounced updates, and mutual exclusivity clearing. `TransformController` handles flip/rotate state. `ScaleManager` owns all slider variables and their ranges. `PlotController` coordinates drawing and canvas refreshes. `HistoryManager` handles undo/redo.

### I want to change the look of buttons, entries, dropdowns, sliders, or checkboxes
**→ `forge_widgets.py`**

All Canvas-drawn UI primitives live here and are shared across the Forge app family. The full widget catalog:

| Class | Replaces | Notes |
|---|---|---|
| `_StyledButton` | `tk.Button` | Rounded, drop-shadow, hover/active/disabled states |
| `_StyledEntry` | `tk.Entry` | Rounded border, focus ring, same `.get()/.insert()/.delete()` API |
| `_StyledStepper` | `tk.Spinbox` | `[− value +]` in one unified Canvas rect |
| `_StyledCombobox` | `ttk.Combobox` | Custom popup, fires `<<ComboboxSelected>>` |
| `_ColorSwatchButton` | — | `_StyledButton` variant with a solid color fill and selection dot |
| `_StyledSlider` | `ttk.Scale` | iOS-style thumb, optional center tick, `.get()/.set()/variable=` |
| `_StyledCheckbox` | `tk.Checkbutton` | Rounded square, macOS-style checkmark, `.get()/.set()/variable=` |

`BTN_H` (default `20`) is a single constant that controls the height of every widget at once. The `_ForgeTheme` dataclass is the single source of truth for all colors and radii — pass a custom instance to any widget to override the theme.

This file has **zero app-specific dependencies** (no imports from `models`, `controllers`, etc.) so it can be dropped into any Forge package as-is.

### I want to change composite mode behavior (dragging, snapping, multi-shape canvas)
**→ `composite_controller.py`**

`CompositeDragController` owns all composite state: shape positions, transforms, selection, labels, dimension lines, drag/marquee, and snap guides. All three mouse event handlers for composite mode live here.

### I want to change standalone annotation behavior (labels, dimension lines on single shapes)
**→ `standalone_controller.py`**

`StandaloneAnnotationController` owns freeform labels, dimension lines, preset snap lines, selection state, and the four mouse event handlers for the standalone (non-composite) canvas.

### I want to change the composite transfer list widget
**→ `widgets.py`**

`CompositeTransferList` is the source → destination list widget with drag-to-reorder support used in composite mode.

## How Imports Work
Files import from each other using relative imports:

```python
from .models import AppConstants, DrawingContext
from .drawing import DrawingUtilities
from .forge_widgets import BTN_H, _StyledButton, _StyledEntry
```

The dot (`.`) means "from this same package." The `__init__.py` file is what makes the folder a Python package.

## Dependencies
- Python 3.10+
- `tkinter` (usually bundled with Python)
- `matplotlib`
- `numpy`
