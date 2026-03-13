from __future__ import annotations

import math
from typing import Any, TypedDict, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum, StrEnum

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from .labels import LabelManager

# Type aliases for geometry primitives
Point = tuple[float, float]
Polygon = list[Point]

class DimLine(TypedDict, total=False):
    """Shared fields for all dimension lines.

    Both standalone and composite dim lines carry these fields.
    Callers must always supply x1/y1/x2/y2/text; the rest are optional
    and default to None / False when absent.
    label_x/label_y: None means midpoint fallback is used at render time.
    """
    x1: float
    y1: float
    x2: float
    y2: float
    text: str
    label_x: float | None
    label_y: float | None
    constraint: str | None  # "height", "width", or None


class StandaloneDimLine(DimLine, total=False):
    """A freeform or preset dimension line on the standalone canvas.

    Extends DimLine with standalone-only fields:
      preset_key:   None for freeform lines, key string for preset snap lines.
      user_dragged: True once the label has been manually repositioned.
    """
    preset_key: str | None
    user_dragged: bool


class CompositeDimLine(DimLine, total=False):
    """A dimension line on the composite canvas.

    Extends DimLine; constraint additionally accepts "free" (composite only).
    No extra fields beyond the base — subclassed for type-checker clarity
    and to allow composite-specific constraint values to be documented here.
    """



class ValidationError(Exception):
    """Raised when shape parameters are invalid."""
    pass


class AppConstants:
    # UI Settings
    WINDOW_TITLE: str = "Geometry Forge"

    # ── Scaling ────────────────────────────────────────────────────────────────
    # UI_SCALE is updated at runtime when the window is resized.
    # All pixel/font constants below are base (scale=1.0) values; use the
    # scaled_* class methods to get the live scaled value.
    UI_SCALE: float = 1.0

    # Base (unscaled) UI font sizes — do not use directly; call scaled_btn_font() etc.
    _BASE_UI_FONT_SIZE: int = 10
    _BASE_HEADER_FONT_SIZE: int = 8
    _BASE_CONTROLS_HEIGHT: int = 155
    _BASE_TOP_BAR_HEIGHT: int = 46
    _BASE_SHORTCUT_BAR_HEIGHT: int = 24

    @classmethod
    def scaled_ui_font_size(cls) -> int:
        return max(6, round(cls._BASE_UI_FONT_SIZE * cls.UI_SCALE))

    @classmethod
    def scaled_btn_font(cls) -> tuple:
        return ("Arial", max(6, round(cls._BASE_UI_FONT_SIZE * cls.UI_SCALE)))

    @classmethod
    def scaled_header_font(cls) -> tuple:
        return ("Arial", max(5, round(cls._BASE_HEADER_FONT_SIZE * cls.UI_SCALE)), "bold")

    @classmethod
    def scaled_controls_height(cls) -> int:
        return max(60, round(cls._BASE_CONTROLS_HEIGHT * cls.UI_SCALE))

    @classmethod
    def scaled_top_bar_height(cls) -> int:
        return max(28, round(cls._BASE_TOP_BAR_HEIGHT * cls.UI_SCALE))

    @classmethod
    def scaled_shortcut_bar_height(cls) -> int:
        return max(16, round(cls._BASE_SHORTCUT_BAR_HEIGHT * cls.UI_SCALE))
    # ── End scaling ────────────────────────────────────────────────────────────

    DEFAULT_FONT_SIZE: int = 12
    MIN_FONT_SIZE: int = 6
    MAX_FONT_SIZE: int = 48
    DEFAULT_FONT_FAMILY: str = "serif"
    BTN_FONT: tuple = ("Arial", 10)   # Standard button font — apply to all tk.Button widgets for consistency
    HEADER_FONT: tuple = ("Arial", 8, "bold")  # Section header font — apply to all column/panel headers
    
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
    ISOSCELES_DEFAULT_BASE: float = 3.0
    ISOSCELES_DEFAULT_HEIGHT: float = 5.0
    SCALENE_DEFAULT_BASE: float = 9.0
    RIGHT_DEFAULT_BASE: float = 4.0
    RIGHT_DEFAULT_LEG: float = 3.0   # vertical leg (l); hypotenuse = 5
    SCALENE_DEFAULT_L: float = 5.0
    SCALENE_DEFAULT_R: float = 7.0
    # Peak offset for default scalene (b=9, l=5, r=7): peak_x=(81+25-49)/18=3.17, peak_ref=3.17/9
    SCALENE_DEFAULT_PEAK: float = 0.351852
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
    PAPER_ASPECT_RATIO: float = 4.0 / 3.0  # Canvas drawing area aspect ratio (4:3)

    # Window layout — all heights in pixels; width is determined at runtime by
    # the menu bar's natural rendered width and never hardcoded.
    CONTROLS_HEIGHT: int = 155           # Fixed height for controls row (sized for triangle — tallest shape)
    TOP_BAR_HEIGHT: int = 46              # Category/Shape/Font selector bar
    SHORTCUT_BAR_HEIGHT: int = 24         # Keyboard shortcut hint bar
    CANVAS_PAPER_MARGIN: float = 0.04    # Grey border around white paper (fraction of figure)

    # All built-in dimension label keys that support selection, drag, and double-click edit.
    # Used in _on_canvas_press to distinguish built-in labels from freeform annotations.
    BUILTIN_LABEL_KEYS: frozenset[str] = frozenset({
        "Circumference", "Radius", "Diameter", "Height", "Slant",
        "Length (Front)", "Width (Side)",
        "Base (Tri)", "Height (Tri)", "Length (Prism)",
    })



class ShapeFeature(StrEnum):
    """Feature key constants for ShapeConfig.features dicts.

    Inherits from StrEnum so members compare equal to their string values
    and can be used directly as dict keys — no .value access needed.
    Typos at call sites are caught at import time rather than silently
    returning False from has_feature().
    """
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
    has_dimension_mode: bool = False  # Intentional: structural on/off flag, not a toggle feature — does not belong in ShapeFeature enum
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


class TriangleType(Enum):
    CUSTOM = "Custom"
    ISOSCELES = "Isosceles"
    SCALENE = "Scalene"
    EQUILATERAL = "Equilateral"
    RIGHT = "Right"


class PolygonType(Enum):
    PENTAGON = "Pentagon"
    HEXAGON = "Hexagon"
    OCTAGON = "Octagon"



@dataclass
class TransformState:
    """Holds current transformation state for shape drawing."""
    flip_h: bool = False
    flip_v: bool = False
    base_side: int = 0


@dataclass
class DrawingContext:
    """Holds common drawing parameters."""
    ax: Axes
    aspect_ratio: float
    font_size: int
    view_scale: float = 1.0
    composite_mode: bool = False  # True when drawing inside _draw_composite_shapes
    show_hashmarks: bool = False  # Mirrors show_hashmarks_var; drawers check this before drawing hashmarks
    line_color: str = field(init=False, default="black")
    line_args: dict = field(init=False)
    dash_args: dict = field(init=False)
    dim_args: dict = field(init=False)
    
    def __post_init__(self):
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
class DrawingDependencies:
    """Dependencies required for shape drawing, passed explicitly to drawers."""
    ax: Axes
    label_manager: LabelManager
    font_size: int



@dataclass(frozen=True)
class ScaleSpec:
    key: str
    min_val: float
    max_val: float
    default: float



# Central capability map — single source of truth for all shape feature sets.
# Edit here; all frozensets in SmartGeometryEngine and GeometricRotation are derived.
#   polygon:       has polygon vertices (used by SmartGeometryEngine.POLYGON_SHAPES)
#   geometric_2d:  2D shape using post-draw artist transform pipeline
#   geometric_3d:  3D shape using post-draw artist transform pipeline
#   arc_symmetric: radially symmetric — flip normalises to base_side rotation
SHAPE_CAPABILITIES: dict[str, dict[str, bool]] = {
    # 2D shapes
    "Rectangle":         {"polygon": True,  "geometric_2d": True,  "geometric_3d": False, "arc_symmetric": False},
    "Square":            {"polygon": True,  "geometric_2d": False, "geometric_3d": False, "arc_symmetric": False},
    "Parallelogram":     {"polygon": True,  "geometric_2d": True,  "geometric_3d": False, "arc_symmetric": False},
    "Trapezoid":         {"polygon": True,  "geometric_2d": True,  "geometric_3d": False, "arc_symmetric": False},
    "Triangle":          {"polygon": True,  "geometric_2d": True,  "geometric_3d": False, "arc_symmetric": False},
    "Polygon":           {"polygon": True,  "geometric_2d": False, "geometric_3d": False, "arc_symmetric": False},
    "Tri Triangle":      {"polygon": False, "geometric_2d": True,  "geometric_3d": False, "arc_symmetric": False},
    # Radially symmetric 2D/3D
    "Sphere":            {"polygon": False, "geometric_2d": False, "geometric_3d": False, "arc_symmetric": True},
    # 3D shapes
    "Hemisphere":        {"polygon": False, "geometric_2d": False, "geometric_3d": True,  "arc_symmetric": True},
    "Cylinder":          {"polygon": False, "geometric_2d": False, "geometric_3d": True,  "arc_symmetric": True},
    "Cone":              {"polygon": False, "geometric_2d": False, "geometric_3d": True,  "arc_symmetric": True},
    "Rectangular Prism": {"polygon": False, "geometric_2d": False, "geometric_3d": True,  "arc_symmetric": False},
    "Triangular Prism":  {"polygon": False, "geometric_2d": False, "geometric_3d": True,  "arc_symmetric": False},
    "Tri Prism":         {"polygon": False, "geometric_2d": False, "geometric_3d": True,  "arc_symmetric": False},
}


def _build_shape_configs() -> dict[str, "ShapeConfig"]:
    """Build and return the complete shape configuration mapping.

    Called exactly once at module load — result stored in _SHAPE_CONFIGS.
    ShapeConfigProvider delegates all lookups to that constant.
    """
    configs: dict[str, ShapeConfig] = {}

    # ── 2D shapes ────────────────────────────────────────────────────────────
    configs["Rectangle"] = ShapeConfig(
        labels=["Top", "Bottom", "Left", "Right"],
        default_values=["a", "a", "b", "b"],
        custom_values=["10", "5"],
        custom_labels=["Length", "Width"],
        features={ShapeFeature.SLIDER_SHAPE: True, ShapeFeature.FLIP: True, ShapeFeature.ROTATE: True},
        num_sides=4,
        has_dimension_mode=True,
        help_text="Default: Clean shape, add labels with presets.\nCustom: Enter Length and Width values."
    )
    configs["Square"] = ShapeConfig(
        labels=["Top", "Bottom", "Left", "Right"],
        default_values=["a", "a", "a", "a"],
        features={ShapeFeature.FLIP: False, ShapeFeature.ROTATE: False},
        has_dimension_mode=False,
        help_text="All sides equal. Labels only mode."
    )
    configs["Circle"] = ShapeConfig(
        labels=["Radius", "Diameter", "Circumference"],
        default_values=["r", "", ""],
        features={ShapeFeature.FLIP: False, ShapeFeature.ROTATE: False},
        has_dimension_mode=False,
        help_text="Defined by radius or diameter. Leave unused fields blank."
    )
    configs["Parallelogram"] = ShapeConfig(
        labels=["Top", "Bottom", "Left Side", "Right Side", "Height"],
        default_values=["a", "a", "b", "b", "h"],
        custom_values=["6", "4", ""],
        custom_labels=["Length", "Height", "Side"],
        features={ShapeFeature.SLIDER_SHAPE: True, ShapeFeature.SLIDER_SLOPE: True, ShapeFeature.FLIP: True, ShapeFeature.ROTATE: True},
        num_sides=4,
        has_dimension_mode=True,
        help_text="Enter Length + Height, OR Length + Side.\nCustom: Slope slider adjusts lean."
    )
    configs["Trapezoid"] = ShapeConfig(
        labels=["Top", "Bottom", "Left Leg", "Right Leg", "Height"],
        default_values=["a", "b", "c", "d", "h"],
        custom_values=["3", "6", "4", "", ""],
        custom_labels=["Top Base", "Bottom Base", "Height", "Left Side", "Right Side"],
        features={ShapeFeature.SLIDER_SHAPE: True, ShapeFeature.FLIP: True, ShapeFeature.ROTATE: True},
        num_sides=4,
        has_dimension_mode=True,
        help_text="Enter Top + Bottom + Height, OR Top + Bottom + both Sides.\nCustom: Numeric values control exact dimensions."
    )
    configs["Polygon"] = ShapeConfig(
        labels=["Side Length"],
        default_values=["a"],
        features={ShapeFeature.HASH_MARKS: True},
        has_dimension_mode=False,
        help_text="Regular polygon. Choose Pentagon, Hexagon, or Octagon using the type buttons."
    )
    # Triangle configurations — base shared between Triangle and Tri Triangle
    _tri_base = dict(
        labels=["Base Width", "Height", "Left Label", "Right Label"],
        default_values=["b", "h", "c", "a"],
        features={ShapeFeature.FLIP: True, ShapeFeature.ROTATE: True},
        has_dimension_mode=False,
    )
    configs["Triangle"] = ShapeConfig(
        **_tri_base,
        num_sides=3,
        help_text="Custom triangle defined by base and height."
    )
    configs["Tri Triangle"] = ShapeConfig(
        **_tri_base,
        num_sides=4,
        uses_base_side_flip=False,
        rotation_labels=["Base Down", "Left Side Down", "Base Up", "Right Side Down"],
        help_text="Triangle (composite). Rotates in 90° steps."
    )
    # Triangle sub-type configs live in _TRIANGLE_SUB_CONFIGS (see below)
    # so this dict only contains true shape names matching SHAPE_CAPABILITIES keys.

    # ── 3D shapes ────────────────────────────────────────────────────────────
    configs["Sphere"] = ShapeConfig(
        labels=["Radius", "Diameter", "Circumference"],
        default_values=["r", "", ""],
        has_dimension_mode=False,
        help_text="Defined by radius or diameter. Slider adjusts scale."
    )
    configs["Hemisphere"] = ShapeConfig(
        labels=["Radius", "Diameter", "Circumference"],
        default_values=["r", "", ""],
        features={ShapeFeature.SLIDER_SHAPE: True, ShapeFeature.FLIP: True, ShapeFeature.ROTATE: True},
        num_sides=4,
        rotation_labels=[],
        has_dimension_mode=False,
        help_text="Defined by radius or diameter. Adjust Shape slider changes dome proportions (squat/tall)."
    )
    configs["Cylinder"] = ShapeConfig(
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
    configs["Cone"] = ShapeConfig(
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
    configs["Rectangular Prism"] = ShapeConfig(
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
    configs["Triangular Prism"] = ShapeConfig(
        **_tri_prism_base,
        num_sides=3,
        rotation_labels=["Base Down", "Left Side Down", "Base Up"],
    )
    configs["Tri Prism"] = ShapeConfig(
        **_tri_prism_base,
        num_sides=4,
        uses_base_side_flip=False,
        rotation_labels=["Base Down", "Left Side Down", "Base Up", "Right Side Down"],
    )

    # ── Angle / line shapes ──────────────────────────────────────────────────
    configs["Angle (Adjustable)"] = ShapeConfig(
        labels=["Angle Label"],
        default_values=["x°"],
        features={ShapeFeature.SLIDER_SHAPE: True, ShapeFeature.FLIP: True},
        has_dimension_mode=False,
        help_text="Adjustable angle. Slider controls angle size (5° to 350°)."
    )
    configs["Parallel Lines & Transversal"] = ShapeConfig(
        labels=["Top Left", "Top Right", "Bottom Left", "Bottom Right"],
        default_values=["x°", "x°", "x°", "x°"],
        features={ShapeFeature.SLIDER_SHAPE: True, ShapeFeature.FLIP: True},
        has_dimension_mode=False,
        help_text="Parallel lines cut by transversal. Slider adjusts gap between parallel lines."
    )
    configs["Complementary Angles"] = ShapeConfig(
        labels=["Angle 1", "Angle 2"],
        default_values=["x°", "x°"],
        features={ShapeFeature.SLIDER_SHAPE: True, ShapeFeature.FLIP: True},
        has_dimension_mode=False,
        help_text="Two angles that sum to 90°. Slider adjusts split between angles."
    )
    configs["Supplementary Angles"] = ShapeConfig(
        labels=["Left Angle", "Right Angle"],
        default_values=["x°", "x°"],
        features={ShapeFeature.SLIDER_SHAPE: True, ShapeFeature.FLIP: True},
        has_dimension_mode=False,
        help_text="Two angles that sum to 180°. Slider adjusts split between angles."
    )
    configs["Vertical Angles"] = ShapeConfig(
        labels=["Top", "Bottom", "Left", "Right"],
        default_values=["x°", "x°", "x°", "x°"],
        features={ShapeFeature.SLIDER_SHAPE: True},
        has_dimension_mode=False,
        help_text="Opposite angles formed by two intersecting lines. Slider adjusts line angle."
    )
    configs["Line Segment"] = ShapeConfig(
        labels=["Point A", "Point B", "Point C", "Left Segment", "Right Segment"],
        default_values=["A", "B", "C", "", ""],
        features={ShapeFeature.SLIDER_SHAPE: True, ShapeFeature.FLIP: True},
        has_dimension_mode=False,
        help_text="Line segment with three labeled points. Slider moves point B along segment."
    )

    # ── Composite shapes ─────────────────────────────────────────────────────
    configs["2D Composite"] = ShapeConfig(
        labels=[],
        default_values=[],
        features={},
        has_dimension_mode=False,
        help_text="Add 2D shapes using the transfer list, then they appear on the canvas."
    )
    configs["3D Composite"] = ShapeConfig(
        labels=[],
        default_values=[],
        features={},
        has_dimension_mode=False,
        help_text="Add 3D shapes using the transfer list, then they appear on the canvas."
    )

    return configs


def _build_triangle_sub_configs() -> dict[str, "ShapeConfig"]:
    """Build triangle sub-type configs (Custom, Isosceles, Scalene, Equilateral).

    Kept separate from _SHAPE_CONFIGS so the main config dict only contains
    true shape names that match SHAPE_CAPABILITIES keys.
    ShapeConfigProvider.get_triangle_config() delegates lookups here.
    """
    configs: dict[str, ShapeConfig] = {}
    configs["Triangle_Custom"] = ShapeConfig(
        labels=["Base Width", "Height", "Left Side", "Right Side"],
        default_values=["b", "h", "c", "a"],
        custom_values=["5", "4", "", ""],
        custom_labels=["Base Width", "Height", "Left Side", "Right Side"],
        features={ShapeFeature.FLIP: True, ShapeFeature.ROTATE: True},
        num_sides=3,
        has_dimension_mode=True,
        help_text="Enter Base + Height OR Base + both Sides (mutually exclusive)."
    )
    configs["Triangle_Isosceles"] = ShapeConfig(
        labels=["Base", "Left", "Right", "Height"],
        default_values=["a", "b", "b", "h"],
        features={ShapeFeature.SLIDER_SHAPE: True, ShapeFeature.FLIP: True, ShapeFeature.ROTATE: True},
        num_sides=3,
        has_dimension_mode=False,
        help_text="Isosceles triangle with two equal sides. Slider adjusts proportions."
    )
    configs["Triangle_Scalene"] = ShapeConfig(
        labels=["Side A (Bottom)", "Side B (Left)", "Side C (Right)", "Height"],
        default_values=["a", "b", "c", "h"],
        features={ShapeFeature.FLIP: True, ShapeFeature.ROTATE: True, ShapeFeature.SLIDER_PEAK: True},
        num_sides=3,
        has_dimension_mode=False,
        help_text="Scalene triangle with all different sides. Slider adjusts peak offset."
    )
    configs["Triangle_Equilateral"] = ShapeConfig(
        labels=["Side A (Bottom)", "Side B (Left)", "Side C (Right)"],
        default_values=["a", "a", "a"],
        features={ShapeFeature.FLIP: True, ShapeFeature.ROTATE: True},
        num_sides=3,
        has_dimension_mode=False,
        help_text="Equilateral triangle with all equal sides."
    )
    configs["Triangle_Right"] = ShapeConfig(
        labels=["Leg A", "Leg B", "Hypotenuse"],
        default_values=["a", "b", "c"],
        features={ShapeFeature.SLIDER_SHAPE: True, ShapeFeature.FLIP: True, ShapeFeature.ROTATE: True},
        num_sides=3,
        has_dimension_mode=False,
        help_text="Right triangle with 90° angle at bottom-left. Slider adjusts base vs leg ratio."
    )
    return configs


# Module-level constants — built once at import time, never mutated.
_SHAPE_CONFIGS: dict[str, "ShapeConfig"] = _build_shape_configs()
_TRIANGLE_SUB_CONFIGS: dict[str, "ShapeConfig"] = _build_triangle_sub_configs()


class ShapeConfigProvider:
    """Read-only access to shape configurations.

    All configs are built once into the module-level _SHAPE_CONFIGS constant.
    This class is a thin lookup façade — it holds no mutable state.
    """

    @staticmethod
    def get(shape_name: str) -> "ShapeConfig":
        """Return config for *shape_name*. Returns empty ShapeConfig if not found."""
        return _SHAPE_CONFIGS.get(shape_name, ShapeConfig())

    @staticmethod
    def get_triangle_config(triangle_type: str) -> "ShapeConfig":
        """Return config for a specific triangle sub-type (Custom/Isosceles/Scalene/Equilateral)."""
        key = f"Triangle_{triangle_type}"
        return _TRIANGLE_SUB_CONFIGS.get(key, _SHAPE_CONFIGS.get("Triangle", ShapeConfig()))

    @staticmethod
    def has_dimension_mode(shape_name: str) -> bool:
        """Return True if *shape_name* supports dimension-mode switching."""
        return ShapeConfigProvider.get(shape_name).has_dimension_mode