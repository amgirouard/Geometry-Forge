"""Geometry Forge — interactive geometry drawing application."""

from __future__ import annotations

import logging

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

# Enable High-DPI scaling for Windows (no-op on macOS/Linux)
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

# Public API
from .models import (
    Point, Polygon, DimLine, StandaloneDimLine, CompositeDimLine,
    AppConstants, ShapeFeature, ShapeConfig, ShapeConfigProvider,
    TriangleType, PolygonType, DrawingContext, TransformState,
)
from .drawing import DrawingUtilities, SmartGeometryEngine, GeometricRotation
from .labels import LabelManager
from .drawers import ShapeDrawer, ShapeRegistry
from .controllers import (
    TransformController, PlotController,
    HistoryManager, ScaleManager,
)
from .core import GeometryCore

__all__ = [
    "GeometryCore",
    "ShapeDrawer",
    "ShapeRegistry",
    "AppConstants",
    "TriangleType",
    "ShapeConfig",
    "ShapeConfigProvider",
    "ShapeFeature",
    "Point",
    "Polygon",
    "DimLine",
    "StandaloneDimLine",
    "CompositeDimLine",
    "TransformController",
    "PlotController",
    "HistoryManager",
    "ScaleManager",
    "LabelManager",
    "DrawingUtilities",
    "SmartGeometryEngine",
    "GeometricRotation",
    "TransformState",
    "DrawingContext",
]
