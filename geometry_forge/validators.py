from __future__ import annotations

import math

from .models import ValidationError


class ShapeValidator:
    """Centralized validation for shape parameters."""
    
    @staticmethod
    def validate_positive(name: str, value: float | None) -> str | None:
        """Check if value is positive. Returns error message or None."""
        if value is None:
            return None
        if not math.isfinite(value):
            return f"{name} must be a finite number"
        if value <= 0:
            return f"{name} must be positive"
        return None
    
    @staticmethod
    def validate_equal(name1: str, val1: float | None, 
                       name2: str, val2: float | None,
                       tolerance: float = 0.001) -> str | None:
        """Check if two values are equal. Returns error message or None."""
        if val1 is not None and not math.isfinite(val1):
            return f"{name1} must be a finite number"
        if val2 is not None and not math.isfinite(val2):
            return f"{name2} must be a finite number"
        if val1 is not None and val1 > 0 and val2 is not None and val2 > 0:
            if abs(val1 - val2) > tolerance:
                return f"{name1} and {name2} must be equal"
        return None
    
    @staticmethod
    def validate_all_equal(values: dict[str, float | None], 
                          tolerance: float = 0.001) -> str | None:
        """Check if all provided values are equal. Returns error message or None."""
        for k, v in values.items():
            if v is not None and not math.isfinite(v):
                return f"{k} must be a finite number"
        numeric_values = [(k, v) for k, v in values.items() 
                         if v is not None and v > 0]
        if len(numeric_values) > 1:
            first_val = numeric_values[0][1]
            for name, val in numeric_values[1:]:
                if abs(val - first_val) > tolerance:
                    return f"All values must be equal ({', '.join(values.keys())})"
        return None
    
    @staticmethod
    def validate_mutually_exclusive(group1_name: str, group1_has: bool,
                                    group2_name: str, group2_has: bool) -> str | None:
        """Check that only one of two groups has values. Returns error message or None."""
        if group1_has and group2_has:
            return f"Enter {group1_name} OR {group2_name}, not both"
        return None
    
    @staticmethod
    def validate_diameter_radius(radius: str | None, diameter: str | None, 
                                  default_radius: float = 3.0) -> tuple[float, str | None]:
        """Validate radius/diameter from string inputs (e.g. tk.StringVar.get()).
        Returns (radius, error_message)."""
        try:
            r_val = float(radius) if radius not in [None, ""] else None
            d_val = float(diameter) if diameter not in [None, ""] else None
        except (ValueError, TypeError):
            return default_radius, "Radius and Diameter must be valid numbers"

        if r_val is not None and not math.isfinite(r_val):
            return default_radius, "Radius must be a finite number"
        if d_val is not None and not math.isfinite(d_val):
            return default_radius, "Diameter must be a finite number"

        if r_val is not None and r_val <= 0:
            return default_radius, "Radius must be positive"
        if d_val is not None and d_val <= 0:
            return default_radius, "Diameter must be positive"
        
        has_radius = r_val is not None and r_val > 0
        has_diameter = d_val is not None and d_val > 0
        
        if has_radius and has_diameter:
            if abs(d_val - (r_val * 2)) > 0.001:
                return default_radius, "Diameter must be exactly 2x the radius"
        
        if has_diameter:
            return d_val / 2, None
        if has_radius:
            return r_val, None
        
        return default_radius, None

