"""forge_widgets.py — Shared styled UI widgets for the Forge app family.

Drop this file into any Forge package and import what you need:

    from .forge_widgets import (
        BTN_H,
        _StyledButton,
        _StyledEntry,
        _StyledStepper,
        _StyledCombobox,
        _ColorSwatchButton,
        _StyledSlider,
        _StyledCheckbox,
    )

Every class is a tk.Canvas (or tk.Frame for _StyledCheckbox) subclass with
zero dependencies outside the standard library + tkinter, so they work in
Fraction Forge, Geometry Forge, Algebra Forge, or any future Forge app.

Design tokens are read from a _ForgeTheme dataclass so you can override the
palette in one place if a future app needs a different accent color.

Compatibility guarantee
-----------------------
_StyledButton   — API-compatible with tk.Button
                  Extra: .set_active(bool), .set_disabled(bool)
_StyledEntry    — API-compatible with tk.Entry
                  Extra: focus ring on FocusIn/FocusOut
_StyledStepper  — API-compatible with tk.Spinbox
                  Extra: [− value +] zones, embedded entry for typing
_StyledCombobox — API-compatible with ttk.Combobox (readonly mode)
                  Extra: custom popup, label= placeholder text
_ColorSwatchButton — _StyledButton subclass; .set_selected(bool)
_StyledSlider   — API-compatible with ttk.Scale
                  Extra: .get(), .set(), show_center_tick=, command=
_StyledCheckbox — API-compatible with tk.Checkbutton
                  Extra: rounded square box, consistent cross-platform look
"""

from __future__ import annotations

import math
import tkinter as tk
from dataclasses import dataclass
from typing import Callable


# ═══════════════════════════════════════════════════════════════════════════════
# Theme tokens — override here if a future app needs a different palette
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class _ForgeTheme:
    normal_fill:   str = "#ffffff"
    normal_border: str = "#d8d8d8"
    hover_fill:    str = "#eef4fc"
    hover_border:  str = "#7aaed4"
    active_fill:   str = "#C0D8F2"
    active_border: str = "#7aaed4"
    disabled_fill: str = "#f0f0f0"
    disabled_text: str = "#aaaaaa"
    text_color:    str = "#1a1a1a"
    shadow_color:  str = "#d0d0d0"
    focus_ring:    str = "#a8c8e8"
    track_color:   str = "#c8c8c8"
    check_color:   str = "#1a5a8a"   # checkmark stroke inside filled box
    radius:        int = 5


# Singleton used by all widgets — replace with _ForgeTheme(active_fill="…")
# at the top of any app that needs a different accent.
THEME = _ForgeTheme()

# ── Global height constant ────────────────────────────────────────────────────
# Change BTN_H to resize ALL Forge widgets at once.
BTN_H: int = 20


# ═══════════════════════════════════════════════════════════════════════════════
# Internal drawing helper — used by all Canvas-based widgets
# ═══════════════════════════════════════════════════════════════════════════════

def _rounded_rect(canvas: tk.Canvas, x1: float, y1: float,
                  x2: float, y2: float, r: int, **kwargs) -> int:
    """Draw a filled/outlined rounded rectangle on *canvas* via polygon arcs.

    Returns the canvas item id of the polygon.
    All kwargs are forwarded to canvas.create_polygon().
    """
    points: list[float] = []
    for cx, cy, a0, a1 in [
        (x2 - r, y1 + r, -90,   0),
        (x2 - r, y2 - r,   0,  90),
        (x1 + r, y2 - r,  90, 180),
        (x1 + r, y1 + r, 180, 270),
    ]:
        for i in range(9):
            a = math.radians(a0 + (a1 - a0) * i / 8)
            points += [cx + r * math.cos(a), cy + r * math.sin(a)]
    return canvas.create_polygon(points, smooth=False, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# _StyledButton
# ═══════════════════════════════════════════════════════════════════════════════

class _StyledButton(tk.Canvas):
    """Canvas-drawn button — consistent look on all platforms.

    Matches macOS native appearance: white fill, rounded corners, drop shadow.
    Active state uses the app's blue accent. Hover tints slightly.

    Usage::

        btn = _StyledButton(parent, text="Save", command=save_fn,
                            width=60, height=BTN_H)
        btn.set_active(True)     # blue highlight
        btn.set_disabled(True)   # greyed out, no cursor
    """

    def __init__(self, parent: tk.Widget, text: str = "",
                 command: Callable | None = None, *,
                 width: int = 80, height: int = BTN_H,
                 font: tuple | None = None,
                 active: bool = False,
                 bg: str | None = None,
                 theme: _ForgeTheme = THEME,
                 **kwargs):
        super().__init__(parent, width=width, height=height,
                         highlightthickness=0,
                         bg=bg or parent.cget("bg"),
                         cursor="hand2", **kwargs)
        self._text     = text
        self._command  = command
        self._font     = font or ("Arial", 10)
        self._active   = active
        self._hover    = False
        self._disabled = False
        self._theme    = theme

        self.bind("<Enter>",           self._on_enter)
        self.bind("<Leave>",           self._on_leave)
        self.bind("<ButtonPress-1>",   self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self._draw()

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_active(self, active: bool) -> None:
        """Toggle the blue active/selected highlight."""
        self._active = active
        self._draw()

    def set_disabled(self, disabled: bool) -> None:
        """Grey out the button and remove the pointer cursor."""
        self._disabled = disabled
        self.config(cursor="" if disabled else "hand2")
        self._draw()

    def configure(self, **kwargs) -> None:
        """Support text=, state=, font=, command= — same as tk.Button."""
        if "text" in kwargs:
            self._text = kwargs.pop("text")
        if "state" in kwargs:
            state = kwargs.pop("state")
            self._disabled = (state == "disabled")
            self.config(cursor="" if self._disabled else "hand2")
        if "font" in kwargs:
            self._font = kwargs.pop("font")
        if "command" in kwargs:
            self._command = kwargs.pop("command")
        # Silently drop legacy tk.Button-only kwargs
        for _drop in ("relief", "fg", "pady", "bd"):
            kwargs.pop(_drop, None)
        if kwargs:
            super().configure(**kwargs)
        self._draw()

    # Allow .config() as alias — mirrors tk.Button convention
    config = configure

    # ── Drawing ────────────────────────────────────────────────────────────────

    def _draw(self) -> None:
        self.delete("all")
        t = self._theme
        w = int(self.cget("width"))
        h = int(self.cget("height"))

        if self._disabled:
            fill, border, shadow = t.disabled_fill, "#ebebeb", False
            text_color           = t.disabled_text
        elif self._active:
            fill, border, shadow = t.active_fill, t.active_border, True
            text_color           = t.text_color
        elif self._hover:
            fill, border, shadow = t.hover_fill, t.hover_border, True
            text_color           = t.text_color
        else:
            fill, border, shadow = t.normal_fill, t.normal_border, True
            text_color           = t.text_color

        if shadow:
            _rounded_rect(self, 3, 3, w - 1, h - 1, t.radius,
                          fill=t.shadow_color, outline="")
        _rounded_rect(self, 1, 1, w - 3, h - 3, t.radius,
                      fill=fill, outline=border, width=1)
        self.create_text(w // 2, (h - 2) // 2, text=self._text,
                         font=self._font, fill=text_color, anchor="center")

    # ── Events ─────────────────────────────────────────────────────────────────

    def _on_enter(self, _=None) -> None:
        if not self._disabled:
            self._hover = True
            self._draw()

    def _on_leave(self, _=None) -> None:
        self._hover = False
        self._draw()

    def _on_press(self, _=None) -> None:
        self.winfo_toplevel().focus_set()
        if not self._disabled and self._command:
            self._command()

    def _on_release(self, _=None) -> None:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# _ColorSwatchButton  (subclass of _StyledButton)
# ═══════════════════════════════════════════════════════════════════════════════

class _ColorSwatchButton(_StyledButton):
    """_StyledButton variant that displays a colour swatch instead of text.

    Normal: swatch fill + drop shadow.
    Selected: swatch fill + small dark dot in centre.
    Hover: slightly lighter swatch fill.

    Usage::

        swatch = _ColorSwatchButton(parent,
                                    shade_color="#4A90D9",
                                    display_color="#9BC2EA",
                                    command=lambda: set_color("#4A90D9"))
        swatch.set_selected(True)
    """

    def __init__(self, parent: tk.Widget,
                 shade_color: str, display_color: str,
                 command: Callable | None = None, *,
                 width: int = 28, height: int = 28,
                 bg: str | None = None,
                 theme: _ForgeTheme = THEME, **kwargs):
        self._shade_color   = shade_color
        self._display_color = display_color
        self._selected      = False
        super().__init__(parent, text="", command=command,
                         width=width, height=height,
                         bg=bg, theme=theme, **kwargs)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._active   = selected
        self._draw()

    def _draw(self) -> None:
        self.delete("all")
        t = self._theme
        w = int(self.cget("width"))
        h = int(self.cget("height"))
        _rounded_rect(self, 3, 3, w - 1, h - 1, t.radius,
                      fill="#c8c8c8", outline="")
        _rounded_rect(self, 1, 1, w - 3, h - 3, t.radius,
                      fill=self._display_color, outline="")
        if self._selected:
            cx, cy, dot_r = w // 2, (h - 2) // 2, 3
            self.create_oval(cx - dot_r, cy - dot_r,
                             cx + dot_r, cy + dot_r,
                             fill="#333333", outline="")


# ═══════════════════════════════════════════════════════════════════════════════
# _StyledEntry
# ═══════════════════════════════════════════════════════════════════════════════

class _StyledEntry(tk.Canvas):
    """tk.Entry wrapped in a Canvas border with rounded corners and a focus ring.

    The inner tk.Entry is embedded via create_window so all native text-editing
    behaviour (cursor, selection, copy/paste, IME) works unchanged.

    Supports .get(), .insert(), .delete(), textvariable=, .bind(),
    .grid(), .grid_remove() — same interface as tk.Entry.

    Usage::

        entry = _StyledEntry(parent, width=6, font=("Arial", 10))
        entry.bind("<Return>", on_submit)
        value = entry.get()
    """

    _PAD = 2   # pixels between canvas edge and inner Entry

    def __init__(self, parent: tk.Widget, *,
                 textvariable: tk.Variable | None = None,
                 width: int = 6,
                 font: tuple | None = None,
                 bg: str | None = None,
                 justify: str = "left",
                 height: int | None = None,
                 theme: _ForgeTheme = THEME,
                 **kwargs):
        bg = bg or parent.cget("bg")
        self._font   = font or ("Arial", 10)
        self._focused = False
        self._theme  = theme

        char_w   = 8           # approximate pixels per character at size 10
        inner_w  = width * char_w
        canvas_h = height if height is not None else BTN_H
        inner_h  = canvas_h - self._PAD * 2 - 2
        canvas_w = inner_w + self._PAD * 2 + 2

        super().__init__(parent, width=canvas_w, height=canvas_h,
                         highlightthickness=0, bg=bg, **kwargs)

        self._canvas_w = canvas_w
        self._canvas_h = canvas_h
        self._inner_w  = inner_w
        self._inner_h  = inner_h

        t = self._theme
        self._entry = tk.Entry(
            self, textvariable=textvariable,
            font=self._font, relief="flat", bd=0,
            bg=t.normal_fill, fg=t.text_color,
            insertbackground=t.text_color,
            selectbackground=t.active_fill,
            selectforeground=t.text_color,
            justify=justify, width=width,
            highlightthickness=0,
        )
        self._win = self.create_window(
            self._PAD + 1, self._PAD + 1,
            anchor="nw", width=inner_w, height=inner_h,
            window=self._entry,
        )
        self._entry.bind("<FocusIn>",  self._on_focus_in)
        self._entry.bind("<FocusOut>", self._on_focus_out)
        self.bind("<Button-1>",
                  lambda e: self.after(1, self._entry.focus_force))
        self._entry.bind("<Button-1>",
                         lambda e: self.after(1, self._entry.focus_force),
                         add="+")
        self._draw()

    # ── Public API (tk.Entry-compatible) ──────────────────────────────────────

    def focus_force(self) -> None:
        self._entry.focus_force()

    def focus_set(self) -> None:
        self._entry.focus_set()

    def get(self) -> str:
        return self._entry.get()

    def insert(self, index, string: str) -> None:
        self._entry.insert(index, string)

    def delete(self, first, last=None) -> None:
        self._entry.delete(first, last)

    def bind(self, sequence=None, func=None, add=None):
        # Forward input events to the inner Entry
        if sequence in ("<Return>", "<FocusOut>", "<KeyRelease>",
                        "<FocusIn>", "<Tab>"):
            return self._entry.bind(sequence, func, add)
        return super().bind(sequence, func, add)

    def configure(self, **kwargs) -> None:
        entry_keys = {"textvariable", "font", "state", "fg", "bg"}
        entry_kw   = {k: v for k, v in kwargs.items() if k in entry_keys}
        canvas_kw  = {k: v for k, v in kwargs.items() if k not in entry_keys}
        if entry_kw:
            self._entry.configure(**entry_kw)
        if canvas_kw:
            super().configure(**canvas_kw)

    config = configure

    # ── Drawing ────────────────────────────────────────────────────────────────

    def _draw(self) -> None:
        tk.Canvas.delete(self, "border_layer")
        t  = self._theme
        w, h, r = self._canvas_w, self._canvas_h, t.radius

        def arc_pts(x1, y1, x2, y2):
            pts = []
            for cx, cy, a0, a1 in [
                (x2-r, y1+r, -90, 0), (x2-r, y2-r, 0, 90),
                (x1+r, y2-r, 90, 180), (x1+r, y1+r, 180, 270),
            ]:
                for i in range(9):
                    a = math.radians(a0 + (a1 - a0) * i / 8)
                    pts += [cx + r * math.cos(a), cy + r * math.sin(a)]
            return pts

        if self._focused:
            self.create_polygon(arc_pts(1, 1, w-1, h-1), smooth=False,
                                fill=t.normal_fill, outline=t.focus_ring,
                                width=1, tags="border_layer")
        else:
            self.create_polygon(arc_pts(3, 3, w-1, h-1), smooth=False,
                                fill=t.shadow_color, outline="",
                                tags="border_layer")
            self.create_polygon(arc_pts(1, 1, w-3, h-3), smooth=False,
                                fill=t.normal_fill, outline="",
                                tags="border_layer")

    def _on_focus_in(self, _=None) -> None:
        self._focused = True
        self._draw()

    def _on_focus_out(self, _=None) -> None:
        self._focused = False
        self._draw()


# ═══════════════════════════════════════════════════════════════════════════════
# _StyledStepper
# ═══════════════════════════════════════════════════════════════════════════════

class _StyledStepper(tk.Canvas):
    """Single-canvas [− value +] stepper — drop-in for tk.Spinbox.

    The left zone is the − button, the right zone is +, the centre shows the
    value in an embedded Entry (so typing works natively). Everything is drawn
    on one Canvas so there are no seams between the three zones.

    Supports .get(), textvariable=, command=, .bind(), .grid(), .grid_remove().

    Usage::

        stepper = _StyledStepper(parent, from_=6, to=48,
                                 textvariable=font_size_var,
                                 command=on_font_change)
        value = stepper.get()   # returns str, same as tk.Spinbox
    """

    _ZONE = 14   # pixel width of each ± hit zone

    def __init__(self, parent: tk.Widget, *,
                 from_: int = 0, to: int = 100,
                 textvariable: tk.IntVar | None = None,
                 command: Callable | None = None,
                 width: int = 3,
                 font: tuple | None = None,
                 bg: str | None = None,
                 theme: _ForgeTheme = THEME,
                 **kwargs):
        bg = bg or parent.cget("bg")
        val_w   = width * 10 + 8
        total_w = self._ZONE * 2 + val_w

        super().__init__(parent, width=total_w, height=BTN_H,
                         highlightthickness=0, bg=bg, cursor="hand2", **kwargs)

        self._from    = from_
        self._to      = to
        self._var     = textvariable or tk.IntVar(value=from_)
        self._cmd     = command
        self._font    = font or ("Arial", 10)
        self._total_w = total_w
        self._val_w   = val_w
        self._theme   = theme
        self._hover_dec = False
        self._hover_inc = False

        self._str_var = tk.StringVar(value=str(self._var.get()))

        self.bind("<Motion>",          self._on_motion)
        self.bind("<Leave>",           self._on_leave)
        self.bind("<ButtonPress-1>",   self._on_click)
        self._var.trace_add("write", self._on_var_write)
        self._draw()

        # Embed entry after first _draw() so it sits on top
        entry_h = BTN_H - 6
        entry_y = (BTN_H - entry_h) // 2
        t = self._theme
        self._entry = tk.Entry(
            self, textvariable=self._str_var,
            font=self._font, relief="flat", bd=0,
            bg=t.normal_fill, fg=t.text_color,
            justify="center", highlightthickness=0,
        )
        self._entry_win = self.create_window(
            self._ZONE + 2, entry_y, anchor="nw",
            width=self._val_w - 4, height=entry_h,
            window=self._entry,
        )
        self._entry.bind("<Return>",   self._commit)
        self._entry.bind("<FocusOut>", self._commit)
        self._entry.bind("<Button-1>",
                         lambda e: self.after(1, self._entry.focus_force),
                         add="+")

    # ── Public API ─────────────────────────────────────────────────────────────

    def get(self) -> str:
        """Return current value as a string — matches tk.Spinbox.get()."""
        return str(self._var.get())

    def configure(self, **kwargs) -> None:
        # Silently drop legacy spinbox-only kwargs
        for _drop in ("state", "relief", "bd", "pady"):
            kwargs.pop(_drop, None)
        if kwargs:
            super().configure(**kwargs)

    config = configure

    # ── Drawing ────────────────────────────────────────────────────────────────

    def _draw(self) -> None:
        self.delete("shapes")
        t = self._theme
        w, h, r, z = self._total_w, BTN_H, t.radius, self._ZONE
        val    = self._var.get()
        at_min = val <= self._from
        at_max = val >= self._to

        def rr(x1, y1, x2, y2, fill, outline="", tags="shapes"):
            _rounded_rect(self, x1, y1, x2, y2, r,
                          fill=fill, outline=outline, width=1, tags=tags)

        rr(3, 3, w-1, h-1, t.shadow_color)
        rr(1, 1, w-3, h-3, t.normal_fill, t.normal_border)

        if self._hover_dec and not at_min:
            rr(1, 1, z+1, h-3, t.hover_fill)
        if self._hover_inc and not at_max:
            rr(w-z-2, 1, w-3, h-3, t.hover_fill)

        sep_y1, sep_y2 = 4, h - 5
        self.create_line(z, sep_y1, z, sep_y2,
                         fill="#d8d8d8", width=1, tags="shapes")
        self.create_line(w-z-3, sep_y1, w-z-3, sep_y2,
                         fill="#d8d8d8", width=1, tags="shapes")

        dec_col = t.disabled_text if at_min else t.text_color
        inc_col = t.disabled_text if at_max else t.text_color
        cy = (h - 2) // 2
        self.create_text(z // 2,         cy, text="−", font=self._font,
                         fill=dec_col, anchor="center", tags="shapes")
        self.create_text(w - 3 - z // 2, cy, text="+", font=self._font,
                         fill=inc_col, anchor="center", tags="shapes")

        if hasattr(self, "_entry_win"):
            self.tag_raise(self._entry_win)

    # ── Events ────────────────────────────────────────────────────────────────

    def _on_motion(self, event) -> None:
        in_dec = event.x < self._ZONE
        in_inc = event.x > self._total_w - self._ZONE - 3
        if in_dec != self._hover_dec or in_inc != self._hover_inc:
            self._hover_dec = in_dec
            self._hover_inc = in_inc
            self._draw()

    def _on_leave(self, _=None) -> None:
        if self._hover_dec or self._hover_inc:
            self._hover_dec = False
            self._hover_inc = False
            self._draw()

    def _on_click(self, event) -> None:
        if event.x < self._ZONE:
            self._commit()
            self.winfo_toplevel().focus_set()
            self._decrement()
        elif event.x > self._total_w - self._ZONE - 3:
            self._commit()
            self.winfo_toplevel().focus_set()
            self._increment()
        else:
            self.after(1, self._entry.focus_force)

    def _on_var_write(self, *_) -> None:
        try:
            self._str_var.set(str(self._var.get()))
        except tk.TclError:
            pass
        self._draw()

    def _commit(self, _=None) -> None:
        if not hasattr(self, "_str_var"):
            return
        try:
            val = int(float(self._str_var.get()))
        except (ValueError, tk.TclError):
            self._str_var.set(str(self._var.get()))
            return
        val = max(self._from, min(self._to, val))
        prev = self._var.get()
        self._str_var.set(str(val))
        if val != prev:
            self._var.set(val)
            if self._cmd:
                self._cmd()

    def _increment(self) -> None:
        if self._var.get() < self._to:
            self._var.set(self._var.get() + 1)
            if self._cmd:
                self._cmd()

    def _decrement(self) -> None:
        if self._var.get() > self._from:
            self._var.set(self._var.get() - 1)
            if self._cmd:
                self._cmd()


# ═══════════════════════════════════════════════════════════════════════════════
# _StyledCombobox
# ═══════════════════════════════════════════════════════════════════════════════

class _StyledCombobox(tk.Canvas):
    """Canvas-drawn dropdown — consistent look on all platforms.

    API-compatible with ttk.Combobox (readonly mode):
    .get(), .set(), .current(), ["values"],
    .bind("<<ComboboxSelected>>", …), .configure(state=), .grid().

    Usage::

        combo = _StyledCombobox(parent, values=["A", "B", "C"],
                                width=15, label="Pick one")
        combo.bind("<<ComboboxSelected>>", on_select)
        value = combo.get()
    """

    def __init__(self, parent: tk.Widget, *,
                 textvariable: tk.StringVar | None = None,
                 values: list[str] | None = None,
                 state: str = "normal",
                 width: int = 15,
                 font: tuple | None = None,
                 label: str = "",
                 bg: str | None = None,
                 theme: _ForgeTheme = THEME,
                 **kwargs):
        self._px_width = width * 8 + 24   # char estimate + chevron zone
        self._height   = BTN_H
        bg = bg or parent.cget("bg")
        super().__init__(parent, width=self._px_width, height=self._height,
                         highlightthickness=0, bg=bg, cursor="hand2", **kwargs)

        self._var      = textvariable or tk.StringVar()
        self._values   = list(values or [])
        self._current  = self._var.get()
        self._disabled = (state == "disabled")
        self._hover    = False
        self._font     = font or ("Arial", 10)
        self._label    = label
        self._theme    = theme
        self._popup: tk.Toplevel | None = None
        self._extra_bindings: list[tuple] = []

        self.bind("<Enter>",         self._on_enter)
        self.bind("<Leave>",         self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self._draw()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get(self) -> str:
        return self._current

    def set(self, value: str) -> None:
        self._current = value
        self._var.set(value)
        self._draw()

    def current(self, index: int | None = None) -> int | None:
        if index is None:
            try:
                return self._values.index(self._current)
            except ValueError:
                return -1
        if 0 <= index < len(self._values):
            self.set(self._values[index])
        return None

    def __setitem__(self, key, value) -> None:
        if key == "values":
            self._values = list(value)
        self._draw()

    def __getitem__(self, key):
        if key == "values":
            return self._values
        raise KeyError(key)

    def configure(self, **kwargs) -> None:
        if "state" in kwargs:
            state = kwargs.pop("state")
            self._disabled = (state == "disabled")
            self.config(cursor="" if self._disabled else "hand2")
        if "width" in kwargs:
            self._px_width = kwargs.pop("width") * 8 + 24
            super().configure(width=self._px_width)
        if kwargs:
            super().configure(**kwargs)
        self._draw()

    config = configure

    def bind(self, sequence=None, func=None, add=None):
        if sequence and sequence not in ("<Enter>", "<Leave>", "<ButtonPress-1>"):
            self._extra_bindings.append((sequence, func))
        return super().bind(sequence, func, add)

    # ── Drawing ────────────────────────────────────────────────────────────────

    def _draw(self) -> None:
        self.delete("all")
        t = self._theme
        w, h, r = self._px_width, self._height, t.radius

        if self._disabled:
            fill, shadow = t.disabled_fill, False
            text_color   = t.disabled_text
        elif self._hover:
            fill, shadow = t.hover_fill, True
            text_color   = t.text_color
        else:
            fill, shadow = t.normal_fill, True
            text_color   = t.text_color

        if shadow:
            _rounded_rect(self, 3, 3, w-1, h-1, r,
                          fill=t.shadow_color, outline="")
        _rounded_rect(self, 1, 1, w-3, h-3, r,
                      fill=fill,
                      outline=t.normal_border if not shadow else "",
                      width=1)

        chev_x = w - 20
        self.create_line(chev_x, 4, chev_x, h-5, fill="#d0d0d0", width=1)
        cx, cy = w - 10, (h - 2) // 2
        self.create_polygon(cx-4, cy-2, cx+4, cy-2, cx, cy+3,
                            fill=text_color, outline="")

        if self._popup and self._label:
            display    = self._label
            text_color = "#999999"
        else:
            display = self._current or "—"
        self.create_text(8, (h-2)//2, text=display, font=self._font,
                         fill=text_color, anchor="w", width=chev_x - 8)

    # ── Events ─────────────────────────────────────────────────────────────────

    def _on_enter(self, _=None) -> None:
        if not self._disabled:
            self._hover = True
            self._draw()

    def _on_leave(self, _=None) -> None:
        self._hover = False
        self._draw()

    def _on_press(self, _=None) -> None:
        if self._disabled or not self._values:
            return
        self._open_popup()

    def _open_popup(self) -> None:
        if self._popup:
            self._close_popup()
            return
        self.update_idletasks()
        ax = self.winfo_rootx()
        ay = self.winfo_rooty() + self._height + 2

        pop = tk.Toplevel(self)
        pop.wm_overrideredirect(True)
        pop.configure(bg="#d0d0d0")
        self._popup = pop

        outer = tk.Frame(pop, bg="#d0d0d0", padx=1, pady=1)
        outer.pack(fill=tk.BOTH, expand=True)
        inner = tk.Frame(outer, bg="white")
        inner.pack(fill=tk.BOTH, expand=True)

        max_items = 12
        visible   = min(len(self._values), max_items)
        t = self._theme

        lb = tk.Listbox(inner, font=self._font, relief="flat", bd=0,
                        selectmode=tk.SINGLE, activestyle="none",
                        selectbackground=t.active_fill,
                        selectforeground=t.text_color,
                        bg="white", fg=t.text_color, height=visible)
        for v in self._values:
            lb.insert(tk.END, "  " + v)
        if self._current in self._values:
            idx = self._values.index(self._current)
            lb.selection_set(idx)
            lb.see(idx)
        if len(self._values) > max_items:
            sb = tk.Scrollbar(inner, orient=tk.VERTICAL, command=lb.yview)
            lb.configure(yscrollcommand=sb.set)
            sb.pack(side=tk.RIGHT, fill=tk.Y)
        lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        pop.update_idletasks()
        pop.wm_geometry(
            f"{self._px_width}x{pop.winfo_reqheight()}+{ax}+{ay}")

        lb.bind("<ButtonRelease-1>", lambda e: self._pick(lb))
        lb.bind("<Return>",          lambda e: self._pick(lb))
        lb.bind("<Escape>",          lambda e: self._close_popup())
        pop.bind("<FocusOut>",
                 lambda e: self.after(100, self._close_popup_if_unfocused))
        pop.focus_set()
        lb.focus_set()
        self._draw()

    def _close_popup_if_unfocused(self) -> None:
        if self._popup and not self._popup.focus_displayof():
            self._close_popup()

    def _close_popup(self) -> None:
        if self._popup:
            self._popup.destroy()
            self._popup = None
        self._hover = False
        self._draw()

    def _pick(self, lb: tk.Listbox) -> None:
        sel = lb.curselection()
        if not sel:
            return
        value = self._values[sel[0]]
        self._close_popup()
        self._current = value
        self._var.set(value)
        self._draw()
        for seq, func in self._extra_bindings:
            if func:
                try:
                    func(None)
                except Exception:
                    pass
        self.event_generate("<<ComboboxSelected>>")


# ═══════════════════════════════════════════════════════════════════════════════
# _StyledSlider  ★ NEW
# ═══════════════════════════════════════════════════════════════════════════════

class _StyledSlider(tk.Canvas):
    """Horizontal slider — plain line track + circular thumb.

    Drop-in for ttk.Scale (horizontal orient only).

    Track: 2px grey line with optional center-tick marker.
    Thumb: 14px white circle with drop shadow; hover tints blue;
           dragging fills the active blue.

    Supports .get(), .set(), textvariable=, command=,
    .bind(), .grid(), .grid_remove().

    Usage::

        slider = _StyledSlider(parent, from_=0.25, to=1.0,
                               variable=scale_var,
                               command=on_change,
                               width=140)

        # Slope slider — show a center reference tick at 0.0
        slope_slider = _StyledSlider(parent, from_=-1.0, to=1.0,
                                     variable=slope_var,
                                     command=on_change,
                                     show_center_tick=True,
                                     center_value=0.0,
                                     width=140)
    """

    _THUMB_R   = 7    # thumb radius in pixels
    _TRACK_H   = 2    # track line thickness
    _TICK_H    = 6    # center-tick height in pixels
    _PAD       = 9    # horizontal padding so thumb never clips the canvas edge

    def __init__(self, parent: tk.Widget, *,
                 from_: float = 0.0,
                 to: float = 1.0,
                 variable: tk.DoubleVar | None = None,
                 command: Callable | None = None,
                 width: int = 140,
                 show_center_tick: bool = False,
                 center_value: float | None = None,
                 bg: str | None = None,
                 theme: _ForgeTheme = THEME,
                 **kwargs):
        bg = bg or parent.cget("bg")
        super().__init__(parent, width=width, height=BTN_H,
                         highlightthickness=0, bg=bg,
                         cursor="hand2", **kwargs)

        self._from       = float(from_)
        self._to         = float(to)
        self._var        = variable if variable is not None else tk.DoubleVar(value=from_)
        self._cmd        = command
        self._theme      = theme
        self._width      = width
        self._dragging   = False
        self._hover      = False

        self._show_tick  = show_center_tick
        # Default center_value to the midpoint if not specified
        self._center_val = (center_value if center_value is not None
                            else (from_ + to) / 2.0)

        # Clamp _center_val to the valid range
        self._center_val = max(self._from, min(self._to, self._center_val))

        self._var.trace_add("write", self._on_var_write)

        self.bind("<Enter>",           self._on_enter)
        self.bind("<Leave>",           self._on_leave)
        self.bind("<ButtonPress-1>",   self._on_press)
        self.bind("<B1-Motion>",       self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)

        self._draw()

    # ── Public API (ttk.Scale-compatible) ─────────────────────────────────────

    def get(self) -> float:
        """Return current value as float — matches ttk.Scale.get()."""
        return self._var.get()

    def set(self, value: float) -> None:
        """Set slider value programmatically."""
        value = max(self._from, min(self._to, float(value)))
        self._var.set(value)

    def configure(self, **kwargs) -> None:
        if "variable" in kwargs:
            self._var.trace_remove("write",
                *[tid for tid in self._var.trace_info()
                  if tid[1] == self._on_var_write.__name__])
            self._var = kwargs.pop("variable")
            self._var.trace_add("write", self._on_var_write)
        if "command" in kwargs:
            self._cmd = kwargs.pop("command")
        if "from_" in kwargs:
            self._from = float(kwargs.pop("from_"))
        if "to" in kwargs:
            self._to = float(kwargs.pop("to"))
        if "state" in kwargs:
            state = kwargs.pop("state")
            self._disabled = (state == "disabled")
            self.config(cursor="" if self._disabled else "hand2")
        for _drop in ("orient", "relief", "bd"):
            kwargs.pop(_drop, None)
        if kwargs:
            super().configure(**kwargs)
        self._draw()

    config = configure

    # ── Geometry helpers ──────────────────────────────────────────────────────

    def _val_to_px(self, value: float) -> float:
        """Convert a data value to a canvas x-pixel position."""
        span = self._to - self._from
        if span == 0:
            return float(self._PAD)
        pct = (value - self._from) / span
        track_w = self._width - self._PAD * 2
        return self._PAD + pct * track_w

    def _px_to_val(self, px: float) -> float:
        """Convert a canvas x-pixel position to a data value."""
        track_w = self._width - self._PAD * 2
        if track_w <= 0:
            return self._from
        pct = (px - self._PAD) / track_w
        pct = max(0.0, min(1.0, pct))
        return self._from + pct * (self._to - self._from)

    # ── Drawing ────────────────────────────────────────────────────────────────

    def _draw(self) -> None:
        self.delete("all")
        t   = self._theme
        w   = self._width
        h   = BTN_H
        cy  = h // 2          # vertical centre of canvas
        pad = self._PAD

        # ── Track line ────────────────────────────────────────────────────────
        self.create_line(pad, cy, w - pad, cy,
                         fill=t.track_color,
                         width=self._TRACK_H,
                         capstyle=tk.ROUND)

        # ── Optional center-reference tick ────────────────────────────────────
        if self._show_tick:
            tx = self._val_to_px(self._center_val)
            half = self._TICK_H // 2
            self.create_line(tx, cy - half, tx, cy + half,
                             fill="#aaaaaa", width=1)

        # ── Thumb ─────────────────────────────────────────────────────────────
        tx = self._val_to_px(self._var.get())
        r  = self._THUMB_R

        # Drop shadow (offset 1 px right + down, matches buttons/entry)
        self.create_oval(tx - r + 2, cy - r + 2,
                         tx + r + 1, cy + r + 1,
                         fill=t.shadow_color, outline="")

        if self._dragging:
            fill, border = t.active_fill, t.active_border
        elif self._hover:
            fill, border = t.hover_fill, t.hover_border
        else:
            fill, border = t.normal_fill, t.normal_border

        self.create_oval(tx - r, cy - r, tx + r - 1, cy + r - 1,
                         fill=fill, outline=border, width=1)

    # ── Events ────────────────────────────────────────────────────────────────

    def _on_enter(self, _=None) -> None:
        self._hover = True
        self._draw()

    def _on_leave(self, _=None) -> None:
        self._hover = False
        self._draw()

    def _on_press(self, event) -> None:
        self._dragging = True
        self._update_from_event(event)

    def _on_drag(self, event) -> None:
        if self._dragging:
            self._update_from_event(event)

    def _on_release(self, event) -> None:
        self._dragging = False
        self._update_from_event(event)
        self._draw()

    def _update_from_event(self, event) -> None:
        new_val = self._px_to_val(event.x)
        self._var.set(new_val)
        self._draw()
        if self._cmd:
            self._cmd(str(new_val))

    def _on_var_write(self, *_) -> None:
        """Redraw when the linked variable changes externally (e.g. reset)."""
        self._draw()


# ═══════════════════════════════════════════════════════════════════════════════
# _StyledCheckbox  ★ NEW
# ═══════════════════════════════════════════════════════════════════════════════

class _StyledCheckbox(tk.Frame):
    """Rounded-square checkbox — consistent look on all platforms.

    Drop-in for tk.Checkbutton.

    Unchecked: white fill + grey border + drop shadow.
    Checked:   active-blue fill + blue border + white angled checkmark.
    Hover:     blue-tint fill preview on the box.
    Disabled:  grey fill, no cursor, no interaction.

    Supports variable=, command=, text=, .get(), .set(),
    .bind(), .grid(), .grid_remove().

    Usage::

        cb = _StyledCheckbox(parent, text="Show Hashmarks",
                             variable=hashmarks_var,
                             command=on_hashmarks_changed)
        checked = cb.get()   # bool
        cb.set(True)
    """

    _BOX_SIZE  = 14   # width and height of the checkbox square
    _MARK_PAD  = 3    # inset from box edge to start of checkmark
    _RADIUS    = 5    # rounded corner radius — matches all other widgets

    def __init__(self, parent: tk.Widget, *,
                 variable: tk.BooleanVar | None = None,
                 command: Callable | None = None,
                 text: str = "",
                 font: tuple | None = None,
                 bg: str | None = None,
                 theme: _ForgeTheme = THEME,
                 takefocus: int = 0,
                 **kwargs):
        bg = bg or parent.cget("bg")
        super().__init__(parent, bg=bg, cursor="hand2", **kwargs)

        self._var      = variable if variable is not None else tk.BooleanVar(value=False)
        self._cmd      = command
        self._text     = text
        self._font     = font or ("Arial", 10)
        self._theme    = theme
        self._bg       = bg
        self._hover    = False
        self._disabled = False

        # ── Box canvas ────────────────────────────────────────────────────────
        s = self._BOX_SIZE
        self._box = tk.Canvas(self, width=s, height=s,
                              highlightthickness=0, bg=bg,
                              cursor="hand2")
        self._box.pack(side=tk.LEFT, padx=(0, 4 if text else 0))

        # ── Optional text label ───────────────────────────────────────────────
        self._label_widget: tk.Label | None = None
        if text:
            self._label_widget = tk.Label(self, text=text, bg=bg,
                                          font=self._font,
                                          cursor="hand2")
            self._label_widget.pack(side=tk.LEFT)

        # ── Bindings — on both the frame, box canvas, and label ───────────────
        for widget in [self, self._box] + (
                [self._label_widget] if self._label_widget else []):
            widget.bind("<Enter>",         self._on_enter)
            widget.bind("<Leave>",         self._on_leave)
            widget.bind("<ButtonPress-1>", self._on_press)

        self._var.trace_add("write", self._on_var_write)
        self._draw()

    # ── Public API (tk.Checkbutton-compatible) ────────────────────────────────

    def get(self) -> bool:
        """Return current checked state."""
        return bool(self._var.get())

    def set(self, value: bool) -> None:
        """Set checked state programmatically."""
        self._var.set(value)

    def configure(self, **kwargs) -> None:
        if "text" in kwargs:
            self._text = kwargs.pop("text")
            if self._label_widget:
                self._label_widget.configure(text=self._text)
        if "variable" in kwargs:
            self._var = kwargs.pop("variable")
            self._var.trace_add("write", self._on_var_write)
        if "command" in kwargs:
            self._cmd = kwargs.pop("command")
        if "state" in kwargs:
            state = kwargs.pop("state")
            self._disabled = (state == "disabled")
            cursor = "" if self._disabled else "hand2"
            self.config(cursor=cursor)
            self._box.config(cursor=cursor)
            if self._label_widget:
                self._label_widget.config(cursor=cursor)
        if "font" in kwargs:
            self._font = kwargs.pop("font")
            if self._label_widget:
                self._label_widget.configure(font=self._font)
        # Drop legacy Checkbutton-only kwargs silently
        for _drop in ("relief", "bd", "pady", "takefocus", "onvalue", "offvalue"):
            kwargs.pop(_drop, None)
        if kwargs:
            super().configure(**kwargs)
        self._draw()

    config = configure

    # ── Drawing ────────────────────────────────────────────────────────────────

    def _draw(self) -> None:
        c   = self._box
        t   = self._theme
        s   = self._BOX_SIZE
        r   = self._RADIUS
        checked = self._var.get()

        c.delete("all")

        # Determine fill and border
        if self._disabled:
            fill   = t.disabled_fill
            border = "#e0e0e0"
            shadow = False
        elif checked:
            fill   = t.active_fill
            border = t.active_border
            shadow = True
        elif self._hover:
            fill   = t.hover_fill
            border = t.hover_border
            shadow = True
        else:
            fill   = t.normal_fill
            border = t.normal_border
            shadow = True

        # Drop shadow
        if shadow:
            _rounded_rect(c, 2, 2, s, s, r,
                          fill=t.shadow_color, outline="")

        # Box body
        _rounded_rect(c, 0, 0, s - 2, s - 2, r,
                      fill=fill, outline=border, width=1)

        # Checkmark — two line segments forming a tick (✓)
        if checked:
            mk = t.check_color if not self._disabled else t.disabled_text
            p  = self._MARK_PAD
            bw = s - 2   # inner box width (accounting for shadow offset)
            # Left leg: bottom-left corner up to the mid-dip
            x1, y1 = p,           bw - p - 1          # bottom-left start
            x2, y2 = bw // 2 - 1, bw - p + 1          # lowest dip point
            # Right leg: mid-dip up to top-right
            x3, y3 = bw - p,      p + 1               # top-right tip
            c.create_line(x1, y1, x2, y2, fill=mk, width=2,
                          capstyle=tk.ROUND, joinstyle=tk.ROUND)
            c.create_line(x2, y2, x3, y3, fill=mk, width=2,
                          capstyle=tk.ROUND, joinstyle=tk.ROUND)

        # Sync label text color
        if self._label_widget:
            self._label_widget.configure(
                fg=t.disabled_text if self._disabled else t.text_color
            )

    # ── Events ────────────────────────────────────────────────────────────────

    def _on_enter(self, _=None) -> None:
        if not self._disabled:
            self._hover = True
            self._draw()

    def _on_leave(self, _=None) -> None:
        self._hover = False
        self._draw()

    def _on_press(self, _=None) -> None:
        if self._disabled:
            return
        self._var.set(not self._var.get())
        if self._cmd:
            self._cmd()

    def _on_var_write(self, *_) -> None:
        """Redraw when the linked variable changes externally."""
        self._draw()
