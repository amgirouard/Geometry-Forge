from __future__ import annotations

import tkinter as tk
from typing import Callable

from .models import AppConstants


class CompositeTransferList(tk.Frame):
    """Transfer list widget for composite figure shape selection.

    Source list (available shapes) on the left, arrow buttons in the centre,
    destination list (selected shapes) on the right — drag-to-reorder supported.
    """

    DISPLAY_NAMES: dict[str, str] = {"Tri Prism": "Triangular Prism", "Tri Triangle": "Triangle"}
    INTERNAL_NAMES: dict[str, str] = {v: k for k, v in DISPLAY_NAMES.items()}

    # Visual feedback colours for drag
    _DRAG_COLOR   = "#cce0ff"   # highlight row being dragged over
    _NORMAL_COLOR = "white"

    def __init__(self, parent: tk.Frame, available_shapes: list[str],
                 on_change_callback: Callable[..., None],
                 on_before_change_callback: Callable[..., None] | None = None, **kwargs):
        super().__init__(parent, bg=AppConstants.BG_COLOR, **kwargs)
        self.available_shapes = list(available_shapes)
        self.on_change_callback = on_change_callback
        self.on_before_change_callback = on_before_change_callback
        self._drag_start_idx: int | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        """Build the three-column transfer list layout."""
        # --- Source column ---
        src_frame = tk.Frame(self, bg=AppConstants.BG_COLOR)
        src_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 2))

        tk.Label(src_frame, text="Available", bg=AppConstants.BG_COLOR,
                 font=("Arial", 8, "bold")).pack(side=tk.TOP)

        self.source_listbox = tk.Listbox(src_frame, width=18, height=5,
                                          selectmode=tk.SINGLE, exportselection=False)
        self.source_listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.source_listbox.bind("<Double-Button-1>", lambda e: self._add_selected())

        for shape in self.available_shapes:
            self.source_listbox.insert(tk.END, self.DISPLAY_NAMES.get(shape, shape))

        # --- Arrow buttons column ---
        btn_frame = tk.Frame(self, bg=AppConstants.BG_COLOR)
        btn_frame.pack(side=tk.LEFT, fill=tk.Y, padx=4)

        tk.Frame(btn_frame, bg=AppConstants.BG_COLOR, height=10).pack(side=tk.TOP)

        self.add_btn = tk.Button(btn_frame, text="→", width=1, font=AppConstants.BTN_FONT,
                                  command=self._add_selected)
        self.add_btn.pack(side=tk.TOP, pady=2)

        self.remove_btn = tk.Button(btn_frame, text="←", width=1, font=AppConstants.BTN_FONT,
                                     command=self._remove_selected)
        self.remove_btn.pack(side=tk.TOP, pady=2)

        # --- Destination column (drag-to-reorder) ---
        dest_outer = tk.Frame(self, bg=AppConstants.BG_COLOR)
        dest_outer.pack(side=tk.LEFT, fill=tk.BOTH, padx=(2, 0))

        tk.Label(dest_outer, text="Selected  ☰ drag to reorder", bg=AppConstants.BG_COLOR,
                 font=("Arial", 8, "bold")).pack(side=tk.TOP)

        self.dest_listbox = tk.Listbox(dest_outer, width=13, height=10,
                                        selectmode=tk.SINGLE, exportselection=False,
                                        cursor="arrow")
        self.dest_listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Drag-to-reorder bindings
        self.dest_listbox.bind("<Button-1>",        self._drag_start)
        self.dest_listbox.bind("<B1-Motion>",        self._drag_motion)
        self.dest_listbox.bind("<ButtonRelease-1>",  self._drag_end)
        self.dest_listbox.bind("<Double-Button-1>",  lambda e: self._remove_selected())

    # ── drag-to-reorder ───────────────────────────────────────────────────────

    def _item_index_at(self, y: int) -> int | None:
        """Return listbox index under pixel y, or None if empty."""
        idx = self.dest_listbox.nearest(y)
        if self.dest_listbox.size() == 0:
            return None
        return idx

    def _drag_start(self, event) -> None:
        idx = self._item_index_at(event.y)
        self._drag_start_idx = idx
        if idx is not None:
            self.dest_listbox.selection_clear(0, tk.END)
            self.dest_listbox.selection_set(idx)

    def _drag_motion(self, event) -> None:
        if self._drag_start_idx is None:
            return
        target = self._item_index_at(event.y)
        if target is None:
            return
        # Visual feedback: highlight target row
        for i in range(self.dest_listbox.size()):
            self.dest_listbox.itemconfig(i, bg=self._DRAG_COLOR if i == target else self._NORMAL_COLOR)

    def _drag_end(self, event) -> None:
        src = self._drag_start_idx
        self._drag_start_idx = None
        # Clear highlights
        for i in range(self.dest_listbox.size()):
            self.dest_listbox.itemconfig(i, bg=self._NORMAL_COLOR)
        if src is None:
            return
        target = self._item_index_at(event.y)
        if target is None or target == src:
            return
        # Move item from src to target
        shape = self.dest_listbox.get(src)
        self.dest_listbox.delete(src)
        self.dest_listbox.insert(target, shape)
        self.dest_listbox.selection_set(target)
        self.on_change_callback(("swap", src, target))

    # ── add / remove ─────────────────────────────────────────────────────────

    def _add_selected(self) -> None:
        sel = self.source_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        shape = self.source_listbox.get(idx)
        self.dest_listbox.insert(tk.END, "☰  " + shape)
        self.source_listbox.selection_set(idx)
        self.on_change_callback(("add", self.dest_listbox.size() - 1))

    def _remove_selected(self) -> None:
        sel = self.dest_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        self.dest_listbox.delete(idx)
        if self.dest_listbox.size() > 0:
            self.dest_listbox.selection_set(min(idx, self.dest_listbox.size() - 1))
        self.on_change_callback(("remove", idx))

    def get_selected_shapes(self) -> list[str]:
        return [self.INTERNAL_NAMES.get(self.dest_listbox.get(i).removeprefix("☰  "), self.dest_listbox.get(i).removeprefix("☰  "))
                for i in range(self.dest_listbox.size())]

    def set_selected_shapes(self, shapes: list[str]) -> None:
        self.dest_listbox.delete(0, tk.END)
        for s in shapes:
            self.dest_listbox.insert(tk.END, "☰  " + self.DISPLAY_NAMES.get(s, s))