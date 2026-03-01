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
                 font=AppConstants.HEADER_FONT).pack(side=tk.TOP)

        self.source_listbox = tk.Listbox(src_frame, width=18, height=5,
                                          selectmode=tk.SINGLE, exportselection=False,
                                          relief="flat", bd=0,
                                          highlightthickness=1,
                                          highlightbackground="#aaaaaa",
                                          highlightcolor="#aaaaaa")
        self.source_listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.source_listbox.bind("<Double-Button-1>", lambda e: self._add_selected())

        for shape in self.available_shapes:
            self.source_listbox.insert(tk.END, self.DISPLAY_NAMES.get(shape, shape))

        tk.Label(src_frame, text="double-click to select", bg=AppConstants.BG_COLOR,
                 font=("Arial", 7), fg="#888888").pack(side=tk.TOP)

        # --- Destination column (drag-to-reorder) ---
        # Gap replaces the old arrow-button column
        dest_outer = tk.Frame(self, bg=AppConstants.BG_COLOR)
        dest_outer.pack(side=tk.LEFT, fill=tk.BOTH, padx=(10, 0))

        tk.Label(dest_outer, text="Selected", bg=AppConstants.BG_COLOR,
                 font=AppConstants.HEADER_FONT).pack(side=tk.TOP)

        # Container so we can overlay a drop-line indicator on top of the listbox
        dest_lb_frame = tk.Frame(dest_outer, bg=AppConstants.BG_COLOR)
        dest_lb_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.dest_listbox = tk.Listbox(dest_lb_frame, width=18, height=10,
                                        selectmode=tk.SINGLE, exportselection=False,
                                        cursor="arrow",
                                        relief="flat", bd=0,
                                        highlightthickness=1,
                                        highlightbackground="#aaaaaa",
                                        highlightcolor="#aaaaaa")
        self.dest_listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Thin Frame used as the drop indicator line — placed absolutely over the listbox
        # using place(), which doesn't affect pack geometry. Hidden until drag starts.
        self._drop_indicator = tk.Frame(dest_lb_frame, bg="#2266cc", height=2, bd=0)
        self._dest_lb_frame = dest_lb_frame

        tk.Label(dest_outer, text="☰ drag to reorder", bg=AppConstants.BG_COLOR,
                 font=("Arial", 7), fg="#888888").pack(side=tk.TOP)

        # Drag-to-reorder bindings
        self.dest_listbox.bind("<Button-1>",        self._drag_start)
        self.dest_listbox.bind("<B1-Motion>",        self._drag_motion)
        self.dest_listbox.bind("<ButtonRelease-1>",  self._drag_end)
        self.dest_listbox.bind("<Double-Button-1>",  lambda e: self._remove_selected())

    # ── drag-to-reorder ───────────────────────────────────────────────────────

    def _item_index_at(self, y: int) -> int | None:
        """Return listbox index under pixel y, or None if empty."""
        if self.dest_listbox.size() == 0:
            return None
        return self.dest_listbox.nearest(y)

    def _row_height(self) -> int:
        """Return the pixel height of one listbox row."""
        try:
            return self.dest_listbox.winfo_reqheight() // max(self.dest_listbox.cget("height"), 1)
        except Exception:
            return 18

    def _drop_insert_pos(self, y: int) -> int:
        """Return the insertion index (0..size) that best matches cursor y.
        
        The cursor is mapped to whichever gap between rows it is closest to.
        Index 0 = above first item, index size = below last item.
        """
        n = self.dest_listbox.size()
        if n == 0:
            return 0
        rh = self._row_height()
        # nearest() gives the closest existing row index
        nearest = self.dest_listbox.nearest(y)
        # pixel centre of that row
        row_top = nearest * rh
        row_mid = row_top + rh / 2
        if y <= row_mid:
            return nearest          # insert ABOVE nearest row
        else:
            return nearest + 1      # insert BELOW nearest row

    def _draw_drop_line(self, insert_pos: int) -> None:
        """Position the drop indicator frame at gap insert_pos."""
        self.dest_listbox.update_idletasks()
        rh = self._row_height()
        # y is relative to dest_lb_frame; offset by listbox's y position within it
        lb_y = self.dest_listbox.winfo_y()
        y = lb_y + insert_pos * rh
        h = self._dest_lb_frame.winfo_height()
        y = max(0, min(y, h - 2))
        w = self._dest_lb_frame.winfo_width()
        self._drop_indicator.place(x=0, y=y, width=w, height=2)
        self._drop_indicator.lift()

    def _clear_drop_line(self) -> None:
        """Hide the drop indicator."""
        self._drop_indicator.place_forget()

    def _drag_start(self, event) -> None:
        idx = self._item_index_at(event.y)
        self._drag_start_idx = idx
        if idx is not None:
            self.dest_listbox.selection_clear(0, tk.END)
            self.dest_listbox.selection_set(idx)

    def _drag_motion(self, event) -> None:
        if self._drag_start_idx is None:
            return
        insert_pos = self._drop_insert_pos(event.y)
        self._draw_drop_line(insert_pos)

    def _drag_end(self, event) -> None:
        src = self._drag_start_idx
        self._drag_start_idx = None
        self._clear_drop_line()
        # Clear any selection highlight on all rows
        for i in range(self.dest_listbox.size()):
            self.dest_listbox.itemconfig(i, bg=self._NORMAL_COLOR)
        if src is None:
            return
        insert_pos = self._drop_insert_pos(event.y)
        if insert_pos == src or insert_pos == src + 1:
            # Dropped back on itself — no move, just re-select
            self.dest_listbox.selection_set(src)
            return
        # Pull the item out, adjust target index for the deletion, then insert
        shape = self.dest_listbox.get(src)
        self.dest_listbox.delete(src)
        target = insert_pos if insert_pos < src else insert_pos - 1
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