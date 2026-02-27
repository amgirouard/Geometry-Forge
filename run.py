"""Run Geometry Forge — convenience launcher."""

import tkinter as tk
from geometry_forge.app import GeometryApp

if __name__ == "__main__":
    root = tk.Tk()
    app = GeometryApp(root)
    root.mainloop()
