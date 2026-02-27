"""Entry point for running Geometry Forge as a package: python -m geometry_forge"""

import tkinter as tk
from .app import GeometryApp


def main():
    root = tk.Tk()
    app = GeometryApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
