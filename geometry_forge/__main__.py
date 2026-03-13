"""Entry point for running Geometry Forge as a package.

Usage:  python -m geometry_forge
"""
import subprocess
import sys
from pathlib import Path


def main() -> None:
    app = Path(__file__).parent.parent / "streamlit_app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app)],
        check=False,
    )


if __name__ == "__main__":
    main()
