"""Run Geometry Forge — convenience launcher.

Usage:  python run.py
"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    app = Path(__file__).parent / "streamlit_app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app)],
        check=False,
    )
