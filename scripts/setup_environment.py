"""
AgentWeather Environment Initializer
Creates virtual environment and prepares project directories.
"""

import sys
import subprocess
from pathlib import Path

def setup():
    root = Path(__file__).resolve().parent.parent
    venv_dir = root / "venv"
    if not venv_dir.exists():
        print("Creating virtual environment at venv...")
        import venv
        venv.create(venv_dir, with_pip=True)
        print("Virtual environment created successfully.")
    else:
        print("Virtual environment already exists.")

if __name__ == "__main__":
    setup()
