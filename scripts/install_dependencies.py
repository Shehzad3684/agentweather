"""
AgentWeather Automated Dependency Installer
Installs all production dependencies pinned in requirements.txt.
"""

import sys
import subprocess
from pathlib import Path

def install_all():
    root_dir = Path(__file__).resolve().parent.parent
    req_file = root_dir / "requirements.txt"
    if not req_file.exists():
        print(f"Error: {req_file} not found.")
        sys.exit(1)
    print(f"Installing dependencies from {req_file}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
    print("Dependencies installed successfully.")

if __name__ == "__main__":
    install_all()
