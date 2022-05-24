"""
Problem: from the notebooks we can't import our library packages, because the notebook is executed from the "notebooks"
directory (instead of the project root).

Fix: we can add the root directory to sys path, such that it can find and import our library packages.

Usage: in the notebooks always first execute "import addroot" before importing any of the lib packages
"""
import sys
from pathlib import Path


notebook_directory = Path(__file__).parent.resolve()
root_directory = (notebook_directory / "..").resolve()

if root_directory not in sys.path:
    sys.path.append(str(root_directory))
