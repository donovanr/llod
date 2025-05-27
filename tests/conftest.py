"""
Configuration for pytest
"""

import sys
from pathlib import Path

# Add the root directory to the path so we can import app
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))
