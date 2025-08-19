#!/usr/bin/env python3
"""
Launcher script for MGF Explorer.
"""

import sys
import os

# Add the src directory to the Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

from mgfexplorer.app import main

if __name__ == "__main__":
    main()
