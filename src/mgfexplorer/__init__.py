"""
MGF Explorer - A tool for exploring and editing MGF (Mascot Generic Format) files.
"""

from .app import main
from .mgf_parser import MGFParser, Spectrum

__version__ = "1.0.0"
__all__ = ["main", "MGFParser", "Spectrum"]
