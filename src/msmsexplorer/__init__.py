"""
MGF Explorer - A tool for exploring and editing MGF (Mascot Generic Format) files.
"""

from ._version import __version__
from .app import main
from .mgf_parser import MGFParser, Spectrum

__all__ = ["main", "MGFParser", "Spectrum", "__version__"]
