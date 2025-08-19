"""
MGF (Mascot Generic Format) file parser for mass spectrometry data.
"""

import numpy as np
from typing import Dict, List, Any, Optional
import re


class Spectrum:
    """Represents a single MS/MS spectrum from an MGF file."""
    
    def __init__(self, spectrum_id: int):
        self.spectrum_id = spectrum_id
        self.metadata: Dict[str, str] = {}
        self.ions: np.ndarray = np.array([])  # 2D array: [mz, intensity]
        
    def add_metadata(self, key: str, value: str):
        """Add metadata key-value pair, handling duplicate keys."""
        original_key = key
        counter = 1
        
        while key in self.metadata:
            key = f"{original_key}_{counter}"
            counter += 1
            
        self.metadata[key] = value
        
    def set_ions(self, mz_values: List[float], intensity_values: List[float]):
        """Set the ion data as a 2D numpy array."""
        if len(mz_values) != len(intensity_values):
            raise ValueError("mz and intensity arrays must have the same length")
        
        self.ions = np.column_stack((mz_values, intensity_values))
        
    def get_metadata_value(self, key: str) -> Optional[str]:
        """Get metadata value by key."""
        return self.metadata.get(key)
        
    def update_metadata(self, key: str, value: str):
        """Update existing metadata value."""
        if key in self.metadata:
            self.metadata[key] = value
            
    def rename_metadata_key(self, old_key: str, new_key: str):
        """Rename a metadata key."""
        if old_key in self.metadata and new_key not in self.metadata:
            self.metadata[new_key] = self.metadata.pop(old_key)


class MGFParser:
    """Parser for MGF (Mascot Generic Format) files."""
    
    def __init__(self):
        self.spectra: List[Spectrum] = []
        
    def parse_file(self, file_path: str) -> List[Spectrum]:
        """Parse an MGF file and return a list of Spectrum objects."""
        self.spectra = []
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            
        # Split content by BEGIN IONS blocks
        blocks = re.split(r'BEGIN IONS\s*\n', content)
        
        # Skip the first block (before any BEGIN IONS)
        for i, block in enumerate(blocks[1:], 1):
            if 'END IONS' in block:
                spectrum = self._parse_spectrum_block(block, i)
                if spectrum:
                    self.spectra.append(spectrum)
                    
        return self.spectra
    
    def _parse_spectrum_block(self, block: str, spectrum_id: int) -> Optional[Spectrum]:
        """Parse a single spectrum block."""
        lines = block.split('\n')
        spectrum = Spectrum(spectrum_id)
        
        mz_values = []
        intensity_values = []
        
        for line in lines:
            line = line.strip()
            
            if not line or line == 'END IONS':
                continue
                
            # Check if line contains key=value metadata
            if '=' in line and not self._is_ion_line(line):
                parts = line.split('=', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    spectrum.add_metadata(key, value)
            
            # Check if line contains ion data (mz intensity)
            elif self._is_ion_line(line):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        mz = float(parts[0])
                        intensity = float(parts[1])
                        mz_values.append(mz)
                        intensity_values.append(intensity)
                    except ValueError:
                        continue
        
        # Set ion data
        if mz_values and intensity_values:
            spectrum.set_ions(mz_values, intensity_values)
            
        return spectrum if spectrum.metadata or len(mz_values) > 0 else None
    
    def _is_ion_line(self, line: str) -> bool:
        """Check if a line contains ion data (mz intensity)."""
        parts = line.split()
        if len(parts) >= 2:
            try:
                float(parts[0])  # mz
                float(parts[1])  # intensity
                return True
            except ValueError:
                return False
        return False
    
    def get_all_metadata_keys(self) -> List[str]:
        """Get all unique metadata keys across all spectra."""
        all_keys = set()
        for spectrum in self.spectra:
            all_keys.update(spectrum.metadata.keys())
        return sorted(list(all_keys))
    
    def get_unique_values_for_key(self, key: str) -> List[str]:
        """Get all unique values for a specific metadata key."""
        values = set()
        for spectrum in self.spectra:
            if key in spectrum.metadata:
                values.add(spectrum.metadata[key])
        return sorted(list(values))
    
    def rename_key_in_all_spectra(self, old_key: str, new_key: str):
        """Rename a metadata key in all spectra."""
        for spectrum in self.spectra:
            spectrum.rename_metadata_key(old_key, new_key)
    
    def update_key_value_in_selected_spectra(self, spectrum_ids: List[int], key: str, value: str):
        """Update a metadata value in selected spectra."""
        for spectrum in self.spectra:
            if spectrum.spectrum_id in spectrum_ids:
                spectrum.update_metadata(key, value)
