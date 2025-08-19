#!/usr/bin/env python3
"""
Test script for the MGF parser functionality.
"""

import sys
import os

# Add the src directory to the Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

from mgfexplorer.mgf_parser import MGFParser

def test_parser():
    """Test the MGF parser with the example file."""
    parser = MGFParser()
    
    # Parse the example file
    mgf_file = os.path.join(os.path.dirname(__file__), 'example', 'sirius_pos.mgf')
    
    if not os.path.exists(mgf_file):
        print(f"Error: File {mgf_file} not found")
        return
        
    print(f"Parsing file: {mgf_file}")
    
    try:
        spectra = parser.parse_file(mgf_file)
        print(f"Successfully parsed {len(spectra)} spectra")
        
        if spectra:
            # Show info about first spectrum
            first_spectrum = spectra[0]
            print(f"\nFirst spectrum (ID: {first_spectrum.spectrum_id}):")
            print("Metadata:")
            for key, value in first_spectrum.metadata.items():
                print(f"  {key}: {value}")
                
            print(f"Ion data shape: {first_spectrum.ions.shape}")
            if first_spectrum.ions.size > 0:
                print(f"m/z range: {first_spectrum.ions[:, 0].min():.3f} - {first_spectrum.ions[:, 0].max():.3f}")
                print(f"Intensity range: {first_spectrum.ions[:, 1].min():.3f} - {first_spectrum.ions[:, 1].max():.3f}")
                
            # Show all unique metadata keys
            all_keys = parser.get_all_metadata_keys()
            print(f"\nAll metadata keys found ({len(all_keys)}):")
            for key in all_keys:
                unique_values = parser.get_unique_values_for_key(key)
                print(f"  {key}: {len(unique_values)} unique values")
                
    except Exception as e:
        print(f"Error parsing file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_parser()
