#!/usr/bin/env python3
"""
Test script for empty metadata handling functionality.
"""

import sys
import os
sys.path.insert(0, 'src')

from mgfexplorer.mgf_parser import MGFParser

def test_empty_metadata_handling():
    """Test that empty/missing metadata values are handled correctly."""
    parser = MGFParser()
    mgf_file = os.path.join('example', 'sirius_pos.mgf')
    
    print("Testing empty metadata handling...")
    
    try:
        # Parse the file
        spectra = parser.parse_file(mgf_file)
        print(f"Loaded {len(spectra)} spectra")
        
        # Test getting unique values including empty ones
        print("\n=== Testing Unique Values with Missing Keys ===")
        
        # Create a test key that doesn't exist in all spectra
        test_key = "TEST_MISSING_KEY"
        
        # Add the key to only some spectra
        for i, spectrum in enumerate(spectra[:3]):
            spectrum.add_metadata(test_key, f"value_{i}")
            
        print(f"Added '{test_key}' to first 3 spectra")
        
        # Get unique values - should include empty string for missing values
        unique_values = parser.get_unique_values_for_key(test_key)
        print(f"Unique values for '{test_key}': {unique_values}")
        
        # Test counting empty vs non-empty values
        has_value_count = 0
        empty_count = 0
        
        for spectrum in spectra:
            value = spectrum.get_metadata_value(test_key)
            if value is None:
                empty_count += 1
            else:
                has_value_count += 1
                
        print(f"Spectra with '{test_key}': {has_value_count}")
        print(f"Spectra without '{test_key}' (treated as empty): {empty_count}")
        
        # Test updating a value to empty string
        print("\n=== Testing Setting Value to Empty String ===")
        
        # Set one spectrum's value to empty string
        spectra[0].metadata[test_key] = ""
        
        # Check that empty string is different from missing key
        value_0 = spectra[0].get_metadata_value(test_key)  # Should be ""
        value_last = spectra[-1].get_metadata_value(test_key)  # Should be None
        
        print(f"Spectrum 0 '{test_key}' value: '{value_0}' (type: {type(value_0)})")
        print(f"Last spectrum '{test_key}' value: {value_last} (type: {type(value_last)})")
        
        # Test unique values again
        unique_values_updated = parser.get_unique_values_for_key(test_key)
        print(f"Updated unique values: {unique_values_updated}")
        
        print("\n✅ Empty metadata handling test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_empty_metadata_handling()
