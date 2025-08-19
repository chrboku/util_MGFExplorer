import sys
import os
sys.path.insert(0, 'src')

from mgfexplorer.mgf_parser import MGFParser

def test_metadata_editing():
    """Test metadata editing functionality."""
    parser = MGFParser()
    mgf_file = os.path.join('example', 'sirius_pos.mgf')
    
    print("Testing metadata editing functionality...")
    
    try:
        # Parse the file
        spectra = parser.parse_file(mgf_file)
        print(f"Loaded {len(spectra)} spectra")
        
        # Test key renaming
        print("\n=== Testing Key Renaming ===")
        original_key = "FEATURE_ID"
        new_key = "FEATURE_ID_RENAMED"
        
        print(f"Original keys: {parser.get_all_metadata_keys()}")
        
        # Rename key for all spectra
        parser.rename_key_in_all_spectra(original_key, new_key)
        
        print(f"After renaming '{original_key}' to '{new_key}': {parser.get_all_metadata_keys()}")
        
        # Test value updating
        print("\n=== Testing Value Updating ===")
        test_spectrum_ids = [1, 2, 3]  # First 3 spectra
        test_key = new_key
        new_value = "TEST_VALUE"
        
        # Show original values
        print("Original values:")
        for spectrum in spectra[:3]:
            print(f"  Spectrum {spectrum.spectrum_id}: {spectrum.get_metadata_value(test_key)}")
        
        # Update values
        parser.update_key_value_in_selected_spectra(test_spectrum_ids, test_key, new_value)
        
        print(f"\nAfter updating to '{new_value}':")
        for spectrum in spectra[:3]:
            print(f"  Spectrum {spectrum.spectrum_id}: {spectrum.get_metadata_value(test_key)}")
        
        # Test adding missing key
        print("\n=== Testing Adding Missing Key ===")
        missing_key = "NEW_TEST_KEY"
        default_value = "DEFAULT_VALUE"
        
        # Add to a few spectra manually first
        spectra[0].add_metadata(missing_key, "EXISTING_VALUE")
        spectra[1].add_metadata(missing_key, "ANOTHER_VALUE")
        
        print(f"Before adding missing key '{missing_key}':")
        for i, spectrum in enumerate(spectra[:5]):
            value = spectrum.get_metadata_value(missing_key)
            print(f"  Spectrum {spectrum.spectrum_id}: {value if value else 'NOT PRESENT'}")
        
        # Add missing key to all spectra
        parser.add_missing_key_to_all_spectra(missing_key, default_value)
        
        print(f"\nAfter adding missing key with default '{default_value}':")
        for i, spectrum in enumerate(spectra[:5]):
            value = spectrum.get_metadata_value(missing_key)
            print(f"  Spectrum {spectrum.spectrum_id}: {value}")
        
        print("\n✅ All metadata editing tests passed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_metadata_editing()
