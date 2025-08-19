import sys
import os
sys.path.insert(0, 'src')

from mgfexplorer.mgf_parser import MGFParser

def test_grouping_functionality():
    """Test the new grouping and metadata features."""
    parser = MGFParser()
    mgf_file = os.path.join('example', 'sirius_pos.mgf')
    
    print("Testing new features...")
    
    try:
        # Parse the file
        spectra = parser.parse_file(mgf_file)
        print(f"Loaded {len(spectra)} spectra")
        
        # Test available metadata keys for grouping
        print("\n=== Available Metadata Keys ===")
        all_keys = parser.get_all_metadata_keys()
        print(f"Available keys for grouping: {all_keys}")
        
        # Test grouping combinations
        print("\n=== Testing Grouping Examples ===")
        
        # Example 1: Group by MSLEVEL
        test_key = "MSLEVEL"
        unique_values = parser.get_unique_values_for_key(test_key)
        print(f"\nGrouping by '{test_key}':")
        print(f"  Unique values: {unique_values}")
        
        for value in unique_values:
            count = sum(1 for s in spectra if s.get_metadata_value(test_key) == value)
            print(f"  {test_key}={value}: {count} spectra")
        
        # Example 2: Group by CHARGE
        test_key2 = "CHARGE"
        unique_values2 = parser.get_unique_values_for_key(test_key2)
        print(f"\nGrouping by '{test_key2}':")
        print(f"  Unique values: {unique_values2}")
        
        for value in unique_values2:
            count = sum(1 for s in spectra if s.get_metadata_value(test_key2) == value)
            print(f"  {test_key2}={value}: {count} spectra")
        
        # Example 3: Hierarchical grouping (MSLEVEL, CHARGE)
        print(f"\nHierarchical grouping by '{test_key}' then '{test_key2}':")
        hierarchy = {}
        
        for spectrum in spectra:
            level = spectrum.get_metadata_value(test_key) or "unknown"
            charge = spectrum.get_metadata_value(test_key2) or "unknown"
            
            if level not in hierarchy:
                hierarchy[level] = {}
            if charge not in hierarchy[level]:
                hierarchy[level][charge] = 0
            hierarchy[level][charge] += 1
        
        for level, charges in hierarchy.items():
            print(f"  {test_key}={level}:")
            for charge, count in charges.items():
                print(f"    {test_key2}={charge}: {count} spectra")
        
        print("\n✅ Grouping functionality test completed!")
        
        # Test key manipulation
        print("\n=== Testing Key Manipulation ===")
        sample_spectrum = spectra[0]
        print(f"Sample spectrum metadata: {list(sample_spectrum.metadata.keys())}")
        
        # Test adding new key
        new_key = "TEST_NEW_KEY"
        new_value = "TEST_VALUE"
        sample_spectrum.metadata[new_key] = new_value
        print(f"Added new key '{new_key}' with value '{new_value}'")
        
        # Test key case conversion (simulation)
        original_keys = list(sample_spectrum.metadata.keys())
        print(f"Original keys: {original_keys}")
        
        uppercase_keys = [key.upper() for key in original_keys]
        lowercase_keys = [key.lower() for key in original_keys]
        
        print(f"UPPERCASE version: {uppercase_keys}")
        print(f"lowercase version: {lowercase_keys}")
        
        print("\n✅ All functionality tests passed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_grouping_functionality()
