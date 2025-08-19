#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify selection works correctly with grouping.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_grouped_selection():
    """Test selection with grouped spectra."""
    try:
        import tkinter as tk
        from mgfexplorer.mgf_parser import MGFParser, Spectrum
        from mgfexplorer.gui_components import SpectrumTreeView
        
        # Create test data
        parser = MGFParser()
        
        # Create test spectra with different metadata values
        for i in range(1, 11):
            spectrum = Spectrum(i)
            spectrum.add_metadata("TITLE", f"Spectrum_{i}")
            spectrum.add_metadata("CHARGE", str(2 if i <= 5 else 3))
            spectrum.add_metadata("PEPMASS", str(100.0 + i))
            spectrum.set_ions([100.0 + i, 200.0 + i], [1000.0, 500.0])
            parser.spectra.append(spectrum)
        
        print(f"Created {len(parser.spectra)} test spectra")
        
        # Create GUI components
        root = tk.Tk()
        root.withdraw()
        
        tree_view = SpectrumTreeView(root)
        tree_view.parser = parser
        
        # Test 1: No grouping
        print("\nTest 1: Selection without grouping")
        tree_view.selected_grouping_tags = []
        tree_view.load_data(parser)
        
        # Try to select spectra 1, 3, 5
        test_ids = [1, 3, 5]
        tree_view.select_spectra_by_ids(test_ids)
        selected = tree_view.get_selected_spectrum_ids()
        print(f"Requested: {test_ids}, Selected: {selected}")
        print(f"Match: {set(test_ids) == set(selected)}")
        
        # Test 2: With grouping by CHARGE
        print("\nTest 2: Selection with CHARGE grouping")
        tree_view.selected_grouping_tags = ["CHARGE"]
        tree_view.load_data(parser)
        
        # Try to select spectra 1, 3, 5 (should be in CHARGE=2 group)
        tree_view.select_spectra_by_ids(test_ids)
        selected = tree_view.get_selected_spectrum_ids()
        print(f"Requested: {test_ids}, Selected: {selected}")
        print(f"Match: {set(test_ids) == set(selected)}")
        
        # Test 3: Select spectra from different groups
        print("\nTest 3: Selection across different groups")
        mixed_ids = [2, 7]  # One from CHARGE=2, one from CHARGE=3
        tree_view.select_spectra_by_ids(mixed_ids)
        selected = tree_view.get_selected_spectrum_ids()
        print(f"Requested: {mixed_ids}, Selected: {selected}")
        print(f"Match: {set(mixed_ids) == set(selected)}")
        
        root.destroy()
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing grouped selection...")
    if test_grouped_selection():
        print("\n+ Grouped selection test completed!")
    else:
        print("\n- Grouped selection test failed!")
        sys.exit(1)
