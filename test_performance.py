#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test script to verify the performance optimizations work correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all imports work correctly."""
    try:
        from mgfexplorer.mgf_parser import MGFParser, Spectrum
        from mgfexplorer.gui_components import (
            SpectrumTreeView, 
            MetadataEditor, 
            SpectrumVisualization, 
            IonDataTable,
            CosineSimilarityVisualization
        )
        from mgfexplorer.app import MGFExplorerApp
        print("+ All imports successful")
        return True
    except ImportError as e:
        print(f"- Import error: {e}")
        return False

def test_cosine_similarity():
    """Test the cosine similarity calculation."""
    try:
        from mgfexplorer.mgf_parser import MGFParser, Spectrum
        import numpy as np
        
        parser = MGFParser()
        
        # Create two test spectra
        spec1 = Spectrum(1)
        spec1.set_ions([100.0, 200.0, 300.0], [1000.0, 2000.0, 1500.0])
        
        spec2 = Spectrum(2)
        spec2.set_ions([100.1, 200.1, 300.1], [900.0, 2100.0, 1400.0])
        
        # Test similarity calculation
        similarity = parser.calculate_cosine_similarity(spec1, spec2, mz_tolerance=0.2)
        
        print(f"+ Cosine similarity calculation works: {similarity:.3f}")
        
        # Test similarity matrix
        parser.spectra = [spec1, spec2]
        matrix = parser.calculate_similarity_matrix([1, 2])
        
        print(f"+ Similarity matrix calculation works: shape {matrix.shape}")
        return True
        
    except Exception as e:
        print(f"- Cosine similarity test failed: {e}")
        return False

def test_performance_limits():
    """Test performance limit constants."""
    try:
        from mgfexplorer.gui_components import CosineSimilarityVisualization
        import tkinter as tk
        
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        viz = CosineSimilarityVisualization(root)
        
        print(f"+ Performance limits: AUTO={viz.MAX_SPECTRA_AUTO}, MANUAL={viz.MAX_SPECTRA_MANUAL}")
        
        root.destroy()
        return True
        
    except Exception as e:
        print(f"- Performance limits test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing MGF Explorer performance optimizations...")
    print()
    
    all_passed = True
    all_passed &= test_imports()
    all_passed &= test_cosine_similarity() 
    all_passed &= test_performance_limits()
    
    print()
    if all_passed:
        print("+ All tests passed!")
    else:
        print("- Some tests failed!")
        sys.exit(1)
