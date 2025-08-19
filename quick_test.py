import sys
import os
sys.path.insert(0, 'src')

from mgfexplorer.mgf_parser import MGFParser

# Test the parser
parser = MGFParser()
mgf_file = os.path.join('example', 'sirius_pos.mgf')
print(f'Parsing file: {mgf_file}')

try:
    spectra = parser.parse_file(mgf_file)
    print(f'Successfully parsed {len(spectra)} spectra')

    if spectra:
        first_spectrum = spectra[0]
        print(f'First spectrum ID: {first_spectrum.spectrum_id}')
        print(f'Metadata keys: {list(first_spectrum.metadata.keys())}')
        print(f'Ion data shape: {first_spectrum.ions.shape}')
        
        # Show all unique metadata keys
        all_keys = parser.get_all_metadata_keys()
        print(f'All metadata keys: {all_keys}')
        
        # Show some sample ion data
        if first_spectrum.ions.size > 0:
            print(f'Sample ions (first 5):')
            for i in range(min(5, len(first_spectrum.ions))):
                mz, intensity = first_spectrum.ions[i]
                print(f'  m/z: {mz:.6f}, intensity: {intensity:.3f}')
                
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
