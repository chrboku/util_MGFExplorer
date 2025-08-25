# MGF Explorer

A comprehensive GUI tool for exploring and editing MGF (Mascot Generic Format) files containing mass spectrometry data.

## Features

- **MGF File Parsing**: Parse MGF files with all content between "BEGIN IONS" and "END IONS" as MS/MS spectra
- **Metadata Management**: 
  - Parse key=value metadata to dictionaries
  - Handle duplicate keys with automatic numbering (_1, _2, etc.)
  - Edit and rename metadata keys
  - Apply changes across multiple spectra
- **Ion Data Visualization**:
  - Parse ion data as 2D numpy arrays (m/z, intensity)
  - Display spectra as stick charts with m/z on x-axis
  - View multiple spectra simultaneously
- **Interactive GUI**:
  - Tree view for spectrum selection and grouping
  - Metadata editor with unique value tracking
  - Table view for ion data
  - Real-time spectrum visualization

## Installation

This project uses `uv` for dependency management. Make sure you have `uv` installed.

1. Clone or download this repository
2. Navigate to the project directory
3. Install dependencies:
```bash
uv sync
```

## Usage

### Running the Application

```bash
uv run mgfexplorer
```

### Using the GUI

1. **Open an MGF File**: Use File → Open MGF File... or Ctrl+O
2. **Select Spectra**: 
   - Use the tree view on the left to select one or more spectra
   - Enable grouping options to organize spectra by metadata fields
3. **Filter Spectra**:
   - Use the filter box to search through spectra
   - Default: searches all metadata fields (e.g., `positive`)
   - `$$ key: value` - search only specific key for exact match (e.g., `$$ IONMODE: Positive`)
   - `$$$ key: regex` - search only specific key using regex (e.g., `$$$ CHARGE: ^[12]\+`)
4. **Edit Metadata**:
   - View and edit metadata in the center panel
   - Rename keys or update values
   - Changes apply to all selected spectra
5. **View Data**:
   - Ion data tables appear in the bottom-left
   - Spectrum visualization in the bottom-right
   - Stick charts show m/z vs intensity

### Key Components

#### Left Panel - Spectrum Tree View
- Lists all spectra in the MGF file
- Advanced filtering: search all fields, specific keys, or use regex patterns
- Grouping checkboxes to organize by metadata fields
- Multi-select support for batch operations

#### Center Panel - Metadata Editor
- Shows metadata keys and values for selected spectra
- Edit values directly or rename keys
- Shows unique values across all spectra for each key

#### Bottom-Left Panel - Ion Data Tables
- Tabbed interface showing ion data for each selected spectrum
- m/z and intensity values in table format

#### Bottom-Right Panel - Spectrum Visualization
- Stick chart visualization of selected spectra
- Automatic scaling and multiple spectrum support

## Example Data

The `example/sirius_pos.mgf` file contains sample mass spectrometry data for testing the application.

## Technical Details

### MGF File Format Support

The parser handles:
- Standard BEGIN IONS / END IONS blocks
- Key=value metadata parsing
- Numeric ion data (m/z intensity pairs)
- Automatic duplicate key handling
- Robust error handling for malformed data

### Data Structures

- **Spectrum Class**: Represents individual MS/MS spectra
  - `spectrum_id`: Unique identifier
  - `metadata`: Dictionary of key-value pairs
  - `ions`: 2D numpy array [m/z, intensity]

- **MGFParser Class**: Handles file parsing and data management
  - File parsing and validation
  - Metadata key management
  - Batch operations across spectra

### Dependencies

- **numpy**: Numerical data handling
- **matplotlib**: Spectrum visualization
- **tkinter**: GUI framework (built-in with Python)

## Development

### Project Structure

```
src/
├── mgfexplorer/
│   ├── __init__.py
│   ├── main.py           # Entry point
│   ├── app.py            # Main application window
│   ├── mgf_parser.py     # MGF file parsing logic
│   └── gui_components.py # GUI components
example/
└── sirius_pos.mgf        # Sample data file
```

### Running Tests

```bash
uv run python quick_test.py
```

## License

This project is open source. See the license file for details.
