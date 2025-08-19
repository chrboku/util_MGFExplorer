# MGF Explorer - Feature Summary

## Recent Updates

### Menu Reorganization

#### File Menu
- Open MGF File... (Ctrl+O)
- Exit

#### Edit Menu (NEW)
- Add New Key-Value Pair...
- Convert Keys to UPPERCASE
- Convert Keys to lowercase

#### View Menu (NEW)
- Spectrum Name
  - Numbered (default - shows spectrum ID)
  - Any metadata field (shows ID: field=value)

#### Help Menu
- About

### Metadata Editor Improvements

#### Inline Editing
- **Double-click** on Key or Value cells to edit directly
- **Enter** to save changes
- **Escape** to cancel editing
- No more separate text boxes and buttons

#### Key Editing Features
- Rename keys with conflict resolution:
  - Update only (for spectra without new key)
  - Merge (overwrite existing values)
  - Abort operation
- Add missing keys to spectra that don't have them

#### Value Editing Features
- Update values in all selected spectra
- Add key-value pairs to spectra that don't have the key

### Tag-Based Grouping System

#### Grouping Interface
- Single text field for comma-separated metadata keys
- Real-time display of available metadata keys
- Apply button to update grouping

#### Hierarchical Tree View
- Multi-level grouping (first tag → second tag → ...)
- Group selection selects all spectra in group and subgroups
- Individual spectrum selection within groups
- Expandable/collapsible groups

### Spectrum Naming Options

#### Numbered Display (Default)
- Shows: "Spectrum {ID}"

#### Field-Based Display
- Shows: "{ID}: {field}={value}"
- Handles missing values: "{ID}: {field}=<missing>"
- Available for any metadata field

### Core Features (Previously Implemented)

#### MGF File Parsing
- ✅ Parse BEGIN IONS / END IONS blocks
- ✅ Handle key=value metadata with duplicate numbering
- ✅ Parse ion data as 2D numpy arrays (m/z, intensity)
- ✅ Robust error handling

#### GUI Layout
- ✅ Left: Spectrum tree with tag-based grouping
- ✅ Center: Metadata editor with inline editing
- ✅ Bottom-left: Ion data tables
- ✅ Bottom-right: Spectrum visualization

#### Data Visualization
- ✅ Stick charts with m/z vs intensity
- ✅ Multiple spectrum support
- ✅ Automatic scaling and labeling

## Usage Guide

### Opening Files
1. Use **File → Open MGF File...** or **Ctrl+O**
2. Select your MGF file
3. Data loads automatically with numbered spectrum names

### Grouping Spectra
1. Enter metadata keys in the grouping field (e.g., "MSLEVEL,CHARGE")
2. Click **Apply Grouping**
3. Tree view shows hierarchical structure
4. Click groups to select all contained spectra

### Editing Metadata
1. Select one or more spectra
2. **Double-click** on any Key or Value cell
3. Edit the text and press **Enter**
4. Changes apply based on edit type:
   - **Key edits**: Asks about global rename with conflict resolution
   - **Value edits**: Updates all selected spectra

### Adding New Metadata
1. Use **Edit → Add New Key-Value Pair...**
2. Enter key name and value
3. Choose scope: Selected spectra or All loaded spectra
4. Click **Add**

### Converting Key Cases
1. Use **Edit → Convert Keys to UPPERCASE** or **lowercase**
2. Confirms before applying to all spectra
3. Updates all displays automatically

### Changing Spectrum Names
1. Use **View → Spectrum Name**
2. Choose **Numbered** for simple ID display
3. Choose any metadata field for descriptive names
4. Tree updates automatically

## Technical Notes

### Dependencies
- numpy: Data processing
- matplotlib: Visualization
- tkinter: GUI framework (built-in)

### Data Structure
- Spectrum objects with metadata dictionaries and ion arrays
- MGFParser handles file I/O and batch operations
- GUI components are modular and interconnected

### Performance
- Efficient parsing of large MGF files
- Real-time updates for metadata changes
- Optimized tree view rendering for hierarchical data
