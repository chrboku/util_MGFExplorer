"""
Main application window for the MGF Explorer.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import os
import re
import numpy as np
from typing import Any, Dict, List, Tuple
from .mgf_parser import MGFParser, Spectrum
from .config import load_config, save_config
from .gui_components import (
    SpectrumTreeView,
    MetadataEditor,
    SpectrumVisualization,
    IonDataTable,
    CosineSimilarityVisualization,
    FileLoadingDialog,
    SmartsFilterDialog,
    IntensityFilterDialog,
    FragmentAnnotationDialog,
    CanonicalSmilesDialog,
    ProgressDialog,
    PPMDeviationPlotDialog,
    SpectrumPopupWindow,
    FragmentDistributionDialog,
)
from .options_dialog import OptionsDialog
from .molecular_formula import (
    FragmentAnnotator,
    MolecularFormula,
)
from concurrent.futures import ThreadPoolExecutor, as_completed

# Try to import tkinterdnd2 for drag and drop support
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    DRAG_DROP_AVAILABLE = True
except ImportError:
    DRAG_DROP_AVAILABLE = False

# Try to import RDKit for SMARTS filtering
try:
    from rdkit import Chem
    from rdkit.Chem import Draw

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


class MGFExplorerApp:
    """Main application class for the MGF Explorer."""

    def __init__(self):
        # Initialize with drag and drop support if available
        if DRAG_DROP_AVAILABLE:
            self.root = TkinterDnD.Tk()
        else:
            self.root = tk.Tk()

        self.root.title("MGF Explorer")
        self.root.geometry("1400x900")
        self.root.minsize(1000, 700)

        self.parser = MGFParser()
        self.current_file = None
        self.used_prefixes = set()  # Track used prefixes to prevent conflicts
        self.config_data = load_config()

        # Selection debouncing
        self.selection_update_job = None
        self.SELECTION_DELAY_MS = 300  # Wait 300ms before updating heavy components

        self._create_widgets()
        self._create_menu()
        self._apply_persistent_options()
        self._setup_drag_drop()

        self._get_ppm_tolerance_for_mz_cache = {}

    def _create_menu(self):
        """Create the application menu."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(
            label="Load MGF File(s)...", command=self.open_files, accelerator="Ctrl+O"
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Clear All Data", command=self.clear_data, accelerator="Ctrl+N"
        )
        file_menu.add_separator()

        export_menu = tk.Menu(file_menu, tearoff=0)
        export_menu.add_command(
            label="Export All Spectra...",
            command=self.export_all_spectra,
            accelerator="Ctrl+E",
        )
        export_menu.add_command(
            label="Export Grouped Spectra...",
            command=self.export_grouped_spectra,
        )
        file_menu.add_cascade(label="Export Spectra", menu=export_menu)

        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(
            label="Add New Key-Value Pair...", command=self._add_new_key_value
        )
        edit_menu.add_separator()
        edit_menu.add_command(
            label="Convert Keys to UPPERCASE", command=self._keys_to_uppercase
        )
        edit_menu.add_command(
            label="Convert Keys to lowercase", command=self._keys_to_lowercase
        )
        edit_menu.add_command(
            label="Canonicalize SMILES...",
            command=self._canonicalize_smiles_metadata,
        )
        edit_menu.add_separator()

        # Intensity Normalization submenu
        normalize_menu = tk.Menu(edit_menu, tearoff=0)
        edit_menu.add_cascade(label="Intensity Normalization", menu=normalize_menu)

        # Range submenus
        range_01_menu = tk.Menu(normalize_menu, tearoff=0)
        normalize_menu.add_cascade(label="Range 0-1", menu=range_01_menu)
        range_01_menu.add_command(
            label="Relative to most abundant signal",
            command=lambda: self._normalize_intensities("max", 1.0),
        )
        range_01_menu.add_command(
            label="Relative to total sum of all signals",
            command=lambda: self._normalize_intensities("sum", 1.0),
        )

        range_0100_menu = tk.Menu(normalize_menu, tearoff=0)
        normalize_menu.add_cascade(label="Range 0-100", menu=range_0100_menu)
        range_0100_menu.add_command(
            label="Relative to most abundant signal",
            command=lambda: self._normalize_intensities("max", 100.0),
        )
        range_0100_menu.add_command(
            label="Relative to total sum of all signals",
            command=lambda: self._normalize_intensities("sum", 100.0),
        )

        range_01000_menu = tk.Menu(normalize_menu, tearoff=0)
        normalize_menu.add_cascade(label="Range 0-1000", menu=range_01000_menu)
        range_01000_menu.add_command(
            label="Relative to most abundant signal",
            command=lambda: self._normalize_intensities("max", 1000.0),
        )
        range_01000_menu.add_command(
            label="Relative to total sum of all signals",
            command=lambda: self._normalize_intensities("sum", 1000.0),
        )

        edit_menu.add_separator()
        edit_menu.add_command(
            label="Delete Selected Spectra", command=self._delete_selected_spectra
        )
        edit_menu.add_separator()
        edit_menu.add_command(
            label="Calculate Average Spectrum per Group",
            command=self._calculate_average_spectra,
        )
        edit_menu.add_separator()

        # Regex Update submenu
        regex_menu = tk.Menu(edit_menu, tearoff=0)
        edit_menu.add_command(label="Regex Update", command=self._open_regex_editor)

        # Filter menu
        filter_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Filter", menu=filter_menu)
        filter_menu.add_command(
            label="SMARTS Substructure Filter...", command=self._open_smarts_filter
        )
        filter_menu.add_command(
            label="Intensity Filter...", command=self._open_intensity_filter
        )

        # Fragment annotation menu
        fragment_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Fragment annotation", menu=fragment_menu)
        fragment_menu.add_command(
            label="Generate subformulas", command=self._generate_subformulas
        )
        fragment_menu.add_separator()
        fragment_menu.add_command(
            label="Clear all annotations", command=self._clear_all_annotations
        )

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)

        # Spectrum Name submenu
        spectrum_name_menu = tk.Menu(view_menu, tearoff=0)
        view_menu.add_cascade(label="Spectrum Name", menu=spectrum_name_menu)

        self.spectrum_name_var = tk.StringVar(value="Numbered")
        spectrum_name_menu.add_radiobutton(
            label="Numbered",
            variable=self.spectrum_name_var,
            value="Numbered",
            command=self._update_spectrum_names,
        )

        # Will be populated when data is loaded
        self.spectrum_name_menu = spectrum_name_menu

        view_menu.add_separator()
        view_menu.add_command(
            label="Fragment Distribution...", command=self._show_fragment_distribution
        )

        # Options menu
        options_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Options", menu=options_menu)
        options_menu.add_command(
            label="Preferences...", command=self._open_options_dialog
        )

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

        # Bind keyboard shortcuts
        self.root.bind("<Control-o>", lambda e: self.open_files())
        self.root.bind("<Control-n>", lambda e: self.clear_data())
        self.root.bind("<Control-e>", lambda e: self.export_all_spectra())

    def _setup_drag_drop(self):
        """Set up drag and drop functionality for MGF files."""
        if not DRAG_DROP_AVAILABLE:
            # Add a note about drag and drop not being available
            print(
                "Note: Drag and drop support not available. Install tkinterdnd2 for this feature."
            )
            return

        # Enable drag and drop for files
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind("<<Drop>>", self._on_file_drop)

        # Optional: Add visual feedback during drag
        self.root.dnd_bind("<<DragEnter>>", self._on_drag_enter)
        self.root.dnd_bind("<<DragLeave>>", self._on_drag_leave)

    def _apply_persistent_options(self):
        """Apply loaded configuration to UI components."""
        grouping_tags = self.config_data.get("default_grouping_tags", [])
        if hasattr(self, "spectrum_tree"):
            self.spectrum_tree.set_grouping_tags(grouping_tags)

        metadata_groups = self.config_data.get("metadata_groups", [])
        if hasattr(self, "metadata_editor"):
            self.metadata_editor.set_metadata_groups(metadata_groups)

    def _open_options_dialog(self):
        """Launch the options dialog and persist any updates."""
        dialog = OptionsDialog(self.root, self.config_data)
        if not dialog.result:
            return

        self.config_data = dialog.result

        try:
            save_config(self.config_data)
        except Exception as exc:
            messagebox.showerror(
                "Save Options",
                f"Failed to save options: {exc}",
            )
            return

        self._apply_persistent_options()

        if hasattr(self, "status_var"):
            self.status_var.set("Options updated")

    def _on_drag_enter(self, event):
        """Handle drag enter event."""
        self.status_var.set("Drop MGF file to open...")

    def _on_drag_leave(self, event):
        """Handle drag leave event."""
        self._restore_status()

    def _on_file_drop(self, event):
        """Handle file drop event."""
        try:
            # Get the dropped files
            files = event.data  # .split()

            # files is '{}' or '{} {}'
            if "} {" in files:
                files = files.split("} {")
                for fi in range(len(files)):
                    files[fi] = files[fi].strip().lstrip("{")
                    files[fi] = files[fi].rstrip("}")
            else:
                files = [files.strip().lstrip("{").rstrip("}")]

            if not files:
                return

            # Process the dropped file
            for file_path in files:
                self._process_dropped_file(file_path)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process dropped file: {str(e)}")
            self._restore_status()

    def _process_dropped_file(self, file_path):
        """Process a file that was dropped onto the window."""
        try:
            # Check if it's an MGF file
            if not file_path.lower().endswith(".mgf"):
                messagebox.showwarning(
                    "Invalid File Type", "Please drop an MGF file (.mgf extension)."
                )
                return

            # Check if file exists
            if not os.path.exists(file_path):
                messagebox.showerror(
                    "File Not Found", f"The file does not exist:\n{file_path}"
                )
                return

            # Load the file
            self._load_mgf_file(file_path)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process dropped file: {str(e)}")

    def _restore_status(self):
        """Restore the status bar to its normal state."""
        if self.current_file:
            filename = os.path.basename(self.current_file)
            spectrum_count = len(self.parser.spectra) if self.parser else 0
            self.status_var.set(f"Loaded {spectrum_count} spectra from {filename}")
        else:
            self.status_var.set("Ready - Open an MGF file to get started")

    def _load_mgf_file(self, file_path):
        """Load an MGF file and append to existing data."""
        try:
            # Show file loading options dialog
            dialog = FileLoadingDialog(self.root, file_path, self.used_prefixes)

            if dialog.result is None:
                # User cancelled
                return

            loading_options = dialog.result

            self.status_var.set("Loading file...")
            self.root.update()

            # Check if we have existing data to append to
            is_first_file = len(self.parser.spectra) == 0

            if is_first_file:
                # First file - use parse_file which clears existing data
                spectra = self.parser.parse_file(file_path)
            else:
                # Additional file - use parse_and_append_file to add to existing data
                # Calculate ID offset to avoid conflicts
                id_offset = self._calculate_id_offset()
                spectra = self.parser.parse_and_append_file(file_path, id_offset)

            if not spectra:
                messagebox.showwarning("Warning", "No spectra found in the file.")
                return

            # Apply loading options to newly loaded spectra only
            self._apply_loading_options(spectra, loading_options)

            # Update current file reference (keep track of the most recent file)
            self.current_file = file_path

            # Update components
            self.spectrum_tree.load_data(self.parser)
            self._update_spectrum_name_menu()
            self._set_components_enabled(True)

            # Update status
            filename = os.path.basename(file_path)
            total_count = len(self.parser.spectra)
            new_count = len(spectra)

            if is_first_file:
                self.status_var.set(f"Loaded {new_count} spectra from {filename}")
                self.root.title(f"MGF Explorer - {filename}")
            else:
                self.status_var.set(
                    f"Added {new_count} spectra from {filename}. Total: {total_count} spectra"
                )
                # Update title to show multiple files
                self.root.title(
                    f"MGF Explorer - {total_count} spectra from multiple files"
                )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
            self.status_var.set("Error loading file")

    def _calculate_id_offset(self):
        """Calculate ID offset for appending new spectra to avoid conflicts."""
        if not self.parser.spectra:
            return 0

        try:
            # Try to get the maximum numeric ID, fall back to length if IDs are strings
            numeric_ids = [
                s.spectrum_id
                for s in self.parser.spectra
                if isinstance(s.spectrum_id, int)
            ]
            if numeric_ids:
                return max(numeric_ids)
            else:
                # All IDs are strings, use length as offset
                return len(self.parser.spectra)
        except (ValueError, TypeError):
            # Fallback to using length
            return len(self.parser.spectra)

    def _apply_loading_options(self, spectra, options):
        """Apply loading options to the loaded spectra."""
        database_identifier = options.get("database_identifier")
        prefix = options.get("prefix")

        # Apply database identifier to all spectra
        if database_identifier:
            for spectrum in spectra:
                spectrum.update_metadata("database_identifier", database_identifier)

        # Apply prefix to spectrum IDs and track the prefix
        if prefix:
            for i, spectrum in enumerate(spectra):
                # Create new spectrum ID with prefix
                new_id = f"{prefix}_{spectrum.spectrum_id}"

                # Update any existing metadata that might reference the spectrum ID
                if "TITLE" in spectrum.metadata:
                    # If there's a TITLE field, prefix it as well
                    original_title = spectrum.metadata["TITLE"]
                    spectrum.update_metadata("TITLE", f"{prefix}_{original_title}")
                else:
                    # If no TITLE, create one with the prefixed ID
                    spectrum.update_metadata("TITLE", new_id)

                # Update the spectrum ID itself (though this is mainly for internal tracking)
                spectrum.spectrum_id = new_id

            # Track this prefix as used
            self.used_prefixes.add(prefix)

    def _create_widgets(self):
        """Create the main application widgets."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Create paned windows for layout
        # Main horizontal split
        main_paned = ttk.PanedWindow(main_frame, orient="horizontal")
        main_paned.pack(fill="both", expand=True)

        # Left panel (spectrum tree)
        left_frame = ttk.Frame(main_paned, width=300)
        main_paned.add(left_frame, weight=1)

        # Right panel (metadata and visualization)
        right_paned = ttk.PanedWindow(main_paned, orient="vertical")
        main_paned.add(right_paned, weight=3)

        # Top right (metadata editor)
        metadata_frame = ttk.Frame(right_paned, height=300)
        right_paned.add(metadata_frame, weight=1)

        # Bottom right (visualization and tables)
        bottom_paned = ttk.PanedWindow(right_paned, orient="horizontal")
        right_paned.add(bottom_paned, weight=2)

        # Bottom left (ion data table)
        table_frame = ttk.Frame(bottom_paned, width=300)
        bottom_paned.add(table_frame, weight=1)

        # Bottom center (spectrum visualization)
        viz_frame = ttk.Frame(bottom_paned, width=500)
        bottom_paned.add(viz_frame, weight=2)

        # Bottom right (cosine similarity)
        similarity_frame = ttk.Frame(bottom_paned, width=400)
        bottom_paned.add(similarity_frame, weight=1)

        # Create components
        self.spectrum_tree = SpectrumTreeView(
            left_frame, on_selection_changed=self._on_spectrum_selection_changed
        )
        self.spectrum_tree.pack(fill="both", expand=True)

        self.metadata_editor = MetadataEditor(
            metadata_frame, on_metadata_changed=self._on_metadata_changed
        )
        self.metadata_editor.pack(fill="both", expand=True)

        self.ion_table = IonDataTable(table_frame)
        self.ion_table.pack(fill="both", expand=True)

        self.spectrum_viz = SpectrumVisualization(viz_frame)
        self.spectrum_viz.pack(fill="both", expand=True)

        self.similarity_viz = CosineSimilarityVisualization(similarity_frame)
        self.similarity_viz.pack(fill="both", expand=True)

        # Set up communication between ion table and spectrum visualization
        self.ion_table.set_spectrum_viz_callback(self._on_ion_selection_changed)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Open an MGF file to get started")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken")
        status_bar.pack(side="bottom", fill="x")

        # Initial state
        self._set_components_enabled(False)

    def _set_components_enabled(self, enabled: bool):
        """Enable or disable components based on data availability."""
        state = "normal" if enabled else "disabled"

        # This could be expanded to actually disable/enable widgets
        # For now, components handle empty data gracefully
        pass

    def open_files(self):
        """Open and parse one or more MGF files."""
        file_paths = filedialog.askopenfilenames(
            title="Load MGF File(s)",
            filetypes=[("MGF files", "*.mgf"), ("All files", "*.*")],
        )

        if not file_paths:
            return

        # Load files sequentially, each with its own dialog
        for file_path in file_paths:
            self._load_mgf_file(file_path)

    def clear_data(self):
        """Clear all loaded data."""
        if self.parser.spectra:
            result = messagebox.askyesno(
                "Clear Data",
                f"This will clear all {len(self.parser.spectra)} loaded spectra. Continue?",
                icon="warning",
            )
            if not result:
                return

        # Clear data
        self.parser.spectra.clear()
        self.used_prefixes.clear()
        self.current_file = None

        # Clear components
        self.spectrum_tree.load_data(self.parser)
        self.metadata_editor.load_data(self.parser, [])
        self.spectrum_viz.clear_plot()
        self.ion_table.clear_data()
        self.similarity_viz.clear_data()

        # Update UI state
        self._set_components_enabled(False)
        self.status_var.set("Ready - Load MGF files to get started")
        self.root.title("MGF Explorer")

    def _on_spectrum_selection_changed(self, selected_spectrum_ids):
        """Handle spectrum selection changes with debouncing for performance."""
        # Cancel any pending update
        if self.selection_update_job:
            self.root.after_cancel(self.selection_update_job)

        # Update status immediately for responsiveness
        if not selected_spectrum_ids:
            self.status_var.set(f"Ready - {len(self.parser.spectra)} spectra loaded")
        else:
            if len(selected_spectrum_ids) == 1:
                self.status_var.set(f"Selected spectrum {selected_spectrum_ids[0]}")
            else:
                self.status_var.set(f"Selected {len(selected_spectrum_ids)} spectra")

        # Schedule the heavy update with debouncing
        self.selection_update_job = self.root.after(
            self.SELECTION_DELAY_MS,
            lambda: self._update_components_with_selection(selected_spectrum_ids),
        )

    def _update_components_with_selection(self, selected_spectrum_ids):
        """Update all components with the selected spectra."""
        self.selection_update_job = None

        if not selected_spectrum_ids:
            # Clear all displays
            self.metadata_editor.load_data(self.parser, [])
            self.ion_table.load_data(self.parser, [])
            self.spectrum_viz.load_data(self.parser, [])
            self.similarity_viz.load_data(self.parser, [])
        else:
            # Update displays with selected spectra
            # Always update metadata editor (it's lightweight)
            self.metadata_editor.load_data(self.parser, selected_spectrum_ids)

            # For very large selections, disable expensive operations
            if len(selected_spectrum_ids) > 1000:
                # Only show metadata for very large selections
                self.ion_table.load_data(self.parser, [])
                self.spectrum_viz.load_data(self.parser, [])
                self.similarity_viz.load_data(self.parser, [])

                # Update status with performance warning
                self.status_var.set(
                    f"Selected {len(selected_spectrum_ids)} spectra "
                    f"(detailed views disabled for performance)"
                )

            elif len(selected_spectrum_ids) > 100:
                # For large selections, limit what we show in detailed views
                limited_ids = selected_spectrum_ids[:10]
                self.ion_table.load_data(self.parser, limited_ids)
                self.spectrum_viz.load_data(self.parser, limited_ids)

                # Disable similarity calculation for large selections
                self.similarity_viz.load_data(self.parser, [])

                # Update status to reflect the limitation
                self.status_var.set(
                    f"Selected {len(selected_spectrum_ids)} spectra "
                    f"(showing details for first 10, similarity disabled)"
                )

            elif len(selected_spectrum_ids) > 10:
                # Only show first 10 spectra in detailed views
                limited_ids = selected_spectrum_ids[:10]
                self.ion_table.load_data(self.parser, limited_ids)
                self.spectrum_viz.load_data(self.parser, limited_ids)

                # Load similarity data (this will handle its own performance limits)
                self.similarity_viz.load_data(self.parser, selected_spectrum_ids)

                # Update status to reflect the limitation
                self.status_var.set(
                    f"Selected {len(selected_spectrum_ids)} spectra "
                    f"(showing details for first 10)"
                )
            else:
                # Small selection - show everything
                self.ion_table.load_data(self.parser, selected_spectrum_ids)
                self.spectrum_viz.load_data(self.parser, selected_spectrum_ids)
                self.similarity_viz.load_data(self.parser, selected_spectrum_ids)

    def _on_metadata_changed(self):
        """Handle metadata changes."""
        # Check for pending selection from metadata editor
        pending_selection = self.metadata_editor.get_pending_selection()
        if pending_selection:
            # Update the spectrum tree to select the specified spectra
            self.spectrum_tree.select_spectra_by_ids(pending_selection)
            return

        # Refresh the spectrum tree to show updated grouping
        self.spectrum_tree.load_data(self.parser)

        # Update spectrum name menu if needed
        self._update_spectrum_name_menu()

        # Get current selection and refresh other components
        selected_ids = self.spectrum_tree.get_selected_spectrum_ids()
        if selected_ids:
            self.metadata_editor.load_data(self.parser, selected_ids)
            self.ion_table.load_data(self.parser, selected_ids)
            self.spectrum_viz.load_data(self.parser, selected_ids)
            self.similarity_viz.load_data(self.parser, selected_ids)

    def _on_ion_selection_changed(
        self, spectrum_id: int, selected_ion_indices: List[int]
    ):
        """Handle ion selection changes in the ion data table."""
        if self.spectrum_viz:
            self.spectrum_viz.highlight_ions(spectrum_id, selected_ion_indices)

    def _add_new_key_value(self):
        """Add a new key-value pair via the Edit menu."""
        if not self.parser or not self.parser.spectra:
            messagebox.showwarning(
                "Warning", "No data loaded. Please open an MGF file first."
            )
            return

        selected_ids = self.spectrum_tree.get_selected_spectrum_ids()
        from .gui_components import AddKeyValueDialog

        dialog = AddKeyValueDialog(self.root, self.parser, selected_ids)
        if dialog.result:
            self._on_metadata_changed()

    def _keys_to_uppercase(self):
        """Convert all key names to uppercase via the Edit menu."""
        if not self.parser or not self.parser.spectra:
            messagebox.showwarning(
                "Warning", "No data loaded. Please open an MGF file first."
            )
            return

        if not messagebox.askyesno(
            "Convert Keys", "Convert all key names to UPPERCASE?"
        ):
            return

        # Get all current keys
        all_keys = self.parser.get_all_metadata_keys()

        for old_key in all_keys:
            new_key = old_key.upper()
            if old_key != new_key:
                self.parser.rename_key_in_all_spectra(old_key, new_key)

        self._on_metadata_changed()

    def _keys_to_lowercase(self):
        """Convert all key names to lowercase via the Edit menu."""
        if not self.parser or not self.parser.spectra:
            messagebox.showwarning(
                "Warning", "No data loaded. Please open an MGF file first."
            )
            return

        if not messagebox.askyesno(
            "Convert Keys", "Convert all key names to lowercase?"
        ):
            return

        # Get all current keys
        all_keys = self.parser.get_all_metadata_keys()

        for old_key in all_keys:
            new_key = old_key.lower()
            if old_key != new_key:
                self.parser.rename_key_in_all_spectra(old_key, new_key)

        self._on_metadata_changed()

    def _canonicalize_smiles_metadata(self):
        """Canonicalize SMILES strings stored in metadata fields."""
        if not self.parser or not self.parser.spectra:
            messagebox.showwarning(
                "Canonicalize SMILES",
                "No data loaded. Please open an MGF file first.",
            )
            return

        result = self._canonicalize_smiles_for_key("smiles")

        if result.get("error"):
            messagebox.showinfo("Canonicalize SMILES", result["error"])
            self.status_var.set(result["error"])
            return

        dialog = CanonicalSmilesDialog(self.root)
        dialog.show(
            rows=result.get("rows", []),
            summary_text=result.get("summary"),
            initial_key="smiles",
            on_key_change=self._handle_canonical_smiles_key_change,
        )

        status_message = result.get("status_message")
        if status_message:
            self.status_var.set(status_message)

    def _handle_canonical_smiles_key_change(self, key: str):
        """Handle metadata key changes triggered from the canonical SMILES dialog."""
        result = self._canonicalize_smiles_for_key(key)
        status_message = result.get("status_message")

        if result.get("error"):
            if result["error"]:
                self.status_var.set(result["error"])
            return result

        if status_message:
            self.status_var.set(status_message)

        return result

    def _canonicalize_smiles_for_key(self, smiles_key: str) -> Dict[str, Any]:
        """Canonicalize SMILES strings for a specific metadata key."""
        outcome: Dict[str, Any] = {
            "rows": [],
            "summary": "",
            "status_message": "",
            "error": None,
        }

        key_trimmed = smiles_key.strip()
        if not key_trimmed:
            outcome["error"] = "SMILES metadata key cannot be empty."
            return outcome

        if not RDKIT_AVAILABLE:
            outcome["error"] = (
                "RDKit is required to canonicalize SMILES strings, but it is not available."
            )
            return outcome

        key_lower = key_trimmed.lower()
        entries: List[Tuple[Spectrum, str, Any]] = []

        for spectrum in self.parser.spectra:
            for meta_key, value in spectrum.metadata.items():
                if meta_key.lower() == key_lower:
                    entries.append((spectrum, meta_key, value))

        if not entries:
            outcome["error"] = f"No metadata values found for key '{key_trimmed}'."
            return outcome

        occurrences_by_raw: Dict[str, List[Tuple[Spectrum, str]]] = {}
        trimmed_map: Dict[str, str] = {}
        order: List[str] = []

        for spectrum, meta_key, raw_value in entries:
            raw_str = "" if raw_value is None else str(raw_value)
            if raw_str not in occurrences_by_raw:
                occurrences_by_raw[raw_str] = []
                trimmed_map[raw_str] = raw_str.strip()
                order.append(raw_str)
            occurrences_by_raw[raw_str].append((spectrum, meta_key))

        rows_for_dialog: List[Dict[str, Any]] = []
        unique_changed = 0
        total_entries_updated = 0
        total_failures = 0

        for raw_value in order:
            occurrences = occurrences_by_raw[raw_value]
            trimmed_value = trimmed_map.get(raw_value, raw_value.strip())

            canonical_value = None
            error_message = None

            if trimmed_value:
                try:
                    mol = Chem.MolFromSmiles(trimmed_value)
                    if mol:
                        canonical_value = Chem.MolToSmiles(mol, canonical=True)
                    else:
                        error_message = "Invalid SMILES string"
                except Exception as exc:
                    error_message = f"Error: {exc}"
            else:
                error_message = "Empty SMILES value"

            highlight = False
            if canonical_value is not None:
                if canonical_value != trimmed_value:
                    unique_changed += 1
                    highlight = True
                elif canonical_value != raw_value:
                    highlight = True

                for spectrum, meta_key in occurrences:
                    current_value = spectrum.metadata.get(meta_key)
                    if current_value != canonical_value:
                        spectrum.metadata[meta_key] = canonical_value
                        total_entries_updated += 1
            else:
                highlight = True
                total_failures += 1

            rows_for_dialog.append(
                {
                    "original": raw_value,
                    "canonical": canonical_value,
                    "error": error_message,
                    "highlight": highlight,
                    "highlight_canonical": highlight,
                    "occurrences": len(occurrences),
                }
            )

        if total_entries_updated > 0:
            self._on_metadata_changed()

        summary_lines = [
            f"Metadata key: {key_trimmed}",
            f"Unique SMILES processed: {len(rows_for_dialog)}",
            f"Unique SMILES changed: {unique_changed}",
            f"Metadata entries updated: {total_entries_updated}",
        ]
        if total_failures:
            summary_lines.append(f"Failed conversions: {total_failures}")
        outcome["summary"] = "\n".join(summary_lines)

        if total_entries_updated > 0:
            outcome["status_message"] = (
                f"Canonicalized SMILES for key '{key_trimmed}' ({total_entries_updated} metadata entries updated)."
            )
        else:
            outcome["status_message"] = (
                f"Canonicalized SMILES for key '{key_trimmed}' (no updates required)."
            )

        if total_failures:
            outcome["status_message"] += (
                f" {total_failures} unique values could not be processed."
            )

        outcome["rows"] = rows_for_dialog
        return outcome

    def _normalize_intensities(self, mode="max", scale=1.0):
        """
        Normalize intensities in all spectra.

        Args:
            mode: "max" to normalize relative to most abundant peak, "sum" to normalize relative to total sum
            scale: target scale (1.0 for 0-1, 100.0 for 0-100, 1000.0 for 0-1000)
        """
        if not self.parser or not self.parser.spectra:
            messagebox.showwarning(
                "Warning", "No data loaded. Please open an MGF file first."
            )
            return

        # Create descriptive text for the confirmation dialog
        mode_text = (
            "most abundant peak" if mode == "max" else "total sum of all signals"
        )
        range_text = f"0-{int(scale)}" if scale != 1.0 else "0-1"

        # Ask for confirmation
        if not messagebox.askyesno(
            "Intensity Normalization",
            f"Normalize intensities in all spectra to range {range_text}, "
            f"relative to {mode_text}?\n\n"
            "This will modify the intensity values and cannot be undone.",
        ):
            return

        try:
            self.status_var.set("Normalizing intensities...")
            self.root.update()

            # Normalize all spectra
            self.parser.normalize_intensities(mode=mode, scale=scale)

            # Update visualization and data
            self._on_metadata_changed()

            self.status_var.set(
                f"Normalized intensities in {len(self.parser.spectra)} spectra"
            )

            messagebox.showinfo(
                "Normalization Complete",
                f"Successfully normalized intensities in {len(self.parser.spectra)} spectra "
                f"to range {range_text}, relative to {mode_text}.",
            )

        except Exception as e:
            messagebox.showerror(
                "Normalization Error", f"Failed to normalize intensities:\n{str(e)}"
            )
            self.status_var.set("Normalization failed")

    def _update_spectrum_name_menu(self):
        """Update the spectrum name menu with available metadata fields."""
        if not self.parser:
            return

        # Clear existing field options (keep "Numbered")
        menu = self.spectrum_name_menu

        # Remove all items except "Numbered"
        last_index = menu.index("end")
        if last_index is not None and last_index > 0:
            menu.delete(1, last_index)

        # Add separator
        menu.add_separator()

        # Add metadata fields as options
        all_keys = self.parser.get_all_metadata_keys()
        for key in all_keys:
            menu.add_radiobutton(
                label=key,
                variable=self.spectrum_name_var,
                value=key,
                command=self._update_spectrum_names,
            )

    def _update_spectrum_names(self):
        """Update how spectrum names are displayed."""
        if not self.parser:
            return

        # Set the naming scheme in the spectrum tree
        naming_scheme = self.spectrum_name_var.get()
        self.spectrum_tree.set_naming_scheme(naming_scheme)

        # Set the naming scheme in the spectrum visualization
        if hasattr(self, "spectrum_viz") and self.spectrum_viz:
            self.spectrum_viz.set_naming_scheme(naming_scheme)

        # Refresh the tree to show updated names
        self.spectrum_tree.load_data(self.parser)

    def _open_regex_editor(self):
        """Open the regex editor dialog."""
        if not self.parser or not self.parser.spectra:
            messagebox.showwarning(
                "Warning", "No data loaded. Please open an MGF file first."
            )
            return

        from .gui_components import RegexEditorDialog

        dialog = RegexEditorDialog(self.root, self.parser)
        if dialog.changes_made:
            self._on_metadata_changed()

    def _delete_selected_spectra(self):
        """Delete the currently selected spectra."""
        if not self.parser or not self.parser.spectra:
            messagebox.showwarning(
                "Warning", "No data loaded. Please open an MGF file first."
            )
            return

        selected_ids = self.spectrum_tree.get_selected_spectrum_ids()
        if not selected_ids:
            messagebox.showwarning("Warning", "No spectra selected.")
            return

        # Confirmation dialog
        num_selected = len(selected_ids)
        total_spectra = len(self.parser.spectra)

        message = (
            f"Are you sure you want to delete {num_selected} selected spectra?\n\n"
            f"This will remove them permanently from the current session.\n"
            f"Remaining spectra: {total_spectra - num_selected}"
        )

        if not messagebox.askyesno("Confirm Delete", message, icon="warning"):
            return

        # Remove selected spectra
        self.parser.spectra = [
            s for s in self.parser.spectra if s.spectrum_id not in selected_ids
        ]

        # Update the status
        remaining_count = len(self.parser.spectra)
        self.status_var.set(
            f"Deleted {num_selected} spectra. {remaining_count} remaining."
        )

        # Refresh all displays
        if remaining_count > 0:
            self.spectrum_tree.load_data(self.parser)
            self._update_spectrum_name_menu()

            # Clear selections since deleted spectra are no longer available
            self.metadata_editor.load_data(self.parser, [])
            self.ion_table.load_data(self.parser, [])
            self.spectrum_viz.load_data(self.parser, [])
        else:
            # No spectra left - reset to initial state
            self._set_components_enabled(False)
            for component in [
                self.spectrum_tree,
                self.metadata_editor,
                self.ion_table,
                self.spectrum_viz,
            ]:
                if hasattr(component, "load_data"):
                    component.load_data(self.parser, [])
            self.root.title("MGF Explorer")

    def _calculate_average_spectra(self):
        """Calculate average spectrum per group."""
        if not self.parser or not self.parser.spectra:
            messagebox.showwarning(
                "Warning", "No data loaded. Please open an MGF file first."
            )
            return

        # Check if grouping is applied
        if not self.spectrum_tree.selected_grouping_tags:
            messagebox.showwarning(
                "Warning", "No grouping applied. Please set grouping tags first."
            )
            return

        # Show parameter dialog
        from .gui_components import AverageSpectrumDialog

        dialog = AverageSpectrumDialog(self.root)
        if not dialog.result:
            return

        # Get parameters from dialog
        binning_mz = dialog.result["binning_mz"]
        averaging_method = dialog.result["averaging_method"]
        new_key = dialog.result["new_key"]
        new_value = dialog.result["new_value"]

        try:
            # Get grouped spectra data
            grouped_spectra = self._get_grouped_spectra_for_averaging()

            if not grouped_spectra:
                messagebox.showwarning("Warning", "No groups found to average.")
                return

            # Calculate average spectra
            new_spectra = self._create_average_spectra(
                grouped_spectra, binning_mz, averaging_method, new_key, new_value
            )

            total_groups = len(grouped_spectra)
            single_spectrum_groups = sum(
                1 for spectra_list in grouped_spectra.values() if len(spectra_list) == 1
            )

            if new_spectra:
                # Add new spectra to parser
                max_id = max([s.spectrum_id for s in self.parser.spectra])
                for i, spectrum in enumerate(new_spectra):
                    spectrum.spectrum_id = max_id + i + 1
                    self.parser.spectra.append(spectrum)

                # Refresh displays
                self._on_metadata_changed()

                message = f"Created {len(new_spectra)} average spectra from {total_groups} groups."
                if single_spectrum_groups > 0:
                    message += f"\nSkipped {single_spectrum_groups} groups with only one spectrum."
                messagebox.showinfo("Success", message)
            else:
                if single_spectrum_groups == total_groups:
                    messagebox.showinfo(
                        "No Averages Created",
                        f"All {total_groups} groups contain only one spectrum each.\n"
                        "No average spectra were created.",
                    )
                else:
                    messagebox.showwarning(
                        "Warning", "No average spectra could be created."
                    )

        except Exception as e:
            messagebox.showerror(
                "Error", f"Failed to calculate average spectra: {str(e)}"
            )

    def _get_grouped_spectra_for_averaging(self):
        """Get grouped spectra organized for averaging."""
        grouped_spectra = {}

        # Build hierarchy similar to the tree view
        hierarchy = {}

        for spectrum in self.parser.spectra:
            # Get values for grouping tags
            path = []
            for tag in self.spectrum_tree.selected_grouping_tags:
                value = spectrum.get_metadata_value(tag)
                if value is None:
                    value = "<missing>"
                path.append(f"{tag}={value}")

            # Build nested dictionary
            current = hierarchy
            for level, path_part in enumerate(path):
                if path_part not in current:
                    current[path_part] = {"spectra": [], "children": {}}
                current = current[path_part]["children"]

            # Add spectrum to the final level
            final_level = hierarchy
            for path_part in path[:-1]:
                final_level = final_level[path_part]["children"]
            if path:
                final_level[path[-1]]["spectra"].append(spectrum)
            else:
                # No valid grouping path, add to root
                if "_ungrouped_" not in hierarchy:
                    hierarchy["_ungrouped_"] = {"spectra": [], "children": {}}
                hierarchy["_ungrouped_"]["spectra"].append(spectrum)

        # Extract groups that have spectra (leaf nodes)
        self._extract_leaf_groups(hierarchy, grouped_spectra, [])

        return grouped_spectra

    def _extract_leaf_groups(self, hierarchy, grouped_spectra, path):
        """Extract leaf groups that contain spectra."""
        for key, data in hierarchy.items():
            current_path = path + [key]

            # If this group has spectra and no children, it's a leaf group
            if data["spectra"] and not data["children"]:
                group_key = " -> ".join(current_path)
                grouped_spectra[group_key] = data["spectra"]

            # If this group has children, recursively check them
            if data["children"]:
                self._extract_leaf_groups(
                    data["children"], grouped_spectra, current_path
                )

            # If this group has both spectra and children, the spectra at this level form a group
            if data["spectra"] and data["children"]:
                group_key = " -> ".join(current_path) + " (direct)"
                grouped_spectra[group_key] = data["spectra"]

    def _create_average_spectra(
        self, grouped_spectra, binning_mz, averaging_method, new_key, new_value
    ):
        """Create average spectra for each group."""
        new_spectra = []
        skipped_groups = 0

        for group_name, spectra_list in grouped_spectra.items():
            if len(spectra_list) < 2:
                skipped_groups += 1
                continue  # Skip groups with only one spectrum

            try:
                average_spectrum = self._calculate_single_average_spectrum(
                    spectra_list,
                    binning_mz,
                    averaging_method,
                    new_key,
                    new_value,
                    group_name,
                )
                if average_spectrum:
                    new_spectra.append(average_spectrum)
            except Exception as e:
                print(f"Error creating average for group {group_name}: {e}")
                continue

        if skipped_groups > 0:
            print(f"Skipped {skipped_groups} groups with only one spectrum")

        return new_spectra

    def _calculate_single_average_spectrum(
        self, spectra_list, binning_mz, averaging_method, new_key, new_value, group_name
    ):
        """Calculate a single average spectrum from a list of spectra."""
        if not spectra_list:
            return None

        # Collect all ions from all spectra
        all_ions = []
        for spectrum in spectra_list:
            if len(spectrum.ions) > 0:
                all_ions.append(spectrum.ions)

        if not all_ions:
            return None  # No ion data to average

        # Find common peaks across all spectra (for normalization)
        common_peaks = self._find_common_peaks(all_ions, binning_mz)

        if len(common_peaks) == 0:
            # Create empty spectrum if no common peaks
            average_spectrum = Spectrum(0)  # ID will be set later
            average_spectrum.set_ions([], [])
        else:
            # Calculate normalized ion data
            normalized_ions = self._normalize_spectra_by_common_peaks(
                all_ions, common_peaks, binning_mz
            )

            # Bin and average the ions
            averaged_ions = self._bin_and_average_ions(
                normalized_ions, binning_mz, averaging_method
            )

            # Create new spectrum
            average_spectrum = Spectrum(0)  # ID will be set later
            if len(averaged_ions) > 0:
                mz_values = [ion[0] for ion in averaged_ions]
                intensity_values = [ion[1] for ion in averaged_ions]
                average_spectrum.set_ions(mz_values, intensity_values)

        # Calculate average metadata
        self._calculate_average_metadata(
            average_spectrum, spectra_list, new_key, new_value, group_name
        )

        return average_spectrum

    def _find_common_peaks(self, all_ions, binning_mz):
        """Find peaks that are present in all spectra within the binning tolerance."""
        if not all_ions:
            return []

        # Get all unique m/z values from the first spectrum as candidates
        first_spectrum_mz = all_ions[0][:, 0]
        common_peaks = []

        for mz in first_spectrum_mz:
            is_common = True
            for ions in all_ions[1:]:
                # Check if this m/z is present in other spectra (within binning tolerance)
                if not self._has_peak_in_range(ions[:, 0], mz, binning_mz):
                    is_common = False
                    break

            if is_common:
                common_peaks.append(mz)

        return common_peaks

    def _has_peak_in_range(self, mz_values, target_mz, tolerance):
        """Check if any peak exists within tolerance of target m/z."""
        return np.any(np.abs(mz_values - target_mz) <= tolerance / 2)

    def _normalize_spectra_by_common_peaks(self, all_ions, common_peaks, binning_mz):
        """Normalize spectra based on common peaks sum."""
        normalized_ions = []

        for ions in all_ions:
            if len(common_peaks) == 0:
                # No common peaks, no normalization
                normalized_ions.append(ions)
                continue

            # Calculate sum of intensities for common peaks
            common_intensity_sum = 0
            for common_mz in common_peaks:
                # Find the closest peak within binning tolerance
                mz_diffs = np.abs(ions[:, 0] - common_mz)
                closest_idx = np.argmin(mz_diffs)
                if mz_diffs[closest_idx] <= binning_mz / 2:
                    common_intensity_sum += ions[closest_idx, 1]

            if common_intensity_sum > 0:
                # Normalize so common peaks sum to 1
                normalization_factor = 1.0 / common_intensity_sum
                normalized_spectrum = ions.copy()
                normalized_spectrum[:, 1] *= normalization_factor
                normalized_ions.append(normalized_spectrum)
            else:
                # No common peaks found, use original
                normalized_ions.append(ions)

        return normalized_ions

    def _bin_and_average_ions(self, normalized_ions, binning_mz, averaging_method):
        """Bin ions by m/z and calculate average intensities and m/z values."""
        # Collect all m/z and intensity pairs from all spectra
        all_peaks = []
        for spectrum_idx, ions in enumerate(normalized_ions):
            for mz, intensity in ions:
                all_peaks.append((mz, intensity, spectrum_idx))

        if not all_peaks:
            return []

        # Sort by m/z for efficient binning
        all_peaks.sort(key=lambda x: x[0])

        # Group peaks that are within binning tolerance
        averaged_ions = []
        current_group = []
        current_bin_center = None

        for mz, intensity, spectrum_idx in all_peaks:
            if current_bin_center is None:
                # Start new bin
                current_bin_center = mz
                current_group = [(mz, intensity, spectrum_idx)]
            elif abs(mz - current_bin_center) <= binning_mz / 2:
                # Add to current bin
                current_group.append((mz, intensity, spectrum_idx))
            else:
                # Process current bin and start new one
                if current_group:
                    averaged_peak = self._process_peak_group(
                        current_group, averaging_method
                    )
                    if averaged_peak:
                        averaged_ions.append(averaged_peak)

                # Start new bin
                current_bin_center = mz
                current_group = [(mz, intensity, spectrum_idx)]

        # Process the last group
        if current_group:
            averaged_peak = self._process_peak_group(current_group, averaging_method)
            if averaged_peak:
                averaged_ions.append(averaged_peak)

        return averaged_ions

    def _process_peak_group(self, peak_group, averaging_method):
        """Process a group of peaks that should be combined into one."""
        if not peak_group:
            return None

        # Extract m/z values and intensities, grouped by spectrum
        spectrum_contributions = {}

        for mz, intensity, spectrum_idx in peak_group:
            if spectrum_idx not in spectrum_contributions:
                spectrum_contributions[spectrum_idx] = []
            spectrum_contributions[spectrum_idx].append((mz, intensity))

        # For each spectrum, take the highest intensity peak if multiple peaks contribute
        final_contributions = []
        for spectrum_idx, peaks in spectrum_contributions.items():
            if len(peaks) == 1:
                final_contributions.append(peaks[0])
            else:
                # Multiple peaks from same spectrum - take the one with highest intensity
                best_peak = max(peaks, key=lambda x: x[1])
                final_contributions.append(best_peak)

        if not final_contributions:
            return None

        # Calculate average m/z weighted by intensity
        mz_values = [peak[0] for peak in final_contributions]
        intensities = [peak[1] for peak in final_contributions]

        # Weighted average m/z (weighted by intensity)
        total_intensity = sum(intensities)
        if total_intensity > 0:
            avg_mz = (
                sum(mz * intensity for mz, intensity in final_contributions)
                / total_intensity
            )
        else:
            avg_mz = sum(mz_values) / len(mz_values)  # Simple average if no intensity

        # Calculate average or median intensity
        if averaging_method == "average":
            avg_intensity = np.mean(intensities)
        else:  # median
            avg_intensity = np.median(intensities)

        return [avg_mz, avg_intensity]

    def _calculate_average_metadata(
        self, average_spectrum, spectra_list, new_key, new_value, group_name
    ):
        """Calculate average metadata for the new spectrum."""
        # Add the new key-value pair
        average_spectrum.add_metadata(new_key, new_value)

        # Calculate average PEPMASS if present
        pepmass_values = []
        for spectrum in spectra_list:
            pepmass = spectrum.get_metadata_value("PEPMASS")
            if pepmass:
                try:
                    # PEPMASS might have intensity as well, extract just the m/z
                    pepmass_parts = pepmass.split()
                    pepmass_mz = float(pepmass_parts[0])
                    pepmass_values.append(pepmass_mz)
                except (ValueError, IndexError):
                    continue

        if pepmass_values:
            avg_pepmass = np.mean(pepmass_values)
            average_spectrum.add_metadata("PEPMASS", f"{avg_pepmass:.6f}")

        # Collect unique values from all other metadata keys
        # TODO: This needs refinement - currently just concatenating unique values
        all_keys = set()
        for spectrum in spectra_list:
            all_keys.update(spectrum.metadata.keys())

        all_keys.discard("PEPMASS")  # Already handled
        all_keys.discard(new_key)  # Already set

        for key in all_keys:
            unique_values = set()
            for spectrum in spectra_list:
                value = spectrum.get_metadata_value(key)
                if value:
                    unique_values.add(value)

            if unique_values:
                if len(unique_values) == 1:
                    # All spectra have the same value
                    average_spectrum.add_metadata(key, list(unique_values)[0])
                else:
                    # Multiple values - concatenate them
                    # TODO: This needs refinement for better handling of different metadata types
                    combined_value = "|".join(sorted(unique_values))
                    average_spectrum.add_metadata(key, combined_value)

        # Add metadata about the averaging
        average_spectrum.add_metadata("AVERAGED_FROM_GROUP", group_name)
        average_spectrum.add_metadata("AVERAGED_FROM_COUNT", str(len(spectra_list)))
        spectrum_ids = [str(s.spectrum_id) for s in spectra_list]
        average_spectrum.add_metadata("AVERAGED_FROM_IDS", ",".join(spectrum_ids))

    def export_grouped_spectra(self):
        """Export spectra into separate MGF files for each active group."""
        if not self.parser.spectra:
            messagebox.showwarning("Warning", "No spectra loaded to export.")
            return

        if not self.spectrum_tree.has_grouping():
            messagebox.showwarning(
                "No Grouping", "Please configure grouping tags before exporting."
            )
            return

        grouping_structure = self.spectrum_tree.get_current_grouping_structure()
        if not grouping_structure:
            messagebox.showwarning(
                "No Groups",
                "No groups available for export with the current grouping or filter.",
            )
            return

        file_path = filedialog.asksaveasfilename(
            title="Export Grouped Spectra",
            defaultextension=".mgf",
            filetypes=[("MGF files", "*.mgf"), ("All files", "*.*")],
        )

        if not file_path:
            return

        base_dir, base_name = os.path.split(file_path)
        if not base_dir:
            base_dir = os.getcwd()
        base_dir = os.path.abspath(base_dir)

        name_root, ext = os.path.splitext(base_name)
        if not name_root:
            name_root = "grouped_export"
        if not ext:
            ext = ".mgf"

        groups = []

        def collect_groups(nodes, path):
            for key, data in nodes.items():
                new_path = path + [key]
                children = data.get("children", {})
                if children:
                    collect_groups(children, new_path)
                elif data.get("spectra"):
                    groups.append((new_path, data["spectra"]))

        collect_groups(grouping_structure, [])

        if not groups:
            messagebox.showwarning(
                "No Groups",
                "No grouped spectra available for export with the current grouping or filter.",
            )
            return

        def sanitize_component(text):
            value = str(text).strip()
            if not value:
                value = "group"
            value = value.replace(" ", "_")
            value = re.sub(r"[\\/:*?\"<>|]", "_", value)
            value = re.sub(r"_+", "_", value)
            value = value.strip("_")
            return value or "group"

        used_suffixes = set()
        total_files = 0
        total_spectra = 0

        try:
            self.status_var.set("Exporting grouped spectra...")
            self.root.update_idletasks()

            for path_components, spectra in groups:
                if not spectra:
                    continue

                value_parts = []
                for component in path_components:
                    if "=" in component:
                        value = component.split("=", 1)[1]
                    else:
                        value = component
                    sanitized = sanitize_component(value or "missing")
                    value_parts.append(sanitized)

                if not value_parts:
                    value_parts.append("group")

                suffix = "_".join(value_parts)
                candidate_suffix = suffix
                counter = 2
                while candidate_suffix in used_suffixes:
                    candidate_suffix = f"{suffix}_{counter}"
                    counter += 1
                used_suffixes.add(candidate_suffix)

                if name_root:
                    output_name = f"{name_root}_{candidate_suffix}{ext}"
                else:
                    output_name = f"{candidate_suffix}{ext}"

                output_path = os.path.join(base_dir, output_name)
                spectrum_ids = [s.spectrum_id for s in spectra]
                self.parser.export_to_mgf(output_path, spectrum_ids)

                total_files += 1
                total_spectra += len(spectra)

            if total_files == 0:
                messagebox.showwarning(
                    "No Groups",
                    "No grouped spectra available for export with the current grouping or filter.",
                )
                self.status_var.set("No grouped spectra exported")
                return

            self.status_var.set(
                f"Exported {total_spectra} spectra into {total_files} grouped file(s)"
            )
            messagebox.showinfo(
                "Export Complete",
                f"Exported {total_spectra} spectra into {total_files} file(s).\n"
                f"Files saved to: {base_dir}",
            )

        except Exception as e:
            messagebox.showerror(
                "Export Error", f"Failed to export grouped spectra:\n{str(e)}"
            )
            self.status_var.set("Export failed")

    def export_all_spectra(self):
        """Export all spectra to a new MGF file."""
        if not self.parser.spectra:
            messagebox.showwarning("Warning", "No spectra loaded to export.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Export All Spectra to MGF File",
            defaultextension=".mgf",
            filetypes=[("MGF files", "*.mgf"), ("All files", "*.*")],
        )

        if not file_path:
            return

        try:
            self.status_var.set("Exporting spectra...")
            self.root.update()

            # Export all spectra
            self.parser.export_to_mgf(file_path)

            # Update status
            self.status_var.set(
                f"Exported {len(self.parser.spectra)} spectra to {os.path.basename(file_path)}"
            )

            messagebox.showinfo(
                "Export Complete",
                f"Successfully exported {len(self.parser.spectra)} spectra to:\n{file_path}",
            )

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export spectra:\n{str(e)}")
            self.status_var.set("Export failed")

    def _open_smarts_filter(self):
        """Open SMARTS substructure filter dialog."""
        if not RDKIT_AVAILABLE:
            messagebox.showerror(
                "RDKit Not Available",
                "RDKit is required for SMARTS filtering. Please install rdkit-pypi.",
            )
            return

        if not self.parser.spectra:
            messagebox.showwarning(
                "No Data", "Please load MGF data before using SMARTS filtering."
            )
            return

        # Create SMARTS filter dialog
        dialog = SmartsFilterDialog(
            self.root, self.parser.spectra, self._apply_smarts_filter
        )

    def _apply_smarts_filter(self, matching_spectra):
        """Apply SMARTS filter by keeping only matching spectra."""
        if not matching_spectra:
            messagebox.showinfo("No Matches", "No spectra matched the SMARTS pattern.")
            return

        # Replace the spectra list with only matching ones
        self.parser.spectra = matching_spectra

        # Update all views - use load_data to refresh the tree view
        self.spectrum_tree.load_data(self.parser)
        self.metadata_editor.clear()
        self.spectrum_viz.clear_plot()

        # Update status
        self.status_var.set(
            f"SMARTS filter applied - {len(matching_spectra)} spectra remaining"
        )

    def _open_intensity_filter(self):
        """Open intensity filter dialog."""
        if not self.parser.spectra:
            messagebox.showwarning(
                "No Data", "Please load MGF data before using intensity filtering."
            )
            return

        # Create intensity filter dialog
        dialog = IntensityFilterDialog(
            self.root, self.parser.spectra, self._apply_intensity_filter
        )

    def _apply_intensity_filter(self):
        """Apply intensity filter - callback after filter is applied."""
        # Update all views to reflect the filtered data
        self.spectrum_tree.load_data(self.parser)
        self.metadata_editor.clear()
        self.spectrum_viz.clear_plot()
        self.ion_table.clear()

        # Update status
        total_fragments = sum(len(spectrum.ions) for spectrum in self.parser.spectra)
        self.status_var.set(
            f"Intensity filter applied - {total_fragments} total fragments remaining"
        )

    def _generate_subformulas(self):
        """Open fragment annotation dialog and generate subformulas."""
        if not self.parser.spectra:
            messagebox.showwarning(
                "No Data", "Please load MGF data before generating subformulas."
            )
            return

        # Open fragment annotation dialog
        dialog = FragmentAnnotationDialog(self.root, spectra=self.parser.spectra)
        config = dialog.show()

        if config is None:  # User cancelled
            return

        # Clear PPM tolerance cache to ensure new tolerance function is used
        self._get_ppm_tolerance_for_mz_cache.clear()

        # Count spectra with formulas first
        spectra_with_formulas = []
        for spectrum in self.parser.spectra:
            formula = self._extract_formula_from_spectrum(
                spectrum, config["formula_tags"]
            )
            if formula:
                spectra_with_formulas.append((spectrum, formula))

        if not spectra_with_formulas:
            messagebox.showwarning(
                "No Formulas Found",
                "No molecular formulas found in the specified metadata tags.\n"
                f"Searched tags: {', '.join(config['formula_tags'])}",
            )
            return

        # Use max_workers from dialog configuration
        max_workers = config.get("max_workers", min(32, (os.cpu_count() or 1) + 2))

        # Show progress dialog
        progress_dialog = ProgressDialog(
            self.root,
            title="Fragment Annotation",
            message=f"Generating subformulas for fragments ({max_workers} threads)...",
        )
        progress_dialog.show(max_value=len(spectra_with_formulas))

        try:
            processed_count = 0
            # Collect all annotation data for the plot
            all_annotations = []

            # Clear existing annotations up front (so workers don't need to modify shared state)
            for spectrum, _ in spectra_with_formulas:
                spectrum.clear_fragment_annotations()

            # Map spectrum_id -> spectrum object for applying results in main thread
            id_to_spectrum = {s.spectrum_id: s for s, _ in spectra_with_formulas}

            # Prepare worker inputs
            worker_inputs = []
            for spectrum, formula in spectra_with_formulas:
                # Convert ions to a plain Python list to avoid accidental numpy/GIL issues in threads
                ions_list = [tuple(x) for x in spectrum.ions]
                worker_inputs.append((spectrum.spectrum_id, ions_list, formula))

            def _annotate_worker(spectrum_id, ions_list, formula):
                """Worker function: compute annotations for one spectrum (no GUI / shared-state mutations)."""
                results = []  # list of per-annotation dicts
                annotator = FragmentAnnotator(
                    precursor_formula=formula,
                    additional_elements=config["additional_elements"],
                )
                annotator.generate_subformulas()

                for ion_index, (mz, intensity) in enumerate(ions_list):
                    ppm_tolerance = self._get_ppm_tolerance_for_mz(mz, config)
                    annotations = annotator.annotate_mz(mz, ppm_tolerance=ppm_tolerance)
                    for annotation_rank, annotation in enumerate(annotations):
                        results.append(
                            {
                                "ion_index": ion_index,
                                "mz": mz,
                                "intensity": intensity,
                                "formula": annotation["formula"],
                                "ppm_error": annotation["ppm_error"],
                                "theoretical_mass": annotation.get("theoretical_mass"),
                                "charge": annotation.get("charge"),
                                "annotation_rank": annotation_rank,
                                "ppm_tolerance_used": ppm_tolerance,
                            }
                        )
                return (spectrum_id, results)

            processed_count = 0
            all_annotations = []

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_id = {
                    executor.submit(_annotate_worker, spec_id, ions, formula): spec_id
                    for spec_id, ions, formula in worker_inputs
                }

                try:
                    for future in as_completed(future_to_id):
                        # If user cancelled, stop applying further results
                        if progress_dialog.is_cancelled():
                            break

                        spec_id, annotations_for_spectrum = future.result()

                        # Apply annotations in the main thread (modify spectrum objects and GUI-safe operations)
                        spectrum = id_to_spectrum.get(spec_id)
                        if spectrum is None:
                            # Spectrum no longer exists (e.g. deleted) - skip
                            continue

                        for ann in annotations_for_spectrum:
                            spectrum.add_fragment_annotation(
                                ion_index=ann["ion_index"],
                                formula=ann["formula"],
                                ppm_error=ann["ppm_error"],
                                additional_info={
                                    "theoretical_mass": ann["theoretical_mass"],
                                    "charge": ann["charge"],
                                },
                            )

                            all_annotations.append(
                                {
                                    "mz": ann["mz"],
                                    "ppm_error": ann["ppm_error"],
                                    "formula": ann["formula"],
                                    "spectrum_id": spectrum.spectrum_id,
                                    "intensity": ann["intensity"],
                                    "annotation_rank": ann["annotation_rank"],
                                    "ppm_tolerance_used": ann["ppm_tolerance_used"],
                                }
                            )

                        # Update progress and UI
                        processed_count += 1
                        progress_dialog.update_progress(
                            processed_count,
                            f"Processed spectrum {processed_count}/{len(spectra_with_formulas)} (ID: {spectrum.spectrum_id})",
                        )

                        # Periodically allow the UI to process events
                        if processed_count % 5 == 0:
                            self.root.update()

                finally:
                    # If cancelled while futures still running, attempt to shut down promptly
                    if progress_dialog.is_cancelled():
                        try:
                            executor.shutdown(wait=False)
                        except Exception:
                            pass

            # Close progress dialog
            progress_dialog.close()

            if progress_dialog.is_cancelled():
                self.status_var.set("Subformula generation cancelled")
                messagebox.showinfo(
                    "Cancelled", "Fragment annotation was cancelled by user."
                )
                return

            # Refresh the ion data table if there are selected spectra
            if (
                hasattr(self, "ion_table")
                and self.spectrum_tree.get_selected_spectrum_ids()
            ):
                selected_ids = self.spectrum_tree.get_selected_spectrum_ids()
                self.ion_table.load_data(self.parser, selected_ids)

            self.status_var.set(
                f"Subformula generation completed for {processed_count} spectra"
            )

            # Show PPM deviation plot if there are annotations
            if all_annotations:
                try:
                    plot_dialog = PPMDeviationPlotDialog(self.root)
                    plot_dialog.show(all_annotations)
                except Exception as e:
                    # If plot fails, just show a warning but don't stop the process
                    messagebox.showwarning(
                        "Plot Warning",
                        f"Could not display PPM deviation plot:\n{str(e)}",
                    )

            messagebox.showinfo(
                "Annotation Complete",
                f"Fragment annotation completed successfully.\n"
                f"Processed {processed_count} spectra with molecular formulas.\n"
                f"Total annotated fragments: {len(all_annotations)}",
            )

        except Exception as e:
            progress_dialog.close()
            messagebox.showerror(
                "Annotation Error", f"Failed to generate subformulas:\n{str(e)}"
            )
            self.status_var.set("Subformula generation failed")

    def _clear_all_annotations(self):
        """Clear all fragment annotations from all spectra."""
        if not self.parser.spectra:
            messagebox.showwarning(
                "No Data", "Please load MGF data before clearing annotations."
            )
            return

        # Count spectra with annotations
        spectra_with_annotations = [
            spectrum
            for spectrum in self.parser.spectra
            if spectrum.fragment_annotations
        ]

        if not spectra_with_annotations:
            messagebox.showinfo(
                "No Annotations", "No fragment annotations found to clear."
            )
            return

        # Ask for confirmation
        result = messagebox.askyesno(
            "Clear All Annotations",
            f"Are you sure you want to clear all fragment annotations?\n\n"
            f"This will remove annotations from {len(spectra_with_annotations)} "
            f"spectra and cannot be undone.",
            icon="warning",
        )

        if not result:
            return

        # Clear annotations from all spectra
        cleared_count = 0
        for spectrum in self.parser.spectra:
            if spectrum.fragment_annotations:
                spectrum.clear_fragment_annotations()
                cleared_count += 1

        # Refresh the ion data table if there are selected spectra
        if (
            hasattr(self, "ion_table")
            and self.spectrum_tree.get_selected_spectrum_ids()
        ):
            selected_ids = self.spectrum_tree.get_selected_spectrum_ids()
            self.ion_table.load_data(self.parser, selected_ids)

        self.status_var.set(f"Cleared annotations from {cleared_count} spectra")

        messagebox.showinfo(
            "Annotations Cleared",
            f"Successfully cleared fragment annotations from {cleared_count} spectra.",
        )

    def _extract_formula_from_spectrum(
        self, spectrum: Spectrum, formula_tags: List[str]
    ) -> str:
        """Extract molecular formula from spectrum metadata."""
        for tag in formula_tags:
            value = spectrum.get_metadata_value(tag)
            if value:
                # Try to extract formula from various formats
                formula = self._parse_formula_from_value(value)
                if formula:
                    return formula
        return ""

    def _parse_formula_from_value(self, value: str) -> str:
        """Parse molecular formula from metadata value."""
        # Remove common prefixes and clean up
        value = value.strip()

        # Handle SMILES format - try to convert to molecular formula
        if any(char in value for char in ["[", "]", "=", "#", "(", ")", "+"]):
            # This might be SMILES, try to extract formula using basic pattern matching
            # For now, skip SMILES conversion and look for explicit formulas
            return ""

        # Look for molecular formula pattern (letters followed by optional numbers)
        import re

        formula_pattern = r"^([A-Z][a-z]?\d*)+$"
        if re.match(formula_pattern, value):
            return value

        # Try to extract formula from longer strings
        formula_match = re.search(r"([A-Z][a-z]?\d*)+", value)
        if formula_match:
            return formula_match.group()

        return ""

    def _get_ppm_tolerance_for_mz(self, mz: float, config: dict) -> float:
        mz = int(mz)
        if mz not in self._get_ppm_tolerance_for_mz_cache:
            tol = self.__get_ppm_tolerance_for_mz(mz, config)
            self._get_ppm_tolerance_for_mz_cache[mz] = tol

        return self._get_ppm_tolerance_for_mz_cache[mz]

    def __get_ppm_tolerance_for_mz(self, mz: float, config: dict) -> float:
        """Calculate PPM tolerance for a given m/z using the custom function if available."""
        ppm_function_points = config.get("ppm_function_points", [])

        if not ppm_function_points:
            # No custom function, use default tolerance
            return config["ppm_tolerance"]

        if len(ppm_function_points) == 1:
            # Single point - constant function
            return ppm_function_points[0][1]

        # Sort points by m/z
        sorted_points = sorted(ppm_function_points, key=lambda x: x[0])

        # Check if mz is before the first point
        if mz <= sorted_points[0][0]:
            return sorted_points[0][1]

        # Check if mz is after the last point
        if mz >= sorted_points[-1][0]:
            return sorted_points[-1][1]

        # Find the two points to interpolate between
        for i in range(len(sorted_points) - 1):
            mz1, ppm1 = sorted_points[i]
            mz2, ppm2 = sorted_points[i + 1]

            if mz1 <= mz <= mz2:
                # Linear interpolation
                if mz2 == mz1:
                    return ppm1
                t = (mz - mz1) / (mz2 - mz1)
                return ppm1 + t * (ppm2 - ppm1)

        # Fallback (should not reach here)
        return config["ppm_tolerance"]

    def _show_fragment_distribution(self):
        """Show the fragment distribution dialog."""
        if not self.parser or not self.parser.spectra:
            messagebox.showinfo(
                "No Data", "No spectra loaded. Please load an MGF file first."
            )
            return

        # Get currently selected spectrum IDs
        selected_ids = self.spectrum_tree.get_selected_spectrum_ids()

        dialog = FragmentDistributionDialog(self.root)
        dialog.show(self.parser, selected_ids)

    def show_about(self):
        """Show about dialog."""
        drag_drop_status = (
            "• Drag and drop MGF files to open\n"
            if DRAG_DROP_AVAILABLE
            else "• Use File > Open to load MGF files\n"
        )

        messagebox.showinfo(
            "About MGF Explorer",
            "MGF Explorer v1.0\n\n"
            "A tool for exploring and editing MGF (Mascot Generic Format) files.\n\n"
            "Features:\n"
            "• Parse and display MS/MS spectra\n"
            "• Group spectra by metadata fields\n"
            "• Edit metadata values\n"
            "• Normalize intensities in spectra\n"
            "• Export spectra to MGF files\n"
            "• Visualize spectra as stick charts\n"
            "• View ion data in tables\n"
            "• Display SMILES molecular structures\n"
            f"{drag_drop_status}",
        )

    def run(self):
        """Start the application."""
        self.root.mainloop()


def main():
    """Main entry point for the application."""
    app = MGFExplorerApp()
    app.run()


if __name__ == "__main__":
    main()
