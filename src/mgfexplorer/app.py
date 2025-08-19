"""
Main application window for the MGF Explorer.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import os
import numpy as np
from .mgf_parser import MGFParser, Spectrum
from .gui_components import (
    SpectrumTreeView, 
    MetadataEditor, 
    SpectrumVisualization, 
    IonDataTable,
    CosineSimilarityVisualization
)


class MGFExplorerApp:
    """Main application class for the MGF Explorer."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MGF Explorer")
        self.root.geometry("1400x900")
        self.root.minsize(1000, 700)
        
        self.parser = MGFParser()
        self.current_file = None
        
        # Selection debouncing
        self.selection_update_job = None
        self.SELECTION_DELAY_MS = 300  # Wait 300ms before updating heavy components
        
        self._create_widgets()
        self._create_menu()
        
    def _create_menu(self):
        """Create the application menu."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open MGF File...", command=self.open_file, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Export All Spectra...", command=self.export_all_spectra, accelerator="Ctrl+E")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Add New Key-Value Pair...", command=self._add_new_key_value)
        edit_menu.add_separator()
        edit_menu.add_command(label="Convert Keys to UPPERCASE", command=self._keys_to_uppercase)
        edit_menu.add_command(label="Convert Keys to lowercase", command=self._keys_to_lowercase)
        edit_menu.add_separator()
        edit_menu.add_command(label="Intensity Normalization", command=self._normalize_intensities)
        edit_menu.add_separator()
        edit_menu.add_command(label="Delete Selected Spectra", command=self._delete_selected_spectra)
        edit_menu.add_separator()
        edit_menu.add_command(label="Calculate Average Spectrum per Group", command=self._calculate_average_spectra)
        edit_menu.add_separator()
        
        # Regex Update submenu
        regex_menu = tk.Menu(edit_menu, tearoff=0)
        edit_menu.add_command(label="Regex Update", command=self._open_regex_editor)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        
        # Spectrum Name submenu
        spectrum_name_menu = tk.Menu(view_menu, tearoff=0)
        view_menu.add_cascade(label="Spectrum Name", menu=spectrum_name_menu)
        
        self.spectrum_name_var = tk.StringVar(value="Numbered")
        spectrum_name_menu.add_radiobutton(label="Numbered", variable=self.spectrum_name_var, 
                                         value="Numbered", command=self._update_spectrum_names)
        
        # Will be populated when data is loaded
        self.spectrum_name_menu = spectrum_name_menu
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self.open_file())
        self.root.bind('<Control-e>', lambda e: self.export_all_spectra())
        
    def _create_widgets(self):
        """Create the main application widgets."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create paned windows for layout
        # Main horizontal split
        main_paned = ttk.PanedWindow(main_frame, orient='horizontal')
        main_paned.pack(fill='both', expand=True)
        
        # Left panel (spectrum tree)
        left_frame = ttk.Frame(main_paned, width=300)
        main_paned.add(left_frame, weight=1)
        
        # Right panel (metadata and visualization)
        right_paned = ttk.PanedWindow(main_paned, orient='vertical')
        main_paned.add(right_paned, weight=3)
        
        # Top right (metadata editor)
        metadata_frame = ttk.Frame(right_paned, height=300)
        right_paned.add(metadata_frame, weight=1)
        
        # Bottom right (visualization and tables)
        bottom_paned = ttk.PanedWindow(right_paned, orient='horizontal')
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
            left_frame, 
            on_selection_changed=self._on_spectrum_selection_changed
        )
        self.spectrum_tree.pack(fill='both', expand=True)
        
        self.metadata_editor = MetadataEditor(
            metadata_frame,
            on_metadata_changed=self._on_metadata_changed
        )
        self.metadata_editor.pack(fill='both', expand=True)
        
        self.ion_table = IonDataTable(table_frame)
        self.ion_table.pack(fill='both', expand=True)
        
        self.spectrum_viz = SpectrumVisualization(viz_frame)
        self.spectrum_viz.pack(fill='both', expand=True)
        
        self.similarity_viz = CosineSimilarityVisualization(similarity_frame)
        self.similarity_viz.pack(fill='both', expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Open an MGF file to get started")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief='sunken')
        status_bar.pack(side='bottom', fill='x')
        
        # Initial state
        self._set_components_enabled(False)
        
    def _set_components_enabled(self, enabled: bool):
        """Enable or disable components based on data availability."""
        state = 'normal' if enabled else 'disabled'
        
        # This could be expanded to actually disable/enable widgets
        # For now, components handle empty data gracefully
        pass
        
    def open_file(self):
        """Open and parse an MGF file."""
        file_path = filedialog.askopenfilename(
            title="Open MGF File",
            filetypes=[
                ("MGF files", "*.mgf"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
            
        try:
            self.status_var.set("Loading file...")
            self.root.update()
            
            # Parse the file
            spectra = self.parser.parse_file(file_path)
            
            if not spectra:
                messagebox.showwarning("Warning", "No spectra found in the file.")
                return
                
            self.current_file = file_path
            
            # Update components
            self.spectrum_tree.load_data(self.parser)
            self._update_spectrum_name_menu()
            self._set_components_enabled(True)
            
            # Update status
            filename = os.path.basename(file_path)
            self.status_var.set(f"Loaded {len(spectra)} spectra from {filename}")
            
            # Update window title
            self.root.title(f"MGF Explorer - {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
            self.status_var.set("Error loading file")
            
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
            lambda: self._update_components_with_selection(selected_spectrum_ids)
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
                self.status_var.set(f"Selected {len(selected_spectrum_ids)} spectra "
                                  f"(detailed views disabled for performance)")
                                  
            elif len(selected_spectrum_ids) > 100:
                # For large selections, limit what we show in detailed views
                limited_ids = selected_spectrum_ids[:10]
                self.ion_table.load_data(self.parser, limited_ids)
                self.spectrum_viz.load_data(self.parser, limited_ids)
                
                # Disable similarity calculation for large selections
                self.similarity_viz.load_data(self.parser, [])
                
                # Update status to reflect the limitation
                self.status_var.set(f"Selected {len(selected_spectrum_ids)} spectra "
                                  f"(showing details for first 10, similarity disabled)")
                                  
            elif len(selected_spectrum_ids) > 10:
                # Only show first 10 spectra in detailed views
                limited_ids = selected_spectrum_ids[:10]
                self.ion_table.load_data(self.parser, limited_ids)
                self.spectrum_viz.load_data(self.parser, limited_ids)
                
                # Load similarity data (this will handle its own performance limits)
                self.similarity_viz.load_data(self.parser, selected_spectrum_ids)
                
                # Update status to reflect the limitation
                self.status_var.set(f"Selected {len(selected_spectrum_ids)} spectra "
                                  f"(showing details for first 10)")
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
            
    def _add_new_key_value(self):
        """Add a new key-value pair via the Edit menu."""
        if not self.parser or not self.parser.spectra:
            messagebox.showwarning("Warning", "No data loaded. Please open an MGF file first.")
            return
            
        selected_ids = self.spectrum_tree.get_selected_spectrum_ids()
        from .gui_components import AddKeyValueDialog
        dialog = AddKeyValueDialog(self.root, self.parser, selected_ids)
        if dialog.result:
            self._on_metadata_changed()
            
    def _keys_to_uppercase(self):
        """Convert all key names to uppercase via the Edit menu."""
        if not self.parser or not self.parser.spectra:
            messagebox.showwarning("Warning", "No data loaded. Please open an MGF file first.")
            return
            
        if not messagebox.askyesno("Convert Keys", "Convert all key names to UPPERCASE?"):
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
            messagebox.showwarning("Warning", "No data loaded. Please open an MGF file first.")
            return
            
        if not messagebox.askyesno("Convert Keys", "Convert all key names to lowercase?"):
            return
            
        # Get all current keys
        all_keys = self.parser.get_all_metadata_keys()
        
        for old_key in all_keys:
            new_key = old_key.lower()
            if old_key != new_key:
                self.parser.rename_key_in_all_spectra(old_key, new_key)
                
        self._on_metadata_changed()
        
    def _normalize_intensities(self):
        """Normalize intensities so that the most abundant peak in each spectrum has intensity 1."""
        if not self.parser or not self.parser.spectra:
            messagebox.showwarning("Warning", "No data loaded. Please open an MGF file first.")
            return
            
        # Ask for confirmation
        if not messagebox.askyesno(
            "Intensity Normalization", 
            "Normalize intensities in all spectra so that the most abundant peak has intensity 1?\n\n"
            "This will modify the intensity values and cannot be undone."
        ):
            return
            
        try:
            self.status_var.set("Normalizing intensities...")
            self.root.update()
            
            # Normalize all spectra
            self.parser.normalize_intensities()
            
            # Update visualization and data
            self._on_metadata_changed()
            
            self.status_var.set(f"Normalized intensities in {len(self.parser.spectra)} spectra")
            
            messagebox.showinfo(
                "Normalization Complete", 
                f"Successfully normalized intensities in {len(self.parser.spectra)} spectra."
            )
            
        except Exception as e:
            messagebox.showerror("Normalization Error", f"Failed to normalize intensities:\n{str(e)}")
            self.status_var.set("Normalization failed")
        
    def _update_spectrum_name_menu(self):
        """Update the spectrum name menu with available metadata fields."""
        if not self.parser:
            return
            
        # Clear existing field options (keep "Numbered")
        menu = self.spectrum_name_menu
        
        # Remove all items except "Numbered"
        last_index = menu.index('end')
        if last_index is not None and last_index > 0:
            menu.delete(1, last_index)
            
        # Add separator
        menu.add_separator()
        
        # Add metadata fields as options
        all_keys = self.parser.get_all_metadata_keys()
        for key in all_keys:
            menu.add_radiobutton(label=key, variable=self.spectrum_name_var, 
                               value=key, command=self._update_spectrum_names)
                               
    def _update_spectrum_names(self):
        """Update how spectrum names are displayed."""
        if not self.parser:
            return
            
        # Set the naming scheme in the spectrum tree
        naming_scheme = self.spectrum_name_var.get()
        self.spectrum_tree.set_naming_scheme(naming_scheme)
        
        # Refresh the tree to show updated names
        self.spectrum_tree.load_data(self.parser)
        
    def _open_regex_editor(self):
        """Open the regex editor dialog."""
        if not self.parser or not self.parser.spectra:
            messagebox.showwarning("Warning", "No data loaded. Please open an MGF file first.")
            return
            
        from .gui_components import RegexEditorDialog
        dialog = RegexEditorDialog(self.root, self.parser)
        if dialog.changes_made:
            self._on_metadata_changed()
            
    def _delete_selected_spectra(self):
        """Delete the currently selected spectra."""
        if not self.parser or not self.parser.spectra:
            messagebox.showwarning("Warning", "No data loaded. Please open an MGF file first.")
            return
            
        selected_ids = self.spectrum_tree.get_selected_spectrum_ids()
        if not selected_ids:
            messagebox.showwarning("Warning", "No spectra selected.")
            return
            
        # Confirmation dialog
        num_selected = len(selected_ids)
        total_spectra = len(self.parser.spectra)
        
        message = (f"Are you sure you want to delete {num_selected} selected spectra?\n\n"
                  f"This will remove them permanently from the current session.\n"
                  f"Remaining spectra: {total_spectra - num_selected}")
        
        if not messagebox.askyesno("Confirm Delete", message, icon='warning'):
            return
            
        # Remove selected spectra
        self.parser.spectra = [s for s in self.parser.spectra if s.spectrum_id not in selected_ids]
        
        # Update the status
        remaining_count = len(self.parser.spectra)
        self.status_var.set(f"Deleted {num_selected} spectra. {remaining_count} remaining.")
        
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
            for component in [self.spectrum_tree, self.metadata_editor, self.ion_table, self.spectrum_viz]:
                if hasattr(component, 'load_data'):
                    component.load_data(self.parser, [])
            self.root.title("MGF Explorer")
            
    def _calculate_average_spectra(self):
        """Calculate average spectrum per group."""
        if not self.parser or not self.parser.spectra:
            messagebox.showwarning("Warning", "No data loaded. Please open an MGF file first.")
            return
            
        # Check if grouping is applied
        if not self.spectrum_tree.selected_grouping_tags:
            messagebox.showwarning("Warning", "No grouping applied. Please set grouping tags first.")
            return
            
        # Show parameter dialog
        from .gui_components import AverageSpectrumDialog
        dialog = AverageSpectrumDialog(self.root)
        if not dialog.result:
            return
            
        # Get parameters from dialog
        binning_mz = dialog.result['binning_mz']
        averaging_method = dialog.result['averaging_method']
        new_key = dialog.result['new_key']
        new_value = dialog.result['new_value']
        
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
            single_spectrum_groups = sum(1 for spectra_list in grouped_spectra.values() if len(spectra_list) == 1)
            
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
                    messagebox.showinfo("No Averages Created", 
                                      f"All {total_groups} groups contain only one spectrum each.\n"
                                      "No average spectra were created.")
                else:
                    messagebox.showwarning("Warning", "No average spectra could be created.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to calculate average spectra: {str(e)}")
            
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
                    current[path_part] = {'spectra': [], 'children': {}}
                current = current[path_part]['children']
                
            # Add spectrum to the final level
            final_level = hierarchy
            for path_part in path[:-1]:
                final_level = final_level[path_part]['children']
            if path:
                final_level[path[-1]]['spectra'].append(spectrum)
            else:
                # No valid grouping path, add to root
                if '_ungrouped_' not in hierarchy:
                    hierarchy['_ungrouped_'] = {'spectra': [], 'children': {}}
                hierarchy['_ungrouped_']['spectra'].append(spectrum)
        
        # Extract groups that have spectra (leaf nodes)
        self._extract_leaf_groups(hierarchy, grouped_spectra, [])
        
        return grouped_spectra
        
    def _extract_leaf_groups(self, hierarchy, grouped_spectra, path):
        """Extract leaf groups that contain spectra."""
        for key, data in hierarchy.items():
            current_path = path + [key]
            
            # If this group has spectra and no children, it's a leaf group
            if data['spectra'] and not data['children']:
                group_key = " -> ".join(current_path)
                grouped_spectra[group_key] = data['spectra']
            
            # If this group has children, recursively check them
            if data['children']:
                self._extract_leaf_groups(data['children'], grouped_spectra, current_path)
                
            # If this group has both spectra and children, the spectra at this level form a group
            if data['spectra'] and data['children']:
                group_key = " -> ".join(current_path) + " (direct)"
                grouped_spectra[group_key] = data['spectra']
    
    def _create_average_spectra(self, grouped_spectra, binning_mz, averaging_method, new_key, new_value):
        """Create average spectra for each group."""
        new_spectra = []
        skipped_groups = 0
        
        for group_name, spectra_list in grouped_spectra.items():
            if len(spectra_list) < 2:
                skipped_groups += 1
                continue  # Skip groups with only one spectrum
                
            try:
                average_spectrum = self._calculate_single_average_spectrum(
                    spectra_list, binning_mz, averaging_method, new_key, new_value, group_name
                )
                if average_spectrum:
                    new_spectra.append(average_spectrum)
            except Exception as e:
                print(f"Error creating average for group {group_name}: {e}")
                continue
        
        if skipped_groups > 0:
            print(f"Skipped {skipped_groups} groups with only one spectrum")
                
        return new_spectra
    
    def _calculate_single_average_spectrum(self, spectra_list, binning_mz, averaging_method, new_key, new_value, group_name):
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
            normalized_ions = self._normalize_spectra_by_common_peaks(all_ions, common_peaks, binning_mz)
            
            # Bin and average the ions
            averaged_ions = self._bin_and_average_ions(normalized_ions, binning_mz, averaging_method)
            
            # Create new spectrum
            average_spectrum = Spectrum(0)  # ID will be set later
            if len(averaged_ions) > 0:
                mz_values = [ion[0] for ion in averaged_ions]
                intensity_values = [ion[1] for ion in averaged_ions]
                average_spectrum.set_ions(mz_values, intensity_values)
        
        # Calculate average metadata
        self._calculate_average_metadata(average_spectrum, spectra_list, new_key, new_value, group_name)
        
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
                    averaged_peak = self._process_peak_group(current_group, averaging_method)
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
            avg_mz = sum(mz * intensity for mz, intensity in final_contributions) / total_intensity
        else:
            avg_mz = sum(mz_values) / len(mz_values)  # Simple average if no intensity
        
        # Calculate average or median intensity
        if averaging_method == "average":
            avg_intensity = np.mean(intensities)
        else:  # median
            avg_intensity = np.median(intensities)
        
        return [avg_mz, avg_intensity]
    
    def _calculate_average_metadata(self, average_spectrum, spectra_list, new_key, new_value, group_name):
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
        all_keys.discard(new_key)    # Already set
        
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
    
    def export_all_spectra(self):
        """Export all spectra to a new MGF file."""
        if not self.parser.spectra:
            messagebox.showwarning("Warning", "No spectra loaded to export.")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Export All Spectra to MGF File",
            defaultextension=".mgf",
            filetypes=[
                ("MGF files", "*.mgf"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
            
        try:
            self.status_var.set("Exporting spectra...")
            self.root.update()
            
            # Export all spectra
            self.parser.export_to_mgf(file_path)
            
            # Update status
            self.status_var.set(f"Exported {len(self.parser.spectra)} spectra to {os.path.basename(file_path)}")
            
            messagebox.showinfo(
                "Export Complete", 
                f"Successfully exported {len(self.parser.spectra)} spectra to:\n{file_path}"
            )
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export spectra:\n{str(e)}")
            self.status_var.set("Export failed")
            
    def show_about(self):
        """Show about dialog."""
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
            "• View ion data in tables"
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
