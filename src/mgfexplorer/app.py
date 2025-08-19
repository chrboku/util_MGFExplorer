"""
Main application window for the MGF Explorer.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import os
from .mgf_parser import MGFParser
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
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Add New Key-Value Pair...", command=self._add_new_key_value)
        edit_menu.add_separator()
        edit_menu.add_command(label="Convert Keys to UPPERCASE", command=self._keys_to_uppercase)
        edit_menu.add_command(label="Convert Keys to lowercase", command=self._keys_to_lowercase)
        edit_menu.add_separator()
        edit_menu.add_command(label="Delete Selected Spectra", command=self._delete_selected_spectra)
        edit_menu.add_separator()
        
        # Regex Update submenu
        regex_menu = tk.Menu(edit_menu, tearoff=0)
        edit_menu.add_cascade(label="Regex Update", menu=regex_menu)
        regex_menu.add_command(label="Open Regex Editor...", command=self._open_regex_editor)
        
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
        """Handle spectrum selection changes."""
        if not selected_spectrum_ids:
            # Clear all displays
            self.metadata_editor.load_data(self.parser, [])
            self.ion_table.load_data(self.parser, [])
            self.spectrum_viz.load_data(self.parser, [])
            self.similarity_viz.load_data(self.parser, [])
            self.status_var.set(f"Ready - {len(self.parser.spectra)} spectra loaded")
        else:
            # Update displays with selected spectra
            self.metadata_editor.load_data(self.parser, selected_spectrum_ids)
            self.ion_table.load_data(self.parser, selected_spectrum_ids)
            self.spectrum_viz.load_data(self.parser, selected_spectrum_ids)
            self.similarity_viz.load_data(self.parser, selected_spectrum_ids)
            
            if len(selected_spectrum_ids) == 1:
                self.status_var.set(f"Selected spectrum {selected_spectrum_ids[0]}")
            else:
                self.status_var.set(f"Selected {len(selected_spectrum_ids)} spectra")
                
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
