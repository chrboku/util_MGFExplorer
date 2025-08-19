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
    IonDataTable
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
        table_frame = ttk.Frame(bottom_paned, width=400)
        bottom_paned.add(table_frame, weight=1)
        
        # Bottom right (spectrum visualization)
        viz_frame = ttk.Frame(bottom_paned, width=600)
        bottom_paned.add(viz_frame, weight=2)
        
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
            self.status_var.set(f"Ready - {len(self.parser.spectra)} spectra loaded")
        else:
            # Update displays with selected spectra
            self.metadata_editor.load_data(self.parser, selected_spectrum_ids)
            self.ion_table.load_data(self.parser, selected_spectrum_ids)
            self.spectrum_viz.load_data(self.parser, selected_spectrum_ids)
            
            if len(selected_spectrum_ids) == 1:
                self.status_var.set(f"Selected spectrum {selected_spectrum_ids[0]}")
            else:
                self.status_var.set(f"Selected {len(selected_spectrum_ids)} spectra")
                
    def _on_metadata_changed(self):
        """Handle metadata changes."""
        # Refresh the spectrum tree to show updated grouping
        self.spectrum_tree.load_data(self.parser)
        
        # Get current selection and refresh other components
        selected_ids = self.spectrum_tree.get_selected_spectrum_ids()
        if selected_ids:
            self.metadata_editor.load_data(self.parser, selected_ids)
            self.ion_table.load_data(self.parser, selected_ids)
            self.spectrum_viz.load_data(self.parser, selected_ids)
            
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
