"""
GUI components for the MGF Explorer application.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from typing import List, Dict, Any, Optional
from .mgf_parser import MGFParser, Spectrum


class SpectrumTreeView(ttk.Frame):
    """Tree view component for displaying spectra list and grouping options."""
    
    def __init__(self, parent, on_selection_changed=None):
        super().__init__(parent)
        self.on_selection_changed = on_selection_changed
        self.parser: Optional[MGFParser] = None
        self.selected_grouping_keys: List[str] = []
        
        self._create_widgets()
        
    def _create_widgets(self):
        """Create the tree view widgets."""
        # Grouping frame
        group_frame = ttk.LabelFrame(self, text="Grouping Options", padding=5)
        group_frame.pack(fill='x', padx=5, pady=5)
        
        self.grouping_vars = {}
        self.grouping_frame = group_frame
        
        # Tree view frame
        tree_frame = ttk.LabelFrame(self, text="Spectra", padding=5)
        tree_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Tree view with scrollbar
        tree_container = ttk.Frame(tree_frame)
        tree_container.pack(fill='both', expand=True)
        
        self.tree = ttk.Treeview(tree_container, selectmode='extended')
        tree_scrollbar = ttk.Scrollbar(tree_container, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scrollbar.set)
        
        self.tree.pack(side='left', fill='both', expand=True)
        tree_scrollbar.pack(side='right', fill='y')
        
        # Bind selection event
        self.tree.bind('<<TreeviewSelect>>', self._on_tree_selection)
        
    def load_data(self, parser: MGFParser):
        """Load spectra data into the tree view."""
        self.parser = parser
        self._update_grouping_options()
        self._populate_tree()
        
    def _update_grouping_options(self):
        """Update the grouping checkboxes based on available metadata keys."""
        # Clear existing checkboxes
        for widget in self.grouping_frame.winfo_children():
            widget.destroy()
            
        self.grouping_vars.clear()
        
        if not self.parser:
            return
            
        # Create checkboxes for each metadata key
        all_keys = self.parser.get_all_metadata_keys()
        
        for i, key in enumerate(all_keys):
            var = tk.BooleanVar()
            cb = ttk.Checkbutton(
                self.grouping_frame, 
                text=key, 
                variable=var,
                command=self._on_grouping_changed
            )
            cb.grid(row=i//3, column=i%3, sticky='w', padx=5, pady=2)
            self.grouping_vars[key] = var
            
    def _on_grouping_changed(self):
        """Handle grouping option changes."""
        self.selected_grouping_keys = [
            key for key, var in self.grouping_vars.items() if var.get()
        ]
        self._populate_tree()
        
    def _populate_tree(self):
        """Populate the tree view with spectra."""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        if not self.parser:
            return
            
        # Configure columns
        columns = ['ID'] + self.selected_grouping_keys
        self.tree['columns'] = columns
        self.tree['show'] = 'tree headings'
        
        # Configure column headings
        self.tree.heading('#0', text='Spectrum')
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)
            
        # Add spectra to tree
        for spectrum in self.parser.spectra:
            values = [spectrum.spectrum_id]
            for key in self.selected_grouping_keys:
                values.append(spectrum.get_metadata_value(key) or '')
                
            self.tree.insert('', 'end', text=f'Spectrum {spectrum.spectrum_id}', 
                           values=values, tags=(spectrum.spectrum_id,))
            
    def _on_tree_selection(self, event):
        """Handle tree selection changes."""
        selected_items = self.tree.selection()
        selected_spectrum_ids = []
        
        for item in selected_items:
            tags = self.tree.item(item, 'tags')
            if tags:
                selected_spectrum_ids.append(int(tags[0]))
                
        if self.on_selection_changed:
            self.on_selection_changed(selected_spectrum_ids)
            
    def get_selected_spectrum_ids(self) -> List[int]:
        """Get currently selected spectrum IDs."""
        selected_items = self.tree.selection()
        selected_ids = []
        
        for item in selected_items:
            tags = self.tree.item(item, 'tags')
            if tags:
                selected_ids.append(int(tags[0]))
                
        return selected_ids


class MetadataEditor(ttk.Frame):
    """Component for viewing and editing metadata."""
    
    def __init__(self, parent, on_metadata_changed=None):
        super().__init__(parent)
        self.on_metadata_changed = on_metadata_changed
        self.parser: Optional[MGFParser] = None
        self.selected_spectrum_ids: List[int] = []
        
        self._create_widgets()
        
    def _create_widgets(self):
        """Create the metadata editor widgets."""
        # Header
        header_frame = ttk.Frame(self)
        header_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(header_frame, text="Metadata Editor", font=('Arial', 12, 'bold')).pack()
        
        # Metadata table
        table_frame = ttk.LabelFrame(self, text="Metadata", padding=5)
        table_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create treeview for metadata
        columns = ('Key', 'Value', 'Unique Values')
        self.metadata_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=10)
        
        for col in columns:
            self.metadata_tree.heading(col, text=col)
            self.metadata_tree.column(col, width=150)
            
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.metadata_tree.yview)
        h_scrollbar = ttk.Scrollbar(table_frame, orient='horizontal', command=self.metadata_tree.xview)
        
        self.metadata_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.metadata_tree.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        # Edit controls
        edit_frame = ttk.Frame(self)
        edit_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(edit_frame, text="Key:").grid(row=0, column=0, sticky='w', padx=5)
        self.key_var = tk.StringVar()
        self.key_entry = ttk.Entry(edit_frame, textvariable=self.key_var, width=20)
        self.key_entry.grid(row=0, column=1, padx=5)
        
        ttk.Label(edit_frame, text="Value:").grid(row=0, column=2, sticky='w', padx=5)
        self.value_var = tk.StringVar()
        self.value_entry = ttk.Entry(edit_frame, textvariable=self.value_var, width=30)
        self.value_entry.grid(row=0, column=3, padx=5)
        
        ttk.Button(edit_frame, text="Update Value", command=self._update_value).grid(row=0, column=4, padx=5)
        ttk.Button(edit_frame, text="Rename Key", command=self._rename_key).grid(row=0, column=5, padx=5)
        
        # Bind selection event
        self.metadata_tree.bind('<<TreeviewSelect>>', self._on_metadata_selection)
        
    def load_data(self, parser: MGFParser, selected_spectrum_ids: List[int]):
        """Load metadata for selected spectra."""
        self.parser = parser
        self.selected_spectrum_ids = selected_spectrum_ids
        self._populate_metadata()
        
    def _populate_metadata(self):
        """Populate the metadata table."""
        # Clear existing items
        for item in self.metadata_tree.get_children():
            self.metadata_tree.delete(item)
            
        if not self.parser or not self.selected_spectrum_ids:
            return
            
        # Get all metadata keys
        all_keys = self.parser.get_all_metadata_keys()
        
        for key in all_keys:
            # Get value from first selected spectrum
            value = ""
            if self.selected_spectrum_ids:
                first_spectrum = next(
                    (s for s in self.parser.spectra if s.spectrum_id == self.selected_spectrum_ids[0]), 
                    None
                )
                if first_spectrum:
                    value = first_spectrum.get_metadata_value(key) or ""
                    
            # Get unique values for this key
            unique_values = self.parser.get_unique_values_for_key(key)
            unique_str = "; ".join(unique_values[:5])  # Show first 5 unique values
            if len(unique_values) > 5:
                unique_str += f" ... ({len(unique_values)} total)"
                
            self.metadata_tree.insert('', 'end', values=(key, value, unique_str))
            
    def _on_metadata_selection(self, event):
        """Handle metadata selection."""
        selected_items = self.metadata_tree.selection()
        if selected_items:
            item = selected_items[0]
            values = self.metadata_tree.item(item, 'values')
            if values:
                self.key_var.set(values[0])
                self.value_var.set(values[1])
                
    def _update_value(self):
        """Update metadata value for selected spectra."""
        key = self.key_var.get().strip()
        value = self.value_var.get().strip()
        
        if not key or not self.parser or not self.selected_spectrum_ids:
            return
            
        self.parser.update_key_value_in_selected_spectra(self.selected_spectrum_ids, key, value)
        self._populate_metadata()
        
        if self.on_metadata_changed:
            self.on_metadata_changed()
            
    def _rename_key(self):
        """Rename a metadata key."""
        old_key = self.key_var.get().strip()
        
        if not old_key:
            return
            
        new_key = tk.simpledialog.askstring("Rename Key", f"Enter new name for '{old_key}':")
        if new_key and new_key != old_key:
            self.parser.rename_key_in_all_spectra(old_key, new_key)
            self._populate_metadata()
            
            if self.on_metadata_changed:
                self.on_metadata_changed()


class SpectrumVisualization(ttk.Frame):
    """Component for visualizing spectra as stick charts."""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.parser: Optional[MGFParser] = None
        self.selected_spectrum_ids: List[int] = []
        
        self._create_widgets()
        
    def _create_widgets(self):
        """Create the visualization widgets."""
        # Header
        header_frame = ttk.Frame(self)
        header_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(header_frame, text="Spectrum Visualization", font=('Arial', 12, 'bold')).pack()
        
        # Matplotlib figure
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
        
    def load_data(self, parser: MGFParser, selected_spectrum_ids: List[int]):
        """Load and visualize selected spectra."""
        self.parser = parser
        self.selected_spectrum_ids = selected_spectrum_ids
        self._plot_spectra()
        
    def _plot_spectra(self):
        """Plot the selected spectra."""
        self.figure.clear()
        
        if not self.parser or not self.selected_spectrum_ids:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No spectra selected', ha='center', va='center', transform=ax.transAxes)
            self.canvas.draw()
            return
            
        # Get selected spectra
        selected_spectra = [
            s for s in self.parser.spectra if s.spectrum_id in self.selected_spectrum_ids
        ]
        
        if not selected_spectra:
            return
            
        # Create subplots
        n_spectra = len(selected_spectra)
        if n_spectra == 1:
            ax = self.figure.add_subplot(111)
            self._plot_single_spectrum(ax, selected_spectra[0])
        else:
            for i, spectrum in enumerate(selected_spectra):
                ax = self.figure.add_subplot(n_spectra, 1, i+1)
                self._plot_single_spectrum(ax, spectrum)
                
        self.figure.tight_layout()
        self.canvas.draw()
        
    def _plot_single_spectrum(self, ax, spectrum: Spectrum):
        """Plot a single spectrum as a stick chart."""
        if spectrum.ions.size == 0:
            ax.text(0.5, 0.5, 'No ion data', ha='center', va='center', transform=ax.transAxes)
            return
            
        mz_values = spectrum.ions[:, 0]
        intensity_values = spectrum.ions[:, 1]
        
        # Create stick plot
        ax.vlines(mz_values, 0, intensity_values, colors='blue', linewidth=1.5)
        ax.set_xlabel('m/z')
        ax.set_ylabel('Intensity')
        ax.set_title(f'Spectrum {spectrum.spectrum_id}')
        ax.grid(True, alpha=0.3)
        
        # Set limits
        if len(mz_values) > 0:
            ax.set_xlim(mz_values.min() * 0.95, mz_values.max() * 1.05)
            ax.set_ylim(0, intensity_values.max() * 1.1)


class IonDataTable(ttk.Frame):
    """Component for displaying ion data in table format."""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.parser: Optional[MGFParser] = None
        self.selected_spectrum_ids: List[int] = []
        
        self._create_widgets()
        
    def _create_widgets(self):
        """Create the table widgets."""
        # Header
        header_frame = ttk.Frame(self)
        header_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(header_frame, text="Ion Data", font=('Arial', 12, 'bold')).pack()
        
        # Notebook for multiple spectra
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
    def load_data(self, parser: MGFParser, selected_spectrum_ids: List[int]):
        """Load ion data for selected spectra."""
        self.parser = parser
        self.selected_spectrum_ids = selected_spectrum_ids
        self._populate_tables()
        
    def _populate_tables(self):
        """Populate tables with ion data."""
        # Clear existing tabs
        for tab in self.notebook.tabs():
            self.notebook.forget(tab)
            
        if not self.parser or not self.selected_spectrum_ids:
            return
            
        # Get selected spectra
        selected_spectra = [
            s for s in self.parser.spectra if s.spectrum_id in self.selected_spectrum_ids
        ]
        
        for spectrum in selected_spectra:
            self._create_table_for_spectrum(spectrum)
            
    def _create_table_for_spectrum(self, spectrum: Spectrum):
        """Create a table tab for a single spectrum."""
        # Create frame for this spectrum
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=f'Spectrum {spectrum.spectrum_id}')
        
        # Create treeview
        columns = ('Index', 'm/z', 'Intensity')
        tree = ttk.Treeview(frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)
            
        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(frame, orient='vertical', command=tree.yview)
        h_scrollbar = ttk.Scrollbar(frame, orient='horizontal', command=tree.xview)
        
        tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        tree.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        
        # Populate with ion data
        if spectrum.ions.size > 0:
            for i, (mz, intensity) in enumerate(spectrum.ions):
                tree.insert('', 'end', values=(i+1, f'{mz:.6f}', f'{intensity:.3f}'))
