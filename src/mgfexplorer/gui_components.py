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
import re
from .mgf_parser import MGFParser, Spectrum


class SpectrumTreeView(ttk.Frame):
    """Tree view component for displaying spectra list with tag-based grouping."""
    
    def __init__(self, parent, on_selection_changed=None):
        super().__init__(parent)
        self.on_selection_changed = on_selection_changed
        self.parser: Optional[MGFParser] = None
        self.selected_grouping_tags: List[str] = []
        self.naming_scheme: str = "Numbered"  # Default naming scheme
        
        self._create_widgets()
        
    def _create_widgets(self):
        """Create the tree view widgets."""
        # Tag selection frame
        tag_frame = ttk.LabelFrame(self, text="Grouping Tags", padding=5)
        tag_frame.pack(fill='x', padx=5, pady=5)
        
        # Instructions
        ttk.Label(tag_frame, text="Enter metadata keys for grouping (comma-separated):").pack(anchor='w')
        
        # Tag entry with autocomplete-like functionality
        self.tag_var = tk.StringVar()
        self.tag_entry = ttk.Entry(tag_frame, textvariable=self.tag_var, width=40)
        self.tag_entry.pack(fill='x', pady=5)
        self.tag_entry.bind('<KeyRelease>', self._on_tag_entry_change)
        self.tag_entry.bind('<Return>', self._apply_grouping)
        
        # Available tags display
        self.tags_label = ttk.Label(tag_frame, text="Available keys: ", wraplength=250)
        self.tags_label.pack(anchor='w', pady=2)
        
        # Apply button
        ttk.Button(tag_frame, text="Apply Grouping", command=self._apply_grouping).pack(pady=5)
        
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
        self._update_available_tags()
        self._populate_tree()
        
    def _update_available_tags(self):
        """Update the display of available metadata keys."""
        if not self.parser:
            self.tags_label.config(text="Available keys: ")
            return
            
        all_keys = self.parser.get_all_metadata_keys()
        keys_text = ", ".join(all_keys) if all_keys else "None"
        self.tags_label.config(text=f"Available keys: {keys_text}")
        
    def _on_tag_entry_change(self, event=None):
        """Handle changes in the tag entry field."""
        # Could add autocomplete functionality here in the future
        pass
        
    def _apply_grouping(self, event=None):
        """Apply the grouping based on entered tags."""
        tag_text = self.tag_var.get().strip()
        if tag_text:
            # Parse comma-separated tags
            self.selected_grouping_tags = [tag.strip() for tag in tag_text.split(',') if tag.strip()]
        else:
            self.selected_grouping_tags = []
            
        self._populate_tree()
        
    def _populate_tree(self):
        """Populate the tree view with spectra using hierarchical grouping."""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        if not self.parser:
            return
            
        # Configure tree display
        self.tree['show'] = 'tree'
        
        if not self.selected_grouping_tags:
            # No grouping - show flat list
            self._populate_flat_list()
        else:
            # Hierarchical grouping
            self._populate_hierarchical_tree()
            
    def _populate_flat_list(self):
        """Populate tree with flat list of spectra."""
        for spectrum in self.parser.spectra:
            display_name = self._get_spectrum_display_name(spectrum)
            item_id = self.tree.insert('', 'end', text=display_name, 
                                     tags=('spectrum', spectrum.spectrum_id))
                                     
    def _populate_hierarchical_tree(self):
        """Populate tree with hierarchical grouping."""
        # Build hierarchy
        hierarchy = {}
        
        for spectrum in self.parser.spectra:
            # Get values for grouping tags
            path = []
            for tag in self.selected_grouping_tags:
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
        
        # Populate tree from hierarchy
        self._add_hierarchy_to_tree('', hierarchy, 0)
        
    def _add_hierarchy_to_tree(self, parent_id: str, hierarchy: dict, level: int):
        """Recursively add hierarchy to tree."""
        for key, data in hierarchy.items():
            # Create group node
            display_text = key if key != '_ungrouped_' else 'Ungrouped'
            group_id = self.tree.insert(parent_id, 'end', text=display_text, 
                                       tags=('group', level))
            
            # Add spectra in this group
            for spectrum in data['spectra']:
                display_name = self._get_spectrum_display_name(spectrum)
                spectrum_id = self.tree.insert(group_id, 'end', 
                                             text=display_name,
                                             tags=('spectrum', spectrum.spectrum_id))
            
            # Recursively add children
            if data['children']:
                self._add_hierarchy_to_tree(group_id, data['children'], level + 1)
                
            # Expand group if it has items
            if data['spectra'] or data['children']:
                self.tree.item(group_id, open=True)
                
    def _on_tree_selection(self, event):
        """Handle tree selection changes."""
        selected_items = self.tree.selection()
        selected_spectrum_ids = []
        
        for item in selected_items:
            tags = self.tree.item(item, 'tags')
            if not tags:
                continue
                
            if tags[0] == 'spectrum':
                # Direct spectrum selection
                selected_spectrum_ids.append(int(tags[1]))
            elif tags[0] == 'group':
                # Group selection - get all spectra in group and children
                group_spectra = self._get_spectra_in_group(item)
                selected_spectrum_ids.extend(group_spectra)
                
        # Remove duplicates and sort
        selected_spectrum_ids = sorted(list(set(selected_spectrum_ids)))
                
        if self.on_selection_changed:
            self.on_selection_changed(selected_spectrum_ids)
            
    def _get_spectra_in_group(self, group_item):
        """Get all spectrum IDs in a group and its children."""
        spectrum_ids = []
        
        def collect_spectra(item):
            for child in self.tree.get_children(item):
                tags = self.tree.item(child, 'tags')
                if tags and tags[0] == 'spectrum':
                    spectrum_ids.append(int(tags[1]))
                else:
                    # Recursive for nested groups
                    collect_spectra(child)
                    
        collect_spectra(group_item)
        return spectrum_ids
        
    def get_selected_spectrum_ids(self) -> List[int]:
        """Get currently selected spectrum IDs."""
        selected_items = self.tree.selection()
        selected_ids = []
        
        for item in selected_items:
            tags = self.tree.item(item, 'tags')
            if not tags:
                continue
                
            if tags[0] == 'spectrum':
                selected_ids.append(int(tags[1]))
            elif tags[0] == 'group':
                group_spectra = self._get_spectra_in_group(item)
                selected_ids.extend(group_spectra)
                
        return sorted(list(set(selected_ids)))
        
    def set_naming_scheme(self, scheme: str):
        """Set the naming scheme for spectrum display."""
        self.naming_scheme = scheme
        
    def _get_spectrum_display_name(self, spectrum) -> str:
        """Get the display name for a spectrum based on the current naming scheme."""
        if self.naming_scheme == "Numbered":
            return f'Spectrum {spectrum.spectrum_id}'
        else:
            # Use the specified metadata field
            value = spectrum.get_metadata_value(self.naming_scheme)
            if value:
                return f'{spectrum.spectrum_id}: {self.naming_scheme}={value}'
            else:
                return f'{spectrum.spectrum_id}: {self.naming_scheme}=<missing>'


class MetadataEditor(ttk.Frame):
    """Component for viewing and editing metadata."""
    
    def __init__(self, parent, on_metadata_changed=None):
        super().__init__(parent)
        self.on_metadata_changed = on_metadata_changed
        self.parser: Optional[MGFParser] = None
        self.selected_spectrum_ids: List[int] = []
        self.edit_var = tk.StringVar()
        self.edit_entry = None
        self.editing_item = None
        self.editing_column = None
        
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
        
        # Instructions
        instructions_frame = ttk.Frame(self)
        instructions_frame.pack(fill='x', padx=5, pady=5)
        
        instructions_text = "Double-click on Key or Value cells to edit. Press Enter to save, Escape to cancel."
        ttk.Label(instructions_frame, text=instructions_text, font=('Arial', 9), foreground='gray').pack()
        
        # Bind selection and editing events
        self.metadata_tree.bind('<<TreeviewSelect>>', self._on_metadata_selection)
        self.metadata_tree.bind('<Double-1>', self._on_double_click)
        self.metadata_tree.bind('<Button-1>', self._on_single_click)
        
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
        # This can be used for future functionality if needed
        pass
                
    def _on_single_click(self, event):
        """Handle single click to close any open editor."""
        self._close_editor()
                
    def _on_double_click(self, event):
        """Handle double click to start editing."""
        item = self.metadata_tree.identify('item', event.x, event.y)
        column = self.metadata_tree.identify('column', event.x, event.y)
        
        if item and column in ('#1', '#2'):  # Key or Value columns
            self._start_editing(item, column)
            
    def _start_editing(self, item, column):
        """Start inline editing of a cell."""
        # Close any existing editor
        self._close_editor()
        
        # Get current value
        values = self.metadata_tree.item(item, 'values')
        if not values:
            return
            
        current_value = values[0] if column == '#1' else values[1]
        self.edit_var.set(current_value)
        
        # Get cell position
        bbox = self.metadata_tree.bbox(item, column)
        if not bbox:
            return
            
        x, y, width, height = bbox
        
        # Create entry widget
        self.edit_entry = ttk.Entry(self.metadata_tree, textvariable=self.edit_var)
        self.edit_entry.place(x=x, y=y, width=width, height=height)
        self.edit_entry.focus()
        self.edit_entry.select_range(0, tk.END)
        
        # Store editing context
        self.editing_item = item
        self.editing_column = column
        
        # Bind events
        self.edit_entry.bind('<Return>', self._finish_editing)
        self.edit_entry.bind('<Escape>', self._cancel_editing)
        self.edit_entry.bind('<FocusOut>', self._cancel_editing)
        
    def _close_editor(self):
        """Close the inline editor."""
        if self.edit_entry:
            self.edit_entry.destroy()
            self.edit_entry = None
            self.editing_item = None
            self.editing_column = None
            
    def _cancel_editing(self, event=None):
        """Cancel editing without saving."""
        self._close_editor()
        
    def _finish_editing(self, event=None):
        """Finish editing and save changes."""
        if not self.editing_item or not self.editing_column:
            return
            
        new_value = self.edit_var.get().strip()
        values = self.metadata_tree.item(self.editing_item, 'values')
        
        if not values:
            self._close_editor()
            return
            
        current_key = values[0]
        current_value = values[1]
        
        if self.editing_column == '#1':  # Editing key
            self._handle_key_edit(current_key, new_value)
        else:  # Editing value
            self._handle_value_edit(current_key, new_value)
            
        self._close_editor()
        
    def _handle_key_edit(self, old_key: str, new_key: str):
        """Handle editing of a metadata key."""
        if not new_key or old_key == new_key:
            return
            
        if not self.parser:
            return
            
        # Check if new key already exists in any spectrum
        existing_spectra_with_new_key = []
        for spectrum in self.parser.spectra:
            if new_key in spectrum.metadata:
                existing_spectra_with_new_key.append(spectrum.spectrum_id)
                
        # Ask user if they want to update for all spectra
        message = f"Do you want to rename the key '{old_key}' to '{new_key}' for all loaded spectra?"
        if not messagebox.askyesno("Rename Key", message):
            return
            
        if existing_spectra_with_new_key:
            # New key already exists in some spectra
            conflict_message = (
                f"The key '{new_key}' already exists in {len(existing_spectra_with_new_key)} spectra.\n\n"
                f"What would you like to do?\n\n"
                f"• Update: Only rename '{old_key}' to '{new_key}' in spectra that don't have '{new_key}'\n"
                f"• Merge: Combine values ('{old_key}' values will overwrite '{new_key}' values)\n"
                f"• Abort: Cancel the operation"
            )
            
            result = messagebox.askyesnocancel(
                "Key Conflict", 
                conflict_message,
                title="Choose Action"
            )
            
            if result is None:  # Cancel
                return
            elif result:  # Yes - Update only
                self._rename_key_selective(old_key, new_key, existing_spectra_with_new_key)
            else:  # No - Merge
                self._rename_key_merge(old_key, new_key)
        else:
            # No conflict, rename for all spectra
            self._rename_key_all(old_key, new_key)
            
    def _rename_key_selective(self, old_key: str, new_key: str, exclude_spectrum_ids: List[int]):
        """Rename key only in spectra that don't have the new key."""
        for spectrum in self.parser.spectra:
            if spectrum.spectrum_id not in exclude_spectrum_ids:
                if old_key in spectrum.metadata:
                    spectrum.rename_metadata_key(old_key, new_key)
                    
        # Add empty key for spectra that don't have the old key
        for spectrum in self.parser.spectra:
            if spectrum.spectrum_id not in exclude_spectrum_ids and new_key not in spectrum.metadata:
                spectrum.add_metadata(new_key, "")
                
        self._refresh_after_change()
        
    def _rename_key_merge(self, old_key: str, new_key: str):
        """Merge keys, with old_key values overwriting new_key values."""
        for spectrum in self.parser.spectra:
            if old_key in spectrum.metadata:
                old_value = spectrum.metadata[old_key]
                spectrum.metadata[new_key] = old_value
                del spectrum.metadata[old_key]
            elif new_key not in spectrum.metadata:
                spectrum.add_metadata(new_key, "")
                
        self._refresh_after_change()
        
    def _rename_key_all(self, old_key: str, new_key: str):
        """Rename key for all spectra."""
        for spectrum in self.parser.spectra:
            if old_key in spectrum.metadata:
                spectrum.rename_metadata_key(old_key, new_key)
            else:
                spectrum.add_metadata(new_key, "")
                
        self._refresh_after_change()
        
    def _handle_value_edit(self, key: str, new_value: str):
        """Handle editing of a metadata value."""
        if not self.parser or not self.selected_spectrum_ids:
            return
            
        # Update value in all selected spectra
        for spectrum in self.parser.spectra:
            if spectrum.spectrum_id in self.selected_spectrum_ids:
                if key in spectrum.metadata:
                    spectrum.metadata[key] = new_value
                else:
                    spectrum.add_metadata(key, new_value)
                    
        self._refresh_after_change()
        
    def _refresh_after_change(self):
        """Refresh the view after metadata changes."""
        self._populate_metadata()
        if self.on_metadata_changed:
            self.on_metadata_changed()


class AddKeyValueDialog:
    """Dialog for adding new key-value pairs."""
    
    def __init__(self, parent, parser: MGFParser, selected_spectrum_ids: List[int]):
        self.parent = parent
        self.parser = parser
        self.selected_spectrum_ids = selected_spectrum_ids
        self.result = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Add New Key-Value Pair")
        self.dialog.geometry("400x200")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.geometry("+%d+%d" % (
            parent.winfo_rootx() + 50,
            parent.winfo_rooty() + 50
        ))
        
        self._create_widgets()
        self.dialog.wait_window()
        
    def _create_widgets(self):
        """Create dialog widgets."""
        main_frame = ttk.Frame(self.dialog, padding=10)
        main_frame.pack(fill='both', expand=True)
        
        # Key input
        ttk.Label(main_frame, text="Key Name:").grid(row=0, column=0, sticky='w', pady=5)
        self.key_var = tk.StringVar()
        key_entry = ttk.Entry(main_frame, textvariable=self.key_var, width=30)
        key_entry.grid(row=0, column=1, padx=(10, 0), pady=5)
        key_entry.focus()
        
        # Value input
        ttk.Label(main_frame, text="Value:").grid(row=1, column=0, sticky='w', pady=5)
        self.value_var = tk.StringVar()
        value_entry = ttk.Entry(main_frame, textvariable=self.value_var, width=30)
        value_entry.grid(row=1, column=1, padx=(10, 0), pady=5)
        
        # Scope selection
        ttk.Label(main_frame, text="Apply to:").grid(row=2, column=0, sticky='w', pady=5)
        self.scope_var = tk.StringVar(value="selected")
        scope_frame = ttk.Frame(main_frame)
        scope_frame.grid(row=2, column=1, padx=(10, 0), pady=5, sticky='w')
        
        selected_text = f"Selected spectra ({len(self.selected_spectrum_ids)})" if self.selected_spectrum_ids else "Selected spectra (none)"
        ttk.Radiobutton(scope_frame, text=selected_text, variable=self.scope_var, value="selected").pack(anchor='w')
        ttk.Radiobutton(scope_frame, text=f"All loaded spectra ({len(self.parser.spectra)})", variable=self.scope_var, value="all").pack(anchor='w')
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="Add", command=self._add_key_value).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Cancel", command=self._cancel).pack(side='left', padx=5)
        
        # Bind Enter key
        self.dialog.bind('<Return>', lambda e: self._add_key_value())
        self.dialog.bind('<Escape>', lambda e: self._cancel())
        
    def _add_key_value(self):
        """Add the new key-value pair."""
        key = self.key_var.get().strip()
        value = self.value_var.get().strip()
        scope = self.scope_var.get()
        
        if not key:
            messagebox.showerror("Error", "Key name cannot be empty")
            return
            
        # Check if key already exists
        existing_keys = self.parser.get_all_metadata_keys()
        if key in existing_keys:
            if not messagebox.askyesno("Key Exists", f"Key '{key}' already exists. Do you want to update its value?"):
                return
                
        # Apply to selected scope
        if scope == "selected":
            if not self.selected_spectrum_ids:
                messagebox.showwarning("Warning", "No spectra selected")
                return
            target_ids = self.selected_spectrum_ids
        else:
            target_ids = [s.spectrum_id for s in self.parser.spectra]
            
        # Add/update the key-value pair
        for spectrum in self.parser.spectra:
            if spectrum.spectrum_id in target_ids:
                spectrum.metadata[key] = value
                
        self.result = True
        self.dialog.destroy()
        
    def _cancel(self):
        """Cancel the dialog."""
        self.result = False
        self.dialog.destroy()


class RegexEditorDialog:
    """Dialog for regex-based metadata editing."""
    
    def __init__(self, parent, parser: MGFParser):
        self.parent = parent
        self.parser = parser
        self.changes_made = False
        self.current_key = None
        self.value_data = {}  # {value: {'count': int, 'updated': str}}
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Regex Metadata Editor")
        self.dialog.geometry("900x600")
        self.dialog.resizable(True, True)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.geometry("+%d+%d" % (
            parent.winfo_rootx() + 50,
            parent.winfo_rooty() + 50
        ))
        
        self._create_widgets()
        self._populate_keys()
        self.dialog.wait_window()
        
    def _create_widgets(self):
        """Create dialog widgets."""
        main_frame = ttk.Frame(self.dialog, padding=10)
        main_frame.pack(fill='both', expand=True)
        
        # Top frame with three columns
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        # Left column - Keys list
        keys_frame = ttk.LabelFrame(top_frame, text="Metadata Keys", padding=5)
        keys_frame.pack(side='left', fill='y', padx=(0, 5))
        
        # Keys listbox with scrollbar
        keys_container = ttk.Frame(keys_frame)
        keys_container.pack(fill='both', expand=True)
        
        self.keys_listbox = tk.Listbox(keys_container, width=20, height=20)
        keys_scrollbar = ttk.Scrollbar(keys_container, orient='vertical', command=self.keys_listbox.yview)
        self.keys_listbox.configure(yscrollcommand=keys_scrollbar.set)
        
        self.keys_listbox.pack(side='left', fill='both', expand=True)
        keys_scrollbar.pack(side='right', fill='y')
        
        self.keys_listbox.bind('<<ListboxSelect>>', self._on_key_selection)
        
        # Middle column - Values table
        values_frame = ttk.LabelFrame(top_frame, text="Values", padding=5)
        values_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        # Values treeview
        columns = ('Original Value', 'Count', 'Updated Value')
        self.values_tree = ttk.Treeview(values_frame, columns=columns, show='headings', height=20)
        
        for col in columns:
            self.values_tree.heading(col, text=col)
            
        # Set column widths
        self.values_tree.column('Original Value', width=200)
        self.values_tree.column('Count', width=80)
        self.values_tree.column('Updated Value', width=200)
        
        # Scrollbars for values tree
        values_v_scrollbar = ttk.Scrollbar(values_frame, orient='vertical', command=self.values_tree.yview)
        values_h_scrollbar = ttk.Scrollbar(values_frame, orient='horizontal', command=self.values_tree.xview)
        
        self.values_tree.configure(yscrollcommand=values_v_scrollbar.set, xscrollcommand=values_h_scrollbar.set)
        
        self.values_tree.grid(row=0, column=0, sticky='nsew')
        values_v_scrollbar.grid(row=0, column=1, sticky='ns')
        values_h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        values_frame.grid_rowconfigure(0, weight=1)
        values_frame.grid_columnconfigure(0, weight=1)
        
        # Bottom frame - Regex editor
        regex_frame = ttk.LabelFrame(main_frame, text="Regex Editor", padding=5)
        regex_frame.pack(fill='x', pady=(0, 10))
        
        # Regex pattern input
        pattern_frame = ttk.Frame(regex_frame)
        pattern_frame.pack(fill='x', pady=5)
        
        ttk.Label(pattern_frame, text="Search Pattern (regex):").pack(anchor='w')
        self.pattern_var = tk.StringVar()
        self.pattern_entry = ttk.Entry(pattern_frame, textvariable=self.pattern_var, width=80)
        self.pattern_entry.pack(fill='x', pady=2)
        
        # Replacement input
        replacement_frame = ttk.Frame(regex_frame)
        replacement_frame.pack(fill='x', pady=5)
        
        ttk.Label(replacement_frame, text="Replacement:").pack(anchor='w')
        self.replacement_var = tk.StringVar()
        self.replacement_entry = ttk.Entry(replacement_frame, textvariable=self.replacement_var, width=80)
        self.replacement_entry.pack(fill='x', pady=2)
        
        # Regex options
        options_frame = ttk.Frame(regex_frame)
        options_frame.pack(fill='x', pady=5)
        
        self.ignore_case_var = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="Ignore case", variable=self.ignore_case_var).pack(side='left', padx=(0, 10))
        
        self.multiline_var = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="Multiline", variable=self.multiline_var).pack(side='left', padx=(0, 10))
        
        # Buttons for regex operations
        regex_buttons_frame = ttk.Frame(regex_frame)
        regex_buttons_frame.pack(fill='x', pady=5)
        
        ttk.Button(regex_buttons_frame, text="Preview", command=self._preview_regex).pack(side='left', padx=(0, 5))
        ttk.Button(regex_buttons_frame, text="Apply", command=self._apply_regex).pack(side='left', padx=(0, 5))
        ttk.Button(regex_buttons_frame, text="Reset", command=self._reset_values).pack(side='left', padx=(0, 5))
        
        # Examples
        examples_frame = ttk.Frame(regex_frame)
        examples_frame.pack(fill='x', pady=5)
        
        ttk.Label(examples_frame, text="Examples:", font=('Arial', 9, 'bold')).pack(anchor='w')
        examples_text = (
            "• Remove prefix: ^prefix_ → (empty)\n"
            "• Replace spaces: \\s+ → _\n"
            "• Extract numbers: .*?([0-9]+).* → \\1\n"
            "• Add suffix: (.*) → \\1_new"
        )
        ttk.Label(examples_frame, text=examples_text, font=('Arial', 8), foreground='gray').pack(anchor='w')
        
        # Bottom buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x')
        
        ttk.Button(button_frame, text="Close", command=self._close_dialog).pack(side='right', padx=5)
        
        # Bind Enter key for quick preview
        self.pattern_entry.bind('<KeyRelease>', self._on_pattern_change)
        self.replacement_entry.bind('<KeyRelease>', self._on_pattern_change)
        
    def _populate_keys(self):
        """Populate the keys listbox."""
        all_keys = self.parser.get_all_metadata_keys()
        for key in all_keys:
            self.keys_listbox.insert(tk.END, key)
            
    def _on_key_selection(self, event):
        """Handle key selection."""
        selection = self.keys_listbox.curselection()
        if not selection:
            return
            
        key = self.keys_listbox.get(selection[0])
        self.current_key = key
        self._load_values_for_key(key)
        
    def _load_values_for_key(self, key: str):
        """Load and display values for the selected key."""
        # Clear existing data
        for item in self.values_tree.get_children():
            self.values_tree.delete(item)
            
        self.value_data.clear()
        
        # Count unique values
        value_counts = {}
        for spectrum in self.parser.spectra:
            value = spectrum.get_metadata_value(key)
            if value is not None:
                value_counts[value] = value_counts.get(value, 0) + 1
                
        # Store and display data
        for value, count in sorted(value_counts.items()):
            self.value_data[value] = {'count': count, 'updated': value}
            self.values_tree.insert('', 'end', values=(value, count, value))
            
    def _on_pattern_change(self, event=None):
        """Handle pattern or replacement changes - auto preview if both fields have content."""
        if self.pattern_var.get().strip() and self.replacement_var.get().strip():
            self._preview_regex()
            
    def _preview_regex(self):
        """Preview the regex transformation."""
        if not self.current_key or not self.value_data:
            return
            
        pattern = self.pattern_var.get()
        replacement = self.replacement_var.get()
        
        if not pattern:
            messagebox.showwarning("Warning", "Please enter a search pattern.")
            return
            
        try:
            # Compile regex with options
            flags = 0
            if self.ignore_case_var.get():
                flags |= re.IGNORECASE
            if self.multiline_var.get():
                flags |= re.MULTILINE
                
            compiled_pattern = re.compile(pattern, flags)
            
            # Update the tree with preview
            for item in self.values_tree.get_children():
                values = self.values_tree.item(item, 'values')
                original_value = values[0]
                
                try:
                    updated_value = compiled_pattern.sub(replacement, original_value)
                    self.value_data[original_value]['updated'] = updated_value
                    
                    # Update tree display
                    self.values_tree.item(item, values=(original_value, values[1], updated_value))
                    
                except Exception as e:
                    # If replacement fails for this value, keep original
                    self.value_data[original_value]['updated'] = original_value
                    self.values_tree.item(item, values=(original_value, values[1], f"ERROR: {str(e)}"))
                    
        except re.error as e:
            messagebox.showerror("Regex Error", f"Invalid regular expression: {str(e)}")
            
    def _apply_regex(self):
        """Apply the regex transformation to all spectra."""
        if not self.current_key or not self.value_data:
            messagebox.showwarning("Warning", "Please select a key and preview changes first.")
            return
            
        # Confirm before applying
        num_affected = sum(data['count'] for data in self.value_data.values() 
                          if data['updated'] != list(self.value_data.keys())[list(self.value_data.values()).index(data)])
        
        if num_affected == 0:
            messagebox.showinfo("Info", "No changes to apply.")
            return
            
        if not messagebox.askyesno("Confirm Changes", 
                                  f"Apply regex transformation to {num_affected} values in key '{self.current_key}'?"):
            return
            
        # Apply changes to all spectra
        changes_made = False
        for spectrum in self.parser.spectra:
            current_value = spectrum.get_metadata_value(self.current_key)
            if current_value in self.value_data:
                new_value = self.value_data[current_value]['updated']
                if new_value != current_value and not new_value.startswith("ERROR:"):
                    spectrum.metadata[self.current_key] = new_value
                    changes_made = True
                    
        if changes_made:
            self.changes_made = True
            messagebox.showinfo("Success", "Regex transformation applied successfully!")
            
            # Reload values to show the changes
            self._load_values_for_key(self.current_key)
            
    def _reset_values(self):
        """Reset all values to original."""
        if not self.current_key:
            return
            
        # Reset the updated values to original
        for original_value in self.value_data:
            self.value_data[original_value]['updated'] = original_value
            
        # Update tree display
        for item in self.values_tree.get_children():
            values = self.values_tree.item(item, 'values')
            original_value = values[0]
            count = values[1]
            self.values_tree.item(item, values=(original_value, count, original_value))
            
    def _close_dialog(self):
        """Close the dialog."""
        self.dialog.destroy()


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
            
        # Calculate global m/z limits for all selected spectra
        global_mz_min = float('inf')
        global_mz_max = float('-inf')
        
        for spectrum in selected_spectra:
            if spectrum.ions.size > 0:
                mz_values = spectrum.ions[:, 0]
                global_mz_min = min(global_mz_min, mz_values.min())
                global_mz_max = max(global_mz_max, mz_values.max())
        
        # Add some padding to the limits
        if global_mz_min != float('inf') and global_mz_max != float('-inf'):
            mz_range = global_mz_max - global_mz_min
            padding = mz_range * 0.02  # 2% padding
            global_mz_min -= padding
            global_mz_max += padding
        else:
            # Fallback if no valid data
            global_mz_min, global_mz_max = 0, 1000
            
        # Create subplots
        n_spectra = len(selected_spectra)
        if n_spectra == 1:
            ax = self.figure.add_subplot(111)
            self._plot_single_spectrum(ax, selected_spectra[0], (global_mz_min, global_mz_max))
        else:
            for i, spectrum in enumerate(selected_spectra):
                ax = self.figure.add_subplot(n_spectra, 1, i+1)
                self._plot_single_spectrum(ax, spectrum, (global_mz_min, global_mz_max))
                
        self.figure.tight_layout()
        self.canvas.draw()
        
    def _plot_single_spectrum(self, ax, spectrum, mz_limits=None):
        """Plot a single spectrum as a stick chart."""
        if spectrum.ions.size == 0:
            ax.text(0.5, 0.5, 'No ion data', ha='center', va='center', transform=ax.transAxes)
            if mz_limits:
                ax.set_xlim(mz_limits)
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
        if mz_limits:
            ax.set_xlim(mz_limits)
        elif len(mz_values) > 0:
            ax.set_xlim(mz_values.min() * 0.95, mz_values.max() * 1.05)
            
        # Set y limits
        if len(intensity_values) > 0:
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
