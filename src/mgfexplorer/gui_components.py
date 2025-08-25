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
import threading
import time
from .mgf_parser import MGFParser, Spectrum

# RDKit imports for SMILES plotting
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


class SpectrumTreeView(ttk.Frame):
    """Tree view component for displaying spectra list with tag-based grouping."""

    def __init__(self, parent, on_selection_changed=None):
        super().__init__(parent)
        self.on_selection_changed = on_selection_changed
        self.parser: Optional[MGFParser] = None
        self.selected_grouping_tags: List[str] = []
        self.naming_scheme: str = "Numbered"  # Default naming scheme
        self.filter_text: str = ""  # Current filter text
        self.filter_job: Optional[str] = None  # Job ID for delayed filtering

        self._create_widgets()

    def _create_widgets(self):
        """Create the tree view widgets."""
        # Tag selection frame
        tag_frame = ttk.LabelFrame(self, text="Grouping Tags", padding=5)
        tag_frame.pack(fill="x", padx=5, pady=5)

        # Tag entry with right-click context menu
        self.tag_var = tk.StringVar()
        self.tag_entry = ttk.Entry(tag_frame, textvariable=self.tag_var, width=40)
        self.tag_entry.pack(fill="x", pady=5)
        self.tag_entry.bind("<KeyRelease>", self._on_tag_entry_change)
        self.tag_entry.bind("<Button-3>", self._show_context_menu)  # Right-click
        self.tag_entry.bind("<Key>", self._on_key_press)  # For intelligent deletion
        self.tag_entry.bind(
            "<Double-Button-1>", self._on_double_click_entry
        )  # Double-click to select field

        # Create context menu for metadata fields
        self.context_menu = tk.Menu(self, tearoff=0)

        # Track previous text for deletion detection
        self.previous_text = ""

        # Filter frame
        filter_frame = ttk.LabelFrame(self, text="Filter", padding=5)
        filter_frame.pack(fill="x", padx=5, pady=5)

        # Filter entry with responsive filtering
        self.filter_var = tk.StringVar()
        self.filter_entry = ttk.Entry(
            filter_frame, textvariable=self.filter_var, width=40
        )
        self.filter_entry.pack(fill="x", pady=5)
        self.filter_entry.bind("<KeyRelease>", self._on_filter_entry_change)

        # Filter help text
        help_text = "Default: search all fields\n$$ key: value (exact)\n$$$ key: regex"
        filter_help = ttk.Label(filter_frame, text=help_text, font=("TkDefaultFont", 8))
        filter_help.pack(fill="x", pady=(0, 5))

        # Tree view frame
        tree_frame = ttk.LabelFrame(self, text="Spectra", padding=5)
        tree_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Tree view with scrollbar
        tree_container = ttk.Frame(tree_frame)
        tree_container.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(tree_container, selectmode="extended")
        tree_scrollbar = ttk.Scrollbar(
            tree_container, orient="vertical", command=self.tree.yview
        )
        self.tree.configure(yscrollcommand=tree_scrollbar.set)

        self.tree.pack(side="left", fill="both", expand=True)
        tree_scrollbar.pack(side="right", fill="y")

        # Bind selection event
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_selection)

    def load_data(self, parser: MGFParser):
        """Load spectra data into the tree view."""
        self.parser = parser
        # Initialize previous text tracking
        self.previous_text = self.tag_var.get()
        # Reset filter when loading new data
        self.filter_var.set("")
        self.filter_text = ""
        if self.filter_job:
            self.after_cancel(self.filter_job)
            self.filter_job = None
        self._populate_tree()

    def _show_context_menu(self, event):
        """Show context menu with available metadata fields."""
        if not self.parser:
            return

        # Clear existing menu items
        self.context_menu.delete(0, "end")

        # Get all available metadata keys
        all_keys = self.parser.get_all_metadata_keys()

        if not all_keys:
            self.context_menu.add_command(
                label="No metadata fields available", state="disabled"
            )
        else:
            # Add each key as a menu item
            for key in sorted(all_keys):
                self.context_menu.add_command(
                    label=key, command=lambda k=key: self._add_tag_from_menu(k)
                )

        # Show the menu at the cursor position
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()

    def _add_tag_from_menu(self, tag):
        """Add a tag from the context menu to the text field."""
        current_text = self.tag_var.get().strip()

        if current_text:
            # Add comma separator if there's existing text
            new_text = f"{current_text}, {tag}"
        else:
            new_text = tag

        self.tag_var.set(new_text)

        # Apply grouping automatically
        self._apply_grouping_auto()

    def _on_tag_entry_change(self, event=None):
        """Handle changes in the tag entry field with automatic grouping and cleanup."""
        current_text = self.tag_var.get()

        # Clean up spacing: remove multiple spaces after commas
        cleaned_text = self._clean_spacing(current_text)
        if cleaned_text != current_text:
            cursor_pos = self.tag_entry.index(tk.INSERT)
            self.tag_var.set(cleaned_text)
            # Try to maintain cursor position, but adjust if needed
            new_cursor_pos = min(cursor_pos, len(cleaned_text))
            self.tag_entry.icursor(new_cursor_pos)
            current_text = cleaned_text

        # Apply grouping automatically when text changes
        self._apply_grouping_auto()

        # Update previous text for next comparison
        self.previous_text = current_text

    def _on_filter_entry_change(self, event=None):
        """Handle changes in the filter entry field with delayed filtering."""
        # Cancel any pending filter job
        if self.filter_job:
            self.after_cancel(self.filter_job)

        # Schedule a new filter job after 2 seconds (2000ms)
        self.filter_job = self.after(2000, self._apply_filter)

    def _apply_filter(self):
        """Apply the current filter to the tree view."""
        self.filter_job = None
        self.filter_text = self.filter_var.get().strip().lower()
        self._populate_tree()

    def _spectrum_matches_filter(self, spectrum):
        """Check if a spectrum matches the current filter."""
        if not self.filter_text:
            return True

        # Check for special filtering syntax
        if self.filter_text.startswith("$$$"):
            # Regex search in specific key: "$$$ key: regex"
            return self._filter_by_key_regex(spectrum, self.filter_text[3:].strip())
        elif self.filter_text.startswith("$$"):
            # Exact submatch in specific key: "$$ key: value"
            return self._filter_by_key_exact(spectrum, self.filter_text[2:].strip())
        else:
            # Default: search in all metadata values
            for key, value in spectrum.metadata.items():
                if value and self.filter_text in str(value).lower():
                    return True
            return False

    def _filter_by_key_exact(self, spectrum, filter_text):
        """Filter by exact submatch in a specific key field."""
        if ":" not in filter_text:
            return False

        key_part, value_part = filter_text.split(":", 1)
        key = key_part.strip()
        search_value = value_part.strip().lower()

        if key in spectrum.metadata:
            metadata_value = spectrum.metadata[key]
            if metadata_value and search_value in str(metadata_value).lower():
                return True
        return False

    def _filter_by_key_regex(self, spectrum, filter_text):
        """Filter by regex match in a specific key field."""
        import re

        if ":" not in filter_text:
            return False

        key_part, regex_part = filter_text.split(":", 1)
        key = key_part.strip()
        regex_pattern = regex_part.strip()

        if key in spectrum.metadata:
            metadata_value = spectrum.metadata[key]
            if metadata_value:
                try:
                    # Case-insensitive regex search
                    if re.search(regex_pattern, str(metadata_value), re.IGNORECASE):
                        return True
                except re.error:
                    # Invalid regex pattern - fall back to literal search
                    if regex_pattern.lower() in str(metadata_value).lower():
                        return True
        return False

    def _clean_spacing(self, text):
        """Clean up spacing in the text - ensure single space after commas."""
        import re

        # Replace comma followed by multiple spaces with comma followed by single space
        cleaned = re.sub(r",\s+", ", ", text)
        # Remove leading/trailing spaces from the entire string
        cleaned = cleaned.strip()
        return cleaned

    def _on_key_press(self, event):
        """Handle key presses for intelligent field deletion."""
        if event.keysym in ("BackSpace", "Delete"):
            return self._handle_intelligent_field_deletion(event)
        return None

    def _handle_intelligent_field_deletion(self, event):
        """Handle backspace/delete to remove entire fields when cursor is in a field name."""
        current_text = self.tag_var.get()
        cursor_pos = self.tag_entry.index(tk.INSERT)

        # Find which field the cursor is currently in
        field_info = self._get_field_at_cursor(current_text, cursor_pos)

        if field_info is None:
            # Cursor not in a field, allow normal behavior
            return None

        field_index, field_start, field_end, field_text = field_info

        # Check if cursor is in the field name (not in separator)
        if field_start <= cursor_pos <= field_end and field_text.strip():
            # Cursor is in a field name, remove the entire field
            self._remove_field_at_index(field_index)
            return "break"  # Prevent default behavior

        # Allow normal behavior for separators or empty areas
        return None

    def _get_field_at_cursor(self, text, cursor_pos):
        """Get information about the field at cursor position."""
        if not text:
            return None

        # Split by commas but keep track of positions
        fields = []
        current_pos = 0
        parts = text.split(",")

        for i, part in enumerate(parts):
            field_start = current_pos
            field_end = current_pos + len(part)
            field_text = part.strip()

            fields.append((i, field_start, field_end, field_text))
            current_pos = field_end + 1  # +1 for comma

        # Find which field contains the cursor
        for field_index, field_start, field_end, field_text in fields:
            if field_start <= cursor_pos <= field_end:
                return (field_index, field_start, field_end, field_text)

        return None

    def _remove_field_at_index(self, field_index):
        """Remove a field at the specified index and update the text."""
        current_text = self.tag_var.get()
        parts = [part.strip() for part in current_text.split(",")]

        if 0 <= field_index < len(parts):
            # Remove the field
            parts.pop(field_index)

            # Rebuild text
            new_text = ", ".join(parts)
            self.tag_var.set(new_text)

            # Position cursor appropriately
            if parts:
                if field_index == 0:
                    # Removed first field, position at start
                    self.tag_entry.icursor(0)
                elif field_index >= len(parts):
                    # Removed last field, position at end
                    self.tag_entry.icursor(len(new_text))
                else:
                    # Position at start of next field
                    pos = len(", ".join(parts[:field_index])) + (
                        2 if field_index > 0 else 0
                    )
                    self.tag_entry.icursor(min(pos, len(new_text)))
            else:
                # No fields left, position at start
                self.tag_entry.icursor(0)

            # Apply grouping with updated text
            self._apply_grouping_auto()

    def _on_double_click_entry(self, event):
        """Handle double-click to select entire field."""
        cursor_pos = self.tag_entry.index(tk.INSERT)
        current_text = self.tag_var.get()

        # Use the same field detection logic
        field_info = self._get_field_at_cursor(current_text, cursor_pos)

        if field_info:
            field_index, field_start, field_end, field_text = field_info

            # Skip leading whitespace
            while field_start < field_end and current_text[field_start] == " ":
                field_start += 1
            # Skip trailing whitespace
            while field_end > field_start and current_text[field_end - 1] == " ":
                field_end -= 1

            # Select the field text
            if field_start < field_end:
                self.tag_entry.selection_range(field_start, field_end)
                self.tag_entry.icursor(field_end)

    def _apply_grouping_auto(self):
        """Apply grouping automatically without user action."""
        tag_text = self.tag_var.get().strip()
        if tag_text:
            # Parse comma-separated tags
            self.selected_grouping_tags = [
                tag.strip() for tag in tag_text.split(",") if tag.strip()
            ]
        else:
            self.selected_grouping_tags = []

        self._populate_tree()

    def _apply_grouping(self, event=None):
        """Apply the grouping based on entered tags (legacy method)."""
        self._apply_grouping_auto()

    def _populate_tree(self):
        """Populate the tree view with spectra using hierarchical grouping."""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)

        if not self.parser:
            return

        # Configure tree display
        self.tree["show"] = "tree"

        if not self.selected_grouping_tags:
            # No grouping - show flat list
            self._populate_flat_list()
        else:
            # Hierarchical grouping
            self._populate_hierarchical_tree()

    def _populate_flat_list(self):
        """Populate tree with flat list of spectra."""
        for spectrum in self.parser.spectra:
            # Apply filter if specified
            if not self._spectrum_matches_filter(spectrum):
                continue

            display_name = self._get_spectrum_display_name(spectrum)
            item_id = self.tree.insert(
                "", "end", text=display_name, tags=("spectrum", spectrum.spectrum_id)
            )

    def _populate_hierarchical_tree(self):
        """Populate tree with hierarchical grouping."""
        # Build hierarchy
        hierarchy = {}

        for spectrum in self.parser.spectra:
            # Apply filter if specified
            if not self._spectrum_matches_filter(spectrum):
                continue

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

        # Populate tree from hierarchy
        self._add_hierarchy_to_tree("", hierarchy, 0)

    def _add_hierarchy_to_tree(self, parent_id: str, hierarchy: dict, level: int):
        """Recursively add hierarchy to tree."""
        for key, data in hierarchy.items():
            # Create group node
            display_text = key if key != "_ungrouped_" else "Ungrouped"
            group_id = self.tree.insert(
                parent_id, "end", text=display_text, tags=("group", level)
            )

            # Add spectra in this group
            for spectrum in data["spectra"]:
                display_name = self._get_spectrum_display_name(spectrum)
                spectrum_id = self.tree.insert(
                    group_id,
                    "end",
                    text=display_name,
                    tags=("spectrum", spectrum.spectrum_id),
                )

            # Recursively add children
            if data["children"]:
                self._add_hierarchy_to_tree(group_id, data["children"], level + 1)

            # Expand group if it has items
            if data["spectra"] or data["children"]:
                self.tree.item(group_id, open=True)

    def _on_tree_selection(self, event):
        """Handle tree selection changes."""
        selected_items = self.tree.selection()
        selected_spectrum_ids = []

        for item in selected_items:
            tags = self.tree.item(item, "tags")
            if not tags:
                continue

            if tags[0] == "spectrum":
                # Direct spectrum selection - spectrum ID could be string or int
                spectrum_id = tags[1]
                # Try to convert to int, but keep as string if it fails (for prefixed IDs)
                try:
                    spectrum_id = int(spectrum_id)
                except ValueError:
                    pass  # Keep as string
                selected_spectrum_ids.append(spectrum_id)
            elif tags[0] == "group":
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
                tags = self.tree.item(child, "tags")
                if tags and tags[0] == "spectrum":
                    spectrum_id = tags[1]
                    # Try to convert to int, but keep as string if it fails (for prefixed IDs)
                    try:
                        spectrum_id = int(spectrum_id)
                    except ValueError:
                        pass  # Keep as string
                    spectrum_ids.append(spectrum_id)
                else:
                    # Recursive for nested groups
                    collect_spectra(child)

        collect_spectra(group_item)
        return spectrum_ids

    def get_selected_spectrum_ids(self) -> List:
        """Get currently selected spectrum IDs."""
        selected_items = self.tree.selection()
        selected_ids = []

        for item in selected_items:
            tags = self.tree.item(item, "tags")
            if not tags:
                continue

            if tags[0] == "spectrum":
                spectrum_id = tags[1]
                # Try to convert to int, but keep as string if it fails (for prefixed IDs)
                try:
                    spectrum_id = int(spectrum_id)
                except ValueError:
                    pass  # Keep as string
                selected_ids.append(spectrum_id)
            elif tags[0] == "group":
                group_spectra = self._get_spectra_in_group(item)
                selected_ids.extend(group_spectra)

        return sorted(list(set(selected_ids)))

    def select_spectra_by_ids(self, spectrum_ids):
        """Select spectra by their IDs with performance optimizations."""
        if not spectrum_ids:
            # Clear selection
            for item in self.tree.selection():
                self.tree.selection_remove(item)
            return

        # Cancel any ongoing async selection
        if hasattr(self, "_async_selection_items"):
            self._cleanup_async_selection()

        # For very large selections, show a progress indicator and use async selection
        if len(spectrum_ids) > 100:
            print(f"Debug: Using async selection for {len(spectrum_ids)} spectra")
            self._select_spectra_async(spectrum_ids)
        else:
            print(f"Debug: Using standard selection for {len(spectrum_ids)} spectra")
            found_items = self._select_spectra_by_ids_standard(spectrum_ids)
            print(
                f"Debug: Standard selection found {len(found_items) if found_items else 0} items"
            )

    def _select_spectra_async(self, spectrum_ids):
        """Asynchronously select large numbers of spectra."""
        # Show progress message
        self.after_idle(lambda: self._show_selection_progress(len(spectrum_ids)))

        # Schedule the selection to collect all items first, then select
        self._async_selection_ids = set(spectrum_ids)
        self._async_selection_items = []

        # Use a more robust approach: collect all items first
        self.after(10, self._collect_all_spectrum_items_async)

    def _collect_all_spectrum_items_async(self):
        """Collect all spectrum items from the tree structure."""
        all_spectrum_items = []

        def collect_items(parent=""):
            for item in self.tree.get_children(parent):
                tags = self.tree.item(item, "tags")
                if tags and tags[0] == "spectrum":
                    spectrum_id = tags[1]  # Don't convert to int - keep original type
                    if spectrum_id in self._async_selection_ids:
                        all_spectrum_items.append(item)
                elif tags and tags[0] == "group":
                    # Recursively check group children
                    collect_items(item)
                else:
                    # Handle items without proper tags
                    collect_items(item)

        # Clear current selection first
        self.tree.selection_remove(*self.tree.selection())

        # Collect all matching items
        collect_items()

        # Store the collected items
        self._async_selection_items = all_spectrum_items
        self._selection_chunk_index = 0

        # Start applying selection in chunks
        self.after(10, self._apply_async_selection)

    def _show_selection_progress(self, count):
        """Show selection progress in the tree."""
        # Temporarily clear selection and show message
        self.tree.selection_remove(*self.tree.selection())

        # You could add a temporary item here showing progress
        # For now, we'll just ensure the UI updates
        self.update_idletasks()

    def _apply_async_selection(self):
        """Apply the collected selection asynchronously."""
        if not hasattr(self, "_async_selection_items"):
            return

        # Apply selection in chunks
        chunk_size = 50
        items = self._async_selection_items

        if not hasattr(self, "_selection_chunk_index"):
            self._selection_chunk_index = 0

        # Process a chunk
        start_idx = self._selection_chunk_index
        end_idx = min(start_idx + chunk_size, len(items))

        for item in items[start_idx:end_idx]:
            self.tree.selection_add(item)

        self._selection_chunk_index = end_idx

        # Check if we're done
        if end_idx >= len(items):
            # Selection complete
            self._cleanup_async_selection()

            # Make first item visible
            if items:
                self.tree.see(items[0])
        else:
            # Continue with next chunk
            self.after(10, self._apply_async_selection)

    def _cleanup_async_selection(self):
        """Clean up async selection variables."""
        for attr in [
            "_async_selection_ids",
            "_async_selection_items",
            "_selection_chunk_index",
        ]:
            if hasattr(self, attr):
                delattr(self, attr)

    def _select_spectra_by_ids_standard(self, spectrum_ids):
        """Standard selection method for moderate number of spectra."""
        # Clear current selection
        for item in self.tree.selection():
            self.tree.selection_remove(item)

        # Convert to set for faster lookup
        target_ids = set(spectrum_ids)
        found_items = []

        # Select matching spectra
        def select_in_tree(parent=""):
            for item in self.tree.get_children(parent):
                tags = self.tree.item(item, "tags")
                if tags and tags[0] == "spectrum":
                    spectrum_id = tags[1]  # Don't convert to int - keep original type
                    if spectrum_id in target_ids:
                        self.tree.selection_add(item)
                        found_items.append(item)
                        # Only ensure visibility for first few items to avoid performance issues
                        if len(found_items) <= 10:
                            self.tree.see(item)
                elif tags and tags[0] == "group":
                    # Recursively check children of group items
                    select_in_tree(item)
                else:
                    # Handle items without proper tags - recursively check children anyway
                    select_in_tree(item)

        select_in_tree()

        # Debug information - can be removed later
        if len(found_items) != len(spectrum_ids):
            print(
                f"Selection mismatch: requested {len(spectrum_ids)}, found {len(found_items)}"
            )
            print(f"Tree has grouping: {bool(self.selected_grouping_tags)}")
            print(f"Grouping tags: {self.selected_grouping_tags}")

        return found_items

    def _select_spectra_by_ids_batch(self, spectrum_ids):
        """Optimized selection method for large number of spectra."""
        # Clear current selection
        self.tree.selection_remove(*self.tree.selection())

        # Convert to set for faster lookup
        target_ids = set(spectrum_ids)
        items_to_select = []

        # Collect all items to select first
        def collect_items(parent=""):
            for item in self.tree.get_children(parent):
                tags = self.tree.item(item, "tags")
                if tags and tags[0] == "spectrum":
                    spectrum_id = tags[1]  # Don't convert to int - keep original type
                    if spectrum_id in target_ids:
                        items_to_select.append(item)
                else:
                    # Recursively check children
                    collect_items(item)

        collect_items()

        # Batch select items to reduce UI updates
        if items_to_select:
            # Disable updates during batch operation
            self.tree.configure(state="disabled")
            try:
                # Select in chunks to avoid overwhelming the UI
                chunk_size = 100
                for i in range(0, len(items_to_select), chunk_size):
                    chunk = items_to_select[i : i + chunk_size]
                    for item in chunk:
                        self.tree.selection_add(item)

                    # Update UI periodically
                    if i % (chunk_size * 5) == 0:
                        self.update_idletasks()

                # Make the first selected item visible
                if items_to_select:
                    self.tree.see(items_to_select[0])

            finally:
                self.tree.configure(state="normal")

    def set_naming_scheme(self, scheme: str):
        """Set the naming scheme for spectrum display."""
        self.naming_scheme = scheme

    def _get_spectrum_display_name(self, spectrum) -> str:
        """Get the display name for a spectrum based on the current naming scheme."""
        if self.naming_scheme == "Numbered":
            return f"S {spectrum.spectrum_id}"
        else:
            # Use the specified metadata field
            value = spectrum.get_metadata_value(self.naming_scheme)
            if value:
                return f"{spectrum.spectrum_id}: {self.naming_scheme}={value}"
            else:
                return f"{spectrum.spectrum_id}: {self.naming_scheme}=<missing>"

    def get_current_grouping_structure(self):
        """Get the current grouping structure for export functionality."""
        if not self.parser or not self.selected_grouping_tags:
            return None

        # Build the same hierarchy structure used in _populate_hierarchical_tree
        hierarchy = {}

        for spectrum in self.parser.spectra:
            # Apply filter if specified
            if not self._spectrum_matches_filter(spectrum):
                continue

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

        return hierarchy

    def has_grouping(self):
        """Check if there is currently active grouping."""
        return bool(self.selected_grouping_tags)

    def get_grouping_tags(self):
        """Get the current grouping tags."""
        return self.selected_grouping_tags.copy()


class MetadataEditor(ttk.Frame):
    """Component for viewing and editing metadata."""

    def __init__(self, parent, on_metadata_changed=None):
        super().__init__(parent)
        self.parent = parent  # Store parent reference
        self.on_metadata_changed = on_metadata_changed
        self.parser: Optional[MGFParser] = None
        self.selected_spectrum_ids: List[int] = []
        self.edit_var = tk.StringVar()
        self.edit_entry = None
        self.editing_item = None
        self.editing_column = None
        self._pending_selection: Optional[List[int]] = None

        self._create_widgets()

    def _create_widgets(self):
        """Create the metadata editor widgets."""
        # Header
        header_frame = ttk.Frame(self)
        header_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(
            header_frame, text="Metadata Editor", font=("Arial", 12, "bold")
        ).pack()

        # Main content area with horizontal split
        content_frame = ttk.Frame(self)
        content_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Create horizontal paned window
        paned_window = ttk.PanedWindow(content_frame, orient="horizontal")
        paned_window.pack(fill="both", expand=True)

        # Left side: Metadata table
        metadata_frame = ttk.LabelFrame(paned_window, text="Metadata", padding=5)
        paned_window.add(metadata_frame, weight=4)

        # Create treeview for metadata
        columns = ("Key", "Value", "Unique Values")
        self.metadata_tree = ttk.Treeview(
            metadata_frame, columns=columns, show="headings", height=10
        )

        for col in columns:
            self.metadata_tree.heading(col, text=col)
            self.metadata_tree.column(col, width=150)

        # Scrollbars for metadata table
        v_scrollbar = ttk.Scrollbar(
            metadata_frame, orient="vertical", command=self.metadata_tree.yview
        )
        h_scrollbar = ttk.Scrollbar(
            metadata_frame, orient="horizontal", command=self.metadata_tree.xview
        )

        self.metadata_tree.configure(
            yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set
        )

        self.metadata_tree.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        metadata_frame.grid_rowconfigure(0, weight=1)
        metadata_frame.grid_columnconfigure(0, weight=1)

        # Right side: SMILES structure plot
        smiles_frame = ttk.LabelFrame(paned_window, text="SMILES Structure", padding=5)
        paned_window.add(smiles_frame, weight=1)

        # Create SMILES plot area
        self._create_smiles_plot(smiles_frame)

        # Instructions
        instructions_frame = ttk.Frame(self)
        instructions_frame.pack(fill="x", padx=5, pady=5)

        instructions_text = (
            "Double-click on Key or Value cells to edit. Press Enter to save, Escape to cancel.\n"
            "Right-click on Value cells to select all spectra with that value (including empty values).\n"
            "SMILES structures are displayed when available and consistent across selected spectra."
        )
        ttk.Label(
            instructions_frame,
            text=instructions_text,
            font=("Arial", 9),
            foreground="gray",
        ).pack()

        # Bind selection and editing events
        self.metadata_tree.bind("<<TreeviewSelect>>", self._on_metadata_selection)
        self.metadata_tree.bind("<Double-1>", self._on_double_click)
        self.metadata_tree.bind("<Button-1>", self._on_single_click)
        self.metadata_tree.bind("<Button-3>", self._on_right_click)  # Right-click

    def _create_smiles_plot(self, parent_frame):
        """Create the SMILES structure plot area."""
        # Create matplotlib figure for SMILES display
        self.smiles_fig = Figure(figsize=(4, 4), dpi=100)
        self.smiles_canvas = FigureCanvasTkAgg(self.smiles_fig, parent_frame)
        self.smiles_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Initially show empty plot with message
        self._show_smiles_message("No SMILES data available")

    def _show_smiles_message(self, message):
        """Show a text message in the SMILES plot area."""
        self.smiles_fig.clear()
        ax = self.smiles_fig.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            fontsize=10,
            wrap=True,
            transform=ax.transAxes,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        self.smiles_canvas.draw()

    def _get_smiles_from_spectra(self, spectrum_ids):
        """Extract SMILES codes from selected spectra."""
        if not self.parser or not spectrum_ids:
            return None, "No spectra selected"

        # Common SMILES key names to check
        smiles_keys = ["smiles", "SMILES", "Smiles", "smiles_code", "SMILES_CODE"]

        smiles_values = []
        found_key = None

        for spectrum_id in spectrum_ids:
            spectrum = next(
                (s for s in self.parser.spectra if s.spectrum_id == spectrum_id), None
            )
            if not spectrum:
                continue

            # Find SMILES key in metadata
            spectrum_smiles = None
            for key in smiles_keys:
                value = spectrum.get_metadata_value(key)
                if value and value.strip():
                    spectrum_smiles = value.strip()
                    found_key = key
                    break

            if spectrum_smiles:
                smiles_values.append(spectrum_smiles)
            else:
                smiles_values.append(None)

        if not any(smiles_values):
            return None, "No SMILES data found in selected spectra"

        # Check if all non-None SMILES are the same
        non_none_smiles = [s for s in smiles_values if s is not None]
        if len(set(non_none_smiles)) > 1:
            return None, f"Different SMILES codes found:\n" + "\n".join(
                set(non_none_smiles)
            ) + ("\n..." if len(set(non_none_smiles)) > 3 else "")

        return non_none_smiles[0], None

    def _plot_smiles(self, smiles_code):
        """Plot a SMILES structure using RDKit."""
        if not RDKIT_AVAILABLE:
            self._show_smiles_message("RDKit not available for SMILES plotting")
            return

        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles_code)
            if mol is None:
                self._show_smiles_message(f"Invalid SMILES code:\n{smiles_code}")
                return

            # Generate 2D coordinates
            from rdkit.Chem import rdDepictor

            rdDepictor.Compute2DCoords(mol)

            # Create drawer
            drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()

            # Get image data
            img_data = drawer.GetDrawingText()

            # Convert to matplotlib image
            from PIL import Image
            import io

            img = Image.open(io.BytesIO(img_data))

            # Clear figure and display image
            self.smiles_fig.clear()
            ax = self.smiles_fig.add_subplot(111)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(
                f"SMILES: {smiles_code[:30]}{'...' if len(smiles_code) > 30 else ''}",
                fontsize=8,
                pad=10,
            )

            self.smiles_canvas.draw()

        except Exception as e:
            self._show_smiles_message(f"Error plotting SMILES:\n{str(e)}")

    def _update_smiles_plot(self):
        """Update the SMILES plot based on current selection."""
        smiles_code, error_msg = self._get_smiles_from_spectra(
            self.selected_spectrum_ids
        )

        if error_msg:
            self._show_smiles_message(error_msg)
        elif smiles_code:
            self._plot_smiles(smiles_code)

    def load_data(self, parser: MGFParser, selected_spectrum_ids: List[int]):
        """Load metadata for selected spectra."""
        self.parser = parser
        self.selected_spectrum_ids = selected_spectrum_ids
        self._populate_metadata()
        self._update_smiles_plot()

    def _populate_metadata(self):
        """Populate the metadata table."""
        # Clear existing items
        for item in self.metadata_tree.get_children():
            self.metadata_tree.delete(item)

        if not self.parser or not self.selected_spectrum_ids:
            return

        # Get all metadata keys
        all_keys = self.parser.get_all_metadata_keys()

        # Get selected spectra objects
        selected_spectra = [
            s
            for s in self.parser.spectra
            if s.spectrum_id in self.selected_spectrum_ids
        ]

        for key in all_keys:
            # Get value from first selected spectrum, treating None as empty string
            value = ""
            if self.selected_spectrum_ids:
                first_spectrum = next(
                    (
                        s
                        for s in self.parser.spectra
                        if s.spectrum_id == self.selected_spectrum_ids[0]
                    ),
                    None,
                )
                if first_spectrum:
                    raw_value = first_spectrum.get_metadata_value(key)
                    value = raw_value if raw_value is not None else ""

            # Get unique values for this key from only the selected spectra
            unique_values = self._get_unique_values_for_selected_spectra(
                key, selected_spectra
            )

            # Only show unique values if there are multiple different values
            if len(unique_values) > 1:
                # Format unique values for display
                unique_str = "; ".join(
                    [f'"{v}"' if v else "<empty>" for v in unique_values[:5]]
                )
                if len(unique_values) > 5:
                    unique_str += f" ... ({len(unique_values)} total)"
            else:
                # Only one unique value, don't show the unique values column
                unique_str = ""

            self.metadata_tree.insert("", "end", values=(key, value, unique_str))

    def _get_unique_values_for_selected_spectra(
        self, key: str, selected_spectra: List[Spectrum]
    ) -> List[str]:
        """Get unique values for a key from only the selected spectra."""
        values = set()
        has_missing = False

        for spectrum in selected_spectra:
            if key in spectrum.metadata:
                values.add(spectrum.metadata[key])
            else:
                has_missing = True

        # Include empty string if any selected spectrum is missing this key
        if has_missing:
            values.add("")

        return sorted(list(values))

    def _on_metadata_selection(self, event):
        """Handle metadata selection."""
        # This can be used for future functionality if needed
        pass

    def _on_single_click(self, event):
        """Handle single click to close any open editor."""
        self._close_editor()

    def _on_right_click(self, event):
        """Handle right-click to show context menu."""
        # Identify the item and column
        item = self.metadata_tree.identify("item", event.x, event.y)
        column = self.metadata_tree.identify("column", event.x, event.y)

        if not item or column not in ("#1", "#2"):  # Only for Key or Value columns
            return

        values = self.metadata_tree.item(item, "values")
        if not values:
            return

        key = values[0]
        value = values[1]

        # Create context menu
        context_menu = tk.Menu(self, tearoff=0)

        # Always show selection option for value column, including empty values
        if column == "#2":  # Value column
            display_value = value if value else "<empty>"
            context_menu.add_command(
                label=f"Select all with '{key}' = '{display_value}'",
                command=lambda: self._select_spectra_by_value(key, value),
            )

        # Show menu if it has items
        if context_menu.index("end") is not None:
            try:
                context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                context_menu.grab_release()

    def _select_spectra_by_value(self, key: str, value: str):
        """Select all spectra that have the specified key-value pair."""
        if not self.parser:
            return

        total_spectra = len(self.parser.spectra)

        # For very large datasets, warn user and ask for confirmation
        if total_spectra > 5000:
            result = messagebox.askyesno(
                "Large Dataset Warning",
                f"This dataset contains {total_spectra} spectra.\n"
                f"Searching and selecting by value may take time and impact performance.\n\n"
                f"Do you want to proceed?",
            )
            if not result:
                return

        # Show progress for large datasets
        if total_spectra > 1000:
            progress_dialog = self._create_progress_dialog("Searching spectra...")
            self.update_idletasks()
        else:
            progress_dialog = None

        try:
            # Find all spectra with this key-value pair
            matching_spectrum_ids = []
            start_time = time.time()

            for i, spectrum in enumerate(self.parser.spectra):
                # Check for timeout (max 30 seconds)
                if time.time() - start_time > 30:
                    if progress_dialog:
                        progress_dialog.destroy()
                    messagebox.showwarning(
                        "Search Timeout",
                        f"Search timed out after checking {i} spectra.\n"
                        f"Found {len(matching_spectrum_ids)} matches so far.\n"
                        f"Consider using a smaller dataset or more specific search criteria.",
                    )
                    if matching_spectrum_ids:
                        # Proceed with partial results
                        break
                    else:
                        return

                # Check if spectrum matches the search criteria
                # Treat None (missing key) as empty string for comparison
                spectrum_value = spectrum.get_metadata_value(key)
                if spectrum_value is None:
                    spectrum_value = ""

                if spectrum_value == value:
                    matching_spectrum_ids.append(spectrum.spectrum_id)

                # Update progress for large datasets
                if progress_dialog and i % 100 == 0:
                    progress = (i / total_spectra) * 100
                    progress_dialog.update_progress(
                        progress,
                        f"Checked {i}/{total_spectra} spectra\nFound {len(matching_spectrum_ids)} matches",
                    )

            if progress_dialog:
                progress_dialog.destroy()

            if matching_spectrum_ids:
                # Check if selection is very large
                if len(matching_spectrum_ids) > 1000:
                    result = messagebox.askyesno(
                        "Very Large Selection",
                        f"Found {len(matching_spectrum_ids)} matching spectra.\n\n"
                        f"Selecting this many spectra will significantly impact performance:\n"
                        f"• Similarity calculations will be disabled\n"
                        f"• Only subset will be shown in detailed views\n"
                        f"• UI may become slow\n\n"
                        f"Do you want to proceed?",
                    )
                    if not result:
                        return
                elif len(matching_spectrum_ids) > 500:
                    result = messagebox.askyesno(
                        "Large Selection",
                        f"Found {len(matching_spectrum_ids)} matching spectra.\n"
                        f"Selecting this many spectra may impact performance.\n\n"
                        f"Do you want to proceed?",
                    )
                    if not result:
                        return

                # Trigger the selection callback to update the main application
                if self.on_metadata_changed:
                    # Store the matching IDs for the parent to handle
                    self._pending_selection = matching_spectrum_ids
                    print(
                        f"Debug: Setting pending selection of {len(matching_spectrum_ids)} spectra"
                    )
                    self.on_metadata_changed()

                    # Verify selection was applied (after a short delay)
                    self.after(
                        500,
                        lambda: self._verify_selection_applied(matching_spectrum_ids),
                    )
            else:
                display_value = value if value else "<empty>"
                messagebox.showinfo(
                    "No Matches", f"No spectra found with {key} = '{display_value}'"
                )

        except Exception as e:
            if progress_dialog:
                progress_dialog.destroy()
            messagebox.showerror("Error", f"Error during search: {str(e)}")

    def _verify_selection_applied(self, expected_ids):
        """Verify that the selection was properly applied (debug method)."""
        # This method can be removed once the issue is resolved
        try:
            # Get the current selection from the parent's spectrum tree
            if hasattr(self.parent, "spectrum_tree"):
                current_selection = (
                    self.parent.spectrum_tree.get_selected_spectrum_ids()
                )
                if set(current_selection) != set(expected_ids):
                    print(f"Debug: Selection mismatch!")
                    print(f"  Expected: {len(expected_ids)} spectra")
                    print(f"  Got: {len(current_selection)} spectra")
                    print(
                        f"  Tree grouped: {bool(self.parent.spectrum_tree.selected_grouping_tags)}"
                    )
                else:
                    print(
                        f"Debug: Selection verified correctly ({len(current_selection)} spectra)"
                    )
        except Exception as e:
            print(f"Debug: Could not verify selection: {e}")

    def _create_progress_dialog(self, title: str):
        """Create a simple progress dialog."""
        dialog = tk.Toplevel(self)
        dialog.title(title)
        dialog.geometry("300x100")
        dialog.transient(self)
        dialog.grab_set()

        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (300 // 2)
        y = (dialog.winfo_screenheight() // 2) - (100 // 2)
        dialog.geometry(f"300x100+{x}+{y}")

        # Progress label
        dialog.progress_label = ttk.Label(dialog, text="Searching...")
        dialog.progress_label.pack(pady=10)

        # Progress bar
        dialog.progress_bar = ttk.Progressbar(dialog, length=250, mode="determinate")
        dialog.progress_bar.pack(pady=10)

        def update_progress(percent, text=""):
            dialog.progress_bar["value"] = percent
            dialog.progress_label.config(text=text)
            dialog.update_idletasks()

        dialog.update_progress = update_progress
        return dialog

    def _on_double_click(self, event):
        """Handle double click to start editing."""
        item = self.metadata_tree.identify("item", event.x, event.y)
        column = self.metadata_tree.identify("column", event.x, event.y)

        if item and column in ("#1", "#2"):  # Key or Value columns
            self._start_editing(item, column)

    def _start_editing(self, item, column):
        """Start inline editing of a cell."""
        # Close any existing editor
        self._close_editor()

        # Get current value
        values = self.metadata_tree.item(item, "values")
        if not values:
            return

        current_value = values[0] if column == "#1" else values[1]
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
        self.edit_entry.bind("<Return>", self._finish_editing)
        self.edit_entry.bind("<Escape>", self._cancel_editing)
        self.edit_entry.bind("<FocusOut>", self._cancel_editing)

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
        values = self.metadata_tree.item(self.editing_item, "values")

        if not values:
            self._close_editor()
            return

        current_key = values[0]
        current_value = values[1]

        if self.editing_column == "#1":  # Editing key
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
                "Key Conflict", conflict_message, title="Choose Action"
            )

            if result is None:  # Cancel
                return
            elif result:  # Yes - Update only
                self._rename_key_selective(
                    old_key, new_key, existing_spectra_with_new_key
                )
            else:  # No - Merge
                self._rename_key_merge(old_key, new_key)
        else:
            # No conflict, rename for all spectra
            self._rename_key_all(old_key, new_key)

    def _rename_key_selective(
        self, old_key: str, new_key: str, exclude_spectrum_ids: List[int]
    ):
        """Rename key only in spectra that don't have the new key."""
        for spectrum in self.parser.spectra:
            if spectrum.spectrum_id not in exclude_spectrum_ids:
                if old_key in spectrum.metadata:
                    spectrum.rename_metadata_key(old_key, new_key)

        # Add empty key for spectra that don't have the old key
        for spectrum in self.parser.spectra:
            if (
                spectrum.spectrum_id not in exclude_spectrum_ids
                and new_key not in spectrum.metadata
            ):
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
                # Always update the metadata - even if new_value is empty string
                spectrum.metadata[key] = new_value

        self._refresh_after_change()

    def _refresh_after_change(self):
        """Refresh the view after metadata changes."""
        self._populate_metadata()
        if self.on_metadata_changed:
            self.on_metadata_changed()

    def get_pending_selection(self) -> Optional[List[int]]:
        """Get and clear any pending selection."""
        if self._pending_selection:
            selection = self._pending_selection
            self._pending_selection = None
            return selection
        return None


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
        self.dialog.geometry(
            "+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50)
        )

        self._create_widgets()
        self.dialog.wait_window()

    def _create_widgets(self):
        """Create dialog widgets."""
        main_frame = ttk.Frame(self.dialog, padding=10)
        main_frame.pack(fill="both", expand=True)

        # Key input
        ttk.Label(main_frame, text="Key Name:").grid(
            row=0, column=0, sticky="w", pady=5
        )
        self.key_var = tk.StringVar()
        key_entry = ttk.Entry(main_frame, textvariable=self.key_var, width=30)
        key_entry.grid(row=0, column=1, padx=(10, 0), pady=5)
        key_entry.focus()

        # Value input
        ttk.Label(main_frame, text="Value:").grid(row=1, column=0, sticky="w", pady=5)
        self.value_var = tk.StringVar()
        value_entry = ttk.Entry(main_frame, textvariable=self.value_var, width=30)
        value_entry.grid(row=1, column=1, padx=(10, 0), pady=5)

        # Scope selection
        ttk.Label(main_frame, text="Apply to:").grid(
            row=2, column=0, sticky="w", pady=5
        )
        self.scope_var = tk.StringVar(value="selected")
        scope_frame = ttk.Frame(main_frame)
        scope_frame.grid(row=2, column=1, padx=(10, 0), pady=5, sticky="w")

        selected_text = (
            f"Selected spectra ({len(self.selected_spectrum_ids)})"
            if self.selected_spectrum_ids
            else "Selected spectra (none)"
        )
        ttk.Radiobutton(
            scope_frame, text=selected_text, variable=self.scope_var, value="selected"
        ).pack(anchor="w")
        ttk.Radiobutton(
            scope_frame,
            text=f"All loaded spectra ({len(self.parser.spectra)})",
            variable=self.scope_var,
            value="all",
        ).pack(anchor="w")

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=20)

        ttk.Button(button_frame, text="Add", command=self._add_key_value).pack(
            side="left", padx=5
        )
        ttk.Button(button_frame, text="Cancel", command=self._cancel).pack(
            side="left", padx=5
        )

        # Bind Enter key
        self.dialog.bind("<Return>", lambda e: self._add_key_value())
        self.dialog.bind("<Escape>", lambda e: self._cancel())

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
            if not messagebox.askyesno(
                "Key Exists",
                f"Key '{key}' already exists. Do you want to update its value?",
            ):
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
        self.dialog.geometry(
            "+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50)
        )

        self._create_widgets()
        self._populate_keys()
        self.dialog.wait_window()

    def _create_widgets(self):
        """Create dialog widgets."""
        main_frame = ttk.Frame(self.dialog, padding=10)
        main_frame.pack(fill="both", expand=True)

        # Top frame with three columns
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill="both", expand=True, pady=(0, 10))

        # Left column - Keys list
        keys_frame = ttk.LabelFrame(top_frame, text="Metadata Keys", padding=5)
        keys_frame.pack(side="left", fill="y", padx=(0, 5))

        # Keys listbox with scrollbar
        keys_container = ttk.Frame(keys_frame)
        keys_container.pack(fill="both", expand=True)

        self.keys_listbox = tk.Listbox(keys_container, width=20, height=20)
        keys_scrollbar = ttk.Scrollbar(
            keys_container, orient="vertical", command=self.keys_listbox.yview
        )
        self.keys_listbox.configure(yscrollcommand=keys_scrollbar.set)

        self.keys_listbox.pack(side="left", fill="both", expand=True)
        keys_scrollbar.pack(side="right", fill="y")

        self.keys_listbox.bind("<<ListboxSelect>>", self._on_key_selection)

        # Middle column - Values table
        values_frame = ttk.LabelFrame(top_frame, text="Values", padding=5)
        values_frame.pack(side="left", fill="both", expand=True, padx=5)

        # Values treeview
        columns = ("Original Value", "Count", "Updated Value")
        self.values_tree = ttk.Treeview(
            values_frame, columns=columns, show="headings", height=20
        )

        for col in columns:
            self.values_tree.heading(col, text=col)

        # Set column widths
        self.values_tree.column("Original Value", width=200)
        self.values_tree.column("Count", width=80)
        self.values_tree.column("Updated Value", width=200)

        # Scrollbars for values tree
        values_v_scrollbar = ttk.Scrollbar(
            values_frame, orient="vertical", command=self.values_tree.yview
        )
        values_h_scrollbar = ttk.Scrollbar(
            values_frame, orient="horizontal", command=self.values_tree.xview
        )

        self.values_tree.configure(
            yscrollcommand=values_v_scrollbar.set, xscrollcommand=values_h_scrollbar.set
        )

        self.values_tree.grid(row=0, column=0, sticky="nsew")
        values_v_scrollbar.grid(row=0, column=1, sticky="ns")
        values_h_scrollbar.grid(row=1, column=0, sticky="ew")

        values_frame.grid_rowconfigure(0, weight=1)
        values_frame.grid_columnconfigure(0, weight=1)

        # Bottom frame - Regex editor
        regex_frame = ttk.LabelFrame(main_frame, text="Regex Editor", padding=5)
        regex_frame.pack(fill="x", pady=(0, 10))

        # Regex pattern input
        pattern_frame = ttk.Frame(regex_frame)
        pattern_frame.pack(fill="x", pady=5)

        ttk.Label(pattern_frame, text="Search Pattern (regex):").pack(anchor="w")
        self.pattern_var = tk.StringVar()
        self.pattern_entry = ttk.Entry(
            pattern_frame, textvariable=self.pattern_var, width=80
        )
        self.pattern_entry.pack(fill="x", pady=2)

        # Replacement input
        replacement_frame = ttk.Frame(regex_frame)
        replacement_frame.pack(fill="x", pady=5)

        ttk.Label(replacement_frame, text="Replacement:").pack(anchor="w")
        self.replacement_var = tk.StringVar()
        self.replacement_entry = ttk.Entry(
            replacement_frame, textvariable=self.replacement_var, width=80
        )
        self.replacement_entry.pack(fill="x", pady=2)

        # Regex options
        options_frame = ttk.Frame(regex_frame)
        options_frame.pack(fill="x", pady=5)

        self.ignore_case_var = tk.BooleanVar()
        ttk.Checkbutton(
            options_frame, text="Ignore case", variable=self.ignore_case_var
        ).pack(side="left", padx=(0, 10))

        self.multiline_var = tk.BooleanVar()
        ttk.Checkbutton(
            options_frame, text="Multiline", variable=self.multiline_var
        ).pack(side="left", padx=(0, 10))

        # Buttons for regex operations
        regex_buttons_frame = ttk.Frame(regex_frame)
        regex_buttons_frame.pack(fill="x", pady=5)

        ttk.Button(
            regex_buttons_frame, text="Preview", command=self._preview_regex
        ).pack(side="left", padx=(0, 5))
        ttk.Button(regex_buttons_frame, text="Apply", command=self._apply_regex).pack(
            side="left", padx=(0, 5)
        )
        ttk.Button(regex_buttons_frame, text="Reset", command=self._reset_values).pack(
            side="left", padx=(0, 5)
        )

        # Examples
        examples_frame = ttk.Frame(regex_frame)
        examples_frame.pack(fill="x", pady=5)

        ttk.Label(examples_frame, text="Examples:", font=("Arial", 9, "bold")).pack(
            anchor="w"
        )
        examples_text = (
            "• Remove prefix: ^prefix_ → (empty)\n"
            "• Replace spaces: \\s+ → _\n"
            "• Extract numbers: .*?([0-9]+).* → \\1\n"
            "• Add suffix: (.*) → \\1_new"
        )
        ttk.Label(
            examples_frame, text=examples_text, font=("Arial", 8), foreground="gray"
        ).pack(anchor="w")

        # Bottom buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x")

        ttk.Button(button_frame, text="Close", command=self._close_dialog).pack(
            side="right", padx=5
        )

        # Bind Enter key for quick preview
        self.pattern_entry.bind("<KeyRelease>", self._on_pattern_change)
        self.replacement_entry.bind("<KeyRelease>", self._on_pattern_change)

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
            self.value_data[value] = {"count": count, "updated": value}
            self.values_tree.insert("", "end", values=(value, count, value))

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
                values = self.values_tree.item(item, "values")
                original_value = values[0]

                try:
                    updated_value = compiled_pattern.sub(replacement, original_value)
                    self.value_data[original_value]["updated"] = updated_value

                    # Update tree display
                    self.values_tree.item(
                        item, values=(original_value, values[1], updated_value)
                    )

                except Exception as e:
                    # If replacement fails for this value, keep original
                    self.value_data[original_value]["updated"] = original_value
                    self.values_tree.item(
                        item, values=(original_value, values[1], f"ERROR: {str(e)}")
                    )

        except re.error as e:
            messagebox.showerror("Regex Error", f"Invalid regular expression: {str(e)}")

    def _apply_regex(self):
        """Apply the regex transformation to all spectra."""
        if not self.current_key or not self.value_data:
            messagebox.showwarning(
                "Warning", "Please select a key and preview changes first."
            )
            return

        # Confirm before applying
        num_affected = sum(
            data["count"]
            for data in self.value_data.values()
            if data["updated"]
            != list(self.value_data.keys())[list(self.value_data.values()).index(data)]
        )

        if num_affected == 0:
            messagebox.showinfo("Info", "No changes to apply.")
            return

        if not messagebox.askyesno(
            "Confirm Changes",
            f"Apply regex transformation to {num_affected} values in key '{self.current_key}'?",
        ):
            return

        # Apply changes to all spectra
        changes_made = False
        for spectrum in self.parser.spectra:
            current_value = spectrum.get_metadata_value(self.current_key)
            if current_value in self.value_data:
                new_value = self.value_data[current_value]["updated"]
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
            self.value_data[original_value]["updated"] = original_value

        # Update tree display
        for item in self.values_tree.get_children():
            values = self.values_tree.item(item, "values")
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
        header_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(
            header_frame, text="Spectrum Visualization", font=("Arial", 12, "bold")
        ).pack()

        # Matplotlib figure
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

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
            ax.text(
                0.5,
                0.5,
                "No spectra selected",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            self.canvas.draw()
            return

        # Get selected spectra
        selected_spectra = [
            s
            for s in self.parser.spectra
            if s.spectrum_id in self.selected_spectrum_ids
        ]

        if not selected_spectra:
            return

        # Performance check: limit to first 5 spectra if more than 10 are selected
        original_count = len(selected_spectra)
        if original_count > 10:
            selected_spectra = selected_spectra[:5]
            # Add a warning text to the plot
            performance_warning = f"Performance limit: Showing first 5 of {original_count} selected spectra"
        else:
            performance_warning = None

        # Calculate global m/z limits for all selected spectra
        global_mz_min = float("inf")
        global_mz_max = float("-inf")

        for spectrum in selected_spectra:
            if spectrum.ions.size > 0:
                mz_values = spectrum.ions[:, 0]
                global_mz_min = min(global_mz_min, mz_values.min())
                global_mz_max = max(global_mz_max, mz_values.max())

        # Add some padding to the limits
        if global_mz_min != float("inf") and global_mz_max != float("-inf"):
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
            self._plot_single_spectrum(
                ax, selected_spectra[0], (global_mz_min, global_mz_max), is_last=True
            )
        else:
            for i, spectrum in enumerate(selected_spectra):
                is_last = i == n_spectra - 1
                ax = self.figure.add_subplot(n_spectra, 1, i + 1)
                self._plot_single_spectrum(
                    ax, spectrum, (global_mz_min, global_mz_max), is_last=is_last
                )

        # Add performance warning if applicable
        if performance_warning:
            self.figure.suptitle(performance_warning, fontsize=10, color="red", y=0.98)

        # Minimize space between subplots
        self.figure.tight_layout(pad=0.5, h_pad=0.2)
        # Adjust layout to make room for the warning if present
        if performance_warning:
            self.figure.subplots_adjust(top=0.94, hspace=0.1)
        else:
            self.figure.subplots_adjust(hspace=0.1)
        self.canvas.draw()

    def _plot_single_spectrum(self, ax, spectrum, mz_limits=None, is_last=False):
        """Plot a single spectrum as a stick chart."""
        if spectrum.ions.size == 0:
            ax.text(
                0.5,
                0.5,
                "No ion data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            if mz_limits:
                ax.set_xlim(mz_limits)
            return

        mz_values = spectrum.ions[:, 0]
        intensity_values = spectrum.ions[:, 1]

        # Create stick plot
        ax.vlines(mz_values, 0, intensity_values, colors="blue", linewidth=1.5)

        # Only show x-axis label and ticks on the last spectrum
        if is_last:
            ax.set_xlabel("m/z")
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", which="both", bottom=False)

        ax.set_ylabel("Intensity")

        # Remove title - spectrum ID will be shown in y-axis label instead
        spectrum_label = f"S {spectrum.spectrum_id}"
        ax.set_ylabel(f"{spectrum_label}\nIntensity", fontsize=9)

        ax.grid(True, alpha=0.3)

        # Set limits
        if mz_limits:
            ax.set_xlim(mz_limits)
        elif len(mz_values) > 0:
            ax.set_xlim(mz_values.min() * 0.95, mz_values.max() * 1.05)

        # Set y limits
        if len(intensity_values) > 0:
            ax.set_ylim(0, intensity_values.max() * 1.1)

    def clear_plot(self):
        """Clear the plot and reset to empty state."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "No spectra loaded",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        self.canvas.draw()
        self.parser = None
        self.selected_spectrum_ids = []

    def clear_plot(self):
        """Clear the spectrum plot and reset data."""
        fig = self.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "No spectra loaded",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        self.canvas.draw()
        self.parser = None
        self.selected_spectrum_ids = []


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
        header_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(header_frame, text="Ion Data", font=("Arial", 12, "bold")).pack()

        # Notebook for multiple spectra
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

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
            s
            for s in self.parser.spectra
            if s.spectrum_id in self.selected_spectrum_ids
        ]

        for spectrum in selected_spectra:
            self._create_table_for_spectrum(spectrum)

    def _create_table_for_spectrum(self, spectrum: Spectrum):
        """Create a table tab for a single spectrum."""
        # Create frame for this spectrum
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=f"S {spectrum.spectrum_id}")

        # Create treeview
        columns = ("Index", "m/z", "Intensity")
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=15)

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)

        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        h_scrollbar = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)

        tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        tree.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        # Populate with ion data
        if spectrum.ions.size > 0:
            for i, (mz, intensity) in enumerate(spectrum.ions):
                tree.insert("", "end", values=(i + 1, f"{mz:.6f}", f"{intensity:.3f}"))

    def clear_data(self):
        """Clear all data from the tables."""
        # Clear all tabs
        for tab_id in self.notebook.tabs():
            self.notebook.forget(tab_id)

        self.parser = None
        self.selected_spectrum_ids = []


class CosineSimilarityVisualization(ttk.Frame):
    """Component for visualizing cosine similarity matrix and statistics."""

    def __init__(self, parent):
        super().__init__(parent)
        self.parser: Optional[MGFParser] = None
        self.selected_spectrum_ids: List[int] = []
        self.similarity_matrix: Optional[np.ndarray] = None
        self.calculation_thread: Optional[threading.Thread] = None
        self.cancel_calculation = False

        # Performance limits
        self.MAX_SPECTRA_AUTO = 20  # Auto-calculate up to this many spectra (reduced)
        self.MAX_SPECTRA_MANUAL = (
            100  # Allow manual calculation up to this many (reduced)
        )

        self._create_widgets()

    def _create_widgets(self):
        """Create the cosine similarity visualization widgets."""
        # Header
        header_frame = ttk.Frame(self)
        header_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(
            header_frame, text="Cosine Similarity", font=("Arial", 12, "bold")
        ).pack()

        # Control frame
        control_frame = ttk.Frame(header_frame)
        control_frame.pack(fill="x", pady=2)

        # Tolerance setting
        ttk.Label(control_frame, text="m/z Tolerance:").pack(side="left")
        self.tolerance_var = tk.DoubleVar(value=0.1)
        tolerance_spinbox = ttk.Spinbox(
            control_frame,
            from_=0.01,
            to=1.0,
            increment=0.01,
            width=8,
            textvariable=self.tolerance_var,
            command=self._on_tolerance_changed,
        )
        tolerance_spinbox.pack(side="left", padx=5)

        # Calculate button
        self.calc_button = ttk.Button(
            control_frame, text="Calculate", command=self._recalculate_similarity
        )
        self.calc_button.pack(side="left", padx=5)

        # Cancel button
        self.cancel_button = ttk.Button(
            control_frame,
            text="Cancel",
            command=self._cancel_calculation,
            state="disabled",
        )
        self.cancel_button.pack(side="left", padx=2)

        # Progress bar
        self.progress_var = tk.StringVar()
        self.progress_label = ttk.Label(control_frame, textvariable=self.progress_var)
        self.progress_label.pack(side="left", padx=10)

        # Main content area
        content_frame = ttk.Frame(self)
        content_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Matplotlib figure for heatmap
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, content_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Statistics frame
        stats_frame = ttk.LabelFrame(
            content_frame, text="Similarity Statistics", padding=5
        )
        stats_frame.pack(fill="x", pady=5)

        self.stats_text = tk.Text(
            stats_frame, height=4, wrap="word", font=("Courier", 9)
        )
        self.stats_text.pack(fill="x")

    def load_data(self, parser: MGFParser, selected_spectrum_ids: List[int]):
        """Load and visualize cosine similarity for selected spectra."""
        # Cancel any ongoing calculation
        self._cancel_calculation()

        self.parser = parser
        self.selected_spectrum_ids = selected_spectrum_ids

        # Check if we should auto-calculate or require manual trigger
        if len(selected_spectrum_ids) <= self.MAX_SPECTRA_AUTO:
            self._calculate_and_display_similarity()
        else:
            self._show_performance_warning()

    def _show_performance_warning(self):
        """Show performance warning for large selections."""
        self.figure.clear()
        self.stats_text.delete(1.0, tk.END)

        n_selected = len(self.selected_spectrum_ids)
        ax = self.figure.add_subplot(111)

        if n_selected > self.MAX_SPECTRA_MANUAL:
            warning_text = (
                f"Too many spectra selected ({n_selected}).\n"
                f"Maximum supported: {self.MAX_SPECTRA_MANUAL}\n\n"
                f"Please select fewer spectra for\n"
                f"similarity analysis."
            )
            self.calc_button.config(state="disabled")
        else:
            warning_text = (
                f"Large selection ({n_selected} spectra).\n"
                f"Similarity calculation may take time.\n\n"
                f'Click "Calculate" to proceed.'
            )
            self.calc_button.config(state="normal")

        ax.text(
            0.5,
            0.5,
            warning_text,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
            color="orange",
        )
        ax.set_xticks([])
        ax.set_yticks([])

        # Show computation complexity estimate
        n_comparisons = n_selected * (n_selected - 1) // 2
        stats_text = (
            f"Selected: {n_selected} spectra\n"
            f"Pairwise comparisons needed: {n_comparisons:,}\n"
            f"Estimated time: {self._estimate_calculation_time(n_selected)}"
        )
        self.stats_text.insert(tk.END, stats_text)

        self.progress_var.set("")
        self.canvas.draw()

    def _estimate_calculation_time(self, n_spectra: int) -> str:
        """Estimate calculation time based on number of spectra."""
        n_comparisons = n_spectra * (n_spectra - 1) // 2

        # Rough estimates based on typical performance
        if n_comparisons < 100:
            return "< 1 second"
        elif n_comparisons < 1000:
            return "1-5 seconds"
        elif n_comparisons < 5000:
            return "5-30 seconds"
        elif n_comparisons < 20000:
            return "30 seconds - 2 minutes"
        else:
            return "> 2 minutes"

    def _on_tolerance_changed(self):
        """Handle tolerance change."""
        # Only auto-recalculate for small selections
        if len(self.selected_spectrum_ids) <= self.MAX_SPECTRA_AUTO:
            self._recalculate_similarity()

    def _recalculate_similarity(self):
        """Recalculate similarity with current tolerance."""
        if not self.parser or not self.selected_spectrum_ids:
            return

        if len(self.selected_spectrum_ids) > self.MAX_SPECTRA_MANUAL:
            messagebox.showwarning(
                "Too Many Spectra",
                f"Cannot calculate similarity for {len(self.selected_spectrum_ids)} spectra.\n"
                f"Maximum supported: {self.MAX_SPECTRA_MANUAL}",
            )
            return

        self._calculate_and_display_similarity()

    def _cancel_calculation(self):
        """Cancel ongoing calculation."""
        if self.calculation_thread and self.calculation_thread.is_alive():
            self.cancel_calculation = True
            # Wait a bit for thread to finish
            self.after(100, self._check_cancellation)

    def _check_cancellation(self):
        """Check if calculation thread has finished after cancellation."""
        if self.calculation_thread and self.calculation_thread.is_alive():
            # Still running, check again later
            self.after(100, self._check_cancellation)
        else:
            # Thread finished
            self._reset_ui_after_calculation()

    def _reset_ui_after_calculation(self):
        """Reset UI state after calculation completes or is cancelled."""
        self.cancel_calculation = False
        self.calc_button.config(state="normal")
        self.cancel_button.config(state="disabled")
        self.progress_var.set("")

    def _calculate_and_display_similarity(self):
        """Calculate and display the cosine similarity matrix and statistics."""
        # Clear display first
        self.figure.clear()
        self.stats_text.delete(1.0, tk.END)

        if not self.parser or len(self.selected_spectrum_ids) < 2:
            ax = self.figure.add_subplot(111)
            if len(self.selected_spectrum_ids) == 1:
                ax.text(
                    0.5,
                    0.5,
                    "Select 2+ spectra\nfor similarity comparison",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                self.stats_text.insert(
                    tk.END,
                    "Single spectrum selected.\nSimilarity: 1.0 (self-similarity)",
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No spectra selected",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                self.stats_text.insert(tk.END, "No spectra selected for comparison.")
            self.canvas.draw()
            return

        # For large calculations, use threading
        if len(self.selected_spectrum_ids) > 10:
            self._start_threaded_calculation()
        else:
            self._calculate_similarity_direct()

    def _start_threaded_calculation(self):
        """Start similarity calculation in a separate thread."""
        # Update UI for calculation in progress
        self.calc_button.config(state="disabled")
        self.cancel_button.config(state="normal")
        self.progress_var.set("Calculating...")

        # Show placeholder while calculating
        ax = self.figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "Calculating similarity matrix...\nPlease wait",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        self.canvas.draw()

        # Start calculation thread
        self.cancel_calculation = False
        self.calculation_thread = threading.Thread(
            target=self._calculate_similarity_threaded, daemon=True
        )
        self.calculation_thread.start()

        # Start monitoring the calculation
        self._check_calculation_progress()

    def _calculate_similarity_threaded(self):
        """Calculate similarity matrix in a separate thread with progress updates."""
        try:
            tolerance = self.tolerance_var.get()
            n_spectra = len(self.selected_spectrum_ids)

            # Get spectra objects
            spectra = [
                s
                for s in self.parser.spectra
                if s.spectrum_id in self.selected_spectrum_ids
            ]

            if n_spectra < 2:
                self.similarity_matrix = np.array([[1.0]] if n_spectra == 1 else [])
                return

            similarity_matrix = np.zeros((n_spectra, n_spectra))
            total_calculations = n_spectra * (n_spectra - 1) // 2
            completed = 0

            # Fill diagonal with 1.0
            for i in range(n_spectra):
                similarity_matrix[i, i] = 1.0

            # Calculate upper triangle only
            for i in range(n_spectra):
                if self.cancel_calculation:
                    return

                for j in range(i + 1, n_spectra):
                    if self.cancel_calculation:
                        return

                    # Calculate similarity
                    sim = self.parser.calculate_cosine_similarity(
                        spectra[i], spectra[j], tolerance
                    )
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim  # Symmetric

                    completed += 1

                    # Update progress periodically
                    if completed % max(1, total_calculations // 20) == 0:
                        progress_pct = (completed / total_calculations) * 100
                        self.after_idle(
                            lambda p=progress_pct: self.progress_var.set(
                                f"Calculating... {p:.0f}%"
                            )
                        )

            if not self.cancel_calculation:
                self.similarity_matrix = similarity_matrix
                # Schedule UI update on main thread
                self.after_idle(self._display_similarity_results)

        except Exception as e:
            # Handle errors
            self.after_idle(lambda: self._show_calculation_error(str(e)))

    def _calculate_similarity_direct(self):
        """Calculate similarity matrix directly (for small datasets)."""
        try:
            tolerance = self.tolerance_var.get()
            self.similarity_matrix = self.parser.calculate_similarity_matrix(
                self.selected_spectrum_ids, tolerance
            )
            self._display_similarity_results()
        except Exception as e:
            self._show_calculation_error(str(e))

    def _show_calculation_error(self, error_msg: str):
        """Show calculation error in the display."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            f"Calculation Error:\n{error_msg}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="red",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        self.canvas.draw()
        self._reset_ui_after_calculation()

    def _check_calculation_progress(self):
        """Check if threaded calculation is complete."""
        if self.calculation_thread and self.calculation_thread.is_alive():
            # Still calculating, check again later
            self.after(100, self._check_calculation_progress)
        else:
            # Calculation finished
            self._reset_ui_after_calculation()

    def _display_similarity_results(self):
        """Display the calculated similarity matrix and statistics."""
        if self.similarity_matrix is None or self.similarity_matrix.size == 0:
            return

        # Create heatmap
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        im = ax.imshow(
            self.similarity_matrix, cmap="viridis", vmin=0, vmax=1, aspect="equal"
        )

        # Add colorbar
        cbar = self.figure.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Cosine Similarity", rotation=270, labelpad=15)

        # Set labels
        spectrum_labels = [f"S{sid}" for sid in self.selected_spectrum_ids]
        ax.set_xticks(range(len(spectrum_labels)))
        ax.set_yticks(range(len(spectrum_labels)))
        ax.set_xticklabels(spectrum_labels, rotation=45, ha="right")
        ax.set_yticklabels(spectrum_labels)

        # Add text annotations for values (only for smaller matrices)
        n = self.similarity_matrix.shape[0]
        if n <= 20:  # Only show values for matrices up to 20x20
            for i in range(n):
                for j in range(n):
                    text = ax.text(
                        j,
                        i,
                        f"{self.similarity_matrix[i, j]:.3f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                    )

        tolerance = self.tolerance_var.get()
        ax.set_title(f"Similarity Matrix (tolerance: {tolerance:.2f})")

        # Adjust layout
        self.figure.tight_layout()
        self.canvas.draw()

        # Calculate and display statistics
        self._display_statistics()

    def _display_statistics(self):
        """Display similarity statistics."""
        if self.similarity_matrix is None or self.similarity_matrix.size == 0:
            return

        # Get upper triangle (excluding diagonal) for statistics
        n = self.similarity_matrix.shape[0]
        if n < 2:
            return

        # Extract unique pairwise similarities (upper triangle, no diagonal)
        triu_indices = np.triu_indices(n, k=1)
        similarities = self.similarity_matrix[triu_indices]

        if len(similarities) == 0:
            return

        # Calculate percentiles
        percentiles = [0, 10, 25, 50, 75, 90, 100]
        values = np.percentile(similarities, percentiles)

        # Format statistics
        stats_text = f"Pairwise Similarities (n={len(similarities)}):\n"
        stats_text += (
            f"Min:    {values[0]:.4f}   10%: {values[1]:.4f}   25%: {values[2]:.4f}\n"
        )
        stats_text += (
            f"Median: {values[3]:.4f}   75%: {values[4]:.4f}   90%: {values[5]:.4f}\n"
        )
        stats_text += f"Max:    {values[6]:.4f}   Mean: {np.mean(similarities):.4f}"

        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, stats_text)

    def clear_data(self):
        """Clear all data from the cosine similarity visualization."""
        self.heatmap_canvas.delete("all")
        self.stats_text.delete(1.0, tk.END)
        self.similarity_matrix = None
        self.selected_spectra = []


class FileLoadingDialog:
    """Dialog for configuring file loading options including database identifier and prefix."""

    def __init__(self, parent, file_to_load, existing_prefixes=None):
        self.parent = parent
        self.file_to_load = file_to_load
        self.existing_prefixes = existing_prefixes or set()
        self.result = None

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("File Loading Options")
        self.dialog.geometry("450x400")
        self.dialog.resizable(True, True)
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Center the dialog
        self.dialog.geometry(
            "+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50)
        )

        self._create_widgets()
        self.dialog.wait_window()

    def _create_widgets(self):
        """Create dialog widgets."""
        main_frame = ttk.Frame(self.dialog, padding=15)
        main_frame.pack(fill="both", expand=True)

        # Instructions
        instructions = ttk.Label(
            main_frame,
            text=f"Configure metadata and naming options for the loaded spectra:\n(* indicates required fields)\nFile is: '{self.file_to_load}'",
            font=("Arial", 10),
        )
        instructions.pack(anchor="w", pady=(0, 15))

        # Database identifier section
        db_frame = ttk.LabelFrame(main_frame, text="Database Information", padding=10)
        db_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(db_frame, text="Database Identifier: *").pack(anchor="w")
        self.database_identifier_var = tk.StringVar()
        db_entry = ttk.Entry(
            db_frame, textvariable=self.database_identifier_var, width=40
        )
        db_entry.pack(fill="x", pady=(5, 0))

        # Bind validation to database identifier entry
        self.database_identifier_var.trace_add("write", self._validate_database_id)

        ttk.Label(
            db_frame,
            text="This identifier will be added to all spectra metadata (required)",
            font=("Arial", 8),
            foreground="gray",
        ).pack(anchor="w", pady=(2, 0))

        # Prefix section
        prefix_frame = ttk.LabelFrame(main_frame, text="Spectrum ID Prefix", padding=10)
        prefix_frame.pack(fill="x", pady=(0, 15))

        ttk.Label(prefix_frame, text="Prefix for spectrum IDs: *").pack(anchor="w")
        self.prefix_var = tk.StringVar()
        self.prefix_entry = ttk.Entry(
            prefix_frame, textvariable=self.prefix_var, width=40
        )
        self.prefix_entry.pack(fill="x", pady=(5, 0))

        # Validation label for prefix
        self.prefix_status_label = ttk.Label(
            prefix_frame,
            text="This prefix will be applied to each spectrum ID",
            font=("Arial", 8),
            foreground="gray",
        )
        self.prefix_status_label.pack(anchor="w", pady=(2, 0))

        # Bind validation to prefix entry
        self.prefix_var.trace_add("write", self._validate_prefix)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(10, 0))

        ttk.Button(button_frame, text="Cancel", command=self._cancel).pack(
            side="right", padx=(5, 0)
        )
        self.ok_button = ttk.Button(button_frame, text="OK", command=self._ok)
        self.ok_button.pack(side="right")

        # Initially disable OK button since fields are required
        self.ok_button.config(state="disabled")

        # Bind Enter and Escape keys
        self.dialog.bind("<Return>", lambda e: self._ok())
        self.dialog.bind("<Escape>", lambda e: self._cancel())

        # Focus on database identifier entry
        db_entry.focus_set()

    def _validate_prefix(self, *args):
        """Validate the prefix to ensure it hasn't been used before."""
        prefix = self.prefix_var.get().strip()

        if not prefix:
            self.prefix_status_label.config(
                text="⚠ Prefix is required",
                foreground="red",
            )
            self._update_ok_button_state()
        elif prefix in self.existing_prefixes:
            self.prefix_status_label.config(
                text="⚠ This prefix has already been used!", foreground="red"
            )
            self._update_ok_button_state()
        else:
            self.prefix_status_label.config(
                text="✓ Prefix is available", foreground="green"
            )
            self._update_ok_button_state()

    def _validate_database_id(self, *args):
        """Validate the database identifier."""
        self._update_ok_button_state()

    def _update_ok_button_state(self):
        """Update the OK button state based on validation."""
        prefix = self.prefix_var.get().strip()
        database_id = self.database_identifier_var.get().strip()

        # Both fields are required and prefix must not be used
        if database_id and prefix and prefix not in self.existing_prefixes:
            self.ok_button.config(state="normal")
        else:
            self.ok_button.config(state="disabled")

    def _ok(self):
        """Handle OK button."""
        prefix = self.prefix_var.get().strip()
        database_id = self.database_identifier_var.get().strip()

        # Validate required fields
        if not database_id:
            messagebox.showerror(
                "Missing Information", "Database identifier is required."
            )
            return

        if not prefix:
            messagebox.showerror("Missing Information", "Prefix is required.")
            return

        # Final validation
        if prefix in self.existing_prefixes:
            messagebox.showerror(
                "Invalid Prefix",
                f"The prefix '{prefix}' has already been used. Please choose a different prefix.",
            )
            return

        self.result = {
            "database_identifier": database_id,
            "prefix": prefix,
        }
        self.dialog.destroy()

    def _cancel(self):
        """Handle Cancel button."""
        self.result = None
        self.dialog.destroy()


class AverageSpectrumDialog:
    """Dialog for configuring spectrum averaging parameters."""

    def __init__(self, parent):
        self.result = None

        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Calculate Average Spectrum per Group")
        self.dialog.geometry("450x450")
        self.dialog.resizable(True, True)

        # Make dialog modal
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Center dialog on parent
        self.dialog.geometry(
            "+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50)
        )

        self._create_widgets()

        # Wait for dialog to close
        self.dialog.wait_window()

    def _create_widgets(self):
        """Create dialog widgets."""
        main_frame = ttk.Frame(self.dialog, padding=20)
        main_frame.pack(fill="both", expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Average Spectrum Parameters",
            font=("TkDefaultFont", 12, "bold"),
        )
        title_label.pack(pady=(0, 20))

        # Parameters frame
        params_frame = ttk.Frame(main_frame)
        params_frame.pack(fill="x", pady=(0, 20))

        # Binning m/z
        binning_frame = ttk.Frame(params_frame)
        binning_frame.pack(fill="x", pady=5)

        ttk.Label(
            binning_frame, text="Binning m/z tolerance:", width=20, anchor="w"
        ).pack(side="left")
        self.binning_var = tk.StringVar(value="0.1")
        binning_entry = ttk.Entry(
            binning_frame, textvariable=self.binning_var, width=12
        )
        binning_entry.pack(side="right")

        # Averaging method
        method_frame = ttk.Frame(params_frame)
        method_frame.pack(fill="x", pady=5)

        ttk.Label(method_frame, text="Averaging method:", width=20, anchor="w").pack(
            side="left"
        )
        self.method_var = tk.StringVar(value="average")
        method_combo = ttk.Combobox(
            method_frame,
            textvariable=self.method_var,
            values=["average", "median"],
            state="readonly",
            width=10,
        )
        method_combo.pack(side="right")

        # New key
        key_frame = ttk.Frame(params_frame)
        key_frame.pack(fill="x", pady=5)

        ttk.Label(key_frame, text="New metadata key:", width=20, anchor="w").pack(
            side="left"
        )
        self.key_var = tk.StringVar(value="SPECTYPE")
        key_entry = ttk.Entry(key_frame, textvariable=self.key_var, width=12)
        key_entry.pack(side="right")

        # New value
        value_frame = ttk.Frame(params_frame)
        value_frame.pack(fill="x", pady=5)

        ttk.Label(value_frame, text="New metadata value:", width=20, anchor="w").pack(
            side="left"
        )
        self.value_var = tk.StringVar(value="Averaged")
        value_entry = ttk.Entry(value_frame, textvariable=self.value_var, width=12)
        value_entry.pack(side="right")

        # Info text
        info_text = (
            "This will create average spectra for each group at the same level.\n"
            "Spectra will be normalized using common peaks and then averaged.\n"
            "Groups with only one spectrum will be skipped.\n\n"
            "Peak combining: Peaks with similar m/z values (within tolerance)\n"
            "will be combined into single peaks with weighted average m/z."
        )
        info_label = ttk.Label(
            main_frame,
            text=info_text,
            wraplength=400,
            justify="left",
            foreground="gray",
        )
        info_label.pack(pady=(0, 20))

        # Buttons - try a completely different approach with regular tkinter buttons
        button_frame = tk.Frame(main_frame, height=60, bg="SystemButtonFace")

        # Use regular tk.Button instead of ttk.Button
        cancel_btn = tk.Button(
            button_frame,
            text="Cancel",
            command=self._cancel,
            width=12,
            height=2,
            font=("TkDefaultFont", 9),
        )
        cancel_btn.pack(side="right", padx=10, pady=15)

        calculate_btn = tk.Button(
            button_frame,
            text="Calculate",
            command=self._ok,
            width=12,
            height=2,
            font=("TkDefaultFont", 9),
        )
        calculate_btn.pack(side="right", padx=5, pady=15)
        button_frame.pack(fill="x", pady=20)
        button_frame.pack_propagate(False)

    def _ok(self):
        """Handle OK button."""
        try:
            binning_mz = float(self.binning_var.get())
            if binning_mz <= 0:
                raise ValueError("Binning m/z must be positive")
        except ValueError:
            messagebox.showerror("Error", "Invalid binning m/z value")
            return

        new_key = self.key_var.get().strip()
        new_value = self.value_var.get().strip()

        if not new_key:
            messagebox.showerror("Error", "New metadata key cannot be empty")
            return

        if not new_value:
            messagebox.showerror("Error", "New metadata value cannot be empty")
            return

        self.result = {
            "binning_mz": binning_mz,
            "averaging_method": self.method_var.get(),
            "new_key": new_key,
            "new_value": new_value,
        }

        self.dialog.destroy()

    def _cancel(self):
        """Handle Cancel button."""
        self.result = None
        self.dialog.destroy()
