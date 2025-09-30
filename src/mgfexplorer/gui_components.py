"""
GUI components for the MGF Explorer application.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from typing import List, Dict, Any, Optional
import re
import threading
import time
from .mgf_parser import MGFParser, Spectrum

# Natural sorting for spectrum names
try:
    from natsort import natsorted
    NATSORT_AVAILABLE = True
except ImportError:
    NATSORT_AVAILABLE = False
    # Simple natural sort implementation
    def natsorted(items, key=None):
        """Simple natural sort implementation."""
        import re
        def natural_key(text):
            if key:
                text = key(text)
            return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', str(text))]
        return sorted(items, key=natural_key)

# RDKit imports for SMILES plotting
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, rdMolDescriptors
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

            # Get molecular formula from RDKit
            molecular_formula = Chem.rdMolDescriptors.CalcMolFormula(mol)

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

            # Create title with SMILES and formula
            title_text = f"SMILES: {smiles_code[:30]}{'...' if len(smiles_code) > 30 else ''}\nFormula: {molecular_formula}"
            ax.set_title(
                title_text,
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

        # Add options for key column
        if column == "#1":  # Key column
            context_menu.add_command(
                label=f"Add '{key}' as grouping tag",
                command=lambda: self._add_key_as_grouping_tag(key),
            )
            context_menu.add_command(
                label=f"Set '{key}' as spectrum name",
                command=lambda: self._set_key_as_spectrum_name(key),
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

    def _add_key_as_grouping_tag(self, key: str):
        """Add the selected key as a grouping tag in the spectrum tree view."""
        # Find the spectrum tree view in the parent application
        root = self.winfo_toplevel()
        spectrum_tree = self._find_spectrum_tree_view(root)
        
        if spectrum_tree:
            # Get current grouping tags
            current_tags = spectrum_tree.tag_var.get().strip()
            
            # Add the new key if it's not already present
            if current_tags:
                tags = [tag.strip() for tag in current_tags.split(',')]
                if key not in tags:
                    tags.append(key)
                    new_tags = ', '.join(tags)
                else:
                    messagebox.showinfo("Already Added", f"Key '{key}' is already in grouping tags.")
                    return
            else:
                new_tags = key
            
            # Update the tag entry
            spectrum_tree.tag_var.set(new_tags)
            spectrum_tree._on_tag_entry_change()
            
        else:
            messagebox.showwarning("Error", "Could not find spectrum tree view.")

    def _set_key_as_spectrum_name(self, key: str):
        """Set the selected key as the spectrum naming scheme."""
        # Try to get the main app through the callback
        main_app = None
        
        # First try: use the callback to find the main app
        if self.on_metadata_changed and hasattr(self.on_metadata_changed, '__self__'):
            potential_app = self.on_metadata_changed.__self__
            if hasattr(potential_app, 'spectrum_name_var') and hasattr(potential_app, 'spectrum_tree'):
                main_app = potential_app
        
        # Second try: search through widget hierarchy
        if not main_app:
            main_app = self._find_main_app(self.winfo_toplevel())
        
        if main_app and hasattr(main_app, 'spectrum_name_var'):
            # Check if this key exists as a naming option
            # First, we need to ensure the key is added to available naming options
            if hasattr(main_app, 'spectrum_tree') and hasattr(main_app.spectrum_tree, 'parser'):
                parser = main_app.spectrum_tree.parser
                if parser:
                    # Add this key to the spectrum naming menu if it doesn't exist
                    self._add_naming_option_if_missing(main_app, key)
                    
                    # Set the naming scheme
                    main_app.spectrum_name_var.set(key)
                    main_app._update_spectrum_names()
                    
                else:
                    messagebox.showwarning("Error", "No data loaded.")
            else:
                messagebox.showwarning("Error", "Could not access spectrum data.")
        else:
            messagebox.showwarning("Error", "Could not find main application.")

    def _find_spectrum_tree_view(self, widget):
        """Recursively find the SpectrumTreeView widget."""
        if isinstance(widget, SpectrumTreeView):
            return widget
        
        for child in widget.winfo_children():
            result = self._find_spectrum_tree_view(child)
            if result:
                return result
        return None

    def _find_main_app(self, widget):
        """Find the main application instance."""
        # First, try to find the main app through the widget hierarchy
        current = widget
        while current:
            # Check if this widget has the main app attributes
            if hasattr(current, 'spectrum_name_var') and hasattr(current, 'spectrum_tree'):
                return current
            
            # Check if current has a reference to main app via callbacks
            if hasattr(current, 'on_metadata_changed') and current.on_metadata_changed:
                # The callback is likely bound to the main app
                try:
                    # Get the instance that the callback method is bound to
                    if hasattr(current.on_metadata_changed, '__self__'):
                        potential_app = current.on_metadata_changed.__self__
                        if hasattr(potential_app, 'spectrum_name_var') and hasattr(potential_app, 'spectrum_tree'):
                            return potential_app
                except:
                    pass
            
            # Try the parent widget
            try:
                current = current.master
            except:
                current = None
        
        # If we can't find it through hierarchy, try a different approach
        # Look through all top-level windows
        root = self.winfo_toplevel()
        try:
            # Check all children of the root window
            for child in root.winfo_children():
                if hasattr(child, 'spectrum_name_var') and hasattr(child, 'spectrum_tree'):
                    return child
                # Recursively check children
                result = self._search_for_main_app_in_children(child)
                if result:
                    return result
        except:
            pass
        
        return None

    def _search_for_main_app_in_children(self, widget):
        """Recursively search for main app in widget children."""
        try:
            if hasattr(widget, 'spectrum_name_var') and hasattr(widget, 'spectrum_tree'):
                return widget
            
            for child in widget.winfo_children():
                result = self._search_for_main_app_in_children(child)
                if result:
                    return result
        except:
            pass
        return None

    def _add_naming_option_if_missing(self, main_app, key: str):
        """Add a naming option to the spectrum name menu if it doesn't exist."""
        if hasattr(main_app, 'spectrum_name_menu'):
            menu = main_app.spectrum_name_menu
            
            # Check if the option already exists
            try:
                last_index = menu.index('end')
                if last_index is not None:
                    for i in range(last_index + 1):
                        try:
                            label = menu.entrycget(i, 'label')
                            if label == key:
                                return  # Already exists
                        except:
                            continue
                
                # Add the new option
                menu.add_radiobutton(
                    label=key,
                    variable=main_app.spectrum_name_var,
                    value=key,
                    command=main_app._update_spectrum_names,
                )
            except:
                # If there's an error with the menu, ignore it
                pass

    def get_pending_selection(self) -> Optional[List[int]]:
        """Get and clear any pending selection."""
        if self._pending_selection:
            selection = self._pending_selection
            self._pending_selection = None
            return selection
        return None

    def clear(self):
        """Clear the metadata editor and reset to empty state."""
        # Clear the metadata tree
        for item in self.metadata_tree.get_children():
            self.metadata_tree.delete(item)

        # Clear the SMILES plot
        self._show_smiles_message("No spectra selected")

        # Reset state
        self.parser = None
        self.selected_spectrum_ids = []
        self._pending_selection = None

        # Close any open editor
        self._close_editor()


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
        self.highlighted_ions: Dict[int, List[int]] = {}  # {spectrum_id: [ion_indices]}
        self.axes: List = []  # Track all subplot axes for zoom synchronization
        self._syncing_zoom = False  # Prevent infinite recursion during sync
        self.show_combined_plot = tk.BooleanVar(value=False)  # Control combined plot display
        self.ppm_tolerance = tk.DoubleVar(value=20.0)  # PPM tolerance for fragment matching
        self.top_fragments_count = tk.IntVar(value=15)  # Number of top fragments to show
        self.naming_scheme = "Numbered"  # Current spectrum naming scheme

        self._create_widgets()

    def _create_widgets(self):
        """Create the visualization widgets."""
        # Header with controls
        header_frame = ttk.Frame(self)
        header_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(
            header_frame, text="Spectrum Visualization", font=("Arial", 12, "bold")
        ).pack(side="left")

        # Controls frame
        controls_frame = ttk.Frame(header_frame)
        controls_frame.pack(side="right")

        # Combined plot checkbox
        ttk.Checkbutton(
            controls_frame,
            text="Show Combined Plot",
            variable=self.show_combined_plot,
            command=self._plot_spectra
        ).pack(side="left", padx=(0, 10))

        # PPM tolerance for fragment matching in combined plot
        ttk.Label(controls_frame, text="PPM tolerance:").pack(side="left", padx=(0, 2))
        ppm_spinbox = ttk.Spinbox(
            controls_frame,
            from_=1.0,
            to=100.0,
            increment=1.0,
            width=8,
            textvariable=self.ppm_tolerance,
            command=self._on_ppm_change
        )
        ppm_spinbox.pack(side="left", padx=(0, 10))
        ppm_spinbox.bind('<KeyRelease>', self._on_ppm_change)

        # Top fragments count for combined plot
        ttk.Label(controls_frame, text="Top fragments:").pack(side="left", padx=(0, 2))
        fragments_spinbox = ttk.Spinbox(
            controls_frame,
            from_=5,
            to=50,
            increment=1,
            width=6,
            textvariable=self.top_fragments_count,
            command=self._on_fragments_count_change
        )
        fragments_spinbox.pack(side="left", padx=(0, 10))
        fragments_spinbox.bind('<KeyRelease>', self._on_fragments_count_change)

        # Popup button
        ttk.Button(
            controls_frame,
            text="Create Popup",
            command=self._create_popup_window
        ).pack(side="left")

        # Matplotlib figure
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=(5, 0))

        # Add navigation toolbar for zoom/pan functionality
        toolbar_frame = ttk.Frame(self)
        toolbar_frame.pack(fill="x", padx=5, pady=(0, 5))
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

    def set_naming_scheme(self, naming_scheme: str):
        """Set the spectrum naming scheme."""
        self.naming_scheme = naming_scheme
        # Replot if we have data to show updated names
        if self.parser and self.selected_spectrum_ids:
            self._plot_spectra()

    def load_data(self, parser: MGFParser, selected_spectrum_ids: List[int]):
        """Load and visualize selected spectra."""
        self.parser = parser
        self.selected_spectrum_ids = selected_spectrum_ids
        self.highlighted_ions = {}  # Clear highlighted ions when loading new data
        self._plot_spectra()

    def highlight_ions(self, spectrum_id: int, ion_indices: List[int]):
        """Highlight specific ions in the spectrum visualization."""
        self.highlighted_ions[spectrum_id] = ion_indices
        self._plot_spectra()  # Replot to show highlights

    def _plot_spectra(self):
        """Plot the selected spectra."""
        self.figure.clear()
        self.axes = []  # Reset axes list

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

        if self.show_combined_plot.get() and len(selected_spectra) > 1:
            self._plot_combined_spectra(selected_spectra)
        else:
            self._plot_individual_spectra(selected_spectra)

    def _plot_individual_spectra(self, selected_spectra):
        """Plot individual spectra in separate subplots."""
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
            self.axes.append(ax)
            self._plot_single_spectrum(
                ax, selected_spectra[0], (global_mz_min, global_mz_max), is_last=True
            )
        else:
            for i, spectrum in enumerate(selected_spectra):
                is_last = i == n_spectra - 1
                ax = self.figure.add_subplot(n_spectra, 1, i + 1)
                self.axes.append(ax)
                self._plot_single_spectrum(
                    ax, spectrum, (global_mz_min, global_mz_max), is_last=is_last
                )

        # Set up zoom synchronization for multiple spectra
        if len(self.axes) > 1:
            self._setup_zoom_synchronization()

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

    def _plot_combined_spectra(self, selected_spectra):
        """Plot all selected spectra in a combined plot with fragment matching."""
        # Limit to 10 spectra for performance
        original_count = len(selected_spectra)
        if original_count > 10:
            selected_spectra = selected_spectra[:10]
            performance_warning = f"Performance limit: Showing first 10 of {original_count} selected spectra"
        else:
            performance_warning = None

        ax = self.figure.add_subplot(111)
        self.axes.append(ax)

        # Collect all fragments from all spectra with sum-scaled intensities
        all_fragments = {}  # {mz_value: [(spectrum_id, relative_intensity), ...]}
        spectrum_colors = plt.cm.tab10(np.linspace(0, 1, min(len(selected_spectra), 10)))
        spectrum_info = {}  # {spectrum_id: {'color': color, 'index': i, 'label': str}}

        for i, spectrum in enumerate(selected_spectra):
            if spectrum.ions.size == 0:
                continue

            mz_values = spectrum.ions[:, 0]
            intensity_values = spectrum.ions[:, 1]
            
            # Sum-scale intensities (normalize to sum = 1)
            total_intensity = np.sum(intensity_values)
            if total_intensity > 0:
                relative_intensities = intensity_values / total_intensity
            else:
                relative_intensities = intensity_values

            spectrum_info[spectrum.spectrum_id] = {
                'color': spectrum_colors[i],
                'index': i,
                'label': self._get_spectrum_display_name(spectrum)
            }

            # Add fragments to collection
            for mz, rel_intensity in zip(mz_values, relative_intensities):
                if mz not in all_fragments:
                    all_fragments[mz] = []
                all_fragments[mz].append((spectrum.spectrum_id, rel_intensity))

        # Group fragments by similar m/z values within PPM tolerance
        fragment_groups = self._group_fragments_by_mz(all_fragments, self.ppm_tolerance.get())

        # Calculate total intensity for each fragment group and sort by intensity
        fragment_intensities = []
        for group_mz, fragments_in_group in fragment_groups.items():
            if len(fragments_in_group) > 1:  # Only consider fragments present in multiple spectra
                total_intensity = sum(rel_intensity for _, rel_intensity in fragments_in_group)
                fragment_intensities.append((total_intensity, group_mz, fragments_in_group))
        
        # Sort by total intensity (descending) and take top N
        top_fragments_count = self.top_fragments_count.get()
        fragment_intensities.sort(key=lambda x: x[0], reverse=True)
        top_fragments = fragment_intensities[:top_fragments_count]

        # Create a plot showing spectra on x-axis and relative abundance on y-axis
        # Each line represents a fragment group (similar m/z values)
        
        # Sort spectrum IDs by their display names using natural sorting
        sorted_spec_ids = list(spectrum_info.keys())
        try:
            sorted_spec_ids = natsorted(sorted_spec_ids, 
                                      key=lambda spec_id: self._get_spectrum_display_name(spec_id))
        except NameError:
            # Fallback to regular sorting if natsort is not available
            sorted_spec_ids = sorted(sorted_spec_ids, 
                                   key=lambda spec_id: self._get_spectrum_display_name(spec_id))
        
        spectrum_positions = {spec_id: i for i, spec_id in enumerate(sorted_spec_ids)}
        spectrum_labels = [self._get_spectrum_display_name(spec_id) for spec_id in sorted_spec_ids]

        # Plot each top fragment group as a line connecting spectra
        for total_intensity, group_mz, fragments_in_group in top_fragments:
            # Create arrays for plotting
            x_positions = []
            y_intensities = []
            
            # Sort fragments by spectrum index for consistent line drawing
            sorted_fragments = sorted(fragments_in_group, 
                                    key=lambda x: spectrum_info.get(x[0], {}).get('index', 999))
            
            for spectrum_id, rel_intensity in sorted_fragments:
                if spectrum_id in spectrum_info:
                    x_positions.append(spectrum_positions[spectrum_id])
                    y_intensities.append(rel_intensity)
            
            if len(x_positions) > 1:
                # Plot line connecting the same fragment across spectra
                ax.plot(x_positions, y_intensities, 'o-', alpha=0.7, linewidth=2, 
                       markersize=6, label=f'm/z {group_mz:.4f}')

        # Set x-axis to show spectrum names
        ax.set_xticks(range(len(spectrum_labels)))
        ax.set_xticklabels(spectrum_labels, rotation=45, ha='right')
        ax.set_xlabel("Spectra")
        ax.set_ylabel("Relative Abundance (Sum-scaled)")
        ax.set_title(f"Combined Spectrum Plot - Fragment Matching ({len(selected_spectra)} spectra)")
        ax.grid(True, alpha=0.3)

        # Add legend for fragment m/z values
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            legend_title = f"Top {min(len(handles), top_fragments_count)} fragments (by intensity)"
            ax.legend(loc='upper right', framealpha=0.9, fontsize=8, title=legend_title)

        # Add performance warning if applicable
        if performance_warning:
            ax.text(0.02, 0.98, performance_warning, transform=ax.transAxes, 
                   fontsize=10, color="red", verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        self.figure.tight_layout()
        self.canvas.draw()

    def _group_fragments_by_mz(self, all_fragments, ppm_tolerance):
        """Group fragments by similar m/z values within PPM tolerance."""
        fragment_groups = {}
        sorted_mz_values = sorted(all_fragments.keys())
        
        for mz in sorted_mz_values:
            # Find if this m/z belongs to an existing group
            group_found = False
            for group_mz in fragment_groups:
                # Calculate PPM difference
                ppm_diff = abs(mz - group_mz) / group_mz * 1e6
                if ppm_diff <= ppm_tolerance:
                    # Add to existing group
                    fragment_groups[group_mz].extend(all_fragments[mz])
                    group_found = True
                    break
            
            if not group_found:
                # Create new group
                fragment_groups[mz] = all_fragments[mz][:]
        
        return fragment_groups

    def _on_ppm_change(self, event=None):
        """Handle PPM tolerance change."""
        if self.show_combined_plot.get():
            self._plot_spectra()

    def _on_fragments_count_change(self, event=None):
        """Handle top fragments count change."""
        if self.show_combined_plot.get():
            self._plot_spectra()

    def _get_spectrum_display_name(self, spectrum_or_id):
        """Get the display name for a spectrum based on the current naming scheme."""
        # Handle both spectrum objects and spectrum IDs
        if isinstance(spectrum_or_id, (str, int)):
            # It's a spectrum ID, find the spectrum object
            spectrum_id = spectrum_or_id
            if not self.parser:
                return f"S {spectrum_id}"
            
            spectrum = next(
                (s for s in self.parser.spectra if s.spectrum_id == spectrum_id), None
            )
            if not spectrum:
                return f"S {spectrum_id}"
        else:
            # It's already a spectrum object
            spectrum = spectrum_or_id
            spectrum_id = spectrum.spectrum_id
        
        # Use the local naming scheme
        if self.naming_scheme == "Numbered":
            return f"S {spectrum_id}"
        else:
            # Use the metadata value for the naming key
            name_value = spectrum.get_metadata_value(self.naming_scheme)
            if name_value:
                return str(name_value)
            else:
                return f"S {spectrum_id}"  # Fall back to numbered if key not found

    def _create_popup_window(self):
        """Create a popup window with current selected spectra and metadata."""
        if not self.parser or not self.selected_spectrum_ids:
            messagebox.showwarning("No Selection", "No spectra selected for popup.")
            return

        # Get the root window from the widget hierarchy
        root = self.winfo_toplevel()
        popup = SpectrumPopupWindow(root, self.parser, self.selected_spectrum_ids, self.naming_scheme)
        popup.show()

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

        # Get highlighted ions for this spectrum
        highlighted_indices = self.highlighted_ions.get(spectrum.spectrum_id, [])

        # Create stick plot with different colors for highlighted vs normal ions
        for i, (mz, intensity) in enumerate(zip(mz_values, intensity_values)):
            color = "green" if i in highlighted_indices else "blue"
            linewidth = 2.0 if i in highlighted_indices else 1.5
            ax.vlines(mz, 0, intensity, colors=color, linewidth=linewidth)

        # Add precursor mass line if available
        precursor_mass = self._get_precursor_mass_from_spectrum(spectrum)
        if precursor_mass is not None:
            # Check if precursor mass is within the current plot range
            x_min, x_max = ax.get_xlim()
            if mz_limits:
                x_min, x_max = mz_limits
            elif len(mz_values) > 0:
                x_min = mz_values.min() * 0.95
                x_max = mz_values.max() * 1.05

            if x_min <= precursor_mass <= x_max:
                y_max = intensity_values.max() if len(intensity_values) > 0 else 1
                ax.axvline(
                    precursor_mass,
                    color="grey",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                    label=f"Precursor: {precursor_mass:.4f}",
                )
                # Add a small text label at the top
                ax.text(
                    precursor_mass,
                    y_max * 1.05,
                    f"M: {precursor_mass:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="grey",
                    rotation=90,
                )

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

    def _get_precursor_mass_from_spectrum(self, spectrum):
        """Extract precursor mass from spectrum metadata."""
        # Common precursor mass key names to check
        mass_keys = [
            "pepmass",
            "PEPMASS",
            "precursor_mass",
            "PRECURSOR_MASS",
            "precursormass",
        ]

        for key in mass_keys:
            value = spectrum.get_metadata_value(key)
            if value:
                try:
                    # Handle different formats like "123.456" or "123.456 2" (mass and charge)
                    mass_str = value.strip().split()[0]  # Take first part (mass)
                    return float(mass_str)
                except (ValueError, IndexError):
                    continue
        return None

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

    def _setup_zoom_synchronization(self):
        """Set up zoom synchronization between all subplot axes."""
        for ax in self.axes:
            ax.callbacks.connect("xlim_changed", self._on_xlims_change)

    def _on_xlims_change(self, ax):
        """Handle x-axis limit changes to synchronize zoom across all subplots."""
        if self._syncing_zoom:
            return  # Prevent infinite recursion

        self._syncing_zoom = True
        try:
            # Get the new x limits from the changed axis
            xlims = ax.get_xlim()

            # Apply the same x limits to all other axes
            for other_ax in self.axes:
                if other_ax != ax:
                    other_ax.set_xlim(xlims)

            # Redraw the canvas
            self.canvas.draw_idle()
        finally:
            self._syncing_zoom = False


class IonDataTable(ttk.Frame):
    """Component for displaying ion data in table format."""

    def __init__(self, parent):
        super().__init__(parent)
        self.parser: Optional[MGFParser] = None
        self.selected_spectrum_ids: List[int] = []
        self.spectrum_viz_callback = None  # Callback to update spectrum visualization
        self.table_data = {}  # Store data for sorting: {tab_id: [(ion_index, mz, intensity, annotations), ...]}

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

    def set_spectrum_viz_callback(self, callback):
        """Set callback function to update spectrum visualization when ions are selected."""
        self.spectrum_viz_callback = callback

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
        tab_text = f"S {spectrum.spectrum_id}"
        self.notebook.add(frame, text=tab_text)

        # Create treeview with additional columns for annotations
        columns = ("Index", "m/z", "Intensity", "Rel. Intensity %", "Annotations")
        tree = ttk.Treeview(
            frame, columns=columns, show="headings", height=15, selectmode="extended"
        )

        # Configure columns with sorting callbacks
        tree.heading(
            "Index",
            text="Index ↕",
            command=lambda: self._sort_table(tree, spectrum, "Index"),
        )
        tree.column("Index", width=80)

        tree.heading(
            "m/z", text="m/z ↕", command=lambda: self._sort_table(tree, spectrum, "m/z")
        )
        tree.column("m/z", width=120)

        tree.heading(
            "Intensity",
            text="Intensity ↕",
            command=lambda: self._sort_table(tree, spectrum, "Intensity"),
        )
        tree.column("Intensity", width=120)

        tree.heading(
            "Rel. Intensity %",
            text="Rel. Intensity % ↕",
            command=lambda: self._sort_table(tree, spectrum, "Rel. Intensity %"),
        )
        tree.column("Rel. Intensity %", width=120)

        tree.heading(
            "Annotations",
            text="Annotations (Formula [ppm]) ↕",
            command=lambda: self._sort_table(tree, spectrum, "Annotations"),
        )
        tree.column("Annotations", width=300)

        # Bind selection event
        tree.bind(
            "<<TreeviewSelect>>", lambda event: self._on_ion_selection(event, spectrum)
        )

        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        h_scrollbar = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)

        tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        tree.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        # Store reference to tree for later access
        frame.tree = tree
        frame.spectrum = spectrum

        # Populate with ion data and store for sorting
        self._populate_table_data(tree, spectrum)

        # Populate with ion data and store for sorting
        self._populate_table_data(tree, spectrum)

    def _populate_table_data(self, tree, spectrum):
        """Populate table with ion data and store for sorting."""
        # Clear existing items
        for item in tree.get_children():
            tree.delete(item)

        if spectrum.ions.size == 0:
            return

        # Get precursor mass for comparison
        precursor_mass = self._get_precursor_mass(spectrum)

        # Calculate total intensity for relative percentage calculation
        total_intensity = (
            float(spectrum.ions[:, 1].sum()) if spectrum.ions.size > 0 else 1.0
        )

        # Store data for this table
        table_id = id(tree)
        self.table_data[table_id] = []

        for i, (mz, intensity) in enumerate(spectrum.ions):
            # Calculate relative intensity percentage
            rel_intensity_percent = (intensity / total_intensity) * 100.0

            # Get fragment annotations for this ion
            annotations = spectrum.get_fragment_annotations(i)

            # Format annotations (sorted by ppm error)
            annotation_text = ""
            if annotations:
                # Sort by ppm error
                sorted_annotations = sorted(annotations, key=lambda x: x["ppm_error"])
                annotation_strings = []
                for ann in sorted_annotations[:3]:  # Show only top 3 matches
                    annotation_strings.append(
                        f"{ann['formula']} [{ann['ppm_error']:.1f}]"
                    )
                annotation_text = "; ".join(annotation_strings)
                if len(annotations) > 3:
                    annotation_text += f" (+{len(annotations) - 3} more)"

            # Check if this fragment is close to precursor mass
            precursor_indicator = ""
            if precursor_mass and abs(mz - precursor_mass) < 0.1:  # Within 0.1 Da
                precursor_indicator = " [M+H]+"
            elif (
                precursor_mass and abs(mz - (precursor_mass - 1.007825)) < 0.1
            ):  # M+ (no proton)
                precursor_indicator = " [M]+"

            # Format m/z with precursor indicator
            mz_text = f"{mz:.6f}{precursor_indicator}"

            # Store raw data for sorting
            self.table_data[table_id].append(
                {
                    "ion_index": i,
                    "index_display": i + 1,
                    "mz_value": mz,
                    "mz_display": mz_text,
                    "intensity": intensity,
                    "intensity_display": f"{intensity:.3f}",
                    "rel_intensity_percent": rel_intensity_percent,
                    "rel_intensity_display": f"{rel_intensity_percent:.2f}%",
                    "annotations": annotation_text,
                    "precursor_indicator": precursor_indicator,
                }
            )

            # Insert into tree
            tree.insert(
                "",
                "end",
                values=(
                    i + 1,
                    mz_text,
                    f"{intensity:.3f}",
                    f"{rel_intensity_percent:.2f}%",
                    annotation_text,
                ),
                tags=(f"ion_{i}",),  # Tag with ion index for highlighting
            )

    def _sort_table(self, tree, spectrum, column):
        """Sort table by the specified column."""
        table_id = id(tree)
        if table_id not in self.table_data:
            return

        data = self.table_data[table_id]

        # Determine sort key based on column
        if column == "Index":
            sort_key = lambda x: x["index_display"]
        elif column == "m/z":
            sort_key = lambda x: x["mz_value"]
        elif column == "Intensity":
            sort_key = lambda x: x["intensity"]
        elif column == "Rel. Intensity %":
            sort_key = lambda x: x["rel_intensity_percent"]
        elif column == "Annotations":
            sort_key = lambda x: x["annotations"]
        else:
            return

        # Check current sort direction (stored as attribute on tree)
        current_sort = getattr(tree, "_sort_column", None)
        reverse = False
        if current_sort == column:
            reverse = not getattr(tree, "_sort_reverse", False)

        # Store sort state
        tree._sort_column = column
        tree._sort_reverse = reverse

        # Sort data
        data.sort(key=sort_key, reverse=reverse)

        # Update column headers to show sort direction
        for col in ["Index", "m/z", "Intensity", "Rel. Intensity %", "Annotations"]:
            if col == column:
                direction = "↓" if reverse else "↑"
                if col == "Annotations":
                    tree.heading(col, text=f"Annotations (Formula [ppm]) {direction}")
                elif col == "Rel. Intensity %":
                    tree.heading(col, text=f"Rel. Intensity % {direction}")
                else:
                    tree.heading(col, text=f"{col} {direction}")
            else:
                if col == "Annotations":
                    tree.heading(col, text="Annotations (Formula [ppm]) ↕")
                elif col == "Rel. Intensity %":
                    tree.heading(col, text="Rel. Intensity % ↕")
                else:
                    tree.heading(col, text=f"{col} ↕")
                    tree.heading(col, text=f"{col} ↕")

        # Clear and repopulate tree
        for item in tree.get_children():
            tree.delete(item)

        for row_data in data:
            tree.insert(
                "",
                "end",
                values=(
                    row_data["index_display"],
                    row_data["mz_display"],
                    row_data["intensity_display"],
                    row_data["rel_intensity_display"],
                    row_data["annotations"],
                ),
                tags=(f"ion_{row_data['ion_index']}",),
            )

    def _on_ion_selection(self, event, spectrum):
        """Handle ion selection in the table."""
        tree = event.widget
        selected_items = tree.selection()

        # Extract ion indices from selected items
        selected_ion_indices = []
        for item in selected_items:
            tags = tree.item(item)["tags"]
            for tag in tags:
                if tag.startswith("ion_"):
                    ion_index = int(tag.split("_")[1])
                    selected_ion_indices.append(ion_index)

        # Update spectrum visualization if callback is set
        if self.spectrum_viz_callback:
            self.spectrum_viz_callback(spectrum.spectrum_id, selected_ion_indices)

    def _get_precursor_mass(self, spectrum):
        """Extract precursor mass from spectrum metadata."""
        # Common precursor mass key names to check
        mass_keys = [
            "pepmass",
            "PEPMASS",
            "precursor_mass",
            "PRECURSOR_MASS",
            "precursormass",
        ]

        for key in mass_keys:
            value = spectrum.get_metadata_value(key)
            if value:
                try:
                    # Handle different formats like "123.456" or "123.456 2" (mass and charge)
                    mass_str = value.strip().split()[0]  # Take first part (mass)
                    return float(mass_str)
                except (ValueError, IndexError):
                    continue
        return None

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
        self.canvas.get_tk_widget().pack(fill="both", expand=True, pady=(0, 5))

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


class SmartsFilterDialog:
    """Dialog for SMARTS substructure filtering."""

    def __init__(self, parent, spectra, apply_callback):
        self.parent = parent
        self.spectra = spectra
        self.apply_callback = apply_callback
        self.matching_spectra = []
        self.unique_smiles = []
        self.smiles_to_spectra = {}

        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("SMARTS Substructure Filter")
        self.dialog.geometry("1200x800")
        self.dialog.resizable(True, True)
        self.dialog.grab_set()  # Make dialog modal

        self._create_widgets()

    def _create_widgets(self):
        """Create dialog widgets."""
        # Main frame
        main_frame = ttk.Frame(self.dialog, padding=10)
        main_frame.pack(fill="both", expand=True)

        # SMARTS input section
        input_frame = ttk.LabelFrame(main_frame, text="SMARTS Pattern", padding=5)
        input_frame.pack(fill="x", pady=(0, 10))

        # SMARTS entry
        ttk.Label(input_frame, text="Enter SMARTS pattern(s):").pack(anchor="w")
        ttk.Label(
            input_frame,
            text="Use ' ' to separate multiple patterns for OR logic (e.g., 'c1ccccc1 C=O N')",
            font=("Arial", 9),
            foreground="gray",
        ).pack(anchor="w", pady=(0, 5))
        self.smarts_var = tk.StringVar()
        self.smarts_entry = ttk.Entry(
            input_frame, textvariable=self.smarts_var, font=("Courier", 12)
        )
        self.smarts_entry.pack(fill="x", pady=(5, 10))
        self.smarts_entry.bind("<KeyRelease>", self._on_smarts_change)

        # Example patterns
        examples_frame = ttk.Frame(input_frame)
        examples_frame.pack(fill="x")
        ttk.Label(examples_frame, text="Examples:").pack(anchor="w")

        example_buttons_frame = ttk.Frame(examples_frame)
        example_buttons_frame.pack(fill="x", pady=5)

        examples = [
            ("Benzene ring", "c1ccccc1"),
            ("Carbonyl", "C=O"),
            ("Hydroxyl", "O"),
            ("Ester", "C(=O)O"),
            (
                "Flavone or Iso-Flavone",
                "[O,o]~[C,c]~1~[C,c]~[C,c](~[O,o]~[C,c]2~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~1~2)~[C,c]~3~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]3  [C,c]~1~[C,c]~[C,c](~[O,o]~[C,c]2~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~1~2)~[C,c]~3~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]3  [O,o]~[C,c]~1~[C,c]~2~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~2~[O,o]~[C,c]~[C,c]~1[C,c]~3~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~3  [C,c]~1~[C,c]~2~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~2~[O,o]~[C,c]~[C,c]~1[C,c]~3~[C,c]~[C,c]~[C,c]~[C,c]~[C,c]~3",
            ),
        ]

        for i, (name, pattern) in enumerate(examples):
            btn = ttk.Button(
                example_buttons_frame,
                text=f"{name}\n({pattern})",
                command=lambda p=pattern: self._set_smarts_pattern(p),
                width=12,
            )
            btn.grid(row=0, column=i, padx=2, sticky="ew")

        # Configure column weights for even distribution
        for i in range(len(examples)):
            example_buttons_frame.columnconfigure(i, weight=1)

        # SMARTS visualization frame
        viz_frame = ttk.LabelFrame(main_frame, text="Pattern Visualization", padding=5)
        viz_frame.pack(fill="x", pady=(0, 10))

        self.pattern_canvas = tk.Canvas(viz_frame, height=150, bg="white")
        self.pattern_canvas.pack(fill="x")

        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="Filtering Results", padding=5)
        results_frame.pack(fill="both", expand=True, pady=(0, 10))

        # Create paned window for matched/unmatched structures
        paned_window = ttk.PanedWindow(results_frame, orient="horizontal")
        paned_window.pack(fill="both", expand=True)

        # Matched structures frame
        matched_frame = ttk.LabelFrame(
            paned_window, text="Matched Structures (0)", padding=5
        )
        paned_window.add(matched_frame, weight=1)

        # Create scrollable frame for matched structures
        self.matched_canvas = tk.Canvas(matched_frame, bg="white")
        matched_scrollbar = ttk.Scrollbar(
            matched_frame, orient="vertical", command=self.matched_canvas.yview
        )
        self.matched_scrollable_frame = ttk.Frame(self.matched_canvas)

        self.matched_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.matched_canvas.configure(
                scrollregion=self.matched_canvas.bbox("all")
            ),
        )

        self.matched_canvas.create_window(
            (0, 0), window=self.matched_scrollable_frame, anchor="nw"
        )
        self.matched_canvas.configure(yscrollcommand=matched_scrollbar.set)

        self.matched_canvas.pack(side="left", fill="both", expand=True)
        matched_scrollbar.pack(side="right", fill="y")

        # Unmatched structures frame
        unmatched_frame = ttk.LabelFrame(
            paned_window, text="Unmatched Structures (0)", padding=5
        )
        paned_window.add(unmatched_frame, weight=1)

        # Create scrollable frame for unmatched structures
        self.unmatched_canvas = tk.Canvas(unmatched_frame, bg="white")
        unmatched_scrollbar = ttk.Scrollbar(
            unmatched_frame, orient="vertical", command=self.unmatched_canvas.yview
        )
        self.unmatched_scrollable_frame = ttk.Frame(self.unmatched_canvas)

        self.unmatched_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.unmatched_canvas.configure(
                scrollregion=self.unmatched_canvas.bbox("all")
            ),
        )

        self.unmatched_canvas.create_window(
            (0, 0), window=self.unmatched_scrollable_frame, anchor="nw"
        )
        self.unmatched_canvas.configure(yscrollcommand=unmatched_scrollbar.set)

        self.unmatched_canvas.pack(side="left", fill="both", expand=True)
        unmatched_scrollbar.pack(side="right", fill="y")

        # Bind mouse wheel to canvases
        self._bind_mousewheel(self.matched_canvas)
        self._bind_mousewheel(self.unmatched_canvas)

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(10, 0))

        ttk.Button(button_frame, text="Apply Filter", command=self._apply_filter).pack(
            side="right", padx=(5, 0)
        )
        ttk.Button(
            button_frame, text="Generate Overview", command=self._generate_overview
        ).pack(side="right", padx=(5, 0))
        ttk.Button(button_frame, text="Cancel", command=self._cancel).pack(side="right")

        # Status label
        self.status_var = tk.StringVar(
            value="Enter a SMARTS pattern and click 'Generate Overview' to begin"
        )
        ttk.Label(button_frame, textvariable=self.status_var).pack(side="left")

    def _bind_mousewheel(self, canvas):
        """Bind mouse wheel scrolling to canvas."""

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_to_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_from_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", _bind_to_mousewheel)
        canvas.bind("<Leave>", _unbind_from_mousewheel)

    def _set_smarts_pattern(self, pattern):
        """Set SMARTS pattern from example button."""
        self.smarts_var.set(pattern)
        self._on_smarts_change()
        # Automatically generate overview when using example patterns
        self._generate_overview()

    def _on_smarts_change(self, event=None):
        """Handle SMARTS pattern change."""
        pattern = self.smarts_var.get().strip()

        if not pattern:
            self._clear_visualization()
            return

        # Only visualize the SMARTS pattern, don't perform filtering yet
        self._visualize_smarts_pattern(pattern)

        # Update status to prompt user to generate overview
        self.status_var.set("Pattern loaded. Click 'Generate Overview' to see matches.")

    def _generate_overview(self):
        """Generate overview of matching and non-matching structures."""
        pattern = self.smarts_var.get().strip()

        if not pattern:
            messagebox.showwarning("No Pattern", "Please enter a SMARTS pattern first.")
            return

        # Perform filtering and display results
        self._perform_filtering(pattern)

    def _visualize_smarts_pattern(self, pattern):
        """Visualize the SMARTS pattern(s)."""
        self.pattern_canvas.delete("all")

        if not RDKIT_AVAILABLE:
            self.pattern_canvas.create_text(
                150, 75, text="RDKit not available", font=("Arial", 12), fill="red"
            )
            return

        try:
            # Split patterns by $$$
            patterns = [p.strip() for p in pattern.split(" ") if p.strip()]

            if not patterns:
                self.pattern_canvas.create_text(
                    150,
                    75,
                    text="No valid patterns found",
                    font=("Arial", 12),
                    fill="red",
                )
                return

            # Get canvas dimensions
            canvas_width = self.pattern_canvas.winfo_width()
            if canvas_width <= 1:  # Canvas not yet drawn
                canvas_width = 800  # Assume larger width for multiple patterns
            canvas_height = 150

            # Calculate layout for multiple patterns
            num_patterns = len(patterns)
            if num_patterns == 1:
                # Single pattern - center it
                pattern_width = 300
                x_positions = [canvas_width // 2]
            else:
                # Multiple patterns - distribute them
                pattern_width = min(250, (canvas_width - 40) // num_patterns)
                spacing = canvas_width / (num_patterns + 1)
                x_positions = [int(spacing * (i + 1)) for i in range(num_patterns)]

            # Generate and display each pattern
            images = []
            for i, single_pattern in enumerate(patterns):
                try:
                    mol = Chem.MolFromSmarts(single_pattern)
                    if mol is None:
                        # Show error for this specific pattern
                        self.pattern_canvas.create_text(
                            x_positions[i],
                            40,
                            text=f"Invalid:\n{single_pattern[:15]}...",
                            font=("Arial", 9),
                            fill="red",
                            justify="center",
                        )
                        continue

                    # Generate image
                    img = Draw.MolToImage(mol, size=(pattern_width, 100))

                    # Convert PIL image to PhotoImage
                    from PIL import ImageTk

                    photo = ImageTk.PhotoImage(img)
                    images.append(photo)  # Keep reference

                    # Display image
                    self.pattern_canvas.create_image(x_positions[i], 60, image=photo)

                    # Add pattern text below image
                    pattern_text = (
                        single_pattern
                        if len(single_pattern) <= 15
                        else single_pattern[:12] + "..."
                    )
                    self.pattern_canvas.create_text(
                        x_positions[i],
                        120,
                        text=pattern_text,
                        font=("Courier", 8),
                        justify="center",
                    )

                except Exception as e:
                    # Show error for this specific pattern
                    self.pattern_canvas.create_text(
                        x_positions[i],
                        60,
                        text=f"Error:\n{str(e)[:20]}",
                        font=("Arial", 9),
                        fill="red",
                        justify="center",
                    )

            # Store images to prevent garbage collection
            self.pattern_canvas.images = images

            # Add OR indicator if multiple patterns
            if num_patterns > 1:
                for i in range(num_patterns - 1):
                    or_x = (x_positions[i] + x_positions[i + 1]) // 2
                    self.pattern_canvas.create_text(
                        or_x, 75, text="OR", font=("Arial", 12, "bold"), fill="blue"
                    )

        except Exception as e:
            self.pattern_canvas.create_text(
                150, 75, text=f"Error: {str(e)}", font=("Arial", 10), fill="red"
            )

    def _clear_visualization(self):
        """Clear pattern visualization."""
        self.pattern_canvas.delete("all")

        # Clear results
        for widget in self.matched_scrollable_frame.winfo_children():
            widget.destroy()
        for widget in self.unmatched_scrollable_frame.winfo_children():
            widget.destroy()

        self.status_var.set(
            "Enter a SMARTS pattern and click 'Generate Overview' to begin"
        )

    def _perform_filtering(self, pattern):
        """Perform SMARTS filtering on spectra with support for multiple patterns (OR logic)."""
        if not RDKIT_AVAILABLE:
            return

        try:
            # Split patterns by $$$
            patterns = [p.strip() for p in pattern.split(" ") if p.strip()]

            if not patterns:
                self.status_var.set("No valid patterns found")
                return

            # Parse all SMARTS patterns
            smarts_mols = []
            valid_patterns = []

            for single_pattern in patterns:
                smarts_mol = Chem.MolFromSmarts(single_pattern)
                if smarts_mol is not None:
                    smarts_mols.append(smarts_mol)
                    valid_patterns.append(single_pattern)
                else:
                    self.status_var.set(f"Invalid SMARTS pattern: {single_pattern}")
                    return

            if not smarts_mols:
                self.status_var.set("No valid SMARTS patterns found")
                return

            # Collect unique SMILES from spectra
            smiles_to_spectra = {}
            unique_smiles = []

            for spectrum in self.spectra:
                smiles = (
                    spectrum.metadata.get("SMILES")
                    if "SMILES" in spectrum.metadata
                    else spectrum.metadata.get("smiles")
                    if "smiles" in spectrum.metadata
                    else ""
                )
                if not smiles:
                    continue

                if smiles not in smiles_to_spectra:
                    smiles_to_spectra[smiles] = []
                    unique_smiles.append(smiles)
                smiles_to_spectra[smiles].append(spectrum)

            if not unique_smiles:
                self.status_var.set("No SMILES found in spectra")
                return

            # Test each unique SMILES against all patterns (OR logic)
            matched_smiles = []
            unmatched_smiles = []

            for smiles in unique_smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Check if ANY of the SMARTS patterns match (OR logic)
                    has_match = any(
                        mol.HasSubstructMatch(smarts_mol) for smarts_mol in smarts_mols
                    )
                    if has_match:
                        matched_smiles.append(smiles)
                    else:
                        unmatched_smiles.append(smiles)
                else:
                    unmatched_smiles.append(smiles)

            # Update display
            self._display_results(matched_smiles, unmatched_smiles, smiles_to_spectra)

            # Store results
            self.matching_spectra = []
            for smiles in matched_smiles:
                self.matching_spectra.extend(smiles_to_spectra[smiles])

            # Update status
            total_matched_spectra = sum(
                len(smiles_to_spectra[smiles]) for smiles in matched_smiles
            )
            pattern_count = len(valid_patterns)
            pattern_text = f"{pattern_count} pattern{'s' if pattern_count > 1 else ''}"
            self.status_var.set(
                f"Found {len(matched_smiles)} matching structures using {pattern_text} "
                f"({total_matched_spectra} spectra)"
            )

        except Exception as e:
            self.status_var.set(f"Error during filtering: {str(e)}")

    def _display_results(self, matched_smiles, unmatched_smiles, smiles_to_spectra):
        """Display matched and unmatched structures."""
        # Clear previous results
        for widget in self.matched_scrollable_frame.winfo_children():
            widget.destroy()
        for widget in self.unmatched_scrollable_frame.winfo_children():
            widget.destroy()

        # Update frame titles
        total_matched_spectra = sum(
            len(smiles_to_spectra[smiles]) for smiles in matched_smiles
        )
        total_unmatched_spectra = sum(
            len(smiles_to_spectra[smiles]) for smiles in unmatched_smiles
        )

        matched_frame = self.matched_canvas.master
        unmatched_frame = self.unmatched_canvas.master
        matched_frame.configure(
            text=f"Matched Structures ({len(matched_smiles)} unique, {total_matched_spectra} spectra)"
        )
        unmatched_frame.configure(
            text=f"Unmatched Structures ({len(unmatched_smiles)} unique, {total_unmatched_spectra} spectra)"
        )

        # Display matched structures
        self._display_smiles_grid(
            self.matched_scrollable_frame, matched_smiles, smiles_to_spectra, True
        )

        # Display unmatched structures
        self._display_smiles_grid(
            self.unmatched_scrollable_frame, unmatched_smiles, smiles_to_spectra, False
        )

    def _display_smiles_grid(
        self, parent_frame, smiles_list, smiles_to_spectra, highlight_match
    ):
        """Display SMILES structures in a grid layout."""
        if not RDKIT_AVAILABLE:
            return

        # Create grid of structures (6 per row)
        for i, smiles in enumerate(smiles_list):
            row = i // 6
            col = i % 6

            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                # Create frame for this structure
                struct_frame = ttk.Frame(parent_frame, padding=2)
                struct_frame.grid(row=row, column=col, padx=2, pady=2, sticky="nsew")

                # Generate structure image
                img_size = (150, 150)
                if highlight_match and hasattr(self, "smarts_var"):
                    # Highlight substructure match for multiple patterns
                    pattern = self.smarts_var.get().strip()
                    if pattern:
                        try:
                            # Split patterns by $$$
                            patterns = [
                                p.strip() for p in pattern.split(" ") if p.strip()
                            ]
                            highlight_atoms = set()

                            # Collect all matching atoms from all patterns
                            for single_pattern in patterns:
                                smarts_mol = Chem.MolFromSmarts(single_pattern)
                                if smarts_mol is not None:
                                    match = mol.GetSubstructMatch(smarts_mol)
                                    if match:
                                        highlight_atoms.update(match)

                            if highlight_atoms:
                                img = Draw.MolToImage(
                                    mol,
                                    size=img_size,
                                    highlightAtoms=list(highlight_atoms),
                                )
                            else:
                                img = Draw.MolToImage(mol, size=img_size)
                        except:
                            img = Draw.MolToImage(mol, size=img_size)
                    else:
                        img = Draw.MolToImage(mol, size=img_size)
                else:
                    img = Draw.MolToImage(mol, size=img_size)

                # Convert to PhotoImage
                from PIL import ImageTk

                photo = ImageTk.PhotoImage(img)

                # Create label with image
                img_label = ttk.Label(struct_frame, image=photo)
                img_label.image = photo  # Keep reference
                img_label.pack()

                # Add click binding to show enlarged structure
                img_label.bind(
                    "<Button-1>",
                    lambda e,
                    s=smiles,
                    h=highlight_match: self._show_enlarged_structure(s, h),
                )
                img_label.configure(
                    cursor="hand2"
                )  # Change cursor to indicate clickable

                # Add SMILES text (truncated if too long)
                smiles_text = smiles if len(smiles) <= 20 else smiles[:17] + "..."
                ttk.Label(struct_frame, text=smiles_text, font=("Courier", 8)).pack()

                # Add spectrum count
                spectrum_count = len(smiles_to_spectra[smiles])
                ttk.Label(
                    struct_frame, text=f"({spectrum_count} spectra)", font=("Arial", 8)
                ).pack()

            except Exception as e:
                # Create error frame
                struct_frame = ttk.Frame(parent_frame, padding=2)
                struct_frame.grid(row=row, column=col, padx=2, pady=2, sticky="nsew")
                ttk.Label(struct_frame, text="Error", foreground="red").pack()
                ttk.Label(struct_frame, text=str(e)[:20], font=("Arial", 8)).pack()

        # Configure column weights
        for col in range(6):
            parent_frame.columnconfigure(col, weight=1)

    def _show_enlarged_structure(self, smiles, highlight_match):
        """Show an enlarged view of the structure in a popup window."""
        if not RDKIT_AVAILABLE:
            return

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                messagebox.showerror("Error", "Cannot parse SMILES structure")
                return

            # Create popup window
            popup = tk.Toplevel(self.dialog)
            popup.title(f"Structure View - {smiles}")
            popup.geometry("600x700")
            popup.resizable(True, True)
            popup.grab_set()  # Make modal

            # Main frame
            main_frame = ttk.Frame(popup, padding=10)
            main_frame.pack(fill="both", expand=True)

            # Title
            ttk.Label(
                main_frame, text="Molecular Structure", font=("Arial", 14, "bold")
            ).pack(pady=(0, 10))

            # SMILES text
            smiles_frame = ttk.LabelFrame(main_frame, text="SMILES", padding=5)
            smiles_frame.pack(fill="x", pady=(0, 10))

            # Create a text widget for SMILES so it's selectable
            smiles_text = tk.Text(
                smiles_frame, height=2, wrap="word", font=("Courier", 10)
            )
            smiles_text.insert("1.0", smiles)
            smiles_text.config(state="disabled")  # Make read-only
            smiles_text.pack(fill="x")

            # Structure frame
            struct_frame = ttk.LabelFrame(main_frame, text="Structure", padding=5)
            struct_frame.pack(fill="both", expand=True, pady=(0, 10))

            # Generate large structure image
            img_size = (500, 400)
            if highlight_match and hasattr(self, "smarts_var"):
                # Highlight substructure match for multiple patterns
                pattern = self.smarts_var.get().strip()
                if pattern:
                    try:
                        # Split patterns by $$$
                        patterns = [p.strip() for p in pattern.split(" ") if p.strip()]
                        highlight_atoms = set()

                        # Collect all matching atoms from all patterns
                        for single_pattern in patterns:
                            smarts_mol = Chem.MolFromSmarts(single_pattern)
                            if smarts_mol is not None:
                                match = mol.GetSubstructMatch(smarts_mol)
                                if match:
                                    highlight_atoms.update(match)

                        if highlight_atoms:
                            img = Draw.MolToImage(
                                mol,
                                size=img_size,
                                highlightAtoms=list(highlight_atoms),
                            )
                        else:
                            img = Draw.MolToImage(mol, size=img_size)
                    except:
                        img = Draw.MolToImage(mol, size=img_size)
                else:
                    img = Draw.MolToImage(mol, size=img_size)
            else:
                img = Draw.MolToImage(mol, size=img_size)

            # Convert to PhotoImage
            from PIL import ImageTk

            photo = ImageTk.PhotoImage(img)

            # Create canvas to display the image
            canvas = tk.Canvas(struct_frame, width=500, height=400, bg="white")
            canvas.pack(expand=True)
            canvas.create_image(250, 200, image=photo)
            canvas.image = photo  # Keep reference

            # Info frame
            info_frame = ttk.LabelFrame(
                main_frame, text="Molecular Information", padding=5
            )
            info_frame.pack(fill="x", pady=(0, 10))

            # Add molecular properties
            try:
                from rdkit.Chem import Descriptors

                mol_weight = Descriptors.MolWt(mol)
                num_atoms = mol.GetNumAtoms()
                num_bonds = mol.GetNumBonds()

                info_text = f"Molecular Weight: {mol_weight:.2f} Da\n"
                info_text += f"Number of Atoms: {num_atoms}\n"
                info_text += f"Number of Bonds: {num_bonds}"

                ttk.Label(info_frame, text=info_text, font=("Arial", 10)).pack(
                    anchor="w"
                )
            except:
                ttk.Label(
                    info_frame,
                    text="Molecular properties unavailable",
                    font=("Arial", 10),
                ).pack(anchor="w")

            # Button frame
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(fill="x", pady=(10, 0))

            # Save image button
            ttk.Button(
                button_frame,
                text="Save Image",
                command=lambda: self._save_structure_image(img, smiles),
            ).pack(side="left")

            # Close button
            ttk.Button(button_frame, text="Close", command=popup.destroy).pack(
                side="right"
            )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to display structure: {str(e)}")

    def _save_structure_image(self, img, smiles):
        """Save the structure image to a file."""
        try:
            # Ask user for file location
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("JPEG files", "*.jpg"),
                    ("All files", "*.*"),
                ],
                title="Save Structure Image",
            )

            if filename:
                img.save(filename)
                messagebox.showinfo("Success", f"Structure image saved to {filename}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {str(e)}")

    def _apply_filter(self):
        """Apply the SMARTS filter."""
        if not self.matching_spectra:
            messagebox.showwarning(
                "No Matches", "No spectra match the current SMARTS pattern."
            )
            return

        # Call the callback with matching spectra
        self.apply_callback(self.matching_spectra)
        self.dialog.destroy()

    def _cancel(self):
        """Cancel the dialog."""
        self.dialog.destroy()


class FragmentAnnotationDialog:
    """Dialog for configuring fragment annotation parameters."""

    def __init__(self, parent, callback=None, spectra=None):
        self.parent = parent
        self.callback = callback
        self.spectra = spectra or []
        self.dialog = None
        self.result = None
        # Interactive PPM tolerance function
        self.ppm_points = []  # List of (mz, ppm) points
        self.figure = None
        self.canvas = None
        self.ax = None
        self.plot_initialized = False  # Track if plot limits have been set
        self.max_precursor_mz = self._calculate_max_precursor_mz()

    def show(self):
        """Show the fragment annotation dialog."""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("Fragment Annotation - Generate Subformulas")
        self.dialog.geometry("900x800")
        self.dialog.resizable(True, True)
        self.dialog.transient(self.parent)
        self.dialog.grab_set()

        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (900 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (800 // 2)
        self.dialog.geometry(f"900x800+{x}+{y}")

        self._create_widgets()

        # Wait for dialog to close
        self.dialog.wait_window()
        return self.result

    def _calculate_max_precursor_mz(self):
        """Calculate the maximum precursor m/z from the spectra."""
        if not self.spectra:
            return 1000  # Default fallback value

        max_mz = 0
        for spectrum in self.spectra:
            # Try to get precursor m/z from metadata
            precursor_mz = None

            # Common metadata keys for precursor m/z
            for key in ["PEPMASS", "pepmass", "precursor_mz", "PRECURSOR_MZ"]:
                value = spectrum.metadata.get(key)
                if value:
                    try:
                        # PEPMASS might be "123.456 1+" format, so take first part
                        precursor_mz = float(str(value).split()[0])
                        break
                    except (ValueError, IndexError):
                        continue

            # If not found in metadata, use the highest m/z from ions as approximation
            if precursor_mz is None and spectrum.ions:
                precursor_mz = max(ion[0] for ion in spectrum.ions)

            if precursor_mz and precursor_mz > max_mz:
                max_mz = precursor_mz

        return max_mz if max_mz > 0 else 1000  # Fallback to 1000 if no valid m/z found

    def _create_widgets(self):
        """Create the dialog widgets."""
        main_frame = ttk.Frame(self.dialog, padding=10)
        main_frame.pack(fill="both", expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Fragment Annotation Configuration",
            font=("Arial", 12, "bold"),
        )
        title_label.pack(pady=(0, 15))

        # Create a paned window to split the dialog
        paned_window = ttk.PanedWindow(main_frame, orient="horizontal")
        paned_window.pack(fill="both", expand=True, pady=(0, 10))

        # Left frame for configuration options
        config_frame = ttk.Frame(paned_window, padding=5)
        paned_window.add(config_frame, weight=1)

        # Right frame for the interactive plot
        plot_frame = ttk.Frame(paned_window, padding=5)
        paned_window.add(plot_frame, weight=2)

        # Configure the left side (configuration options)
        self._create_config_widgets(config_frame)

        # Configure the right side (interactive plot)
        self._create_plot_widgets(plot_frame)

        # Buttons at the bottom
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(10, 0))

        ttk.Button(button_frame, text="Cancel", command=self._cancel).pack(
            side="right", padx=(10, 0)
        )
        ttk.Button(button_frame, text="OK", command=self._ok).pack(side="right")

    def _create_config_widgets(self, parent_frame):
        """Create the configuration widgets on the left side."""
        # Formula tag order section
        formula_frame = ttk.LabelFrame(
            parent_frame, text="Formula Tag Order", padding=10
        )
        formula_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(
            formula_frame,
            text="Specify the order of metadata tags to search for molecular formulas:",
            wraplength=300,
        ).pack(anchor="w")

        self.formula_tags_var = tk.StringVar(value="formula, FORMULA, smiles, SMILES")
        formula_entry = ttk.Entry(formula_frame, textvariable=self.formula_tags_var)
        formula_entry.pack(fill="x", pady=(5, 0))

        ttk.Label(
            formula_frame,
            text="(Comma-separated list, will try in order until a formula is found)",
            font=("Arial", 8),
            foreground="gray",
            wraplength=300,
        ).pack(anchor="w")

        # Default PPM tolerance section
        default_ppm_frame = ttk.LabelFrame(
            parent_frame, text="Default PPM Tolerance", padding=10
        )
        default_ppm_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(
            default_ppm_frame,
            text="This value is used when no custom function is defined:",
            wraplength=300,
        ).pack(anchor="w")

        ppm_input_frame = ttk.Frame(default_ppm_frame)
        ppm_input_frame.pack(fill="x", pady=(5, 0))

        ttk.Label(ppm_input_frame, text="PPM deviation:").pack(side="left")
        self.ppm_var = tk.StringVar(value="50")
        ppm_spinbox = ttk.Spinbox(
            ppm_input_frame, from_=1, to=1000, textvariable=self.ppm_var, width=10
        )
        ppm_spinbox.pack(side="left", padx=(10, 0))

        # Additional elements section
        elements_frame = ttk.LabelFrame(
            parent_frame, text="Additional Elements", padding=10
        )
        elements_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(
            elements_frame,
            text="Additional elements that might be added during fragmentation:",
            wraplength=300,
        ).pack(anchor="w")

        # Common additional elements with checkboxes
        self.additional_elements = {}
        elements_grid = ttk.Frame(elements_frame)
        elements_grid.pack(fill="x", pady=(5, 0))

        common_elements = [
            ("H", "Hydrogen"),
            ("O", "Oxygen"),
            ("Na", "Sodium"),
            ("K", "Potassium"),
            ("NH3", "Ammonia"),
            ("H2O", "Water"),
        ]

        row = 0
        for element, description in common_elements:
            var = tk.BooleanVar()
            self.additional_elements[element] = var
            cb = ttk.Checkbutton(
                elements_grid, text=f"{element} ({description})", variable=var
            )
            cb.grid(row=row, column=0, sticky="w", pady=1)
            row += 1

        # Custom additional elements
        custom_frame = ttk.Frame(elements_frame)
        custom_frame.pack(fill="x", pady=(10, 0))

        ttk.Label(
            custom_frame, text="Custom elements (comma-separated):", wraplength=300
        ).pack(anchor="w")
        self.custom_elements_var = tk.StringVar()
        custom_entry = ttk.Entry(custom_frame, textvariable=self.custom_elements_var)
        custom_entry.pack(fill="x", pady=(5, 0))

        # Max workers section
        workers_frame = ttk.LabelFrame(
            parent_frame, text="Parallel Processing", padding=10
        )
        workers_frame.pack(fill="x", pady=(10, 0))

        ttk.Label(
            workers_frame,
            text="Maximum worker threads for parallel processing:",
            wraplength=300,
        ).pack(anchor="w")

        workers_input_frame = ttk.Frame(workers_frame)
        workers_input_frame.pack(fill="x", pady=(5, 0))

        ttk.Label(workers_input_frame, text="Max workers:").pack(side="left")
        self.max_workers_var = tk.StringVar(value="600")
        workers_spinbox = ttk.Spinbox(
            workers_input_frame,
            from_=1,
            to=9000,
            textvariable=self.max_workers_var,
            width=10,
        )
        workers_spinbox.pack(side="left", padx=(10, 0))

        ttk.Label(
            workers_frame,
            text="Higher values may speed up processing but use more system resources.\nAllows up to 9000 workers for very large datasets.",
            font=("Arial", 8),
            foreground="gray",
            wraplength=300,
        ).pack(anchor="w", pady=(5, 0))

    def _create_plot_widgets(self, parent_frame):
        """Create the interactive plot widgets on the right side."""
        # Instructions
        instructions_frame = ttk.LabelFrame(
            parent_frame, text="Interactive PPM Tolerance Function", padding=5
        )
        instructions_frame.pack(fill="x", pady=(0, 10))

        instructions_text = (
            "Define a custom PPM tolerance function:\n"
            "• Left-click to add points\n"
            "• Right-click on points to remove them\n"
            "• Function interpolates linearly between points\n"
            "• Constant values outside the range\n"
            "• Use toolbar below to zoom/pan the plot"
        )
        ttk.Label(
            instructions_frame,
            text=instructions_text,
            font=("Arial", 9),
            justify="left",
        ).pack(anchor="w")

        # Plot frame
        plot_container = ttk.Frame(parent_frame)
        plot_container.pack(fill="both", expand=True)

        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 5), dpi=80)
        self.canvas = FigureCanvasTkAgg(self.figure, plot_container)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Add navigation toolbar for zoom/pan functionality
        toolbar_frame = ttk.Frame(plot_container)
        toolbar_frame.pack(fill="x")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # Initialize the plot
        self._init_plot()

        # Control buttons
        controls_frame = ttk.Frame(parent_frame)
        controls_frame.pack(fill="x", pady=(5, 0))

        ttk.Button(
            controls_frame, text="Clear All Points", command=self._clear_all_points
        ).pack(side="left", padx=(0, 5))

        ttk.Button(
            controls_frame, text="Add Default Points", command=self._add_default_points
        ).pack(side="left")

    def _init_plot(self):
        """Initialize the interactive plot."""
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("m/z", fontsize=12)
        self.ax.set_ylabel("PPM Tolerance", fontsize=12)
        self.ax.set_title(
            "Interactive PPM Tolerance Function", fontsize=14, fontweight="bold"
        )
        self.ax.grid(True, alpha=0.3)

        # Set initial axis limits
        max_x = self.max_precursor_mz * 1.15  # Add 15% to maximum precursor m/z
        self.ax.set_xlim(0, max_x)
        self.ax.set_ylim(0, 55)

        # Connect mouse events
        # Use press+release to detect a "real" single click and ignore press-and-hold.
        # Store reference to original handler and replace with a no-op so any direct
        # bindings to button_press_event (later in this method) won't trigger behavior.
        self._raw_on_click = self._on_click
        self._on_click = lambda event: None  # temporarily disable direct press handling

        # State for detecting short clicks
        self._mouse_press_time = None
        self._mouse_press_event = None
        self._click_threshold = 0.35  # seconds - max duration to consider a click
        self._move_threshold = 5  # pixels - max movement to consider a click

        def _on_mouse_press(event):
            # Only track presses inside the axes
            if event.inaxes != self.ax:
                self._mouse_press_time = None
                self._mouse_press_event = None
                return
            self._mouse_press_time = time.time()
            self._mouse_press_event = event

        def _on_mouse_release(event):
            # Only consider releases inside the axes and if we previously recorded a press
            if self._mouse_press_time is None or event.inaxes != self.ax:
                self._mouse_press_time = None
                self._mouse_press_event = None
                return

            duration = time.time() - self._mouse_press_time

            # Compute movement in display (pixel) coordinates if available
            try:
                dx = abs(event.x - self._mouse_press_event.x)
                dy = abs(event.y - self._mouse_press_event.y)
            except Exception:
                dx = dy = 0

            moved = max(dx, dy)

            # Ignore double-click events and long presses or significant movement
            if getattr(event, "dblclick", False):
                pass
            elif duration <= self._click_threshold and moved <= self._move_threshold:
                # Treat as a single click -> call original handler with the release event
                try:
                    self._raw_on_click(event)
                except Exception:
                    # Be conservative: swallow exceptions from callback to avoid crashing UI
                    pass

                # Reset press state
                self._mouse_press_time = None
                self._mouse_press_event = None

        # Connect press/release handlers instead of direct press handling
        self.canvas.mpl_connect("button_press_event", _on_mouse_press)
        self.canvas.mpl_connect("button_release_event", _on_mouse_release)

        # Mark plot as initialized before the first update
        self.plot_initialized = True

        # Initial plot update
        self._update_plot()

    def _on_click(self, event):
        """Handle mouse clicks on the plot."""
        if event.inaxes != self.ax:
            return

        if event.button == 1:  # Left click - add point
            self._add_point(event.xdata, event.ydata)
        elif event.button == 3:  # Right click - remove point
            self._remove_nearest_point(event.xdata, event.ydata)

    def _add_point(self, mz, ppm):
        """Add a point to the PPM function."""
        if mz is None or ppm is None:
            return

        # Ensure positive PPM values
        ppm = max(1, ppm)

        # Add the point
        self.ppm_points.append((mz, ppm))

        # Sort points by m/z
        self.ppm_points.sort(key=lambda x: x[0])

        self._update_plot()

    def _remove_nearest_point(self, mz, ppm):
        """Remove the nearest point to the click location."""
        if not self.ppm_points or mz is None or ppm is None:
            return

        # Find the nearest point
        min_distance = float("inf")
        nearest_index = -1

        for i, (point_mz, point_ppm) in enumerate(self.ppm_points):
            # Calculate distance (normalized by axis ranges)
            mz_range = self.ax.get_xlim()[1] - self.ax.get_xlim()[0]
            ppm_range = self.ax.get_ylim()[1] - self.ax.get_ylim()[0]

            normalized_mz_dist = (mz - point_mz) / mz_range
            normalized_ppm_dist = (ppm - point_ppm) / ppm_range

            distance = (normalized_mz_dist**2 + normalized_ppm_dist**2) ** 0.5

            if distance < min_distance:
                min_distance = distance
                nearest_index = i

        # Remove the nearest point if it's close enough (within 5% of the plot)
        if nearest_index >= 0 and min_distance < 0.05:
            self.ppm_points.pop(nearest_index)
            self._update_plot()

    def _clear_all_points(self):
        """Clear all points from the function."""
        self.ppm_points.clear()
        self._update_plot()

    def _add_default_points(self):
        """Add some default points to demonstrate the function."""
        self.ppm_points = [(100, 50), (300, 30), (500, 20), (800, 40)]
        self._update_plot()

    def _update_plot(self):
        """Update the plot display."""

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        self.ax.clear()
        self.ax.set_xlabel("m/z", fontsize=12)
        self.ax.set_ylabel("PPM Tolerance", fontsize=12)
        self.ax.set_title(
            "Interactive PPM Tolerance Function", fontsize=14, fontweight="bold"
        )
        self.ax.grid(True, alpha=0.3)

        if not self.ppm_points:
            # No points - show default constant function
            default_ppm = (
                float(self.ppm_var.get())
                if self.ppm_var.get().replace(".", "").isdigit()
                else 50
            )
            self.ax.axhline(
                y=default_ppm,
                color="blue",
                linestyle="--",
                alpha=0.7,
                label=f"Default: {default_ppm} PPM",
            )
            self.ax.legend()
        else:
            # Plot the function
            if len(self.ppm_points) == 1:
                # Single point - constant function
                mz, ppm = self.ppm_points[0]
                self.ax.axhline(
                    y=ppm,
                    color="blue",
                    linestyle="-",
                    linewidth=2,
                    label=f"Constant: {ppm:.1f} PPM",
                )
                self.ax.plot(mz, ppm, "ro", markersize=8, label="Control Point")
            else:
                # Multiple points - interpolated function
                mz_values = [point[0] for point in self.ppm_points]
                ppm_values = [point[1] for point in self.ppm_points]

                # Create continuous function across the full x-axis range
                max_x = self.max_precursor_mz * 1.15
                mz_extended = list(range(0, int(max_x) + 1, 10))

                ppm_extended = []
                for mz in mz_extended:
                    ppm_extended.append(self._interpolate_ppm(mz))

                # Plot the function
                self.ax.plot(
                    mz_extended, ppm_extended, "b-", linewidth=2, label="PPM Function"
                )
                self.ax.plot(
                    mz_values, ppm_values, "ro", markersize=8, label="Control Points"
                )

            self.ax.legend()

        # Set reasonable axis limits only during initial setup
        if not self.plot_initialized:
            max_x = self.max_precursor_mz * 1.15  # Add 15% to maximum precursor m/z
            self.ax.set_xlim(
                0, max_x
            )  # Always use full range from 0 to max precursor + 15%
            self.ax.set_ylim(0, 52)

            self.plot_initialized = True

        else:
            self.ax.set_xlim(*xlim)
            self.ax.set_ylim(*ylim)

        self.canvas.draw()

    def _interpolate_ppm(self, mz):
        """Interpolate PPM value for a given m/z using the defined function."""
        if not self.ppm_points:
            return (
                float(self.ppm_var.get())
                if self.ppm_var.get().replace(".", "").isdigit()
                else 50
            )

        if len(self.ppm_points) == 1:
            return self.ppm_points[0][1]

        # Sort points by m/z
        sorted_points = sorted(self.ppm_points, key=lambda x: x[0])

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
        return sorted_points[0][1]

    def _ok(self):
        """Handle OK button click."""
        try:
            # Validate PPM value
            ppm_value = float(self.ppm_var.get())
            if ppm_value <= 0:
                raise ValueError("PPM deviation must be positive")

            # Validate max_workers value
            max_workers = int(self.max_workers_var.get())
            if max_workers <= 0:
                raise ValueError("Max workers must be positive")
            if max_workers > 9000:
                raise ValueError("Max workers cannot exceed 9000")

            # Parse formula tags
            formula_tags = [
                tag.strip()
                for tag in self.formula_tags_var.get().split(",")
                if tag.strip()
            ]
            if not formula_tags:
                raise ValueError("At least one formula tag must be specified")

            # Get selected additional elements
            selected_elements = []
            for element, var in self.additional_elements.items():
                if var.get():
                    selected_elements.append(element)

            # Add custom elements
            custom_elements = [
                elem.strip()
                for elem in self.custom_elements_var.get().split(",")
                if elem.strip()
            ]
            selected_elements.extend(custom_elements)

            # Prepare result
            self.result = {
                "formula_tags": formula_tags,
                "ppm_tolerance": ppm_value,
                "additional_elements": selected_elements,
                "ppm_function_points": self.ppm_points.copy(),  # Include the custom function
                "max_workers": max_workers,
            }

            self.dialog.destroy()

        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))

    def _cancel(self):
        """Handle Cancel button click."""
        self.result = None
        self.dialog.destroy()


class ProgressDialog:
    """Dialog for showing progress during long-running operations."""

    def __init__(self, parent, title="Processing...", message="Please wait..."):
        self.parent = parent
        self.title = title
        self.message = message
        self.dialog = None
        self.progress_var = None
        self.progress_bar = None
        self.status_label = None
        self.cancelled = False

    def show(self, max_value=100):
        """Show the progress dialog."""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title(self.title)
        self.dialog.geometry("400x220")
        self.dialog.resizable(False, False)
        self.dialog.transient(self.parent)
        self.dialog.grab_set()

        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (400 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (220 // 2)
        self.dialog.geometry(f"400x220+{x}+{y}")

        # Prevent closing with X button
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_close)

        self._create_widgets(max_value)

    def _create_widgets(self, max_value):
        """Create the progress dialog widgets."""
        main_frame = ttk.Frame(self.dialog, padding=20)
        main_frame.pack(fill="both", expand=True)

        # Message label
        message_label = ttk.Label(main_frame, text=self.message, font=("Arial", 10))
        message_label.pack(pady=(0, 15))

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            main_frame,
            variable=self.progress_var,
            maximum=max_value,
            length=350,
            mode="determinate",
        )
        self.progress_bar.pack(pady=(0, 10))

        # Status label
        self.status_label = ttk.Label(main_frame, text="Starting...", font=("Arial", 9))
        self.status_label.pack(pady=(0, 15))

        # Cancel button
        cancel_button = ttk.Button(main_frame, text="Cancel", command=self._cancel)
        cancel_button.pack()

    def update_progress(self, value, status_text=""):
        """Update the progress bar and status text."""
        if self.dialog and self.progress_var:
            self.progress_var.set(value)
            if status_text and self.status_label:
                self.status_label.config(text=status_text)
            self.dialog.update()

    def _cancel(self):
        """Handle cancel button click."""
        self.cancelled = True

    def _on_close(self):
        """Handle dialog close attempt."""
        self.cancelled = True

    def close(self):
        """Close the progress dialog."""
        if self.dialog:
            self.dialog.destroy()
            self.dialog = None

    def is_cancelled(self):
        """Check if the operation was cancelled."""
        return self.cancelled


class PPMDeviationPlotDialog:
    """Dialog for displaying a plot of m/z vs PPM deviation for annotated fragments."""

    def __init__(self, parent):
        """Initialize the dialog."""
        self.parent = parent
        self.dialog = None

    def show(self, annotated_data):
        """
        Show the PPM deviation plot dialog.

        Args:
            annotated_data: List of dictionaries containing:
                - mz: m/z value
                - ppm_error: PPM deviation
                - formula: molecular formula
                - spectrum_id: spectrum ID (optional)
        """
        if not annotated_data:
            messagebox.showinfo("No Data", "No annotated fragments to display.")
            return

        # Create dialog window
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("PPM Deviation Plot - Annotated Fragments")
        self.dialog.geometry("800x600")
        self.dialog.resizable(True, True)
        self.dialog.grab_set()  # Make modal

        # Main frame
        main_frame = ttk.Frame(self.dialog, padding=10)
        main_frame.pack(fill="both", expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="PPM Deviation vs m/z for Annotated Fragments",
            font=("Arial", 14, "bold"),
        )
        title_label.pack(pady=(0, 10))

        # Create matplotlib figure
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, main_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, pady=(0, 5))

        # Add navigation toolbar for zoom/pan functionality
        toolbar_frame = ttk.Frame(main_frame)
        toolbar_frame.pack(fill="x", pady=(0, 10))
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # Plot the data
        self._plot_ppm_deviation(annotated_data)

        # Info frame with statistics
        info_frame = ttk.LabelFrame(main_frame, text="Statistics", padding=5)
        info_frame.pack(fill="x", pady=(0, 10))

        self._display_statistics(annotated_data, info_frame)

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x")

        # Save plot button
        ttk.Button(
            button_frame,
            text="Save Plot",
            command=lambda: self._save_plot(annotated_data),
        ).pack(side="left", padx=(0, 5))

        # Close button
        ttk.Button(button_frame, text="Close", command=self._close).pack(side="right")

        # Center the dialog
        self.dialog.transient(self.parent)
        self.dialog.wait_window()

    def _plot_ppm_deviation(self, annotated_data):
        """Create the PPM deviation plot."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Extract data
        mz_values = [item["mz"] for item in annotated_data]
        ppm_errors = [item["ppm_error"] for item in annotated_data]
        formulas = [item.get("formula", "") for item in annotated_data]
        annotation_ranks = [item.get("annotation_rank", 0) for item in annotated_data]

        # Define colors based on annotation rank with 30% transparency (alpha=0.3)
        def get_color(rank):
            if rank == 0:
                return (0.0, 0.8, 0.0, 0.3)  # Green for 1st (best match)
            elif rank == 1:
                return (0.0, 0.0, 1.0, 0.3)  # Blue for 2nd
            elif rank == 2:
                return (1.0, 0.65, 0.0, 0.3)  # Orange for 3rd
            else:
                return (1.0, 0.0, 0.0, 0.3)  # Red for 4th and beyond

        # Create colors array
        colors = [get_color(rank) for rank in annotation_ranks]

        # Create scatter plot with color coding
        scatter = ax.scatter(
            mz_values,
            ppm_errors,
            alpha=1.0,  # Set to 1.0 since alpha is included in colors
            s=50,
            c=colors,
            edgecolors="black",
            linewidth=0.5,
        )

        # Set labels and title
        ax.set_xlabel("m/z", fontsize=12)
        ax.set_ylabel("PPM Deviation", fontsize=12)
        ax.set_title(
            "PPM Deviation vs m/z for Annotated Fragments",
            fontsize=14,
            fontweight="bold",
        )

        # Create legend for color coding
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(
                facecolor=(0.0, 0.8, 0.0, 0.3),
                edgecolor="black",
                label="1st match (best)",
            ),
            Patch(facecolor=(0.0, 0.0, 1.0, 0.3), edgecolor="black", label="2nd match"),
            Patch(
                facecolor=(1.0, 0.65, 0.0, 0.3), edgecolor="black", label="3rd match"
            ),
            Patch(
                facecolor=(1.0, 0.0, 0.0, 0.3), edgecolor="black", label="4th+ match"
            ),
        ]

        # Check if PPM tolerance function was used and plot it
        ppm_tolerances_used = [
            item.get("ppm_tolerance_used") for item in annotated_data
        ]
        unique_tolerances = set(filter(None, ppm_tolerances_used))

        if len(unique_tolerances) > 1:
            # Variable tolerance was used - try to reconstruct and show the function
            # Group by m/z ranges to show the tolerance function
            mz_tolerance_pairs = []
            for item in annotated_data:
                mz = item["mz"]
                tolerance = item.get("ppm_tolerance_used")
                if tolerance is not None:
                    mz_tolerance_pairs.append((mz, tolerance))

            if mz_tolerance_pairs:
                # Sort by m/z and create a smooth tolerance line
                mz_tolerance_pairs.sort()
                tolerance_mz = [pair[0] for pair in mz_tolerance_pairs]
                tolerance_values = [pair[1] for pair in mz_tolerance_pairs]

                # Plot tolerance function as positive and negative bounds
                ax.plot(
                    tolerance_mz,
                    tolerance_values,
                    "r--",
                    alpha=0.7,
                    linewidth=2,
                    label="PPM Tolerance (upper limit)",
                )
                ax.plot(
                    tolerance_mz,
                    [-t for t in tolerance_values],
                    "r--",
                    alpha=0.7,
                    linewidth=2,
                    label="PPM Tolerance (lower limit)",
                )

                # Add to legend
                legend_elements.append(
                    Patch(
                        facecolor="none",
                        edgecolor="red",
                        linestyle="--",
                        label="PPM tolerance function",
                    )
                )

        ax.legend(handles=legend_elements, loc="upper right")

        # Add grid
        ax.grid(True, alpha=0.3)

        # Add horizontal line at y=0 for reference
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.7, linewidth=1)

        # Set axis limits with some padding
        if mz_values and ppm_errors:
            mz_range = max(mz_values) - min(mz_values)
            ppm_range = max(ppm_errors) - min(ppm_errors)

            ax.set_xlim(
                min(mz_values) - mz_range * 0.05, max(mz_values) + mz_range * 0.05
            )
            ax.set_ylim(
                min(ppm_errors) - ppm_range * 0.1, max(ppm_errors) + ppm_range * 0.1
            )

        # Add annotation on hover (if matplotlib supports it)
        try:
            # Create annotation box
            self.annot = ax.annotate(
                "",
                xy=(0, 0),
                xytext=(20, 20),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w", alpha=0.8),
                arrowprops=dict(arrowstyle="->"),
            )
            self.annot.set_visible(False)

            def update_annot(ind):
                """Update annotation with point information."""
                pos = scatter.get_offsets()[ind["ind"][0]]
                self.annot.xy = pos
                idx = ind["ind"][0]
                rank = annotation_ranks[idx] + 1  # Convert to 1-based for display
                text = f"m/z: {mz_values[idx]:.4f}\nPPM: {ppm_errors[idx]:.2f}\nFormula: {formulas[idx]}\nRank: {rank}"
                self.annot.set_text(text)
                self.annot.get_bbox_patch().set_facecolor("white")
                self.annot.get_bbox_patch().set_alpha(0.8)

            def hover(event):
                """Handle hover events over data points."""
                if event.inaxes == ax:
                    cont, ind = scatter.contains(event)
                    if cont:
                        update_annot(ind)
                        self.annot.set_visible(True)
                        self.figure.canvas.draw()
                    else:
                        if self.annot.get_visible():
                            self.annot.set_visible(False)
                            self.figure.canvas.draw()

            self.figure.canvas.mpl_connect("motion_notify_event", hover)
        except:
            # If hover annotation fails, continue without it
            pass

        # Adjust layout
        self.figure.tight_layout()
        self.canvas.draw()

    def _display_statistics(self, annotated_data, parent_frame):
        """Display statistics about the annotated data."""
        if not annotated_data:
            return

        ppm_errors = [item["ppm_error"] for item in annotated_data]
        annotation_ranks = [item.get("annotation_rank", 0) for item in annotated_data]

        # Calculate statistics
        min_ppm = min(ppm_errors)
        max_ppm = max(ppm_errors)
        mean_ppm = sum(ppm_errors) / len(ppm_errors)

        # Calculate median
        sorted_ppm = sorted(ppm_errors)
        n = len(sorted_ppm)
        median_ppm = (
            sorted_ppm[n // 2]
            if n % 2 == 1
            else (sorted_ppm[n // 2 - 1] + sorted_ppm[n // 2]) / 2
        )

        # Calculate rank distribution
        rank_counts = {}
        for rank in annotation_ranks:
            rank_display = rank + 1  # Convert to 1-based for display
            if rank_display <= 3:
                rank_counts[rank_display] = rank_counts.get(rank_display, 0) + 1
            else:
                rank_counts["4+"] = rank_counts.get("4+", 0) + 1

        # Create statistics text
        stats_text = (
            f"Total annotated fragments: {len(annotated_data)}\n"
            f"PPM deviation range: {min_ppm:.2f} - {max_ppm:.2f}\n"
            f"Mean PPM deviation: {mean_ppm:.2f}\n"
            f"Median PPM deviation: {median_ppm:.2f}\n\n"
            f"Annotation rank distribution:\n"
        )

        # Add rank distribution
        for rank in [1, 2, 3, "4+"]:
            count = rank_counts.get(rank, 0)
            percentage = (count / len(annotated_data)) * 100
            stats_text += f"  Rank {rank}: {count} ({percentage:.1f}%)\n"

        stats_label = ttk.Label(parent_frame, text=stats_text, font=("Arial", 10))
        stats_label.pack(anchor="w")

    def _save_plot(self, annotated_data):
        """Save the plot to a file."""
        from tkinter import filedialog

        filename = filedialog.asksaveasfilename(
            title="Save PPM Deviation Plot",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
                ("All files", "*.*"),
            ],
        )

        if filename:
            try:
                self.figure.savefig(filename, dpi=300, bbox_inches="tight")
                messagebox.showinfo(
                    "Success", f"Plot saved successfully to:\n{filename}"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save plot:\n{str(e)}")

    def _close(self):
        """Close the dialog."""
        if self.dialog:
            self.dialog.destroy()


class SpectrumPopupWindow:
    """Popup window for displaying selected spectra and metadata."""

    def __init__(self, parent, parser: MGFParser, selected_spectrum_ids: List[int], naming_scheme: str = "Numbered"):
        self.parent = parent
        self.parser = parser
        self.selected_spectrum_ids = selected_spectrum_ids[:]  # Create a copy
        self.naming_scheme = naming_scheme
        self.popup = None

    def show(self):
        """Show the popup window."""
        if not self.parser or not self.selected_spectrum_ids:
            return

        # Create popup window
        self.popup = tk.Toplevel(self.parent)
        self.popup.title(f"Spectrum Popup - {len(self.selected_spectrum_ids)} Spectra")
        self.popup.geometry("1200x800")
        self.popup.resizable(True, True)

        # Make it a regular window (not modal)
        # self.popup.transient(self.parent)
        # self.popup.grab_set()

        # Create main frame
        main_frame = ttk.Frame(self.popup, padding=10)
        main_frame.pack(fill="both", expand=True)

        # Create paned window for layout
        paned_window = ttk.PanedWindow(main_frame, orient="horizontal")
        paned_window.pack(fill="both", expand=True)

        # Left side: Spectrum visualization
        viz_frame = ttk.LabelFrame(paned_window, text="Spectrum Visualization", padding=5)
        paned_window.add(viz_frame, weight=2)

        # Create spectrum visualization for popup
        self.spectrum_viz = SpectrumVisualizationPopup(viz_frame)
        self.spectrum_viz.naming_scheme = self.naming_scheme  # Set the naming scheme
        self.spectrum_viz.pack(fill="both", expand=True)
        self.spectrum_viz.load_data(self.parser, self.selected_spectrum_ids)

        # Right side: Metadata and ion data
        right_frame = ttk.Frame(paned_window)
        paned_window.add(right_frame, weight=1)

        # Metadata section
        metadata_frame = ttk.LabelFrame(right_frame, text="Metadata", padding=5)
        metadata_frame.pack(fill="both", expand=True, pady=(0, 5))

        # Create text widget with scrollbar for metadata
        metadata_text_frame = ttk.Frame(metadata_frame)
        metadata_text_frame.pack(fill="both", expand=True)

        self.metadata_text = tk.Text(metadata_text_frame, wrap="word", height=15)
        metadata_scrollbar = ttk.Scrollbar(metadata_text_frame, orient="vertical", 
                                         command=self.metadata_text.yview)
        self.metadata_text.configure(yscrollcommand=metadata_scrollbar.set)

        self.metadata_text.pack(side="left", fill="both", expand=True)
        metadata_scrollbar.pack(side="right", fill="y")

        # Ion data section
        ion_frame = ttk.LabelFrame(right_frame, text="Ion Data Summary", padding=5)
        ion_frame.pack(fill="both", expand=True)

        # Create text widget with scrollbar for ion data
        ion_text_frame = ttk.Frame(ion_frame)
        ion_text_frame.pack(fill="both", expand=True)

        self.ion_text = tk.Text(ion_text_frame, wrap="word", height=15)
        ion_scrollbar = ttk.Scrollbar(ion_text_frame, orient="vertical", 
                                    command=self.ion_text.yview)
        self.ion_text.configure(yscrollcommand=ion_scrollbar.set)

        self.ion_text.pack(side="left", fill="both", expand=True)
        ion_scrollbar.pack(side="right", fill="y")

        # Populate metadata and ion data
        self._populate_metadata()
        self._populate_ion_data()

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(10, 0))

        ttk.Button(button_frame, text="Close", command=self._close_popup).pack(side="right")

        # Center the window
        self.popup.update_idletasks()
        width = self.popup.winfo_width()
        height = self.popup.winfo_height()
        x = (self.popup.winfo_screenwidth() // 2) - (width // 2)
        y = (self.popup.winfo_screenheight() // 2) - (height // 2)
        self.popup.geometry(f"{width}x{height}+{x}+{y}")

    def _populate_metadata(self):
        """Populate the metadata text widget."""
        self.metadata_text.delete(1.0, tk.END)
        
        # Get selected spectra
        selected_spectra = [
            s for s in self.parser.spectra 
            if s.spectrum_id in self.selected_spectrum_ids
        ]

        for i, spectrum in enumerate(selected_spectra):
            self.metadata_text.insert(tk.END, f"=== Spectrum {spectrum.spectrum_id} ===\n")
            
            if spectrum.metadata:
                for key, value in spectrum.metadata.items():
                    self.metadata_text.insert(tk.END, f"{key}: {value}\n")
            else:
                self.metadata_text.insert(tk.END, "No metadata available\n")
            
            if i < len(selected_spectra) - 1:
                self.metadata_text.insert(tk.END, "\n")

        self.metadata_text.config(state="disabled")

    def _populate_ion_data(self):
        """Populate the ion data text widget with summary."""
        self.ion_text.delete(1.0, tk.END)
        
        # Get selected spectra
        selected_spectra = [
            s for s in self.parser.spectra 
            if s.spectrum_id in self.selected_spectrum_ids
        ]

        for i, spectrum in enumerate(selected_spectra):
            self.ion_text.insert(tk.END, f"=== Spectrum {spectrum.spectrum_id} ===\n")
            
            if spectrum.ions.size > 0:
                self.ion_text.insert(tk.END, f"Number of ions: {len(spectrum.ions)}\n")
                
                # Basic statistics
                mz_values = spectrum.ions[:, 0]
                intensity_values = spectrum.ions[:, 1]
                
                self.ion_text.insert(tk.END, f"m/z range: {mz_values.min():.4f} - {mz_values.max():.4f}\n")
                self.ion_text.insert(tk.END, f"Intensity range: {intensity_values.min():.2f} - {intensity_values.max():.2f}\n")
                
                # Top 5 most intense ions
                sorted_indices = np.argsort(intensity_values)[::-1]
                self.ion_text.insert(tk.END, "\nTop 5 most intense ions:\n")
                for j in range(min(5, len(sorted_indices))):
                    idx = sorted_indices[j]
                    mz = mz_values[idx]
                    intensity = intensity_values[idx]
                    self.ion_text.insert(tk.END, f"  {j+1}. m/z {mz:.4f}, intensity {intensity:.2f}\n")
            else:
                self.ion_text.insert(tk.END, "No ion data available\n")
            
            if i < len(selected_spectra) - 1:
                self.ion_text.insert(tk.END, "\n")

        self.ion_text.config(state="disabled")

    def _close_popup(self):
        """Close the popup window."""
        if self.popup:
            self.popup.destroy()


class SpectrumVisualizationPopup(ttk.Frame):
    """Spectrum visualization component for popup windows (simplified version)."""

    def __init__(self, parent):
        super().__init__(parent)
        self.parser: Optional[MGFParser] = None
        self.selected_spectrum_ids: List[int] = []
        self.show_combined_plot = tk.BooleanVar(value=False)
        self.ppm_tolerance = tk.DoubleVar(value=20.0)
        self.top_fragments_count = tk.IntVar(value=15)  # Number of top fragments to show
        self.naming_scheme = "Numbered"  # Current spectrum naming scheme

        self._create_widgets()

    def _create_widgets(self):
        """Create the visualization widgets."""
        # Header with controls
        header_frame = ttk.Frame(self)
        header_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(
            header_frame, text="Spectrum Visualization", font=("Arial", 12, "bold")
        ).pack(side="left")

        # Controls frame
        controls_frame = ttk.Frame(header_frame)
        controls_frame.pack(side="right")

        # Combined plot checkbox
        ttk.Checkbutton(
            controls_frame,
            text="Show Combined Plot",
            variable=self.show_combined_plot,
            command=self._plot_spectra
        ).pack(side="left", padx=(0, 10))

        # PPM tolerance for fragment matching in combined plot
        ttk.Label(controls_frame, text="PPM tolerance:").pack(side="left", padx=(0, 2))
        ppm_spinbox = ttk.Spinbox(
            controls_frame,
            from_=1.0,
            to=100.0,
            increment=1.0,
            width=8,
            textvariable=self.ppm_tolerance,
            command=self._on_ppm_change
        )
        ppm_spinbox.pack(side="left", padx=(0, 10))
        ppm_spinbox.bind('<KeyRelease>', self._on_ppm_change)

        # Top fragments count for combined plot
        ttk.Label(controls_frame, text="Top fragments:").pack(side="left", padx=(0, 2))
        fragments_spinbox = ttk.Spinbox(
            controls_frame,
            from_=5,
            to=50,
            increment=1,
            width=6,
            textvariable=self.top_fragments_count,
            command=self._on_fragments_count_change
        )
        fragments_spinbox.pack(side="left")
        fragments_spinbox.bind('<KeyRelease>', self._on_fragments_count_change)

        # Matplotlib figure
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=(5, 0))

        # Add navigation toolbar
        toolbar_frame = ttk.Frame(self)
        toolbar_frame.pack(fill="x", padx=5, pady=(0, 5))
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

    def _get_spectrum_display_name(self, spectrum_or_id):
        """Get the display name for a spectrum based on the current naming scheme."""
        # Handle both spectrum objects and spectrum IDs
        if isinstance(spectrum_or_id, (str, int)):
            # It's a spectrum ID, find the spectrum object
            spectrum_id = spectrum_or_id
            if not self.parser:
                return f"S {spectrum_id}"
            
            spectrum = next(
                (s for s in self.parser.spectra if s.spectrum_id == spectrum_id), None
            )
            if not spectrum:
                return f"S {spectrum_id}"
        else:
            # It's already a spectrum object
            spectrum = spectrum_or_id
            spectrum_id = spectrum.spectrum_id
        
        # Use the local naming scheme
        if self.naming_scheme == "Numbered":
            return f"S {spectrum_id}"
        else:
            # Use the metadata value for the naming key
            name_value = spectrum.get_metadata_value(self.naming_scheme)
            if name_value:
                return str(name_value)
            else:
                return f"S {spectrum_id}"  # Fall back to numbered if key not found

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
            ax.text(0.5, 0.5, "No spectra selected", ha="center", va="center", 
                   transform=ax.transAxes)
            self.canvas.draw()
            return

        # Get selected spectra
        selected_spectra = [
            s for s in self.parser.spectra
            if s.spectrum_id in self.selected_spectrum_ids
        ]

        if not selected_spectra:
            return

        if self.show_combined_plot.get() and len(selected_spectra) > 1:
            self._plot_combined_spectra(selected_spectra)
        else:
            self._plot_individual_spectra(selected_spectra)

    def _plot_individual_spectra(self, selected_spectra):
        """Plot individual spectra in separate subplots."""
        # Limit to 10 spectra for popup
        if len(selected_spectra) > 10:
            selected_spectra = selected_spectra[:10]

        # Calculate global m/z limits
        global_mz_min = float("inf")
        global_mz_max = float("-inf")

        for spectrum in selected_spectra:
            if spectrum.ions.size > 0:
                mz_values = spectrum.ions[:, 0]
                global_mz_min = min(global_mz_min, mz_values.min())
                global_mz_max = max(global_mz_max, mz_values.max())

        # Add padding
        if global_mz_min != float("inf") and global_mz_max != float("-inf"):
            mz_range = global_mz_max - global_mz_min
            padding = mz_range * 0.02
            global_mz_min -= padding
            global_mz_max += padding
        else:
            global_mz_min, global_mz_max = 0, 1000

        # Create subplots
        n_spectra = len(selected_spectra)
        for i, spectrum in enumerate(selected_spectra):
            is_last = i == n_spectra - 1
            ax = self.figure.add_subplot(n_spectra, 1, i + 1)
            self._plot_single_spectrum(ax, spectrum, (global_mz_min, global_mz_max), is_last)

        self.figure.tight_layout(pad=0.5, h_pad=0.2)
        self.canvas.draw()

    def _plot_single_spectrum(self, ax, spectrum, mz_limits=None, is_last=False):
        """Plot a single spectrum as a stick chart."""
        if spectrum.ions.size == 0:
            ax.text(0.5, 0.5, "No ion data", ha="center", va="center", 
                   transform=ax.transAxes)
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

        ax.set_ylabel(f"S {spectrum.spectrum_id}\nIntensity", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Set limits
        if mz_limits:
            ax.set_xlim(mz_limits)
        elif len(mz_values) > 0:
            ax.set_xlim(mz_values.min() * 0.95, mz_values.max() * 1.05)

        if len(intensity_values) > 0:
            ax.set_ylim(0, intensity_values.max() * 1.1)

    def _plot_combined_spectra(self, selected_spectra):
        """Plot all selected spectra in a combined plot with fragment matching."""
        # Limit to 10 spectra
        if len(selected_spectra) > 10:
            selected_spectra = selected_spectra[:10]

        ax = self.figure.add_subplot(111)

        # Use the same combined plotting logic as the main visualization
        all_fragments = {}
        spectrum_colors = plt.cm.tab10(np.linspace(0, 1, min(len(selected_spectra), 10)))
        spectrum_info = {}

        for i, spectrum in enumerate(selected_spectra):
            if spectrum.ions.size == 0:
                continue

            mz_values = spectrum.ions[:, 0]
            intensity_values = spectrum.ions[:, 1]
            
            # Sum-scale intensities
            total_intensity = np.sum(intensity_values)
            if total_intensity > 0:
                relative_intensities = intensity_values / total_intensity
            else:
                relative_intensities = intensity_values

            spectrum_info[spectrum.spectrum_id] = {
                'color': spectrum_colors[i],
                'index': i,
                'label': self._get_spectrum_display_name(spectrum)
            }

            for mz, rel_intensity in zip(mz_values, relative_intensities):
                if mz not in all_fragments:
                    all_fragments[mz] = []
                all_fragments[mz].append((spectrum.spectrum_id, rel_intensity))

        # Group fragments by similar m/z values
        fragment_groups = self._group_fragments_by_mz(all_fragments, self.ppm_tolerance.get())

        # Calculate total intensity for each fragment group and sort by intensity
        fragment_intensities = []
        for group_mz, fragments_in_group in fragment_groups.items():
            if len(fragments_in_group) > 1:  # Only consider fragments present in multiple spectra
                total_intensity = sum(rel_intensity for _, rel_intensity in fragments_in_group)
                fragment_intensities.append((total_intensity, group_mz, fragments_in_group))
        
        # Sort by total intensity (descending) and take top N
        top_fragments_count = self.top_fragments_count.get()
        fragment_intensities.sort(key=lambda x: x[0], reverse=True)
        top_fragments = fragment_intensities[:top_fragments_count]

        # Create a plot showing spectra on x-axis and relative abundance on y-axis
        
        # Sort spectrum IDs by their display names using natural sorting
        sorted_spec_ids = list(spectrum_info.keys())
        try:
            sorted_spec_ids = natsorted(sorted_spec_ids, 
                                      key=lambda spec_id: self._get_spectrum_display_name(spec_id))
        except NameError:
            # Fallback to regular sorting if natsort is not available
            sorted_spec_ids = sorted(sorted_spec_ids, 
                                   key=lambda spec_id: self._get_spectrum_display_name(spec_id))
        
        spectrum_positions = {spec_id: i for i, spec_id in enumerate(sorted_spec_ids)}
        spectrum_labels = [self._get_spectrum_display_name(spec_id) for spec_id in sorted_spec_ids]

        # Plot each top fragment group as a line connecting spectra
        for total_intensity, group_mz, fragments_in_group in top_fragments:
            x_positions = []
            y_intensities = []
            
            sorted_fragments = sorted(fragments_in_group, 
                                    key=lambda x: spectrum_info.get(x[0], {}).get('index', 999))
            
            for spectrum_id, rel_intensity in sorted_fragments:
                if spectrum_id in spectrum_info:
                    x_positions.append(spectrum_positions[spectrum_id])
                    y_intensities.append(rel_intensity)
            
            if len(x_positions) > 1:
                ax.plot(x_positions, y_intensities, 'o-', alpha=0.7, linewidth=2, 
                       markersize=6, label=f'm/z {group_mz:.4f}')

        # Set x-axis to show spectrum names
        ax.set_xticks(range(len(spectrum_labels)))
        ax.set_xticklabels(spectrum_labels, rotation=45, ha='right')
        ax.set_xlabel("Spectra")
        ax.set_ylabel("Relative Abundance (Sum-scaled)")
        ax.set_title(f"Combined Spectrum Plot - Fragment Matching ({len(selected_spectra)} spectra)")
        ax.grid(True, alpha=0.3)

        # Add legend for fragment m/z values
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            legend_title = f"Top {min(len(handles), top_fragments_count)} fragments (by intensity)"
            ax.legend(loc='upper right', framealpha=0.9, fontsize=8, title=legend_title)

        self.figure.tight_layout()
        self.canvas.draw()

    def _group_fragments_by_mz(self, all_fragments, ppm_tolerance):
        """Group fragments by similar m/z values within PPM tolerance."""
        fragment_groups = {}
        sorted_mz_values = sorted(all_fragments.keys())
        
        for mz in sorted_mz_values:
            group_found = False
            for group_mz in fragment_groups:
                ppm_diff = abs(mz - group_mz) / group_mz * 1e6
                if ppm_diff <= ppm_tolerance:
                    fragment_groups[group_mz].extend(all_fragments[mz])
                    group_found = True
                    break
            
            if not group_found:
                fragment_groups[mz] = all_fragments[mz][:]
        
        return fragment_groups

    def _on_ppm_change(self, event=None):
        """Handle PPM tolerance change."""
        if self.show_combined_plot.get():
            self._plot_spectra()

    def _on_fragments_count_change(self, event=None):
        """Handle top fragments count change."""
        if self.show_combined_plot.get():
            self._plot_spectra()
