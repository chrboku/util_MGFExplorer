"""Options dialog for configuring MGF Explorer behaviour."""

from __future__ import annotations

import copy
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Dict, List, Optional


class OptionsDialog:
    """Dialog for editing persistent application options."""

    def __init__(self, parent: tk.Misc, initial_config: Dict[str, Any]):
        self.parent = parent
        self.initial_config = copy.deepcopy(initial_config)
        self.result: Optional[Dict[str, Any]] = None

        self._groups: List[Dict[str, Any]] = []
        self._active_group_index: Optional[int] = None

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Options")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.resizable(True, True)
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_cancel)

        # Centre dialog relative to parent
        self.dialog.geometry(
            "+%d+%d"
            % (
                parent.winfo_rootx() + 60,
                parent.winfo_rooty() + 60,
            )
        )

        self._build_ui()
        self._load_initial_state()
        self.dialog.wait_window()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        main_frame = ttk.Frame(self.dialog, padding=12)
        main_frame.pack(fill="both", expand=True)

        # Default grouping tags ------------------------------------------------
        tags_frame = ttk.LabelFrame(main_frame, text="Default Grouping Tags", padding=8)
        tags_frame.pack(fill="x", expand=False, pady=(0, 10))

        ttk.Label(
            tags_frame,
            text="Comma-separated list applied to the grouping field when the app starts.",
        ).pack(anchor="w", pady=(0, 4))

        self.default_tags_var = tk.StringVar()
        tags_entry = ttk.Entry(tags_frame, textvariable=self.default_tags_var)
        tags_entry.pack(fill="x")

        # Metadata groups ------------------------------------------------------
        groups_frame = ttk.LabelFrame(main_frame, text="Metadata Groups", padding=8)
        groups_frame.pack(fill="both", expand=True)

        container = ttk.Frame(groups_frame)
        container.pack(fill="both", expand=True)

        # Left side: list of groups
        left_frame = ttk.Frame(container)
        left_frame.pack(side="left", fill="y", padx=(0, 8))

        self.group_listbox = tk.Listbox(left_frame, height=10, exportselection=False)
        self.group_listbox.pack(fill="y", expand=True)
        self.group_listbox.bind("<<ListboxSelect>>", self._on_group_selection)

        group_buttons = ttk.Frame(left_frame)
        group_buttons.pack(fill="x", pady=(6, 0))

        ttk.Button(group_buttons, text="Add", command=self._add_group).grid(
            row=0, column=0, padx=2
        )
        ttk.Button(group_buttons, text="Remove", command=self._remove_group).grid(
            row=0, column=1, padx=2
        )
        ttk.Button(group_buttons, text="Up", command=lambda: self._move_group(-1)).grid(
            row=0, column=2, padx=2
        )
        ttk.Button(
            group_buttons, text="Down", command=lambda: self._move_group(1)
        ).grid(row=0, column=3, padx=2)

        # Right side: group detail editor
        detail_frame = ttk.Frame(container)
        detail_frame.pack(side="left", fill="both", expand=True)

        name_frame = ttk.Frame(detail_frame)
        name_frame.pack(fill="x")

        ttk.Label(name_frame, text="Group Name:").pack(anchor="w")
        self.group_name_var = tk.StringVar()
        self.group_name_var.trace_add("write", self._on_group_name_change)
        self.group_name_entry = ttk.Entry(name_frame, textvariable=self.group_name_var)
        self.group_name_entry.pack(fill="x", pady=(0, 8))

        keys_frame = ttk.Frame(detail_frame)
        keys_frame.pack(fill="both", expand=True)

        ttk.Label(keys_frame, text="Keys in this group (order preserved):").pack(
            anchor="w"
        )

        self.keys_listbox = tk.Listbox(keys_frame, height=10, exportselection=False)
        self.keys_listbox.pack(fill="both", expand=True)

        key_controls = ttk.Frame(keys_frame)
        key_controls.pack(fill="x", pady=(6, 0))

        self.new_key_var = tk.StringVar()
        ttk.Entry(key_controls, textvariable=self.new_key_var, width=20).grid(
            row=0, column=0, padx=2
        )
        ttk.Button(key_controls, text="Add", command=self._add_key).grid(
            row=0, column=1, padx=2
        )
        ttk.Button(key_controls, text="Remove", command=self._remove_key).grid(
            row=0, column=2, padx=2
        )
        ttk.Button(key_controls, text="Up", command=lambda: self._move_key(-1)).grid(
            row=0, column=3, padx=2
        )
        ttk.Button(key_controls, text="Down", command=lambda: self._move_key(1)).grid(
            row=0, column=4, padx=2
        )

        # Buttons --------------------------------------------------------------
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(12, 0))

        ttk.Button(button_frame, text="Cancel", command=self._on_cancel).pack(
            side="right", padx=(5, 0)
        )
        ttk.Button(button_frame, text="Save", command=self._on_save).pack(side="right")

        self._update_detail_state(False)

    # ------------------------------------------------------------------
    # Data initialisation and helpers
    # ------------------------------------------------------------------
    def _load_initial_state(self) -> None:
        tags = ", ".join(self.initial_config.get("default_grouping_tags", []))
        self.default_tags_var.set(tags)

        groups = self.initial_config.get("metadata_groups", [])
        if isinstance(groups, list):
            self._groups = [
                {
                    "name": str(group.get("name", "")).strip(),
                    "keys": list(group.get("keys", [])),
                }
                for group in groups
                if isinstance(group, dict)
            ]
        else:
            self._groups = []

        self._refresh_group_listbox()
        if self._groups:
            self.group_listbox.selection_set(0)
            self._on_group_selection()

    def _refresh_group_listbox(self) -> None:
        self.group_listbox.delete(0, tk.END)
        for group in self._groups:
            display_name = group["name"] or "(unnamed group)"
            self.group_listbox.insert(tk.END, display_name)

    def _update_detail_state(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        widgets = [
            self.group_name_entry,
            self.keys_listbox,
        ]
        for widget in widgets:
            widget.configure(state=state)

    # ------------------------------------------------------------------
    # Group operations
    # ------------------------------------------------------------------
    def _on_group_selection(self, event: Optional[tk.Event] = None) -> None:
        if not self.group_listbox.curselection():
            self._active_group_index = None
            self._update_detail_state(False)
            self.group_name_var.set("")
            self.keys_listbox.delete(0, tk.END)
            return

        self._active_group_index = int(self.group_listbox.curselection()[0])
        self._update_detail_state(True)

        active_group = self._groups[self._active_group_index]
        self.group_name_var.set(active_group["name"])

        self.keys_listbox.delete(0, tk.END)
        for key in active_group["keys"]:
            self.keys_listbox.insert(tk.END, key)

    def _add_group(self) -> None:
        new_index = len(self._groups) + 1
        proposed_name = f"Group {new_index}"

        self._groups.append({"name": proposed_name, "keys": []})
        self._refresh_group_listbox()
        last_index = len(self._groups) - 1
        self.group_listbox.selection_clear(0, tk.END)
        self.group_listbox.selection_set(last_index)
        self.group_listbox.see(last_index)
        self._on_group_selection()

    def _remove_group(self) -> None:
        if self._active_group_index is None:
            return
        del self._groups[self._active_group_index]
        self._refresh_group_listbox()
        if self._groups:
            new_index = min(self._active_group_index, len(self._groups) - 1)
            self.group_listbox.selection_set(new_index)
            self._on_group_selection()
        else:
            self.group_listbox.selection_clear(0, tk.END)
            self._on_group_selection()

    def _move_group(self, delta: int) -> None:
        if self._active_group_index is None:
            return
        new_index = self._active_group_index + delta
        if new_index < 0 or new_index >= len(self._groups):
            return
        self._groups[self._active_group_index], self._groups[new_index] = (
            self._groups[new_index],
            self._groups[self._active_group_index],
        )
        self._refresh_group_listbox()
        self.group_listbox.selection_set(new_index)
        self.group_listbox.see(new_index)
        self._active_group_index = new_index
        self._on_group_selection()

    def _on_group_name_change(self, *_: Any) -> None:
        if self._active_group_index is None:
            return
        name = self.group_name_var.get().strip()
        self._groups[self._active_group_index]["name"] = name
        self._refresh_group_listbox()
        self.group_listbox.selection_set(self._active_group_index)

    # ------------------------------------------------------------------
    # Key operations
    # ------------------------------------------------------------------
    def _add_key(self) -> None:
        if self._active_group_index is None:
            messagebox.showwarning(
                "No Group Selected", "Select a group before adding keys."
            )
            return

        key = self.new_key_var.get().strip()
        if not key:
            return

        active_group = self._groups[self._active_group_index]
        if key in active_group["keys"]:
            messagebox.showinfo("Duplicate Key", f"'{key}' is already in this group.")
            return

        active_group["keys"].append(key)
        self.keys_listbox.insert(tk.END, key)
        self.keys_listbox.selection_clear(0, tk.END)
        self.keys_listbox.selection_set(tk.END)
        self.new_key_var.set("")

    def _remove_key(self) -> None:
        if self._active_group_index is None:
            return
        selection = self.keys_listbox.curselection()
        if not selection:
            return
        index = int(selection[0])
        del self._groups[self._active_group_index]["keys"][index]
        self.keys_listbox.delete(index)

    def _move_key(self, delta: int) -> None:
        if self._active_group_index is None:
            return
        selection = self.keys_listbox.curselection()
        if not selection:
            return
        index = int(selection[0])
        new_index = index + delta
        keys = self._groups[self._active_group_index]["keys"]
        if new_index < 0 or new_index >= len(keys):
            return
        keys[index], keys[new_index] = keys[new_index], keys[index]
        self._on_group_selection()
        self.keys_listbox.selection_set(new_index)
        self.keys_listbox.see(new_index)

    # ------------------------------------------------------------------
    # Dialog result handling
    # ------------------------------------------------------------------
    def _collect_result(self) -> Optional[Dict[str, Any]]:
        cleaned_groups: List[Dict[str, Any]] = []
        group_names = set()
        for group in self._groups:
            name = group.get("name", "").strip()
            keys = [
                str(key).strip() for key in group.get("keys", []) if str(key).strip()
            ]

            if not name:
                messagebox.showerror("Invalid Group", "Each group must have a name.")
                return None

            if name in group_names:
                messagebox.showerror(
                    "Duplicate Group",
                    f"There are multiple groups named '{name}'. Please use unique names.",
                )
                return None

            group_names.add(name)
            cleaned_groups.append({"name": name, "keys": keys})

        tags = [
            part.strip()
            for part in self.default_tags_var.get().split(",")
            if part.strip()
        ]

        result = copy.deepcopy(self.initial_config)
        result["default_grouping_tags"] = tags
        result["metadata_groups"] = cleaned_groups
        return result

    def _on_save(self) -> None:
        collected = self._collect_result()
        if collected is None:
            return
        self.result = collected
        self.dialog.destroy()

    def _on_cancel(self) -> None:
        self.dialog.destroy()
