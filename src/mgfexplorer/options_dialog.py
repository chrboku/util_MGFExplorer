"""Options dialog for configuring MGF Explorer behaviour."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QPushButton,
    QLabel,
    QLineEdit,
    QListWidget,
    QDialogButtonBox,
    QMessageBox,
    QWidget,
)


class OptionsDialog(QDialog):
    """Dialog for editing persistent application options."""

    def __init__(self, parent, initial_config: Dict[str, Any]):
        super().__init__(parent)
        self.initial_config = copy.deepcopy(initial_config)
        self.result: Optional[Dict[str, Any]] = None

        self._groups: List[Dict[str, Any]] = []
        self._active_group_index: Optional[int] = None

        self.setWindowTitle("Options")
        self.setModal(True)
        self.resize(600, 500)

        self._build_ui()
        self._load_initial_state()
        self.exec()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)

        # Default grouping tags ------------------------------------------------
        tags_group = QGroupBox("Default Grouping Tags")
        tags_layout = QVBoxLayout(tags_group)

        tags_layout.addWidget(
            QLabel(
                "Comma-separated list applied to the grouping field when the app starts."
            )
        )

        self.default_tags_edit = QLineEdit()
        tags_layout.addWidget(self.default_tags_edit)
        main_layout.addWidget(tags_group)

        # Metadata groups ------------------------------------------------------
        groups_group = QGroupBox("Metadata Groups")
        groups_layout = QVBoxLayout(groups_group)

        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        # Left side: list of groups
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 8, 0)

        self.group_list = QListWidget()
        self.group_list.setFixedWidth(180)
        left_layout.addWidget(self.group_list)
        self.group_list.currentRowChanged.connect(self._on_group_selection)

        group_buttons_widget = QWidget()
        group_buttons_layout = QGridLayout(group_buttons_widget)
        group_buttons_layout.setContentsMargins(0, 0, 0, 0)

        add_group_btn = QPushButton("Add")
        add_group_btn.clicked.connect(self._add_group)
        group_buttons_layout.addWidget(add_group_btn, 0, 0)

        remove_group_btn = QPushButton("Remove")
        remove_group_btn.clicked.connect(self._remove_group)
        group_buttons_layout.addWidget(remove_group_btn, 0, 1)

        up_group_btn = QPushButton("Up")
        up_group_btn.clicked.connect(lambda: self._move_group(-1))
        group_buttons_layout.addWidget(up_group_btn, 0, 2)

        down_group_btn = QPushButton("Down")
        down_group_btn.clicked.connect(lambda: self._move_group(1))
        group_buttons_layout.addWidget(down_group_btn, 0, 3)

        left_layout.addWidget(group_buttons_widget)
        container_layout.addWidget(left_widget)

        # Right side: group detail editor
        detail_widget = QWidget()
        detail_layout = QVBoxLayout(detail_widget)
        detail_layout.setContentsMargins(0, 0, 0, 0)

        name_widget = QWidget()
        name_layout = QVBoxLayout(name_widget)
        name_layout.setContentsMargins(0, 0, 0, 0)
        name_layout.addWidget(QLabel("Group Name:"))
        self.group_name_entry = QLineEdit()
        self.group_name_entry.textChanged.connect(self._on_group_name_change)
        name_layout.addWidget(self.group_name_entry)
        detail_layout.addWidget(name_widget)

        keys_widget = QWidget()
        keys_layout = QVBoxLayout(keys_widget)
        keys_layout.setContentsMargins(0, 0, 0, 0)
        keys_layout.addWidget(QLabel("Keys in this group (order preserved):"))
        self.keys_list = QListWidget()
        keys_layout.addWidget(self.keys_list)

        key_controls_widget = QWidget()
        key_controls_layout = QGridLayout(key_controls_widget)
        key_controls_layout.setContentsMargins(0, 0, 0, 0)

        self.new_key_edit = QLineEdit()
        self.new_key_edit.setFixedWidth(140)
        key_controls_layout.addWidget(self.new_key_edit, 0, 0)

        add_key_btn = QPushButton("Add")
        add_key_btn.clicked.connect(self._add_key)
        key_controls_layout.addWidget(add_key_btn, 0, 1)

        remove_key_btn = QPushButton("Remove")
        remove_key_btn.clicked.connect(self._remove_key)
        key_controls_layout.addWidget(remove_key_btn, 0, 2)

        up_key_btn = QPushButton("Up")
        up_key_btn.clicked.connect(lambda: self._move_key(-1))
        key_controls_layout.addWidget(up_key_btn, 0, 3)

        down_key_btn = QPushButton("Down")
        down_key_btn.clicked.connect(lambda: self._move_key(1))
        key_controls_layout.addWidget(down_key_btn, 0, 4)

        keys_layout.addWidget(key_controls_widget)
        detail_layout.addWidget(keys_widget)
        container_layout.addWidget(detail_widget, stretch=1)

        groups_layout.addWidget(container)
        main_layout.addWidget(groups_group, stretch=1)

        # Buttons --------------------------------------------------------------
        button_box = QDialogButtonBox()
        save_btn = button_box.addButton("Save", QDialogButtonBox.ButtonRole.AcceptRole)
        cancel_btn = button_box.addButton(
            "Cancel", QDialogButtonBox.ButtonRole.RejectRole
        )
        save_btn.clicked.connect(self._on_save)
        cancel_btn.clicked.connect(self._on_cancel)
        main_layout.addWidget(button_box)

        self._update_detail_state(False)

    # ------------------------------------------------------------------
    # Data initialisation and helpers
    # ------------------------------------------------------------------
    def _load_initial_state(self) -> None:
        tags = ", ".join(self.initial_config.get("default_grouping_tags", []))
        self.default_tags_edit.setText(tags)

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
            self.group_list.setCurrentRow(0)

    def _refresh_group_listbox(self) -> None:
        self.group_list.blockSignals(True)
        self.group_list.clear()
        for group in self._groups:
            display_name = group["name"] or "(unnamed group)"
            self.group_list.addItem(display_name)
        self.group_list.blockSignals(False)

    def _update_detail_state(self, enabled: bool) -> None:
        self.group_name_entry.setEnabled(enabled)
        self.keys_list.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Group operations
    # ------------------------------------------------------------------
    def _on_group_selection(self, row: int) -> None:
        if row < 0:
            self._active_group_index = None
            self._update_detail_state(False)
            self.group_name_entry.blockSignals(True)
            self.group_name_entry.setText("")
            self.group_name_entry.blockSignals(False)
            self.keys_list.clear()
            return

        self._active_group_index = row
        self._update_detail_state(True)

        active_group = self._groups[self._active_group_index]
        self.group_name_entry.blockSignals(True)
        self.group_name_entry.setText(active_group["name"])
        self.group_name_entry.blockSignals(False)

        self.keys_list.clear()
        for key in active_group["keys"]:
            self.keys_list.addItem(key)

    def _add_group(self) -> None:
        new_index = len(self._groups) + 1
        proposed_name = f"Group {new_index}"

        self._groups.append({"name": proposed_name, "keys": []})
        self._refresh_group_listbox()
        last_index = len(self._groups) - 1
        self.group_list.setCurrentRow(last_index)

    def _remove_group(self) -> None:
        if self._active_group_index is None:
            return
        del self._groups[self._active_group_index]
        self._refresh_group_listbox()
        if self._groups:
            new_index = min(self._active_group_index, len(self._groups) - 1)
            self.group_list.setCurrentRow(new_index)
        else:
            self._active_group_index = None
            self._update_detail_state(False)
            self.group_name_entry.blockSignals(True)
            self.group_name_entry.setText("")
            self.group_name_entry.blockSignals(False)
            self.keys_list.clear()

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
        self.group_list.setCurrentRow(new_index)
        self._active_group_index = new_index

    def _on_group_name_change(self, text: str) -> None:
        if self._active_group_index is None:
            return
        name = text.strip()
        self._groups[self._active_group_index]["name"] = name
        self._refresh_group_listbox()
        self.group_list.blockSignals(True)
        self.group_list.setCurrentRow(self._active_group_index)
        self.group_list.blockSignals(False)

    # ------------------------------------------------------------------
    # Key operations
    # ------------------------------------------------------------------
    def _add_key(self) -> None:
        if self._active_group_index is None:
            QMessageBox.warning(
                self, "No Group Selected", "Select a group before adding keys."
            )
            return

        key = self.new_key_edit.text().strip()
        if not key:
            return

        active_group = self._groups[self._active_group_index]
        if key in active_group["keys"]:
            QMessageBox.information(
                self, "Duplicate Key", f"'{key}' is already in this group."
            )
            return

        active_group["keys"].append(key)
        self.keys_list.addItem(key)
        self.keys_list.setCurrentRow(self.keys_list.count() - 1)
        self.new_key_edit.setText("")

    def _remove_key(self) -> None:
        if self._active_group_index is None:
            return
        row = self.keys_list.currentRow()
        if row < 0:
            return
        del self._groups[self._active_group_index]["keys"][row]
        self.keys_list.takeItem(row)

    def _move_key(self, delta: int) -> None:
        if self._active_group_index is None:
            return
        row = self.keys_list.currentRow()
        if row < 0:
            return
        new_index = row + delta
        keys = self._groups[self._active_group_index]["keys"]
        if new_index < 0 or new_index >= len(keys):
            return
        keys[row], keys[new_index] = keys[new_index], keys[row]
        # Refresh keys display preserving selection
        saved_group = self._active_group_index
        self.keys_list.clear()
        for key in keys:
            self.keys_list.addItem(key)
        self.keys_list.setCurrentRow(new_index)

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
                QMessageBox.critical(
                    self, "Invalid Group", "Each group must have a name."
                )
                return None

            if name in group_names:
                QMessageBox.critical(
                    self,
                    "Duplicate Group",
                    f"There are multiple groups named '{name}'. Please use unique names.",
                )
                return None

            group_names.add(name)
            cleaned_groups.append({"name": name, "keys": keys})

        tags = [
            part.strip()
            for part in self.default_tags_edit.text().split(",")
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
        self.accept()

    def _on_cancel(self) -> None:
        self.reject()
