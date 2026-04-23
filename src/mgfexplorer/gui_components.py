"""
GUI components for the MGF Explorer application.
Rewritten for PyQt6 from tkinter.
"""

import re
import threading
import time
import io
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from matplotlib.widgets import RectangleSelector

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter,
    QLabel, QLineEdit, QPushButton, QTreeWidget, QTreeWidgetItem,
    QTabWidget, QDialog, QDialogButtonBox, QMessageBox, QScrollArea,
    QFrame, QTextEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
    QGroupBox, QRadioButton, QButtonGroup, QProgressBar, QApplication,
    QAbstractItemView, QHeaderView, QSizePolicy, QMenu, QListWidget,
    QListWidgetItem, QFileDialog,
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QSize, QPoint,
)
from PyQt6.QtGui import (
    QFont, QPixmap, QImage, QCursor, QAction, QActionGroup,
    QDoubleValidator, QIntValidator,
)

from .mgf_parser import MGFParser, Spectrum

# Natural sorting
try:
    from natsort import natsorted
    NATSORT_AVAILABLE = True
except ImportError:
    NATSORT_AVAILABLE = False

    def natsorted(items, key=None):
        def natural_key(text):
            if key:
                text = key(text)
            return [
                int(c) if c.isdigit() else c.lower()
                for c in re.split("([0-9]+)", str(text))
            ]
        return sorted(items, key=natural_key)

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D
    from PIL import Image
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


class ToolTip:
    """Thin wrapper — PyQt6 uses setToolTip() natively."""

    @staticmethod
    def add_tooltip(widget: QWidget, text: str):
        widget.setToolTip(text)

    def __init__(self, widget: QWidget, text: str = ""):
        """Legacy constructor: just sets the tooltip on the widget."""
        if widget is not None:
            widget.setToolTip(text)


# ---------------------------------------------------------------------------
# SpectrumTreeView
# ---------------------------------------------------------------------------

class SpectrumTreeView(QWidget):
    """Tree view for displaying spectra with grouping/filtering."""

    def __init__(self, parent=None, on_selection_changed=None):
        super().__init__(parent)
        self._on_selection_changed_cb = on_selection_changed
        self.parser: Optional[MGFParser] = None
        self.selected_grouping_tags: List[str] = []
        self._naming_scheme: str = "Numbered"
        self._filter_text: str = ""
        self._filter_timer: QTimer = QTimer(self)
        self._filter_timer.setSingleShot(True)
        self._filter_timer.timeout.connect(self._apply_filter)
        self._block_selection_signal = False
        self._create_widgets()

    def _create_widgets(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Grouping tags
        tags_box = QGroupBox("Grouping Tags")
        tags_layout = QVBoxLayout(tags_box)
        self._tags_entry = QLineEdit()
        self._tags_entry.setPlaceholderText("Tag1, Tag2, …  (right-click for fields)")
        self._tags_entry.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tags_entry.customContextMenuRequested.connect(self._show_tags_context_menu)
        self._tags_entry.textChanged.connect(self._on_tags_changed)
        tags_layout.addWidget(self._tags_entry)
        layout.addWidget(tags_box)

        # Filter
        filter_box = QGroupBox("Filter")
        filter_layout = QVBoxLayout(filter_box)
        self._filter_entry = QLineEdit()
        self._filter_entry.setPlaceholderText("Search…  ($$key:val  $$$key:regex)")
        self._filter_entry.setToolTip(
            "Default: case-insensitive substring in any value\n"
            "$$key: value — exact match in field\n"
            "$$$key: regex — regex match in field"
        )
        self._filter_entry.textChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self._filter_entry)
        layout.addWidget(filter_box)

        # Tree
        tree_box = QGroupBox("Spectra")
        tree_layout = QVBoxLayout(tree_box)
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Spectrum"])
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._tree.itemSelectionChanged.connect(self._on_tree_selection)
        tree_layout.addWidget(self._tree)
        layout.addWidget(tree_box, stretch=1)

    # ------------------------------------------------------------------
    def _show_tags_context_menu(self, pos):
        if not self.parser:
            return
        menu = QMenu(self)
        all_keys = self.parser.get_all_metadata_keys()
        for key in sorted(all_keys):
            act = menu.addAction(key)
            act.triggered.connect(lambda checked, k=key: self._add_tag(k))
        menu.exec(self._tags_entry.mapToGlobal(pos))

    def _add_tag(self, tag: str):
        current = self._tags_entry.text().strip()
        new_text = f"{current}, {tag}" if current else tag
        self._tags_entry.setText(new_text)

    def _on_tags_changed(self, text: str):
        tags = [t.strip() for t in text.split(",") if t.strip()]
        self.selected_grouping_tags = tags
        self._populate_tree()

    def _on_filter_changed(self, text: str):
        self._filter_timer.stop()
        self._filter_timer.start(600)

    def _apply_filter(self):
        self._filter_text = self._filter_entry.text().strip()
        self._populate_tree()

    # ------------------------------------------------------------------
    def load_data(self, parser: MGFParser):
        self.parser = parser
        self._filter_entry.clear()
        self._filter_text = ""
        self._populate_tree()

    def _populate_tree(self):
        self._block_selection_signal = True
        self._tree.clear()
        self._block_selection_signal = False

        if not self.parser:
            return

        spectra = [s for s in self.parser.spectra if self._spectrum_matches_filter(s)]

        if self.selected_grouping_tags:
            self._populate_grouped(spectra)
        else:
            self._populate_flat(spectra)

        self._tree.expandAll()

    def _populate_flat(self, spectra):
        for spectrum in spectra:
            item = QTreeWidgetItem(self._tree)
            item.setText(0, self._get_display_name(spectrum))
            item.setData(0, Qt.ItemDataRole.UserRole, spectrum.spectrum_id)

    def _populate_grouped(self, spectra):
        groups: Dict[str, List] = {}
        for spectrum in spectra:
            key_parts = []
            for tag in self.selected_grouping_tags:
                val = spectrum.get_metadata_value(tag) or "N/A"
                key_parts.append(str(val))
            group_key = " | ".join(key_parts)
            groups.setdefault(group_key, []).append(spectrum)

        for group_name in natsorted(groups.keys()):
            group_item = QTreeWidgetItem(self._tree)
            group_item.setText(0, group_name)
            group_item.setData(0, Qt.ItemDataRole.UserRole, None)
            for spectrum in groups[group_name]:
                child = QTreeWidgetItem(group_item)
                child.setText(0, self._get_display_name(spectrum))
                child.setData(0, Qt.ItemDataRole.UserRole, spectrum.spectrum_id)

    def _get_display_name(self, spectrum) -> str:
        if self._naming_scheme == "Numbered":
            return f"Spectrum {spectrum.spectrum_id}"
        val = spectrum.get_metadata_value(self._naming_scheme)
        if val:
            return str(val)
        return f"Spectrum {spectrum.spectrum_id}"

    # ------------------------------------------------------------------
    def _on_tree_selection(self):
        if self._block_selection_signal:
            return
        ids = self.get_selected_spectrum_ids()
        if self._on_selection_changed_cb:
            self._on_selection_changed_cb(ids)

    def get_selected_spectrum_ids(self) -> List:
        ids = []
        for item in self._tree.selectedItems():
            sid = item.data(0, Qt.ItemDataRole.UserRole)
            if sid is not None:
                ids.append(sid)
        return ids

    def select_spectra_by_ids(self, ids):
        self._block_selection_signal = True
        self._tree.clearSelection()
        def _visit(item):
            sid = item.data(0, Qt.ItemDataRole.UserRole)
            if sid in ids:
                item.setSelected(True)
            for i in range(item.childCount()):
                _visit(item.child(i))
        for i in range(self._tree.topLevelItemCount()):
            _visit(self._tree.topLevelItem(i))
        self._block_selection_signal = False

    def set_grouping_tags(self, tags: List[str]):
        self.selected_grouping_tags = tags
        self._tags_entry.setText(", ".join(tags))

    def get_grouping_tags(self) -> List[str]:
        return self.selected_grouping_tags

    def set_naming_scheme(self, scheme: str):
        self._naming_scheme = scheme
        if self.parser:
            self._populate_tree()

    def get_current_grouping_structure(self) -> Dict:
        result = {}
        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            if item.childCount() > 0:
                children = []
                for j in range(item.childCount()):
                    child = item.child(j)
                    sid = child.data(0, Qt.ItemDataRole.UserRole)
                    if sid is not None:
                        children.append(sid)
                result[item.text(0)] = children
        return result

    def has_grouping(self) -> bool:
        return len(self.selected_grouping_tags) > 0

    def _spectrum_matches_filter(self, spectrum) -> bool:
        if not self._filter_text:
            return True
        ft = self._filter_text
        if ft.startswith("$$$"):
            return self._filter_key_regex(spectrum, ft[3:].strip())
        elif ft.startswith("$$"):
            return self._filter_key_exact(spectrum, ft[2:].strip())
        else:
            low = ft.lower()
            for val in spectrum.metadata.values():
                if val and low in str(val).lower():
                    return True
            return False

    def _filter_key_exact(self, spectrum, filter_text: str) -> bool:
        if ":" not in filter_text:
            return False
        key, value = filter_text.split(":", 1)
        key = key.strip()
        value = value.strip().lower()
        meta_val = spectrum.metadata.get(key)
        if meta_val and value in str(meta_val).lower():
            return True
        return False

    def _filter_key_regex(self, spectrum, filter_text: str) -> bool:
        if ":" not in filter_text:
            return False
        key, pattern = filter_text.split(":", 1)
        key = key.strip()
        pattern = pattern.strip()
        meta_val = spectrum.metadata.get(key)
        if meta_val:
            try:
                if re.search(pattern, str(meta_val), re.IGNORECASE):
                    return True
            except re.error:
                if pattern.lower() in str(meta_val).lower():
                    return True
        return False


# ---------------------------------------------------------------------------
# MetadataEditor
# ---------------------------------------------------------------------------

class MetadataEditor(QWidget):
    """Component for viewing and editing spectrum metadata."""

    def __init__(self, parent=None, on_metadata_changed=None):
        super().__init__(parent)
        self._on_metadata_changed_cb = on_metadata_changed
        self.parser: Optional[MGFParser] = None
        self.selected_spectrum_ids: List = []
        self._pending_selection: Optional[List] = None
        self.metadata_groups: List[Dict[str, List[str]]] = []
        self._others_group_name = "others"
        self._create_widgets()

    def _create_widgets(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left: metadata table
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self._metadata_tree = QTreeWidget()
        self._metadata_tree.setHeaderLabels(["Group / Key", "Value", "Unique Values"])
        self._metadata_tree.setColumnWidth(0, 180)
        self._metadata_tree.setColumnWidth(1, 200)
        self._metadata_tree.setColumnWidth(2, 120)
        self._metadata_tree.setAlternatingRowColors(True)
        self._metadata_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._metadata_tree.customContextMenuRequested.connect(self._on_right_click)
        self._metadata_tree.itemDoubleClicked.connect(self._on_double_click)
        self._metadata_tree.itemChanged.connect(self._on_item_changed)
        left_layout.addWidget(self._metadata_tree)
        splitter.addWidget(left)

        # Right: SMILES plot
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self._smiles_fig = Figure(figsize=(4, 4), dpi=80)
        self._smiles_canvas = FigureCanvasQTAgg(self._smiles_fig)
        right_layout.addWidget(self._smiles_canvas)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        self._show_smiles_message("No SMILES data available")
        self._editing_item = None
        self._editing_column = None

    # ------------------------------------------------------------------
    def _show_smiles_message(self, msg: str):
        self._smiles_fig.clear()
        ax = self._smiles_fig.add_subplot(111)
        ax.text(0.5, 0.5, msg, ha="center", va="center",
                fontsize=9, wrap=True, transform=ax.transAxes)
        ax.axis("off")
        self._smiles_canvas.draw()

    def _plot_smiles(self, smiles_code: str):
        if not RDKIT_AVAILABLE:
            self._show_smiles_message("RDKit not available")
            return
        try:
            mol = Chem.MolFromSmiles(smiles_code)
            if mol is None:
                self._show_smiles_message(f"Invalid SMILES:\n{smiles_code[:40]}")
                return
            drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
            drawer.drawOptions().addStereoAnnotation = True
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            png = drawer.GetDrawingText()
            qimg = QImage.fromData(png)
            pixmap = QPixmap.fromImage(qimg)
            self._smiles_fig.clear()
            ax = self._smiles_fig.add_subplot(111)
            ax.imshow(
                np.frombuffer(qimg.bits(), dtype=np.uint8).reshape(
                    qimg.height(), qimg.width(), 4
                )
            )
            ax.axis("off")
            self._smiles_canvas.draw()
        except Exception as e:
            self._show_smiles_message(f"SMILES error:\n{e}")

    # ------------------------------------------------------------------
    def load_data(self, parser: MGFParser, ids: List):
        self.parser = parser
        self.selected_spectrum_ids = ids
        self._populate_tree()
        self._update_smiles_display()

    def _populate_tree(self):
        self._metadata_tree.blockSignals(True)
        self._metadata_tree.clear()

        if not self.parser or not self.selected_spectrum_ids:
            self._metadata_tree.blockSignals(False)
            return

        spectra = [s for s in self.parser.spectra
                   if s.spectrum_id in self.selected_spectrum_ids]

        # Collect all keys and values
        all_keys = set()
        for s in spectra:
            all_keys.update(s.metadata.keys())

        key_data: Dict[str, List[str]] = {k: [] for k in all_keys}
        for s in spectra:
            for k in all_keys:
                val = s.metadata.get(k, "")
                if val is not None:
                    key_data[k].append(str(val))
                else:
                    key_data[k].append("")

        # Group keys
        def make_group_item(label: str, parent=None) -> QTreeWidgetItem:
            if parent is None:
                item = QTreeWidgetItem(self._metadata_tree)
            else:
                item = QTreeWidgetItem(parent)
            item.setText(0, label)
            item.setData(0, Qt.ItemDataRole.UserRole, None)
            font = item.font(0)
            font.setBold(True)
            item.setFont(0, font)
            return item

        def make_key_item(key: str, values: List[str], parent: QTreeWidgetItem):
            item = QTreeWidgetItem(parent)
            item.setText(0, key)
            item.setData(0, Qt.ItemDataRole.UserRole, key)
            unique_vals = sorted(set(v for v in values if v))
            display_val = values[0] if len(set(v for v in values if v)) <= 1 else "<multiple>"
            item.setText(1, display_val if display_val else "")
            item.setText(2, str(len(unique_vals)))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            return item

        grouped_keys: set = set()
        if self.metadata_groups:
            for group_def in self.metadata_groups:
                for group_name, keys in group_def.items():
                    group_item = make_group_item(group_name)
                    for k in keys:
                        if k in key_data:
                            make_key_item(k, key_data[k], group_item)
                            grouped_keys.add(k)

        remaining = sorted(all_keys - grouped_keys)
        if remaining:
            if self.metadata_groups:
                others_item = make_group_item(self._others_group_name)
                for k in remaining:
                    make_key_item(k, key_data[k], others_item)
            else:
                for k in remaining:
                    make_key_item(k, key_data[k], self._metadata_tree.invisibleRootItem())

        self._metadata_tree.expandAll()
        self._metadata_tree.blockSignals(False)

    def _update_smiles_display(self):
        if not self.parser or not self.selected_spectrum_ids:
            self._show_smiles_message("No spectra selected")
            return
        smiles_keys = ["smiles", "SMILES", "Smiles", "smiles_code", "SMILES_CODE"]
        found = None
        for sid in self.selected_spectrum_ids:
            spectrum = next((s for s in self.parser.spectra if s.spectrum_id == sid), None)
            if not spectrum:
                continue
            for k in smiles_keys:
                v = spectrum.get_metadata_value(k)
                if v and v.strip():
                    found = v.strip()
                    break
            if found:
                break
        if found:
            self._plot_smiles(found)
        else:
            self._show_smiles_message("No SMILES data found")

    def _on_double_click(self, item: QTreeWidgetItem, column: int):
        if item.data(0, Qt.ItemDataRole.UserRole) is None:
            return  # group node
        if column in (0, 1):
            self._metadata_tree.editItem(item, column)

    def _on_item_changed(self, item: QTreeWidgetItem, column: int):
        key_data = item.data(0, Qt.ItemDataRole.UserRole)
        if key_data is None:
            return
        if column == 1 and self.parser:
            new_value = item.text(1)
            original_key = item.text(0)
            for s in self.parser.spectra:
                if s.spectrum_id in self.selected_spectrum_ids:
                    s.metadata[original_key] = new_value
            if self._on_metadata_changed_cb:
                self._on_metadata_changed_cb()

    def _on_right_click(self, pos):
        item = self._metadata_tree.itemAt(pos)
        if not item:
            return
        key = item.data(0, Qt.ItemDataRole.UserRole)
        if key is None:
            return
        menu = QMenu(self)
        act_select = menu.addAction("Select by value")
        act_group = menu.addAction("Add as grouping tag")
        act_name = menu.addAction("Set as spectrum name")
        action = menu.exec(self._metadata_tree.viewport().mapToGlobal(pos))
        if action == act_select:
            self._select_by_value(item)
        elif action == act_group:
            self._add_as_grouping_tag(key)
        elif action == act_name:
            self._set_as_spectrum_name(key)

    def _select_by_value(self, item: QTreeWidgetItem):
        key = item.data(0, Qt.ItemDataRole.UserRole)
        value = item.text(1)
        if not self.parser or key is None:
            return
        matching = [s.spectrum_id for s in self.parser.spectra
                    if str(s.metadata.get(key, "")) == value]
        self._pending_selection = matching
        if self._on_metadata_changed_cb:
            self._on_metadata_changed_cb()

    def _add_as_grouping_tag(self, key: str):
        # Try to notify main app callback
        if self._on_metadata_changed_cb:
            self._on_metadata_changed_cb()

    def _set_as_spectrum_name(self, key: str):
        if self._on_metadata_changed_cb:
            self._on_metadata_changed_cb()

    def set_metadata_groups(self, groups: List[Dict[str, List[str]]]):
        self.metadata_groups = groups

    def get_pending_selection(self) -> Optional[List]:
        if self._pending_selection is not None:
            sel = self._pending_selection
            self._pending_selection = None
            return sel
        return None

    def clear(self):
        self._metadata_tree.clear()
        self._show_smiles_message("No spectra selected")
        self.parser = None
        self.selected_spectrum_ids = []
        self._pending_selection = None


# ---------------------------------------------------------------------------
# AddKeyValueDialog
# ---------------------------------------------------------------------------

class AddKeyValueDialog(QDialog):
    """Dialog for adding a new key-value pair to spectra."""

    def __init__(self, parent, parser: MGFParser, selected_spectrum_ids: List):
        super().__init__(parent)
        self.parser = parser
        self.selected_spectrum_ids = selected_spectrum_ids
        self.result = None
        self.setWindowTitle("Add New Key-Value Pair")
        self.setMinimumWidth(400)
        self._build_ui()
        self.exec()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        grid = QGridLayout()
        grid.addWidget(QLabel("Key Name:"), 0, 0)
        self._key_edit = QLineEdit()
        grid.addWidget(self._key_edit, 0, 1)

        grid.addWidget(QLabel("Value:"), 1, 0)
        self._value_edit = QLineEdit()
        grid.addWidget(self._value_edit, 1, 1)

        grid.addWidget(QLabel("Apply to:"), 2, 0)
        scope_widget = QWidget()
        scope_layout = QVBoxLayout(scope_widget)
        scope_layout.setContentsMargins(0, 0, 0, 0)
        self._rb_selected = QRadioButton(
            f"Selected spectra ({len(self.selected_spectrum_ids)})"
        )
        self._rb_all = QRadioButton(
            f"All loaded spectra ({len(self.parser.spectra)})"
        )
        self._rb_selected.setChecked(True)
        scope_layout.addWidget(self._rb_selected)
        scope_layout.addWidget(self._rb_all)
        grid.addWidget(scope_widget, 2, 1)
        layout.addLayout(grid)

        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        bb.accepted.connect(self._on_ok)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)
        self._key_edit.setFocus()

    def _on_ok(self):
        key = self._key_edit.text().strip()
        value = self._value_edit.text().strip()
        if not key:
            QMessageBox.critical(self, "Error", "Key name cannot be empty.")
            return
        existing = self.parser.get_all_metadata_keys()
        if key in existing:
            ans = QMessageBox.question(
                self, "Key Exists",
                f"Key '{key}' already exists. Update its value?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ans != QMessageBox.StandardButton.Yes:
                return
        if self._rb_selected.isChecked():
            if not self.selected_spectrum_ids:
                QMessageBox.warning(self, "Warning", "No spectra selected.")
                return
            target_ids = self.selected_spectrum_ids
        else:
            target_ids = [s.spectrum_id for s in self.parser.spectra]
        for s in self.parser.spectra:
            if s.spectrum_id in target_ids:
                s.metadata[key] = value
        self.result = True
        self.accept()


# ---------------------------------------------------------------------------
# RegexEditorDialog
# ---------------------------------------------------------------------------

class RegexEditorDialog(QDialog):
    """Dialog for regex-based bulk metadata editing."""

    def __init__(self, parent, parser: MGFParser):
        super().__init__(parent)
        self.parser = parser
        self.changes_made = False
        self.current_key: Optional[str] = None
        self.value_data: Dict = {}
        self.setWindowTitle("Regex Metadata Editor")
        self.resize(900, 600)
        self._build_ui()
        self._populate_keys()
        self.exec()

    def _build_ui(self):
        layout = QHBoxLayout(self)

        # Keys list
        keys_box = QGroupBox("Metadata Keys")
        keys_layout = QVBoxLayout(keys_box)
        self._keys_list = QListWidget()
        self._keys_list.currentRowChanged.connect(self._on_key_changed)
        keys_layout.addWidget(self._keys_list)
        layout.addWidget(keys_box, 1)

        # Values table
        values_box = QGroupBox("Values")
        values_layout = QVBoxLayout(values_box)
        self._values_tree = QTreeWidget()
        self._values_tree.setHeaderLabels(["Original Value", "Count", "Updated Value"])
        self._values_tree.setColumnWidth(0, 200)
        self._values_tree.setColumnWidth(1, 60)
        self._values_tree.setColumnWidth(2, 200)
        values_layout.addWidget(self._values_tree)
        layout.addWidget(values_box, 2)

        # Right panel: regex editor
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        regex_box = QGroupBox("Regex Replace")
        regex_inner = QGridLayout(regex_box)
        regex_inner.addWidget(QLabel("Pattern:"), 0, 0)
        self._pattern_edit = QLineEdit()
        regex_inner.addWidget(self._pattern_edit, 0, 1)
        regex_inner.addWidget(QLabel("Replacement:"), 1, 0)
        self._replacement_edit = QLineEdit()
        regex_inner.addWidget(self._replacement_edit, 1, 1)
        self._preview_btn = QPushButton("Preview")
        self._preview_btn.clicked.connect(self._preview_changes)
        regex_inner.addWidget(self._preview_btn, 2, 0)
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.clicked.connect(self._apply_changes)
        regex_inner.addWidget(self._apply_btn, 2, 1)
        right_layout.addWidget(regex_box)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        right_layout.addWidget(close_btn)
        right_layout.addStretch()

        layout.addWidget(right_panel, 1)

    def _populate_keys(self):
        if not self.parser:
            return
        for key in sorted(self.parser.get_all_metadata_keys()):
            self._keys_list.addItem(key)

    def _on_key_changed(self, row: int):
        if row < 0:
            return
        self.current_key = self._keys_list.item(row).text()
        self._populate_values()

    def _populate_values(self):
        self._values_tree.clear()
        if not self.current_key or not self.parser:
            return
        counts: Dict[str, int] = {}
        for s in self.parser.spectra:
            val = s.metadata.get(self.current_key, "")
            if val is not None:
                counts[str(val)] = counts.get(str(val), 0) + 1
        for val, cnt in sorted(counts.items()):
            item = QTreeWidgetItem(self._values_tree)
            item.setText(0, val)
            item.setText(1, str(cnt))
            item.setText(2, val)

    def _preview_changes(self):
        pattern = self._pattern_edit.text()
        replacement = self._replacement_edit.text()
        if not pattern or not self.current_key:
            return
        try:
            for i in range(self._values_tree.topLevelItemCount()):
                item = self._values_tree.topLevelItem(i)
                original = item.text(0)
                updated = re.sub(pattern, replacement, original)
                item.setText(2, updated)
        except re.error as e:
            QMessageBox.critical(self, "Regex Error", str(e))

    def _apply_changes(self):
        if not self.current_key or not self.parser:
            return
        mapping: Dict[str, str] = {}
        for i in range(self._values_tree.topLevelItemCount()):
            item = self._values_tree.topLevelItem(i)
            original = item.text(0)
            updated = item.text(2)
            if original != updated:
                mapping[original] = updated
        if not mapping:
            return
        for s in self.parser.spectra:
            val = s.metadata.get(self.current_key)
            if val is not None and str(val) in mapping:
                s.metadata[self.current_key] = mapping[str(val)]
        self.changes_made = True
        self._populate_values()


# ---------------------------------------------------------------------------
# SpectrumVisualization
# ---------------------------------------------------------------------------

class SpectrumVisualization(QWidget):
    """Spectrum stick-plot visualization widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parser: Optional[MGFParser] = None
        self.selected_spectrum_ids: List = []
        self.highlighted_ions: Dict = {}
        self.naming_scheme: str = "Numbered"
        self.axes: List = []
        self._syncing_zoom = False
        self._create_widgets()

    def _create_widgets(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Controls
        ctrl = QWidget()
        ctrl_layout = QHBoxLayout(ctrl)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)

        self._combined_cb = QCheckBox("Show Combined Plot")
        self._combined_cb.stateChanged.connect(self._plot_spectra)
        ctrl_layout.addWidget(self._combined_cb)

        ctrl_layout.addWidget(QLabel("PPM tolerance:"))
        self._ppm_spin = QDoubleSpinBox()
        self._ppm_spin.setRange(1.0, 500.0)
        self._ppm_spin.setValue(10.0)
        self._ppm_spin.valueChanged.connect(self._on_ppm_change)
        ctrl_layout.addWidget(self._ppm_spin)

        ctrl_layout.addWidget(QLabel("Top fragments:"))
        self._top_fragments_spin = QSpinBox()
        self._top_fragments_spin.setRange(1, 100)
        self._top_fragments_spin.setValue(10)
        self._top_fragments_spin.valueChanged.connect(self._on_fragments_count_change)
        ctrl_layout.addWidget(self._top_fragments_spin)

        popup_btn = QPushButton("Create Popup")
        popup_btn.clicked.connect(self._create_popup_window)
        ctrl_layout.addWidget(popup_btn)
        ctrl_layout.addStretch()
        layout.addWidget(ctrl)

        # Figure
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas, stretch=1)

        toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(toolbar)

    def _on_ppm_change(self):
        if self._combined_cb.isChecked():
            self._plot_spectra()

    def _on_fragments_count_change(self):
        if self._combined_cb.isChecked():
            self._plot_spectra()

    def set_naming_scheme(self, scheme: str):
        self.naming_scheme = scheme
        if self.parser and self.selected_spectrum_ids:
            self._plot_spectra()

    def load_data(self, parser: MGFParser, selected_spectrum_ids: List):
        self.parser = parser
        self.selected_spectrum_ids = selected_spectrum_ids
        self.highlighted_ions = {}
        self._plot_spectra()

    def highlight_ions(self, spectrum_id, ion_indices: List[int]):
        self.highlighted_ions[spectrum_id] = ion_indices
        self._plot_spectra()

    def clear_plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, "No spectra loaded", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        self.canvas.draw()
        self.parser = None
        self.selected_spectrum_ids = []

    def _get_spectrum_display_name(self, spectrum) -> str:
        if self.naming_scheme == "Numbered":
            return f"Spectrum {spectrum.spectrum_id}"
        val = spectrum.get_metadata_value(self.naming_scheme)
        return str(val) if val else f"Spectrum {spectrum.spectrum_id}"

    def _plot_spectra(self):
        self.figure.clear()
        self.axes = []
        if not self.parser or not self.selected_spectrum_ids:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No spectra selected", ha="center", va="center",
                    transform=ax.transAxes)
            ax.axis("off")
            self.canvas.draw()
            return

        selected_spectra = [
            s for s in self.parser.spectra
            if s.spectrum_id in self.selected_spectrum_ids
        ]
        if not selected_spectra:
            return

        if self._combined_cb.isChecked() and len(selected_spectra) > 1:
            self._plot_combined_spectra(selected_spectra)
        else:
            self._plot_individual_spectra(selected_spectra)

    def _plot_individual_spectra(self, selected_spectra):
        original_count = len(selected_spectra)
        warning = None
        if original_count > 10:
            selected_spectra = selected_spectra[:5]
            warning = f"Performance limit: Showing first 5 of {original_count} selected spectra"

        global_mz_min, global_mz_max = float("inf"), float("-inf")
        for s in selected_spectra:
            if s.ions.size > 0:
                global_mz_min = min(global_mz_min, s.ions[:, 0].min())
                global_mz_max = max(global_mz_max, s.ions[:, 0].max())
        if global_mz_min == float("inf"):
            global_mz_min, global_mz_max = 0, 1000
        else:
            rng = global_mz_max - global_mz_min
            pad = rng * 0.02
            global_mz_min -= pad
            global_mz_max += pad

        n = len(selected_spectra)
        for i, spectrum in enumerate(selected_spectra):
            ax = self.figure.add_subplot(n, 1, i + 1)
            self.axes.append(ax)
            self._plot_single_spectrum(ax, spectrum, (global_mz_min, global_mz_max),
                                       is_last=(i == n - 1))

        if len(self.axes) > 1:
            self._setup_zoom_synchronization()
        if warning:
            self.figure.suptitle(warning, fontsize=9, color="red", y=0.99)
        self.figure.tight_layout(pad=0.5, h_pad=0.2)
        self.figure.subplots_adjust(hspace=0.1)
        self.canvas.draw()

    def _plot_single_spectrum(self, ax, spectrum, mz_limits, is_last=True):
        if spectrum.ions.size == 0:
            ax.text(0.5, 0.5, "No ion data", ha="center", va="center",
                    transform=ax.transAxes)
            if mz_limits:
                ax.set_xlim(mz_limits)
            return
        mz = spectrum.ions[:, 0]
        intensity = spectrum.ions[:, 1]
        highlighted = self.highlighted_ions.get(spectrum.spectrum_id, [])
        for i, (m, inten) in enumerate(zip(mz, intensity)):
            color = "green" if i in highlighted else "blue"
            lw = 2.0 if i in highlighted else 1.5
            ax.vlines(m, 0, inten, colors=color, linewidth=lw)

        precursor = self._get_precursor_mass(spectrum)
        if precursor is not None:
            y_max = intensity.max() if len(intensity) > 0 else 1
            ax.axvline(precursor, color="grey", linestyle="--",
                       linewidth=1.5, alpha=0.7)
            ax.text(precursor, y_max * 1.05, f"M: {precursor:.2f}",
                    ha="center", va="bottom", fontsize=8, color="grey", rotation=90)

        if is_last:
            ax.set_xlabel("m/z")
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", which="both", bottom=False)
        label = self._get_spectrum_display_name(spectrum)
        ax.set_ylabel(f"{label}\nIntensity", fontsize=9)
        ax.grid(True, alpha=0.3)
        if mz_limits:
            ax.set_xlim(mz_limits)
        elif len(mz) > 0:
            ax.set_xlim(mz.min() * 0.95, mz.max() * 1.05)
        if len(intensity) > 0:
            ax.set_ylim(0, intensity.max() * 1.1)

    def _get_precursor_mass(self, spectrum) -> Optional[float]:
        for key in ["pepmass", "PEPMASS", "precursor_mass", "PRECURSOR_MASS"]:
            val = spectrum.get_metadata_value(key)
            if val:
                try:
                    return float(val.strip().split()[0])
                except (ValueError, IndexError):
                    continue
        return None

    def _plot_combined_spectra(self, selected_spectra):
        original_count = len(selected_spectra)
        warning = None
        if original_count > 10:
            selected_spectra = selected_spectra[:10]
            warning = f"Performance limit: Showing first 10 of {original_count} spectra"

        ax = self.figure.add_subplot(111)
        self.axes.append(ax)
        spectrum_colors = plt.cm.tab10(np.linspace(0, 1, min(len(selected_spectra), 10)))
        all_fragments: Dict = {}
        spectrum_info: Dict = {}

        for i, spectrum in enumerate(selected_spectra):
            if spectrum.ions.size == 0:
                continue
            mz = spectrum.ions[:, 0]
            intensity = spectrum.ions[:, 1]
            total = np.sum(intensity)
            rel = intensity / total if total > 0 else intensity
            spectrum_info[spectrum.spectrum_id] = {
                "color": spectrum_colors[i],
                "label": self._get_spectrum_display_name(spectrum),
            }
            for m, r in zip(mz, rel):
                all_fragments.setdefault(m, []).append((spectrum.spectrum_id, r))

        groups = self._group_fragments_by_mz(all_fragments, self._ppm_spin.value())
        fragment_intensities = []
        for group_mz, frags in groups.items():
            if len(frags) > 1:
                total = sum(r for _, r in frags)
                fragment_intensities.append((total, group_mz, frags))

        fragment_intensities.sort(key=lambda x: x[0], reverse=True)
        top = fragment_intensities[:self._top_fragments_spin.value()]

        sorted_ids = list(spectrum_info.keys())
        try:
            sorted_ids = natsorted(sorted_ids,
                                   key=lambda sid: spectrum_info[sid]["label"])
        except Exception:
            pass

        x_positions = {sid: i for i, sid in enumerate(sorted_ids)}
        for _, group_mz, frags in top:
            x_vals = [x_positions[sid] for sid, _ in frags if sid in x_positions]
            y_vals = [r for sid, r in frags if sid in x_positions]
            if len(x_vals) > 1:
                ax.plot(x_vals, y_vals, "o-", alpha=0.7, linewidth=1.5,
                        label=f"m/z {group_mz:.2f}")

        ax.set_xticks(range(len(sorted_ids)))
        ax.set_xticklabels(
            [spectrum_info[sid]["label"] for sid in sorted_ids],
            rotation=45, ha="right", fontsize=8
        )
        ax.set_xlabel("Spectrum")
        ax.set_ylabel("Relative Intensity (sum-scaled)")
        ax.set_title(f"Top {self._top_fragments_spin.value()} shared fragments")
        ax.grid(True, alpha=0.3)
        if top:
            ax.legend(fontsize=7, loc="upper right", ncol=2)
        if warning:
            self.figure.suptitle(warning, fontsize=9, color="red")
        self.figure.tight_layout()
        self.canvas.draw()

    def _group_fragments_by_mz(self, fragments: Dict, ppm_tolerance: float) -> Dict:
        sorted_mzs = sorted(fragments.keys())
        groups: Dict = {}
        used: set = set()
        for mz in sorted_mzs:
            if mz in used:
                continue
            group_mz = mz
            group_frags = list(fragments[mz])
            for other_mz in sorted_mzs:
                if other_mz == mz or other_mz in used:
                    continue
                ppm = abs(mz - other_mz) / mz * 1e6
                if ppm <= ppm_tolerance:
                    group_frags.extend(fragments[other_mz])
                    used.add(other_mz)
            groups[group_mz] = group_frags
            used.add(mz)
        return groups

    def _setup_zoom_synchronization(self):
        for ax in self.axes:
            ax.callbacks.connect("xlim_changed", self._on_xlims_change)

    def _on_xlims_change(self, ax):
        if self._syncing_zoom:
            return
        self._syncing_zoom = True
        try:
            xlims = ax.get_xlim()
            for other in self.axes:
                if other is not ax:
                    other.set_xlim(xlims)
            self.canvas.draw_idle()
        finally:
            self._syncing_zoom = False

    def _create_popup_window(self):
        if not self.parser or not self.selected_spectrum_ids:
            QMessageBox.information(self, "No Data", "No spectra selected.")
            return
        popup = SpectrumPopupWindow(self, self.parser, self.selected_spectrum_ids,
                                    self.naming_scheme)
        popup.show()


# ---------------------------------------------------------------------------
# IonDataTable
# ---------------------------------------------------------------------------

class IonDataTable(QWidget):
    """Tabbed ion data table."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parser: Optional[MGFParser] = None
        self.selected_spectrum_ids: List = []
        self.spectrum_viz_callback = None
        self._create_widgets()

    def _create_widgets(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        lbl = QLabel("Ion Data")
        lbl.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        layout.addWidget(lbl)
        self.notebook = QTabWidget()
        layout.addWidget(self.notebook, stretch=1)

    def set_spectrum_viz_callback(self, callback):
        self.spectrum_viz_callback = callback

    def load_data(self, parser: MGFParser, selected_spectrum_ids: List):
        self.parser = parser
        self.selected_spectrum_ids = selected_spectrum_ids
        self._populate_tables()

    def _populate_tables(self):
        self.notebook.clear()
        if not self.parser or not self.selected_spectrum_ids:
            return
        spectra = [s for s in self.parser.spectra
                   if s.spectrum_id in self.selected_spectrum_ids]
        for spectrum in spectra:
            self._create_table_for_spectrum(spectrum)

    def _create_table_for_spectrum(self, spectrum: "Spectrum"):
        tree = QTreeWidget()
        tree.setHeaderLabels(["Index", "m/z", "Intensity", "Rel. Intensity %", "Annotations"])
        tree.setColumnWidth(0, 70)
        tree.setColumnWidth(1, 110)
        tree.setColumnWidth(2, 110)
        tree.setColumnWidth(3, 110)
        tree.setColumnWidth(4, 260)
        tree.setAlternatingRowColors(True)
        tree.setSortingEnabled(True)
        tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        tree.customContextMenuRequested.connect(
            lambda pos, s=spectrum, t=tree: self._on_right_click(pos, s, t)
        )
        tree.itemSelectionChanged.connect(
            lambda s=spectrum, t=tree: self._on_ion_selection(s, t)
        )
        self._populate_table_data(tree, spectrum)
        self.notebook.addTab(tree, f"S {spectrum.spectrum_id}")

    def _populate_table_data(self, tree: QTreeWidget, spectrum: "Spectrum"):
        tree.clear()
        if spectrum.ions.size == 0:
            return
        mz = spectrum.ions[:, 0]
        intensity = spectrum.ions[:, 1]
        max_inten = intensity.max() if len(intensity) > 0 else 1.0

        for i, (m, inten) in enumerate(zip(mz, intensity)):
            rel = (inten / max_inten * 100) if max_inten > 0 else 0
            annotations = ""
            if hasattr(spectrum, "annotations") and spectrum.annotations:
                ann = spectrum.annotations.get(i, [])
                if ann:
                    annotations = "; ".join(
                        f"{a.get('formula','?')} [{a.get('ppm_error', 0):.1f}ppm]"
                        for a in ann
                    )
            item = QTreeWidgetItem()
            item.setData(0, Qt.ItemDataRole.UserRole, i)
            item.setText(0, str(i))
            item.setText(1, f"{m:.4f}")
            item.setText(2, f"{inten:.2f}")
            item.setText(3, f"{rel:.2f}")
            item.setText(4, annotations)
            tree.addTopLevelItem(item)

    def _on_ion_selection(self, spectrum: "Spectrum", tree: QTreeWidget):
        if self.spectrum_viz_callback is None:
            return
        selected_indices = [
            item.data(0, Qt.ItemDataRole.UserRole)
            for item in tree.selectedItems()
        ]
        self.spectrum_viz_callback(spectrum.spectrum_id, selected_indices)

    def _on_right_click(self, pos, spectrum: "Spectrum", tree: QTreeWidget):
        item = tree.itemAt(pos)
        if not item:
            return
        menu = QMenu(self)
        del_act = menu.addAction("Delete fragment")
        action = menu.exec(tree.viewport().mapToGlobal(pos))
        if action == del_act:
            idx = item.data(0, Qt.ItemDataRole.UserRole)
            if idx is not None and spectrum.ions.size > 0:
                spectrum.ions = np.delete(spectrum.ions, idx, axis=0)
                if hasattr(spectrum, "annotations") and spectrum.annotations:
                    spectrum.annotations.pop(idx, None)
                self._populate_table_data(tree, spectrum)

    def clear_data(self):
        self.notebook.clear()
        self.parser = None
        self.selected_spectrum_ids = []

    def clear(self):
        self.clear_data()


# ---------------------------------------------------------------------------
# CosineSimilarityVisualization
# ---------------------------------------------------------------------------

class CosineSimilarityVisualization(QWidget):
    """Heatmap visualization of cosine similarity between spectra."""

    MAX_SPECTRA_AUTO = 50
    MAX_SPECTRA_MANUAL = 500

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parser: Optional[MGFParser] = None
        self.selected_spectrum_ids: List = []
        self.similarity_matrix: Optional[np.ndarray] = None
        self.calculation_thread: Optional[threading.Thread] = None
        self.cancel_calculation = False
        self._calculation_error: Optional[str] = None
        self._create_widgets()

    def _create_widgets(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Controls
        ctrl = QWidget()
        ctrl_layout = QHBoxLayout(ctrl)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.addWidget(QLabel("Tolerance (Da):"))
        self._tolerance_spin = QDoubleSpinBox()
        self._tolerance_spin.setRange(0.0, 1.0)
        self._tolerance_spin.setValue(0.02)
        self._tolerance_spin.setSingleStep(0.01)
        self._tolerance_spin.valueChanged.connect(self._on_tolerance_changed)
        ctrl_layout.addWidget(self._tolerance_spin)

        self._calc_btn = QPushButton("Calculate")
        self._calc_btn.clicked.connect(self._recalculate_similarity)
        ctrl_layout.addWidget(self._calc_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel_calculation)
        ctrl_layout.addWidget(self._cancel_btn)

        self._progress_label = QLabel("")
        ctrl_layout.addWidget(self._progress_label)
        ctrl_layout.addStretch()
        layout.addWidget(ctrl)

        # Figure
        self.figure = Figure(figsize=(6, 5), dpi=90)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas, stretch=2)

        # Stats
        self._stats_text = QTextEdit()
        self._stats_text.setReadOnly(True)
        self._stats_text.setMaximumHeight(80)
        layout.addWidget(self._stats_text)

    def load_data(self, parser: MGFParser, selected_spectrum_ids: List):
        self.parser = parser
        self.selected_spectrum_ids = selected_spectrum_ids
        self.similarity_matrix = None
        self._stats_text.clear()
        if len(selected_spectrum_ids) <= self.MAX_SPECTRA_AUTO:
            self._calculate_and_display_similarity()
        else:
            self._show_large_dataset_warning()

    def clear_data(self):
        self.figure.clear()
        self._stats_text.clear()
        self.similarity_matrix = None
        self.selected_spectrum_ids = []
        self.canvas.draw()

    def _show_large_dataset_warning(self):
        n = len(self.selected_spectrum_ids)
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if n > self.MAX_SPECTRA_MANUAL:
            msg = (f"Too many spectra ({n}).\n"
                   f"Maximum: {self.MAX_SPECTRA_MANUAL}")
            self._calc_btn.setEnabled(False)
        else:
            msg = (f"Large selection ({n} spectra).\n"
                   f'Click "Calculate" to proceed.')
            self._calc_btn.setEnabled(True)
        ax.text(0.5, 0.5, msg, ha="center", va="center",
                transform=ax.transAxes, color="orange", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        self.canvas.draw()

    def _on_tolerance_changed(self):
        if len(self.selected_spectrum_ids) <= self.MAX_SPECTRA_AUTO:
            self._recalculate_similarity()

    def _recalculate_similarity(self):
        if not self.parser or not self.selected_spectrum_ids:
            return
        if len(self.selected_spectrum_ids) > self.MAX_SPECTRA_MANUAL:
            QMessageBox.warning(self, "Too Many Spectra",
                                f"Cannot calculate for {len(self.selected_spectrum_ids)} spectra.\n"
                                f"Maximum: {self.MAX_SPECTRA_MANUAL}")
            return
        self._calculate_and_display_similarity()

    def _calculate_and_display_similarity(self):
        self.figure.clear()
        self._stats_text.clear()
        if not self.parser or len(self.selected_spectrum_ids) < 2:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "Select 2+ spectra for comparison",
                    ha="center", va="center", transform=ax.transAxes)
            self.canvas.draw()
            return
        if len(self.selected_spectrum_ids) > 10:
            self._start_threaded_calculation()
        else:
            self._calculate_similarity_direct()

    def _start_threaded_calculation(self):
        self._calc_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._progress_label.setText("Calculating...")
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, "Calculating...\nPlease wait",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        self.canvas.draw()
        self.cancel_calculation = False
        self._calculation_error = None
        self.calculation_thread = threading.Thread(
            target=self._calculate_similarity_threaded, daemon=True
        )
        self.calculation_thread.start()
        self._check_calculation_progress()

    def _calculate_similarity_threaded(self):
        try:
            tolerance = self._tolerance_spin.value()
            spectra = [s for s in self.parser.spectra
                       if s.spectrum_id in self.selected_spectrum_ids]
            n = len(spectra)
            if n < 2:
                self.similarity_matrix = np.array([[1.0]] if n == 1 else [])
                return
            sim_matrix = np.zeros((n, n))
            total = n * (n - 1) // 2
            completed = 0
            for i in range(n):
                sim_matrix[i, i] = 1.0
            for i in range(n):
                if self.cancel_calculation:
                    return
                for j in range(i + 1, n):
                    if self.cancel_calculation:
                        return
                    sim = self.parser.calculate_cosine_similarity(
                        spectra[i], spectra[j], tolerance
                    )
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim
                    completed += 1
                    if completed % max(1, total // 20) == 0:
                        pct = completed / total * 100
                        QTimer.singleShot(
                            0,
                            lambda p=pct: self._progress_label.setText(
                                f"Calculating... {p:.0f}%"
                            )
                        )
            if not self.cancel_calculation:
                self.similarity_matrix = sim_matrix
        except Exception as e:
            self._calculation_error = str(e)

    def _check_calculation_progress(self):
        if self.calculation_thread and self.calculation_thread.is_alive():
            QTimer.singleShot(100, self._check_calculation_progress)
        else:
            self._reset_ui_after_calculation()
            if self._calculation_error:
                self._show_calculation_error(self._calculation_error)
                self._calculation_error = None
            elif self.similarity_matrix is not None:
                self._display_similarity_results()

    def _calculate_similarity_direct(self):
        try:
            tolerance = self._tolerance_spin.value()
            self.similarity_matrix = self.parser.calculate_similarity_matrix(
                self.selected_spectrum_ids, tolerance
            )
            self._display_similarity_results()
        except Exception as e:
            self._show_calculation_error(str(e))

    def _cancel_calculation(self):
        self.cancel_calculation = True

    def _reset_ui_after_calculation(self):
        self.cancel_calculation = False
        self._calc_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._progress_label.setText("")

    def _show_calculation_error(self, msg: str):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, f"Error:\n{msg}", ha="center", va="center",
                transform=ax.transAxes, color="red")
        ax.set_xticks([]); ax.set_yticks([])
        self.canvas.draw()

    def _display_similarity_results(self):
        if self.similarity_matrix is None or self.similarity_matrix.size == 0:
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        im = ax.imshow(self.similarity_matrix, cmap="viridis", vmin=0, vmax=1,
                       aspect="equal")
        cbar = self.figure.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Cosine Similarity", rotation=270, labelpad=15)
        labels = [f"S{sid}" for sid in self.selected_spectrum_ids]
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        n = self.similarity_matrix.shape[0]
        if n <= 20:
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, f"{self.similarity_matrix[i, j]:.3f}",
                            ha="center", va="center", color="white", fontsize=8)
        tol = self._tolerance_spin.value()
        ax.set_title(f"Similarity Matrix (tolerance: {tol:.2f})")
        self.figure.tight_layout()
        self.canvas.draw()
        self._display_statistics()

    def _display_statistics(self):
        if self.similarity_matrix is None:
            return
        n = self.similarity_matrix.shape[0]
        if n < 2:
            return
        triu = np.triu_indices(n, k=1)
        sims = self.similarity_matrix[triu]
        if len(sims) == 0:
            return
        pcts = np.percentile(sims, [0, 10, 25, 50, 75, 90, 100])
        txt = (f"Pairwise Similarities (n={len(sims)}):\n"
               f"Min: {pcts[0]:.4f}  10%: {pcts[1]:.4f}  25%: {pcts[2]:.4f}\n"
               f"Median: {pcts[3]:.4f}  75%: {pcts[4]:.4f}  90%: {pcts[5]:.4f}\n"
               f"Max: {pcts[6]:.4f}  Mean: {np.mean(sims):.4f}")
        self._stats_text.setPlainText(txt)


# ---------------------------------------------------------------------------
# FileLoadingDialog
# ---------------------------------------------------------------------------

class FileLoadingDialog(QDialog):
    """Dialog for file loading options: database identifier and prefix."""

    def __init__(self, parent, file_to_load, existing_prefixes=None):
        super().__init__(parent)
        self.file_to_load = file_to_load
        self.existing_prefixes = set(existing_prefixes or [])
        self.result = None
        self.setWindowTitle("File Loading Options")
        self.setMinimumWidth(450)
        self._build_ui()
        self.exec()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        lbl = QLabel(
            f"Configure options for:\n'{self.file_to_load}'\n(* = required)"
        )
        layout.addWidget(lbl)

        db_box = QGroupBox("Database Information")
        db_layout = QVBoxLayout(db_box)
        db_layout.addWidget(QLabel("Database Identifier: *"))
        self._db_edit = QLineEdit()
        self._db_edit.textChanged.connect(self._validate)
        db_layout.addWidget(self._db_edit)
        layout.addWidget(db_box)

        prefix_box = QGroupBox("Spectrum ID Prefix")
        prefix_layout = QVBoxLayout(prefix_box)
        prefix_layout.addWidget(QLabel("Prefix for spectrum IDs: *"))
        self._prefix_edit = QLineEdit()
        self._prefix_edit.textChanged.connect(self._validate)
        prefix_layout.addWidget(self._prefix_edit)
        self._prefix_status = QLabel("")
        prefix_layout.addWidget(self._prefix_status)
        layout.addWidget(prefix_box)

        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._ok_btn = bb.button(QDialogButtonBox.StandardButton.Ok)
        self._ok_btn.setEnabled(False)
        bb.accepted.connect(self._on_ok)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)
        self._db_edit.setFocus()

    def _validate(self):
        prefix = self._prefix_edit.text().strip()
        db_id = self._db_edit.text().strip()
        if not prefix:
            self._prefix_status.setText("⚠ Prefix is required")
            self._ok_btn.setEnabled(False)
        elif prefix in self.existing_prefixes:
            self._prefix_status.setText("⚠ Prefix already used")
            self._ok_btn.setEnabled(False)
        else:
            self._prefix_status.setText("✓ Available")
            self._ok_btn.setEnabled(bool(db_id))

    def _on_ok(self):
        prefix = self._prefix_edit.text().strip()
        db_id = self._db_edit.text().strip()
        if not db_id:
            QMessageBox.critical(self, "Missing", "Database identifier is required.")
            return
        if not prefix:
            QMessageBox.critical(self, "Missing", "Prefix is required.")
            return
        if prefix in self.existing_prefixes:
            QMessageBox.critical(self, "Invalid", f"Prefix '{prefix}' already used.")
            return
        self.result = {"database_identifier": db_id, "prefix": prefix}
        self.accept()


# ---------------------------------------------------------------------------
# AverageSpectrumDialog
# ---------------------------------------------------------------------------

class AverageSpectrumDialog(QDialog):
    """Dialog for average spectrum parameters."""

    def __init__(self, parent):
        super().__init__(parent)
        self.result = None
        self.setWindowTitle("Calculate Average Spectrum per Group")
        self.setMinimumWidth(420)
        self._build_ui()
        self.exec()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("Average Spectrum Parameters")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title)

        params_box = QGroupBox("Parameters")
        grid = QGridLayout(params_box)

        grid.addWidget(QLabel("Binning m/z tolerance:"), 0, 0)
        self._binning_edit = QDoubleSpinBox()
        self._binning_edit.setRange(0.001, 10.0)
        self._binning_edit.setValue(0.1)
        self._binning_edit.setDecimals(3)
        grid.addWidget(self._binning_edit, 0, 1)

        grid.addWidget(QLabel("Averaging method:"), 1, 0)
        self._method_combo = QComboBox()
        self._method_combo.addItems(["average", "median"])
        grid.addWidget(self._method_combo, 1, 1)

        grid.addWidget(QLabel("New metadata key:"), 2, 0)
        self._key_edit = QLineEdit("group_average")
        grid.addWidget(self._key_edit, 2, 1)

        grid.addWidget(QLabel("New metadata value:"), 3, 0)
        self._value_edit = QLineEdit("averaged_spectrum")
        grid.addWidget(self._value_edit, 3, 1)

        layout.addWidget(params_box)

        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        bb.accepted.connect(self._on_ok)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

    def _on_ok(self):
        try:
            binning = self._binning_edit.value()
        except Exception:
            QMessageBox.critical(self, "Error", "Invalid binning value.")
            return
        self.result = {
            "binning_mz": binning,
            "averaging_method": self._method_combo.currentText(),
            "new_key": self._key_edit.text().strip(),
            "new_value": self._value_edit.text().strip(),
        }
        self.accept()


# ---------------------------------------------------------------------------
# SmartsFilterDialog
# ---------------------------------------------------------------------------

class SmartsFilterDialog:
    """SMARTS substructure filter dialog (non-modal)."""

    def __init__(self, parent, spectra, apply_callback):
        self.parent = parent
        self.spectra = spectra
        self.apply_callback = apply_callback
        self.matching_spectra: List = []
        self._dialog = QDialog(parent)
        self._dialog.setWindowTitle("SMARTS Substructure Filter")
        self._dialog.resize(1000, 700)
        self._examples = [
            ("Benzene ring", "c1ccccc1"),
            ("Carbonyl", "C=O"),
            ("Hydroxyl", "O"),
            ("Ester", "C(=O)O"),
        ]
        self._build_ui()
        self._dialog.show()

    def _build_ui(self):
        layout = QVBoxLayout(self._dialog)

        # SMARTS input
        input_box = QGroupBox("SMARTS Pattern")
        input_layout = QVBoxLayout(input_box)
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Enter SMARTS pattern(s) (space-separated for OR):"))
        ex_btn = QPushButton("Examples")
        ex_btn.clicked.connect(self._show_examples_menu)
        top_row.addWidget(ex_btn)
        input_layout.addLayout(top_row)
        self._smarts_edit = QLineEdit()
        self._smarts_edit.setFont(QFont("Courier", 12))
        self._smarts_edit.textChanged.connect(self._on_smarts_change)
        input_layout.addWidget(self._smarts_edit)
        layout.addWidget(input_box)

        # Results
        results_box = QGroupBox("Results")
        results_layout = QHBoxLayout(results_box)
        self._matched_list = QListWidget()
        matched_group = QGroupBox("Matched (0)")
        ml = QVBoxLayout(matched_group)
        ml.addWidget(self._matched_list)
        results_layout.addWidget(matched_group)
        self._matched_group_label = matched_group

        self._unmatched_list = QListWidget()
        unmatched_group = QGroupBox("Unmatched (0)")
        ul = QVBoxLayout(unmatched_group)
        ul.addWidget(self._unmatched_list)
        results_layout.addWidget(unmatched_group)
        self._unmatched_group_label = unmatched_group
        layout.addWidget(results_box)

        # Buttons
        btn_row = QHBoxLayout()
        self._status_lbl = QLabel("Enter a SMARTS pattern to begin")
        btn_row.addWidget(self._status_lbl, stretch=1)
        apply_btn = QPushButton("Apply Filter")
        apply_btn.clicked.connect(self._apply_filter)
        btn_row.addWidget(apply_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self._dialog.close)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    def _show_examples_menu(self):
        menu = QMenu(self._dialog)
        for name, pattern in self._examples:
            act = menu.addAction(f"{name} ({pattern})")
            act.triggered.connect(lambda checked, p=pattern: self._set_pattern(p))
        menu.exec(QCursor.pos())

    def _set_pattern(self, pattern: str):
        self._smarts_edit.setText(pattern)
        self._perform_filtering(pattern)

    def _on_smarts_change(self, text: str):
        if not text.strip():
            self._matched_list.clear()
            self._unmatched_list.clear()
            return

    def _perform_filtering(self, pattern: str):
        if not RDKIT_AVAILABLE:
            self._status_lbl.setText("RDKit not available")
            return
        patterns_list = [p.strip() for p in pattern.split() if p.strip()]
        query_mols = []
        for p in patterns_list:
            mol = Chem.MolFromSmarts(p)
            if mol:
                query_mols.append(mol)
        if not query_mols:
            self._status_lbl.setText("Invalid SMARTS pattern")
            return

        self.matching_spectra = []
        unmatched = []
        smiles_keys = ["smiles", "SMILES", "Smiles"]
        for s in self.spectra:
            smiles = None
            for k in smiles_keys:
                v = s.metadata.get(k)
                if v and str(v).strip():
                    smiles = str(v).strip()
                    break
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    matched = any(mol.HasSubstructMatch(q) for q in query_mols)
                    if matched:
                        self.matching_spectra.append(s)
                    else:
                        unmatched.append(s)
                else:
                    unmatched.append(s)
            else:
                unmatched.append(s)

        self._matched_list.clear()
        for s in self.matching_spectra:
            self._matched_list.addItem(f"Spectrum {s.spectrum_id}")
        self._unmatched_list.clear()
        for s in unmatched:
            self._unmatched_list.addItem(f"Spectrum {s.spectrum_id}")
        self._matched_group_label.setTitle(f"Matched ({len(self.matching_spectra)})")
        self._unmatched_group_label.setTitle(f"Unmatched ({len(unmatched)})")
        self._status_lbl.setText(
            f"Found {len(self.matching_spectra)} matching spectra"
        )

    def _apply_filter(self):
        pattern = self._smarts_edit.text().strip()
        if pattern:
            self._perform_filtering(pattern)
        if self.apply_callback:
            self.apply_callback(self.matching_spectra)
        self._dialog.close()

    def destroy(self):
        self._dialog.close()


# ---------------------------------------------------------------------------
# IntensityFilterDialog
# ---------------------------------------------------------------------------

class IntensityFilterDialog:
    """Intensity filter dialog (non-modal)."""

    def __init__(self, parent, spectra, apply_callback):
        self.parent = parent
        self.spectra = spectra
        self.apply_callback = apply_callback
        self._dialog = QDialog(parent)
        self._dialog.setWindowTitle("Intensity Filter")
        self._dialog.resize(450, 350)
        self._build_ui()
        self._dialog.show()

    def _build_ui(self):
        layout = QVBoxLayout(self._dialog)

        # Global threshold
        gt_box = QGroupBox("Global Threshold")
        gt_layout = QGridLayout(gt_box)
        self._gt_enable = QCheckBox("Enable")
        gt_layout.addWidget(self._gt_enable, 0, 0)
        gt_layout.addWidget(QLabel("Min intensity:"), 1, 0)
        self._gt_spin = QDoubleSpinBox()
        self._gt_spin.setRange(0.0, 1e12)
        self._gt_spin.setValue(0.0)
        gt_layout.addWidget(self._gt_spin, 1, 1)
        layout.addWidget(gt_box)

        # Relative to max
        rm_box = QGroupBox("Relative to Maximum")
        rm_layout = QGridLayout(rm_box)
        self._rm_enable = QCheckBox("Enable")
        rm_layout.addWidget(self._rm_enable, 0, 0)
        rm_layout.addWidget(QLabel("Min % of max:"), 1, 0)
        self._rm_spin = QDoubleSpinBox()
        self._rm_spin.setRange(0.0, 100.0)
        self._rm_spin.setValue(1.0)
        rm_layout.addWidget(self._rm_spin, 1, 1)
        layout.addWidget(rm_box)

        # Relative to sum
        rs_box = QGroupBox("Relative to Sum")
        rs_layout = QGridLayout(rs_box)
        self._rs_enable = QCheckBox("Enable")
        rs_layout.addWidget(self._rs_enable, 0, 0)
        rs_layout.addWidget(QLabel("Min % of sum:"), 1, 0)
        self._rs_spin = QDoubleSpinBox()
        self._rs_spin.setRange(0.0, 100.0)
        self._rs_spin.setValue(0.1)
        rs_layout.addWidget(self._rs_spin, 1, 1)
        layout.addWidget(rs_box)

        # Buttons
        btn_row = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._apply)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self._dialog.close)
        btn_row.addStretch()
        btn_row.addWidget(apply_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    def _apply(self):
        for spectrum in self.spectra:
            if spectrum.ions.size == 0:
                continue
            ions = spectrum.ions
            mask = np.ones(len(ions), dtype=bool)
            intensities = ions[:, 1]
            if self._gt_enable.isChecked():
                mask &= intensities >= self._gt_spin.value()
            if self._rm_enable.isChecked():
                max_inten = intensities.max()
                if max_inten > 0:
                    mask &= (intensities / max_inten * 100) >= self._rm_spin.value()
            if self._rs_enable.isChecked():
                total = intensities.sum()
                if total > 0:
                    mask &= (intensities / total * 100) >= self._rs_spin.value()
            spectrum.ions = ions[mask]
        if self.apply_callback:
            self.apply_callback()
        self._dialog.close()

    def destroy(self):
        self._dialog.close()


# ---------------------------------------------------------------------------
# FragmentAnnotationDialog
# ---------------------------------------------------------------------------

class FragmentAnnotationDialog:
    """Dialog for fragment annotation configuration."""

    def __init__(self, parent, callback=None, spectra=None):
        self.parent = parent
        self.callback = callback
        self.spectra = spectra or []
        self.result = None
        self._window: Optional[QDialog] = None

    def show(self):
        self._window = QDialog(self.parent)
        self._window.setWindowTitle("Fragment Annotation")
        self._window.resize(600, 500)
        self._build_ui()
        self._window.exec()
        return self.result

    def _build_ui(self):
        layout = QVBoxLayout(self._window)

        # Formula tags
        tags_box = QGroupBox("Formula Tags (comma-separated)")
        tags_layout = QVBoxLayout(tags_box)
        self._formula_tags_edit = QLineEdit("H2O, CO, CO2, NH3")
        tags_layout.addWidget(self._formula_tags_edit)
        layout.addWidget(tags_box)

        # PPM tolerance
        ppm_box = QGroupBox("PPM Tolerance")
        ppm_layout = QHBoxLayout(ppm_box)
        ppm_layout.addWidget(QLabel("PPM tolerance:"))
        self._ppm_spin = QDoubleSpinBox()
        self._ppm_spin.setRange(0.1, 1000.0)
        self._ppm_spin.setValue(10.0)
        ppm_layout.addWidget(self._ppm_spin)
        ppm_layout.addStretch()
        layout.addWidget(ppm_box)

        # Additional elements
        elem_box = QGroupBox("Additional Elements")
        elem_layout = QVBoxLayout(elem_box)
        self._add_elements_edit = QLineEdit("")
        self._add_elements_edit.setPlaceholderText("e.g. Na, K (optional)")
        elem_layout.addWidget(self._add_elements_edit)
        layout.addWidget(elem_box)

        # Max workers
        workers_box = QGroupBox("Processing")
        workers_layout = QHBoxLayout(workers_box)
        workers_layout.addWidget(QLabel("Max workers:"))
        self._workers_spin = QSpinBox()
        self._workers_spin.setRange(1, 32)
        self._workers_spin.setValue(4)
        workers_layout.addWidget(self._workers_spin)
        workers_layout.addStretch()
        layout.addWidget(workers_box)

        # PPM function points
        points_box = QGroupBox("PPM Function Points (m/z:ppm, comma-separated)")
        points_layout = QVBoxLayout(points_box)
        self._points_edit = QLineEdit("")
        self._points_edit.setPlaceholderText("e.g. 100:20, 500:10, 1000:5")
        points_layout.addWidget(self._points_edit)
        layout.addWidget(points_box)

        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        bb.accepted.connect(self._on_ok)
        bb.rejected.connect(self._window.reject)
        layout.addWidget(bb)

    def _on_ok(self):
        tags_raw = self._formula_tags_edit.text()
        formula_tags = [t.strip() for t in tags_raw.split(",") if t.strip()]

        extra_raw = self._add_elements_edit.text()
        additional_elements = [e.strip() for e in extra_raw.split(",") if e.strip()]

        points_raw = self._points_edit.text().strip()
        ppm_points = []
        if points_raw:
            try:
                for pair in points_raw.split(","):
                    mz_str, ppm_str = pair.strip().split(":")
                    ppm_points.append((float(mz_str.strip()), float(ppm_str.strip())))
            except Exception:
                ppm_points = []

        self.result = {
            "formula_tags": formula_tags,
            "ppm_tolerance": self._ppm_spin.value(),
            "additional_elements": additional_elements,
            "ppm_function_points": ppm_points,
            "max_workers": self._workers_spin.value(),
        }
        if self.callback:
            self.callback(self.result)
        self._window.accept()


# ---------------------------------------------------------------------------
# CanonicalSmilesDialog
# ---------------------------------------------------------------------------

class CanonicalSmilesDialog:
    """Non-modal dialog showing canonical SMILES results."""

    def __init__(self, parent):
        self.parent = parent
        self._window: Optional[QDialog] = None
        self._on_key_change = None

    def show(self, rows, summary_text=None, initial_key: str = "smiles",
             on_key_change=None):
        self._on_key_change = on_key_change
        if self._window and not self._window.isHidden():
            self._window.close()
        self._window = QDialog(self.parent)
        self._window.setWindowTitle("Canonical SMILES")
        self._window.resize(900, 500)
        layout = QVBoxLayout(self._window)

        if summary_text:
            lbl = QLabel(summary_text)
            lbl.setWordWrap(True)
            layout.addWidget(lbl)

        # Key selector
        key_row = QHBoxLayout()
        key_row.addWidget(QLabel("Metadata key:"))
        self._key_edit = QLineEdit(initial_key)
        self._key_edit.returnPressed.connect(self._on_key_changed)
        key_row.addWidget(self._key_edit)
        update_btn = QPushButton("Update")
        update_btn.clicked.connect(self._on_key_changed)
        key_row.addWidget(update_btn)
        key_row.addStretch()
        layout.addLayout(key_row)

        # Table
        self._table = QTreeWidget()
        self._table.setHeaderLabels(["Spectrum ID", "Original SMILES",
                                     "Canonical SMILES", "Error"])
        self._table.setColumnWidth(0, 100)
        self._table.setColumnWidth(1, 220)
        self._table.setColumnWidth(2, 220)
        self._table.setColumnWidth(3, 200)
        for row in rows:
            item = QTreeWidgetItem()
            item.setText(0, str(row.get("spectrum_id", "")))
            item.setText(1, str(row.get("original", "")))
            item.setText(2, str(row.get("canonical", "")))
            item.setText(3, str(row.get("error", "")))
            if row.get("highlighted"):
                for col in range(4):
                    item.setBackground(col, Qt.GlobalColor.yellow)
            self._table.addTopLevelItem(item)
        layout.addWidget(self._table)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self._window.close)
        layout.addWidget(close_btn)
        self._window.show()

    def _on_key_changed(self):
        if self._on_key_change:
            self._on_key_change(self._key_edit.text().strip())

    def close(self):
        if self._window:
            self._window.close()


# ---------------------------------------------------------------------------
# ProgressDialog
# ---------------------------------------------------------------------------

class ProgressDialog:
    """Modal progress dialog."""

    def __init__(self, parent, title: str = "Processing...",
                 message: str = "Please wait..."):
        self.parent = parent
        self._title = title
        self._message = message
        self.cancelled = False
        self._dialog: Optional[QDialog] = None

    def show(self, max_value: int = 100):
        self._dialog = QDialog(self.parent)
        self._dialog.setWindowTitle(self._title)
        self._dialog.setModal(True)
        self._dialog.setMinimumWidth(350)
        layout = QVBoxLayout(self._dialog)
        self._msg_label = QLabel(self._message)
        self._msg_label.setWordWrap(True)
        layout.addWidget(self._msg_label)
        self._bar = QProgressBar()
        self._bar.setRange(0, max_value)
        self._bar.setValue(0)
        layout.addWidget(self._bar)
        self._status_label = QLabel("")
        layout.addWidget(self._status_label)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self._on_cancel)
        layout.addWidget(cancel_btn)
        self._dialog.show()
        QApplication.processEvents()

    def update_progress(self, value: int, status_text: str = ""):
        if self._dialog is None:
            return
        self._bar.setValue(value)
        if status_text:
            self._status_label.setText(status_text)
        QApplication.processEvents()

    def _on_cancel(self):
        self.cancelled = True

    def close(self):
        if self._dialog:
            self._dialog.close()
            self._dialog = None

    def is_cancelled(self) -> bool:
        return self.cancelled


# ---------------------------------------------------------------------------
# PPMDeviationPlotDialog
# ---------------------------------------------------------------------------

class PPMDeviationPlotDialog:
    """Dialog showing a PPM deviation scatter plot for annotated fragments."""

    def __init__(self, parent):
        self.parent = parent

    def show(self, annotated_data):
        dialog = QDialog(self.parent)
        dialog.setWindowTitle("PPM Deviation Plot")
        dialog.resize(800, 600)
        layout = QVBoxLayout(dialog)

        fig = Figure(figsize=(8, 5), dpi=100)
        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas)
        toolbar = NavigationToolbar2QT(canvas, dialog)
        layout.addWidget(toolbar)

        # Stats panel
        stats_edit = QTextEdit()
        stats_edit.setReadOnly(True)
        stats_edit.setMaximumHeight(80)
        layout.addWidget(stats_edit)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)

        ax = fig.add_subplot(111)
        if annotated_data:
            all_mz = []
            all_ppm = []
            all_rank = []
            for entry in annotated_data:
                mz = entry.get("mz", 0)
                ppm = entry.get("ppm_error", 0)
                rank = entry.get("annotation_rank", 0)
                all_mz.append(mz)
                all_ppm.append(ppm)
                all_rank.append(rank)
            sc = ax.scatter(all_mz, all_ppm, c=all_rank, cmap="coolwarm",
                            alpha=0.7, s=20)
            fig.colorbar(sc, ax=ax, label="Annotation Rank")
            ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
            ax.set_xlabel("m/z")
            ax.set_ylabel("PPM Error")
            ax.set_title("PPM Deviation vs m/z")
            ax.grid(True, alpha=0.3)
            ppm_arr = np.array(all_ppm)
            stats_edit.setPlainText(
                f"N={len(ppm_arr)}  Mean PPM={np.mean(ppm_arr):.2f}"
                f"  Std={np.std(ppm_arr):.2f}  "
                f"Min={np.min(ppm_arr):.2f}  Max={np.max(ppm_arr):.2f}"
            )
        else:
            ax.text(0.5, 0.5, "No annotated data available",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
        canvas.draw()
        dialog.exec()


# ---------------------------------------------------------------------------
# SpectrumVisualizationPopup
# ---------------------------------------------------------------------------

class SpectrumVisualizationPopup(QWidget):
    """Simplified spectrum visualization for popup windows."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.naming_scheme: str = "Numbered"
        self.parser: Optional[MGFParser] = None
        self.selected_spectrum_ids: List = []
        self._syncing_zoom = False
        self.axes: List = []
        self._create_widgets()

    def _create_widgets(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.figure = Figure(figsize=(8, 6), dpi=90)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas, stretch=1)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(toolbar)

    def load_data(self, parser: MGFParser, selected_spectrum_ids: List):
        self.parser = parser
        self.selected_spectrum_ids = selected_spectrum_ids
        self._plot_spectra()

    def _get_display_name(self, spectrum) -> str:
        if self.naming_scheme == "Numbered":
            return f"Spectrum {spectrum.spectrum_id}"
        val = spectrum.get_metadata_value(self.naming_scheme)
        return str(val) if val else f"Spectrum {spectrum.spectrum_id}"

    def _plot_spectra(self):
        self.figure.clear()
        self.axes = []
        if not self.parser or not self.selected_spectrum_ids:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No spectra selected", ha="center", va="center",
                    transform=ax.transAxes)
            ax.axis("off")
            self.canvas.draw()
            return
        spectra = [s for s in self.parser.spectra
                   if s.spectrum_id in self.selected_spectrum_ids]
        if not spectra:
            return
        original_count = len(spectra)
        if original_count > 10:
            spectra = spectra[:5]
        global_mz_min, global_mz_max = float("inf"), float("-inf")
        for s in spectra:
            if s.ions.size > 0:
                global_mz_min = min(global_mz_min, s.ions[:, 0].min())
                global_mz_max = max(global_mz_max, s.ions[:, 0].max())
        if global_mz_min == float("inf"):
            global_mz_min, global_mz_max = 0, 1000
        n = len(spectra)
        for i, spectrum in enumerate(spectra):
            ax = self.figure.add_subplot(n, 1, i + 1)
            self.axes.append(ax)
            if spectrum.ions.size > 0:
                mz = spectrum.ions[:, 0]
                intensity = spectrum.ions[:, 1]
                ax.vlines(mz, 0, intensity, colors="blue", linewidth=1.5)
                if len(mz) > 0:
                    ax.set_xlim(global_mz_min * 0.98, global_mz_max * 1.02)
                if len(intensity) > 0:
                    ax.set_ylim(0, intensity.max() * 1.1)
            else:
                ax.text(0.5, 0.5, "No ion data", ha="center", va="center",
                        transform=ax.transAxes)
            label = self._get_display_name(spectrum)
            ax.set_ylabel(f"{label}\nIntensity", fontsize=8)
            if i == n - 1:
                ax.set_xlabel("m/z")
            else:
                ax.set_xticklabels([])
            ax.grid(True, alpha=0.3)
        self.figure.tight_layout(pad=0.5, h_pad=0.2)
        self.figure.subplots_adjust(hspace=0.1)
        self.canvas.draw()


# ---------------------------------------------------------------------------
# SpectrumPopupWindow
# ---------------------------------------------------------------------------

class SpectrumPopupWindow:
    """Non-modal popup window showing spectrum plot + metadata."""

    def __init__(self, parent, parser: MGFParser, selected_spectrum_ids: List,
                 naming_scheme: str = "Numbered"):
        self.parent = parent
        self.parser = parser
        self.selected_spectrum_ids = selected_spectrum_ids
        self.naming_scheme = naming_scheme
        self._window: Optional[QDialog] = None

    def show(self):
        self._window = QDialog(self.parent)
        self._window.setWindowTitle("Spectrum Popup")
        self._window.resize(1100, 700)
        self._window.setModal(False)
        layout = QHBoxLayout(self._window)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left: visualization
        viz = SpectrumVisualizationPopup()
        viz.naming_scheme = self.naming_scheme
        viz.load_data(self.parser, self.selected_spectrum_ids)
        splitter.addWidget(viz)

        # Right: text info
        right = QWidget()
        right_layout = QVBoxLayout(right)
        meta_edit = QTextEdit()
        meta_edit.setReadOnly(True)
        right_layout.addWidget(meta_edit)
        ion_edit = QTextEdit()
        ion_edit.setReadOnly(True)
        right_layout.addWidget(ion_edit)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        # Populate text
        meta_lines = []
        ion_lines = []
        for sid in self.selected_spectrum_ids:
            spectrum = next(
                (s for s in self.parser.spectra if s.spectrum_id == sid), None
            )
            if spectrum:
                meta_lines.append(f"=== Spectrum {sid} ===")
                for k, v in spectrum.metadata.items():
                    meta_lines.append(f"  {k}: {v}")
                meta_lines.append("")
                ion_lines.append(f"=== Spectrum {sid} ===")
                if spectrum.ions.size > 0:
                    for i, (m, inten) in enumerate(spectrum.ions):
                        ion_lines.append(f"  {i}: m/z={m:.4f}  I={inten:.2f}")
                else:
                    ion_lines.append("  No ions")
                ion_lines.append("")
        meta_edit.setPlainText("\n".join(meta_lines))
        ion_edit.setPlainText("\n".join(ion_lines))
        self._window.show()


# ---------------------------------------------------------------------------
# FragmentDistributionDialog
# ---------------------------------------------------------------------------

class FragmentDistributionDialog:
    """Dialog showing fragment distribution scatter plot with selection."""

    def __init__(self, parent):
        self.parent = parent
        self._window: Optional[QDialog] = None

    def show(self, parser: MGFParser, selected_ids: List):
        self._window = QDialog(self.parent)
        self._window.setWindowTitle("Fragment Distribution")
        self._window.resize(1000, 700)
        self._window.setModal(True)
        layout = QVBoxLayout(self._window)

        # Figure
        fig = Figure(figsize=(9, 5), dpi=90)
        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas)
        toolbar = NavigationToolbar2QT(canvas, self._window)
        layout.addWidget(toolbar)

        # Result tables
        splitter = QSplitter(Qt.Orientation.Horizontal)
        frags_tree = QTreeWidget()
        frags_tree.setHeaderLabels(["m/z", "Intensity", "Spectrum ID"])
        frags_tree.setColumnWidth(0, 120)
        frags_tree.setColumnWidth(1, 120)
        ann_tree = QTreeWidget()
        ann_tree.setHeaderLabels(["m/z", "Formula", "PPM", "Spectrum ID"])
        ann_tree.setColumnWidth(0, 120)
        ann_tree.setColumnWidth(1, 120)
        splitter.addWidget(frags_tree)
        splitter.addWidget(ann_tree)
        layout.addWidget(splitter)

        # Buttons
        btn_row = QHBoxLayout()
        remove_btn = QPushButton("Remove Selected Fragments")
        export_btn = QPushButton("Export")
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self._window.close)
        btn_row.addWidget(remove_btn)
        btn_row.addWidget(export_btn)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        # Gather data and plot
        all_mz = []
        all_inten = []
        all_spec_ids = []
        spectra = [s for s in parser.spectra if s.spectrum_id in selected_ids]
        for s in spectra:
            if s.ions.size > 0:
                for m, inten in s.ions:
                    all_mz.append(m)
                    all_inten.append(inten)
                    all_spec_ids.append(s.spectrum_id)

        ax = fig.add_subplot(111)
        if all_mz:
            ax.scatter(all_mz, all_inten, alpha=0.5, s=10, c="blue")
            ax.set_xlabel("m/z")
            ax.set_ylabel("Intensity")
            ax.set_title(f"Fragment Distribution ({len(all_mz)} fragments from "
                         f"{len(spectra)} spectra)")
            ax.grid(True, alpha=0.3)

            # Rectangle selector
            self._selected_rect: Optional[Tuple] = None
            def on_select(eclick, erelease):
                x1, x2 = sorted([eclick.xdata, erelease.xdata])
                y1, y2 = sorted([eclick.ydata, erelease.ydata])
                frags_tree.clear()
                for mz, inten, sid in zip(all_mz, all_inten, all_spec_ids):
                    if x1 <= mz <= x2 and y1 <= inten <= y2:
                        item = QTreeWidgetItem()
                        item.setText(0, f"{mz:.4f}")
                        item.setText(1, f"{inten:.2f}")
                        item.setText(2, str(sid))
                        frags_tree.addTopLevelItem(item)

            selector = RectangleSelector(
                ax, on_select, useblit=True,
                button=[1], minspanx=5, minspany=5, spancoords="pixels",
                interactive=True
            )
            canvas._selector = selector  # keep reference

            # Populate fragments tree
            for mz, inten, sid in zip(all_mz[:500], all_inten[:500], all_spec_ids[:500]):
                item = QTreeWidgetItem()
                item.setText(0, f"{mz:.4f}")
                item.setText(1, f"{inten:.2f}")
                item.setText(2, str(sid))
                frags_tree.addTopLevelItem(item)
        else:
            ax.text(0.5, 0.5, "No fragment data available",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
        canvas.draw()

        def _remove_selected():
            # Gather selected m/z values from frags_tree
            to_remove = set()
            for item in frags_tree.selectedItems():
                try:
                    to_remove.add(float(item.text(0)))
                except ValueError:
                    pass
            if not to_remove:
                return
            for s in spectra:
                if s.ions.size == 0:
                    continue
                mask = np.array([m not in to_remove for m in s.ions[:, 0]])
                s.ions = s.ions[mask]
            frags_tree.clear()

        remove_btn.clicked.connect(_remove_selected)
        self._window.exec()
