"""
cht_dialogs.py  —  BodyEditorDialog
"""
from pathlib import Path
from copy import deepcopy
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QLineEdit,
    QDialogButtonBox, QWidget, QFileDialog,
)
from PyQt5.QtCore import Qt
from UI_components.cht_constants import BUILD_DIRS, BODY_COLORS, BODY_ROLES, SOLID_PRESETS
from UI_components.cht_models    import SolidBodySpec


class BodyEditorDialog(QDialog):
    def __init__(self, body=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Solid Body")
        self.setMinimumWidth(460)
        self.setModal(True)
        self._body = deepcopy(body) if body is not None else SolidBodySpec()

        lay  = QVBoxLayout(self)
        lay.setSpacing(10)
        form = QFormLayout()
        form.setSpacing(8)
        form.setLabelAlignment(Qt.AlignLeft)

        stl_row = QWidget()
        stl_h   = QHBoxLayout(stl_row)
        stl_h.setContentsMargins(0, 0, 0, 0)
        stl_h.setSpacing(4)
        self._stl_btn = QPushButton("  Browse \u2026")
        self._stl_btn.setObjectName("btn_browse")
        self._stl_btn.setMinimumHeight(28)
        self._stl_lbl = QLabel(
            Path(self._body.stl_path).name if self._body.stl_path else "No file"
        )
        self._stl_lbl.setStyleSheet("color:#4A7AAA; font-size:14px; font-style:italic;")
        stl_h.addWidget(self._stl_btn, 1)
        stl_h.addWidget(self._stl_lbl, 2)
        self._stl_btn.clicked.connect(self._browse_stl)
        form.addRow("STL file", stl_row)

        self._name = QLineEdit(self._body.name)
        form.addRow("Body name", self._name)

        self._build_dir = QComboBox()
        self._build_dir.addItems(BUILD_DIRS)
        self._build_dir.setCurrentText(self._body.build_dir)
        form.addRow("Build direction", self._build_dir)

        self._material = QComboBox()
        self._material.addItems(list(SOLID_PRESETS.keys()))
        self._material.setCurrentText(self._body.material)
        form.addRow("Material", self._material)

        self._color = QComboBox()
        self._color.addItems(BODY_COLORS)
        self._color.setCurrentText(self._body.color)
        form.addRow("Display color", self._color)

        self._role = QComboBox()
        self._role.addItems(BODY_ROLES)
        self._role.setCurrentText(self._body.role)
        form.addRow("Role", self._role)

        tip = QLabel("fluid_base = first body defining domain origin\nstack_below = stacked under base along flow axis")
        tip.setObjectName("hint")
        tip.setWordWrap(True)
        form.addRow("", tip)
        lay.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def _browse_stl(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select STL", "", "STL Files (*.stl *.STL)")
        if path:
            self._body.stl_path = path
            self._stl_lbl.setText(Path(path).name)
            self._stl_lbl.setStyleSheet("color:#4FC3A1; font-size:14px;")
            if self._name.text().startswith("body_"):
                self._name.setText(Path(path).stem)

    def _accept(self):
        if not self._body.stl_path:
            return
        self._body.name      = self._name.text().strip() or self._body.name
        self._body.build_dir = self._build_dir.currentText()
        self._body.material  = self._material.currentText()
        self._body.color     = self._color.currentText()
        self._body.role      = self._role.currentText()
        self.accept()

    def result_body(self):
        return self._body
