"""
cht_widgets.py  (v4 — Domains/BCs tabs restructure)
Tab 1: Domains  — Fluid Domain (size + medium) + Solid Domain (bodies + GPU RT)
Tab 2: BCs      — Fluid BCs (flow/velocity/temps) + Solid BCs (heat source) + Voxelisation
Tab 3: Solver   — Flow Solver + Thermal Solver
Tab 4: Visualize
"""
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (
    QGridLayout, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QDoubleSpinBox, QTabWidget, QGroupBox, QFormLayout, QFrame,
    QProgressBar, QScrollArea, QTextEdit, QSpinBox, QListWidget,
    QListWidgetItem, QDialog, QStackedWidget, QRadioButton, QButtonGroup,
    QLineEdit, QCheckBox,
)
from PyQt5.QtCore  import Qt, pyqtSignal, QTimer
from PyQt5.QtGui   import QColor
try:
    from pyvistaqt import QtInteractor
    import pyvista as pv
    HAS_PVQT = True
except ImportError:
    HAS_PVQT = False; QtInteractor = QWidget

from UI_components.cht_constants import FLUID_PRESETS, SOLID_PRESETS, BODY_COLORS, FLOW_DIRS, FLUX_AXES, COLLISION_MODES, OUTLET_MODES
from UI_components.cht_dialogs   import BodyEditorDialog
from UI_components.cht_models    import SolidBodySpec
from LBM_CHT import STLVoxelizer, LBMCHT3D_Torch


# ── BodiesPanel ───────────────────────────────────────────────────────────────

class BodiesPanel(QWidget):
    bodies_changed = pyqtSignal(list)
    def __init__(self, parent=None):
        super().__init__(parent)
        self._bodies = []
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(4)
        tb = QHBoxLayout(); tb.setSpacing(4)
        self._btn_add  = QPushButton("\uff0b"); self._btn_add.setObjectName("btn_icon");  self._btn_add.setToolTip("Add solid body")
        self._btn_edit = QPushButton("\u270f"); self._btn_edit.setObjectName("btn_icon"); self._btn_edit.setToolTip("Edit selected"); self._btn_edit.setEnabled(False)
        self._btn_rem  = QPushButton("\u2715"); self._btn_rem.setObjectName("btn_icon");  self._btn_rem.setToolTip("Remove selected"); self._btn_rem.setEnabled(False)
        self._btn_up   = QPushButton("\u25b2"); self._btn_up.setObjectName("btn_icon");   self._btn_up.setToolTip("Move up");   self._btn_up.setEnabled(False)
        self._btn_dn   = QPushButton("\u25bc"); self._btn_dn.setObjectName("btn_icon");   self._btn_dn.setToolTip("Move down"); self._btn_dn.setEnabled(False)
        for b in (self._btn_add, self._btn_edit, self._btn_rem, self._btn_up, self._btn_dn): tb.addWidget(b)
        tb.addStretch()
        hint = QLabel("First body = domain reference (fluid_base)"); hint.setObjectName("hint"); tb.addWidget(hint)
        lay.addLayout(tb)
        self._list = QListWidget(); self._list.setMinimumHeight(130); self._list.setMaximumHeight(180)
        lay.addWidget(self._list)
        self._btn_add.clicked.connect(self._add_body);   self._btn_edit.clicked.connect(self._edit_body)
        self._btn_rem.clicked.connect(self._remove_body); self._btn_up.clicked.connect(self._move_up)
        self._btn_dn.clicked.connect(self._move_down);   self._list.itemSelectionChanged.connect(self._selection_changed)
        self._list.itemDoubleClicked.connect(lambda: self._edit_body())
    def _refresh_list(self):
        self._list.clear()
        for b in self._bodies:
            item = QListWidgetItem(f"  \u25cf  {b.label()}"); item.setForeground(QColor("#C8D8F0")); self._list.addItem(item)
        self.bodies_changed.emit([b.name for b in self._bodies])
    def _selection_changed(self):
        has = bool(self._list.selectedItems()); row = self._list.currentRow()
        self._btn_edit.setEnabled(has); self._btn_rem.setEnabled(has)
        self._btn_up.setEnabled(has and row > 0); self._btn_dn.setEnabled(has and row < len(self._bodies)-1)
    def _add_body(self):
        dlg = BodyEditorDialog(parent=self)
        if dlg.exec_() == QDialog.Accepted: self._bodies.append(dlg.result_body()); self._refresh_list()
    def _edit_body(self):
        row = self._list.currentRow()
        if row < 0: return
        dlg = BodyEditorDialog(body=self._bodies[row], parent=self)
        if dlg.exec_() == QDialog.Accepted: self._bodies[row] = dlg.result_body(); self._refresh_list()
    def _remove_body(self):
        row = self._list.currentRow()
        if row < 0 or len(self._bodies) <= 1: return
        self._bodies.pop(row); self._refresh_list()
    def _move_up(self):
        row = self._list.currentRow()
        if row > 0: self._bodies[row], self._bodies[row-1] = self._bodies[row-1], self._bodies[row]; self._refresh_list(); self._list.setCurrentRow(row-1)
    def _move_down(self):
        row = self._list.currentRow()
        if row < len(self._bodies)-1: self._bodies[row], self._bodies[row+1] = self._bodies[row+1], self._bodies[row]; self._refresh_list(); self._list.setCurrentRow(row+1)
    @property
    def bodies(self): return self._bodies
    def set_bodies(self, bodies_list): self._bodies = list(bodies_list); self._refresh_list()


# ── DomainsTab ────────────────────────────────────────────────────────────────

class DomainsTab(QWidget):
    bodies_changed     = pyqtSignal(list)
    save_model_clicked = pyqtSignal()
    load_model_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner  = QWidget(); inner.setStyleSheet("background:transparent;")
        iv = QVBoxLayout(inner); iv.setContentsMargins(8,8,8,4); iv.setSpacing(8)
        scroll.setWidget(inner)
        root = QVBoxLayout(self); root.setContentsMargins(0,0,0,0); root.addWidget(scroll)

        # Save / Load row
        mdl_row = QHBoxLayout(); mdl_row.setSpacing(6)
        self.btn_save_mdl = QPushButton("\U0001f4be  Save Model"); self.btn_save_mdl.setObjectName("btn_mdl"); self.btn_save_mdl.setMinimumHeight(28)
        self.btn_load_mdl = QPushButton("\U0001f4c2  Load Model"); self.btn_load_mdl.setObjectName("btn_mdl"); self.btn_load_mdl.setMinimumHeight(28)
        mdl_row.addWidget(self.btn_save_mdl); mdl_row.addWidget(self.btn_load_mdl); mdl_row.addStretch()
        mdl_hint = QLabel("Save / restore all settings as .chtmdl"); mdl_hint.setObjectName("hint"); mdl_row.addWidget(mdl_hint)
        iv.addLayout(mdl_row)

        # ── Fluid Domain ──────────────────────────────────────────────────
        grp_fdom = QGroupBox("Fluid Domain")
        fl = QFormLayout(grp_fdom); fl.setSpacing(7); fl.setLabelAlignment(Qt.AlignLeft)
        self.spn_Lx = QDoubleSpinBox(); self.spn_Lx.setRange(0.01,20); self.spn_Lx.setDecimals(3); self.spn_Lx.setValue(5.0); self.spn_Lx.setSuffix(" m")
        self.spn_Ly = QDoubleSpinBox(); self.spn_Ly.setRange(0.01,20); self.spn_Ly.setDecimals(3); self.spn_Ly.setValue(1.0); self.spn_Ly.setSuffix(" m")
        self.spn_Lz = QDoubleSpinBox(); self.spn_Lz.setRange(0.01,20); self.spn_Lz.setDecimals(3); self.spn_Lz.setValue(1.0); self.spn_Lz.setSuffix(" m")
        fl.addRow("Lx", self.spn_Lx); fl.addRow("Ly", self.spn_Ly); fl.addRow("Lz", self.spn_Lz)
        self.spn_j0 = QDoubleSpinBox(); self.spn_j0.setRange(1,10); self.spn_j0.setDecimals(2); self.spn_j0.setValue(2.5)
        j0_hint = QLabel("Gap \u00f7 j0_divisor = body offset from inlet along flow axis"); j0_hint.setObjectName("hint"); j0_hint.setWordWrap(True)
        fl.addRow("j0_divisor", self.spn_j0); fl.addRow("", j0_hint)
        self.cmb_fluid_medium = QComboBox(); self.cmb_fluid_medium.addItems(list(FLUID_PRESETS.keys())); self.cmb_fluid_medium.setCurrentText("Air (LBM-scaled)")
        fl.addRow("Fluid medium", self.cmb_fluid_medium)
        iv.addWidget(grp_fdom)

        # ── Solid Domain ──────────────────────────────────────────────────
        grp_sdom = QGroupBox("Solid Domain")
        sl = QVBoxLayout(grp_sdom); sl.setContentsMargins(6,12,6,6); sl.setSpacing(6)
        self.bodies_panel = BodiesPanel()
        self.bodies_panel.bodies_changed.connect(self.bodies_changed)
        sl.addWidget(self.bodies_panel)
        self.chk_gpu_rt = QCheckBox("GPU ray tracing")
        self.chk_gpu_rt.setChecked(True)
        self.chk_gpu_rt.setToolTip("Checked = use patched CUDA voxelizer (faster)\nUnchecked = use default CPU voxelizer")
        sl.addWidget(self.chk_gpu_rt)
        iv.addWidget(grp_sdom)

        iv.addStretch()

        self.btn_save_mdl.clicked.connect(self.save_model_clicked)
        self.btn_load_mdl.clicked.connect(self.load_model_clicked)

    @property
    def domain(self): return dict(Lx=self.spn_Lx.value(), Ly=self.spn_Ly.value(), Lz=self.spn_Lz.value())

    @property
    def j0_divisor(self): return self.spn_j0.value()

# ── FaceAssignmentWidget ──────────────────────────────────────────────────────

_ALL_FACES = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
_OPP_FACE  = {"+X":"-X", "-X":"+X", "+Y":"-Y", "-Y":"+Y", "+Z":"-Z", "-Z":"+Z"}

class FaceAssignmentWidget(QWidget):
    """
    Checkbox grid:
        +X   -X   +Y   -Y   +Z   -Z
Inlet  (o)  ( )  ( )  ( )  ( )  ( )   ← radio, auto from flow_dir
Outlet [ ]  [ ]  [ ]  [ ]  [ ]  [ ]   ← multi-check
Wall   [ ]  [ ]  [ ]  [ ]  [ ]  [ ]   ← multi-check

    Faces unchecked in both Outlet and Wall → periodic (omitted from dicts).
    """
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:transparent;")
        g = QGridLayout(self)
        g.setContentsMargins(0, 2, 0, 2)
        g.setSpacing(4)
        g.setColumnStretch(0, 0)

        # ── column headers ────────────────────────────────────────────────
        g.addWidget(QLabel(""), 0, 0)   # blank corner
        for c, face in enumerate(_ALL_FACES, 1):
            lbl = QLabel(face)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font-family:monospace;font-weight:700;"
                              "color:#C8D8F0;font-size:12px;")
            g.addWidget(lbl, 0, c)

        # ── row labels ────────────────────────────────────────────────────
        row_labels = ["Inlet", "Outlet", "Wall"]
        row_colors = ["#4CAF82", "#FFD580", "#80AAFF"]
        for r, (txt, col) in enumerate(zip(row_labels, row_colors), 1):
            lbl = QLabel(txt)
            lbl.setStyleSheet(f"color:{col};font-size:12px;font-weight:600;min-width:44px;")
            g.addWidget(lbl, r, 0)

        # ── Inlet row — radio buttons (exactly one) ───────────────────────
        self._inlet_grp = QButtonGroup(self)
        self._inlet_grp.setExclusive(True)
        self._inlet_radios = {}
        for c, face in enumerate(_ALL_FACES, 1):
            rb = QRadioButton()
            rb.setStyleSheet("QRadioButton::indicator { width:14px; height:14px; }")
            self._inlet_grp.addButton(rb)
            self._inlet_radios[face] = rb
            g.addWidget(rb, 1, c, Qt.AlignCenter)
        self._inlet_grp.buttonToggled.connect(lambda *_: self.changed.emit())

        # ── Outlet / Wall rows — independent checkboxes ───────────────────
        self._outlet_chks = {}
        self._wall_chks   = {}
        for c, face in enumerate(_ALL_FACES, 1):
            for row, store in ((2, self._outlet_chks), (3, self._wall_chks)):
                cb = QCheckBox()
                cb.setStyleSheet("QCheckBox::indicator { width:14px; height:14px; }")
                store[face] = cb
                g.addWidget(cb, row, c, Qt.AlignCenter)
                cb.stateChanged.connect(lambda state, f=face, s=store: self._on_check(f, s))

        self.set_flow_dir("+X")

    def _on_check(self, face, changed_store):
        """Ensure a face can't be both Outlet and Wall simultaneously."""
        other = self._wall_chks if changed_store is self._outlet_chks else self._outlet_chks
        if changed_store[face].isChecked():
            other[face].blockSignals(True)
            other[face].setChecked(False)
            other[face].blockSignals(False)
        self.changed.emit()

    def set_flow_dir(self, flow_dir):
        """Auto-populate: inlet face = radio checked, outlet face = outlet checked,
           transverse faces = wall checked. User can override after."""
        inlet  = _OPP_FACE.get(flow_dir, "-X")   # flow enters here
        outlet = flow_dir                          # flow exits here
        transverse = [f for f in _ALL_FACES if f not in (inlet, outlet)]

        # block all signals during batch update
        for rb in self._inlet_radios.values():   rb.blockSignals(True)
        for cb in self._outlet_chks.values():    cb.blockSignals(True)
        for cb in self._wall_chks.values():      cb.blockSignals(True)

        for face in _ALL_FACES:
            self._inlet_radios[face].setChecked(face == inlet)
            self._outlet_chks[face].setChecked(face == outlet)
            self._wall_chks[face].setChecked(face in transverse)
            # inlet face: disable outlet/wall checkboxes
            is_inlet = (face == inlet)
            self._outlet_chks[face].setEnabled(not is_inlet)
            self._wall_chks[face].setEnabled(not is_inlet)

        for rb in self._inlet_radios.values():   rb.blockSignals(False)
        for cb in self._outlet_chks.values():    cb.blockSignals(False)
        for cb in self._wall_chks.values():      cb.blockSignals(False)

        # also disable outlet/wall for whichever face the inlet radio points to
        self._inlet_grp.buttonToggled.connect(self._sync_inlet_disables)
        self.changed.emit()

    def _sync_inlet_disables(self):
        """When user manually moves inlet radio, disable that face's checkboxes."""
        for face in _ALL_FACES:
            is_inlet = self._inlet_radios[face].isChecked()
            self._outlet_chks[face].setEnabled(not is_inlet)
            self._wall_chks[face].setEnabled(not is_inlet)
            if is_inlet:
                self._outlet_chks[face].setChecked(False)
                self._wall_chks[face].setChecked(False)

    @property
    def inlet_face(self):
        for face, rb in self._inlet_radios.items():
            if rb.isChecked(): return face
        return None

    @property
    def domain_outlets(self):
        return [f for f, cb in self._outlet_chks.items() if cb.isChecked()]

    @property
    def domain_walls(self):
        return [f for f, cb in self._wall_chks.items() if cb.isChecked()]

    def get_state(self):
        return {
            "inlet":   self.inlet_face,
            "outlets": self.domain_outlets,
            "walls":   self.domain_walls,
        }

    def set_state(self, state: dict):
        inlet   = state.get("inlet")
        outlets = state.get("outlets", [])
        walls   = state.get("walls",   [])
        for rb in self._inlet_radios.values():  rb.blockSignals(True)
        for cb in self._outlet_chks.values():   cb.blockSignals(True)
        for cb in self._wall_chks.values():     cb.blockSignals(True)
        for face in _ALL_FACES:
            if inlet: self._inlet_radios[face].setChecked(face == inlet)
            self._outlet_chks[face].setChecked(face in outlets)
            self._wall_chks[face].setChecked(face in walls)
        for rb in self._inlet_radios.values():  rb.blockSignals(False)
        for cb in self._outlet_chks.values():   cb.blockSignals(False)
        for cb in self._wall_chks.values():     cb.blockSignals(False)
        self._sync_inlet_disables()
        self.changed.emit()

# ── BCsTab ────────────────────────────────────────────────────────────────────

class BCsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner  = QWidget(); inner.setStyleSheet("background:transparent;")
        iv = QVBoxLayout(inner); iv.setContentsMargins(8,8,8,4); iv.setSpacing(8)
        scroll.setWidget(inner)
        root = QVBoxLayout(self); root.setContentsMargins(0,0,0,0); root.addWidget(scroll)

        # ── Fluid BCs ─────────────────────────────────────────────────────
        grp_fbc = QGroupBox("Fluid BCs")
        ff = QFormLayout(grp_fbc); ff.setSpacing(7); ff.setLabelAlignment(Qt.AlignLeft)
        self.cmb_dir = QComboBox(); self.cmb_dir.addItems(FLOW_DIRS); self.cmb_dir.setCurrentText("+X")
        ff.addRow("Flow direction", self.cmb_dir)
        self.spn_uin = QDoubleSpinBox(); self.spn_uin.setRange(1e-4,1000); self.spn_uin.setDecimals(4); self.spn_uin.setValue(1.0); self.spn_uin.setSuffix(" m/s")
        ff.addRow("Inlet velocity", self.spn_uin)
        self.spn_tin  = QDoubleSpinBox(); self.spn_tin.setRange(-50,500);  self.spn_tin.setDecimals(1); self.spn_tin.setValue(20);  self.spn_tin.setSuffix(" \u00b0C")
        ff.addRow("Inlet temperature", self.spn_tin)
        self.spn_tamb = QDoubleSpinBox(); self.spn_tamb.setRange(-50,500); self.spn_tamb.setDecimals(1); self.spn_tamb.setValue(20); self.spn_tamb.setSuffix(" \u00b0C")
        ff.addRow("Ambient temperature", self.spn_tamb)
        # Face assignment
        face_lbl = QLabel("Face boundary types")
        face_lbl.setObjectName("hint")
        ff.addRow("", face_lbl)
        self.face_widget = FaceAssignmentWidget()
        ff.addRow("", self.face_widget)
        # auto-update when flow direction changes
        self.cmb_dir.currentTextChanged.connect(self.face_widget.set_flow_dir)        
        iv.addWidget(grp_fbc)

        # ── Solid BCs ─────────────────────────────────────────────────────
        grp_sbc = QGroupBox("Solid BCs")
        hs_lay = QVBoxLayout(grp_sbc); hs_lay.setSpacing(8)
        type_row = QHBoxLayout(); type_row.setSpacing(12)
        self._bg_bc = QButtonGroup(self)
        self._rb_off  = QRadioButton("Off"); self._rb_flux = QRadioButton("Surface Flux"); self._rb_vol = QRadioButton("Volumetric Heat")
        self._rb_off.setChecked(True)
        for i, rb in enumerate((self._rb_off, self._rb_flux, self._rb_vol)): self._bg_bc.addButton(rb, i); type_row.addWidget(rb)
        hs_lay.addLayout(type_row)
        self._bc_stack = QStackedWidget()
        page_off = QLabel("No heat source applied."); page_off.setObjectName("hint"); page_off.setAlignment(Qt.AlignCenter); page_off.setMinimumHeight(40)
        self._bc_stack.addWidget(page_off)
        page_flux = QWidget(); page_flux.setStyleSheet("background:transparent;"); pff = QFormLayout(page_flux); pff.setSpacing(6); pff.setLabelAlignment(Qt.AlignLeft)
        self.cmb_flux_solid = QComboBox(); pff.addRow("Target solid", self.cmb_flux_solid)
        self.cmb_flux_axis  = QComboBox(); self.cmb_flux_axis.addItems(FLUX_AXES); self.cmb_flux_axis.setCurrentText("-Z"); pff.addRow("Flux axis", self.cmb_flux_axis)
        self.spn_flux_L = QDoubleSpinBox(); self.spn_flux_L.setRange(1,5000); self.spn_flux_L.setDecimals(1); self.spn_flux_L.setValue(400); self.spn_flux_L.setSuffix(" mm"); pff.addRow("Surface L", self.spn_flux_L)
        self.spn_flux_W = QDoubleSpinBox(); self.spn_flux_W.setRange(1,5000); self.spn_flux_W.setDecimals(1); self.spn_flux_W.setValue(200); self.spn_flux_W.setSuffix(" mm"); pff.addRow("Surface W", self.spn_flux_W)
        self.spn_flux_q = QDoubleSpinBox(); self.spn_flux_q.setRange(0,1e7); self.spn_flux_q.setDecimals(1); self.spn_flux_q.setValue(1800); self.spn_flux_q.setSuffix(" W/m\u00b2"); pff.addRow("Heat flux q\u2033", self.spn_flux_q)
        self.chk_flux_autocenter = QCheckBox("Auto-center on solid centroid"); self.chk_flux_autocenter.setChecked(True); pff.addRow("", self.chk_flux_autocenter)
        self._bc_stack.addWidget(page_flux)
        page_vol = QWidget(); page_vol.setStyleSheet("background:transparent;"); pvf = QFormLayout(page_vol); pvf.setSpacing(6); pvf.setLabelAlignment(Qt.AlignLeft)
        self.cmb_vol_solid = QComboBox(); pvf.addRow("Target solid", self.cmb_vol_solid)
        self.spn_vol_Q = QDoubleSpinBox(); self.spn_vol_Q.setRange(0,100000); self.spn_vol_Q.setDecimals(1); self.spn_vol_Q.setValue(50); self.spn_vol_Q.setSuffix(" W"); pvf.addRow("Total power", self.spn_vol_Q)
        self._bc_stack.addWidget(page_vol)
        hs_lay.addWidget(self._bc_stack)
        iv.addWidget(grp_sbc)
        self._rb_off.toggled.connect(lambda v: v and self._bc_stack.setCurrentIndex(0))
        self._rb_flux.toggled.connect(lambda v: v and self._bc_stack.setCurrentIndex(1))
        self._rb_vol.toggled.connect(lambda v: v and self._bc_stack.setCurrentIndex(2))

        # ── Voxelisation ──────────────────────────────────────────────────
        grp_vox = QGroupBox("Voxelisation")
        vf = QFormLayout(grp_vox); vf.setSpacing(7); vf.setLabelAlignment(Qt.AlignLeft)
        self.spn_dx = QDoubleSpinBox(); self.spn_dx.setRange(0.1,100); self.spn_dx.setDecimals(1); self.spn_dx.setValue(10); self.spn_dx.setSuffix(" mm")
        vf.addRow("Voxel size", self.spn_dx)
        vox_hint = QLabel("Voxel cell pitch \u2014 smaller = finer geometry but more cells"); vox_hint.setObjectName("hint"); vox_hint.setWordWrap(True)
        vf.addRow("", vox_hint)
        self.lbl_grid = QLabel("\u2014"); self.lbl_grid.setObjectName("hint")
        vf.addRow("Resolved grid", self.lbl_grid)
        iv.addWidget(grp_vox)

        iv.addStretch()

    def _sync_bc_combos(self, names):
        for cmb in (self.cmb_flux_solid, self.cmb_vol_solid):
            cur = cmb.currentText(); cmb.clear(); cmb.addItems(names)
            if cur in names: cmb.setCurrentText(cur)

    def update_grid_label(self, dx_mm, Lx, Ly, Lz):
        nx=round(Lx*1000/dx_mm); ny=round(Ly*1000/dx_mm); nz=round(Lz*1000/dx_mm)
        self.lbl_grid.setText(f"{nx} \u00d7 {ny} \u00d7 {nz}   ({nx*ny*nz/1e6:.2f} M cells)")

    @property
    def bc_type(self):
        if self._rb_flux.isChecked(): return "surface_flux"
        if self._rb_vol.isChecked():  return "volumetric"
        return "off"

    @property
    def bc_params(self):
        if self.bc_type == "surface_flux":
            return dict(solid_name=self.cmb_flux_solid.currentText(), axis=self.cmb_flux_axis.currentText(),
                        L_mm=self.spn_flux_L.value(), W_mm=self.spn_flux_W.value(),
                        q_flux=self.spn_flux_q.value(), auto_center=self.chk_flux_autocenter.isChecked())
        if self.bc_type == "volumetric":
            return dict(solid_name=self.cmb_vol_solid.currentText(), Q_watts=self.spn_vol_Q.value())
        return {}

    @property
    def domain_walls(self):   return self.face_widget.domain_walls
    @property
    def domain_outlets(self): return self.face_widget.domain_outlets

# ── SolverTab ─────────────────────────────────────────────────────────────────

class SolverTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        lay = QVBoxLayout(self); lay.setContentsMargins(8,8,8,8); lay.setSpacing(8)

        grp_fs = QGroupBox("Flow Solver")
        fsf = QFormLayout(grp_fs); fsf.setSpacing(7); fsf.setLabelAlignment(Qt.AlignLeft)
        self.spn_tol_u    = QDoubleSpinBox(); self.spn_tol_u.setRange(1e-6,1); self.spn_tol_u.setDecimals(5); self.spn_tol_u.setValue(5e-3); fsf.addRow("Convergence tol u_ema", self.spn_tol_u)
        self.cmb_collision = QComboBox(); self.cmb_collision.addItems(COLLISION_MODES); self.cmb_collision.setCurrentText("mrt_smag"); fsf.addRow("Collision model", self.cmb_collision)
        self.cmb_outlet    = QComboBox(); self.cmb_outlet.addItems(OUTLET_MODES);       self.cmb_outlet.setCurrentText("convective");  fsf.addRow("Outlet BC mode",    self.cmb_outlet)
        lay.addWidget(grp_fs)

        grp_ts = QGroupBox("Thermal Solver")
        tsf = QFormLayout(grp_ts); tsf.setSpacing(7); tsf.setLabelAlignment(Qt.AlignLeft)
        self.spn_max_outer = QSpinBox();       self.spn_max_outer.setRange(100,200000); self.spn_max_outer.setValue(50000); tsf.addRow("Max outer iterations", self.spn_max_outer)
        self.spn_tol_dTs   = QDoubleSpinBox(); self.spn_tol_dTs.setRange(1e-6,1); self.spn_tol_dTs.setDecimals(5); self.spn_tol_dTs.setValue(5e-3); tsf.addRow("dT solid tol",  self.spn_tol_dTs)
        self.spn_tol_dTf   = QDoubleSpinBox(); self.spn_tol_dTf.setRange(1e-6,1); self.spn_tol_dTf.setDecimals(5); self.spn_tol_dTf.setValue(5e-3); tsf.addRow("dT fluid tol",  self.spn_tol_dTf)
        self.spn_dt_max    = QDoubleSpinBox(); self.spn_dt_max.setRange(1,1000);   self.spn_dt_max.setDecimals(1);  self.spn_dt_max.setValue(100);  tsf.addRow("dt_scale_max",  self.spn_dt_max)
        self.spn_max_mg    = QSpinBox();       self.spn_max_mg.setRange(1,100);    self.spn_max_mg.setValue(15);                                     tsf.addRow("Max MG cycles", self.spn_max_mg)
        lay.addWidget(grp_ts)
        lay.addStretch()


# ── VisualizeTab ──────────────────────────────────────────────────────────────

class VisualizeTab(QWidget):
    plot_requested = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._body_checkboxes = {}
        self._npz_path = ""

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner = QWidget(); inner.setStyleSheet("background:transparent;")
        iv = QVBoxLayout(inner); iv.setContentsMargins(8,8,8,4); iv.setSpacing(8)
        scroll.setWidget(inner)
        root = QVBoxLayout(self); root.setContentsMargins(0,0,0,0); root.addWidget(scroll)

        grp_src = QGroupBox("Results File")
        sf = QVBoxLayout(grp_src); sf.setSpacing(4)
        file_row = QHBoxLayout()
        self.lbl_npz = QLabel("No file loaded"); self.lbl_npz.setObjectName("hint"); self.lbl_npz.setWordWrap(True)
        self.btn_browse = QPushButton("Browse…"); self.btn_browse.setMaximumWidth(80)
        file_row.addWidget(self.lbl_npz, 1); file_row.addWidget(self.btn_browse)
        sf.addLayout(file_row)
        self.btn_plot = QPushButton("\U0001f4ca  Plot in PyVista Window")
        self.btn_plot.setMinimumHeight(36); self.btn_plot.setObjectName("btn_run")
        sf.addWidget(self.btn_plot)
        self.chk_low_quality = QCheckBox("Low quality (faster, less lag)"); self.chk_low_quality.setChecked(True)
        sf.addWidget(self.chk_low_quality)
        iv.addWidget(grp_src)

        grp_fld = QGroupBox("Field")
        ff = QFormLayout(grp_fld); ff.setLabelAlignment(Qt.AlignLeft)
        self.cmb_field = QComboBox(); self.cmb_field.addItems(["T  (temperature)", "speed  (velocity mag)", "rho  (density)"])
        ff.addRow("Scalar field", self.cmb_field)
        self.cmb_cmap = QComboBox(); self.cmb_cmap.addItems(["inferno","turbo","hot","coolwarm","viridis","jet"])
        ff.addRow("Colormap", self.cmb_cmap)
        iv.addWidget(grp_fld)

        grp_dom = QGroupBox("Domains to show")
        self._dom_layout = QVBoxLayout(grp_dom); self._dom_layout.setSpacing(3)
        self.chk_fluid = QCheckBox("Fluid domain")
        self.spn_fluid_opacity = QDoubleSpinBox()
        self.spn_fluid_opacity.setRange(0.05,1.0); self.spn_fluid_opacity.setSingleStep(0.05)
        self.spn_fluid_opacity.setValue(0.3); self.spn_fluid_opacity.setDecimals(2); self.spn_fluid_opacity.setMaximumWidth(70)
        fluid_row = QHBoxLayout(); fluid_row.addWidget(self.chk_fluid)
        fluid_row.addWidget(QLabel("opacity")); fluid_row.addWidget(self.spn_fluid_opacity); fluid_row.addStretch()
        self._dom_layout.addLayout(fluid_row)
        self._bodies_vbox = QVBoxLayout(); self._bodies_vbox.setSpacing(2)
        self._dom_layout.addLayout(self._bodies_vbox)
        hint = QLabel("Bodies auto-populated when NPZ is loaded"); hint.setObjectName("hint"); self._dom_layout.addWidget(hint)
        iv.addWidget(grp_dom)

        grp_sl = QGroupBox("Streamlines")
        slf = QFormLayout(grp_sl); slf.setLabelAlignment(Qt.AlignLeft)
        self.chk_stream = QCheckBox("Show streamlines"); self.chk_stream.setChecked(True)
        self.spn_nseeds = QSpinBox(); self.spn_nseeds.setRange(10,2000); self.spn_nseeds.setValue(150)
        self.cmb_seed_mode = QComboBox(); self.cmb_seed_mode.addItems(["inlet_plane","line","sphere"])
        slf.addRow("", self.chk_stream); slf.addRow("N seeds", self.spn_nseeds); slf.addRow("Seed mode", self.cmb_seed_mode)
        iv.addWidget(grp_sl)
        iv.addStretch()

        self.btn_browse.clicked.connect(self._browse_npz)

    def _browse_npz(self):
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(None,"Load Results","","NumPy Archive (*.npz);;All Files (*)")
        if path: self.set_npz(path)

    def set_npz(self, path):
        self._npz_path = path
        self.lbl_npz.setText(path.replace("\\","/").split("/")[-1])
        self._populate_bodies(path)

    def _populate_bodies(self, path):
        for cb in self._body_checkboxes.values():
            self._bodies_vbox.removeWidget(cb); cb.deleteLater()
        self._body_checkboxes.clear()
        try:
            data = np.load(path, allow_pickle=True)
            names = list(data["body_names"]) if "body_names" in data.files else []
        except Exception:
            names = []
        for name in names:
            cb = QCheckBox(name); cb.setChecked(True)
            self._bodies_vbox.addWidget(cb)
            self._body_checkboxes[name] = cb

    @property
    def field_key(self): return self.cmb_field.currentText().split()[0]

    def get_plot_params(self):
        return dict(
            npz_path     = self._npz_path,
            field        = self.field_key,
            cmap         = self.cmb_cmap.currentText(),
            show_bodies  = [n for n,cb in self._body_checkboxes.items() if cb.isChecked()],
            show_fluid   = self.chk_fluid.isChecked(),
            fluid_opacity= self.spn_fluid_opacity.value(),
            streamlines  = self.chk_stream.isChecked(),
            n_seeds      = self.spn_nseeds.value(),
            seed_mode    = self.cmb_seed_mode.currentText(),
            low_quality  = self.chk_low_quality.isChecked(),
        )


# ── ThermalDiagnosticsDialog ──────────────────────────────────────────────────

class ThermalDiagnosticsDialog(QDialog):
    def __init__(self, rows, parent=None):
        """
        rows: list of dicts with keys: name, tmin, tmean, tmax, nvox
              Last two rows are ALL SOLID / ALL FLUID summaries (nvox as int or str).
        """
        super().__init__(parent)
        self.setWindowTitle("Thermal Diagnostics")
        self.setMinimumWidth(560)
        self.setAttribute(Qt.WA_DeleteOnClose)

        lay = QVBoxLayout(self); lay.setSpacing(10)

        hdr = QLabel("Thermal Diagnostics — Post-Solve Summary")
        hdr.setStyleSheet("font-size:15px;font-weight:700;color:#C8D8F0;")
        lay.addWidget(hdr)

        from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
        cols = ["Body", "T min (°C)", "T mean (°C)", "T max (°C)", "N voxels"]
        tbl = QTableWidget(len(rows), len(cols))
        tbl.setHorizontalHeaderLabels(cols)
        tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, len(cols)):
            tbl.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        tbl.setSelectionBehavior(QTableWidget.SelectRows)
        tbl.setAlternatingRowColors(True)
        tbl.verticalHeader().setVisible(False)

        n_body_rows = len(rows) - 2   # last two are summary rows
        for r, row in enumerate(rows):
            is_summary = r >= n_body_rows
            items = [
                row["name"],
                f"{row['tmin']:.2f}"  if isinstance(row["tmin"],  float) else "—",
                f"{row['tmean']:.2f}" if isinstance(row["tmean"], float) else "—",
                f"{row['tmax']:.2f}"  if isinstance(row["tmax"],  float) else "—",
                f"{row['nvox']:,}"    if isinstance(row["nvox"],   int)   else "—",
            ]
            for c, val in enumerate(items):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter if c > 0 else Qt.AlignLeft | Qt.AlignVCenter)
                if is_summary:
                    item.setForeground(QColor("#FFD580" if "SOLID" in row["name"] else "#80D4FF"))
                    font = item.font(); font.setBold(True); item.setFont(font)
                tbl.setItem(r, c, item)

        lay.addWidget(tbl)

        btn = QPushButton("Close"); btn.setMaximumWidth(100)
        btn.clicked.connect(self.accept)
        btn_row = QHBoxLayout(); btn_row.addStretch(); btn_row.addWidget(btn)
        lay.addLayout(btn_row)

# ── ControlPanel ──────────────────────────────────────────────────────────────

class ControlPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ctrl_panel")
        root = QVBoxLayout(self); root.setContentsMargins(12,10,12,10); root.setSpacing(8)

        hdr = QLabel("FluxCore3D v1.0"); hdr.setStyleSheet("font-size:21px;font-weight:800;color:#C8D8F0;letter-spacing:1px;"); root.addWidget(hdr)
        sub = QLabel("GPU-Native  \u00b7  Conjugate Heat Transfer  \u00b7  Multi-Body Assembly"); sub.setStyleSheet("font-size:14px;color:#2E3A52;margin-top:-4px;"); root.addWidget(sub)
        root.addWidget(self._div())

        self.ctrl_tabs = QTabWidget(); self.ctrl_tabs.setObjectName("ctrl_tabs"); root.addWidget(self.ctrl_tabs, 1)
        self.domains_tab   = DomainsTab()
        self.bcs_tab       = BCsTab()
        self.solver_tab    = SolverTab()
        self.visualize_tab = VisualizeTab()
        self.ctrl_tabs.addTab(self.domains_tab,   "  \u25a0  Domains  ")
        self.ctrl_tabs.addTab(self.bcs_tab,       "  \U0001f4a7  BCs  ")
        self.ctrl_tabs.addTab(self.solver_tab,    "  \u2699  Solver  ")
        self.ctrl_tabs.addTab(self.visualize_tab, "  \U0001f4ca  Visualize  ")

        # Cross-tab wiring
        self.bcs_tab.spn_dx.valueChanged.connect(self._on_dx_changed)
        self.domains_tab.spn_Lx.valueChanged.connect(self._on_dx_changed)
        self.domains_tab.spn_Ly.valueChanged.connect(self._on_dx_changed)
        self.domains_tab.spn_Lz.valueChanged.connect(self._on_dx_changed)
        self.domains_tab.bodies_changed.connect(self.bcs_tab._sync_bc_combos)
        self.domains_tab.bodies_changed.connect(lambda names: self.btn_preview.setEnabled(len(names)>0))

        root.addWidget(self._div())

        self.progress_bar = QProgressBar(); self.progress_bar.setValue(0); self.progress_bar.setTextVisible(False); self.progress_bar.setMaximumHeight(5); root.addWidget(self.progress_bar)
        self.lbl_status = QLabel("Ready"); self.lbl_status.setStyleSheet("color:#2E3A52;font-size:14px;padding:1px 0;"); root.addWidget(self.lbl_status)

        btn_row = QHBoxLayout(); btn_row.setSpacing(6)
        self.btn_preview = QPushButton("\u2bc1   Preview")
        self.btn_preview.setObjectName("btn_preview"); self.btn_preview.setMinimumHeight(40); self.btn_preview.setEnabled(False)
        self.btn_run = QPushButton("\u25b6   Run Model")
        self.btn_run.setObjectName("btn_run"); self.btn_run.setMinimumHeight(40)
        btn_row.addWidget(self.btn_preview, 2); btn_row.addWidget(self.btn_run, 3)
        root.addLayout(btn_row)

        self.log = QTextEdit(); self.log.setObjectName("log"); self.log.setReadOnly(True); self.log.setMaximumHeight(100); self.log.setMinimumHeight(70); root.addWidget(self.log)
        self._on_dx_changed()

    def _div(self):
        f = QFrame(); f.setObjectName("div"); return f

    def _on_dx_changed(self, *_):
        d = self.domains_tab.domain; dx = self.bcs_tab.spn_dx.value()
        self.bcs_tab.update_grid_label(dx, d["Lx"], d["Ly"], d["Lz"])

    def log_msg(self, msg):
        self.log.append(msg); self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def get_all_params(self):
        dt, bt, sv = self.domains_tab, self.bcs_tab, self.solver_tab
        return dict(
            domain      = dt.domain,
            j0_divisor  = dt.j0_divisor,
            bodies      = dt.bodies_panel.bodies,
            fluid       = dt.cmb_fluid_medium.currentText(),
            t_in_C      = bt.spn_tin.value(),
            t_amb_C     = bt.spn_tamb.value(),
            flow_dir    = bt.cmb_dir.currentText(),
            u_in        = bt.spn_uin.value(),
            collision   = sv.cmb_collision.currentText(),
            outlet_bc   = sv.cmb_outlet.currentText(),
            bc_type     = bt.bc_type,
            bc_params   = bt.bc_params,
            dx_mm       = bt.spn_dx.value(),
            tol_u_ema   = sv.spn_tol_u.value(),
            max_outer   = sv.spn_max_outer.value(),
            tol_dTs     = sv.spn_tol_dTs.value(),
            tol_dTf     = sv.spn_tol_dTf.value(),
            dt_scale_max= sv.spn_dt_max.value(),
            max_mg      = sv.spn_max_mg.value(),
            gpu_raytrace= dt.chk_gpu_rt.isChecked(),
            domain_walls   = bt.domain_walls,
            domain_outlets = bt.domain_outlets,
            face_states    = bt.face_widget.get_state(),            
        )

    def restore(self, state):
        dt, bt, sv = self.domains_tab, self.bcs_tab, self.solver_tab
        dom = state.get("domain", {})
        if "Lx" in dom: dt.spn_Lx.setValue(dom["Lx"])
        if "Ly" in dom: dt.spn_Ly.setValue(dom["Ly"])
        if "Lz" in dom: dt.spn_Lz.setValue(dom["Lz"])
        if "j0_divisor"   in state: dt.spn_j0.setValue(state["j0_divisor"])
        if "fluid"        in state: dt.cmb_fluid_medium.setCurrentText(state["fluid"])
        if "bodies"       in state: dt.bodies_panel.set_bodies(state["bodies"])
        if "gpu_raytrace" in state: dt.chk_gpu_rt.setChecked(state["gpu_raytrace"])
        if "flow_dir"     in state: bt.cmb_dir.setCurrentText(state["flow_dir"])
        if "u_in"         in state: bt.spn_uin.setValue(state["u_in"])
        if "t_in_C"       in state: bt.spn_tin.setValue(state["t_in_C"])
        if "t_amb_C"      in state: bt.spn_tamb.setValue(state["t_amb_C"])
        if "dx_mm"        in state: bt.spn_dx.setValue(state["dx_mm"])
        bct = state.get("bc_type","off"); bcp = state.get("bc_params",{})
        if bct == "off": bt._rb_off.setChecked(True)
        elif bct == "surface_flux":
            bt._rb_flux.setChecked(True)
            bt.cmb_flux_axis.setCurrentText(bcp.get("axis","-Z"))
            bt.spn_flux_L.setValue(bcp.get("L_mm",400)); bt.spn_flux_W.setValue(bcp.get("W_mm",200))
            bt.spn_flux_q.setValue(bcp.get("q_flux",1800)); bt.chk_flux_autocenter.setChecked(bcp.get("auto_center",True))
            if bcp.get("solid_name"): bt.cmb_flux_solid.setCurrentText(bcp["solid_name"])
        elif bct == "volumetric":
            bt._rb_vol.setChecked(True); bt.spn_vol_Q.setValue(bcp.get("Q_watts",50))
            if bcp.get("solid_name"): bt.cmb_vol_solid.setCurrentText(bcp["solid_name"])
        if "collision"    in state: sv.cmb_collision.setCurrentText(state["collision"])
        if "outlet_bc"    in state: sv.cmb_outlet.setCurrentText(state["outlet_bc"])
        if "tol_u_ema"    in state: sv.spn_tol_u.setValue(state["tol_u_ema"])
        if "max_outer"    in state: sv.spn_max_outer.setValue(int(state["max_outer"]))
        if "tol_dTs"      in state: sv.spn_tol_dTs.setValue(state["tol_dTs"])
        if "tol_dTf"      in state: sv.spn_tol_dTf.setValue(state["tol_dTf"])
        if "dt_scale_max" in state: sv.spn_dt_max.setValue(state["dt_scale_max"])
        if "max_mg"       in state: sv.spn_max_mg.setValue(int(state["max_mg"]))
        if "face_states" in state: bt.face_widget.set_state(state["face_states"])


# ── _StableInteractor ─────────────────────────────────────────────────────────

if HAS_PVQT:
    class _StableInteractor(QtInteractor):
        def __init__(self, viewport, *args, **kwargs):
            self._vp = viewport; self._replay_scheduled = False
            super().__init__(*args, **kwargs)
        def resizeEvent(self, event):
            super().resizeEvent(event)
            if self._vp._pending is not None and not self._replay_scheduled:
                self._replay_scheduled = True; QTimer.singleShot(0, self._deferred_replay)
        def showEvent(self, event):
            super().showEvent(event)
            if self._vp._pending is not None and not self._replay_scheduled:
                self._replay_scheduled = True; QTimer.singleShot(0, self._deferred_replay)
        def _deferred_replay(self):
            self._replay_scheduled = False
            if self._vp._pending is None: return
            if not self._vp._building: self._vp._replay_pending()
else:
    _StableInteractor = None


# ── ViewportWidget ────────────────────────────────────────────────────────────

_FLOW_AXIS = {
    "+X":(0, 1,(1,0,0)), "-X":(0,-1,(-1,0,0)),
    "+Y":(1, 1,(0,1,0)), "-Y":(1,-1,(0,-1,0)),
    "+Z":(2, 1,(0,0,1)), "-Z":(2,-1,(0,0,-1)),
}

class ViewportWidget(QWidget):
    def __init__(self, parent=None, use_stable=True):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        self._dark = True; self._pending = None; self._building = False
        if HAS_PVQT:
            if use_stable and _StableInteractor is not None:
                self.plotter = _StableInteractor(self, self)
            else:
                self.plotter = QtInteractor(self)
            self._set_bg(); lay.addWidget(self.plotter)
        else:
            lbl = QLabel("\u26a0  pyvistaqt not installed\n\npip install pyvistaqt")
            lbl.setAlignment(Qt.AlignCenter); lbl.setStyleSheet("color:#2E3A52;font-size:14px;")
            lay.addWidget(lbl); self.plotter = None

    def _set_bg(self):
        if self.plotter: self.plotter.set_background("#07080E" if self._dark else "#F4F6FA")

    def toggle_background(self):
        self._dark = not self._dark; self._set_bg()
        if self.plotter:
            if self._pending is not None: self._replay_pending()
            else: self.plotter.render()

    def is_dark(self): return self._dark
    def _replay_pending(self):
        if self._pending is not None and self.plotter is not None: self.plotter.render()

    def closeEvent(self, event):
        if self.plotter is not None:
            try: self.plotter.close()
            except Exception: pass
        super().closeEvent(event)

    def show_domain(self, nx, ny, nz, dx_mm, body_data=None,
                    flux_bcs=None, vol_bcs=None, annotation="", flow_dir="+X"):
        self._pending = ("domain", nx, ny, nz, dx_mm, body_data, flux_bcs, vol_bcs, annotation, flow_dir)
        self._building = True
        try: self._render_domain_impl(nx, ny, nz, dx_mm, body_data or [], flux_bcs or [], vol_bcs or [], annotation, flow_dir)
        finally: self._building = False

    def _render_domain_impl(self, nx, ny, nz, dx_mm, bodies, flux_bcs, vol_bcs, annotation, flow_dir):
        if not self.plotter: return
        self.plotter.clear(); self._set_bg(); pl = self.plotter
        wire_col = "#2A3A5A" if self._dark else "#888888"
        leg_bg   = "#0D1420" if self._dark else "#FFFFFF"
        txt_col  = "#3A5A8A" if self._dark else "#445566"
        fallback = ["steelblue","darkorange","mediumseagreen","mediumpurple","tomato","gold","deepskyblue","sienna"]
        region = np.zeros((nx,ny,nz), dtype=np.int32)
        for idx,b in enumerate(bodies,1):
            try:
                vox = np.load(b["npz"], allow_pickle=True)["voxel_solid"].astype(bool)
                i0,j0,k0 = (int(x) for x in b["offset_ijk"]); Gx,Gy,Gz=vox.shape
                di0=max(i0,0);di1=min(i0+Gx,nx); dj0=max(j0,0);dj1=min(j0+Gy,ny); dk0=max(k0,0);dk1=min(k0+Gz,nz)
                si0=di0-i0;si1=si0+(di1-di0); sj0=dj0-j0;sj1=sj0+(dj1-dj0); sk0=dk0-k0;sk1=sk0+(dk1-dk0)
                region[di0:di1,dj0:dj1,dk0:dk1][vox[si0:si1,sj0:sj1,sk0:sk1]] = idx
            except Exception: continue
        nb = len(bodies)
        for fi,bc in enumerate(flux_bcs): region[bc["mask_np"]] = nb+1+fi
        for vi,bc in enumerate(vol_bcs):  region[bc["mask_np"]] = nb+1+len(flux_bcs)+vi
        grid = pv.ImageData(dimensions=(nx+1,ny+1,nz+1), spacing=(1,1,1), origin=(0,0,0))
        grid.cell_data["region"] = region.reshape(-1, order="F").astype(np.float32)
        pl.add_mesh(pv.Box(bounds=(0,nx,0,ny,0,nz)), style="wireframe", color=wire_col, line_width=1.5)
        for idx,b in enumerate(bodies,1):
            color=b.get("color",fallback[(idx-1)%len(fallback)]); name=b.get("name",f"body_{idx}")
            mesh=grid.threshold([idx-0.5,idx+0.5],scalars="region")
            if mesh.n_cells>0: pl.add_mesh(mesh,color=color,opacity=0.75,smooth_shading=True,specular=0.2,label=name)
        flux_cols=["limegreen","springgreen","greenyellow"]
        for fi,bc in enumerate(flux_bcs):
            val=float(nb+1+fi); mesh=grid.threshold([val-0.5,val+0.5],scalars="region")
            if mesh.n_cells>0:
                pl.add_mesh(mesh,color=flux_cols[fi%len(flux_cols)],opacity=1.0,
                            label=f"Flux {bc.get('axis','')}  {bc.get('q',0)/1000:.2f} kW/m\u00b2  ({bc.get('n_vox',0):,} vox)")
        vol_cols=["tomato","magenta","yellow","cyan"]
        for vi,bc in enumerate(vol_bcs):
            val=float(nb+1+len(flux_bcs)+vi); mesh=grid.threshold([val-0.5,val+0.5],scalars="region")
            if mesh.n_cells>0:
                pl.add_mesh(mesh,color=vol_cols[vi%len(vol_cols)],opacity=0.9,
                            label=f"Vol.Heat {bc.get('name','src')}  {bc.get('Q',0):.1f} W  ({bc.get('n_vox',0):,} vox)")
        if bodies or flux_bcs or vol_bcs:
            n_ent=len(bodies)+len(flux_bcs)+len(vol_bcs)
            try:
                pl.add_legend(bcolor=leg_bg, border=False, size=(0.26, 0.06*max(1,n_ent)),
                              loc="lower right", label_font_size=16, face="rectangle")
            except TypeError:
                pl.add_legend(bcolor=leg_bg, border=False, size=(0.26, 0.06*max(1,n_ent)), loc="lower right")
        ax_idx,sign,uvec = _FLOW_AXIS.get(flow_dir,(0,1,(1,0,0)))
        dims=[nx,ny,nz]; c=[d/2 for d in dims]
        shaft_len=dims[ax_idx]*0.35
        tip=list(c); tip[ax_idx]=0.0 if sign>0 else dims[ax_idx]
        start=list(tip); start[ax_idx]-=sign*shaft_len
        arrow_col="#00CCFF" if self._dark else "#005599"
        arrow_mesh=pv.Arrow(start=start,direction=uvec,scale=shaft_len,tip_length=0.35,tip_radius=0.08,shaft_radius=0.03)
        pl.add_mesh(arrow_mesh,color=arrow_col,opacity=0.9)
        label_pos=list(start); label_pos[ax_idx]-=sign*(shaft_len*0.15)
        perp=(ax_idx+1)%3; label_pos[perp]+=dims[perp]*0.12
        pl.add_point_labels([label_pos],[f"Flow {flow_dir}"],font_size=14,text_color=arrow_col,bold=True,shape=None,always_visible=True,shadow=False)
        info=annotation or f"Domain {nx*dx_mm:.0f}\u00d7{ny*dx_mm:.0f}\u00d7{nz*dx_mm:.0f} mm  |  voxel {dx_mm:.1f} mm"
        pl.add_text(info,position="upper_left",font_size=14,color=txt_col)
        pl.add_axes(); pl.reset_camera(); pl.render()

    def show_results_embedded(self, npz_path, flow_dir="+X", cmap="inferno"):
        if not self.plotter: return
        from UI_components.cht_main_window import launch_results_viewer
        self._pending = None
        launch_results_viewer(
            params=dict(
                npz_path=npz_path, field="T", cmap=cmap,
                show_bodies=None, show_fluid=False, fluid_opacity=0.3,
                streamlines=False, n_seeds=150, seed_mode="inlet_plane",
            ),
            flow_dir=flow_dir, plotter=self.plotter, dark=self._dark,
        )

    def show_placeholder(self, msg):
        self._pending=("placeholder",msg); self._building=True
        try:
            if not self.plotter: return
            self.plotter.clear(); self._set_bg()
            self.plotter.add_text(msg,position="upper_edge",font_size=11,color="#1E2840" if self._dark else "#AABBCC")
            self.plotter.render()
        finally: self._building=False


# ── ViewerPanel ───────────────────────────────────────────────────────────────

class ViewerPanel(QWidget):
    load_results_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(6,6,6,6); lay.setSpacing(4)

        tb_row = QHBoxLayout(); tb_row.setSpacing(6)
        self._lbl_info = QLabel("Domain preview"); self._lbl_info.setStyleSheet("color:#2E3A52;font-size:14px;")
        tb_row.addWidget(self._lbl_info, 1)

        proj_lbl = QLabel("Project:"); proj_lbl.setStyleSheet("color:#4A6A9A;font-size:14px;")
        self.txt_project = QLineEdit(); self.txt_project.setPlaceholderText("project_name")
        self.txt_project.setMinimumWidth(140); self.txt_project.setMaximumWidth(220)
        self.txt_project.setMinimumHeight(26); self.txt_project.setToolTip("Output .npz filename (without extension)")
        tb_row.addWidget(proj_lbl); tb_row.addWidget(self.txt_project)

        self._btn_load_res = QPushButton("\U0001f4c2  Load Results\u2026")
        self._btn_load_res.setObjectName("btn_load_res"); self._btn_load_res.setMinimumHeight(26)
        tb_row.addWidget(self._btn_load_res)

        self._btn_fullviewer = QPushButton("\u25a1  Full Interactive Viewer")
        self._btn_fullviewer.setObjectName("btn_bg"); self._btn_fullviewer.setMinimumHeight(26); self._btn_fullviewer.setEnabled(False)
        tb_row.addWidget(self._btn_fullviewer)

        self._btn_bg = QPushButton("\u2600  Light mode")
        self._btn_bg.setObjectName("btn_bg"); self._btn_bg.setMinimumHeight(26); self._btn_bg.setMaximumWidth(120)
        tb_row.addWidget(self._btn_bg)
        lay.addLayout(tb_row)

        self.view_tabs = QTabWidget(); self.view_tabs.setObjectName("view_tabs"); lay.addWidget(self.view_tabs, 1)
        self.vp_geo = ViewportWidget(); self.vp_res = ViewportWidget(use_stable=False)
        self.view_tabs.addTab(self.vp_geo, "  \u2bc1  Domain Preview  ")
        self.view_tabs.addTab(self.vp_res, "  \U0001f321  Results  ")
        self.vp_geo.show_placeholder("Load solids\nthen click  'Preview'")
        self.vp_res.show_placeholder("Run the simulation\nor load a results file.")

        self._btn_bg.clicked.connect(self._toggle_bg)
        self._btn_load_res.clicked.connect(self.load_results_clicked)

    @property
    def project_name(self): return self.txt_project.text().strip()

    def _toggle_bg(self):
        for vp in (self.vp_geo, self.vp_res): vp.toggle_background()
        self._btn_bg.setText("\u2600  Light mode" if self.vp_geo.is_dark() else "\U0001f319  Dark mode")

    def set_info(self, msg): self._lbl_info.setText(msg)
    def enable_full_viewer(self, en): self._btn_fullviewer.setEnabled(en)
