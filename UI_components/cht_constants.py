"""
cht_constants.py
All application-wide presets, enum lists, color maps and the QSS stylesheet.
No Qt imports — pure data.
"""

FLUID_PRESETS = {
    "Water":            dict(nu_m2_s=1.0e-6,  rho_kg_m3=997.0, k_W_mK=0.600, cp_J_kgK=4182.0, tin_C=25.0),
    "Air (physical)":   dict(nu_m2_s=15.0e-6, rho_kg_m3=1.25,  k_W_mK=0.024, cp_J_kgK=1006.0, tin_C=20.0),
    "Air (LBM-scaled)": dict(nu_m2_s=0.02,    rho_kg_m3=1.0,   k_W_mK=1.0,   cp_J_kgK=50.0,   tin_C=20.0),
}

SOLID_PRESETS = {
    "Aluminum (physical)":   dict(k_W_mK=237.0, rho_kg_m3=2700.0, cp_J_kgK=900.0),
    "Aluminum (LBM-scaled)": dict(k_W_mK=5.0,   rho_kg_m3=1.0,    cp_J_kgK=80.0),
    "Copper":                dict(k_W_mK=401.0, rho_kg_m3=8960.0, cp_J_kgK=385.0),
    "TIM (silicone pad)":    dict(k_W_mK=6.0,   rho_kg_m3=2500.0, cp_J_kgK=800.0),
    "Transformer core":      dict(k_W_mK=25.0,  rho_kg_m3=7650.0, cp_J_kgK=490.0),
    "Ferrite core":          dict(k_W_mK=4.0,   rho_kg_m3=4800.0, cp_J_kgK=700.0),
    "Steel":                 dict(k_W_mK=50.0,  rho_kg_m3=7850.0, cp_J_kgK=490.0),
}

BODY_COLORS  = ["steelblue", "darkorange", "gold", "sienna", "mediumpurple",
                "tomato", "limegreen", "hotpink", "deepskyblue", "coral"]
BODY_ROLES   = ["fluid_base", "stack_below", "stack_above", "fixed"]
BUILD_DIRS   = ["+Z", "-Z", "+Y", "-Y", "+X", "-X"]
FLOW_DIRS    = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
FLUX_AXES    = ["-Z", "+Z", "-Y", "+Y", "-X", "+X"]
COLLISION_MODES = ["bgk", "trt", "mrt", "mrt_smag"]
OUTLET_MODES    = ["convective", "extrapolation", "neumann"]

COLOR_HEX = {
    "steelblue": "#4682B4", "darkorange": "#FF8C00", "gold": "#FFD700",
    "sienna": "#A0522D", "mediumpurple": "#9370DB", "tomato": "#FF6347",
    "limegreen": "#32CD32", "hotpink": "#FF69B4", "deepskyblue": "#00BFFF",
    "coral": "#FF7F50",
}

MODEL_VERSION = "1.0"

# ─────────────────────────────────────────────────────────────────────────────
#  APPLICATION STYLESHEET
# ─────────────────────────────────────────────────────────────────────────────
STYLESHEET_old = """
* { font-family: "Segoe UI", "Inter", Arial, sans-serif; font-size: 16px; color: #E0E6F0; }
QMainWindow, QWidget { background-color: #0F1117; }
#ctrl_panel { background-color: #131820; border-right: 1px solid #1E2840; }

QTabWidget#ctrl_tabs::pane { border: none; background-color: #131820; }
QTabWidget#ctrl_tabs QTabBar::tab {
    background: #0F1117; color: #3B4A62; border: 1px solid #1A2236;
    border-bottom: none; padding: 7px 14px; font-size: 16px; font-weight: 600;
    border-radius: 5px 5px 0 0; margin-right: 2px;
}
QTabWidget#ctrl_tabs QTabBar::tab:selected { background: #131820; color: #7BBFFF; border-color: #2A5FA8; border-bottom-color: #131820; }
QTabWidget#ctrl_tabs QTabBar::tab:hover:!selected { color: #8BAAC8; background: #161C2A; }

QTabWidget#view_tabs::pane { border: 1px solid #1E2840; border-radius: 0 5px 5px 5px; background: #08090E; }
QTabWidget#view_tabs QTabBar::tab {
    background: #0F1117; color: #3B4A62; border: 1px solid #1A2236;
    border-bottom: none; padding: 7px 18px; font-weight: 500;
    border-radius: 5px 5px 0 0; margin-right: 2px;
}
QTabWidget#view_tabs QTabBar::tab:selected { background: #08090E; color: #E0E6F0; border-color: #2A5FA8; border-bottom-color: #08090E; }
QTabWidget#view_tabs QTabBar::tab:hover:!selected { color: #A8C8F0; background: #10151F; }

QGroupBox {
    font-weight: 700; font-size: 14px; color: #4A7AAA; border: 1px solid #1E2840;
    border-radius: 5px; margin-top: 12px; padding-top: 8px;
    background-color: #0F1421; letter-spacing: 0.8px; text-transform: uppercase;
}
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 1px 6px; left: 8px; color: #4A7AAA; background: #0F1421; }

QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit {
    background: #141B2E; border: 1px solid #1E2840; border-radius: 4px;
    padding: 4px 7px; color: #C8D8F0; min-height: 26px;
}
QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus { border-color: #2A5FA8; }
QComboBox::drop-down { border: none; width: 18px; }
QComboBox::down-arrow { border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 5px solid #4A7AAA; margin-right: 4px; }
QComboBox QAbstractItemView { background: #141B2E; border: 1px solid #1E2840; selection-background-color: #1E3A6A; color: #C8D8F0; outline: none; }
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button, QSpinBox::up-button, QSpinBox::down-button { background: #1E2840; border: none; width: 16px; }

QLabel { color: #8090A8; }
QLabel#hint { color: #2E3A52; font-size: 14px; font-style: italic; }

QCheckBox, QRadioButton { spacing: 7px; color: #8090A8; }
QCheckBox::indicator, QRadioButton::indicator { width: 15px; height: 15px; border: 2px solid #1E2840; border-radius: 3px; background: #141B2E; }
QRadioButton::indicator { border-radius: 8px; }
QCheckBox::indicator:checked, QRadioButton::indicator:checked { background: #1A5FB4; border-color: #1A5FB4; }
QCheckBox::indicator:hover, QRadioButton::indicator:hover { border-color: #2A5FA8; }

QPushButton { border-radius: 5px; padding: 6px 12px; font-weight: 600; font-size: 14px; border: none; color: #C8D8F0; }
QPushButton#btn_icon { background: #141B2E; border: 1px solid #1E2840; color: #4A7AAA; min-width: 28px; max-width: 34px; padding: 4px; border-radius: 4px; }
QPushButton#btn_icon:hover { background: #1A2436; border-color: #2A5FA8; color: #7BBFFF; }
QPushButton#btn_icon:disabled { color: #1E2840; border-color: #141B2E; }
QPushButton#btn_browse { background: #141B2E; border: 1px dashed #1E2840; color: #4A7AAA; text-align: left; padding-left: 8px; }
QPushButton#btn_browse:hover { background: #1A2436; border-color: #2A5FA8; color: #7BBFFF; }
QPushButton#btn_preview { background: #0E2040; border: 1px solid #1A4070; color: #5A9AE0; }
QPushButton#btn_preview:hover { background: #122850; }
QPushButton#btn_preview:disabled { background: #0A1020; color: #1E2840; border-color: #141B2E; }
QPushButton#btn_run { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #0F3A7A,stop:1 #0A5A44); color:#FFFFFF; font-size:14px; padding:10px; border:1px solid #1A5FA8; }
QPushButton#btn_run:hover { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #154A9A,stop:1 #0E7055); }
QPushButton#btn_run:disabled { background: #0A1020; color: #1E2840; border-color: #141B2E; }
QPushButton#btn_run[running="true"] { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #5A0A0A,stop:1 #6A2A0A); color:#FFAAAA; border-color:#8A2A2A; }
QPushButton#btn_bg { background: #141B2E; border: 1px solid #1E2840; color: #8090A8; padding: 4px 8px; font-size: 14px; border-radius: 4px; }
QPushButton#btn_bg:hover { background: #1A2436; color: #C8D8F0; border-color: #2A5FA8; }
QPushButton#btn_mdl { background: #0A1828; border: 1px solid #1A3050; color: #4A8AA8; padding: 5px 10px; font-size: 14px; border-radius: 4px; font-weight: 600; }
QPushButton#btn_mdl:hover { background: #102035; color: #7ABBD0; border-color: #2A6080; }
QPushButton#btn_load_res { background: #0A1828; border: 1px solid #1A3050; color: #4A8AA8; padding: 4px 8px; font-size: 14px; border-radius: 4px; }
QPushButton#btn_load_res:hover { background: #102035; color: #7ABBD0; border-color: #2A6080; }

QListWidget { background: #0A0E18; border: 1px solid #1E2840; border-radius: 4px; outline: none; color: #C8D8F0; }
QListWidget::item { padding: 6px 8px; border-bottom: 1px solid #101520; }
QListWidget::item:selected { background: #0E2040; color: #7BBFFF; border-left: 2px solid #2A5FA8; }
QListWidget::item:hover:!selected { background: #0F1830; }

QProgressBar { background: #0A0E18; border: 1px solid #1E2840; border-radius: 3px; height: 5px; color: transparent; }
QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #1A5FB4,stop:1 #0E7055); border-radius: 3px; }

QStatusBar { background: #08090E; color: #2E3A52; border-top: 1px solid #141B2E; font-size: 14px; }
QTextEdit#log { background: #06080F; border: 1px solid #141B2E; border-radius: 3px; color: #3A8A5A; font-family: "Consolas","Courier New",monospace; font-size: 14px; padding: 3px; }

QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: #0A0E18; width: 7px; margin: 0; }
QScrollBar::handle:vertical { background: #1E2840; border-radius: 3px; min-height: 18px; }
QScrollBar::handle:vertical:hover { background: #2A3A5A; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QFrame#div { background: #1A2236; max-height: 1px; min-height: 1px; }

QDialog { background: #0F1421; }
QDialogButtonBox QPushButton { background: #141B2E; border: 1px solid #1E2840; color: #8090A8; padding: 5px 14px; min-width: 70px; }
QDialogButtonBox QPushButton:default { background: #0F3A7A; border-color: #1A5FA8; color: #7BBFFF; }
QDialogButtonBox QPushButton:hover { background: #1A2436; color: #C8D8F0; }
"""

# REPLACE the entire STYLESHEET string WITH:
STYLESHEET = """
* { font-family: "Segoe UI", "Inter", Arial, sans-serif; font-size: 16px; color: #E8EAED; }
QMainWindow, QWidget { background-color: #1E1E1E; }
#ctrl_panel { background-color: #252526; border-right: 1px solid #333333; }

QTabWidget#ctrl_tabs::pane { border: none; background-color: #252526; }
QTabWidget#ctrl_tabs QTabBar::tab {
    background: #1E1E1E; color: #888888; border: 1px solid #333333;
    border-bottom: none; padding: 7px 14px; font-size: 16px; font-weight: 600;
    border-radius: 5px 5px 0 0; margin-right: 2px;
}
QTabWidget#ctrl_tabs QTabBar::tab:selected { background: #252526; color: #FFFFFF; border-color: #555555; border-bottom-color: #252526; }
QTabWidget#ctrl_tabs QTabBar::tab:hover:!selected { color: #CCCCCC; background: #2A2A2A; }

QTabWidget#view_tabs::pane { border: 1px solid #333333; border-radius: 0 5px 5px 5px; background: #141414; }
QTabWidget#view_tabs QTabBar::tab {
    background: #1E1E1E; color: #888888; border: 1px solid #333333;
    border-bottom: none; padding: 7px 18px; font-weight: 500;
    border-radius: 5px 5px 0 0; margin-right: 2px;
}
QTabWidget#view_tabs QTabBar::tab:selected { background: #141414; color: #FFFFFF; border-color: #555555; border-bottom-color: #141414; }
QTabWidget#view_tabs QTabBar::tab:hover:!selected { color: #CCCCCC; background: #222222; }

QGroupBox {
    font-weight: 700; font-size: 14px; color: #AAAAAA; border: 1px solid #333333;
    border-radius: 5px; margin-top: 12px; padding-top: 8px;
    background-color: #252526; letter-spacing: 0.8px; text-transform: uppercase;
}
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 1px 6px; left: 8px; color: #AAAAAA; background: #252526; }

QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit {
    background: #2D2D2D; border: 1px solid #3F3F3F; border-radius: 4px;
    padding: 4px 7px; color: #E8EAED; min-height: 26px;
}
QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus { border-color: #666666; }
QComboBox::drop-down { border: none; width: 18px; }
QComboBox::down-arrow { border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 5px solid #AAAAAA; margin-right: 4px; }
QComboBox QAbstractItemView { background: #2D2D2D; border: 1px solid #3F3F3F; selection-background-color: #404040; color: #E8EAED; outline: none; }
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button, QSpinBox::up-button, QSpinBox::down-button { background: #3A3A3A; border: none; width: 16px; }

QLabel { color: #CCCCCC; }
QLabel#hint { color: #666666; font-size: 14px; font-style: italic; }

QCheckBox, QRadioButton { spacing: 7px; color: #CCCCCC; }
QCheckBox::indicator, QRadioButton::indicator { width: 15px; height: 15px; border: 2px solid #444444; border-radius: 3px; background: #1A1A1A; }

QRadioButton::indicator { border-radius: 8px; }
QCheckBox::indicator:checked, QRadioButton::indicator:checked { 
    background: #1A7FD4; border-color: #2A9FFF; 
    image: url(none);
}
QCheckBox::indicator:checked { 
    background-color: #1A7FD4; 
    border: 2px solid #2A9FFF;
}
QCheckBox::indicator:hover, QRadioButton::indicator:hover { border-color: #888888; }

QPushButton { border-radius: 5px; padding: 6px 12px; font-weight: 600; font-size: 14px; border: none; color: #E8EAED; }
QPushButton#btn_icon { background: #2D2D2D; border: 1px solid #3F3F3F; color: #AAAAAA; min-width: 28px; max-width: 34px; padding: 4px; border-radius: 4px; }
QPushButton#btn_icon:hover { background: #383838; border-color: #666666; color: #FFFFFF; }
QPushButton#btn_icon:disabled { color: #444444; border-color: #2D2D2D; }
QPushButton#btn_browse { background: #2D2D2D; border: 1px dashed #3F3F3F; color: #AAAAAA; text-align: left; padding-left: 8px; }
QPushButton#btn_browse:hover { background: #383838; border-color: #666666; color: #FFFFFF; }
QPushButton#btn_preview { background: #2D2D2D; border: 1px solid #4A4A4A; color: #CCCCCC; }
QPushButton#btn_preview:hover { background: #383838; }
QPushButton#btn_preview:disabled { background: #1E1E1E; color: #444444; border-color: #2D2D2D; }
QPushButton#btn_run { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #0F3A7A,stop:1 #0A5A44); color:#FFFFFF; font-size:16px; padding:10px; border:1px solid #1A5FA8; }
QPushButton#btn_run:hover { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #154A9A,stop:1 #0E7055); }
QPushButton#btn_run:disabled { background: #1E1E1E; color: #444444; border-color: #2D2D2D; }
QPushButton#btn_run[running="true"] { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #5A0A0A,stop:1 #6A2A0A); color:#FFAAAA; border-color:#8A2A2A; }
QPushButton#btn_bg { background: #2D2D2D; border: 1px solid #3F3F3F; color: #CCCCCC; padding: 4px 8px; font-size: 14px; border-radius: 4px; }
QPushButton#btn_bg:hover { background: #383838; color: #FFFFFF; border-color: #666666; }
QPushButton#btn_mdl { background: #2D2D2D; border: 1px solid #4A4A4A; color: #CCCCCC; padding: 5px 10px; font-size: 14px; border-radius: 4px; font-weight: 600; }
QPushButton#btn_mdl:hover { background: #383838; color: #FFFFFF; border-color: #666666; }
QPushButton#btn_load_res { background: #2D2D2D; border: 1px solid #4A4A4A; color: #CCCCCC; padding: 4px 8px; font-size: 14px; border-radius: 4px; }
QPushButton#btn_load_res:hover { background: #383838; color: #FFFFFF; border-color: #666666; }

QListWidget { background: #1A1A1A; border: 1px solid #333333; border-radius: 4px; outline: none; color: #E8EAED; }
QListWidget::item { padding: 6px 8px; border-bottom: 1px solid #2A2A2A; }
QListWidget::item:selected { background: #2D2D2D; color: #FFFFFF; border-left: 2px solid #666666; }
QListWidget::item:hover:!selected { background: #252525; }

QProgressBar { background: #1A1A1A; border: 1px solid #333333; border-radius: 3px; height: 5px; color: transparent; }
QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #1A5FB4,stop:1 #0E7055); border-radius: 3px; }

QStatusBar { background: #141414; color: #666666; border-top: 1px solid #2A2A2A; font-size: 14px; }
QTextEdit#log { background: #141414; border: 1px solid #2A2A2A; border-radius: 3px; color: #3A8A5A; font-family: "Consolas","Courier New",monospace; font-size: 14px; padding: 3px; }

QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: #1A1A1A; width: 7px; margin: 0; }
QScrollBar::handle:vertical { background: #3A3A3A; border-radius: 3px; min-height: 18px; }
QScrollBar::handle:vertical:hover { background: #555555; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QFrame#div { background: #333333; max-height: 1px; min-height: 1px; }

QDialog { background: #252526; }
QDialogButtonBox QPushButton { background: #2D2D2D; border: 1px solid #3F3F3F; color: #CCCCCC; padding: 5px 14px; min-width: 70px; }
QDialogButtonBox QPushButton:default { background: #0F3A7A; border-color: #1A5FA8; color: #FFFFFF; }
QDialogButtonBox QPushButton:hover { background: #383838; color: #FFFFFF; }
"""