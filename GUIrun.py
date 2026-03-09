# #!/usr/bin/env python3
import sys,os
_here=os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path: sys.path.insert(0,_here)
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QColor,QPalette
from UI_components.cht_constants import STYLESHEET
from UI_components.cht_main_window import MainWindow
def main():
    app=QApplication(sys.argv); app.setStyle("Fusion"); app.setStyleSheet(STYLESHEET)
    pal=QPalette()
    pal.setColor(QPalette.Window,QColor("#0F1117")); pal.setColor(QPalette.WindowText,QColor("#E0E6F0"))
    pal.setColor(QPalette.Base,QColor("#141B2E")); pal.setColor(QPalette.AlternateBase,QColor("#0F1421"))
    pal.setColor(QPalette.Text,QColor("#C8D8F0")); pal.setColor(QPalette.Button,QColor("#141B2E"))
    pal.setColor(QPalette.ButtonText,QColor("#C8D8F0")); pal.setColor(QPalette.Highlight,QColor("#1A5FB4"))
    pal.setColor(QPalette.HighlightedText,QColor("#FFFFFF")); pal.setColor(QPalette.ToolTipBase,QColor("#0F1117"))
    pal.setColor(QPalette.ToolTipText,QColor("#C8D8F0")); app.setPalette(pal)
    win=MainWindow(); win.show(); sys.exit(app.exec_())
if __name__=="__main__":
    main()

# #!/usr/bin/env python3
# import sys, os
# _here = os.path.dirname(os.path.abspath(__file__))
# if _here not in sys.path: sys.path.insert(0, _here)

# import qdarktheme
# from PyQt5.QtWidgets import QApplication
# from UI_components.cht_main_window import MainWindow

# def main():
#     app = QApplication(sys.argv)
#     base_qss = qdarktheme.load_stylesheet("dark")
#     font_qss = """
#     * { font-size: 16px; }
#     QTabBar::tab { font-size: 16px; padding: 8px 16px; }
#     QGroupBox { font-size: 15px; font-weight: 700; }
#     QLabel { font-size: 16px; }
#     QCheckBox { font-size: 16px; }
#     QComboBox { font-size: 16px; min-height: 30px; }
#     QDoubleSpinBox, QSpinBox, QLineEdit { font-size: 16px; min-height: 30px; }
#     QPushButton { font-size: 16px; padding: 7px 14px; }
#     QTextEdit { font-size: 14px; }
#     """
#     app.setStyleSheet(base_qss + font_qss)
#     win = MainWindow()
#     win.show()
#     sys.exit(app.exec_())

# if __name__ == "__main__":
#     main()