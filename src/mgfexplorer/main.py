import sys
from PyQt6.QtWidgets import QApplication
from .app import MGFExplorerApp


def main():
    app = QApplication(sys.argv)
    window = MGFExplorerApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
